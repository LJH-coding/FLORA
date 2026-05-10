#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from datasets import load_dataset
from torchvision.io import write_png

from nnet.modules.twister.encoder_network import EncoderNetwork
from nnet.modules.twister.decoder_network import DecoderNetwork


class HuggingFaceImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


class AutoEncoder(nn.Module):
    def __init__(self, stoch_size=32, discrete=32, dim_cnn=32, image_channels=3):
        super().__init__()
        self.encoder = EncoderNetwork(
            dim_input_cnn=image_channels,
            dim_cnn=dim_cnn,
            stoch_size=stoch_size,
            discrete=discrete,
        )
        self.decoder = DecoderNetwork(
            dim_output_cnn=image_channels,
            feat_size=stoch_size * discrete,
            dim_cnn=dim_cnn,
        )

    def forward(self, x, return_first=False):
        latent = self.encoder(x)
        stoch = latent["stoch"]
        logits = latent["logits"]
        outputs = self.decoder(stoch, return_first=return_first)
        if return_first:
            recon_dist, first_feat = outputs
        else:
            recon_dist, first_feat = outputs, None
        return recon_dist, stoch, logits, first_feat


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_model_range(x):
    return x - 0.5


def to_uint8(x):
    x = (x + 0.5).clamp(0, 1)
    return (x * 255).to(torch.uint8)


def pca_feature_map_to_rgb(feat_map):
    # feat_map: (B, C, H, W) or (B, N, C, H, W)
    if feat_map.dim() == 5:
        feat_map = feat_map[:, 0]
    b, c, h, w = feat_map.shape
    feat = feat_map.permute(0, 2, 3, 1).reshape(b, h * w, c)
    rgb = []
    for i in range(b):
        x = feat[i]
        x = x - x.mean(dim=0, keepdim=True)
        _, _, v = torch.pca_lowrank(x, q=3)
        proj = x @ v[:, :3]
        min_val = proj.min(dim=0, keepdim=True).values
        max_val = proj.max(dim=0, keepdim=True).values
        proj = (proj - min_val) / (max_val - min_val + 1e-8)
        rgb_img = proj.reshape(h, w, 3).permute(2, 0, 1)
        rgb.append(rgb_img)
    return torch.stack(rgb, dim=0)


def visualize_batch(gt, feat_map, recon, out_path, indices, writer=None, step=0):
    pca_img = pca_feature_map_to_rgb(feat_map)
    # Upsample PCA map to match GT size
    if pca_img.shape[-1] != gt.shape[-1]:
        pca_img = torch.nn.functional.interpolate(pca_img, size=gt.shape[-2:], mode="nearest")
    gt_vis = (gt + 0.5).clamp(0, 1)
    recon_vis = (recon + 0.5).clamp(0, 1)

    batch = []
    for i in indices:
        if i < 0 or i >= gt.shape[0]:
            continue
        batch.extend([gt_vis[i], pca_img[i], recon_vis[i]])
    if not batch:
        for i in range(min(gt.shape[0], 8)):
            batch.extend([gt_vis[i], pca_img[i], recon_vis[i]])
    grid = make_grid(batch, nrow=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path))
    if writer is not None:
        writer.add_image("ae/gt_pca_recon", grid, step)


def categorical_kl(logits):
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    num_classes = logits.shape[-1]
    log_uniform = -torch.log(torch.tensor(float(num_classes), device=logits.device, dtype=logits.dtype))
    kl = (probs * (log_probs - log_uniform)).sum(dim=-1)
    return kl


def l2_distance(x, recon):
    diff = (x - recon).flatten(start_dim=1)
    return torch.sqrt((diff * diff).sum(dim=-1)).mean()


def train_one_epoch(model, loader, optimizer, device, kl_weight):
    model.train()
    total_loss = 0.0
    num = 0
    for x in loader:
        x = x.to(device)
        recon_dist, _, logits, _ = model(x)
        recon_loss = -recon_dist.log_prob(x).mean()
        kl_loss = categorical_kl(logits).mean()
        loss = recon_loss + kl_weight * kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        num += x.size(0)
    return total_loss / max(1, num)


@torch.no_grad()
def evaluate(model, loader, device, fid_num_samples=5000, use_fid=True, kl_weight=0.0, fid_mode="clean", fid_out_dir=None):
    model.eval()
    total_loss = 0.0
    num = 0
    real_dir = None
    fake_dir = None
    if use_fid:
        if fid_out_dir is None:
            raise ValueError("fid_out_dir is required for clean-fid")
        real_dir = Path(fid_out_dir) / "real"
        fake_dir = Path(fid_out_dir) / "fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)
        # clean previous files
        for p in real_dir.glob("*.png"):
            p.unlink()
        for p in fake_dir.glob("*.png"):
            p.unlink()
        sample_idx = 0
    for x in loader:
        x = x.to(device)
        recon_dist, _, logits, _ = model(x)
        recon = recon_dist.mode()
        recon_loss = -recon_dist.log_prob(x).mean()
        kl_loss = categorical_kl(logits).mean()
        loss = recon_loss + kl_weight * kl_loss
        l2 = l2_distance(x, recon)
        total_loss += loss.item() * x.size(0)
        num += x.size(0)
        if use_fid and fid_num_samples > 0:
            real = to_uint8(x)
            fake = to_uint8(recon)
            for i in range(real.shape[0]):
                if fid_num_samples <= 0:
                    break
                write_png(real[i].cpu(), str(real_dir / f"{sample_idx:08d}.png"))
                write_png(fake[i].cpu(), str(fake_dir / f"{sample_idx:08d}.png"))
                sample_idx += 1
                fid_num_samples -= 1
    fid_score = None
    if use_fid:
        try:
            from cleanfid import fid as cleanfid
            fid_score = cleanfid.compute_fid(str(real_dir), str(fake_dir), mode=fid_mode)
        except Exception:
            fid_score = None
    return total_loss / max(1, num), fid_score, l2.item()


def build_loaders(dataset_name, train_split, val_split, cache_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_model_range),
    ])
    train_ds = load_dataset(dataset_name, split=train_split, cache_dir=cache_dir)
    val_ds = load_dataset(dataset_name, split=val_split, cache_dir=cache_dir)
    train_set = HuggingFaceImageDataset(train_ds, transform=transform)
    val_set = HuggingFaceImageDataset(val_ds, transform=transform)
    # Disable pin_memory to avoid deprecation warnings in recent PyTorch
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", type=str, default="zh-plus/tiny-imagenet")
    parser.add_argument("--hf_train_split", type=str, default="train")
    parser.add_argument("--hf_val_split", type=str, default="valid")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="ae")
    parser.add_argument("--fid_samples", type=int, default=5000)
    parser.add_argument("--fid_mode", type=str, default="clean")
    parser.add_argument("--fid_every", type=int, default=5)
    parser.add_argument("--stoch_size", type=int, default=32)
    parser.add_argument("--discrete", type=int, default=32)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader = build_loaders(
        args.hf_dataset,
        args.hf_train_split,
        args.hf_val_split,
        args.hf_cache_dir,
        args.batch_size,
        args.num_workers,
    )

    model = AutoEncoder(
        stoch_size=args.stoch_size,
        discrete=args.discrete,
    ).to(device)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Encoder params: {encoder_params:,}")
    print(f"Decoder params: {decoder_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"runs/{args.run_name}")
    except Exception:
        writer = None

    out_dir = Path(f"outputs/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    vis_indices = None
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, kl_weight=args.kl_weight)
        end_time = time.time()
        training_time = end_time - start_time
        fid_score = None
        if args.fid_every > 0 and (epoch + 1) % args.fid_every == 0:
            fid_dir = out_dir / f"fid_epoch_{epoch+1:03d}"
            _, fid_score, _ = evaluate(
                model,
                val_loader,
                device,
                fid_num_samples=args.fid_samples,
                kl_weight=args.kl_weight,
                fid_mode=args.fid_mode,
                fid_out_dir=fid_dir,
            )
        val_loss, _, l2_val = evaluate(
            model,
            val_loader,
            device,
            fid_num_samples=0,
            kl_weight=args.kl_weight,
            fid_mode=args.fid_mode,
            fid_out_dir=None,
            use_fid=False,
        )
        if writer is not None:
            writer.add_scalar("ae/train_loss", train_loss, epoch)
            writer.add_scalar("ae/val_loss", val_loss, epoch)
            writer.add_scalar("ae/l2", l2_val, epoch)
            if fid_score is not None:
                writer.add_scalar("ae/fid", fid_score, epoch)

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | fid={fid_score} | training_time={training_time:.2f}s")

        # visualization
        batch = next(iter(val_loader))
        batch = batch.to(device)
        if vis_indices is None:
            bsz = batch.shape[0]
            sample_n = min(8, bsz)
            vis_indices = rng.sample(range(bsz), k=sample_n)
        recon_dist, _, _, first_feat = model(batch, return_first=True)
        recon = recon_dist.mode()
        vis_path = out_dir / f"epoch_{epoch+1:03d}.png"
        visualize_batch(batch, first_feat, recon, vis_path, indices=vis_indices, writer=writer, step=epoch)

        # checkpoint
        torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "ae_last.pt")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
