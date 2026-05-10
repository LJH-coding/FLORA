# PyTorch
import torch

# Other
import os
import glob

def find_last_checkpoint(callback_path, return_full_path=False):

    # All Checkpoints
    checkpoints = glob.glob(os.path.join(callback_path, "checkpoints_*.ckpt"))

    # Select Last Checkpoint else None
    max_steps = 0
    last_checkpoint = None
    for checkpoint in checkpoints:
        checkpoint = checkpoint.split("/")[-1]
        checkpoint_steps = int(checkpoint.split("_")[-1].replace(".ckpt", ""))
        if checkpoint_steps > max_steps:
            max_steps = checkpoint_steps
            last_checkpoint = checkpoint

    # Join path
    if last_checkpoint != None and return_full_path:
        last_checkpoint = os.path.join(callback_path, last_checkpoint)

    return last_checkpoint

def load_model(args):

    # Model Device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    if "cuda" in str(device):
        print("device: {}, {}, {}MB".format(device, torch.cuda.get_device_properties(device).name, int(torch.cuda.get_device_properties(device).total_memory // 1e6)))
        args.num_gpus = torch.cuda.device_count()
    else:
        print("device: {}".format(device))
        args.num_gpus = 1

    # Set Model Device
    model = args.config.model.to(device)

    # Set Callback Path
    args.config.callback_path = getattr(args.config, "callback_path", os.path.join("callbacks", "/".join(args.config_file.replace(".py", "").split("/")[1:])))
    # Append callback Tag
    if hasattr(args.config, "callback_tag"):
        args.config.callback_path = os.path.join(args.config.callback_path, args.config.callback_tag)

    # Last Checkpoint
    if args.load_last:
        last_checkpoint = find_last_checkpoint(args.config.callback_path)
        if last_checkpoint != None:
            args.checkpoint = last_checkpoint

    # Load Checkpoint
    if args.checkpoint is not None:
        model.load(os.path.join(args.config.callback_path, args.checkpoint))

    # Model Summary
    model.summary(show_dict=args.show_dict, show_modules=args.show_modules)
    
    return model

def load_datasets(args):
    raise RuntimeError(
        "Dataset pipeline has been removed from this distribution."
    )