# PyTorch
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

# NeuralNets
from nnet import models
from nnet import optimizers
from nnet import envs
from nnet import modules
from nnet.modules import twister as twister_networks
from nnet.structs import AttrDict

# Other
import copy
import itertools
import os
import glob
from math import log2
from einops import rearrange
from copy import deepcopy
import ruamel.yaml as yaml
import pathlib
import sys

# Schedulers
from nnet import schedulers
from nnet.modules.diffusion import FM, ShortCut

class TWISTER(models.Model):

    def __init__(self, env_name, override_config={}, name="Transformer-based World model wIth contraSTivE Representations (TWISTER)"):
        super(TWISTER, self).__init__(name=name)

        # Env Type
        env_name = env_name.split("-")
        self.env_type = env_name[0]
        assert self.env_type in ["dmc", "atari100k", "diambra", "crafter", "dmcgb2"]

        configs = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent / "./configs/config.yaml").read_text()
        )

        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value

        name_list = ["default", self.env_type]

        # Config
        self.config = AttrDict()
        self.config.env_name = env_name
        self.config.env_type = env_name[0]

        for name in name_list:
            recursive_update(self.config, configs[name])

        # Model Sizes
        model_sizes = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent / "./configs/model_size.yaml").read_text()
        )
        for key, value in model_sizes.items():
            model_sizes[key] = AttrDict(value)

        # Env
        if self.env_type == "dmc":
            self.config.env_class = envs.dm_control.dm_control_dict[env_name[1]]
            self.config.env_params.update({"task": env_name[2]})
        elif self.env_type == "dmcgb2":
            self.config.loss_reward_scale = 1000.0 # Follow DreamaerPro's setting.
            self.config.env_class = envs.dmcgb2.dmcgb2_dict[env_name[1]]
            self.config.env_params = {"task": env_name[2], "mode": env_name[3], "seed": 0, "history_frames": 1, "img_size": (64, 64), "action_repeat": 2}
        elif self.env_type == "atari100k":
            self.config.env_class = envs.atari.AtariEnv
            self.config.env_params.update({"game": env_name[1]})
        elif self.env_type == "diambra":
            self.config.env_class = envs.diambra.DiambraEnv
            self.config.env_params.update({"game": env_name[1]})
        elif self.env_type == "crafter":
            self.config.env_class = envs.crafter.CrafterEnv

        env_steps_per_update = (self.config.batch_size * self.config.L) / (self.config.env_step_period * self.config.num_envs)
        total_env_steps= self.config.epochs * self.config.epoch_length * env_steps_per_update * self.config.env_params["action_repeat"] * self.config.num_envs
        print(f"Total env steps: {total_env_steps}")

        # Optimizer
        self.config.precision = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[self.config.precision]

        # World Model Params
        model_params = model_sizes[self.config.model_size]
        self.config.repr_layers = model_params.num_layers
        self.config.repr_hidden_size = model_params.hidden_size
        self.config.stoch_size = model_params.stoch_size # means number of tokens for each image
        self.config.discrete = model_params.discrete # means codebook size of each token
        self.config.model_hidden_size = model_params.hidden_size
        self.config.action_hidden_size = model_params.hidden_size
        self.config.value_hidden_size = model_params.hidden_size
        self.config.reward_hidden_size = model_params.hidden_size
        self.config.discount_hidden_size = model_params.hidden_size
        self.config.action_layers = model_params.num_layers
        self.config.value_layers = model_params.num_layers
        self.config.reward_layers = model_params.num_layers
        self.config.discount_layers = model_params.num_layers
        self.config.dim_cnn = model_params.dim_cnn

        # TSSM
        self.config.num_blocks_trans = model_params.num_blocks_trans
        self.config.ff_ratio_trans = model_params.ff_ratio_trans
        self.config.num_heads_trans = model_params.num_heads_trans
        self.config.drop_rate_trans = model_params.drop_rate_trans
        self.config.encoder_cnn_norm = {"class": "LayerNorm", "params": {"eps": 1e-3, "convert_float32": True}}
        self.config.module_pre_norm = False
        self.config.detach_decoder = False

        # Contrastive
        self.config.contrastive_augments = torchvision.transforms.RandomResizedCrop(size=(64, 64), antialias=True, scale=(0.25, 1))
        self.config.contrastive_hidden_size = self.config.model_hidden_size
        self.config.contrastive_out_size = self.config.contrastive_hidden_size

        # Override Config
        for key, value in override_config.items():
            assert key in self.config, "{} not in config".format(key)

            if key=="precision":
                self.config[key] = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[value]
            else:
                self.config[key] = value

        self.use_semantic_latent = bool(self.config.barlow_twins["enable"])

        # Config asserts
        assert self.config.att_context_left <= self.config.L

        # Create Training Envs
        self.env = envs.wrappers.BatchEnv([
            envs.wrappers.ResetOnException(
                envs.wrappers.TimeLimit(
                    self.config.env_class(**dict(deepcopy(self.config.env_params), **deepcopy(self.config.train_env_params))), 
                    time_limit=self.config.time_limit
                )
            )
        for env_i in range(self.config.num_envs)])
                
        # Create Evaluation Env
        if self.config.eval_episodes > 0:
            self.env_eval = envs.wrappers.RecordEnv(envs.wrappers.ResetOnException(
                envs.wrappers.TimeLimit(
                    self.config.env_class(**dict(deepcopy(self.config.env_params), **deepcopy(self.config.eval_env_params))),
                    time_limit=self.config.time_limit_eval
                )
            ), record_path=self.config.record_path, max_gifs=self.config.eval_episodes)
        else:
            self.env_eval = None

        self.encoder_network = twister_networks.EncoderNetwork(
            dim_input_cnn=self.config.image_channels,
            dim_cnn=self.config.dim_cnn,
            cnn_norm=self.config.encoder_cnn_norm,
            stoch_size=self.config.stoch_size,
            discrete=self.config.discrete,
        )
        self.embedding_network = None
        self.decoder_network = twister_networks.DecoderNetwork(
            dim_output_cnn=self.config.image_channels,
            feat_size=self.config.stoch_size * self.config.discrete,
            dim_cnn=self.config.dim_cnn,
            cnn_norm=self.config.norm,
        )
        self.reward_scale, self.target_reward_scale = 1, 0
        latent_size = self.config.stoch_size * self.config.discrete
        feat_size = latent_size + model_params.hidden_size
        self.projector = None
        if self.use_semantic_latent:
            self.embedding_network = twister_networks.EmbeddingNetwork(
                dim_input_cnn=self.config.image_channels,
                dim_cnn=self.config.dim_cnn,
                cnn_norm=self.config.encoder_cnn_norm,
                stoch_size=self.config.stoch_size,
                discrete=self.config.discrete,
            )
            feat_size = 2 * latent_size + model_params.hidden_size
            self.projector = modules.Linear(
                in_features=latent_size,
                out_features=self.embedding_network.dim_concat,
            )
        diffusion_model = None
        if self.config.use_diffusion:
            diffusion_model = ShortCut(
                in_features=self.config.model_hidden_size + (2 * latent_size if self.use_semantic_latent else latent_size),
                stoch_size=(2 * self.config.stoch_size if self.use_semantic_latent else self.config.stoch_size),
                discrete=self.config.discrete,
                hidden_size=self.config.model_hidden_size,
            )

        self.rssm = twister_networks.TSSM(
            num_actions=self.env.num_actions, 
            stoch_size=self.config.stoch_size, 
            discrete=self.config.discrete, 
            learn_initial=self.config.learn_initial,
            norm=self.config.norm,
            hidden_size=self.config.model_hidden_size,
            num_blocks=self.config.num_blocks_trans,
            ff_ratio=self.config.ff_ratio_trans,
            num_heads=self.config.num_heads_trans,
            drop_rate=self.config.drop_rate_trans,
            att_context_left=self.config.att_context_left,
            module_pre_norm=self.config.module_pre_norm,
            diffusion_model=diffusion_model,
            use_semantic_latent=self.use_semantic_latent,
        )
        self.policy_network = twister_networks.PolicyNetwork(
            num_actions=self.env.num_actions, 
            hidden_size=self.config.action_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.action_layers, 
            discrete=self.config.policy_discrete,
            norm=self.config.norm,
            sampling_tmp=self.config.sampling_tmp
        )
        self.value_network = twister_networks.ValueNetwork(
            hidden_size=self.config.value_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.value_layers,
            norm=self.config.norm
        )
        self.reward_network = twister_networks.RewardNetwork(
            hidden_size=self.config.reward_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.reward_layers,
            norm=self.config.norm
        )
        self.continue_network = twister_networks.ContinueNetwork(
            hidden_size=self.config.discount_hidden_size, 
            feat_size=feat_size, 
            num_mlp_layers=self.config.discount_layers,
            norm=self.config.norm
        )
        self.contrastive_network = nn.ModuleList([twister_networks.ContrastiveNetwork(
            feat_size=feat_size + t * self.env.num_actions,
            embed_size=self.config.stoch_size * self.config.discrete,
            hidden_size=self.config.contrastive_hidden_size,
            out_size=self.config.contrastive_out_size,
            num_layers=self.config.contrastive_layers
        ) for t in range(self.config.contrastive_steps)])

        self.harmony = model_params.harmony
        if self.harmony:
            self.harmony_s1 = nn.Parameter(-torch.log(torch.tensor(1.0)))  # obs_loss
            self.harmony_s2 = nn.Parameter(-torch.log(torch.tensor(1.0)))  # recon_loss
            self.harmony_s3 = nn.Parameter(-torch.log(torch.tensor(1.0)))  # rew_loss

        # Slow Moving Networks
        self.add_frozen("v_target", copy.deepcopy(self.value_network))

        # Percentiles
        self.register_buffer("perc_low", torch.tensor(0.0))
        self.register_buffer("perc_high", torch.tensor(0.0))
        
        # Training Infos
        self.register_buffer("episodes", torch.tensor(0))
        self.register_buffer("ep_rewards", torch.zeros(self.config.num_envs), persistent=False)
        self.register_buffer("action_step", torch.tensor(0))

        # World Model
        self.world_model = self.WorldModel(outer=self)

        # Actor Model
        self.actor_model = self.ActorModel(outer=self)

        # Critic Model
        self.critic_model = self.CriticModel(outer=self)

    def summary(self, show_dict=False, show_modules=False):

        # Model Name
        print("Model name: {}".format(self.name))

        # Number Params
        print("World Model Parameters: {:,}".format(self.num_params(self.world_model)))
        print("Actor Parameters: {:,}".format(self.num_params(self.actor_model)))
        print("Critic Parameters: {:,}".format(self.num_params(self.critic_model)))

        # Options
        if show_dict:
            self.show_dict()
        if show_modules:
            self.show_modules()

    def preprocess_inputs(self, state, time_stacked):

        def norm_image(image):

            assert image.dtype == torch.uint8

            return image.type(torch.float32) / 255 - 0.5

        # List of Inputs
        if isinstance(state, list):
            state = [norm_image(s) if s.dim()==(5 if time_stacked else 4) and s.dtype == torch.uint8 else s for s in state]

        # State (could be image or lowd)
        else:
            state = norm_image(state) if state.dim()==(5 if time_stacked else 4) and state.dtype == torch.uint8 else state

        return state

    def merge_semantic_latent(self, latent, embedding):
        if not self.use_semantic_latent:
            return latent
        return {
            "stoch": torch.cat([latent["stoch"], embedding["stoch"]], dim=-1),
            "logits": torch.cat([latent["logits"], embedding["logits"]], dim=-2),
        }

    def save(self, path, save_optimizer=True, keep_last_k=None):
        
        # Keep checkpoint key-space compile-agnostic.
        model_state_dict = {k.replace("._orig_mod.", "."): v for k, v in self.state_dict().items()}
        
        # Save Model Checkpoint
        torch.save({
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": None if not save_optimizer else {key: value.state_dict() for key, value in self.optimizer.items()} if isinstance(self.optimizer, dict) else self.optimizer.state_dict(),
            "model_step": self.model_step,
            "grad_scaler_state_dict": self.grad_scaler.state_dict() if hasattr(self, "grad_scaler") else None,
            "replay_buffer_state_dict": self.replay_buffer.state_dict()
        }, path)
        
        # Save Buffer
        self.replay_buffer.save()

        # Print Model state
        print("Model saved at step {}: {}".format(self.model_step, path))

        # Keep last k checkpoints
        if keep_last_k != None:

            # List checkpoints
            save_dir = os.path.dirname(path)
            checkpoints_list = glob.glob(os.path.join(save_dir, "*.ckpt"))
            checkpoints_list = sorted(checkpoints_list, key=lambda s: int(os.path.splitext(s)[0].split("/")[-1].split("_")[-1]))

            # Remove older_checkpoint
            while len(checkpoints_list) > keep_last_k:

                # Pop older_checkpoint
                older_checkpoint = checkpoints_list.pop(0)

                # Remove older_checkpoint
                os.remove(older_checkpoint)

                # Print
                print("Removed old checkpoint {}".format(older_checkpoint))

    def load(self, path, load_optimizer=True, verbose=True, strict=True):

        # Print Load state
        if verbose:
            print("Load Model from {}".format(path))

        # Load Model Checkpoint
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load Model State Dict
        model_state_dict = {key.replace("._orig_mod.", "."): value for key, value in checkpoint["model_state_dict"].items()}
        self.load_state_dict(model_state_dict, strict=strict)

        # Load Optimizer State Dict
        if load_optimizer and checkpoint["optimizer_state_dict"] is not None:

            if isinstance(self.optimizer, dict):
                for key, value in self.optimizer.items():
                    value.load_state_dict(checkpoint["optimizer_state_dict"][key])
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Model Step, already loaded from optm
            self.model_step.fill_(checkpoint["model_step"])

        # Load replay Buffer State Dict
        if self.config.load_replay_buffer_state_dict:
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer_state_dict"])
        elif verbose:
            print("load_replay_buffer_state_dict set to False: replay buffer state dict not loaded")

        # Load Grad Scaler
        if "grad_scaler_state_dict" in checkpoint:
            self.grad_scaler_state_dict = checkpoint["grad_scaler_state_dict"]

        # Print Model state
        if verbose:
            print("Model loaded at step {}".format(self.model_step))

    def set_replay_buffer(self, replay_buffer):

        # Replay Buffer
        self.replay_buffer = replay_buffer

        # Set History
        obs_reset = self.env.reset()
        self.episode_history = AttrDict(
            ep_step=torch.zeros(self.config.num_envs), # (N,)
            hidden=(self.rssm.initial(batch_size=self.config.num_envs, seq_length=1, dtype=torch.float32, detach_learned=True), torch.zeros(self.config.num_envs, self.env.num_actions, dtype=torch.float32)), 
            state=obs_reset.state,
            episodes=[AttrDict(
                states=[obs_reset.state[env_i]],
                actions=[torch.zeros(self.env.num_actions, dtype=torch.float32)],
                rewards=[obs_reset.reward[env_i]],
                dones=[obs_reset.done[env_i]],
                is_firsts=[obs_reset.is_first[env_i]],
                model_steps=[self.model_step.clone()]
            ) for env_i in range(self.config.num_envs)]
        )

        # Update Buffer with reset step
        for env_i in range(self.config.num_envs):
            sample = []
            sample.append(obs_reset.state[env_i])
            sample.append(torch.zeros(self.env.num_actions, dtype=torch.float32))
            sample.append(obs_reset.reward[env_i])
            sample.append(obs_reset.done[env_i])
            sample.append(obs_reset.is_first[env_i])
            sample.append(self.model_step.clone())
            buffer_infos = self.replay_buffer.append_step(sample, env_i)

        # Add Buffer Infos
        for key, value in buffer_infos.items():
            self.add_info(key, value)

    def on_train_begin(self):

        # Pre Fill Buffer
        if self.config.pre_fill_steps > 0 and self.replay_buffer.num_steps < self.config.pre_fill_steps:
            print("Prefill dataset with {} steps, policy={}".format(self.config.pre_fill_steps, "random" if self.config.random_pre_fill_steps else "sample"))
            while self.replay_buffer.num_steps < self.config.pre_fill_steps:
                self.env_step()

    def compile(self):
        if self.config.use_cosine_annealing:
            model_lr_scheduler = schedulers.CosineAnnealingLR(
                lr_max=self.config.model_lr,
                T_max=self.config.cosine_T_max,
                eta_min=self.config.cosine_eta_min
            )
        else:
            model_lr_scheduler = self.config.model_lr
        
        # Compile World Model
        model_param_groups = [
            self.encoder_network.parameters(),
            self.decoder_network.parameters(),
            self.rssm.parameters(),
            self.reward_network.parameters(),
            self.continue_network.parameters(),
            self.contrastive_network.parameters(),
            [self.harmony_s1, self.harmony_s2, self.harmony_s3] if self.harmony else [],
        ]
        if self.use_semantic_latent:
            model_param_groups.extend([
                self.embedding_network.parameters(),
                self.projector.parameters(),
            ])
        model_params = itertools.chain(*model_param_groups)

        if self.config.use_sam:
            from nnet.optimizers.sam import SAM
            print("🔥 USING SAM OPTIMIZER FOR WORLD MODEL 🔥")
            print(f"SAM Parameters: rho={self.config.sam['rho']}, adaptive={self.config.sam['use_adaptive']}")
            world_model_optimizer = SAM(
                params=[{"params": model_params, "lr": self.config.model_lr, "grad_max_norm": self.config.model_grad_max_norm, "eps": self.config.model_eps}],
                base_optimizer=optimizers.Adam,
                rho=self.config.sam['rho'],
                adaptive=self.config.sam['use_adaptive'],
                weight_decay=self.config.opt_weight_decay
            )
        else:
            # Use regular Adam optimizer
            world_model_optimizer = optimizers.Adam(params=[
                {"params": model_params, "lr": self.config.model_lr, "grad_max_norm": self.config.model_grad_max_norm, "eps": self.config.model_eps}, 
            ], weight_decay=self.config.opt_weight_decay)


        self.world_model.compile(
            optimizer=world_model_optimizer, 
            losses={},
            loss_weights={},
            metrics=None,
            decoders=None
        )

        # Compile Actor Model
        self.actor_model.compile(
            optimizer=optimizers.Adam(params=[
                {"params": self.policy_network.parameters(), "lr": self.config.actor_lr, "grad_max_norm": self.config.actor_grad_max_norm, "eps": self.config.actor_eps},
            ], weight_decay=self.config.opt_weight_decay), 
            losses={},
            loss_weights={},
            metrics=None,
            decoders=None
        )

        # Compile Critic Model
        self.critic_model.compile(
            optimizer=optimizers.Adam(params=[
                {"params": self.value_network.parameters(), "lr": self.config.critic_lr, "grad_max_norm": self.config.critic_grad_max_norm, "eps": self.config.critic_eps}, 
            ], weight_decay=self.config.opt_weight_decay), 
            losses={},
            loss_weights={},
            metrics=None,
            decoders=None
        )

        # Model Step
        self.model_step = self.world_model.optimizer.param_groups[0]["lr_scheduler"].model_step

        # Optimizer
        self.optimizer = {"world_model": self.world_model.optimizer, "actor_model": self.actor_model.optimizer, "critic_model": self.critic_model.optimizer}

        # Set Compiled to True
        self.compiled = True

    def env_step(self):

        # Eval Mode
        training = self.training
        self.encoder_network.eval()
        if self.use_semantic_latent:
            self.embedding_network.eval()
        self.rssm.eval()
        self.policy_network.eval()

        ###############################################################################
        # Forward / Env Step
        ###############################################################################

        # Recover State / hidden
        state = self.episode_history.state
        hidden = self.episode_history.hidden

        # Unpack hidden
        prev_latent, action = hidden

        # Transfer to device
        state = self.transfer_to_device(state)
        prev_latent = self.transfer_to_device(prev_latent)
        action = self.transfer_to_device(action)

        # Forward Policy Network
        with torch.no_grad():

            # Repr State (B, ...)
            state_proc = self.preprocess_inputs(state, time_stacked=False)
            latent = self.encoder_network(state_proc)
            if self.use_semantic_latent:
                embedding = self.embedding_network(state_proc)
                latent = self.merge_semantic_latent(latent, embedding)

            # Unsqueeze Time dim (B, 1, ...)
            latent = {key: value.unsqueeze(dim=1) for key, value in latent.items()}

            # Generate is_firsts_hidden for forward
            if prev_latent["hidden"] != None:
                is_firsts_hidden = torch.zeros(self.config.num_envs, self.rssm.get_hidden_len(prev_latent["hidden"]), dtype=torch.float32, device=action.device)
                for env_i in range(self.config.num_envs):
                    env_i_length = len(self.episode_history.episodes[env_i].is_firsts) - 1
                    if 0 < env_i_length <= is_firsts_hidden.shape[1]:
                        is_firsts_hidden[env_i, -env_i_length] = 1.0
            else:
                is_firsts_hidden = None

            # RSSM (B, 1, ...)
            latent, _ = self.rssm(
                states=latent, 
                prev_states=prev_latent, 
                prev_actions=action.unsqueeze(dim=1), 
                is_firsts=torch.tensor([1.0 if len(self.episode_history.episodes[env_i].is_firsts) == 1 else 0.0 for env_i in range(self.config.num_envs)], dtype=torch.float32, device=action.device).unsqueeze(dim=1),
                is_firsts_hidden=is_firsts_hidden
            )

            # Get feat (B, Dfeat)
            feat = self.rssm.get_feat(latent).squeeze(dim=1)

            # Policy Sample
            action = self.policy_network(feat).sample().cpu()

        # Update Hidden
        latent["hidden"] = self.rssm.slice_hidden(latent["hidden"])
        hidden = (latent, action)

        # Clip Action
        if not self.config.policy_discrete:
            action = action.type(torch.float32).clip(self.env.clip_low, self.env.clip_high)

        # Env Step
        if (self.replay_buffer.num_steps < self.config.pre_fill_steps) and self.config.random_pre_fill_steps:
            action = self.env.sample()
        obs = self.env.step(action.argmax(dim=-1) if self.config.policy_discrete else action)

        ###############################################################################
        # Update Infos / Buffer
        ###############################################################################

        # Update training_infos
        self.action_step += self.env.action_repeat * self.config.num_envs
        self.ep_rewards += obs.reward.to(self.ep_rewards.device)

        # Update History State
        self.episode_history.state = obs.state
        self.episode_history.hidden = hidden
        self.episode_history.ep_step += self.env.action_repeat
        # Update History Episodes
        for env_i in range(self.config.num_envs):
            if not obs.error[env_i]:
                self.episode_history.episodes[env_i].states.append(obs.state[env_i])
                self.episode_history.episodes[env_i].actions.append(action[env_i])
                self.episode_history.episodes[env_i].rewards.append(obs.reward[env_i])
                self.episode_history.episodes[env_i].dones.append(obs.done[env_i])
                self.episode_history.episodes[env_i].is_firsts.append(obs.is_first[env_i])
                self.episode_history.episodes[env_i].model_steps.append(self.model_step.clone())

        # Update Traj Buffer
        for env_i in range(self.config.num_envs):
            if not obs.error[env_i]:
                sample = []
                sample.append(obs.state[env_i])
                sample.append(action[env_i])
                sample.append(obs.reward[env_i])
                sample.append(obs.done[env_i])
                sample.append(obs.is_first[env_i])
                sample.append(self.model_step.clone())
                buffer_infos = self.replay_buffer.append_step(sample, env_i)

                # Add Buffer Infos
                for key, value in buffer_infos.items():
                    self.add_info(key, value)

        ###############################################################################
        # Reset Env
        ###############################################################################

        # Is_last / Time Limit
        for env_i in range(self.config.num_envs):
            if obs.is_last[env_i]:

                # Set finished_episode
                finished_episode = []
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].states, dim=0))
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].actions, dim=0))
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].rewards, dim=0))
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].dones, dim=0))
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].is_firsts, dim=0))
                finished_episode.append(torch.stack(self.episode_history.episodes[env_i].model_steps, dim=0))

                # Copy Episode
                finished_episode = copy.deepcopy(finished_episode)

                # Add Infos
                self.add_info("episode_steps", self.episode_history.ep_step[env_i].item())
                self.add_info("episode_reward_total", self.ep_rewards[env_i].item())

                # Reset Episode Step
                self.episode_history.ep_step[env_i] = 0

                # Reset Hidden
                latent = self.rssm.initial(batch_size=1, dtype=torch.float32, detach_learned=True)
                action = torch.zeros(self.env.num_actions, dtype=torch.float32)
                self.episode_history.hidden[1][env_i] = action
                for key in self.episode_history.hidden[0]:

                    # Do not reset hidden
                    if key != "hidden":
                        self.episode_history.hidden[0][key][env_i] = latent[key].squeeze(dim=0)

                # Reset Env
                obs_reset = self.env.envs[env_i].reset()
                self.episode_history.state[env_i] = obs_reset.state

                # Reset Episode History
                self.episode_history.episodes[env_i] = AttrDict(
                    states=[obs_reset.state],
                    actions=[torch.zeros(self.env.num_actions, dtype=torch.float32)],
                    rewards=[obs_reset.reward],
                    dones=[obs_reset.done],
                    is_firsts=[obs_reset.is_first],
                    model_steps=[self.model_step.clone()]
                )

                # Update Traj Buffer
                sample = []
                sample.append(obs_reset.state)
                sample.append(torch.zeros(self.env.num_actions, dtype=torch.float32))
                sample.append(obs_reset.reward)
                sample.append(obs_reset.done)
                sample.append(obs_reset.is_first)
                sample.append(self.model_step.clone())
                buffer_infos = self.replay_buffer.append_step(sample, env_i)

                # Add Buffer Infos
                for key, value in buffer_infos.items():
                    self.add_info(key, value)

                # Update training_infos
                self.episodes += 1
                self.ep_rewards[env_i] = 0.0

        # Default Mode
        self.encoder_network.train(mode=training)
        if self.use_semantic_latent:
            self.embedding_network.train(mode=training)
        self.rssm.train(mode=training)
        self.policy_network.train(mode=training)

    def update_target_networks(self):

        # Update Target Networks
        if 0 <= self.config.critic_ema_decay <= 1:

            # Soft Update
            for param_target, param_net in zip(self.v_target.parameters(), self.value_network.parameters()):
                param_target.mul_(1 - self.config.critic_ema_decay)
                param_target.add_(self.config.critic_ema_decay * param_net.detach())
        else:

            # Hard Update
            if self.model_step % self.config.critic_ema_decay == 0:
                self.v_target.load_state_dict(self.value_network.state_dict())

    def train_step(self, inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training):

        # Init Dict
        batch_losses = {}
        batch_metrics = {}

        # Preprocess state (uint8 to float32)
        inputs = self.preprocess_inputs(inputs, time_stacked=True)

        ###############################################################################
        # World Train Step
        ###############################################################################

        # World Model Step
        self.set_require_grad([self.policy_network, self.value_network], False)
        self.set_require_grad([self.encoder_network, self.embedding_network, self.projector, self.decoder_network, self.rssm, self.reward_network, self.continue_network], True)
        world_model_batch_losses, world_model_batch_metrics, _ = self.world_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        batch_losses.update({"world_model_" + key: value for key, value in world_model_batch_losses.items()})
        batch_metrics.update({"world_model_" + key: value for key, value in world_model_batch_metrics.items()})
        self.infos.update({"world_model_" + key: value for key, value in self.world_model.infos.items()})

        ###############################################################################
        # Actor Model Step
        ###############################################################################

        # Eval Mode: Disable Dropout
        self.rssm.eval()

        self.set_require_grad(self.policy_network, True)
        self.set_require_grad([self.value_network, self.encoder_network, self.embedding_network, self.projector, self.decoder_network, self.rssm, self.reward_network, self.continue_network], False)
        actor_model_batch_losses, actor_model_batch_metrics, _ = self.actor_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        batch_losses.update({"actor_model_" + key: value for key, value in actor_model_batch_losses.items()})
        batch_metrics.update({"actor_model_" + key: value for key, value in actor_model_batch_metrics.items()})
        self.infos.update({"actor_model_" + key: value for key, value in self.actor_model.infos.items()})

        ###############################################################################
        # Value Model Step
        ###############################################################################

        self.set_require_grad(self.value_network, True)
        self.set_require_grad([self.policy_network, self.encoder_network, self.embedding_network, self.projector, self.decoder_network, self.rssm, self.reward_network, self.continue_network], False)
        critic_model_batch_losses, critic_model_batch_metrics, _ = self.critic_model.train_step(inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training)
        batch_losses.update({"critic_model_" + key: value for key, value in critic_model_batch_losses.items()})
        batch_metrics.update({"critic_model_" + key: value for key, value in critic_model_batch_metrics.items()})
        self.infos.update({"critic_model_" + key: value for key, value in self.critic_model.infos.items()})

        # Train Mode
        self.rssm.train()

        ###############################################################################
        # Update Target Networks
        ###############################################################################

        # Update value target
        self.update_target_networks()

        ###############################################################################
        # Env Step
        ###############################################################################

        # Env Step
        num_env_steps = (self.config.batch_size * self.config.L) / (self.config.env_step_period * self.config.num_envs)

        # Env step every n model step
        if 0 < num_env_steps < 1:
            model_step_period = 1 / num_env_steps
            if self.model_step % model_step_period == 0:
                with torch.cuda.amp.autocast(enabled=precision!=torch.float32, dtype=precision):
                    self.env_step()
            
        # n env steps per model step
        else:
            with torch.cuda.amp.autocast(enabled=precision!=torch.float32, dtype=precision):
                for i in range(int(num_env_steps)):
                    self.env_step()

        # Update Infos
        self.infos["episodes"] = self.episodes.item()
        for env_i in range(self.config.num_envs):
            self.infos["ep_rewards_{}".format(env_i)] = round(self.ep_rewards[env_i].item(), 2)
        self.infos["step"] = self.model_step
        self.infos["action_step"] = self.action_step.item()

        # Built
        if not self.built:
            self.built = True

        return batch_losses, batch_metrics, _
    
    class WorldModel(models.Model):

        def __init__(self, outer):
            super().__init__(name="World Model")
            object.__setattr__(self, "outer", outer)
            self.encoder_network = self.outer.encoder_network
            self.embedding_network = self.outer.embedding_network
            self.decoder_network = self.outer.decoder_network
            self.continue_network = self.outer.continue_network
            self.reward_network = self.outer.reward_network
            self.rssm = self.outer.rssm
            self.contrastive_network = self.outer.contrastive_network

        def __getattr__(self, name):
            return getattr(self.outer, name)

        def train_step(self, inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training):
            """Custom train_step to support SAM optimizer for world model."""
            from nnet.optimizers.sam import SAM

            # Default path for non-SAM optimizers.
            if not isinstance(self.optimizer, SAM):
                return super().train_step(
                    inputs=inputs,
                    targets=targets,
                    precision=precision,
                    grad_scaler=grad_scaler,
                    accumulated_steps=accumulated_steps,
                    acc_step=acc_step,
                    eval_training=eval_training,
                )

            # SAM needs two full forward/backward passes on the same batch.
            if accumulated_steps != 1:
                raise ValueError("SAM world model currently requires accumulated_steps == 1.")

            # Keep SAM path numerically stable by running both passes in fp32.
            self.optimizer.zero_grad()
            if grad_scaler.is_enabled():
                self.add_info("grad_scale", grad_scaler.get_scale())

            if "cuda" in str(self.device):
                with torch.cuda.amp.autocast(enabled=False):
                    batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, compute_metrics=eval_training)
                    loss = batch_losses["loss"]
            else:
                batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, compute_metrics=eval_training)
                loss = batch_losses["loss"]

            acc_step += 1
            loss.backward()

            # SAM ascent step: w -> w + e(w)
            self.optimizer.first_step(zero_grad=True)

            # Second pass at perturbed weights.
            if "cuda" in str(self.device):
                with torch.cuda.amp.autocast(enabled=False):
                    batch_losses_2, _, _, _ = self.forward_model(inputs, targets, compute_metrics=False)
                    loss_2 = batch_losses_2["loss"]
            else:
                batch_losses_2, _, _, _ = self.forward_model(inputs, targets, compute_metrics=False)
                loss_2 = batch_losses_2["loss"]

            loss_2.backward()

            # SAM descent step: restore w and apply base optimizer update.
            self.optimizer.second_step(zero_grad=True)
            acc_step = 0

            # Update optimizer/model infos.
            if len(self.optimizer.param_groups) > 1:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    self.add_info("lr_{}".format(i), float(param_group["lr"]))
                    if "grad_norm" in param_group:
                        self.add_info("grad_norm_{}".format(i), round(float(param_group["grad_norm"]), 4))
            else:
                self.add_info("lr", float(self.optimizer.param_groups[0]["lr"]))
                if "grad_norm" in self.optimizer.param_groups[0]:
                    self.add_info("grad_norm", round(float(self.optimizer.param_groups[0]["grad_norm"]), 4))

            self.add_info("step", self.model_step.item())

            return batch_losses, batch_metrics, acc_step

        def forward(self, inputs):

            # Unpack Inputs 
            states, actions, rewards, dones, is_firsts, model_steps = inputs

            # Outputs
            outputs = {}

            ###############################################################################
            # Model Forward
            ###############################################################################

            assert actions.shape[1] == self.config.L

            # Forward Representation Network (B, L, ...)
            latent = self.encoder_network(states)
            embedding = None
            if self.use_semantic_latent:
                embedding = self.embedding_network(states)
                latent = self.merge_semantic_latent(latent, embedding)

            # Model Observe (B, L, D)
            posts, priors = self.rssm.observe(
                states=latent, 
                prev_actions=actions, 
                is_firsts=is_firsts, 
                prev_state=None, 
                is_firsts_hidden=None
            )

            # Update Hidden States
            is_firsts_hidden_concat = is_firsts

            # Get feat (B, L, Dfeat)
            feats = self.rssm.get_feat(posts)

            # Predict reward (B, L, 1)
            model_rewards = self.reward_network(feats)

            # Rec Images (B, L, ...)
            z_size = self.config.stoch_size * self.config.discrete
            model_stoch = posts["stoch"][..., :z_size] if self.use_semantic_latent else posts["stoch"]
            states_pred = self.decoder_network(model_stoch.detach() if self.config.detach_decoder else model_stoch)

            # Predict Discounts
            discount_pred = self.continue_network(feats)

            ###############################################################################
            # Model Contrastive Loss
            ###############################################################################

            # Augment
            states_flat = states.flatten(0, 1).contiguous()   # (B*L, C, H, W)
            states_aug_flat = self.config.contrastive_augments(states_flat)  # batched
            states_aug = states_aug_flat.view_as(states)      # (B, L, C, H, W)

            # # Forward
            posts_con = self.encoder_network(states_aug)

            # Contrastive steps loop
            for t in range(self.config.contrastive_steps):

                # Action condition (B, L-t, A*t)
                if t > 0:
                    actions_cond = torch.cat([actions[:, 1+t_:min(actions.shape[1], actions.shape[1]+1+t_-t)] for t_ in range(t)], dim=-1)

                # Contrastive features (B, L-t, D)
                features_feats, features_embed = self.contrastive_network[t](
                    feats=self.rssm.get_feat(priors) if t==0 else torch.cat([self.rssm.get_feat(priors)[:, :-t], actions_cond], dim=-1), 
                    embed=posts_con["stoch"] if t==0 else posts_con["stoch"][:, t:]
                )
                    
                # Compute contrastive loss
                if features_feats.dtype != torch.float32:
                    with torch.cuda.amp.autocast(enabled=False):
                        info_nce_loss, acc_con = self.compute_contrastive_loss(features_feats.type(torch.float32), features_embed.type(torch.float32))
                        info_nce_loss = info_nce_loss.type(features_feats.dtype)
                else:
                    info_nce_loss, acc_con = self.compute_contrastive_loss(features_feats, features_embed)

                # Add Loss
                self.add_loss(
                    name="model_contrastive_{}".format(t), 
                    loss=- info_nce_loss.mean(), 
                    weight=self.config.loss_contrastive_scale * (self.config.contrastive_exp_lambda ** t) * ( (1.0 / sum([self.config.contrastive_exp_lambda ** t_ for t_ in range(self.config.contrastive_steps)])))
                )

                # Add Accuracy                    
                self.add_metric("acc_con" if t==0 else "acc_con_{}".format(t), acc_con)

            ###############################################################################
            # Model Dynamic Loss
            ###############################################################################

            # KL
            kl_prior = torch.distributions.kl.kl_divergence(self.rssm.get_dist({k: v if k == "hidden" else v.detach() for k, v in posts.items()}), self.rssm.get_dist(priors))
            kl_post = torch.distributions.kl.kl_divergence(self.rssm.get_dist(posts), self.rssm.get_dist({k: v if k == "hidden" else v.detach() for k, v in priors.items()}))

            if self.harmony:
                harmony_weight = 1.0 / torch.exp(self.harmony_s1)
                prior_scale = self.config.loss_kl_prior_scale / (self.config.loss_kl_prior_scale + self.config.loss_kl_post_scale)
                post_scale = self.config.loss_kl_post_scale / (self.config.loss_kl_prior_scale + self.config.loss_kl_post_scale)
                self.add_loss("kl_prior", torch.mean(torch.clip(kl_prior, min=self.config.free_nats)), weight=harmony_weight * prior_scale)
                self.add_loss("kl_post", torch.mean(torch.clip(kl_post, min=self.config.free_nats)), weight=harmony_weight * post_scale)
            else:
                self.add_loss("kl_prior", torch.mean(torch.clip(kl_prior, min=self.config.free_nats)), weight=self.config.loss_kl_prior_scale)
                self.add_loss("kl_post", torch.mean(torch.clip(kl_post, min=self.config.free_nats)), weight=self.config.loss_kl_post_scale)

            ###############################################################################
            # Model Embedding Prediction Loss
            ###############################################################################

            if self.use_semantic_latent:
                # Barlow Twins: project world features to encoder CNN embedding space.
                B, T = feats.shape[:2]
                x1 = self.projector(posts["stoch"][..., z_size:].reshape(B * T, -1))
                x2 = embedding["embed"].reshape(B * T, -1).detach()

                x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
                x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

                c = torch.mm(x1_norm.T, x2_norm) / (B * T)
                invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
                off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
                redundancy_loss = c[off_diag_mask].pow(2).sum()
                self.add_loss("model_embed", invariance_loss + self.config.barlow_twins["lambd"] * redundancy_loss, weight=self.config.barlow_twins["bt"])
                self.add_metric("embed variance", torch.var(embedding["embed"]))

            # Model Image Loss
            if self.harmony:
                self.add_loss("model_image", - states_pred.log_prob(states.detach()).mean(), weight=1.0 / torch.exp(self.harmony_s2))
            else:
                self.add_loss("model_image", - states_pred.log_prob(states.detach()).mean(), weight=self.config.loss_recon_scale)

            # Model Diffusion Loss
            if self.rssm.diffusion_model is not None:
                condition = torch.cat([priors["deter"], priors["stoch"]], dim=-1)
                diffusion_target = posts
                diffusion_loss = self.rssm.diffusion_model.loss(diffusion_target, condition)
                self.add_loss("model_diffusion", diffusion_loss, weight=self.config.loss_diffusion_scale)

            ###############################################################################
            # Model Reward Loss
            ###############################################################################

            # Model Reward Loss
            if self.harmony:
                self.add_loss("model_reward", - model_rewards.log_prob(rewards.unsqueeze(dim=-1).detach()).mean(), weight=1.0 / torch.exp(self.harmony_s3))
            else:
                self.add_loss("model_reward", - model_rewards.log_prob(rewards.unsqueeze(dim=-1).detach()).mean(), weight=self.config.loss_reward_scale)

            ###############################################################################
            # Harmony Regularizer
            ###############################################################################

            if self.harmony:
                self.add_loss("harmony_reg", torch.log(torch.exp(self.harmony_s1)+1) 
                + torch.log(torch.exp(self.harmony_s2)+1) 
                + torch.log(torch.exp(self.harmony_s3)+1)
                , weight=1.0)

            ###############################################################################
            # Model Discount Loss
            ###############################################################################

            # Model Discount Loss
            self.add_loss("model_discount", - discount_pred.log_prob((1.0 - dones).unsqueeze(dim=-1).detach()).mean(), self.config.loss_discount_scale)

            ###############################################################################
            # Flatten and Detach Posts
            ###############################################################################

            # K, V: (B, C+L, D) -> (B*L, C, D)
            hidden_flatten = [
                (
                    # Key (B*L, C, D)
                    torch.stack([

                        # Padd hidden if not enough left context (B, C, D)
                        torch.cat([
                            # Zero Padding to reach length (C,): max(0, L+C-1-t - len(h))
                            hidden_blk[0].new_zeros(hidden_blk[0].shape[0], max(0, self.config.L+self.config.att_context_left-1-t - hidden_blk[0].shape[1]), hidden_blk[0].shape[2]), 
                            # hidden [-L+t+1 - C:-L+t+1]
                            hidden_blk[0][:, max(0, hidden_blk[0].shape[1]-self.config.L+t+1 - self.config.att_context_left):hidden_blk[0].shape[1]-self.config.L+t+1]
                        ], dim=1) 

                    for t in range(0, self.config.L)], dim=1).flatten(start_dim=0, end_dim=1).detach(), # (B, L, C, D) -> (B*L, C, D)

                    # Value (B*L, C, D)
                    torch.stack([

                        # Padd hidden if not enough left context (B, C, D)
                        torch.cat([
                            # zeros max(0, L+C-1-t - len(h))
                            hidden_blk[1].new_zeros(hidden_blk[1].shape[0], max(0, self.config.L+self.config.att_context_left-1-t - hidden_blk[1].shape[1]), hidden_blk[1].shape[2]), 
                            # hidden [-L+t+1 - C:-L+t+1]
                            hidden_blk[1][:, max(0, hidden_blk[1].shape[1]-self.config.L+t+1 - self.config.att_context_left):hidden_blk[1].shape[1]-self.config.L+t+1]
                        ], dim=1) 

                    for t in range(0, self.config.L)], dim=1).flatten(start_dim=0, end_dim=1).detach(), # (B, L, C, D) -> (B*L, C, D)
                )
            for hidden_blk in posts["hidden"]]

            # is_firsts flatten (B, L) -> (B*L, 1), will result in masking hidden if true
            self.outer.detached_is_firsts = is_firsts.flatten(start_dim=0, end_dim=1).unsqueeze(dim=1).detach()

            # is_firsts hidden flatten (B, C+L) -> (B*L, C)
            self.outer.detached_is_firsts_hidden = torch.stack([
                torch.cat([
                    # Zero Padding to reach length (C,): max(0, L+C-1-t - len(h))
                    is_firsts_hidden_concat.new_zeros(is_firsts_hidden_concat.shape[0], max(0, self.config.L+self.config.att_context_left-1-t - is_firsts_hidden_concat.shape[1])),  
                    # set first element to True in order to mask padding (1,)
                    is_firsts_hidden_concat.new_ones(is_firsts_hidden_concat.shape[0], 1),
                    # is_firsts [t-C + 1:t]
                    is_firsts_hidden_concat[:, max(0, is_firsts_hidden_concat.shape[1]-self.config.L+t+1-self.config.att_context_left):is_firsts_hidden_concat.shape[1]-self.config.L+t]
                ], dim=1) 
            for t in range(0, self.config.L)], dim=1).flatten(start_dim=0, end_dim=1).detach()

            # Flatten and detach post (B, L, D) -> (B*L, 1, D) = (B', 1, D)
            self.outer.detached_posts = {k: hidden_flatten if k == "hidden" else v.flatten(start_dim=0, end_dim=1).unsqueeze(dim=1).detach() for k, v in posts.items()}

            return outputs
        
    class ActorModel(models.Model):

        def __init__(self, outer):
            super().__init__(name="Actor Model")
            object.__setattr__(self, "outer", outer)
            self.policy_network = self.outer.policy_network

        def __getattr__(self, name):
            return getattr(self.outer, name)
        
        def forward(self, inputs):

            # Unpack Inputs 
            states, actions, rewards, dones, is_firsts, model_steps  = inputs[:6]

            # Outputs
            outputs = {}

            ###############################################################################
            # Policy Forward
            ###############################################################################

            prev_state = self.detached_posts

            # Model Imagine H next states (B', 1+H, D) with state synchronized actions
            img_states = self.rssm.imagine(
                p_net=self.policy_network, 
                prev_state=prev_state, 
                img_steps=self.config.H,
                is_firsts=self.detached_is_firsts,
                is_firsts_hidden=self.detached_is_firsts_hidden
            )

            # Get feat (B', 1+H, Dfeat)
            feats = self.rssm.get_feat(img_states)

            # Predict rewards (B', 1+H, 1)
            model_rewards = self.reward_network(feats)

            # Predict Values (B', 1+H, 1)
            if self.config.target_value_reg:
                values = self.value_network(feats)
            else:
                values = self.v_target(feats)

            # Predict Discounts (B', 1+H, 1)
            discounts = self.continue_network(feats).mode # 0 / 1

            # Override discount prediction for the first step with the true
            # discount factor from the replay buffer.
            true_first = (1.0 - dones.flatten(start_dim=0, end_dim=1)).unsqueeze(dim=-1).unsqueeze(dim=-1) # 0 or 1
            discounts = torch.cat([true_first, discounts[:, 1:]], dim=1)

            ###############################################################################
            # Policy Loss
            ###############################################################################

            # (B', 1+H, 1)
            weights = torch.cumprod(self.config.gamma * discounts, dim=1).detach() / self.config.gamma

            # Compute lambda returns (B', H, 1), one action grad lost because of next value
            returns = self.compute_td_lambda(rewards=model_rewards.mode()[:, 1:], values=values.mode()[:, 1:], discounts=self.config.gamma * discounts[:, 1:])
            self.add_info("returns_mean", returns.mean().item())

            # Update Perc
            offset, invscale = self.update_perc(returns)

            # Norm Returns using quantiles ema ~ [0:1]
            normed_returns = (returns - offset) / invscale # 1:H+1
            normed_base = (values.mode()[:, :-1] - offset) / invscale # 0:H

            # advantage (B', H)
            advantage = (normed_returns - normed_base).squeeze(dim=-1)

            # Policy Dist (B', 1+H, A)
            policy_dist = self.policy_network(feats.detach()) 

            # Actor Loss
            if self.config.actor_grad == "dynamics":
                actor_loss = advantage
            elif self.config.actor_grad == "reinforce":
                actor_loss = policy_dist.log_prob(img_states["action"].detach())[:, :-1] * advantage.detach()
            else:
                raise Exception("Unknown actor grad: {}".format(self.actor_grad))
            
            # Add Negative Entropy loss
            policy_ent = policy_dist.entropy()[:, :-1]
            self.add_info("policy_ent", policy_ent.mean().item())
            actor_loss += self.config.eta_entropy * policy_ent

            # Apply weights
            actor_loss *= weights[:, :-1].squeeze(dim=-1)

            # Add loss
            self.add_loss("actor", - actor_loss.mean())  

            self.outer.detached_feats = feats.detach()
            self.outer.detached_returns = returns.detach()
            self.outer.detached_weights = weights.detach()

            return outputs
        
    class CriticModel(models.Model):

        def __init__(self, outer):
            super().__init__(name="Critic Model")
            object.__setattr__(self, "outer", outer)
            self.value_network = self.outer.value_network

        def __getattr__(self, name):
            return getattr(self.outer, name)
        
        def forward(self, inputs):

            # Unpack Inputs 
            states, actions, rewards, dones, is_firsts, model_steps  = inputs[:6]

            # Outputs
            outputs = {}

            ###############################################################################
            # Value Loss
            ###############################################################################

            feats = self.detached_feats
            returns = self.detached_returns
            weights = self.detached_weights

            # Value (B', H, 1)
            value_dist = self.value_network(feats.detach()[:, :-1])

            # Value Loss
            value_loss = value_dist.log_prob(returns.detach())
            
            # Add Regularization
            if self.config.target_value_reg:
                with torch.no_grad():
                    value_target = self.v_target(feats.detach()[:, :-1]).mode()
                value_loss += self.config.critic_slow_reg_scale * value_dist.log_prob(value_target.detach())

            # Weight loss
            value_loss *= weights[:, :-1].squeeze(dim=-1)

            # Add Loss
            self.add_loss("value", - value_loss.mean())

            return outputs
    
    def update_perc(self, returns):

        # Compute percentiles (,)
        low = torch.quantile(returns.detach(), q=self.config.return_norm_perc_low)
        high = torch.quantile(returns.detach(), q=self.config.return_norm_perc_high)

        # Update percentiles ema
        self.perc_low = self.config.return_norm_decay * self.perc_low + (1 - self.config.return_norm_decay) * low
        self.perc_high = self.config.return_norm_decay * self.perc_high + (1 - self.config.return_norm_decay) * high
        self.add_info("perc_low", self.perc_low.item())
        self.add_info("perc_high", self.perc_high.item())

        # Compute offset, invscale
        offset = self.perc_low
        invscale = torch.clip(self.perc_high - self.perc_low, min=1.0 / self.config.return_norm_limit)

        return offset.detach(), invscale.detach()
    
    def get_perc(self):

        # Compute offset, invscale
        offset = self.perc_low
        invscale = torch.clip(self.perc_high - self.perc_low, min=1.0 / self.config.return_norm_limit)

        return offset.detach(), invscale.detach()
    
    def compute_td_lambda(self, rewards, values, discounts):

        # Init for loop
        interm = rewards + discounts * (1 - self.config.lambda_td) * values
        vals = [values[:, -1]]

        # Recurrence loop
        for t in reversed(range(interm.shape[1])):
            vals.append(interm[:, t] + discounts[:, t] * self.config.lambda_td * vals[-1])

        # Stack and slice init val
        lambda_values = torch.stack(list(reversed(vals))[:-1], dim=1)

        return lambda_values
    
    def compute_contrastive_loss(self, features_x, features_y):

        # Flatten (B', D)
        features_x = features_x.flatten(start_dim=0, end_dim=1)
        features_y = features_y.flatten(start_dim=0, end_dim=1)

        # Matmul (B', B')
        features = features_x.matmul(features_y.transpose(0, 1))

        # Diag (B',)
        features_pos = torch.diag(features)

        # Exp -> Sum -> Log: (B',)
        features_all = torch.logsumexp(features, dim=-1)
                        
        # Info NCE Loss: (B',)
        info_nce_loss = features_pos - features_all

        # Accuracy Contrastive
        acc_con = torch.mean(torch.where(features.argmax(dim=-1).cpu() == torch.arange(0, features.shape[0]), 1.0, 0.0))

        return info_nce_loss, acc_con

    def compute_simsiam_loss(self, z_1, z_2, p_1, p_2):
        # Flatten (B', D)
        z_1 = z_1.flatten(start_dim=0, end_dim=1)
        z_2 = z_2.flatten(start_dim=0, end_dim=1)
        p_1 = p_1.flatten(start_dim=0, end_dim=1)
        p_2 = p_2.flatten(start_dim=0, end_dim=1)

        loss_1 = - F.cosine_similarity(p_1, z_2.detach(), dim=-1).mean()
        loss_2 = - F.cosine_similarity(p_2, z_1.detach(), dim=-1).mean()
        loss = (loss_1 + loss_2) / 2.0

        return loss

    def play(self, verbose=False, return_att_w=False):

        # Reset
        obs = self.env_eval.reset()

        # Transfer to device
        state = self.transfer_to_device(obs.state)
        prev_latent = self.transfer_to_device(self.rssm.initial(batch_size=1, seq_length=1, dtype=obs.reward.dtype, detach_learned=True))
        prev_action = self.transfer_to_device(torch.zeros(1, self.env.num_actions, dtype=obs.reward.dtype))

        # Create hidden
        hidden = (prev_latent, prev_action)

        # Init values
        total_rewards = 0
        step = 0

        # att weights
        if return_att_w:
            att_ws = []

        # Episode loop
        while 1:

            # Unpack hidden
            prev_latent, prev_action = hidden

            # Representation Network
            with torch.no_grad():

                # Repr State (1, ...)
                state_proc = self.preprocess_inputs(state.unsqueeze(dim=0), time_stacked=False)
                latent = self.encoder_network(state_proc)
                if self.use_semantic_latent:
                    embedding = self.embedding_network(state_proc)
                    latent = self.merge_semantic_latent(latent, embedding)

                # Unsqueeze Time dim (B, 1, ...)
                latent = {key: value.unsqueeze(dim=1) for key, value in latent.items()}

                # RSSM (B, 1, ...)
                latent, _ = self.rssm(
                    states=latent, 
                    prev_states=prev_latent,
                    prev_actions=prev_action.unsqueeze(dim=1), 
                    is_firsts=torch.zeros(1, 1),
                    return_att_w=return_att_w
                )

                # Get feat (B, Dfeat)
                feat = self.rssm.get_feat(latent).squeeze(dim=1)

                # att weights
                if return_att_w:
                    att_ws.append(latent["att_w"])

                # Policy
                action = self.policy_network(feat).mode()

            # Update Hidden
            latent["hidden"] = self.rssm.slice_hidden(latent["hidden"])
            hidden = (latent, action)

            # Forward Env
            obs = self.env_eval.step(action.argmax(dim=-1).squeeze(dim=0) if self.config.policy_discrete else action.squeeze(dim=0))
            state = self.transfer_to_device(obs.state)
            step += self.env_eval.action_repeat
            total_rewards += obs.reward

            # Done / Time Limit
            if obs.done or step >= self.config.time_limit_eval:
                break

        outputs = AttrDict({"score": total_rewards, "steps": step})
        if return_att_w:
            outputs.att_w = att_ws

        return outputs

    def eval_step(self, inputs, targets, verbose=False):

        # play
        outputs_ = self.play(verbose=verbose)
        outputs = {"score": torch.tensor(outputs_.score), "steps": torch.tensor(outputs_.steps)}

        # Update Infos
        for key, value in outputs.items():
            self.infos["ep_{}".format(key)] = value.item()

        # batch_losses, batch_metrics, batch_truths, batch_preds
        return {}, outputs, {}, {}
    
    def log_figure(self, step, inputs, targets, writer, tag, save_image=False): 

        # Eval Mode
        mode = self.training
        self.eval()

        # Preprocess state (uint8 to float32)
        inputs = self.preprocess_inputs(inputs, time_stacked=True)

        # Unpack Inputs 
        states, actions, rewards, dones, is_firsts, model_steps = inputs[:6]
        
        # Number of Rows
        states = states[:self.config.log_figure_batch]
        actions = actions[:self.config.log_figure_batch]
        is_firsts = is_firsts[:self.config.log_figure_batch]

        with torch.no_grad():

            # Forward Representation Network (B, L, D)
            latent = self.encoder_network(states)
            if self.use_semantic_latent:
                embedding = self.embedding_network(states)
                latent = self.merge_semantic_latent(latent, embedding)

            ###############################################################################
            # Model
            ###############################################################################

            # Model Observe (B, L, D)
            posts, priors = self.rssm.observe(
                states=latent, 
                prev_actions=actions, 
                is_firsts=is_firsts, 
                prev_state=None,
                is_firsts_hidden=None,
            )

            # Get feat (B, L, 2*D)
            feats = self.rssm.get_feat(posts)

            # Rec States (B, L, ...)
            z_size = self.config.stoch_size * self.config.discrete
            posts_z = posts["stoch"][..., :z_size] if self.use_semantic_latent else posts["stoch"]
            states_rec = self.decoder_network(posts_z).mode()

            ###############################################################################
            # Imaginary
            ###############################################################################

            # Initial State
            if self.config.log_figure_context_frames == 0:
                # No context, No hidden
                prev_state = self.transfer_to_device(self.rssm.initial(batch_size=feats.shape[0], seq_length=1, dtype=feats.dtype))
            else:
                # context + hidden
                hidden_len = self.rssm.get_hidden_len(posts["hidden"])
                prev_state = {k: [
                    (
                        v_blk[0][:, max(0, hidden_len-self.config.L+self.config.log_figure_context_frames-self.config.att_context_left):hidden_len-self.config.L+self.config.log_figure_context_frames], 
                        v_blk[1][:, max(0, hidden_len-self.config.L+self.config.log_figure_context_frames-self.config.att_context_left):hidden_len-self.config.L+self.config.log_figure_context_frames]
                    ) for v_blk in v] if k == "hidden" else v[:, self.config.log_figure_context_frames-1:self.config.log_figure_context_frames] for k, v in posts.items()
                }

            # Model Imagine (B, 1+L-C, D)
            img_states = self.rssm.imagine(
                p_net=self.policy_network, 
                prev_state=prev_state, 
                img_steps=self.config.L-self.config.log_figure_context_frames,
                is_firsts=None,
                is_firsts_hidden=None
            )

            # Img States (B, L, ...)
            img_stoch = img_states["stoch"][..., :z_size] if self.use_semantic_latent else img_states["stoch"]
            states_img = self.decoder_network(torch.cat([
                posts_z[:, :self.config.log_figure_context_frames],
                img_stoch[:, 1:]
            ], dim=1)).mode()

        # Shift to 0 1
        states_shift = states.clip(-0.5, 0.5) + 0.5
        states_rec_shift = states_rec.clip(-0.5, 0.5) + 0.5
        error_shift = 1 - torch.abs(states_rec_shift - states_shift).mean(dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
        states_img_shift = states_img.clip(-0.5, 0.5) + 0.5
        
        # Expand is Firsts
        is_firsts = is_firsts.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(states) * states_shift

        # Concat Outputs
        outputs = torch.concat([
            is_firsts, 
            states_shift, 
            states_rec_shift, 
            error_shift, 
            states_img_shift,
        ], dim=1).flatten(start_dim=0, end_dim=1)

        # Add Figure to logs
        if writer != None:

            # Log Image
            fig = torchvision.utils.make_grid(outputs, nrow=self.config.L, normalize=False, scale_each=False).cpu()
            writer.add_image(tag, fig, step)

        # Default Mode
        self.train(mode=mode)
