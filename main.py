# Solve dm_control bug
import os
os.environ["MUJOCO_GL"] = "egl"

# PyTorch
import torch

# Functions
import functions

# Other
import os
import argparse
import importlib
import warnings

# Disable Warnings
warnings.filterwarnings("ignore")

def main(args):

    ###############################################################################
    # Init
    ###############################################################################

    # Print Mode
    print("Mode: {}".format(args.mode))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Load Config
    args.config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))

    # Load Model
    model = functions.load_model(args)

    # Optional torch.compile on major submodules
    if args.compile:
        if torch.cuda.is_available():
            compile_kwargs = {"dynamic": False}
            compile_targets = [
                "encoder_network",
                "decoder_network",
            ]
            for name in compile_targets:
                if hasattr(model, name):
                    setattr(model, name, torch.compile(getattr(model, name), **compile_kwargs))
            print("torch.compile enabled for model submodules.")
        else:
            print("torch.compile requested but CUDA not available; skipping.")

    # Load Dataset
    dataset_train, dataset_eval = functions.load_datasets(args)

    ###############################################################################
    # Modes
    ###############################################################################

    # Training
    if args.mode == "training":

        model.fit(
            dataset_train=dataset_train, 
            epochs=getattr(args.config, "epochs", 1000), 
            dataset_eval=dataset_eval, 
            initial_epoch=int(args.checkpoint.split("_")[2]) if args.checkpoint != None else 0, 
            callback_path=args.config.callback_path,
            precision=getattr(args.config, "precision", torch.float32),
            accumulated_steps=getattr(args.config, "accumulated_steps", 1),
            eval_period_step=getattr(args.config, "eval_period_step", args.eval_period_step),
            eval_period_epoch=getattr(args.config, "eval_period_epoch", args.eval_period_epoch),
            saving_period_epoch=getattr(args.config, "saving_period_epoch", args.saving_period_epoch),
            log_figure_period_step=getattr(args.config, "log_figure_period_step", args.log_figure_period_step),
            log_figure_period_epoch=getattr(args.config, "log_figure_period_epoch", args.log_figure_period_epoch),
            step_log_period=args.step_log_period,
            grad_init_scale=getattr(args.config, "grad_init_scale", 65536.0),
            detect_anomaly=getattr(args.config, "detect_anomaly", args.detect_anomaly),
            recompute_metrics=getattr(args.config, "recompute_metrics", False),
            wandb_logging=args.wandb,
            verbose_progress_bar=args.verbose_progress_bar,
            keep_last_k=args.keep_last_k,
            enable_profiling=getattr(args.config, "enable_profiling", False),
        )

        final_eval_episodes = getattr(model.config, 'final_eval_episodes', model.config.eval_episodes)
        dataset_eval.dataset.num_steps = final_eval_episodes
        print(f"\n{'='*50}\nFinal Evaluation with {final_eval_episodes} episodes\n{'='*50}")
        from torch.utils.tensorboard import SummaryWriter
        final_writer = SummaryWriter(os.path.join(args.config.callback_path, "logs"))
        model._evaluate(
            dataset_eval,
            writer=final_writer,
            recompute_metrics=getattr(args.config, "recompute_metrics", False),
            tag="Final-Evaluation",
            verbose_progress_bar=args.verbose_progress_bar
        )

    # Evaluation
    elif args.mode == "evaluation":

        final_eval_episodes = getattr(model.config, 'final_eval_episodes', model.config.eval_episodes)
        dataset_eval.dataset.num_steps = final_eval_episodes
        model._evaluate(
            dataset_eval, 
            writer=None,
            recompute_metrics=getattr(args.config, "recompute_metrics", False),
            verbose_progress_bar=args.verbose_progress_bar,
        )

    # Pass
    elif args.mode == "pass":
        pass

if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",          type=str,   default="configs/twister.py",                                       help="Python configuration file containing model hyperparameters")
    parser.add_argument("-m", "--mode",                 type=str,   default="training", choices=["training", "evaluation", "pass"],     help="Mode: training, validation-clean, test-clean, eval_time-dev-clean, ...")
    parser.add_argument("-i", "--checkpoint",           type=str,   default=None,                                                       help="Load model from checkpoint name")
    parser.add_argument("--cpu",                        action="store_true",                                                            help="Load model on cpu")
    parser.add_argument("--load_last",                  action="store_true",                                                            help="Load last model checkpoint")
    parser.add_argument("--wandb",                      action="store_true",                                                            help="Initialize wandb logging")
    parser.add_argument("--verbose_progress_bar",       type=int,   default=1,                                                          help="Verbose level of progress bar display")

    # Training
    parser.add_argument("--saving_period_epoch",        type=int,   default=1,                                                          help="Model saving every 'n' epochs")
    parser.add_argument("--log_figure_period_step",     type=int,   default=None,                                                       help="Log figure every 'n' steps")
    parser.add_argument("--log_figure_period_epoch",    type=int,   default=1,                                                          help="Log figure every 'n' epochs")
    parser.add_argument("--step_log_period",            type=int,   default=100,                                                        help="Training step log period")
    parser.add_argument("--keep_last_k",                type=int,   default=3,                                                          help="Keep last k checkpoints")

    # Eval
    parser.add_argument("--eval_period_epoch",          type=int,   default=5,                                                          help="Model evaluation every 'n' epochs")
    parser.add_argument("--eval_period_step",           type=int,   default=None,                                                       help="Model evaluation every 'n' steps")

    # Info
    parser.add_argument("--show_dict",                  action="store_true",                                                            help="Show model dict summary")
    parser.add_argument("--show_modules",               action="store_true",                                                            help="Show model named modules")
    
    # Debug
    parser.add_argument("--detect_anomaly",             action="store_true",                                                            help="Enable or disable the autograd anomaly detection")
    parser.add_argument("--compile",                    action="store_true",                                                            help="Enable torch.compile for major model submodules")
    
    # Parse Args
    args = parser.parse_args()

    # Run main
    main(args)
