import argparse
import sys
import logging
from pathlib import Path

from config import TrainingConfig
from trainer import Trainer
from utils import setup_logging, validate_environment
from data_loader import load_alpaca_data, DataValidationError


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5-0.5B-Instruct model with SFT",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Start training")
    train_parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to training configuration file"
    )
    train_parser.add_argument(
        "-r", "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    train_parser.add_argument(
        "-d", "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training device (default: cuda)"
    )
    train_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    train_parser.add_argument(
        "--skip-dependency-check",
        action="store_true",
        help="Skip dependency validation before training"
    )
    train_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    train_parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name (overrides config)"
    )
    train_parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name (overrides config)"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration and data")
    validate_parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to training configuration file"
    )
    validate_parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only validate data file"
    )
    validate_parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only validate configuration file"
    )
    validate_parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies only"
    )
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument(
        "-i", "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory path"
    )
    export_parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory path"
    )
    
    return parser.parse_args()


def cmd_train(args: argparse.Namespace) -> int:
    """Run training command.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Check dependencies unless skipped
        if not args.skip_dependency_check:
            if not validate_environment():
                return 4  # Device/dependency error
        
        # Load configuration
        config = TrainingConfig.load(args.config)
        
        # Override device if specified
        if args.device:
            config.device = args.device
        
        # Override wandb settings if specified
        if args.no_wandb:
            config.use_wandb = False
        if args.wandb_project:
            config.wandb_project = args.wandb_project
        if args.wandb_run_name:
            config.wandb_run_name = args.wandb_run_name
        
        # Setup trainer
        trainer = Trainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.resume_from_checkpoint(args.resume)
        
        # Run training
        trainer.train()
        
        return 0
    
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return 2
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        return 1
    except DataValidationError as e:
        logging.error(f"Data validation error: {e}")
        return 2
    except Exception as e:
        logging.error(f"Training error: {e}", exc_info=True)
        return 3


def cmd_validate(args: argparse.Namespace) -> int:
    """Run validation command.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        if args.check_deps:
            # Check dependencies only
            if validate_environment():
                return 0
            else:
                return 4
        
        if not args.data_only:
            # Validate configuration
            print("Validating configuration...")
            config = TrainingConfig.load(args.config)
            print("✓ Configuration is valid")
        
        if not args.config_only:
            # Validate data
            print("\nValidating data...")
            config = TrainingConfig.load(args.config)
            data = load_alpaca_data(config.data_file)
            print(f"✓ Data is valid ({len(data)} samples)")
        
        print("\n✓ All validations passed!")
        return 0
    
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        return 2
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return 1
    except DataValidationError as e:
        print(f"✗ Data validation error: {e}")
        return 2
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return 3


def cmd_export(args: argparse.Namespace) -> int:
    """Run export command.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        print(f"Exporting model from {args.checkpoint} to {args.output}")
        
        # Copy checkpoint files
        import shutil
        checkpoint_path = Path(args.checkpoint)
        output_path = Path(args.output)
        
        if not checkpoint_path.exists():
            print(f"✗ Checkpoint not found: {args.checkpoint}")
            return 10
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy all files from checkpoint
        for item in checkpoint_path.iterdir():
            if item.is_file():
                shutil.copy2(item, output_path / item.name)
        
        print(f"✓ Model exported to {args.output}")
        return 0
    
    except Exception as e:
        print(f"✗ Export error: {e}")
        return 3


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG"
    logger = setup_logging(log_level)
    
    # Dispatch command
    if args.command == "train":
        return cmd_train(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "export":
        return cmd_export(args)
    else:
        print("Error: No command specified")
        print("Use -h/--help for usage information")
        return 1


if __name__ == "__main__":
    sys.exit(main())

