"""Training loop and model management for Qwen2.5 SFT."""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from config import TrainingConfig
from data_loader import load_alpaca_data, format_alpaca_for_training
from utils import MetricsLogger, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


class AlpacaDataset(Dataset):
    """Dataset class for Alpaca format data."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 2048
    ):
        """Initialize dataset.
        
        Args:
            data: List of Alpaca samples
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item."""
        sample = self.data[idx]
        messages = format_alpaca_for_training(sample)
        
        # Apply chat template to format the conversation
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Use dynamic padding instead of max_length padding
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class Trainer:
    """Training loop for Qwen2.5 SFT."""
    
    def __init__(self, config: TrainingConfig) -> None:
        """Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.save(str(self.output_dir / "config.json"))
        
        # Setup metrics logger
        self.metrics_logger = MetricsLogger(str(self.output_dir))
        
        # Initialize wandb if enabled
        self.wandb_initialized = False
        if self.config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
        
        # Initialize model, tokenizer, and data
        self._setup_model_and_tokenizer()
        self._setup_data()
        self._setup_optimizer_and_scheduler()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
    def _setup_model_and_tokenizer(self) -> None:
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Load model
        torch_dtype = None
        if self.config.mixed_precision == "bf16":
            torch_dtype = torch.bfloat16
        elif self.config.mixed_precision == "fp16":
            torch_dtype = torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype if torch_dtype else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
        )
        
        # Enable gradient checkpointing to save memory
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        self.model = self.model.to(self.device)
        
        # Log memory-saving settings
        if torch_dtype:
            logger.info(f"Using {self.config.mixed_precision} precision for memory efficiency")
        else:
            logger.warning("Mixed precision not enabled - consider enabling bf16/fp16 to save memory")
        
        logger.info(f"Model loaded on {self.device}")
    
    def _init_wandb(self) -> None:
        """Initialize wandb for experiment tracking."""
        if not WANDB_AVAILABLE:
            logger.warning("wandb not available, skipping initialization")
            return
        
        # Generate run name if not provided
        run_name = self.config.wandb_run_name
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"qwen25-sft-{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            tags=self.config.wandb_tags or [],
            config=self.config.to_dict(),
            dir=str(self.output_dir)
        )
        
        self.wandb_initialized = True
        logger.info(f"Wandb initialized: project={self.config.wandb_project}, run={run_name}")
    
    def _setup_data(self) -> None:
        """Load and prepare training data."""
        logger.info(f"Loading data from: {self.config.data_file}")
        
        data = load_alpaca_data(self.config.data_file)
        
        self.dataset = AlpacaDataset(
            data,
            self.tokenizer,
            self.config.max_length
        )
        
        # Custom collate function for dynamic padding
        def collate_fn(batch):
            """Collate function with dynamic padding to max length in batch."""
            input_ids = [item["input_ids"] for item in batch]
            attention_mask = [item["attention_mask"] for item in batch]
            labels = [item["labels"] for item in batch]
            
            # Find max length in this batch
            max_len = max(len(ids) for ids in input_ids)
            max_len = min(max_len, self.config.max_length)  # Cap at max_length
            
            # Pad to max length in batch (not global max_length)
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            
            for ids, attn, lbls in zip(input_ids, attention_mask, labels):
                pad_length = max_len - len(ids)
                if pad_length > 0:
                    ids = torch.cat([ids, torch.full((pad_length,), pad_token_id, dtype=ids.dtype)])
                    attn = torch.cat([attn, torch.zeros(pad_length, dtype=attn.dtype)])
                    lbls = torch.cat([lbls, torch.full((pad_length,), -100, dtype=lbls.dtype)])  # -100 is ignored in loss
                
                padded_input_ids.append(ids[:max_len])
                padded_attention_mask.append(attn[:max_len])
                padded_labels.append(lbls[:max_len])
            
            return {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_mask),
                "labels": torch.stack(padded_labels)
            }
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        logger.info(f"Dataset loaded: {len(self.dataset)} samples")
    
    def _setup_optimizer_and_scheduler(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Prepare model for training
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Setup scheduler
        total_steps = len(self.dataloader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Training steps per epoch: {len(self.dataloader)}")
        logger.info(f"Total training steps: {total_steps}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        while len(checkpoints) > self.config.save_total_limit:
            old_checkpoint = checkpoints.pop()
            shutil.rmtree(old_checkpoint)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def _save_checkpoint(self, epoch: float) -> None:
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        
        save_checkpoint(
            self.model,
            self.tokenizer,
            self.optimizer,
            self.scheduler,
            str(checkpoint_dir),
            self.global_step,
            epoch,
            self.config.to_dict()
        )
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def train(self) -> None:
        """Run training loop."""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Mixed precision: {self.config.mixed_precision}")
        logger.info(f"Gradient checkpointing: {self.config.gradient_checkpointing}")
        
        self.model.train()
        
        # Setup mixed precision scaler if using fp16
        scaler = None
        if self.config.mixed_precision == "fp16":
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Using FP16 with automatic mixed precision (AMP)")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            
            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass with mixed precision if enabled
                if scaler is not None:
                    # FP16 with AMP
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                        loss = loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass with scaler
                    scaler.scale(loss).backward()
                elif self.config.mixed_precision == "bf16":
                    # BF16 with autocast
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                        loss = loss / self.config.gradient_accumulation_steps
                    
                    loss.backward()
                else:
                    # Standard FP32
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                
                epoch_loss += loss.item()
                
                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Calculate gradient norm before clipping (for logging)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                    
                    # Clip gradients and update parameters
                    if scaler is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Log metrics
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    metrics = {
                        "step": self.global_step,
                        "loss": loss.item(),
                        "learning_rate": current_lr,
                        "gradient_norm": grad_norm.item(),
                        "epoch": epoch + (batch_idx + 1) / len(self.dataloader)
                    }
                    
                    self.metrics_logger.log(metrics)
                    
                    # Log to wandb
                    if self.wandb_initialized:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/learning_rate": current_lr,
                            "train/gradient_norm": grad_norm.item(),
                            "train/epoch": epoch + (batch_idx + 1) / len(self.dataloader),
                            "train/step": self.global_step
                        }, step=self.global_step)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{current_lr:.2e}"
                    })
                    
                    # Save checkpoint
                    if self.global_step % self.config.checkpoint_steps == 0:
                        self._save_checkpoint(epoch + (batch_idx + 1) / len(self.dataloader))
            
            avg_loss = epoch_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
            # Log epoch metrics to wandb
            if self.wandb_initialized:
                wandb.log({
                    "train/epoch_loss": avg_loss,
                    "train/epoch": epoch + 1
                }, step=self.global_step)
        
        # Save final checkpoint
        final_checkpoint_dir = self.output_dir / "checkpoint-final"
        save_checkpoint(
            self.model,
            self.tokenizer,
            self.optimizer,
            self.scheduler,
            str(final_checkpoint_dir),
            self.global_step,
            self.config.num_epochs,
            self.config.to_dict()
        )
        
        logger.info("Training completed!")
        
        # Finish wandb run
        if self.wandb_initialized:
            wandb.finish()
            logger.info("Wandb run finished")
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        self.global_step, self.current_epoch = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler
        )
        
        logger.info(f"Resumed at step {self.global_step}, epoch {self.current_epoch}")

