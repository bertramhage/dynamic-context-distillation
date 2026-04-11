"""Teacher-student distillation training loop for the hypernetwork.

Teacher = Chronos-2 with full long+short context (frozen).
Student = Chronos-2 with short context + LoRA from hypernetwork.

Loss = KL divergence between teacher and student quantile predictions.
Only hypernetwork parameters are trained.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import wandb

from src.training.hypernet import HyperLoRA
from src.training.lora_injection import apply_lora_to_model, remove_lora
from src.orchestration.context_encoder import ChronosContextEncoder


def quantile_kl_divergence(
    teacher_quantiles: torch.Tensor,
    student_quantiles: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between teacher and student quantile predictions.

    Chronos-2 outputs quantile values (not probabilities), so we convert
    to a pseudo-distribution by treating adjacent quantile differences as
    bin probabilities (histogram-style), then compute KL.

    Args:
        teacher_quantiles: [batch, n_quantiles, pred_steps]
        student_quantiles: [batch, n_quantiles, pred_steps]

    Returns:
        Scalar loss averaged over batch and prediction steps.
    """
    # Compute bin widths (differences between adjacent quantiles)
    # This converts quantile values to something proportional to PDF
    eps = 1e-8

    # Use softmax on negative squared differences from median to create
    # a proper distribution from quantile values.
    # Simpler approach: use MSE on quantile values directly as a proxy.
    # KL on quantile functions is equivalent to Wasserstein-like distance.
    # For simplicity and stability, use smooth L1 on quantile values.
    loss = F.smooth_l1_loss(student_quantiles, teacher_quantiles, reduction="none")
    return loss.mean()


def _compute_num_output_patches(prediction_length: int, output_patch_size: int) -> int:
    """Compute number of output patches needed for a given prediction length."""
    return math.ceil(prediction_length / output_patch_size)


class HypernetTrainer:
    """Training loop for hypernetwork distillation."""

    def __init__(
        self,
        hypernetwork: HyperLoRA,
        pipeline,
        context_encoder: ChronosContextEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        lora_scaling: float = 2.0,
        max_epochs: int = 50,
        patience: int = 5,
        checkpoint_dir: str = "checkpoints",
        l1_reg_coef: float = 0.0,
        grad_clip: float = 1.0,
        log_every: int = 10,
        device: str = "cpu",
        gradient_accumulation_steps: int = 8,
        warmup_steps: int = 100,
        train_teacher_cache: torch.Tensor | None = None,
        val_teacher_cache: torch.Tensor | None = None,
    ):
        self.hypernetwork = hypernetwork.to(device)
        self.pipeline = pipeline
        self.model = pipeline.model
        self.context_encoder = context_encoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lora_scaling = lora_scaling
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.l1_reg_coef = l1_reg_coef
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.device = device

        self.grad_accum_steps = gradient_accumulation_steps
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.train_teacher_cache = train_teacher_cache
        self.val_teacher_cache = val_teacher_cache

        # Optimizer — only hypernetwork params
        self.optimizer = torch.optim.AdamW(
            hypernetwork.parameters(), lr=lr, weight_decay=weight_decay,
        )

        # LR scheduler: linear warmup then cosine decay
        total_steps = max_epochs * len(train_loader) // gradient_accumulation_steps
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=1e-2, total_iters=warmup_steps),
                CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7),
            ],
            milestones=[warmup_steps],
        )

        # Output patch size from model config
        self.output_patch_size = self.model.chronos_config.output_patch_size

        # Freeze base model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _lookup_cached_teacher_preds(
        self,
        cache_tensor: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Fetch cached teacher predictions for a batch of sample indices."""
        cpu_idx = sample_indices.to(dtype=torch.int64, device="cpu")
        cached = cache_tensor.index_select(0, cpu_idx)
        return cached.to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def _teacher_forward(
        self, teacher_contexts: torch.Tensor, prediction_length: int,
    ) -> torch.Tensor:
        """Run teacher (full context) and return quantile predictions.

        Args:
            teacher_contexts: [batch, n_queries, teacher_ctx_len]
            prediction_length: number of forecast steps

        Returns:
            [batch, n_queries, n_quantiles, pred_steps]
        """
        B, Q, T = teacher_contexts.shape
        # Flatten batch and queries
        flat = teacher_contexts.reshape(B * Q, T).to(self.device)
        num_patches = _compute_num_output_patches(prediction_length, self.output_patch_size)

        out = self.model.forward(flat, num_output_patches=num_patches)
        # quantile_preds: [B*Q, n_quantiles, total_steps]
        qp = out.quantile_preds[:, :, :prediction_length]
        n_q = qp.shape[1]
        return qp.reshape(B, Q, n_q, prediction_length)

    def _student_forward(
        self,
        short_contexts: torch.Tensor,
        lora_dict: dict[str, dict[str, torch.Tensor]],
        prediction_length: int,
    ) -> torch.Tensor:
        """Run student (short context + LoRA) and return quantile predictions.

        Args:
            short_contexts: [batch, n_queries, short_ctx_len]
            lora_dict: hypernetwork output for this batch
            prediction_length: number of forecast steps

        Returns:
            [batch, n_queries, n_quantiles, pred_steps]
        """
        B, Q, T = short_contexts.shape
        num_patches = _compute_num_output_patches(prediction_length, self.output_patch_size)

        # Flatten queries so student runs once on [B*Q, T] instead of Q runs on [B, T].
        flat_ctx = short_contexts.to(self.device).reshape(B * Q, T)

        # LoRA is generated per context (B); repeat each context's LoRA Q times
        # to align with flattened query rows [b0q0, b0q1, ..., b1q0, ...].
        expanded_lora = {
            module_name: {
                "A": module_weights["A"].repeat_interleave(Q, dim=0),
                "B": module_weights["B"].repeat_interleave(Q, dim=0),
            }
            for module_name, module_weights in lora_dict.items()
        }

        patches = apply_lora_to_model(self.model, expanded_lora, self.lora_scaling)
        try:
            out = self.model.forward(flat_ctx, num_output_patches=num_patches)
        finally:
            remove_lora(patches)

        qp = out.quantile_preds[:, :, :prediction_length]
        n_q = qp.shape[1]
        return qp.reshape(B, Q, n_q, prediction_length)

    def _compute_l1_reg(self, lora_dict: dict) -> torch.Tensor:
        """L1 regularization on generated LoRA weights."""
        l1 = torch.tensor(0.0, device=self.device)
        n = 0
        for module_weights in lora_dict.values():
            l1 = l1 + module_weights["A"].abs().mean()
            l1 = l1 + module_weights["B"].abs().mean()
            n += 2
        return l1 / max(n, 1)

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average loss."""
        self.hypernetwork.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            long_ctx = batch["long_context"].to(self.device)       # [B, long_len]
            short_ctx = batch["short_contexts"]                     # [B, Q, short_len]
            sample_indices = batch["sample_indices"]
            prediction_length = batch["targets"].shape[-1]

            # 1. Encode long context through frozen context encoder
            with torch.no_grad():
                ctx_features = self.context_encoder.encode_last_hidden(long_ctx)

            # 2. Generate LoRA weights via hypernetwork
            lora_dict = self.hypernetwork(ctx_features)

            # 3. Teacher predictions (cached, no grad)
            if self.train_teacher_cache is not None:
                teacher_preds = self._lookup_cached_teacher_preds(
                    self.train_teacher_cache, sample_indices,
                )
            else:
                teacher_ctx = batch["teacher_contexts"]
                teacher_preds = self._teacher_forward(teacher_ctx, prediction_length)

            # 4. Student predictions (grad flows through LoRA -> hypernetwork)
            student_preds = self._student_forward(short_ctx, lora_dict, prediction_length)

            # 5. Loss (scaled for gradient accumulation)
            loss = quantile_kl_divergence(teacher_preds.detach(), student_preds)

            if self.l1_reg_coef > 0:
                l1_reg = self._compute_l1_reg(lora_dict)
                loss = loss + self.l1_reg_coef * l1_reg

            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.hypernetwork.parameters(), self.grad_clip,
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

            if wandb.run is not None and (batch_idx + 1) % self.log_every == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/step": epoch * len(self.train_loader) + batch_idx,
                })

            if (batch_idx + 1) % self.log_every == 0:
                avg = total_loss / n_batches
                print(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={loss.item():.6f} avg={avg:.6f}"
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Run validation. Returns average loss."""
        if self.val_loader is None:
            return float("inf")

        self.hypernetwork.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            long_ctx = batch["long_context"].to(self.device)
            short_ctx = batch["short_contexts"]
            sample_indices = batch["sample_indices"]
            prediction_length = batch["targets"].shape[-1]

            ctx_features = self.context_encoder.encode_last_hidden(long_ctx)

            lora_dict = self.hypernetwork(ctx_features)

            if self.val_teacher_cache is not None:
                teacher_preds = self._lookup_cached_teacher_preds(
                    self.val_teacher_cache, sample_indices,
                )
            else:
                teacher_ctx = batch["teacher_contexts"]
                teacher_preds = self._teacher_forward(teacher_ctx, prediction_length)
            student_preds = self._student_forward(short_ctx, lora_dict, prediction_length)

            loss = quantile_kl_divergence(teacher_preds, student_preds)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if wandb.run is not None:
            wandb.log({"val/loss": avg_loss, "val/epoch": epoch})

        return avg_loss

    def train(self) -> Path:
        """Full training loop with early stopping. Returns best checkpoint path."""
        best_val_loss = float("inf")
        patience_counter = 0
        best_path = self.checkpoint_dir / "best_hypernet.pt"

        print(f"Starting training for up to {self.max_epochs} epochs")
        print(f"  Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Train teacher cache: {self.train_teacher_cache is not None}")
        if self.val_loader is not None:
            print(f"  Val teacher cache: {self.val_teacher_cache is not None}")
        print(f"  Hypernetwork params: {sum(p.numel() for p in self.hypernetwork.parameters()):,}")

        for epoch in range(self.max_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} time={elapsed:.1f}s"
            )

            if wandb.run is not None:
                wandb.log({
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "train/epoch_time": elapsed,
                })

            # Early stopping on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.hypernetwork, best_path)
                print(f"  Saved best checkpoint (val_loss={val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        # Save final checkpoint
        final_path = self.checkpoint_dir / "final_hypernet.pt"
        torch.save(self.hypernetwork, final_path)
        print(f"Saved final checkpoint to {final_path}")
        print(f"Best checkpoint at {best_path} (val_loss={best_val_loss:.6f})")

        return best_path
