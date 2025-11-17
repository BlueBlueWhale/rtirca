from __future__ import annotations

from pathlib import Path

from .model import RTIRCA
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK


class RTIRCATrainer(DetectionTrainer):
    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> RTIRCA:
        """Return RTIRCAModel initialized with specified config and weights."""
        # Ensure float parameters are correctly converted from YAML
        alpha = float(self.args.alpha) if hasattr(self.args, 'alpha') else 1e-3
        beta = float(self.args.beta) if hasattr(self.args, 'beta') else 1e-8
        gamma = float(self.args.gamma) if hasattr(self.args, 'gamma') else 1e-6
        temperature = float(self.args.temperature) if hasattr(self.args, 'temperature') else 0.5
        
        model = RTIRCA(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            temperature=temperature,
            has_rev=self.args.has_rev,
            has_irca=self.args.has_irca,
            has_mut=self.args.has_mut,
            activation_layers=self.args.activation_layers,
            student_channels=self.args.student_channels,
            teacher_channels=self.args.teacher_channels,
            teacher_ckpt=self.args.teacher_ckpt,
        )
        if weights:
            model.load(weights)

        return model

    def save_model(self):
        # Remove hooks and criterion from EMA model to avoid thread lock issues during deepcopy
        criterion = getattr(self.ema.ema, "criterion", None)

        # Remove hooks only if EMA model's criterion exists and is not None
        if criterion is not None:
            student_activation_hooks = getattr(criterion, "student_activation_hooks", None)
            if student_activation_hooks is not None:
                student_activation_hooks.remove_hooks()
            teacher_activation_hooks = getattr(criterion, "teacher_activation_hooks", None)
            if teacher_activation_hooks is not None:
                teacher_activation_hooks.remove_hooks()

        # Delete EMA model's criterion if exists
        if hasattr(self.ema.ema, "criterion"):
            delattr(self.ema.ema, "criterion")

        super().save_model()
