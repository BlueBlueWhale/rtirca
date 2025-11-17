"""Activation hooks utility for PyTorch models."""

from __future__ import annotations

from typing import Any, Callable

import torch


class ActivationHooks:
    """Manages forward hooks to capture layer activations in PyTorch models."""

    def __init__(self):
        """Initialize with empty activations and hooks."""
        self.activations: dict[str, torch.Tensor] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(self, model: torch.nn.Module, activation_layers: list[int], prefix: str = "") -> None:
        """Register hooks on specified layer indices."""
        self.remove_hooks()  # Remove any existing hooks first

        for idx in activation_layers:
            if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
                # For models with a 'model' attribute (like DetectionModel)
                layer = model.model[idx]
            else:
                # For regular torch modules
                layer = model[idx]

            hook = layer.register_forward_hook(self._get_activation_hook(f"{prefix}{idx}"))
            self.hooks.append(hook)

    def register_hooks_by_names(self, model: torch.nn.Module, layer_names: list[str]) -> None:
        """Register hooks on layers specified by name."""
        self.remove_hooks()

        for name in layer_names:
            layer = self._get_layer_by_name(model, name)
            if layer is not None:
                hook = layer.register_forward_hook(self._get_activation_hook(name))
                self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all hooks and clear activations."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.clear_activations()

    def clear_activations(self) -> None:
        """Clear activation dictionary."""
        self.activations.clear()

    def _get_activation_hook(self, name: str) -> Callable:
        """Create hook function to capture activations."""

        def hook(module: torch.nn.Module, inputs: Any, outputs: torch.Tensor) -> None:
            self.activations[name] = outputs

        return hook

    def _get_layer_by_name(self, model: torch.nn.Module, layer_name: str) -> torch.nn.Module | None:
        """Get layer by name from model."""
        if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
            model_to_search = model.model
        else:
            model_to_search = model

        # Try to get the layer by name
        try:
            return dict(model_to_search.named_modules())[layer_name]
        except KeyError:
            # Try to get by index if layer_name is numeric
            try:
                idx = int(layer_name)
                if hasattr(model_to_search, "__getitem__"):
                    return model_to_search[idx]
            except (ValueError, IndexError):
                pass

        return None

    def get_activation(self, name: str) -> torch.Tensor:
        """Get activation tensor by name."""
        return self.activations[name]

    def has_activation(self, name: str) -> bool:
        """Check if activation exists."""
        return name in self.activations

    def get_activation_names(self) -> list[str]:
        """Get a list of all activation names."""
        return list(self.activations.keys())


def create_activation_hooks() -> ActivationHooks:
    """Create new ActivationHooks instance."""
    return ActivationHooks()
