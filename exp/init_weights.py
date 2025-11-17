"""This module provides functionality for mapping weights from pretrained yolo models to RTIRCA models."""

from __future__ import annotations

import argparse
import logging

import torch

from exp.config.config import MODEL_FILES, MODEL_STRUCTURES
from ultralytics import YOLO

# module logger (do not configure globally here; configure in __main__ when run as script)
logger = logging.getLogger(__name__)


class RTIRCAInit:
    """RTIRCA model initialization, weight mapping and verification."""

    def __init__(self, rtirca_name: str):
        self.rtirca_name = rtirca_name
        self.rtirca_yaml = MODEL_FILES[rtirca_name].get("yaml")
        self.rtirca_ckpt = MODEL_FILES[rtirca_name].get("ckpt")

        # get pretrained YOLO counterpart for RTIRCA
        yolo_name = MODEL_STRUCTURES[rtirca_name].get("pretrained_weights")
        self.yolo_ckpt = MODEL_FILES[yolo_name].get("ckpt")
        if self.yolo_ckpt is None:
            raise KeyError(f"No YOLO checkpoint found for derived model name '{yolo_name}'")

        self.insertion = MODEL_STRUCTURES[rtirca_name].get("insertion")

        # Weights will be lazily initialized, mapped, and verified
        self._yolo: YOLO | None = None
        self._rtirca: YOLO | None = None

    @property
    def yolo(self) -> YOLO:
        """Lazy loading of YOLO model."""
        if self._yolo is None:
            logger.debug("Loading checkpoint %s", self.yolo_ckpt)
            self._yolo = YOLO(self.yolo_ckpt)
        return self._yolo

    @property
    def rtirca(self) -> YOLO:
        """Lazy loading of RTIRCA model."""
        if self._rtirca is None:
            logger.debug("Initializing from configuration %s", self.rtirca_yaml)
            self._rtirca = YOLO(self.rtirca_yaml)
        return self._rtirca

    def map_weights(self) -> None:
        """Map RTIRCA model weights.

        Returns:
            None
        """
        logger.debug("Start mapping %s model weights", self.rtirca_name)

        yolo_state_dict = self.yolo.model.model.state_dict()
        rtirca_state_dict = self.rtirca.model.model.state_dict()

        # Create weight mapping dictionary
        new_state_dict = self._create_weight_mapping(yolo_state_dict, rtirca_state_dict)

        # Load mapped weights and save initialized model
        self.rtirca.model.model.load_state_dict(new_state_dict, strict=False)
        self.rtirca.save(self.rtirca_ckpt)
        logger.debug("%s model weights mapped successfully!", self.rtirca_name)

    def verify_model_weights(self) -> None:
        """Verify if model weights are correctly mapped."""
        yolo_state_dict = self.yolo.model.model.state_dict()
        rtirca_state_dict = self.rtirca.model.model.state_dict()

        logger.debug("Starting weight mapping verification for %s", self.rtirca_name)

        # Statistics
        total_param_groups = 0
        successfully_mapped = 0

        for name, param_tensor in yolo_state_dict.items():
            layer_idx = self._get_layer_index(name)
            if layer_idx is None:
                continue

            total_param_groups += 1
            target_name = self._map_parameter_name(name, layer_idx)

            if (
                target_name in rtirca_state_dict
                and rtirca_state_dict[target_name].shape == param_tensor.shape
                and torch.allclose(param_tensor, rtirca_state_dict[target_name])
            ):
                successfully_mapped += 1

        self._print_verification_results(total_param_groups, successfully_mapped)
        logger.debug("Verification completed for %s", self.rtirca_name)

    def _get_layer_index(self, param_name: str) -> int | None:
        """Extract layer index from parameter name.

        Args:
            param_name: Parameter name in the format 'X.param_name' where X is the layer index

        Returns:
            Optional[int]: Layer index if found, None otherwise
        """
        # param_name is in the form of '0.conv.weight' or 'model.0.conv'
        parts = param_name.split(".")
        # Return the first standalone numeric token
        for p in parts:
            if p.isdigit():
                return int(p)

        return None

    def _map_parameter_name(self, param_name: str, layer_idx: int) -> str:
        """Map parameter name based on layer index.

        Args:
            param_name: Original parameter name from the source model
            layer_idx: Layer index extracted from the parameter name

        Returns:
            str: Mapped parameter name for the target model architecture
        """
        # Replace only the specific numeric token corresponding to the layer index
        parts = param_name.split(".")
        for i, p in enumerate(parts):
            if p.isdigit() and int(p) == layer_idx:
                if layer_idx <= self.insertion:
                    return param_name
                parts[i] = str(layer_idx + 1)
                return ".".join(parts)

        # Fallback behavior: use simple replace (keeps prior semantics)
        if layer_idx <= self.insertion:
            return param_name
        new_layer_idx = layer_idx + 1
        return param_name.replace(f"{layer_idx}.", f"{new_layer_idx}.", 1)

    def _create_weight_mapping(self, yolo_state_dict: dict, rtirca_state_dict: dict) -> dict:
        """Create weight mapping dictionary.

        Args:
            yolo_state_dict: State dictionary of the yolo model
            rtirca_state_dict: State dictionary of the RTIRCA model

        Returns:
            Dict: Mapping dictionary with RTIRCA parameter names as keys and yolo parameter tensors as values
        """
        new_state_dict = {}

        for name, param_tensor in yolo_state_dict.items():
            layer_idx = self._get_layer_index(name)
            if layer_idx is None:
                continue

            target_name = self._map_parameter_name(name, layer_idx)

            # Check if parameter exists and shape matches
            if target_name in rtirca_state_dict and rtirca_state_dict[target_name].shape == param_tensor.shape:
                new_state_dict[target_name] = param_tensor

        return new_state_dict

    def _print_verification_results(self, total_param_groups: int, successfully_mapped: int) -> None:
        """Print verification result statistics.

        Args:
            total_param_groups: Total number of parameter groups in the model
            successfully_mapped: Number of parameter groups successfully mapped
        """
        logger.debug("  Total parameter groups: %d", total_param_groups)
        logger.debug("  Successfully mapped parameter groups: %d", successfully_mapped)


def batch_init_and_verify(model_names: list[str] | None = None) -> None:
    """Batch initialize and verify multiple models.

    This function runs mapping and verification for each provided model name. If no models are provided (None or empty
    list), the function returns early.
    """
    if not model_names:
        logger.info("No models provided to batch_init_and_verify; nothing to do.")
        return

    for name in model_names:
        try:
            logger.debug("Processing model: %s", name)

            # Initialize model
            init = RTIRCAInit(name)
            init.map_weights()

            # Verify weights
            init.verify_model_weights()

        except Exception as e:
            logger.error("Error processing model %s: %s", name, e, exc_info=True)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Batch initialize and verify RTIRCA models for training.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging output")
    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        required=True,
        help=("Models to be initialized."),
    )

    args = parser.parse_args()

    models_to_init = args.models

    # enable debug logging if requested
    if getattr(args, "verbose", False):
        logger.setLevel(logging.DEBUG)

    batch_init_and_verify(models_to_init)
