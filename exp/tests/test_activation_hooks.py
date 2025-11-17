import pytest

from ultralytics import YOLO
from ultralytics.models.rtirca.activation_hooks import ActivationHooks
from ultralytics.utils import ASSETS


@pytest.fixture
def model():
    """Create and return a RT-IRCA model for testing."""
    return YOLO("exp/config/rtirca11n.yaml")


@pytest.fixture
def activation_hooks():
    """Create and return an ActivationHooks instance for testing."""
    return ActivationHooks()


@pytest.fixture
def activation_layers():
    """Define the layers to hook for testing."""
    return [14, 17, 20, 23, 24]


@pytest.fixture
def registered_hooks(model, activation_hooks, activation_layers):
    """Register hooks and return the activation_hooks instance."""
    activation_hooks.register_hooks(model.model, activation_layers)
    return activation_hooks


def test_activation_hooks_registration(registered_hooks, activation_layers):
    """Test that activation hooks can be properly registered to the model."""
    # Verify all specified layers are registered
    assert len(registered_hooks.hooks) == len(activation_layers), (
        f"Expected {len(activation_layers)} hooks, got {len(registered_hooks.hooks)}"
    )

    # Verify activations are empty before inference
    assert hasattr(registered_hooks, "activations"), "Activations attribute not found"
    assert len(registered_hooks.activations) == 0, "Activations should be empty before inference"


def test_activation_hooks_capture(model, registered_hooks, activation_layers):
    """Test that activation hooks properly capture feature maps during inference."""
    # Run inference (this is when activations are captured)
    results = model(ASSETS / "bus.jpg")

    # Verify inference results are valid
    assert len(results) > 0, "No results from inference"

    # Verify activations were captured
    assert len(registered_hooks.activations) > 0, "No activations captured"

    # Verify all expected layers have activations
    assert len(registered_hooks.activations) == len(activation_layers), "Not all activations captured"

    # Verify each layer has activations with reasonable shape
    for layer_id in activation_layers:
        assert str(layer_id) in registered_hooks.activations, f"No activations captured for layer {layer_id}"

        # Verify activation shape is reasonable (not empty)
        activation = registered_hooks.activations[str(layer_id)]

        # Handle case where activation might be a tuple (common in detection models)
        if isinstance(activation, tuple):
            # Use the first element of the tuple which is typically the feature map
            activation_shape = activation[0].shape if len(activation) > 0 else None
        else:
            # Direct tensor case
            activation_shape = activation.shape

        assert activation_shape is not None and len(activation_shape) >= 3, (
            f"Unexpected activation for layer {layer_id}: {type(activation)}"
        )

        # Print activation shapes for debugging (optional)
        print(
            f"Layer {layer_id} activation shape: {activation_shape}, type: {type(registered_hooks.activations[str(layer_id)])}"
        )


def test_activation_hooks_cleanup(registered_hooks):
    """Test that activation hooks can be properly removed."""
    # Verify hooks are registered
    assert len(registered_hooks.hooks) > 0, "No hooks registered"

    # Remove hooks
    registered_hooks.remove_hooks()

    # Verify hooks are removed
    assert len(registered_hooks.hooks) == 0, "Hooks not properly removed"

    # Verify activations are cleared
    assert len(registered_hooks.activations) == 0, "Activations not cleared after hook removal"
