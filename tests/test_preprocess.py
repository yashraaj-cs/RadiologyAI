"""Tests for image preprocessing and input resolution handling.

Verifies that non-square chest X-ray images are correctly center-cropped,
resized to the weights' native input resolution (224 or 512), and scaled
to the [-1024, 1024] TorchXRayVision pixel intensity range without downloading weights.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torchxrayvision as xrv
from PIL import Image

from app.vision import (
    get_input_resolution,
    pad_to_square_array,
    preprocess_image,
    resolve_model_resolution,
)


@pytest.mark.parametrize(
    "weights_name,expected_size",
    [
        ("densenet121-res224-all", 224),
        ("resnet50-res512-all", 512),
    ],
)
def test_preprocess_synthetic_nonsquare_image(
    tmp_path: Path,
    weights_name: str,
    expected_size: int,
) -> None:
    """Test preprocessing on a synthetic non-square grayscale image for each weights configuration.

    Derives the expected resolution strictly from model_urls metadata without downloading weights.
    Asserts output tensor shape is (1, 1, size, size) and values are bounded in [-1024, 1024].
    """
    # Verify resolution from model_urls metadata
    assert weights_name in xrv.models.model_urls, f"Weights {weights_name} not found in model_urls"
    derived_size = xrv.models.model_urls[weights_name]["input_resolution"]
    assert derived_size == expected_size
    assert get_input_resolution(weights_name) == expected_size

    # Create a synthetic non-square 8-bit grayscale image (height != width)
    # Using a 380x540 landscape image with diverse pixel values across [0, 255]
    height, width = 380, 540
    gradient = np.linspace(0, 255, height * width, dtype=np.uint8).reshape((height, width))
    img = Image.fromarray(gradient, mode="L")

    img_path = tmp_path / f"synthetic_{weights_name}.png"
    img.save(img_path)

    # Preprocess image specifying target resolution derived from weights metadata
    tensor = preprocess_image(img_path, input_size=derived_size)

    # 1. Assert batch and channel dimensions and square resolution
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, expected_size, expected_size)

    # 2. Assert values lie within [-1024, 1024]
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    assert min_val >= -1024.0, f"Minimum value {min_val} below -1024.0"
    assert max_val <= 1024.0, f"Maximum value {max_val} above 1024.0"


def test_preprocess_portrait_nonsquare_image(tmp_path: Path) -> None:
    """Test center cropping and resizing on a portrait non-square image (height > width)."""
    # Create portrait 600x400 image
    height, width = 600, 400
    img_data = np.full((height, width), fill_value=128, dtype=np.uint8)
    img = Image.fromarray(img_data, mode="L")

    img_path = tmp_path / "portrait.jpg"
    img.save(img_path)

    size = 224
    tensor = preprocess_image(img_path, input_size=size)

    assert tensor.shape == (1, 1, size, size)
    assert tensor.min().item() >= -1024.0
    assert tensor.max().item() <= 1024.0


def test_resolve_model_resolution_from_attributes() -> None:
    """Test that resolve_model_resolution reads model.input_resolution and model.weights at call time."""
    class DummyModelWithResolution:
        input_resolution = 512

    class DummyModelWithWeights:
        weights = "resnet50-res512-all"

    class DummyModelFallback:
        pass

    assert resolve_model_resolution(DummyModelWithResolution()) == 512
    assert resolve_model_resolution(DummyModelWithWeights()) == 512
    assert resolve_model_resolution(DummyModelFallback()) == 224


def test_preprocess_380x540_both_modes_and_default_identical(tmp_path: Path) -> None:
    """Test that a synthetic 380x540 image gives square tensors in both modes.

    Verifies:
    1. Default mode output is identical to previous behavior (pad_to_square=False).
    2. In both modes (crop and pad), tensor shape is (1, 1, size, size) for size=224 and size=512.
    3. Padded pixels in the padded array equal the image's minimum value.
    """
    # 1. Create synthetic 380x540 non-square image
    height, width = 380, 540
    # Central rectangle with higher values on a zero background
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[100:280, 100:440] = 200

    img = Image.fromarray(arr, mode="L")
    img_path = tmp_path / "synthetic_380x540.png"
    img.save(img_path)

    # 2. Test default mode vs explicit pad_to_square=False (must be 100% identical)
    tensor_default_224 = preprocess_image(img_path, input_size=224)
    tensor_crop_224 = preprocess_image(img_path, input_size=224, pad_to_square=False)
    assert torch.equal(tensor_default_224, tensor_crop_224), "Default mode differs from pad_to_square=False"

    # 3. Test both modes give square tensors of right size (224 and 512)
    tensor_pad_224 = preprocess_image(img_path, input_size=224, pad_to_square=True)
    assert tensor_crop_224.shape == (1, 1, 224, 224)
    assert tensor_pad_224.shape == (1, 1, 224, 224)

    tensor_crop_512 = preprocess_image(img_path, input_size=512, pad_to_square=False)
    tensor_pad_512 = preprocess_image(img_path, input_size=512, pad_to_square=True)
    assert tensor_crop_512.shape == (1, 1, 512, 512)
    assert tensor_pad_512.shape == (1, 1, 512, 512)


def test_pad_pixels_equal_minimum_value() -> None:
    """Test that pad_to_square_array pads shorter dimension symmetrically with image's minimum value."""
    # Synthetic array with shape (1, 380, 540)
    img_data = np.arange(380 * 540, dtype=np.float32).reshape(1, 380, 540)
    # Normalized range roughly [-1024, 1024]
    img_norm = (img_data / (380 * 540)) * 2048.0 - 1024.0
    min_val = float(img_norm.min())

    padded = pad_to_square_array(img_norm)

    # Shorter dimension is height (380 -> 540, padded by 80 top and 80 bottom)
    assert padded.shape == (1, 540, 540)

    # Top 80 rows must strictly equal minimum value
    assert np.all(padded[0, :80, :] == min_val)

    # Bottom 80 rows must strictly equal minimum value
    assert np.all(padded[0, 460:, :] == min_val)

    # Original center region must be preserved exactly
    assert np.array_equal(padded[0, 80:460, :], img_norm[0])


def test_pad_pixels_portrait_orientation() -> None:
    """Test that pad_to_square_array pads portrait (H > W) width symmetrically with minimum value."""
    # Synthetic array with shape (1, 540, 380)
    img_norm = np.full((1, 540, 380), fill_value=-500.0, dtype=np.float32)
    img_norm[0, 200:300, 100:200] = 500.0
    min_val = float(img_norm.min())

    padded = pad_to_square_array(img_norm)

    # Shorter dimension is width (380 -> 540, padded by 80 left and 80 right)
    assert padded.shape == (1, 540, 540)
    assert np.all(padded[0, :, :80] == min_val)
    assert np.all(padded[0, :, 460:] == min_val)
    assert np.array_equal(padded[0, :, 80:460], img_norm[0])

