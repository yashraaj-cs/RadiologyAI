"""Vision module for loading TorchXRayVision models and running inference.

Loads the pretrained TorchXRayVision DenseNet classifier (cached in memory)
and applies official 2D chest X-ray image preprocessing and transforms.
"""

import logging
import sys
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import torch
import torchvision
import torchxrayvision as xrv

from app.config import MODEL_WEIGHTS

logger = logging.getLogger(__name__)

# Module-level cache for the loaded vision model
_CACHED_MODEL: Any = None
_CACHED_WEIGHTS: str | None = None

# Supported 2D image extensions
VALID_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg"}


def detect_unsupported_pathologies(model: Any) -> list[str]:
    """Detect pathologies whose operating threshold (op_threshs) is NaN or missing.

    Pathologies with NaN operating points are uncalibrated and return uninformative
    constant scores (0.5 in TorchXRayVision's op_norm).

    Args:
        model: TorchXRayVision model instance.

    Returns:
        List of pathology names with NaN operating thresholds.
    """
    threshs = getattr(model, "op_threshs", None)
    if threshs is None:
        weights_name = getattr(model, "weights", None)
        if weights_name and hasattr(xrv.models, "model_urls") and weights_name in xrv.models.model_urls:
            threshs = xrv.models.model_urls[weights_name].get("op_threshs")

    if threshs is None:
        return []

    if isinstance(threshs, torch.Tensor):
        threshs_arr = threshs.detach().cpu().numpy().flatten()
    else:
        threshs_arr = np.array(threshs).flatten()

    pathologies = getattr(model, "pathologies", [])
    if not pathologies and hasattr(xrv.models, "model_urls"):
        weights_name = getattr(model, "weights", None)
        if weights_name and weights_name in xrv.models.model_urls:
            pathologies = xrv.models.model_urls[weights_name].get("labels", [])

    unsupported = [
        pathology
        for pathology, thresh in zip(pathologies, threshs_arr)
        if np.isnan(thresh)
    ]
    return unsupported


def get_input_resolution(weights: str | None = None) -> int:
    """Derive input resolution from model_urls metadata for the given weights.

    Falls back to 224 if weights or input_resolution is not specified in model_urls.

    Args:
        weights: Optional weights name (defaults to MODEL_WEIGHTS from config).

    Returns:
        Square input resolution (e.g., 224 for DenseNet, 512 for ResNet).
    """
    target_weights = weights or MODEL_WEIGHTS
    if hasattr(xrv.models, "model_urls") and target_weights in xrv.models.model_urls:
        res = xrv.models.model_urls[target_weights].get("input_resolution")
        if res is not None:
            return int(res)
    return 224


def resolve_model_resolution(model: Any) -> int:
    """Resolve input resolution from loaded model instance or its weights at call time.

    Derives the required resolution directly from the model object or model_urls
    metadata at invocation time, never relying on stale module-level caches.

    Args:
        model: TorchXRayVision model instance.

    Returns:
        Square input resolution (e.g., 224 for DenseNet, 512 for ResNet).
    """
    res = getattr(model, "input_resolution", None)
    if res is not None:
        return int(res)

    weights_name = getattr(model, "weights", None)
    if weights_name and hasattr(xrv.models, "model_urls") and weights_name in xrv.models.model_urls:
        url_meta = xrv.models.model_urls[weights_name]
        if "input_resolution" in url_meta:
            return int(url_meta["input_resolution"])

    return 224


def load_model(weights: str | None = None) -> Any:
    """Load and cache the pretrained TorchXRayVision classifier.

    Supports both DenseNet and ResNet architectures based on weights name.
    Reuses the cached instance on subsequent calls with matching weights.

    Args:
        weights: Optional model weights identifier. Defaults to MODEL_WEIGHTS
                 from app.config.

    Returns:
        Evaluated, cached model instance.
    """
    global _CACHED_MODEL, _CACHED_WEIGHTS

    target_weights: str = weights or MODEL_WEIGHTS

    # Return cached model if already loaded with identical weights
    if _CACHED_MODEL is not None and _CACHED_WEIGHTS == target_weights:
        return _CACHED_MODEL

    # Instantiate model using get_model to support densenet or resnet families
    if hasattr(xrv.models, "get_model"):
        model = xrv.models.get_model(weights=target_weights)
    elif target_weights.startswith("resnet"):
        model = xrv.models.ResNet(weights=target_weights)
    else:
        model = xrv.models.DenseNet(weights=target_weights)

    model.eval()

    # Detect pathologies whose op_threshs entry is NaN and record them as unsupported
    unsupported = detect_unsupported_pathologies(model)
    setattr(model, "unsupported_pathologies", unsupported)
    if unsupported:
        logger.warning(
            "Model '%s' has unsupported pathologies with NaN op_threshs (dropped from predictions): %s",
            target_weights,
            unsupported,
        )

    _CACHED_MODEL = model
    _CACHED_WEIGHTS = target_weights
    return _CACHED_MODEL


def pad_to_square_array(img: np.ndarray) -> np.ndarray:
    """Pad the shorter side of an image symmetrically with its minimum value to make it square.

    Args:
        img: Input NumPy array with shape (1, H, W) or (H, W).

    Returns:
        Square NumPy array of shape (1, max(H, W), max(H, W)) or (max(H, W), max(H, W)).
    """
    has_channel = img.ndim == 3
    h = img.shape[-2]
    w = img.shape[-1]
    if h == w:
        return img

    min_val = float(img.min())

    if h < w:
        diff = w - h
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        pad_left, pad_right = 0, 0
    else:
        diff = h - w
        pad_left = diff // 2
        pad_right = diff - pad_left
        pad_top, pad_bottom = 0, 0

    if has_channel:
        pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    else:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))

    return np.pad(img, pad_width, mode="constant", constant_values=min_val)


def preprocess_image(
    image_path: str | Path,
    input_size: int | None = None,
    pad_to_square: bool = False,
) -> torch.Tensor:
    """Load a 2D chest X-ray image and apply TorchXRayVision preprocessing.

    Pipeline:
    1. Open image with PIL and convert to 8-bit grayscale ('L').
    2. Normalize 8-bit [0, 255] values to [-1024, 1024] using xrv.datasets.normalize.
    3. Add channel dimension: (1, H, W).
    4. Make square: either pad the shorter side with min value (if pad_to_square=True)
       or apply XRayCenterCrop (default: pad_to_square=False), then resize to input_size.
    5. Convert to PyTorch Tensor with batch dimension: (1, 1, input_size, input_size).

    Args:
        image_path: Path to PNG or JPEG image.
        input_size: Target square image dimension (default: derived from weights metadata).
        pad_to_square: If True, pad shorter side symmetrically with image's minimum value
                       instead of center-cropping. Default is False.

    Returns:
        PyTorch float tensor with shape (1, 1, input_size, input_size).
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    target_size = input_size if input_size is not None else get_input_resolution()

    # 1. Convert to 8-bit grayscale
    pil_img = Image.open(path).convert("L")
    img_np = np.array(pil_img, dtype=np.float32)

    # 2. Normalize 8-bit values to [-1024, 1024]
    img_norm = xrv.datasets.normalize(img_np, maxval=255.0)

    # 3. Add channel dimension: (1, H, W)
    img_ch = img_norm[None, ...]

    # 4. Make square (pad or crop) and resize to target_size
    if pad_to_square:
        img_padded = pad_to_square_array(img_ch)
        resizer = xrv.datasets.XRayResizer(target_size)
        img_transformed = resizer(img_padded)
    else:
        transforms = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(target_size),
        ])
        img_transformed = transforms(img_ch)

    # 5. Convert to tensor with batch dimension: (1, 1, target_size, target_size)
    tensor = torch.from_numpy(img_transformed).unsqueeze(0).float()
    return tensor


def predict(
    image_path: str | Path,
    model: Any = None,
    pad_to_square: bool = False,
) -> dict[str, float]:
    """Preprocess a 2D chest X-ray image and return pathology model scores.

    Args:
        image_path: Path to the image file (PNG/JPEG).
        model: Preloaded TorchXRayVision model, or None to load default cached model.
        pad_to_square: If True, pad shorter side to square instead of center-cropping.

    Returns:
        Dictionary mapping pathology names to model scores (0.0 to 1.0).
    """
    net = model if model is not None else load_model()
    target_res = resolve_model_resolution(net)
    tensor = preprocess_image(image_path, input_size=target_res, pad_to_square=pad_to_square)

    # Assert that preprocessed tensor spatial dimensions equal model's native resolution
    _, _, h, w = tensor.shape
    assert h == target_res and w == target_res, (
        f"Preprocessed tensor spatial dimensions ({h}, {w}) do not match "
        f"model native input resolution ({target_res}, {target_res})."
    )

    with torch.no_grad():
        outputs = net(tensor)

    # Convert model outputs to 1D float array
    scores = outputs[0].detach().cpu().numpy()

    # Drop pathologies marked as unsupported due to NaN op_threshs
    unsupported = set(getattr(net, "unsupported_pathologies", None) or detect_unsupported_pathologies(net))

    return {
        pathology: float(score)
        for pathology, score in zip(net.pathologies, scores)
        if pathology not in unsupported
    }
