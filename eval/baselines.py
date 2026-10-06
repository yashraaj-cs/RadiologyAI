"""Baseline model evaluators stub (Variant A: multimodal LLM, Variant B: template)."""


def run_variant_a_baseline(image_path: str, prompt: str) -> str:
    """Run Variant A: Raw multimodal LLM directly on image without vision extraction pipeline.

    Args:
        image_path: Path to chest X-ray image.
        prompt: Zero-shot or few-shot radiologist report prompt.

    Returns:
        Generated report string (stub).
    """
    # Stub implementation: direct multimodal LLM invocation
    raise NotImplementedError("run_variant_a_baseline stub - to be implemented in eval phase")


def run_variant_b_baseline(image_path: str) -> str:
    """Run Variant B: Vision model + deterministic template without LLM.

    Args:
        image_path: Path to chest X-ray image.

    Returns:
        Template report string (stub).
    """
    # Stub implementation: vision model -> rules -> template
    raise NotImplementedError("run_variant_b_baseline stub - to be implemented in eval phase")
