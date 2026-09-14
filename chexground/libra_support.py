import sys
from pathlib import Path


def ensure_libra_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    libra_root = repo_root / "Libra"
    libra_root_str = str(libra_root)
    if libra_root_str not in sys.path:
        sys.path.insert(0, libra_root_str)
    return libra_root


LIBRA_ROOT = ensure_libra_path()

from libra.conversation import (  # noqa: E402
    Conversation,
    SeparatorStyle,
    conv_templates as libra_conv_templates,
)
from libra.model.language_model.libra_llama import LibraConfig, LibraLlamaForCausalLM  # noqa: E402
from libra.model.multimodal_encoder.builder import build_vision_tower  # noqa: E402


__all__ = [
    "LIBRA_ROOT",
    "Conversation",
    "LibraConfig",
    "LibraLlamaForCausalLM",
    "SeparatorStyle",
    "build_vision_tower",
    "ensure_libra_path",
    "libra_conv_templates",
]
