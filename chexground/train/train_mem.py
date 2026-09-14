from chexground.openmmlab_support import ensure_openmmlab_paths
from chexground.train.train import train
import torch

_original_load = torch.load
def _patched_load(*args, **kwargs):
    if kwargs.get("weights_only", False):
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

ensure_openmmlab_paths()

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
