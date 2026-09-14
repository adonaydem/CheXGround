import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoImageProcessor, GenerationConfig

from chexground.eval.run_chexground import CheXGroundConfig, CheXGroundModel, NON_LORA_TRAINABLES_NAME, build_tokenizer


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def merge_lora(base_model, lora_path, output_dir, cache_dir=None):
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    adapter = Path(lora_path).expanduser()
    if any(Path(source).expanduser().resolve() in output.parents for source in (base_model, lora_path)):
        raise ValueError("Export directory must be outside the source checkpoints.")
    config_payload, _ = CheXGroundConfig.get_config_dict(str(adapter), cache_dir=cache_dir)
    export_config = CheXGroundConfig(**config_payload)
    for config, payload in ((export_config, config_payload), (export_config.libra_cfg, config_payload["libra_cfg"])):
        if "attn_implementation" in payload:
            config.attn_implementation = payload["attn_implementation"]
    config = copy.deepcopy(export_config)
    config.libra_cfg._attn_implementation = "eager"
    config.libra_cfg.attn_implementation = "eager"
    model, loading_info = CheXGroundModel.from_pretrained(
        base_model, config=config, cache_dir=cache_dir, torch_dtype=torch.float32, attn_implementation="eager", output_loading_info=True
    )
    if any(loading_info.values()):
        raise ValueError(f"Base checkpoint loading failed: {loading_info}")
    estimated_bytes = sum(t.numel() * (2 if t.is_floating_point() else t.element_size()) for t in model.state_dict().values())
    existing_parent = next(parent for parent in output.parents if parent.exists())
    if shutil.disk_usage(existing_parent).free < estimated_bytes + 1024**3:
        raise OSError(f"Export requires {estimated_bytes + 1024**3} free bytes including reserve.")

    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    adapter_state = (load_file(adapter / "adapter_model.safetensors") if (adapter / "adapter_model.safetensors").is_file()
                     else torch.load(adapter / "adapter_model.bin", map_location="cpu", weights_only=True))
    loaded_adapter = get_peft_model_state_dict(model)
    if adapter_state.keys() != loaded_adapter.keys():
        raise ValueError("Adapter parameter names do not match the model.")
    for name, tensor in adapter_state.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite adapter tensor: {name}")
        if tensor.shape != loaded_adapter[name].shape or not torch.equal(tensor, loaded_adapter[name]):
            raise ValueError(f"Adapter restoration failed: {name}")
    parameters = dict(model.named_parameters(remove_duplicate=False))
    non_lora_state = torch.load(adapter / NON_LORA_TRAINABLES_NAME, map_location="cpu", weights_only=True)
    for name, tensor in non_lora_state.items():
        if name not in parameters or tensor.shape != parameters[name].shape:
            raise ValueError(f"Non-LoRA parameter mismatch: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite non-LoRA tensor: {name}")
    model.load_state_dict(non_lora_state, strict=False)
    for name, tensor in non_lora_state.items():
        if not torch.equal(parameters[name], tensor.to(parameters[name].dtype)):
            raise ValueError(f"Non-LoRA restoration failed: {name}")

    expected_targets = {}
    for name, module in model.get_base_model().named_modules():
        if hasattr(module, "lora_A"):
            expected = module.weight + module.get_delta_weight(module.active_adapter)
            if module.weight.dtype != torch.float32 or not torch.isfinite(expected).all():
                raise ValueError(f"Invalid FP32 merge target: {name}")
            expected_targets[name + ".weight"] = tensor_digest(expected)
    if not expected_targets:
        raise ValueError("Adapter contains no LoRA targets.")
    del parameters, expected, adapter_state, loaded_adapter
    merged = model.merge_and_unload()
    for name, expected_hash in expected_targets.items():
        if tensor_digest(merged.get_parameter(name)) != expected_hash:
            raise ValueError(f"Merged weight mismatch: {name}")

    merged.config = export_config
    text_config_payload = config_payload.get("grpa_cfg", {}).get("text_encoder_cfg") or {}
    if "auto_map" in text_config_payload:
        merged.config.grpa_cfg.text_encoder_cfg.auto_map = copy.deepcopy(text_config_payload["auto_map"])
    merged.generation_config = GenerationConfig.from_pretrained(str(adapter), cache_dir=cache_dir)
    tokenizer = build_tokenizer(str(adapter), 2048, cache_dir, int(export_config.max_temporal_frames))
    image_processor = AutoImageProcessor.from_pretrained(export_config.image_processor_name, cache_dir=cache_dir, trust_remote_code=True)
    merged.to(dtype=torch.bfloat16)
    for name, tensor in merged.state_dict().items():
        if name in expected_targets or "base_model.model." + name in non_lora_state:
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite BF16 export tensor: {name}")
    export_state = {name: tensor.detach().contiguous().clone() for name, tensor in merged.state_dict().items()}
    expected_state = {name: tensor_digest(tensor) for name, tensor in export_state.items()}
    output.mkdir(parents=True)
    merged.save_pretrained(output, state_dict=export_state, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output)
    image_processor.save_pretrained(output)
    saved_state = {}
    for shard in sorted(output.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if name in saved_state:
                    raise ValueError(f"Duplicate exported tensor: {name}")
                saved_state[name] = tensor_digest(tensors.get_tensor(name))
    if saved_state != expected_state:
        raise ValueError("Exported tensors differ from the merged state.")
    return {"output_dir": str(output), "non_lora_tensors": len(non_lora_state), "merged_targets": len(expected_targets),
            "saved_tensors": len(saved_state), "loading_info": loading_info}


def main():
    parser = argparse.ArgumentParser(description="Export a complete CheXGround inference checkpoint with merged LoRA weights.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    print(json.dumps(merge_lora(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
