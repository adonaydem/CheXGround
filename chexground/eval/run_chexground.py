#!/usr/bin/env python3
"""Run CheXGround inference from self-contained JSON or JSONL requests."""

import argparse
import contextlib
import importlib.metadata
import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import transformers
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chexground.openmmlab_support import ensure_openmmlab_paths

ensure_openmmlab_paths()

from chexground.constants import DEFAULT_TOKENS
from chexground.data.conversation import conv_templates
from chexground.data.datasets.chest_det import _PaddedProcessedDetDataset
from chexground.model.chexground import STAGE3_REGION_TOKENS, CheXGroundConfig, CheXGroundModel, frame_token_name

DEFAULT_CONV_TEMPLATE = "chexground_finetune"
NON_LORA_TRAINABLES_NAME = "non_lora_trainables.bin"
VISUAL_PROMPT_PREFIX_RE = re.compile(r"^\s*<curr>\s+Current Image:\s+<image>", re.IGNORECASE)
UNKNOWN_REF_PLACEHOLDER_RE = re.compile(r"\s*<roi>\s*<refer_box>\s*</roi>\s*<refer_feat>", re.IGNORECASE)


class ImagePreprocessor(_PaddedProcessedDetDataset):
    """Expose the exact image preprocessing used by CheXGround training."""

    def __init__(self, image_processor, image_geometry_mode: str):
        super().__init__(source="inference", length=0, image_geometry_mode=image_geometry_mode)
        self.image_processor = image_processor


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return " ".join(str(text).replace("\x00", " ").split()).strip()


def read_entries(input_path: str) -> Tuple[List[Dict[str, Any]], pathlib.Path]:
    path = pathlib.Path(input_path).expanduser().resolve(strict=True)

    with path.open("r", encoding="utf-8") as input_file:
        if path.suffix.lower() == ".jsonl":
            payload = [json.loads(line) for line in input_file if line.strip()]
        else:
            payload = json.load(input_file)

    if isinstance(payload, dict):
        entries = [payload]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise TypeError(f"Expected a JSON object or list of objects, got {type(payload).__name__}.")

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"Input entry {index} must be a JSON object, got {type(entry).__name__}.")
    if not entries:
        raise ValueError("Input contains no inference requests.")
    return entries, path.parent


def build_tokenizer(tokenizer_path: str, model_max_length: int, cache_dir: Optional[str], max_temporal_frames: int):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path, cache_dir=cache_dir, model_max_length=model_max_length, padding_side="right", use_fast=False, trust_remote_code=True
    )
    special_tokens_to_add: Dict[str, Any] = {}
    if tokenizer.pad_token is None:
        special_tokens_to_add["pad_token"] = DEFAULT_TOKENS["pad"]

    extended_map = getattr(tokenizer, "special_tokens_map_extended", {}) or {}
    registered_tokens = {str(token) for token in (extended_map.get("additional_special_tokens", []) or [])}
    tokens = [
        DEFAULT_TOKENS["image"], DEFAULT_TOKENS["region"], DEFAULT_TOKENS["bor"], DEFAULT_TOKENS["eor"], DEFAULT_TOKENS["rbox"], DEFAULT_TOKENS["rfeat"],
        DEFAULT_TOKENS["gbox"]
    ]
    tokens.extend(frame_token_name(frame_index) for frame_index in range(max_temporal_frames))
    tokens.extend(STAGE3_REGION_TOKENS)
    additional_tokens = [token for token in tokens if token not in registered_tokens]
    if additional_tokens:
        special_tokens_to_add["additional_special_tokens"] = additional_tokens
    if special_tokens_to_add:
        tokenizer.add_special_tokens(special_tokens_to_add)
    return tokenizer


def load_model_and_tokenizer(args):
    if args.load_in_4bit:
        try:
            bnb_version = importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("--load-in-4bit requires bitsandbytes to be installed.") from exc
        if tuple(int(piece) for piece in re.findall(r"\d+", bnb_version)[:3]) < (0, 43, 2):
            raise RuntimeError(f"--load-in-4bit requires bitsandbytes>=0.43.2; found bitsandbytes=={bnb_version}.")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dtype == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            requested_dtype = torch.bfloat16
        elif device.type == "cuda":
            requested_dtype = torch.float16
        else:
            requested_dtype = torch.float32
    else:
        requested_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}[args.dtype]
    model_dtype = requested_dtype if device.type == "cuda" else torch.float32

    local_lora = pathlib.Path(args.lora_path).expanduser() if args.lora_path else None
    config_source = str(local_lora) if local_lora is not None and local_lora.is_dir() and (local_lora / "config.json").exists() else args.base_model
    checkpoint_config = CheXGroundConfig.from_pretrained(config_source, cache_dir=args.cache_dir, trust_remote_code=True)
    max_temporal_frames = int(checkpoint_config.max_temporal_frames)

    tokenizer_files = ("tokenizer_config.json", "tokenizer.json", "tokenizer.model", "special_tokens_map.json")
    if args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
    elif local_lora is not None and local_lora.is_dir() and any((local_lora / name).exists() for name in tokenizer_files):
        tokenizer_path = str(local_lora)
    else:
        tokenizer_path = args.base_model

    tokenizer = build_tokenizer(
        tokenizer_path=tokenizer_path, model_max_length=args.model_max_length, cache_dir=args.cache_dir, max_temporal_frames=max_temporal_frames
    )

    model_kwargs: Dict[str, Any] = {}
    if args.load_in_8bit or args.load_in_4bit:
        model_kwargs["device_map"] = "auto"
    model = CheXGroundModel.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, torch_dtype=model_dtype, attn_implementation=args.attn_implementation, load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit, **model_kwargs
    )
    model.init_special_token_id(tokenizer)

    if args.lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)
        non_lora_path = local_lora / NON_LORA_TRAINABLES_NAME
        if non_lora_path.is_file():
            non_lora_state = torch.load(non_lora_path, map_location="cpu")
            load_result = model.load_state_dict(non_lora_state, strict=False)
            print(
                f"[chexground] loaded non-LoRA trainables from {non_lora_path} "
                f"(missing={len(load_result.missing_keys)} "
                f"unexpected={len(load_result.unexpected_keys)})", file=sys.stderr, flush=True
            )

    if not (args.load_in_8bit or args.load_in_4bit):
        model.to(device)
    model.eval()

    base_model = model.get_base_model() if args.lora_path else model
    base_model.config.use_cache = bool(args.use_cache)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    image_processor_name = getattr(base_model.config, "image_processor_name", None)
    if not args.lora_path and (pathlib.Path(args.base_model).expanduser() / "preprocessor_config.json").is_file():
        image_processor_name = args.base_model
    if not image_processor_name:
        raise ValueError("CheXGround config.image_processor_name is missing.")
    image_processor = transformers.AutoImageProcessor.from_pretrained(image_processor_name, cache_dir=args.cache_dir, trust_remote_code=True)
    return model, tokenizer, image_processor, device, model_dtype, max_temporal_frames


def preprocess_images(image_paths: Sequence[str], image_processor, image_geometry_mode: str, max_temporal_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
    processor = ImagePreprocessor(image_processor, image_geometry_mode=image_geometry_mode)
    images = []
    pixel_masks = []
    for image_path in image_paths:
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        image_tensor, pixel_mask = processor._preprocess_image(image)
        images.append(image_tensor)
        pixel_masks.append(pixel_mask)

    image_tensor = torch.stack(images, dim=0)
    pixel_mask = torch.stack(pixel_masks, dim=0)
    pad_frames = max_temporal_frames - image_tensor.shape[0]
    if pad_frames > 0:
        image_pad = torch.zeros((pad_frames, *image_tensor.shape[1:]), dtype=image_tensor.dtype)
        mask_pad = torch.zeros((pad_frames, *pixel_mask.shape[1:]), dtype=pixel_mask.dtype)
        image_tensor = torch.cat([image_pad, image_tensor], dim=0)
        pixel_mask = torch.cat([mask_pad, pixel_mask], dim=0)
    return image_tensor, pixel_mask


def run_one(
    entry: Dict[str, Any], *, input_dir: pathlib.Path, model, tokenizer, image_processor, device: torch.device, dtype: torch.dtype, args,
    max_temporal_frames: int
) -> Dict[str, Any]:
    image_refs = entry.get("image_refs")
    if not isinstance(image_refs, list) or not image_refs:
        raise ValueError("'image_refs' must be a non-empty list of image file paths.")
    if len(image_refs) > max_temporal_frames:
        raise ValueError(f"'image_refs' has {len(image_refs)} frames, but this model supports at most {max_temporal_frames}.")
    image_paths = []
    for image_ref in image_refs:
        if not isinstance(image_ref, str) or not image_ref.strip():
            raise ValueError("Every value in 'image_refs' must be a non-empty file path string.")
        path = pathlib.Path(image_ref).expanduser()
        if not path.is_absolute():
            path = input_dir / path
        image_paths.append(str(path.resolve(strict=True)))

    prompt_value = entry.get("prompt")
    prompt_text = normalize_text(prompt_value) if isinstance(prompt_value, str) else ""
    if not prompt_text:
        raise ValueError("'prompt' must be a non-empty string.")
    has_visual_prefix = VISUAL_PROMPT_PREFIX_RE.search(prompt_text) is not None
    if has_visual_prefix:
        # Requests carry no ROI slot IDs, so remove only unresolved reference placeholders.
        prompt_text = UNKNOWN_REF_PLACEHOLDER_RE.sub("", prompt_text)
    else:
        prompt_text = prompt_text.replace(DEFAULT_TOKENS["image"], " ").strip()
    ground_truth_value = entry.get("ground_truth")
    if ground_truth_value is not None and not isinstance(ground_truth_value, str):
        raise ValueError("'ground_truth' must be a string when provided.")
    ground_truth = normalize_text(ground_truth_value) or None

    if has_visual_prefix:
        user_text = prompt_text
    else:
        roi_block = " ".join(f"{token} {DEFAULT_TOKENS['region']}" for token in STAGE3_REGION_TOKENS)
        blocks = []
        for frame_index in range(len(image_paths)):
            frame_label = "Current Image" if frame_index == 0 else "Prior Image" if frame_index == 1 else f"Prior Image {frame_index}"
            blocks.append(f"{frame_token_name(frame_index)} {frame_label}: {DEFAULT_TOKENS['image']}")
            if frame_index == 0:
                blocks.append(f"Anatomy Regions: {roi_block}")
        visual_prefix = "\n".join(blocks)
        user_text = f"{visual_prefix}\n{prompt_text}" if prompt_text else visual_prefix
    placeholder_count = user_text.count(DEFAULT_TOKENS["image"])
    if placeholder_count != len(image_paths):
        raise ValueError(f"Prompt image count {placeholder_count} does not match 'image_refs' count {len(image_paths)}.")

    conversation = conv_templates[args.conv_template].copy()
    conversation.append_message(conversation.roles[0], user_text)
    conversation.append_message(conversation.roles[1], "")
    prompt = conversation.get_prompt()
    tokenized = tokenizer(prompt, return_tensors="pt", padding="longest", truncation=True, max_length=args.model_max_length)
    images, pixel_mask = preprocess_images(
        image_paths=image_paths, image_processor=image_processor, image_geometry_mode=args.image_geometry_mode, max_temporal_frames=max_temporal_frames
    )
    batch = {
        "input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask, "images": images.unsqueeze(0), "pixel_mask": pixel_mask.unsqueeze(0),
        "temporal_lengths": torch.tensor([len(image_paths)], dtype=torch.long), "refer_slot_ids": [[]], "ground_slot_ids": [[]]
    }
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    generation_kwargs = {
        **batch, "do_sample": bool(args.do_sample), "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None, "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens, "use_cache": bool(args.use_cache),
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
    with torch.no_grad(), (
        torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16) else contextlib.nullcontext()
    ):
        output_ids = model.generate(**generation_kwargs)

    generated_ids = output_ids[0, batch["input_ids"].shape[1]:].detach().cpu()
    prediction_raw = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=False).strip()
    prediction = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()

    result = {"image_refs": image_paths, "prompt": user_text, "ground_truth": ground_truth, "prediction": prediction, "prediction_raw": prediction_raw}
    if "id" in entry:
        result["id"] = entry["id"]
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run a CheXGround checkpoint with an optional LoRA adapter over JSON/JSONL requests.")
    parser.add_argument("--base-model", required=True, help="CheXGround base or merged model path.")
    parser.add_argument("--lora-path", help="Optional CheXGround LoRA adapter path; omit for a merged model.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL request file.")
    parser.add_argument("--output", help="Optional JSONL output file; defaults to stdout.")
    parser.add_argument("--conv-template", default=DEFAULT_CONV_TEMPLATE)
    parser.add_argument("--image-geometry-mode", default="one_side_pad", choices=("center_pad", "one_side_pad", "square_resize"))
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=("auto", "bf16", "fp16", "float16", "fp32", "float32"))
    parser.add_argument("--cache-dir")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--attn-implementation", default=None)
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument("--load-in-8bit", action="store_true", help="Load the base model in 8-bit precision.")
    precision_group.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit precision.")
    return parser.parse_args()


def main():
    args = parse_args()
    entries, input_dir = read_entries(args.input)
    model, tokenizer, image_processor, device, dtype, max_temporal_frames = load_model_and_tokenizer(args)
    print(
        f"[chexground] base={args.base_model} lora={args.lora_path} "
        f"device={device} dtype={dtype} max_temporal_frames={max_temporal_frames}", file=sys.stderr, flush=True
    )

    with open(args.output, "w", encoding="utf-8") if args.output else contextlib.nullcontext(sys.stdout) as output_file:
        for index, entry in enumerate(entries):
            result = run_one(
                entry, input_dir=input_dir, model=model, tokenizer=tokenizer, image_processor=image_processor, device=device, dtype=dtype, args=args,
                max_temporal_frames=max_temporal_frames
            )
            result["index"] = index
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()


if __name__ == "__main__":
    main()

# Input: one JSON object, a JSON array of objects, or JSONL with one object per line.
# image_refs: non-empty chronological paths, oldest prior first and current last;
#             at most the checkpoint's max_temporal_frames (normally two).
# prompt: a non-empty instruction or a prepared current-first visual prompt.
# Optional id is copied unchanged; ground_truth is normalized and never sent to the model.
# Absolute paths are accepted; relative paths are resolved from the input file's directory.
# Example: {"image_refs": ["prior.png", "current.png"], "prompt": "Describe interval change."}
