import argparse
import contextlib
import json
from pathlib import Path
import re
import sys

import torch

from chexground.eval.run_chexground import (
    DEFAULT_CONV_TEMPLATE, DEFAULT_TOKENS, STAGE3_REGION_TOKENS, VISUAL_PROMPT_PREFIX_RE,
    conv_templates, frame_token_name, load_model_and_tokenizer, normalize_text, preprocess_images, read_entries,
)


def prepare_request(entry, index, *, input_dir, tokenizer, args, anatomy_classes, max_temporal_frames):
    legacy_images = "image_refs" not in entry
    image_refs = entry["image"] if legacy_images else entry["image_refs"]
    if legacy_images and isinstance(image_refs, str):
        image_refs = [image_refs]
    if not isinstance(image_refs, list) or not image_refs:
        raise ValueError("Images must be a non-empty list of paths.")
    image_root = Path(args.image_root).expanduser() if args.image_root else input_dir
    image_paths = []
    for image_ref in image_refs:
        if not isinstance(image_ref, str) or not image_ref.strip():
            raise ValueError("Image paths must be non-empty strings.")
        path = Path(image_ref).expanduser()
        image_paths.append(str((path if path.is_absolute() else image_root / path).resolve(strict=True)))
    if legacy_images:
        image_paths.reverse()
        if len(image_paths) == 2 and image_paths[0] == image_paths[1]:
            image_paths = image_paths[:1]
    if len(image_paths) > max_temporal_frames:
        raise ValueError(f"This checkpoint supports at most {max_temporal_frames} frames.")

    prompt_value = entry["prompt"] if "prompt" in entry else (entry.get("metadata", {}).get("raw_text") or entry["text"])
    if not isinstance(prompt_value, str) or not normalize_text(prompt_value):
        raise ValueError("Prompt must be a non-empty string.")
    prompt_text = normalize_text(prompt_value)
    explicit_slots = entry.get("refer_slot_ids", [])
    if not isinstance(explicit_slots, list) or any(type(slot) is not int or not 0 <= slot < len(STAGE3_REGION_TOKENS) for slot in explicit_slots):
        raise ValueError("refer_slot_ids must contain anatomy slot indices.")
    if prompt_text.count(DEFAULT_TOKENS["rbox"]) != len(explicit_slots):
        raise ValueError("Reference placeholders and refer_slot_ids must match.")

    slot_by_name = {name: slot for slot, name in enumerate(anatomy_classes)}
    reserved_tokens = set(DEFAULT_TOKENS.values()) | set(STAGE3_REGION_TOKENS)
    reserved_tokens.update(frame_token_name(frame) for frame in range(max_temporal_frames))
    parts, refer_slots = [], []
    cursor = explicit_index = 0
    for match in re.finditer(r"<([^<>\n]+)>", prompt_text):
        tag = match.group(0)
        name = normalize_text(match.group(1)).lower()
        parts.append(prompt_text[cursor:match.start()])
        if tag == DEFAULT_TOKENS["rbox"]:
            refer_slots.append(explicit_slots[explicit_index])
            explicit_index += 1
        elif tag not in reserved_tokens:
            refer_slots.append(slot_by_name[name])
            tag = "".join(DEFAULT_TOKENS[key] for key in ("bor", "rbox", "eor", "rfeat"))
        parts.append(tag)
        cursor = match.end()
    parts.append(prompt_text[cursor:])
    prompt_text = "".join(parts)

    if VISUAL_PROMPT_PREFIX_RE.search(prompt_text):
        user_text = prompt_text
    else:
        prompt_text = prompt_text.replace(DEFAULT_TOKENS["image"], " ").strip()
        roi_block = " ".join(f"{token} {DEFAULT_TOKENS['region']}" for token in STAGE3_REGION_TOKENS)
        blocks = []
        for frame in range(len(image_paths)):
            label = "Current Image" if frame == 0 else "Prior Image" if frame == 1 else f"Prior Image {frame}"
            blocks.append(f"{frame_token_name(frame)} {label}: {DEFAULT_TOKENS['image']}")
            if frame == 0:
                blocks.append(f"Anatomy Regions: {roi_block}")
        visual_prefix = "\n".join(blocks)
        user_text = f"{visual_prefix}\n{prompt_text}" if prompt_text else visual_prefix

    conversation = conv_templates[args.conv_template].copy()
    conversation.append_message(conversation.roles[0], user_text)
    conversation.append_message(conversation.roles[1], "")
    tokenized = tokenizer(conversation.get_prompt(), return_tensors="pt", padding="longest", truncation=True, max_length=args.model_max_length)
    input_ids, attention_mask = tokenized.input_ids[0], tokenized.attention_mask[0]
    expected_counts = {"image": len(image_paths), "region": len(STAGE3_REGION_TOKENS), "rbox": len(refer_slots), "rfeat": len(refer_slots), "gbox": 0}
    for key, expected in expected_counts.items():
        token_id = tokenizer.convert_tokens_to_ids(DEFAULT_TOKENS[key])
        if int(input_ids.eq(token_id).sum()) != expected:
            raise ValueError(f"{DEFAULT_TOKENS[key]} count is incompatible with this request after tokenization.")
    if not attention_mask.bool().all() or input_ids.eq(tokenizer.pad_token_id).any():
        raise ValueError("Request prompts must not contain padding tokens.")
    return {"index": index, "question_id": entry.get("question_id", entry.get("id", index)), "image_refs": image_paths,
            "prompt": user_text, "input_ids": input_ids, "attention_mask": attention_mask, "refer_slot_ids": refer_slots}


def run_batch(items, *, model, tokenizer, image_processor, device, dtype, args, anatomy_classes, max_temporal_frames):
    processed = [preprocess_images(item["image_refs"], image_processor, args.image_geometry_mode, max_temporal_frames) for item in items]
    batch = {
        "input_ids": torch.stack([item["input_ids"] for item in items]),
        "attention_mask": torch.stack([item["attention_mask"] for item in items]),
        "images": torch.stack([images for images, _ in processed]),
        "pixel_mask": torch.stack([mask for _, mask in processed]),
        "temporal_lengths": torch.tensor([len(item["image_refs"]) for item in items], dtype=torch.long),
    }
    batch = {key: value.to(device) for key, value in batch.items()}
    del processed
    base_model = model.get_base_model() if args.lora_path else model
    generation_kwargs = {
        **batch, "refer_slot_ids": [list(item["refer_slot_ids"]) for item in items for _ in range(args.num_beams)],
        "ground_slot_ids": [[] for _ in range(len(items) * args.num_beams)],
        "do_sample": bool(args.do_sample), "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None, "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens,
        "use_cache": bool(args.use_cache), "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
    with torch.no_grad(), (
        torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16) else contextlib.nullcontext()
    ):
        backbone = base_model._extract_shared_backbone_outputs(batch["images"], batch["temporal_lengths"])
        roi_outputs = base_model.grpa.extract_roi_features(
            images=batch["images"], pixel_mask=batch["pixel_mask"], temporal_lengths=batch["temporal_lengths"], backbone_outputs=backbone
        )
        boxes = roi_outputs["pred_boxes"].detach().cpu()
        valid_boxes = roi_outputs["roi_valid_mask"].detach().cpu()
        del roi_outputs, backbone
        output_ids = model.generate(**generation_kwargs)

    generated = output_ids[:, batch["input_ids"].shape[1]:].detach().cpu().tolist()
    for index, token_ids in enumerate(generated):
        if tokenizer.eos_token_id in token_ids:
            generated[index] = token_ids[:token_ids.index(tokenizer.eos_token_id) + 1]
    raw_answers = tokenizer.batch_decode(generated, skip_special_tokens=False)
    answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
    token_to_slot = {token_id: slot for slot, token_id in enumerate(base_model.region_index_token_ids)}
    results = []
    for index, item in enumerate(items):
        regions = []
        for token_id in generated[index]:
            if token_id not in token_to_slot:
                continue
            slot = token_to_slot[token_id]
            if not valid_boxes[index, slot]:
                continue
            region = {"tag": f"r{slot}", "slot_id": slot, "bbox": boxes[index, slot].tolist()}
            if anatomy_classes:
                region["region"] = anatomy_classes[slot]
            regions.append(region)
        results.append({
            "index": item["index"], "question_id": item["question_id"], "image_refs": item["image_refs"], "prompt": item["prompt"],
            "text": answers[index].strip(), "prediction_raw": raw_answers[index].strip(), "predicted_regions": regions,
            "predicted_boxes": [region["bbox"] for region in regions], "bbox_format": "cxcywh", "bbox_normalized": True,
            "bbox_coordinate_space": "processed_image", "image_geometry_mode": args.image_geometry_mode,
        })
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run batched CheXGround inference with model-predicted grounding boxes.")
    parser.add_argument("--base-model", required=True, help="CheXGround base or merged model path.")
    parser.add_argument("--lora-path", help="Optional CheXGround LoRA adapter path; omit for a merged model.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL request file.")
    parser.add_argument("--output", help="Optional JSONL output file; defaults to stdout.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-root", help="Root for relative image paths; defaults to the input file directory.")
    parser.add_argument("--anatomy-json", help="Ordered anatomy class list matching the checkpoint slots.")
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
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main():
    args = parse_args()
    entries, input_dir = read_entries(args.input)
    anatomy_classes = []
    if args.anatomy_json:
        with Path(args.anatomy_json).expanduser().open(encoding="utf-8") as handle:
            payload = json.load(handle)
        anatomy_classes = payload["anatomy_classes"] if isinstance(payload, dict) else payload
        if not isinstance(anatomy_classes, list) or any(not isinstance(name, str) or not normalize_text(name) for name in anatomy_classes):
            raise ValueError("Anatomy classes must be an ordered list of non-empty names.")
        anatomy_classes = [normalize_text(name).lower() for name in anatomy_classes]
        if len(anatomy_classes) != len(STAGE3_REGION_TOKENS) or len(set(anatomy_classes)) != len(anatomy_classes):
            raise ValueError("Anatomy classes must contain one unique name per checkpoint slot.")
    model, tokenizer, image_processor, device, dtype, max_temporal_frames = load_model_and_tokenizer(args)
    groups = {}
    for index, entry in enumerate(entries):
        item = prepare_request(entry, index, input_dir=input_dir, tokenizer=tokenizer, args=args,
                               anatomy_classes=anatomy_classes, max_temporal_frames=max_temporal_frames)
        key = (item["input_ids"].numel(), len(item["image_refs"]))
        groups.setdefault(key, []).append(item)
    pending, next_index = {}, 0
    with open(args.output, "w", encoding="utf-8") if args.output else contextlib.nullcontext(sys.stdout) as output_file:
        for items in groups.values():
            for start in range(0, len(items), args.batch_size):
                results = run_batch(items[start:start + args.batch_size], model=model, tokenizer=tokenizer, image_processor=image_processor,
                                    device=device, dtype=dtype, args=args, anatomy_classes=anatomy_classes, max_temporal_frames=max_temporal_frames)
                pending.update((result["index"], result) for result in results)
                while next_index in pending:
                    output_file.write(json.dumps(pending.pop(next_index), ensure_ascii=False) + "\n")
                    output_file.flush()
                    next_index += 1


if __name__ == "__main__":
    main()
