import copy
import csv
import json
import os
import re
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor

from chexground.constants import DEFAULT_TOKENS, IGNORE_INDEX
from chexground.data.conversation import conv_templates
from chexground.data.datasets.chest_det import _PaddedProcessedDetDataset
from chexground.data.datasets.chest_temporal_grounding import (
    MIMICTemporalGroundingDataset,
    _normalize_region_name,
    _normalize_scalar_id,
)


STAGE3_REGION_TOKENS = [f"<r{i}>" for i in range(29)]
_WHITESPACE_RE = re.compile(r"\s+")
_ANATOMY_TAG_RE = re.compile(r"<([^<>]+)>")
_MIMIC_PATH_RE = re.compile(
    r"^files/(?P<bucket>p\d+)/(?P<patient>p\d+)/(?P<study>s\d+)/(?P<image>[^/]+)\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE,
)


def _normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", str(text).replace("\x00", " ")).strip()


def _parse_study_datetime(value: Optional[str]) -> Optional[datetime]:
    text = _normalize_text(value)
    if not text:
        return None

    candidates = []
    if text.endswith(" UTC"):
        base = text[:-4].strip()
        candidates.extend([f"{base}+00:00", base])
    if text.endswith("Z"):
        candidates.append(f"{text[:-1]}+00:00")
    candidates.append(text)

    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    for fmt in ("%Y-%m-%d %H:%M:%S %Z", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


class BaseMedicalLLMDataset(MIMICTemporalGroundingDataset):
    default_instruction = ""

    def __init__(
        self,
        ann_file: str,
        tokenizer,
        image_root: str = "/path/to/images",
        bbox_json_path: str = "/path/to/anatomy_boxes",
        labels_csv_path: str = "/path/to/abnormality_labels.csv",
        temporal_metadata_csv_path: Optional[str] = None,
        region_json_root: Optional[str] = None,
        img_processor=None,
        image_processor_name: Optional[str] = None,
        split: Optional[str] = None,
        conv_temp: str = "chexground_pretrain",
        max_temporal_frames: int = 2,
        source: str = "mimic_llm",
        length: Optional[int] = None,
        **kwargs,
    ):
        kwargs.pop("use_augmentation", None)
        anatomy_num_queries = int(kwargs.pop("anatomy_num_queries", 0) or 0)
        abnormality_num_classes = int(kwargs.pop("abnormality_num_classes", 0) or 0)
        image_geometry_mode = kwargs.pop("image_geometry_mode", "one_side_pad")

        _PaddedProcessedDetDataset.__init__(
            self,
            source=source,
            length=0,
            anatomy_num_queries=max(1, anatomy_num_queries),
            abnormality_num_classes=max(1, abnormality_num_classes),
            image_geometry_mode=image_geometry_mode,
            **kwargs,
        )

        self.ann_file = ann_file
        self.image_root = image_root
        self.bbox_dir = bbox_json_path
        self.bbox_json_path = bbox_json_path
        self.labels_csv_path = labels_csv_path
        self.temporal_metadata_csv_path = temporal_metadata_csv_path
        self.region_json_root = region_json_root
        self.split = split
        self.tokenizer = tokenizer
        self.conv_template = conv_templates[conv_temp]
        self.max_temporal_frames = int(max_temporal_frames)
        self.source = source

        if img_processor is not None:
            self.image_processor = img_processor
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(image_processor_name)

        self.transform = None

        anatomy_classes_path = os.path.join(self.bbox_dir, "anatomy_classes.json")
        with open(anatomy_classes_path, "r", encoding="utf-8") as f:
            bbox_payload = json.load(f)
        anatomy_classes = bbox_payload["anatomy_classes"]
        self.anatomy_num_queries = len(anatomy_classes)
        self.anatomy_class_to_idx = {name: idx for idx, name in enumerate(anatomy_classes)}
        self.static_region_name_to_idx = {
            _normalize_region_name(name): idx for name, idx in self.anatomy_class_to_idx.items()
        }

        self.label_map, self.image_split_map = self._load_label_sources()
        self.image_datetime_map: Dict[str, datetime] = {}
        self.study_datetime_map: Dict[str, datetime] = {}
        if self.temporal_metadata_csv_path:
            self._load_temporal_metadata_map()
        self.abnormality_num_classes = next(
            iter(self.label_map.values())
        )["image_labels"].shape[0]

        self._payload_cache_max_entries = 2048
        self._bbox_payload_cache = OrderedDict()
        self._scene_graph_payload_cache = OrderedDict()

        with open(self.ann_file, "r", encoding="utf-8") as f:
            if self.ann_file.endswith(".jsonl"):
                payload = [json.loads(line.strip()) for line in f if line.strip()]
            else:
                payload = json.load(f)
        self.samples = []
        for entry in payload:
            sample = self._build_sample(entry)
            if sample is not None:
                self.samples.append(sample)
        if length is not None:
            self.samples = self.samples[:length]
        self.length = len(self.samples)

    def _load_label_sources(self) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, str]]:
        label_map: Dict[str, Dict[str, torch.Tensor]] = {}
        split_map: Dict[str, str] = {}
        with open(self.labels_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fixed_meta_columns = {"image_id", "subjectid_studyid", "split", "patient_id", "study_id", "viewpoint"}
            abnormality_columns = sorted(name for name in reader.fieldnames if name not in fixed_meta_columns)

            for row in reader:
                image_id = _normalize_scalar_id(row.get("image_id"))
                if not image_id:
                    continue
                values = []
                masks = []
                for col in abnormality_columns:
                    value = float(row[col])
                    if value < 0:
                        values.append(0.0)
                        masks.append(False)
                    else:
                        values.append(value)
                        masks.append(True)
                label_map[image_id] = {
                    "image_labels": torch.tensor(values, dtype=torch.float32),
                    "image_label_mask": torch.tensor(masks, dtype=torch.bool),
                    "subjectid_studyid": row.get("subjectid_studyid") or image_id,
                }
                split_map[image_id] = _normalize_text(row.get("split"))

        return label_map, split_map

    def _parse_image_ref(self, image_ref: str) -> Dict[str, str]:
        normalized = _normalize_text(image_ref).replace("\\", "/")
        if os.path.isabs(normalized):
            root_prefix = os.path.abspath(self.image_root).replace("\\", "/").rstrip("/") + "/"
            normalized_abs = os.path.abspath(normalized).replace("\\", "/")
            normalized = normalized_abs[len(root_prefix) :]
        while normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = re.sub(r"/+", "/", normalized)
        if normalized.startswith("MIMIC-CXR-JPG/"):
            normalized = normalized[len("MIMIC-CXR-JPG/") :]
        if not normalized.startswith("files/"):
            normalized = f"files/{normalized}"
        match = _MIMIC_PATH_RE.match(normalized)
        patient_id = match.group("patient")
        return {
            "relative_path": normalized,
            "image_id": os.path.splitext(os.path.basename(normalized))[0],
            "study_id": match.group("study"),
            "patient_id": patient_id,
        }

    def _build_user_text(
        self,
        instruction_text: str,
        temporal_length: int,
        sample_meta: Dict[str, object],
    ) -> str:
        roi_block = " ".join(f"{token} {DEFAULT_TOKENS['region']}" for token in STAGE3_REGION_TOKENS)
        image_ids = sample_meta["temporal_image_ids"]
        study_ids = sample_meta["temporal_study_ids"]
        current_dt = self.image_datetime_map.get(image_ids[-1]) or self.study_datetime_map.get(study_ids[-1])
        visual_blocks = []
        for frame_index in range(temporal_length):
            if frame_index == 0:
                token = DEFAULT_TOKENS["curr"]
                image_label = "Current Image"
            else:
                token = DEFAULT_TOKENS.get(f"previm{frame_index}", f"<previm{frame_index}>")
                image_label = "Prior Image" if frame_index == 1 else f"Prior Image {frame_index}"
                source_index = temporal_length - 1 - frame_index
                prior_dt = (
                    self.image_datetime_map.get(image_ids[source_index])
                    or self.study_datetime_map.get(study_ids[source_index])
                )
                if current_dt is not None and prior_dt is not None:
                    delta_seconds = int((current_dt - prior_dt).total_seconds())
                    if delta_seconds > 0:
                        days, remainder = divmod(delta_seconds, 24 * 60 * 60)
                        hours, remainder = divmod(remainder, 60 * 60)
                        minutes = remainder // 60
                        parts = []
                        if days > 0:
                            parts.append(f"{days} {'day' if days == 1 else 'days'}")
                        if hours > 0:
                            parts.append(f"{hours} {'hour' if hours == 1 else 'hours'}")
                        if not parts and minutes > 0:
                            parts.append(f"{minutes} {'minute' if minutes == 1 else 'minutes'}")
                        if parts:
                            image_label += f" taken {' and '.join(parts)} ago"
            visual_blocks.append(f"{token} {image_label}: {DEFAULT_TOKENS['image']}")
            if frame_index == 0:
                visual_blocks.append(f"Anatomy Regions: {roi_block}")
        visual_prefix = "\n".join(visual_blocks)
        if instruction_text:
            return f"{visual_prefix}\n{instruction_text}"
        return visual_prefix

    def _load_temporal_metadata_map(self) -> None:
        with open(self.temporal_metadata_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            study_id_field = next(
                (name for name in ("study_id", "subjectid_studyid") if name in reader.fieldnames), None
            )
            datetime_field = next(
                name for name in ("StudyDateTime", "study_datetime", "study_datetime_utc") if name in reader.fieldnames
            )
            for row in reader:
                image_id = _normalize_scalar_id(row.get("image_id"))
                study_dt = _parse_study_datetime(row.get(datetime_field))
                if study_dt is None:
                    continue
                if image_id and image_id not in self.image_datetime_map:
                    self.image_datetime_map[image_id] = study_dt
                if study_id_field is None:
                    continue
                study_id = _normalize_scalar_id(row.get(study_id_field))
                if study_id and study_id not in self.study_datetime_map:
                    self.study_datetime_map[study_id] = study_dt

    def _tokenize_example(self, user_text: str, answer_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        full_conv = self.conv_template.copy()
        full_conv.append_message(full_conv.roles[0], user_text)
        full_conv.append_message(full_conv.roles[1], answer_text)
        full_prompt = full_conv.get_prompt()

        prefix_conv = self.conv_template.copy()
        prefix_conv.append_message(prefix_conv.roles[0], user_text)
        prefix_conv.append_message(prefix_conv.roles[1], "")
        prefix_prompt = prefix_conv.get_prompt()

        input_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids[0]
        prefix_ids = self.tokenizer(
            prefix_prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids[0]

        labels = input_ids.clone()
        labels[: min(prefix_ids.shape[0], labels.shape[0])] = IGNORE_INDEX
        return input_ids, labels

    def _extract_prompt_pair(self, conversations: Sequence[Dict[str, str]]) -> Tuple[str, str]:
        human_text = ""
        gpt_text = ""
        for conversation in conversations:
            role = _normalize_text(conversation.get("from")).lower()
            if role == "human" and not human_text:
                human_text = _normalize_text(conversation.get("value")).replace("<image>", " ").strip()
            elif role == "gpt" and not gpt_text:
                gpt_text = _normalize_text(conversation.get("value"))
            if human_text and gpt_text:
                break
        return human_text, gpt_text

    def _normalize_alignment_image_sequence(
        self,
        image_field: Sequence[str],
    ) -> Tuple[List[str], Dict[str, str]]:
        parsed_refs = [self._parse_image_ref(image_ref) for image_ref in image_field]
        if len(parsed_refs) == 1:
            return [parsed_refs[0]["relative_path"]], parsed_refs[0]

        current, prior = parsed_refs
        ordered_refs = [prior["relative_path"], current["relative_path"]]
        if current["image_id"] == prior["image_id"]:
            ordered_refs = [current["relative_path"]]
        return ordered_refs, current

    def _finalize_sample(
        self,
        *,
        image_refs: Sequence[str],
        instruction: str,
        answer: str,
        current_image_id: str,
        current_study_id: str,
        metadata: Optional[Dict[str, object]] = None,
        refer_slot_ids: Optional[Sequence[int]] = None,
        ground_slot_ids: Optional[Sequence[int]] = None,
    ) -> Optional[Dict[str, object]]:
        normalized_refs = []
        image_ids = []
        study_ids = []
        for image_ref in image_refs[-self.max_temporal_frames :]:
            parsed = self._parse_image_ref(image_ref)
            normalized_refs.append(parsed["relative_path"])
            image_ids.append(parsed["image_id"])
            study_ids.append(parsed["study_id"])

        if self.split is not None and any(self.image_split_map[image_id] != self.split for image_id in image_ids):
            return None

        bbox_json_path = os.path.join(self.bbox_dir, f"{current_image_id}.json")
        scene_graph_path = self._scene_graph_path(current_image_id)

        if not _normalize_text(answer):
            return None

        return {
            "temporal_image_refs": normalized_refs,
            "temporal_image_ids": image_ids,
            "temporal_study_ids": study_ids,
            "temporal_length": len(normalized_refs),
            "current_image_id": current_image_id,
            "current_study_id": current_study_id,
            "bbox_json_path": bbox_json_path,
            "scene_graph_path": scene_graph_path,
            "instruction": _normalize_text(instruction or self.default_instruction).replace("<image>", " ").strip(),
            "answer": _normalize_text(answer),
            "refer_slot_ids": list(refer_slot_ids or []),
            "ground_slot_ids": list(ground_slot_ids or []),
            "metadata": metadata or {},
        }

    def get_temporal_length(self, idx: int) -> int:
        return int(self.samples[idx]["temporal_length"])

    def __getitem__(self, idx):
        sample_meta = self.samples[idx]
        current_image_id = sample_meta["current_image_id"]
        bbox_sample = copy.deepcopy(self._load_bbox_payload(sample_meta["bbox_json_path"]))

        temporal_images = []
        temporal_pixel_masks = []
        temporal_abnormality_labels = []
        temporal_abnormality_masks = []
        current_width = None
        current_height = None

        for time_idx, image_ref in enumerate(sample_meta["temporal_image_refs"]):
            parsed = self._parse_image_ref(image_ref)
            image_id = parsed["image_id"]
            image_path = os.path.join(self.image_root, parsed["relative_path"])
            if not os.path.exists(image_path):
                image_path = os.path.join(self.image_root, f"{image_id}.jpg")
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            width, height = image.size
            if time_idx == sample_meta["temporal_length"] - 1:
                current_width = width
                current_height = height

            image_tensor, pixel_mask = self._preprocess_image(image)
            temporal_images.append(image_tensor)
            temporal_pixel_masks.append(pixel_mask)
            temporal_abnormality = {key: self.label_map[image_id][key].clone() for key in ("image_labels", "image_label_mask")}
            temporal_abnormality_labels.append(temporal_abnormality["image_labels"])
            temporal_abnormality_masks.append(temporal_abnormality["image_label_mask"])

        image_tensor = torch.stack(temporal_images, dim=0)
        pixel_mask = torch.stack(temporal_pixel_masks, dim=0)
        temporal_image_labels = torch.stack(temporal_abnormality_labels, dim=0)
        temporal_image_label_mask = torch.stack(temporal_abnormality_masks, dim=0)
        anatomy_boxes, anatomy_slot_mask = self._build_anatomy_targets(
            bbox_sample,
            current_width,
            current_height,
        )

        abnormality = {
            "image_labels": temporal_image_labels[-1].clone(),
            "image_label_mask": temporal_image_label_mask[-1].clone(),
            "temporal_image_labels": temporal_image_labels,
            "temporal_image_label_mask": temporal_image_label_mask,
        }
        phrase_id_to_text, region_records = self._load_supervision_payload(sample_meta["scene_graph_path"])

        user_text = self._build_user_text(
            sample_meta["instruction"],
            sample_meta["temporal_length"],
            sample_meta=sample_meta,
        )
        input_ids, labels = self._tokenize_example(user_text, sample_meta["answer"])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "image": image_tensor,
            "pixel_mask": pixel_mask,
            "temporal_length": sample_meta["temporal_length"],
            "image_id": current_image_id,
            "study_id": sample_meta["current_study_id"],
            "image_size": (current_height, current_width),
            "phrase_id_to_text": phrase_id_to_text,
            "region_records": region_records,
            "source": self.source,
            "index": idx,
            "image_refs": list(sample_meta["temporal_image_refs"]),
            "answer_text": sample_meta["answer"],
            "user_text": user_text,
            "refer_slot_ids": list(sample_meta.get("refer_slot_ids") or []),
            "ground_slot_ids": list(sample_meta.get("ground_slot_ids") or []),
            "labels_meta": {
                "anatomy": {"boxes": anatomy_boxes, "slot_mask": anatomy_slot_mask},
                "abnormality": abnormality,
            },
            "metadata": copy.deepcopy(sample_meta["metadata"]),
        }


class MIMICSingleReportDataset(BaseMedicalLLMDataset):
    def __init__(self, **kwargs):
        super().__init__(source=kwargs.pop("source", "mimic_llm_report_single"), **kwargs)

    def _build_sample(self, entry: Dict[str, object]) -> Optional[Dict[str, object]]:
        parsed = self._parse_image_ref(entry["image"])
        instruction, answer = self._extract_prompt_pair(entry["conversations"])
        if not instruction or not answer:
            return None
        return self._finalize_sample(
            image_refs=[parsed["relative_path"]],
            instruction=instruction,
            answer=answer,
            current_image_id=parsed["image_id"],
            current_study_id=parsed["study_id"],
            metadata={"raw_image": entry["image"]},
        )


class MIMICTemporalReportDataset(BaseMedicalLLMDataset):
    default_instruction = "Provide a radiology report for this Chest X-Ray in comparison to the prior exam."

    def __init__(self, instruction: Optional[str] = None, **kwargs):
        if instruction is not None:
            self.default_instruction = _normalize_text(instruction).replace("<image>", " ").strip()
        super().__init__(source=kwargs.pop("source", "mimic_llm_report_temporal"), **kwargs)

    def _build_sample(self, entry: Dict[str, object]) -> Optional[Dict[str, object]]:
        current_parsed = self._parse_image_ref(entry["image"])
        current_image_id = _normalize_scalar_id(entry["current_image_id"])
        current_study_id = _normalize_scalar_id(entry["current_study_id"])
        patient_id = _normalize_scalar_id(entry["patient_id"])
        prior_study_id = _normalize_scalar_id(entry["prior_study_id"])
        prior_image_id = _normalize_scalar_id(entry["prior_image_id"])
        prior_relative_path = f"files/{patient_id[:3]}/{patient_id}/{prior_study_id}/{prior_image_id}.jpg"
        return self._finalize_sample(
            image_refs=[prior_relative_path, current_parsed["relative_path"]],
            instruction=self.default_instruction,
            answer=_normalize_text(entry["temporal_gpt"]),
            current_image_id=current_image_id,
            current_study_id=current_study_id,
            metadata={
                "raw_image": entry["image"],
                "prior_image_id": _normalize_scalar_id(entry.get("prior_image_id")),
                "prior_study_id": _normalize_scalar_id(entry.get("prior_study_id")),
                "original_gpt": _normalize_text(entry.get("original_gpt")),
            },
        )


class MIMICAlignmentDataset(BaseMedicalLLMDataset):
    def __init__(self, **kwargs):
        super().__init__(source=kwargs.pop("source", "mimic_llm_alignment"), **kwargs)

    def _build_sample(self, entry: Dict[str, object]) -> Optional[Dict[str, object]]:
        image_field = entry["image"]
        ordered_refs, current_parsed = self._normalize_alignment_image_sequence(image_field)

        instruction, answer = self._extract_prompt_pair(entry["conversations"])
        if not instruction or not answer:
            return None

        return self._finalize_sample(
            image_refs=ordered_refs,
            instruction=instruction,
            answer=answer,
            current_image_id=current_parsed["image_id"],
            current_study_id=current_parsed["study_id"],
            metadata={"row_id": entry.get("id"), "raw_image": list(image_field)},
        )


class MIMICGroundingDataset(BaseMedicalLLMDataset):
    def __init__(self, **kwargs):
        super().__init__(source=kwargs.pop("source", "mimic_llm_grounding"), **kwargs)

    def _replace_anatomy_tags(
        self,
        text: str,
        *,
        tag_mode: str,
    ) -> Tuple[str, List[int]]:
        slot_ids: List[int] = []

        def _replace(match: re.Match[str]) -> str:
            tag_name = _normalize_region_name(match.group(1))
            if tag_name == _normalize_region_name(DEFAULT_TOKENS["image"].strip("<>")):
                return match.group(0)

            slot_ids.append(int(self.static_region_name_to_idx[tag_name]))
            if tag_mode == "refer":
                return (
                    DEFAULT_TOKENS["bor"]
                    + DEFAULT_TOKENS["rbox"]
                    + DEFAULT_TOKENS["eor"]
                    + DEFAULT_TOKENS["rfeat"]
                )
            return DEFAULT_TOKENS["bor"] + DEFAULT_TOKENS["gbox"] + DEFAULT_TOKENS["eor"]

        rewritten = _ANATOMY_TAG_RE.sub(_replace, _normalize_text(text))
        return rewritten, slot_ids

    def _build_sample(self, entry: Dict[str, object]) -> Optional[Dict[str, object]]:
        image_field = entry["image"]
        ordered_refs, current_parsed = self._normalize_alignment_image_sequence(image_field)

        instruction, answer = self._extract_prompt_pair(entry["conversations"])
        if not instruction or not answer:
            return None

        rewritten_instruction, refer_slot_ids = self._replace_anatomy_tags(instruction, tag_mode="refer")
        rewritten_answer, ground_slot_ids = self._replace_anatomy_tags(answer, tag_mode="ground")

        return self._finalize_sample(
            image_refs=ordered_refs,
            instruction=rewritten_instruction,
            answer=rewritten_answer,
            current_image_id=current_parsed["image_id"],
            current_study_id=current_parsed["study_id"],
            refer_slot_ids=refer_slot_ids,
            ground_slot_ids=ground_slot_ids,
            metadata={
                "row_id": entry.get("id"),
                "raw_image": list(image_field),
                "raw_instruction": instruction,
                "raw_answer": answer,
            },
        )
