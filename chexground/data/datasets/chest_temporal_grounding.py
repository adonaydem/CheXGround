import csv
import copy
import json
import os
import re
from collections import OrderedDict

import torch
from PIL import Image

from .chest_det import MIMICDetDataset


def _normalize_region_name(name: str) -> str:
    if name is None:
        return ""
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name

def _normalize_scalar_id(value) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return ""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return value
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return value


def _dedupe_keep_order(items):
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


class MIMICTemporalGroundingDataset(MIMICDetDataset):
    SCENE_GRAPH_SUFFIX = "_SceneGraph.json"

    def __init__(
        self,
        image_root="/path/to/images",
        bbox_json_path="/path/to/anatomy_boxes",
        labels_csv_path="/path/to/abnormality_labels.csv",
        region_json_root=None,
        sequence_csv_path=None,
        split=None,
        image_processor_name=None,
        length=None,
        **kwargs,
    ):
        if region_json_root is None:
            raise ValueError("region_json_root is required for MIMICTemporalGroundingDataset.")
        if sequence_csv_path is None:
            raise ValueError("sequence_csv_path is required for MIMICTemporalGroundingDataset.")

        super().__init__(
            image_root=image_root,
            bbox_dir=bbox_json_path,
            labels_csv_path=labels_csv_path,
            split=split,
            image_processor_name=image_processor_name,
            length=None,
            **kwargs,
        )
        self.source = "mimic_temporal_grounding"

        self.bbox_json_path = bbox_json_path
        self.region_json_root = region_json_root
        self.sequence_csv_path = sequence_csv_path
        if not os.path.isdir(self.region_json_root):
            raise FileNotFoundError(f"region_json_root not found: {self.region_json_root}")
        if not os.path.exists(self.sequence_csv_path):
            raise FileNotFoundError(f"sequence_csv_path not found: {self.sequence_csv_path}")
        self.image_split_map = self._load_image_split_map()
        self._payload_cache_max_entries = 2048
        self._bbox_payload_cache = OrderedDict()
        self._scene_graph_payload_cache = OrderedDict()

        # DDETR anatomy slot i is trained against anatomy_classes[i]; use that exact order for bbox_name alignment.
        self.static_region_name_to_idx = {
            _normalize_region_name(name): idx for name, idx in self.anatomy_class_to_idx.items()
        }

        self.samples = self._build_sequence_samples()
        if length is not None:
            self.samples = self.samples[: min(length, len(self.samples))]
        self.length = len(self.samples)

        if self.length == 0:
            raise ValueError(
                "MIMICTemporalGroundingDataset has zero samples after joining temporal CSV, scene graph, and labels. "
                f"Requested split={self.split!r}."
            )

    def _scene_graph_path(self, image_id: str) -> str:
        return os.path.join(self.region_json_root, f"{image_id}{self.SCENE_GRAPH_SUFFIX}")

    def _get_cached_payload(self, cache: OrderedDict, key: str, loader):
        cached_value = cache.get(key)
        if cached_value is not None:
            cache.move_to_end(key)
            return cached_value

        loaded_value = loader()
        cache[key] = loaded_value
        if len(cache) > self._payload_cache_max_entries:
            cache.popitem(last=False)
        return loaded_value

    def _load_bbox_payload(self, bbox_json_path: str):
        def _loader():
            with open(bbox_json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        return self._get_cached_payload(self._bbox_payload_cache, bbox_json_path, _loader)

    def _load_image_split_map(self):
        split_map = {}
        with open(self.labels_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("labels_csv_path has no header.")
            if "image_id" not in reader.fieldnames:
                raise ValueError("labels_csv_path is missing required column 'image_id'.")
            for row in reader:
                image_id = _normalize_scalar_id(row.get("image_id"))
                if not image_id:
                    continue
                split_map[image_id] = (row.get("split") or "").strip()
        return split_map

    def _build_sequence_samples(self):
        with open(self.sequence_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("sequence_csv_path has no header.")

            required_columns = {"unique_id", "patient_id", "num_studies"}
            if not required_columns.issubset(reader.fieldnames):
                raise ValueError(
                    "sequence_csv_path is missing required columns. "
                    f"Required={sorted(required_columns)} actual={reader.fieldnames}"
                )

            image_columns = sorted(
                [column for column in reader.fieldnames if re.fullmatch(r"im\d+", column)],
                key=lambda name: int(name[2:]),
            )
            if not image_columns:
                raise ValueError("sequence_csv_path must contain im1..imT columns.")

            samples = []
            for row in reader:
                sample = self._build_sequence_sample(row=row, max_temporal_slots=len(image_columns))
                if sample is not None:
                    samples.append(sample)
        return samples

    def _build_sequence_sample(self, row, max_temporal_slots: int):
        num_studies_raw = row.get("num_studies")
        try:
            num_studies = int(float(num_studies_raw))
        except (TypeError, ValueError):
            return None
        if num_studies <= 0 or num_studies > max_temporal_slots:
            return None

        temporal_image_ids = []
        temporal_study_ids = []
        for time_idx in range(1, num_studies + 1):
            image_id = _normalize_scalar_id(row.get(f"im{time_idx}"))
            if not image_id:
                return None
            temporal_image_ids.append(image_id)
            temporal_study_ids.append(_normalize_scalar_id(row.get(f"s{time_idx}")) or image_id)

        current_image_id = temporal_image_ids[-1]
        if current_image_id not in self.label_map:
            return None
        if self.split is not None:
            for image_id in temporal_image_ids:
                if self.image_split_map.get(image_id) != self.split:
                    return None

        for image_id in temporal_image_ids:
            if not os.path.exists(os.path.join(self.image_root, f"{image_id}.jpg")):
                return None

        current_bbox_json_path = os.path.join(self.bbox_dir, f"{current_image_id}.json")
        current_scene_graph_path = self._scene_graph_path(current_image_id)
        if not os.path.exists(current_bbox_json_path):
            return None
        if not os.path.exists(current_scene_graph_path):
            return None

        return {
            "temporal_image_ids": temporal_image_ids,
            "temporal_length": len(temporal_image_ids),
            "current_image_id": current_image_id,
            "current_study_id": self.label_map[current_image_id].get("subjectid_studyid", current_image_id),
            "bbox_json_path": current_bbox_json_path,
            "scene_graph_path": current_scene_graph_path,
        }

    def _build_phrase_id_to_text(self, attributes):
        phrase_id_to_text = OrderedDict()
        for attribute in attributes:
            phrase_ids = attribute.get("phrase_IDs") or []
            phrases = attribute.get("phrases") or []
            for phrase_id, phrase_text in zip(phrase_ids, phrases):
                normalized_phrase_id = _normalize_scalar_id(phrase_id)
                cleaned_phrase_text = str(phrase_text).strip() if phrase_text is not None else ""
                if not normalized_phrase_id or not cleaned_phrase_text:
                    continue
                if normalized_phrase_id not in phrase_id_to_text:
                    phrase_id_to_text[normalized_phrase_id] = cleaned_phrase_text
        return phrase_id_to_text

    def _build_region_records(self, attributes, phrase_id_to_text):
        merged = OrderedDict()
        for attribute in attributes:
            bbox_name = attribute.get("bbox_name")
            slot_index = self.static_region_name_to_idx.get(_normalize_region_name(bbox_name))
            if slot_index is None:
                continue

            object_id = _normalize_scalar_id(attribute.get("object_id")) or None
            phrase_ids = []
            for phrase_id in attribute.get("phrase_IDs") or []:
                normalized_phrase_id = _normalize_scalar_id(phrase_id)
                if normalized_phrase_id and normalized_phrase_id in phrase_id_to_text:
                    phrase_ids.append(normalized_phrase_id)
            phrase_ids = _dedupe_keep_order(phrase_ids)
            if not phrase_ids:
                continue

            merge_key = object_id if object_id is not None else f"slot::{slot_index}"
            current = merged.setdefault(
                merge_key,
                {
                    "object_id": object_id,
                    "slot_index": slot_index,
                    "bbox_name": bbox_name,
                    "phrase_ids": [],
                },
            )
            current["phrase_ids"].extend(phrase_ids)

        output = []
        for region in merged.values():
            phrase_ids = _dedupe_keep_order(region["phrase_ids"])
            if not phrase_ids:
                continue
            output.append(
                {
                    "object_id": region["object_id"],
                    "slot_index": region["slot_index"],
                    "bbox_name": region["bbox_name"],
                    "phrase_ids": phrase_ids,
                }
            )
        output.sort(key=lambda record: (int(record["slot_index"]), record["object_id"] or ""))
        return output

    def _load_supervision_payload(self, scene_graph_path):
        def _loader():
            with open(scene_graph_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            attributes = payload.get("attributes") or []
            phrase_id_to_text = self._build_phrase_id_to_text(attributes)
            region_records = self._build_region_records(attributes, phrase_id_to_text)
            return phrase_id_to_text, region_records

        return self._get_cached_payload(self._scene_graph_payload_cache, scene_graph_path, _loader)

    def _build_anatomy_targets(self, sample, width, height):
        anatomy_boxes = torch.zeros((self.anatomy_num_queries, 4), dtype=torch.float32)
        anatomy_slot_mask = torch.zeros((self.anatomy_num_queries,), dtype=torch.bool)
        for anatomy_item in sample.get("anatomy", []):
            class_name = anatomy_item.get("class")
            if class_name not in self.anatomy_class_to_idx:
                continue
            slot_idx = self.anatomy_class_to_idx[class_name]
            if anatomy_slot_mask[slot_idx]:
                continue
            square_box = self._convert_box_to_model_space(
                anatomy_item.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                width,
                height,
            )
            anatomy_boxes[slot_idx] = torch.tensor(square_box, dtype=torch.float32)
            anatomy_slot_mask[slot_idx] = True
        return anatomy_boxes, anatomy_slot_mask

    def _apply_current_image_transform(self, image, bbox_sample):
        if getattr(self, "transform", None) is None:
            width, height = image.size
            return image, width, height, bbox_sample

        bboxes = []
        labels = []
        for anatomy_item in bbox_sample.get("anatomy", []):
            class_name = anatomy_item.get("class")
            raw_bbox = anatomy_item.get("bbox")
            if class_name not in self.anatomy_class_to_idx or raw_bbox is None:
                continue
            cx, cy, w, h = raw_bbox
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            if w <= 0.0 or h <= 0.0:
                continue
            bboxes.append([cx, cy, w, h])
            labels.append(class_name)

        import numpy as np

        transformed = self.transform(image=np.array(image), bboxes=bboxes, labels=labels)
        image = Image.fromarray(transformed["image"])
        width, height = image.size

        bbox_sample = dict(bbox_sample)
        bbox_sample["anatomy"] = [
            {"class": label, "bbox": bbox}
            for bbox, label in zip(transformed["bboxes"], transformed["labels"])
        ]
        return image, width, height, bbox_sample

    def get_temporal_length(self, idx: int) -> int:
        if idx < 0 or idx >= self.length:
            raise IndexError(f"index out of range: {idx} (len={self.length})")
        return int(self.samples[idx]["temporal_length"])

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"index out of range: {idx} (len={self.length})")

        sample_meta = self.samples[idx]
        current_image_id = sample_meta["current_image_id"]
        bbox_sample = copy.deepcopy(self._load_bbox_payload(sample_meta["bbox_json_path"]))

        temporal_images = []
        temporal_pixel_masks = []
        temporal_abnormality_labels = []
        temporal_abnormality_masks = []
        current_width = None
        current_height = None
        for time_idx, image_id in enumerate(sample_meta["temporal_image_ids"]):
            image_path = os.path.join(self.image_root, f"{image_id}.jpg")
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            width, height = image.size
            if time_idx == sample_meta["temporal_length"] - 1:
                image, width, height, bbox_sample = self._apply_current_image_transform(image, bbox_sample)
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

        return {
            "image": image_tensor,
            "pixel_mask": pixel_mask,
            "labels": {
                "anatomy": {"boxes": anatomy_boxes, "slot_mask": anatomy_slot_mask},
                "abnormality": abnormality,
            },
            "image_id": current_image_id,
            "study_id": sample_meta["current_study_id"],
            "image_size": (current_height, current_width),
            "phrase_id_to_text": phrase_id_to_text,
            "region_records": region_records,
            "source": self.source,
            "index": idx,
        }
