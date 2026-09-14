from dataclasses import dataclass
import re

import torch
from transformers import AutoTokenizer


_SECTION_HEADER_RE = re.compile(r"^(findings?|impression|wet read(?: version)?|portable ap chest radiograph)\s*:\s*", re.IGNORECASE)


def _normalize_text_for_tokenizer(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _SECTION_HEADER_RE.sub("", text)
    text = re.sub(r"^[\.\,\;\:\-]+\s*(?=[A-Za-z0-9])", "", text)
    return text.strip()


def _format_report_like_text(phrases):
    cleaned_phrases = []
    for phrase in phrases:
        phrase = _normalize_text_for_tokenizer(phrase)
        if not phrase:
            continue
        cleaned_phrases.append(phrase.rstrip(" ."))
    if not cleaned_phrases:
        return ""
    return ". ".join(cleaned_phrases) + "."


def _ensure_temporal_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image to have shape [T, C, H, W] or [C, H, W], got {tuple(image.shape)}.")
    return image


def _ensure_temporal_mask(pixel_mask: torch.Tensor) -> torch.Tensor:
    if pixel_mask.ndim == 2:
        return pixel_mask.unsqueeze(0)
    if pixel_mask.ndim != 3:
        raise ValueError(f"Expected pixel_mask to have shape [T, H, W] or [H, W], got {tuple(pixel_mask.shape)}.")
    return pixel_mask


@dataclass
class DataCollatorForTemporalGroundingDataset(object):
    text_encoder_name: str
    text_max_length: int = 96

    def __post_init__(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name, trust_remote_code=True)

    def _tokenize_texts(self, texts):
        if len(texts) == 0:
            return {
                "input_ids": torch.zeros((0, 1), dtype=torch.long),
                "attention_mask": torch.zeros((0, 1), dtype=torch.long),
            }

        encoded = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )
        return {key: value.to(dtype=torch.long) for key, value in encoded.items()}

    def __call__(self, instances):
        if not instances:
            raise ValueError("DataCollatorForTemporalGroundingDataset received an empty batch.")

        temporal_lengths = [_ensure_temporal_image(instance["image"]).shape[0] for instance in instances]
        unique_temporal_lengths = sorted(set(temporal_lengths))
        if len(unique_temporal_lengths) != 1:
            raise ValueError(
                f"All samples in a batch must have the same temporal_length, got {unique_temporal_lengths}."
            )

        images = torch.stack([_ensure_temporal_image(instance["image"]) for instance in instances], dim=0)
        pixel_mask = torch.stack([_ensure_temporal_mask(instance["pixel_mask"]) for instance in instances], dim=0)

        current_image_ids = [instance["image_id"] for instance in instances]
        current_study_ids = [instance.get("study_id", current_image_ids[idx]) for idx, instance in enumerate(instances)]

        image_labels = torch.stack(
            [instance["labels"]["abnormality"]["image_labels"] for instance in instances],
            dim=0,
        )
        image_label_mask = torch.stack(
            [instance["labels"]["abnormality"]["image_label_mask"] for instance in instances],
            dim=0,
        )
        temporal_image_labels = torch.stack(
            [instance["labels"]["abnormality"]["temporal_image_labels"] for instance in instances],
            dim=0,
        )
        temporal_image_label_mask = torch.stack(
            [instance["labels"]["abnormality"]["temporal_image_label_mask"] for instance in instances],
            dim=0,
        )

        num_images = len(instances)
        num_slots = instances[0]["labels"]["anatomy"]["boxes"].shape[0]
        same_study_mask = torch.zeros((num_images, num_images), dtype=torch.bool)
        for i, study_id_i in enumerate(current_study_ids):
            for j, study_id_j in enumerate(current_study_ids):
                if i != j and study_id_i == study_id_j:
                    same_study_mask[i, j] = True

        phrase_ids = []
        phrase_texts = []
        report_texts = []
        report_valid_mask = []
        phrase_batch_indices = []
        phrase_index_by_image_and_id = []

        for batch_image_index, instance in enumerate(instances):
            phrase_map = instance["phrase_id_to_text"]
            phrase_index_by_id = {}
            image_phrase_texts = []
            for phrase_id, phrase_text in phrase_map.items():
                phrase_id = str(phrase_id).strip()
                phrase_text = _normalize_text_for_tokenizer(phrase_text)
                if not phrase_id or not phrase_text:
                    continue
                phrase_index_by_id[phrase_id] = len(phrase_texts)
                phrase_ids.append(phrase_id)
                phrase_texts.append(phrase_text)
                image_phrase_texts.append(phrase_text)
                phrase_batch_indices.append(batch_image_index)

            phrase_index_by_image_and_id.append(phrase_index_by_id)
            report_texts.append(_format_report_like_text(image_phrase_texts))
            report_valid_mask.append(len(image_phrase_texts) > 0)

        num_phrases = len(phrase_texts)
        phrase_positive_roi_mask = torch.zeros((num_phrases, num_slots), dtype=torch.bool)

        for batch_image_index, instance in enumerate(instances):
            phrase_index_by_id = phrase_index_by_image_and_id[batch_image_index]
            for region_record in instance["region_records"]:
                slot_index = int(region_record["slot_index"])
                if slot_index < 0 or slot_index >= num_slots:
                    continue
                for phrase_id in region_record["phrase_ids"]:
                    phrase_index = phrase_index_by_id.get(str(phrase_id))
                    if phrase_index is None:
                        continue
                    phrase_positive_roi_mask[phrase_index, slot_index] = True

        phrase_tokens = self._tokenize_texts(phrase_texts)
        report_tokens = self._tokenize_texts(report_texts)

        batch = {
            "images": images,
            "pixel_mask": pixel_mask,
            "same_study_mask": same_study_mask,
            "image_labels": image_labels,
            "image_label_mask": image_label_mask,
            "temporal_image_labels": temporal_image_labels,
            "temporal_image_label_mask": temporal_image_label_mask,
            "phrase_ids": phrase_ids,
            "phrase_batch_index": torch.tensor(phrase_batch_indices, dtype=torch.long),
            "phrase_positive_roi_mask": phrase_positive_roi_mask,
            "report_valid_mask": torch.tensor(report_valid_mask, dtype=torch.bool),
            "phrase_input_ids": phrase_tokens["input_ids"],
            "phrase_attention_mask": phrase_tokens["attention_mask"],
            "report_input_ids": report_tokens["input_ids"],
            "report_attention_mask": report_tokens["attention_mask"],
            "image_ids": current_image_ids,
            "study_ids": current_study_ids,
            "temporal_lengths": torch.tensor(temporal_lengths, dtype=torch.long),
        }
        if "token_type_ids" in phrase_tokens:
            batch["phrase_token_type_ids"] = phrase_tokens["token_type_ids"]
        if "token_type_ids" in report_tokens:
            batch["report_token_type_ids"] = report_tokens["token_type_ids"]
        return batch
