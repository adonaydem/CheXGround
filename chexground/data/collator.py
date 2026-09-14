import torch
import transformers
from dataclasses import dataclass

from chexground.constants import IGNORE_INDEX


@dataclass
class DataCollatorForHybridDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        meta_keys = ('input_ids', 'labels', 'image', 'source')
        input_ids, labels, images, sources = tuple(
            [instance.get(key, None) for instance in instances] for key in meta_keys)
        refer_boxes = [instance.get('refer_boxes', torch.empty(0, 4)) for instance in instances]
        ground_boxes = [instance.get('ground_boxes', torch.empty(0, 4)) for instance in instances]
        if all([x is not None for x in images]):
            images = torch.stack(images)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            images=images,
            refer_boxes=refer_boxes,
            ground_boxes=ground_boxes,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
        return batch


@dataclass
class DataCollatorForMedicalVLDataset(object):
    """Batch contract for the CheXGround LLM path.

    Required per-sample fields:
    - input_ids: 1D token ids
    - labels: 1D token labels aligned with input_ids
    - image: [T, C, H, W]
    - pixel_mask: [T, H, W]
    - temporal_length: logical frame count before left padding
    """

    tokenizer: transformers.PreTrainedTokenizer
    max_temporal_frames: int = 2

    def __call__(self, instances):
        if not instances:
            raise ValueError("DataCollatorForMedicalVLDataset requires at least one instance.")

        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        temporal_lengths = torch.tensor([int(instance["temporal_length"]) for instance in instances], dtype=torch.long)
        if (temporal_lengths <= 0).any():
            raise ValueError(f"temporal_length must be positive for every sample, got {temporal_lengths.tolist()}.")
        if (temporal_lengths > self.max_temporal_frames).any():
            raise ValueError(
                f"temporal_length exceeds max_temporal_frames={self.max_temporal_frames}: {temporal_lengths.tolist()}."
            )

        padded_images = []
        padded_masks = []
        for instance in instances:
            image = instance["image"]
            pixel_mask = instance["pixel_mask"]
            temporal_length = int(instance["temporal_length"])
            if image.ndim != 4:
                raise ValueError(f"Expected image to have shape [T, C, H, W], got {tuple(image.shape)}.")
            if pixel_mask.ndim != 3:
                raise ValueError(f"Expected pixel_mask to have shape [T, H, W], got {tuple(pixel_mask.shape)}.")
            if image.shape[0] != temporal_length:
                raise ValueError(
                    f"image frame count must match temporal_length. Got image={tuple(image.shape)} temporal_length={temporal_length}."
                )
            if pixel_mask.shape[0] != temporal_length:
                raise ValueError(
                    f"pixel_mask frame count must match temporal_length. Got pixel_mask={tuple(pixel_mask.shape)} temporal_length={temporal_length}."
                )
            if image.shape[0] > self.max_temporal_frames:
                raise ValueError(
                    f"image has more frames than max_temporal_frames={self.max_temporal_frames}: {tuple(image.shape)}."
                )
            if image.shape[0] != pixel_mask.shape[0]:
                raise ValueError(
                    f"image and pixel_mask frame counts must match, got image={tuple(image.shape)} pixel_mask={tuple(pixel_mask.shape)}."
                )
            if tuple(image.shape[-2:]) != tuple(pixel_mask.shape[-2:]):
                raise ValueError(
                    f"image and pixel_mask spatial shapes must match, got image={tuple(image.shape)} pixel_mask={tuple(pixel_mask.shape)}."
                )
            pad_frames = self.max_temporal_frames - image.shape[0]
            if pad_frames > 0:
                image_pad = torch.zeros((pad_frames, *image.shape[1:]), dtype=image.dtype)
                mask_pad = torch.zeros((pad_frames, *pixel_mask.shape[1:]), dtype=pixel_mask.dtype)
                image = torch.cat([image_pad, image], dim=0)
                pixel_mask = torch.cat([mask_pad, pixel_mask], dim=0)
            padded_images.append(image)
            padded_masks.append(pixel_mask)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "images": torch.stack(padded_images, dim=0),
            "pixel_mask": torch.stack(padded_masks, dim=0),
            "temporal_lengths": temporal_lengths,
            "image_ids": [instance["image_id"] for instance in instances],
            "study_ids": [instance["study_id"] for instance in instances],
            "sources": [instance.get("source") for instance in instances],
            "image_refs": [instance.get("image_refs") for instance in instances],
            "user_texts": [instance.get("user_text") for instance in instances],
            "answer_texts": [instance.get("answer_text") for instance in instances],
            "refer_slot_ids": [list(instance.get("refer_slot_ids") or []) for instance in instances],
            "ground_slot_ids": [list(instance.get("ground_slot_ids") or []) for instance in instances],
            "labels_meta": [instance.get("labels_meta") for instance in instances],
            "phrase_id_to_text": [instance.get("phrase_id_to_text") for instance in instances],
            "region_records": [instance.get("region_records") for instance in instances],
            "image_sizes": [instance.get("image_size") for instance in instances],
            "indices": [instance.get("index") for instance in instances],
            "metadata": [instance.get("metadata") for instance in instances],
        }


@dataclass
class DataCollatorForDetDataset(object):
    def __call__(self, instances):
        images = [instance["image"] for instance in instances]
        images = torch.stack(images)
        pixel_mask = [instance["pixel_mask"] for instance in instances]
        pixel_mask = torch.stack(pixel_mask)
        if tuple(pixel_mask.shape) != (images.shape[0], images.shape[-2], images.shape[-1]):
            raise ValueError(
                "pixel_mask must batch to [batch, height, width] aligned with images. "
                f"Got pixel_mask={tuple(pixel_mask.shape)} images={tuple(images.shape)}."
            )
        labels = [instance["labels"] for instance in instances]
        batch = dict(images=images, pixel_mask=pixel_mask, labels=labels)
        return batch


@dataclass
class DataCollatorForDetEvalDataset(object):
    def __call__(self, instances):
        meta_keys = ('image', 'ori_shape')
        images, ori_shapes = tuple([instance.get(key, None) for instance in instances] for key in meta_keys)
        images = torch.stack(images)
        ori_shapes = torch.stack([torch.tensor(x[:2]) for x in ori_shapes])
        batch = dict(images=images, ori_shapes=ori_shapes)
        return batch
