import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor
from transformers.image_transforms import center_crop
from transformers.image_utils import ChannelDimension, PILImageResampling

IMAGE_GEOMETRY_MODES = frozenset({"center_pad", "one_side_pad", "square_resize"})


class _PaddedProcessedDetDataset(Dataset):
    def __init__(
        self,
        source,
        length=128,
        image_size=518,
        image_geometry_mode="one_side_pad",
        anatomy_num_queries=300,
        abnormality_num_classes=2,
        **_,
    ):
        self.source = source
        self.length = length
        self.image_size = image_size
        self.image_geometry_mode = image_geometry_mode
        self.anatomy_num_queries = anatomy_num_queries
        self.abnormality_num_classes = abnormality_num_classes

    def __len__(self):
        return self.length

    def _preprocess_valid_mask(self, valid_mask, output_size):
        mask = np.asarray(valid_mask, dtype=np.uint8)
        mask = mask[..., None]
        input_data_format = ChannelDimension.LAST

        if getattr(self.image_processor, "do_resize", False):
            mask = self.image_processor.resize(
                image=mask,
                size=self.image_processor.size,
                resample=PILImageResampling.NEAREST,
                input_data_format=input_data_format,
            )

        if getattr(self.image_processor, "do_center_crop", False):
            crop_size = self.image_processor.crop_size
            mask = center_crop(
                image=mask,
                size=(crop_size["height"], crop_size["width"]),
                data_format=ChannelDimension.LAST,
                input_data_format=input_data_format,
            )

        mask = mask.squeeze(-1)

        mask_tensor = torch.as_tensor(mask > 0, dtype=torch.bool)
        if tuple(mask_tensor.shape) != tuple(output_size):
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor[None, None].float(),
                size=output_size,
                mode="nearest",
            )[0, 0] > 0.5
        return mask_tensor

    def _convert_box_to_model_space(self, box, width, height):
        if self.image_geometry_mode == "center_pad":
            cx, cy, bw, bh = box
            side = float(max(width, height))
            pad_x = (side - float(width)) / 2.0
            pad_y = (side - float(height)) / 2.0

            cx_abs = float(cx) * float(width)
            cy_abs = float(cy) * float(height)
            bw_abs = float(bw) * float(width)
            bh_abs = float(bh) * float(height)

            cx_sq = (cx_abs + pad_x) / side
            cy_sq = (cy_abs + pad_y) / side
            bw_sq = bw_abs / side
            bh_sq = bh_abs / side
            return [max(0.0, min(1.0, v)) for v in (cx_sq, cy_sq, bw_sq, bh_sq)]
        if self.image_geometry_mode == "one_side_pad":
            cx, cy, bw, bh = box
            side = float(max(width, height))
            cx_sq = float(cx) * float(width) / side
            cy_sq = float(cy) * float(height) / side
            bw_sq = float(bw) * float(width) / side
            bh_sq = float(bh) * float(height) / side
            return [max(0.0, min(1.0, v)) for v in (cx_sq, cy_sq, bw_sq, bh_sq)]
        cx, cy, bw, bh = box
        return [max(0.0, min(1.0, float(v))) for v in (cx, cy, bw, bh)]

    def _preprocess_image(self, image):
        processor_kwargs = {"return_tensors": "pt"}
        width, height = image.size
        pixel_mask = None

        if self.image_geometry_mode in {"center_pad", "one_side_pad"}:
            side = max(width, height)
            model_image = image
            if width != height:
                model_image = Image.new(image.mode, (side, side), (0, 0, 0))
                offset = ((side - width) // 2, (side - height) // 2) if self.image_geometry_mode == "center_pad" else (0, 0)
                model_image.paste(image, offset)
            if self.image_geometry_mode == "one_side_pad":
                pixel_mask = Image.new("L", (side, side), color=0)
                content_mask = Image.new("L", (width, height), color=255)
                pixel_mask.paste(content_mask, (0, 0))
        else:
            model_image = image.resize((self.image_size, self.image_size), resample=Image.Resampling.BICUBIC)
            processor_kwargs["do_resize"] = False
            if getattr(self.image_processor, "do_center_crop", False):
                processor_kwargs["do_center_crop"] = False

        pixel_values = self.image_processor(images=model_image, **processor_kwargs)["pixel_values"].squeeze(0)
        if pixel_mask is None:
            pixel_mask = torch.ones(pixel_values.shape[-2:], dtype=torch.bool)
        else:
            pixel_mask = self._preprocess_valid_mask(pixel_mask, pixel_values.shape[-2:])
        return pixel_values, pixel_mask


class MIMICDetDataset(_PaddedProcessedDetDataset):
    def __init__(
        self,
        image_root="/path/to/images",
        bbox_dir="/path/to/anatomy_boxes",
        labels_csv_path="/path/to/abnormality_labels.csv",
        split=None,
        image_processor_name=None,
        length=None,
        **kwargs,
    ):
        super().__init__(
            source="mimic",
            length=length if length is not None else 0,
            **kwargs,
        )

        self.image_root = image_root
        self.bbox_dir = kwargs.pop("bbox_json_path", bbox_dir)
        self.labels_csv_path = labels_csv_path
        self.split = split
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_name)

        if self.split == 'train' and kwargs.get("use_augmentation", False):
            import albumentations as A
            self.transform = A.Compose([
                A.Affine(
                    scale=(0.90, 1.10),
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    rotate=(-7, 7),
                    shear={"x": (0, 0), "y": (0, 0)},
                    mode=0,
                    cval=0,
                    p=0.7
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.5
                ),
                A.RandomGamma(
                    gamma_limit=(85, 115),
                    p=0.3
                ),
                A.GaussNoise(
                    var_limit=(10.0, 30.0),
                    p=0.2
                ),
                A.GaussianBlur(
                    blur_limit=(3, 5),
                    p=0.15
                ),
            ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.3, label_fields=["labels"]))
        else:
            self.transform = None

        anatomy_classes_path = os.path.join(self.bbox_dir, "anatomy_classes.json")
        with open(anatomy_classes_path, "r") as f:
            bbox_payload = json.load(f)

        anatomy_classes = bbox_payload.get("anatomy_classes", [])
        self.anatomy_class_to_idx = {name: idx for idx, name in enumerate(anatomy_classes)}

        label_map = {}
        with open(self.labels_csv_path, "r") as f:
            reader = csv.DictReader(f)
            fixed_meta_columns = {"image_id", "subjectid_studyid", "split", "patient_id", "study_id", "viewpoint"}
            abnormality_columns = sorted([name for name in reader.fieldnames if name not in fixed_meta_columns])

            for row in reader:
                if self.split is not None and row.get("split") != self.split:
                    continue
                image_id = row["image_id"]
                values = []
                masks = []
                for col in abnormality_columns:
                    val = float(row[col])
                    if val < 0:
                        values.append(0.0)
                        masks.append(False)
                    else:
                        values.append(val)
                        masks.append(True)
                label_map[image_id] = {
                    "image_labels": torch.tensor(values, dtype=torch.float32),
                    "image_label_mask": torch.tensor(masks, dtype=torch.bool),
                    "subjectid_studyid": row.get("subjectid_studyid") or image_id,
                }

        self.samples = []
        for image_id in label_map.keys():
            if os.path.exists(os.path.join(self.bbox_dir, f"{image_id}.json")):
                self.samples.append({"image_id": image_id})

        if length is not None:
            self.samples = self.samples[:length]
        self.label_map = label_map
        self.length = len(self.samples)

    def __getitem__(self, idx):
        sample_meta = self.samples[idx]
        image_id = sample_meta["image_id"]

        with open(os.path.join(self.bbox_dir, f"{image_id}.json"), "r") as f:
            sample = json.load(f)

        image_path = os.path.join(self.image_root, f"{image_id}.jpg")
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        width, height = image.size

        if self.transform is not None:
            bboxes = []
            labels = []
            for anatomy_item in sample.get("anatomy", []):
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

            img_arr = np.array(image)
            transformed = self.transform(image=img_arr, bboxes=bboxes, labels=labels)
            image = Image.fromarray(transformed["image"])
            width, height = image.size

            new_anatomy = []
            for bbox, label in zip(transformed["bboxes"], transformed["labels"]):
                new_anatomy.append({"class": label, "bbox": bbox})
            sample["anatomy"] = new_anatomy

        image_tensor, pixel_mask = self._preprocess_image(image)

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

        abnormality = {key: self.label_map[image_id][key].clone() for key in ("image_labels", "image_label_mask")}

        return {
            "image": image_tensor,
            "pixel_mask": pixel_mask,
            "labels": {
                "anatomy": {"boxes": anatomy_boxes, "slot_mask": anatomy_slot_mask},
                "abnormality": abnormality,
            },
            "source": self.source,
            "index": idx,
            "image_id": image_id,
            "image_size": (height, width),
        }
