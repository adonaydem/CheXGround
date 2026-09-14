dataset = {
    "type": "mimic_llm_grounding",
    "image_root": "/path/to/images",
    "bbox_json_path": "/path/to/anatomy_boxes",
    "labels_csv_path": "/path/to/abnormality_labels.csv",
    "region_json_root": "/path/to/regions",
    "temporal_metadata_csv_path": "/path/to/temporal_metadata.csv",
    "conv_temp": "chexground_finetune",
    "max_temporal_frames": 2,
    "ratio": 1.0
}

train_datasets = [dict(dataset, split="train", ann_file="/path/to/train_annotations.json")]
val_datasets = [dict(dataset, split="val", ann_file="/path/to/val_annotations.json")]
