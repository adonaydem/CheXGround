dataset = {
    "type": "mimic_temporal_grounding",
    "image_root": "/path/to/images",
    "bbox_json_path": "/path/to/anatomy_boxes",
    "labels_csv_path": "/path/to/abnormality_labels.csv",
    "sequence_csv_path": "/path/to/sequences.csv",
    "ratio": 1.0
}

train_datasets = [dict(dataset, split="train")]
val_datasets = [dict(dataset, split="val")]
test_datasets = [dict(dataset, split="test")]
