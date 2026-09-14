import copy

import numpy as np
from chexground.openmmlab_support import ensure_openmmlab_paths

ensure_openmmlab_paths()

try:
    from mmcv.utils import Config
except ImportError:
    try:
        from mmengine.config import Config
    except ImportError:
        from mmcv import Config
from torch.utils.data import ConcatDataset, Subset
from chexground.data.datasets.chest_det import MIMICDetDataset
from chexground.data.datasets.chest_temporal_grounding import MIMICTemporalGroundingDataset
from chexground.data.datasets.medical_llm import (
    MIMICAlignmentDataset,
    MIMICGroundingDataset,
    MIMICSingleReportDataset,
    MIMICTemporalReportDataset,
)


DATASET_REGISTRY = {
    "mimic_det": MIMICDetDataset,
    "mimic_temporal_grounding": MIMICTemporalGroundingDataset,
    "mimic_llm_report_single": MIMICSingleReportDataset,
    "mimic_llm_report_temporal": MIMICTemporalReportDataset,
    "mimic_llm_alignment": MIMICAlignmentDataset,
    "mimic_llm_grounding": MIMICGroundingDataset,
}


def _select_stage_dataset_cfgs(config, stage):
    stage_key = f"{stage}_datasets"
    if hasattr(config, stage_key):
        dataset_cfgs = getattr(config, stage_key)
    elif stage == "train" and hasattr(config, "datasets"):
        dataset_cfgs = config.datasets
    else:
        return None

    if dataset_cfgs is None:
        return None
    if not isinstance(dataset_cfgs, list):
        raise TypeError(f"{stage_key if hasattr(config, stage_key) else 'datasets'} must be a list.")
    if len(dataset_cfgs) == 0:
        return None
    return dataset_cfgs


def build_multi_datasets(dataset_cfg_file, tokenizer=None, stage="train", **kwargs):
    config = Config.fromfile(dataset_cfg_file)
    dataset_cfgs = _select_stage_dataset_cfgs(config, stage=stage)
    if dataset_cfgs is None:
        return None

    datasets = []
    for cfg in dataset_cfgs:
        dataset = build_dataset(cfg, tokenizer=tokenizer, **kwargs)
        if stage != "train" and len(dataset) == 0:
            continue
        datasets.append(dataset)

    if len(datasets) == 0:
        return None
    return ConcatDataset(datasets)


def _apply_dataset_ratio(dataset, ratio):
    if ratio <= 0:
        raise ValueError(f"dataset ratio must be positive, got {ratio}.")
    if ratio == 1:
        return dataset

    dataset_length = len(dataset)
    if dataset_length == 0:
        return dataset

    if ratio < 1:
        sample_count = max(1, int(dataset_length * ratio))
        random_indices = np.random.choice(dataset_length, sample_count, replace=False)
        return Subset(dataset, random_indices.tolist())

    repeat_count = int(ratio)
    fractional = ratio - repeat_count
    datasets = [dataset for _ in range(repeat_count)]
    if fractional > 0:
        sample_count = max(1, int(dataset_length * fractional))
        random_indices = np.random.choice(dataset_length, sample_count, replace=False)
        datasets.append(Subset(dataset, random_indices.tolist()))
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_dataset(dataset_cfg, tokenizer=None, **kwargs):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop('type')
    ratio = dataset_cfg.pop('ratio', 1)

    if dataset_type not in DATASET_REGISTRY:
        raise NotImplementedError(f"Unknown dataset type: {dataset_type}")

    dataset_cls = DATASET_REGISTRY[dataset_type]
    if dataset_type in {
        "mimic_llm_report_single",
        "mimic_llm_report_temporal",
        "mimic_llm_alignment",
        "mimic_llm_grounding",
    }:
        dataset = dataset_cls(**dataset_cfg, tokenizer=tokenizer, **kwargs)
    else:
        dataset = dataset_cls(**dataset_cfg, **kwargs)

    return _apply_dataset_ratio(dataset, ratio)
