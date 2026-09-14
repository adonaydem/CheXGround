from bisect import bisect_right
from collections import defaultdict
from typing import Iterator, List

import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler, Subset


def get_temporal_length(dataset: Dataset, index: int) -> int:
    if isinstance(dataset, Subset):
        return get_temporal_length(dataset.dataset, int(dataset.indices[index]))

    if isinstance(dataset, ConcatDataset):
        dataset_idx = bisect_right(dataset.cumulative_sizes, index)
        sample_offset = 0 if dataset_idx == 0 else dataset.cumulative_sizes[dataset_idx - 1]
        return get_temporal_length(dataset.datasets[dataset_idx], index - sample_offset)

    if hasattr(dataset, "get_temporal_length"):
        return int(dataset.get_temporal_length(index))

    samples = getattr(dataset, "samples", None)
    if samples is not None:
        temporal_length = samples[index].get("temporal_length")
        if temporal_length is not None:
            return int(temporal_length)

    raise TypeError(f"Dataset does not expose temporal lengths: {type(dataset).__name__}")


def has_temporal_lengths(dataset: Dataset) -> bool:
    try:
        return len(dataset) > 0 and get_temporal_length(dataset, 0) > 0
    except Exception:
        return False


class TemporalLengthBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        generator: torch.Generator = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def _get_generator(self) -> torch.Generator:
        if self.generator is not None:
            return self.generator
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _build_batches(self) -> List[List[int]]:
        grouped_indices = defaultdict(list)
        for index in range(len(self.dataset)):
            grouped_indices[get_temporal_length(self.dataset, index)].append(index)

        temporal_lengths = sorted(grouped_indices)
        batches = []
        generator = self._get_generator() if self.shuffle else None

        for temporal_length in temporal_lengths:
            group_indices = grouped_indices[temporal_length]
            if self.shuffle:
                perm = torch.randperm(len(group_indices), generator=generator).tolist()
                group_indices = [group_indices[i] for i in perm]
            for start in range(0, len(group_indices), self.batch_size):
                batch = group_indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle and batches:
            perm = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in perm]
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._build_batches()

    def __len__(self) -> int:
        total_batches = 0
        counts_by_length = defaultdict(int)
        for index in range(len(self.dataset)):
            counts_by_length[get_temporal_length(self.dataset, index)] += 1
        for count in counts_by_length.values():
            if self.drop_last:
                total_batches += count // self.batch_size
            else:
                total_batches += (count + self.batch_size - 1) // self.batch_size
        return total_batches
