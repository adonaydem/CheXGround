from typing import Iterator, List, Optional, Sized

import torch
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import has_length


class RandomBatchSampler(Sampler):
    data_source: Sized

    def __init__(
            self,
            data_source: Sized,
            batch_size: int,
            dataset_sizes: List[int],
            generator=None
    ) -> None:

        self.data_source = data_source
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes
        self.generator = generator
        self._usable_dataset_sizes = [max(0, int(size) - int(size) % self.batch_size) for size in dataset_sizes]
        self._num_samples = sum(self._usable_dataset_sizes)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        indices = torch.arange(n)
        chunked_indices = torch.split(indices, self.dataset_sizes)
        # sample-level permutation in each dataset
        inner_perm_indices = [x[torch.randperm(len(x), generator=generator)] for x in chunked_indices]
        inner_perm_indices = [
            x[: usable_size]
            for x, usable_size in zip(inner_perm_indices, self._usable_dataset_sizes)
            if usable_size > 0
        ]
        # split into batches
        outer_perm_indices = [torch.split(x, self.batch_size) for x in inner_perm_indices]
        outer_perm_indices = [y for x in outer_perm_indices for y in x]
        if not outer_perm_indices:
            return
        # batch-level permutation
        outer_perm_indices = [outer_perm_indices[i] for i in
                              torch.randperm(len(outer_perm_indices), generator=generator)]
        outer_perm_indices = torch.cat(outer_perm_indices, dim=0)
        yield from outer_perm_indices.tolist()

    def __len__(self) -> int:
        return self.num_samples


class CheXGroundTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_data_source:
            if not hasattr(self.train_dataset, "cumulative_sizes"):
                raise ValueError(
                    "group_by_data_source=True requires a ConcatDataset-style train_dataset with cumulative_sizes."
                )
            cumu_sizes = self.train_dataset.cumulative_sizes
            dataset_sizes = [cumu_sizes[0]] + [cumu_sizes[i] - cumu_sizes[i - 1] for i in range(1, len(cumu_sizes))]
            return RandomBatchSampler(
                self.train_dataset,
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset_sizes
            )
        return super()._get_train_sampler()
