# Batcher: takes data config, dataset reader cache (add more as needed), tokenizer. It creates some kind of dataloader, and returns (i) a generator of batches.
import torch

from typing import List, Optional, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

T_co = TypeVar("T_co", covariant=True)


def create_collate_fn(pad_token_id, max_length=None):
    def collate_fn(batch):
        fields = {field for example in batch for field in example.keys()}
        output_batch = {}
        for key in fields:
            if key in ["input_ids", "target_ids"]:
                output_batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [example[key] for example in batch],
                    batch_first=True,
                    padding_value=pad_token_id,
                )
                if max_length is not None:
                    output_batch[key] = output_batch[key][..., :max_length]
                # cast to long
                output_batch[key] = output_batch[key].long()
            elif key == "answer_choices_ids":
                flat_answer_choice_ids = [
                    choice for example in batch for choice in example[key]
                ]
                num_choice = [len(example[key]) for example in batch]
                if max(num_choice) == 0:
                    continue
                # if max(num_choice) != min(num_choice) or max(num_choice) == 0:
                #     continue
                #     raise NotImplementedError(
                #         "The collate_fn is not implmented for variable number of choices"
                #     )
                if max(num_choice) != min(num_choice):
                    # print(f"Encountered variable number of choices: {num_choice}")
                    padded_choices = []
                    for example in batch:
                        example_choices = example[key]
                        while len(example_choices) < max(num_choice):
                            example_choices.append(torch.tensor([pad_token_id]))
                        padded_choices.extend(example_choices)
                    flat_answer_choice_ids = padded_choices
                flat_answer_choices_ids = torch.nn.utils.rnn.pad_sequence(
                    flat_answer_choice_ids, batch_first=True, padding_value=pad_token_id
                )
                output_batch[key] = flat_answer_choices_ids.view(
                    len(batch), max(num_choice), -1
                )
                if max_length is not None:
                    output_batch[key] = output_batch[key][..., :max_length]
            elif key == "label":
                output_batch[key] = torch.cat([example[key] for example in batch])
            else:
                output_batch[key] = [example[key] for example in batch]

        return output_batch

    return collate_fn


class BaseBatcher(object):
    def __init__(self, shuffle, drop_last, num_workers):
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.seed = None
        self._rng = None

    def set_seed(self, seed):
        self.seed = seed

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def build(self, datasets):
        raise NotImplementedError()


class MultiTaskBatcher(BaseBatcher):
    def __init__(
        self,
        shuffle,
        drop_last,
        num_workers,
        temperature,
        num_replicas=-1,
        rank=-1,
        mixing_ratio=None,
    ):
        super().__init__(shuffle, drop_last, num_workers)
        self.temperature = temperature
        self.mixing_ratio = mixing_ratio
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.rank = rank

        assert drop_last, "drop_last must be True for MultiTaskBatcher"

    def build(self, datasets):
        joint_dataset = torch.utils.data.ConcatDataset(datasets)
        max_length = max([dataset.max_length for dataset in datasets])
        # TODO: assuming that all datasets have the same tokenizer
        dataloader = torch.utils.data.DataLoader(
            joint_dataset,
            batch_sampler=MultiTaskBatchSampler(
                dataset_sizes=[len(dataset) for dataset in datasets],
                batch_sizes=[dataset.batch_size for dataset in datasets],
                temperature=self.temperature,
                seed=self.seed,
                shuffle=self.shuffle,
                mixing_ratio=self.mixing_ratio,
                num_replicas=self.num_replicas,
                rank=self.rank,
            ),
            num_workers=self.num_workers,
            collate_fn=create_collate_fn(
                datasets[0].tokenizer.pad_token_id, max_length
            ),
        )
        return dataloader


# https://pytorch.org/docs/stable/notes/randomness.html
class SingleTaskBatcher(BaseBatcher):
    def build(self, dataset):
        if isinstance(dataset, list):
            assert len(dataset) == 1, "SingleTaskBatcher only supports one dataset"
            dataset = dataset[0]
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            collate_fn=create_collate_fn(
                dataset.tokenizer.pad_token_id, dataset.max_length
            ),
            generator=generator,
        )
        return data_loader


"""Implements a distributed sampler to sample different tasks with
temperature sampling in a way to make sure that the same task is
selected in each core. Modified from Hyperformer codebase."""


class MultiTaskBatchSampler(Sampler[T_co]):
    """Defines a sampler to sample multiple datasets with temperature sampling
    in a distributed fashion."""

    def __init__(
        self,
        dataset_sizes: List[int],
        batch_sizes: List[int],
        temperature: float,
        shuffle: bool = True,
        num_replicas: Optional[int] = -1,
        rank: Optional[int] = -1,
        seed: int = 42,
        mixing_ratio: Optional[List[float]] = None,
    ) -> None:
        """Constructor for MultiTaskBatchSampler.
        Args:
            dataset_sizes: a list of integers, specifies the number of samples in
                each dataset.
            batch_sizes: a list of integer, specifies the batch size in each dataset.
            temperature: float, temperature used for temperature sampling. The larger
                the value, the datasets are sampled equally, and for value of 0, the datasets
                will be sampled according to their number of samples.
            num_replicas: integer, specifies the number of processes.
            rank: integer, specifies the rank of the current process.
            seed: integer, random seed.
            shuffle: bool, if set to true, the datasets will be shuffled in each epoch.
        """
        if num_replicas == -1:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank == -1:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_sizes = batch_sizes
        self.dataset_sizes = dataset_sizes
        # By default we drop the last elements if dataset is not divisible by the number of ranks.
        self.rank_dataset_sizes = [
            dataset_size // self.num_replicas for dataset_size in self.dataset_sizes
        ]
        self.dataset_offsets = torch.cumsum(torch.LongTensor([0] + dataset_sizes), 0)
        self.total_sizes = [
            dataset_size
            // (self.num_replicas * batch_size)
            * (self.num_replicas * batch_size)
            for dataset_size, batch_size in zip(self.dataset_sizes, self.batch_sizes)
        ]
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0
        self.num_batches_per_epoch = sum(
            [
                dataset_size // (self.num_replicas * batch_size)
                for dataset_size, batch_size in zip(
                    self.dataset_sizes, self.batch_sizes
                )
            ]
        )
        self.shuffle = shuffle
        self.mixing_ratio = mixing_ratio
        self.batch_size = min(self.batch_sizes)

    def generate_tasks_distribution(self):
        """Given the dataset sizes computes the weights to sample each dataset
        according to the temperature sampling."""
        if self.mixing_ratio is not None:
            assert len(self.mixing_ratio) == len(
                self.dataset_sizes
            ), f"Size mismatch between mixing ratio {len(self.mixing_ratio)} and number of datasets: {self.dataset_sizes}"
            return torch.as_tensor(self.mixing_ratio, dtype=torch.double)

        total_size = sum(self.dataset_sizes)
        weights = np.array(
            [
                (size / total_size) ** (1.0 / self.temperature)
                for size in self.dataset_sizes
            ]
        )
        weights = weights / np.sum(weights)
        return torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        # print(f"the value of epoch is {self.epoch}")
        # Defines torch generator, to make random choices consistent across cores in
        # different epochs, the seed needs to be set based on seed and epoch.
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # Shuffles the datasets if shuffle is set to true, and shards the datasets per rank.
        self.rank_indices = []
        for dataset_size, total_size in zip(self.dataset_sizes, self.total_sizes):
            if self.shuffle:
                dataset_indices = torch.randperm(
                    dataset_size, generator=generator
                ).tolist()
            else:
                dataset_indices = list(range(dataset_size))
            self.rank_indices.append(
                dataset_indices[self.rank : total_size : self.num_replicas]
            )

        # To make the model consistent across different processes, since the
        # model is based on tasks, we need to make sure the same task is selected
        # across different processes.
        tasks_distribution: torch.Tensor = self.generate_tasks_distribution()

        # Chooses the tasks which will be used in each batch in one epoch.
        # With passing generator, we make sure this choice is consistent across
        # different processes.
        batch_task_assignments = torch.multinomial(
            tasks_distribution,
            self.num_batches_per_epoch,
            replacement=True,
            generator=generator,
        )
        pointers = [0 for i in range(len(self.rank_indices))]
        # print(f"the batch_task_assignments are {batch_task_assignments}")
        for batch_task in batch_task_assignments:
            # Gets the number of samples of the selected datasets available for the
            # current rank.
            batch_size = self.batch_sizes[batch_task]
            if pointers[batch_task] >= len(self.rank_indices[batch_task]):
                # shuffle the list self.rank_indices[batch_task]
                # print(f"reshuffling indices are {self.rank_indices[batch_task]}")
                self.rank_indices[batch_task] = [
                    self.rank_indices[batch_task][i]
                    for i in torch.randperm(len(self.rank_indices[batch_task]))
                ]
                # print(f"shuffled indices are {self.rank_indices[batch_task]}")
                pointers[batch_task] = 0
            # samples are already randomized in self.rank_indices
            results = (
                self.dataset_offsets[batch_task]
                + torch.tensor(
                    self.rank_indices[batch_task][
                        pointers[batch_task] : pointers[batch_task] + batch_size
                    ]
                )
            ).tolist()
            pointers[batch_task] += batch_size
            # # update self.rank_indices
            # self.rank_indices[batch_task] = (
            #     self.rank_indices[batch_task][batch_size:]
            #     + self.rank_indices[batch_task][:batch_size]
            # )

            # num_task_samples = self.rank_dataset_sizes[batch_task]
            # # Computes the random samples from the chosen dataset.
            # indices = torch.randint(
            #     low=0,
            #     high=num_task_samples,
            #     size=(batch_size,),
            #     generator=generator,
            # ).tolist()
            # # Converts the selected indices to the global indices on the given dataset.
            # results = (
            #     self.dataset_offsets[batch_task]
            #     + torch.tensor(self.rank_indices[batch_task])[indices]
            # ).tolist()
            yield results

        # update self.epoch to have different random samples in the next epoch
        self.epoch += 1

    def __len__(self):
        return self.num_batches_per_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch
        # TODO: Find a way to make DDP work without explicitly setting the epoch.
        # What is an epoch when we have temperature > 1? This feels really weird.
