import gzip
import json
import typing
import math

import torch
import numpy as np

import utils

LOGGER = utils.get_logger(__name__)
DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INDEX_TO_DNA = {v: k for k, v in DNA_ALPHABET.items()}
lookup_array = np.array([INDEX_TO_DNA[i] for i in range(len(INDEX_TO_DNA))])

DATA_PATH = '/mnt/ssd1/code/v1_unified/data/processed/viral_tiles_struct.jsonl.gz'


def dna_detokenize(seq):
    return ''.join([INDEX_TO_DNA[int(i)] for i in seq])


def batch_dna_detokenize(batch_seq):
    """batch_seq: numpy array [batch_size, seq_len] -> list of strings"""
    detokenized_batch = lookup_array[batch_seq]
    return [''.join(seq) for seq in detokenized_batch]


def dna_tokenize(seq):
    return [DNA_ALPHABET[c] for c in seq]


class NarryKimDataset(torch.utils.data.Dataset):
    def __init__(self, data_path=DATA_PATH):
        LOGGER.info(f'Loading sequences from {data_path}...')
        sequences = []
        with gzip.open(data_path, 'rt') as f:
            for line in f:
                record = json.loads(line)
                seq = record['sequence'].upper()
                # Skip sequences with non-ACGT characters
                if all(c in DNA_ALPHABET for c in seq):
                    sequences.append(dna_tokenize(seq))
        self.seqs = torch.tensor(sequences, dtype=torch.long)
        LOGGER.info(f'Loaded {len(self.seqs)} sequences, shape: {self.seqs.shape}')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            'seqs': self.seqs[idx],
            'attention_mask': torch.ones(self.seqs.shape[1]),
        }


def get_dataloaders(config, skip_valid=False, valid_seed=None):
    num_gpus = torch.cuda.device_count()
    if config.loader.global_batch_size % (
        num_gpus * config.trainer.accumulate_grad_batches) != 0:
        raise ValueError(
            f'Train Batch Size {config.loader.global_batch_size}'
            f' not divisible by {num_gpus} gpus with accumulation '
            f'{config.trainer.accumulate_grad_batches}.')
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
            f'Eval Batch Size {config.loader.eval_global_batch_size}'
            f' not divisible by {num_gpus}.')

    full_dataset = NarryKimDataset()

    # 90/10 train/val split
    n_total = len(full_dataset)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=generator)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=True,
        persistent_workers=True)

    if skip_valid:
        valid_loader = None
        test_loader = None
    else:
        valid_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False)
        test_loader = valid_loader

    return train_loader, valid_loader, test_loader


# Samplers for distributed training
class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):
    def __init__(self, *args, generator=None, **kwargs):
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop('shuffle', None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'random_state': self.generator.get_state(),
                'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get('random_state'))
        self.counter = state_dict['counter']
        self.restarting = True

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)
        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()
        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False
        for index in indices:
            self.counter += 1
            yield index
        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'epoch': self.epoch, 'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.counter = state_dict['counter']
        self.restarting = True

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index
        self.counter = 0


if __name__ == '__main__':
    ds = NarryKimDataset()
    print(f'Dataset size: {len(ds)}')
    print(f'Sequence shape: {ds.seqs.shape}')
    print(f'Token range: {ds.seqs.min().item()} - {ds.seqs.max().item()}')
    sample = ds[0]
    print(f'Sample seq: {dna_detokenize(sample["seqs"].numpy()[:20])}...')
