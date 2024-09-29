import json
from collections import defaultdict
from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class SkipGramDataset(Dataset):
    """
    This class represents a dataset for training a SkipGram model.
    """

    def __init__(
        self,
        sequences: List[int],
        interacted=defaultdict(set),
        item_freq=defaultdict(int),
        window_size=2,
        negative_samples=5,
        id_to_idx=None,
    ):
        """
        Args:
            sequences (list of list of int): The sequences of item indices.
            interacted_dict (defaultdict(set)): A dictionary that keeps track of the other items that shared the same basket with the target item. Those items are ignored when negative sampling.
            item_freq (defaultdict(int)): A dictionary that keeps track the item frequency. It's used to
            window_size (int): The context window size.
            negative_samples (int): Number of negative samples for each positive pair.

        The reason that interacted_dict and item_freq can be passed into the initialization is that at val dataset creation we want to do negative sampling based on the data from the train set as well.
        """
        self.sequences = sequences
        self.window_size = window_size
        self.negative_samples = negative_samples

        # Convert the input IDs into sequence integer for easier processing
        if id_to_idx is None:
            self.id_to_idx = dict()
            self.idx_to_id = dict()
        else:
            self.id_to_idx = id_to_idx
            self.idx_to_id = {v: k for k, v in id_to_idx.items()}

        # Next two lines are used to map the index of __getitem__ to the sequence
        self.idx_to_seq = []
        self.idx_to_seq_idx = []

        self.interacted = deepcopy(interacted)
        self.item_freq = deepcopy(item_freq)

        # Keep tracked of which item-pair co-occur in one basket
        # When doing negative sampling we do not consider the other items that the target item has shared basket
        logger.info("Processing sequences to build interaction data...")
        for seq_idx, seq in tqdm(
            enumerate(sequences),
            desc="Building interactions",
            total=len(sequences),
            leave=True,
        ):
            # Add the sequence to the index to sequence mapping
            self.idx_to_seq.extend([seq_idx] * len(seq))
            self.idx_to_seq_idx.extend(np.arange(len(seq)))

            for item in seq:
                idx = self.id_to_idx.get(item)
                if idx is None:
                    idx = len(self.id_to_idx)
                    self.id_to_idx[item] = idx
                    self.idx_to_id[idx] = item

            seq_idx_set = set([self.id_to_idx[id_] for id_ in seq])
            for idx in seq_idx_set:
                # An item can be considered that it has interacted with itself
                # This helps with negative sampling later
                self.interacted[idx].update(seq_idx_set)
                self.item_freq[idx] += 1

        # Total number of unique items
        self.vocab_size = len(self.item_freq)

        # Create a list of items and corresponding probabilities for sampling
        items, frequencies = zip(*self.item_freq.items())
        self.item_freq_array = np.zeros(self.vocab_size)
        self.item_freq_array[np.array(items)] = frequencies

        self.items = np.arange(self.vocab_size)

        # Use a smoothed frequency distribution for negative sampling
        # The smoothing factor (0.75) can be tuned
        self.sampling_probs = self.item_freq_array**0.75
        self.sampling_probs /= self.sampling_probs.sum()

    def __len__(self):
        # Get index based on the item in the sequence instead of the sequence
        # This is to make tensors in one batch having the same shape
        # and to make batch_size works with Dataloader wrapper
        return len(self.idx_to_seq)

    def __getitem__(self, idx):
        sequence_idx = self.idx_to_seq[idx]
        sequence = self.sequences[sequence_idx]
        sequence = [self.id_to_idx[item] for item in sequence]
        i = self.idx_to_seq_idx[idx]
        target_item = sequence[i]

        positive_pairs = []
        labels = []

        start = max(i - self.window_size, 0)
        end = min(i + self.window_size + 1, len(sequence))

        for j in range(start, end):
            if i != j:
                context_item = sequence[j]
                positive_pairs.append((target_item, context_item))
                labels.append(1)  # Positive label

        # Generate negative samples based on item frequency
        negative_pairs = []

        for target_item, _ in positive_pairs:
            # Mask out the items that the target item has interacted with
            # Then sample the remaining items based on the item frequency as negative items
            negative_sampling_probs = deepcopy(self.sampling_probs)
            negative_sampling_probs[list(self.interacted[target_item])] = 0
            negative_sampling_probs /= negative_sampling_probs.sum()

            negative_items = np.random.choice(
                self.items,
                size=self.negative_samples,
                p=negative_sampling_probs,
                replace=False,
            )

            for negative_item in negative_items:
                negative_pairs.append((target_item, negative_item))
                labels.append(0)

        # Combine positive and negative pairs
        pairs = positive_pairs + negative_pairs

        # Convert to tensor
        target_items = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
        context_items = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            "target_items": target_items,
            "context_items": context_items,
            "labels": labels,
        }

    def collate_fn(self, batch):
        target_items = []
        context_items = []
        labels = []
        for record in batch:
            target_items.append(record["target_items"])
            context_items.append(record["context_items"])
            labels.append(record["labels"])
        return {
            "target_items": torch.cat(target_items, dim=0),
            "context_items": torch.cat(context_items, dim=0),
            "labels": torch.cat(labels, dim=0),
        }

    def save_id_mappings(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(
                {
                    "id_to_idx": self.id_to_idx,
                    "idx_to_id": self.idx_to_id,
                },
                f,
            )

    @classmethod
    def get_default_loss_fn(cls):
        loss_fn = nn.BCELoss()
        return loss_fn

    @classmethod
    def forward(cls, model, batch_input, loss_fn=None, device="cpu"):
        predictions = model.predict_train_batch(batch_input, device=device).squeeze()
        labels = batch_input["labels"].float().to(device).squeeze()

        if loss_fn is None:
            loss_fn = cls.get_default_loss_fn()

        loss = loss_fn(predictions, labels)
        return loss
