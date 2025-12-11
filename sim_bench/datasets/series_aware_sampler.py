"""
Series-aware batch sampler for PhotoTriage dataset.

Ensures that each batch contains pairs from different series, preventing the model
from seeing multiple pairs from the same series in a single batch. This helps
prevent overfitting to series-specific patterns.
"""

from typing import Iterator, List
import numpy as np
from torch.utils.data import Sampler


class SeriesAwareBatchSampler(Sampler):
    """
    Batch sampler that ensures each batch contains pairs from different series.

    This prevents the model from seeing multiple pairs from the same series
    in a single batch, which helps prevent overfitting to series-specific patterns.

    Algorithm:
    1. Group pair indices by series_id
    2. For each batch, sample one pair from each of N different series
    3. Shuffle series order between epochs

    Example:
        Series 1: [pair_0, pair_1, pair_2]  # 3 pairs
        Series 2: [pair_3, pair_4]          # 2 pairs
        Series 3: [pair_5]                  # 1 pair

        Batch 1: [pair_0, pair_3, pair_5]  # One from each series
        Batch 2: [pair_1, pair_4]           # Series 3 exhausted
        Batch 3: [pair_2]                   # Series 2 exhausted
    """

    def __init__(self, series_ids: List[int], batch_size: int, drop_last: bool = False, shuffle: bool = True):
        """
        Args:
            series_ids: List of series_id for each sample (same length as dataset)
            batch_size: Target batch size
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle series order between epochs
        """
        self.series_ids = np.array(series_ids)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Group indices by series
        self.series_to_indices = {}
        for idx, series_id in enumerate(series_ids):
            if series_id not in self.series_to_indices:
                self.series_to_indices[series_id] = []
            self.series_to_indices[series_id].append(idx)

        # Convert to lists for easier manipulation
        self.unique_series = list(self.series_to_indices.keys())
        self.num_series = len(self.unique_series)

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of indices."""
        # Create a copy of indices for each series (so we can pop from them)
        series_indices = {
            series_id: indices.copy()
            for series_id, indices in self.series_to_indices.items()
        }

        # Shuffle indices within each series if shuffle is enabled
        if self.shuffle:
            for indices in series_indices.values():
                np.random.shuffle(indices)

        # Get list of series (shuffle order if enabled)
        series_order = self.unique_series.copy()
        if self.shuffle:
            np.random.shuffle(series_order)

        # Track which series still have samples
        active_series = series_order.copy()

        # Generate batches
        while len(active_series) > 0:
            batch = []

            # Try to fill batch with one sample from each series
            series_to_remove = []
            for series_id in active_series[:self.batch_size]:
                if len(series_indices[series_id]) > 0:
                    # Pop one sample from this series
                    idx = series_indices[series_id].pop(0)
                    batch.append(idx)

                    # Mark series for removal if exhausted
                    if len(series_indices[series_id]) == 0:
                        series_to_remove.append(series_id)

            # Remove exhausted series
            for series_id in series_to_remove:
                active_series.remove(series_id)

            # Yield batch if it meets size requirements
            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and len(batch) > 0:
                yield batch

    def __len__(self) -> int:
        """Estimate number of batches."""
        # Calculate total number of pairs
        total_pairs = sum(len(indices) for indices in self.series_to_indices.values())

        if self.drop_last:
            return total_pairs // self.batch_size
        else:
            return (total_pairs + self.batch_size - 1) // self.batch_size


class BalancedSeriesBatchSampler(Sampler):
    """
    Advanced sampler that ensures:
    1. Each batch contains pairs from different series (one pair per series)
    2. Batch composition is shuffled between epochs (different series in each batch)

    This prevents overfitting to series-specific patterns by forcing diverse
    examples in each batch AND preventing the model from seeing the same
    series combinations repeatedly.

    Algorithm per epoch:
    1. Shuffle series order (different each epoch)
    2. Shuffle pairs within each series
    3. Round-robin through series, taking one pair from each
    4. Create batches sequentially from round-robin order
    5. Result: Different series appear in different batches each epoch
    """

    def __init__(self, series_ids: List[int], batch_size: int, drop_last: bool = False, shuffle: bool = True):
        """
        Args:
            series_ids: List of series_id for each sample (parallel to dataset)
            batch_size: Target batch size
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle (both series order AND pairs within series)
        """
        self.series_ids = np.array(series_ids)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Group indices by series
        self.series_to_indices = {}
        for idx, series_id in enumerate(series_ids):
            if series_id not in self.series_to_indices:
                self.series_to_indices[series_id] = []
            self.series_to_indices[series_id].append(idx)

        self.unique_series = list(self.series_to_indices.keys())

        print(f"BalancedSeriesBatchSampler initialized:")
        print(f"  Total pairs: {len(series_ids)}")
        print(f"  Unique series: {len(self.unique_series)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Shuffle enabled: {shuffle}")

    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate balanced batches with shuffling between epochs.

        Key difference from previous implementation:
        - Series order is shuffled EACH EPOCH (line below)
        - This ensures different series appear in different batches each epoch
        """
        # CRITICAL: Shuffle series order each epoch
        series_order = self.unique_series.copy()
        if self.shuffle:
            np.random.shuffle(series_order)  # Different order each epoch!

        # Create iterators for each series with shuffled pairs
        series_iterators = {}
        for series_id, indices in self.series_to_indices.items():
            series_indices = indices.copy()
            if self.shuffle:
                np.random.shuffle(series_indices)  # Different pair order each epoch!
            series_iterators[series_id] = iter(series_indices)

        all_indices = []

        # Round-robin through series in THIS EPOCH's order
        while series_iterators:
            for series_id in series_order.copy():
                if series_id not in series_iterators:
                    continue

                try:
                    idx = next(series_iterators[series_id])
                    all_indices.append(idx)
                except StopIteration:
                    # This series is exhausted
                    del series_iterators[series_id]
                    series_order.remove(series_id)

        # Create batches from collected indices
        # Because series_order was shuffled, batches contain different series each epoch
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and len(batch) > 0:
                yield batch

    def __len__(self) -> int:
        """Estimate number of batches."""
        total_pairs = sum(len(indices) for indices in self.series_to_indices.values())

        if self.drop_last:
            return total_pairs // self.batch_size
        else:
            return (total_pairs + self.batch_size - 1) // self.batch_size
