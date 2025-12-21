

__all__ = ['TimeSeriesLoader', 'BaseTimeSeriesDataset', 'TimeSeriesDataset', 'LocalFilesTimeSeriesDataset',
           'TimeSeriesDataModule']


from collections.abc import Mapping
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import utilsforecast.processing as ufp
from torch.utils.data import DataLoader, Dataset
from utilsforecast.compat import DataFrame, pl_Series


class TimeSeriesLoader(DataLoader):
    """TimeSeriesLoader DataLoader.

    Small change to PyTorch's Data loader.
    Combines a dataset and a sampler, and provides an iterable over the given dataset.

    The class `~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    Args:
        dataset: Dataset to load data from.
        batch_size (int, optional): How many samples per batch to load. Defaults to 1.
        shuffle (bool, optional): Set to True to have the data reshuffled at every epoch. Defaults to False.
        sampler (Sampler or Iterable, optional): Defines the strategy to draw samples from the dataset. 
        Can be any Iterable with __len__ implemented. If specified, shuffle must not be specified. Defaults to None.
        drop_last (bool, optional): Set to True to drop the last incomplete batch. Defaults to False.
        **kwargs: Additional keyword arguments for DataLoader.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            kwargs.pop("collate_fn")
        kwargs_ = {**kwargs, **dict(collate_fn=self._collate_fn)}
        DataLoader.__init__(self, dataset=dataset, **kwargs_)

    def _collate_fn(self, batch):
        elem = batch[0]
        elem_type = type(elem)

        if isinstance(elem, torch.Tensor):
            if len(batch) == 1:
                return elem.unsqueeze(0)
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel, device=elem.device)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            return torch.stack(batch, 0, out=out)

        elif isinstance(elem, Mapping):
            if elem["static"] is None:
                return dict(
                    temporal=self.collate_fn([d["temporal"] for d in batch]),
                    temporal_cols=elem["temporal_cols"],
                    y_idx=elem["y_idx"],
                )

            return dict(
                static=self.collate_fn([d["static"] for d in batch]),
                static_cols=elem["static_cols"],
                temporal=self.collate_fn([d["temporal"] for d in batch]),
                temporal_cols=elem["temporal_cols"],
                y_idx=elem["y_idx"],
            )

        raise TypeError(f"Unknown {elem_type}")


class BaseTimeSeriesDataset(Dataset):
    """Base class for time series datasets.

    Args:
        temporal_cols: Column names for temporal features.
        max_size (int): Maximum size of time series.
        min_size (int): Minimum size of time series.
        y_idx (int): Index of target variable.
        static (Optional): Static features array.
        static_cols (Optional): Column names for static features.
    """

    def __init__(
        self,
        temporal_cols,
        max_size: int,
        min_size: int,
        y_idx: int,
        static=None,
        static_cols=None,
    ):
        super().__init__()
        self.temporal_cols = pd.Index(list(temporal_cols))

        if static is not None:
            self.static = self._as_torch_copy(static)
            self.static_cols = static_cols
        else:
            self.static = static
            self.static_cols = static_cols

        self.max_size = max_size
        self.min_size = min_size
        self.y_idx = y_idx

        # Upadated flag. To protect consistency, dataset can only be updated once
        self.updated = False

    def __len__(self):
        return self.n_groups

    def _as_torch_copy(
        self,
        x: Union[np.ndarray, torch.Tensor],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(dtype, copy=False).clone()

    @staticmethod
    def _ensure_available_mask(data: np.ndarray, temporal_cols):
        if "available_mask" not in temporal_cols:
            available_mask = np.ones((len(data), 1), dtype=np.float32)
            temporal_cols = temporal_cols.append(pd.Index(["available_mask"]))
            data = np.append(data, available_mask, axis=1)
        return data, temporal_cols

    @staticmethod
    def _extract_static_features(static_df, id_col):
        if static_df is not None:
            static_df = ufp.sort(static_df, by=id_col)
            static_cols = [col for col in static_df.columns if col != id_col]
            static = ufp.to_numpy(static_df[static_cols])
            static_cols = pd.Index(static_cols)
        else:
            static = None
            static_cols = None
        return static, static_cols


class TimeSeriesDataset(BaseTimeSeriesDataset):
    """Time series dataset implementation.

    Args:
        temporal: Temporal data array.
        temporal_cols: Column names for temporal features.
        indptr: Index pointers for time series grouping.
        y_idx (int): Index of target variable.
        static (Optional): Static features array.
        static_cols (Optional): Column names for static features.
    """

    def __init__(
        self,
        temporal,
        temporal_cols,
        indptr,
        y_idx: int,
        static=None,
        static_cols=None,
    ):
        self.temporal = self._as_torch_copy(temporal)
        self.indptr = indptr
        self.n_groups = self.indptr.size - 1
        sizes = np.diff(indptr)
        super().__init__(
            temporal_cols=temporal_cols,
            max_size=sizes.max().item(),
            min_size=sizes.min().item(),
            y_idx=y_idx,
            static=static,
            static_cols=static_cols,
        )

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # Parse temporal data and pad its left

            temporal_size = (len(self.temporal_cols), self.max_size)
            ts = self.temporal[self.indptr[idx] : self.indptr[idx + 1], :]
            if temporal_size == (ts.shape[1], ts.shape[0]):
                temporal = ts.permute(1, 0)
            else:
                temporal = torch.zeros(
                    size=(len(self.temporal_cols), self.max_size), dtype=torch.float32
                )
                temporal[: len(self.temporal_cols), -len(ts) :] = ts.permute(1, 0)

            # Add static data if available
            static = None if self.static is None else self.static[idx, :]

            item = dict(
                temporal=temporal,
                temporal_cols=self.temporal_cols,
                static=static,
                static_cols=self.static_cols,
                y_idx=self.y_idx,
            )

            return item
        raise ValueError(f"idx must be int, got {type(idx)}")

    def __repr__(self):
        return f"TimeSeriesDataset(n_data={self.temporal.shape[0]:,}, n_groups={self.n_groups:,})"

    def __eq__(self, other):
        if not hasattr(other, "data") or not hasattr(other, "indptr"):
            return False
        return np.allclose(self.data, other.data) and np.array_equal(
            self.indptr, other.indptr
        )

    def align(
        self, df: DataFrame, id_col: str, time_col: str, target_col: str
    ) -> "TimeSeriesDataset":
        # Protect consistency
        df = ufp.copy_if_pandas(df, deep=False)

        # Add Nones to missing columns (without available_mask)
        temporal_cols = self.temporal_cols.copy()
        for col in temporal_cols:
            if col not in df.columns:
                df = ufp.assign_columns(df, col, np.nan)
            if col == "available_mask":
                df = ufp.assign_columns(df, col, 1.0)

        # Sort columns to match self.temporal_cols (without available_mask)
        df = df[[id_col, time_col] + temporal_cols.tolist()]

        # Process future_df
        dataset, *_ = TimeSeriesDataset.from_df(
            df=df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        return dataset

    def append(self, futr_dataset: "TimeSeriesDataset") -> "TimeSeriesDataset":
        """Add future observations to the dataset.

        Args:
            futr_dataset (TimeSeriesDataset): Future dataset to append.

        Returns:
            TimeSeriesDataset: Copy of dataset with future observations appended.

        Raises:
            ValueError: If datasets have different number of groups.
        """
        if self.indptr.size != futr_dataset.indptr.size:
            raise ValueError(
                "Cannot append `futr_dataset` with different number of groups."
            )
        # Define and fill new temporal with updated information
        len_temporal, col_temporal = self.temporal.shape
        len_futr = futr_dataset.temporal.shape[0]
        new_temporal = torch.empty(size=(len_temporal + len_futr, col_temporal))
        new_indptr = self.indptr + futr_dataset.indptr

        for i in range(self.n_groups):
            curr_slice = slice(self.indptr[i], self.indptr[i + 1])
            curr_size = curr_slice.stop - curr_slice.start
            futr_slice = slice(futr_dataset.indptr[i], futr_dataset.indptr[i + 1])
            new_temporal[new_indptr[i] : new_indptr[i] + curr_size] = self.temporal[
                curr_slice
            ]
            new_temporal[new_indptr[i] + curr_size : new_indptr[i + 1]] = (
                futr_dataset.temporal[futr_slice]
            )

        # Define new dataset
        return TimeSeriesDataset(
            temporal=new_temporal,
            temporal_cols=self.temporal_cols.copy(),
            indptr=new_indptr,
            static=self.static,
            y_idx=self.y_idx,
            static_cols=self.static_cols,
        )

    @staticmethod
    def update_dataset(
        dataset, futr_df, id_col="unique_id", time_col="ds", target_col="y"
    ):
        futr_dataset = dataset.align(
            futr_df, id_col=id_col, time_col=time_col, target_col=target_col
        )
        return dataset.append(futr_dataset)

    @staticmethod
    def trim_dataset(dataset, left_trim: int = 0, right_trim: int = 0):
        """Trim temporal information from a dataset.

        Returns temporal indexes [t+left:t-right] for all series.

        Args:
            dataset: Dataset to trim.
            left_trim (int, optional): Number of observations to trim from the left. Defaults to 0.
            right_trim (int, optional): Number of observations to trim from the right. Defaults to 0.

        Returns:
            TimeSeriesDataset: Trimmed dataset.

        Raises:
            Exception: If trim size exceeds minimum series length.
        """
        if dataset.min_size <= left_trim + right_trim:
            raise Exception(
                f"left_trim + right_trim ({left_trim} + {right_trim}) \
                                must be lower than the shorter time series ({dataset.min_size})"
            )

        # Define and fill new temporal with trimmed information
        len_temporal, col_temporal = dataset.temporal.shape
        total_trim = (left_trim + right_trim) * dataset.n_groups
        new_temporal = torch.zeros(size=(len_temporal - total_trim, col_temporal))
        new_indptr = [0]

        acum = 0
        for i in range(dataset.n_groups):
            series_length = dataset.indptr[i + 1] - dataset.indptr[i]
            new_length = series_length - left_trim - right_trim
            new_temporal[acum : (acum + new_length), :] = dataset.temporal[
                dataset.indptr[i] + left_trim : dataset.indptr[i + 1] - right_trim, :
            ]
            acum += new_length
            new_indptr.append(acum)

        # Define new dataset
        return TimeSeriesDataset(
            temporal=new_temporal,
            temporal_cols=dataset.temporal_cols.copy(),
            indptr=np.array(new_indptr, dtype=np.int32),
            y_idx=dataset.y_idx,
            static=dataset.static,
            static_cols=dataset.static_cols,
        )

    @staticmethod
    def from_df(df, static_df=None, id_col="unique_id", time_col="ds", target_col="y"):
        # TODO: protect on equality of static_df + df indexes
        # Define indices if not given and then extract static features
        static, static_cols = TimeSeriesDataset._extract_static_features(
            static_df, id_col
        )

        ids, times, data, indptr, sort_idxs = ufp.process_df(
            df, id_col, time_col, target_col
        )
        # processor sets y as the first column
        temporal_cols = pd.Index(
            [target_col]
            + [c for c in df.columns if c not in (id_col, time_col, target_col)]
        )
        temporal = data.astype(np.float32, copy=False)
        indices = ids
        if isinstance(df, pd.DataFrame):
            dates = pd.Index(times, name=time_col)
        else:
            dates = pl_Series(time_col, times)

        # Add Available mask efficiently (without adding column to df)
        temporal, temporal_cols = TimeSeriesDataset._ensure_available_mask(
            data, temporal_cols
        )

        dataset = TimeSeriesDataset(
            temporal=temporal,
            temporal_cols=temporal_cols,
            static=static,
            static_cols=static_cols,
            indptr=indptr,
            y_idx=0,
        )
        ds = df[time_col].to_numpy()
        if sort_idxs is not None:
            ds = ds[sort_idxs]
        return dataset, indices, dates, ds
