"""Data abstractions."""
import copy
import time
from typing import List

import numpy as np
import pandas as pd

import torch
from torch.utils import data

import made
import util
from estimators import torch_OPS
from util import get_device, ConvertOPSBinToSQLQuery
from multiprocessing.pool import ThreadPool
import mysampler

PREDICATE_OPS = [
    '=',
    '>',
    '<',
    '>=',
    "<="
]
range_fn = [None, torch.less, torch.greater, torch.less_equal, torch.greater_equal]


# Na/NaN/NaT Semantics
#
# Some input columns may naturally contain missing values.  These are handled
# by the corresponding numpy/pandas semantics.
#
# Specifically, for any value (e.g.,.to(dtype=torch.float32, copy=False), int, or np.nan) v:
#
#   np.nan <op> v == False.
#
# This means that in progressive sampling, if a column's domain contains np.nan
# (at the first position in the domain), it will never be a valid sample
# target.
#
# The above evaluation is consistent with SQL semantics.


class Column(object):
    """A column.  Data is write-once, immutable-after.

    Typical usage:
      col = Column('Attr1').Fill(data, infer_dist=True)

    The passed-in 'data' is copied by reference.
    """

    def __init__(self, name, distribution_size=None, pg_name=None):
        self.name = name

        # Data related fields.
        self.data = None
        self.all_distinct_values = None
        self.all_distinct_values_gpu = None
        self.all_discretize_distinct_values = None
        self.all_discretize_distinct_values_gpu = None
        self.distribution_size = distribution_size
        self.has_none = False
        self.nan_ind = None
        # pg_name is the name of the corresponding column in Postgres.  This is
        # put here since, e.g., PG disallows whitespaces in names.
        self.pg_name = pg_name if pg_name else name

    def Name(self):
        """Name of this column."""
        return self.name

    def DistributionSize(self):
        """This column will take on discrete values in [0, N).

        Used to dictionary-encode values to this discretized range.
        """
        return self.distribution_size

    def ValToBin(self, val):
        if isinstance(self.all_distinct_values, list):
            return self.all_distinct_values.index(val)
        inds = np.where(self.all_distinct_values == val)
        assert len(inds[0]) > 0, val

        return inds[0][0]

    def SetDistribution(self, distinct_values):
        """This is all the values this column will ever see."""
        assert self.all_distinct_values is None
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        is_nan = pd.isnull(distinct_values)
        self.nan_ind = is_nan
        contains_nan = np.any(is_nan)
        dv_no_nan = distinct_values[~is_nan]
        # NOTE: np.sort puts NaT values at beginning, and NaN values at end.
        # For our purposes we always add any null value to the beginning.
        vs = np.sort(np.unique(dv_no_nan))
        if contains_nan and np.issubdtype(distinct_values.dtype, np.datetime64):
            vs = np.insert(vs, 0, np.datetime64('NaT'))
        elif contains_nan:
            vs = np.insert(vs, 0, np.nan)
        if self.distribution_size is not None:
            assert len(vs) == self.distribution_size
        self.all_distinct_values = vs
        if distinct_values.dtype != np.object_ and distinct_values.dtype != np.datetime64 and not np.issubdtype(
                distinct_values.dtype, np.datetime64):
            self.all_distinct_values_gpu = torch.as_tensor(vs, device=get_device())
        else:
            self.all_distinct_values_gpu = vs
        self.distribution_size = len(vs)
        # [前提]：pd.Categorical保序且Discritze时dvs有序，nan在最前
        self.all_discretize_distinct_values = np.arange(self.distribution_size)
        self.all_discretize_distinct_values_gpu = torch.as_tensor(self.all_discretize_distinct_values,
                                                                  device=get_device())
        return self

    def Fill(self, data_instance, infer_dist=False):
        assert self.data is None
        self.data = data_instance
        # If no distribution is currently specified, then infer distinct values
        # from data.
        if infer_dist:
            self.SetDistribution(self.data)
        return self

    def __repr__(self):
        return 'Column({}, distribution_size={})'.format(
            self.name, self.distribution_size)


class Table(object):
    """A collection of Columns."""

    def __init__(self, name, columns, pg_name=None):
        """Creates a Table.

        Args:
            name: Name of this table object.
            columns: List of Column instances to populate this table.
            pg_name: name of the corresponding table in Postgres.
        """
        self.name = name
        self.cardinality = self._validate_cardinality(columns)
        self.columns = columns

        self.val_to_bin_funcs = [c.ValToBin for c in columns]
        self.name_to_index = {c.Name(): i for i, c in enumerate(self.columns)}
        self.columns_size = np.array([col.distribution_size for col in columns])
        if pg_name:
            self.pg_name = pg_name
        else:
            self.pg_name = name

    def __repr__(self):
        return '{}({})'.format(self.name, self.columns)

    def _validate_cardinality(self, columns):
        """Checks that all the columns have same the number of rows."""
        cards = [len(c.data) for c in columns]
        c = np.unique(cards)
        assert len(c) == 1, c
        return c[0]

    def Name(self):
        """Name of this table."""
        return self.name

    def Columns(self):
        """Return the list of Columns under this table."""
        return self.columns

    def ColumnIndex(self, name):
        """Returns index of column with the specified name."""
        assert name in self.name_to_index
        return self.name_to_index[name]


class CsvTable(Table):
    """Wraps a CSV file or pd.DataFrame as a Table."""

    def __init__(self,
                 name,
                 filename_or_df,
                 cols,
                 type_casts={},
                 pg_name=None,
                 pg_cols=None,
                 **kwargs):
        """Accepts the same arguments as pd.read_csv().

        Args:
            filename_or_df: pass in str to reload; otherwise accepts a loaded
              pd.Dataframe.
            cols: list of column names to load; can be a subset of all columns.
            type_casts: optional, dict mapping column name to the desired numpy
              datatype.
            pg_name: optional str, a convenient field for specifying what name
              this table holds in a Postgres database.
            pg_name: optional list of str, a convenient field for specifying
              what names this table's columns hold in a Postgres database.
            **kwargs: keyword arguments that will be pass to pd.read_csv().
        """
        self.name = name
        self.pg_name = pg_name

        if isinstance(filename_or_df, str):
            self.data = self._load(filename_or_df, cols, **kwargs, low_memory=False)
        else:
            assert (isinstance(filename_or_df, pd.DataFrame))
            self.data = filename_or_df

        self.columns = self._build_columns(self.data, cols, type_casts, pg_cols)
        self.columns_dict = {}
        for c in self.columns:
            self.columns_dict[c.name] = c
        super(CsvTable, self).__init__(name, self.columns, pg_name)

    def _load(self, filename, cols, **kwargs):
        print('Loading csv...', end=' ')
        s = time.time()
        df = pd.read_csv(filename, usecols=cols, **kwargs)
        if cols is not None:
            df = df[cols]
        print('done, took {:.1f}s'.format(time.time() - s))
        return df

    def _build_columns(self, data, cols, type_casts, pg_cols):
        """Example args:

            cols = ['Model Year', 'Reg Valid Date', 'Reg Expiration Date']
            type_casts = {'Model Year': int}

        Returns: a list of Columns.
        """
        print('Parsing...', end=' ')
        s = time.time()
        for col, typ in type_casts.items():
            if col not in data:
                continue
            if typ != np.datetime64:
                data[col] = data[col].astype(typ, copy=False)
            else:
                # Both infer_datetime_format and cache are critical for perf.
                data[col] = pd.to_datetime(data[col],
                                           infer_datetime_format=True,
                                           cache=True)

        # Discretize & create Columns.
        if cols is None:
            cols = data.columns
        columns = []
        if pg_cols is None:
            pg_cols = [None] * len(cols)
        for c, p in zip(cols, pg_cols):
            col = Column(c, pg_name=p)
            col.Fill(data[c])

            # dropna=False so that if NA/NaN is present in data,
            # all_distinct_values will capture it.
            #
            # For numeric: np.nan
            # For datetime: np.datetime64('NaT')
            col.SetDistribution(data[c].value_counts(dropna=False).index.values)
            col.has_none = pd.isnull(col.all_distinct_values[0])
            columns.append(col)
        print('done, took {:.1f}s'.format(time.time() - s))
        return columns


class TableDataset(data.Dataset):
    """Wraps a Table and yields each row as a PyTorch Dataset element."""

    def __init__(self, table: CsvTable, bs=1, expand_factor=4, queries=None, true_cards=None, inp=None,
                 valid_i_list=None, wild_card_mask=None, model=None):
        super(TableDataset, self).__init__()
        self.table = copy.deepcopy(table)
        self.expand_factor = expand_factor
        self.bs = bs
        self.num_queries = len(true_cards) if true_cards is not None else 1
        if queries is not None:
            assert len(queries[0]) == len(true_cards)
        self.queries = [np.array(queries[1]), np.array(queries[2])] if queries else None
        self.wild_card_mask = wild_card_mask
        self.true_cards = np.array(true_cards)
        self.model = model
        self.inp = inp
        self.valid_i_list = valid_i_list
        print('Discretizing table...', end=' ')
        s = time.time()
        # [cardianlity, num cols].
        self.tuples_np = np.stack(
            [self.Discretize(c) for c in self.table.Columns()], axis=1)
        self.tuples = torch.as_tensor(self.tuples_np).long().pin_memory()

        self.device = util.get_device()
        self.new_tuples = torch.tile(
            torch.zeros((self.bs * self.expand_factor, self.tuples.shape[-1]), requires_grad=False,
                        dtype=torch.long).unsqueeze(1), dims=(1, 2, 1)).pin_memory() - 1
        self.preds = torch.zeros((self.bs * self.expand_factor, 2, 5 * len(self.table.columns)), requires_grad=False,
                                 dtype=torch.long).pin_memory()
        self.has_nones = [c.has_none for c in self.table.columns]
        self.columns_size = [c.distribution_size for c in self.table.columns]
        print('done, took {:.1f}s'.format(time.time() - s))

    def Discretize(self, col):
        """Discretize values into its Column's bins.

        Args:
          col: the Column.
        Returns:
          col_data: discretized version; an np.ndarray of type np.int32.
        """
        return Discretize(col)

    def size(self):
        return len(self.tuples)

    def __len__(self):
        if self.queries is not None:
            return max(len(self.tuples), len(self.queries))
        else:
            return len(self.tuples)

    def __getitem__(self, idx):
        return idx

    def collect_fn(self, samples):
        with torch.no_grad():
            num_samples = len(samples)
            tuples_idxs = torch.as_tensor(samples, dtype=torch.long)
            if self.queries:
                query_start = np.random.randint(len(self.queries[0]))
                query_end = query_start + num_samples
                if query_end > len(self.queries[0]):
                    query_end = len(self.queries[0])
                queries = [self.queries[0][query_start:query_end], self.queries[1][query_start:query_end]]
                true_cards = self.true_cards[query_start:query_end]
                wild_card_mask = self.wild_card_mask[query_start:query_end].to(self.device, non_blocking=True)
                inps = self.inp[query_start:query_end].to(self.device, non_blocking=True)
                # 更新inp中的wild_card
                for i in range(self.model.nin):
                    if i == 0:
                        s = 0
                    else:
                        s = self.model.input_bins_encoded_cumsum[i - 1]
                    e = self.model.input_bins_encoded_cumsum[i]
                    # torch与numpybutong，mask与slice合用时mask只能为一维向量
                    if isinstance(self.model, made.MADE):
                        pred_shift = 5
                    else:
                        pred_shift = 0
                    inps[wild_card_mask[:,0,i], 0,s:e-pred_shift] = 0
                    inps[wild_card_mask[:, 1, i], 1, s:e - pred_shift] = 0
                    inps[..., s:e-pred_shift] = inps[..., s:e-pred_shift] + wild_card_mask.narrow(-1, i, 1).float() * self.model.unk_embeddings[i]
                valid_i_lists = [self.valid_i_list[i][query_start:query_end] for i in range(len(self.valid_i_list))]

            tuples = self.tuples[tuples_idxs]
            if self.expand_factor>1:
                tuples = torch.tile(tuples, dims=(self.expand_factor, 1))
            num_samples = tuples.shape[0]
            # -1 by default
            new_tuples = self.new_tuples[:num_samples]
            new_tuples.fill_(-1)
            new_preds = self.preds[:num_samples]
            new_preds.zero_()
            mysampler.sample(tuples, new_tuples, new_preds, self.columns_size, self.has_nones, num_samples)
            tuples = tuples.to(self.device, non_blocking=True)
            new_tuples = new_tuples.to(self.device, non_blocking=True)
            new_preds = new_preds.to(self.device, non_blocking=True)
            # for i, column_size in enumerate(self.columns_size):
            #     if new_tuples[...,i].max().item()>=column_size:
            #         print(torch.where(new_tuples[...,i]>=column_size))
            #         assert False
            # util.check_sample(self.table, tuples, new_tuples, new_preds)
            if self.queries is None:
                return [new_tuples, new_preds, tuples]
            else:
                return [new_tuples, new_preds, tuples, queries, true_cards, inps, valid_i_lists]


class WorkloadDataset(data.Dataset):
    def __init__(self, table_dataset: TableDataset, queries, true_cards, inp, valid_i_list):
        super(WorkloadDataset, self).__init__()
        self.table = table_dataset.table
        self.num_queries = len(true_cards)
        assert len(queries[0]) == len(true_cards)
        self.queries = queries
        self.true_cards = true_cards
        self.inp = inp
        self.valid_i_list = valid_i_list
        self.tuples_np = table_dataset.tuples_np
        self.tuples = table_dataset.tuples

    def size(self):
        return self.num_queries

    def __len__(self):
        return self.num_queries

    def __getitem__(self, idx):
        cols = self.queries[0][idx]
        ops = self.queries[1][idx]
        vals = self.queries[2][idx]
        return (cols, ops, vals), self.true_cards[idx], idx

    def collect_fn(self, samples):
        queries = list(map(lambda x: x[0], samples))
        true_cards = list(map(lambda x: x[1], samples))
        idxs = torch.as_tensor(list(map(lambda x: x[2], samples)), dtype=torch.long, device=get_device())
        inps = self.inp[idxs]
        valid_i_lists = []
        for i in range(len(self.valid_i_list)):
            valid_i_lists.append(self.valid_i_list[i][idxs])
        return [queries, true_cards, inps, valid_i_lists]


def Discretize(col, data=None):
    """Transforms data values into integers using a Column's vocab.

    Args:
        col: the Column.
        data: list-like data to be discretized.  If None, defaults to col.data.

    Returns:
        col_data: discretized version; an np.ndarray of type np.int32.
    """
    # pd.Categorical() does not allow categories be passed in an array
    # containing np.nan.  It makes it a special case to return code -1
    # for NaN values.

    if data is None:
        data = col.data

    # pd.isnull returns true for both np.nan and np.datetime64('NaT').
    isnan = pd.isnull(col.all_distinct_values)
    if isnan.any():
        # We always add nan or nat to the beginning.
        assert isnan.sum() == 1, isnan
        assert isnan[0], isnan

        dvs = col.all_distinct_values[1:]
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data)

        # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
        # add 1 to everybody
        bin_ids = bin_ids + 1
    else:
        # This column has no nan or nat values.
        dvs = col.all_distinct_values
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data), (len(bin_ids), len(data))

    bin_ids = bin_ids.astype(np.int32, copy=False)
    assert (bin_ids >= 0).all(), (col, data, bin_ids)
    return bin_ids
