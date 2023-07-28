"""A suite of cardinality estimators.

In practicular, inference algorithms for autoregressive density estimators can
be found in 'ProgressiveSampling'.
"""
import bisect
import collections
import copy
import itertools
import json
import multiprocessing
import operator
import time

import mysampler
import numpy as np
import pandas as pd
import torch

import common
import made
import transformer
import util
from util import get_device
from multiprocessing.pool import ThreadPool

OPS_array = np.array(['=', '>', '<', '>=', '<='])

OPS_dict = {'=': 0,
            '>': 1,
            '<': 2,
            '>=': 3,
            '<=': 4}
OPS = {
    '=': np.equal,
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
}

torch_OPS = np.array([torch.eq, torch.greater, torch.less, torch.greater_equal, torch.less_equal])

all_op_pairs = list(filter(lambda x: x[0] is not None, itertools.product([None] + list(range(5)), repeat=2)))


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def clear_errs(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))


def QueryToPredicate(columns, operators, vals, wrap_as_string_cols=None):
    """Converts from (c,o,v) to sql string (for Postgres)."""
    v_s = [
        str(v).replace('T', ' ') if type(v) is np.datetime64 else v
        for v in vals
    ]
    v_s = ["\'" + v + "\'" if type(v) is str else str(v) for v in v_s]

    if wrap_as_string_cols is not None:
        for i in range(len(columns)):
            if columns[i].name in wrap_as_string_cols:
                v_s[i] = "'" + str(v_s[i]) + "'"

    preds = [
        c.pg_name + ' ' + o + ' ' + v
        for c, o, v in zip(columns, operators, v_s)
    ]
    s = ' and '.join(preds)
    return ' where ' + s


def FillInUnqueriedColumns(table, columns, operators, vals, replace_to_torch=True):
    """Allows for some columns to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [[None, None]] * ncols, [[None, None]] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.ColumnIndex(c.name)
        os[idx] = o
        vs[idx] = torch.as_tensor(list(map(lambda x: -1 if x is None else x, v))).unsqueeze(
            0).unsqueeze(-1).pin_memory() if replace_to_torch else v
    if replace_to_torch:
        vs = torch.cat(vs, dim=-1)
    return cs, os, vs


class DirectEstimator(CardEst):
    """Progressive sampling."""

    def __init__(
            self,
            model,
            table,
            device=None,
            seed=False,
            cardinality=None,
            requires_grad=False,
            batch_size=1
    ):
        super(DirectEstimator, self).__init__()
        self.encoding_dur_ms = []
        self.infer_dur_ms = []
        if not requires_grad:
            torch.set_grad_enabled(False)
        self.model = model
        self.table = table
        self.shortcircuit = True

        self.seed = seed
        self.device = device

        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.cardinality

        # with torch.no_grad():
        #     self.init_logits = self.model(
        #         [torch.zeros(1, self.model.nin, device=device), torch.zeros(1, self.model.nin * 5, device=device),
        #          torch.zeros(1, self.model.nin, device=device)])

        self.dom_sizes = [c.DistributionSize() for c in self.table.columns]
        self.dom_sizes = np.cumsum(self.dom_sizes)

        # Inference optimizations below.

        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        if 'MADE' in str(model) and not requires_grad:
            model.bin_as_onehot_shifts = [x.cpu().pin_memory() for x in model.bin_as_onehot_shifts]
            for layer in model.net:
                if type(layer) == made.MaskedLinear:
                    if layer.masked_weight is None:
                        layer.masked_weight = layer.mask * layer.weight
                        # print('Setting masked_weight in MADE, do not retrain!')
        if not requires_grad:
            model.unk_embeddings = torch.nn.ParameterList([x.cpu().pin_memory() for x in model.unk_embeddings])
            for p in model.parameters():
                p.detach_()
                p.requires_grad = False
        self.is_made = isinstance(model, made.MADE)
        self.multi_pred_embed_nn = None
        if self.is_made:
            self.model.unk_embedding_pred_cache = [
                torch.cat([emb, torch.zeros(*emb.shape[:-1], 5, device=emb.device)], dim=-1) for emb
                in self.model.unk_embeddings]
            if not requires_grad:
                self.model.unk_embedding_pred_cache = list(
                    map(lambda x: x.cpu().pin_memory(), self.model.unk_embedding_pred_cache))
                if self.model.multi_pred_embedding == 'mlp' and isinstance(self.model.multi_pred_embed_nn, torch.nn.ModuleList):
                    self.multi_pred_embed_nn = self.model.multi_pred_embed_nn
                    self.model.multi_pred_embed_nn = made.seqList_2_EnsembleSeq(self.model.multi_pred_embed_nn)
                    self.model.is_ensembled = True
        else:
            if not requires_grad:
                self.model.embeddings = self.model.embeddings.cpu()
                self.model.unk_embeddings = self.model.unk_embeddings.cpu()
        if self.is_made:
            self.inp = torch.tile(
                torch.cat([self.model.unk_embedding_pred_cache[natural_idx] for natural_idx in self.model.m[-1]],
                          dim=-1).unsqueeze(1), dims=(1, 2, 1))
            # self.inp = torch.zeros((batch_size, 2, np.array(self.model.input_bins_encoded).sum())).pin_memory()
        else:
            self.inp = torch.zeros((batch_size, 2, self.model.d_model * self.model.nin))
        if not requires_grad:
            self.inp = self.inp.pin_memory()
        # self.init_logits.detach_()
        self.preds = [torch.zeros((1, 2, 5)).pin_memory() for _ in range(len(self.table.columns))]
        self.valid_i_list_cache = []
        self.pool = ThreadPool()

    def clear_errs(self):
        super().clear_errs()
        self.encoding_dur_ms = []
        self.infer_dur_ms = []

    def merge_stats(self, state):
        super().merge_stats(state)
        self.encoding_dur_ms.extend(state[5])
        self.infer_dur_ms.extend(state[6])

    def __str__(self):
        return 'psample'

    def _put_samples_as_input(self,
                              data_to_encode,
                              preds,
                              inp,
                              natural_idx,
                              sampling_order_idx=None,
                              batch_ind=None):
        model = self.model
        """Puts [bs, 1] sampled values approipately into inp."""
        if not isinstance(model, transformer.Transformer):
            if natural_idx == 0:
                if batch_ind is None:
                    out = inp[:, :, :model.input_bins_encoded_cumsum[0]]
                else:
                    out = inp[batch_ind, :, :model.input_bins_encoded_cumsum[0]]
            else:
                l = model.input_bins_encoded_cumsum[natural_idx - 1]
                r = model.input_bins_encoded_cumsum[natural_idx]
                if batch_ind is None:
                    out = inp[:, :, l:r]
                else:
                    out = inp[batch_ind, :, l:r]
            if natural_idx == 0:
                model.EncodeInput(
                    data_to_encode,
                    preds,
                    natural_col=0,
                    out=out)
            else:
                model.EncodeInput(data_to_encode,
                                  preds,
                                  natural_col=natural_idx, out=out)
        else:
            # Transformer.  Need special treatment due to
            # right-shift.
            l = (natural_idx + 1) * model.d_model
            r = l + model.d_model
            target = inp[:, :, :model.d_model]
            if sampling_order_idx == 0:
                # Let's also add E_pos=0 to SOS (if enabled).
                # This is a no-op if disabled pos embs.
                model.EncodeInput(
                    data_to_encode,  # Will ignore.
                    preds,
                    natural_col=-1,  # Signals SOS.
                    out=target)
            target = inp[:, :, l:r]
            if transformer.MASK_SCHEME == 1:
                # Should encode natural_col \in [0, ncols).
                model.EncodeInput(data_to_encode,
                                  preds,
                                  natural_col=natural_idx,
                                  out=target)
            elif natural_idx < model.nin - 1:
                # If scheme is 0, should not encode the last
                # variable.
                model.EncodeInput(data_to_encode,
                                  preds,
                                  natural_col=natural_idx,
                                  out=target)

    def ProcessQuery(self,
                     ordering,
                     columns,
                     operators,
                     vals, ):
        ncols = len(columns)
        valid_i_list = []  # None means all valid.
        for i in range(ncols):
            natural_idx = ordering[i]
            # Column i.
            op = operators[natural_idx]
            val = vals[..., natural_idx]
            dvs = columns[natural_idx].all_discretize_distinct_values_gpu

            if op[0] is not None:
                # There exists a filter.
                valid_i = torch_OPS[op[0]](dvs.view(1, -1), val.squeeze()[0].item())
                if op[1] is not None:
                    valid_i = torch.bitwise_and(valid_i, torch_OPS[op[1]](dvs.view(1, -1), val.squeeze()[1].item()))
                if self.table.columns[natural_idx].nan_ind.any():
                    valid_i[...,0] = False
                # This line triggers a host -> gpu copy, showing up as a
                # hotspot in cprofile.
                # (1, col_dvs_size)
                valid_i_list.append(torch.as_tensor(valid_i, device=self.device).T)
            else:
                valid_i_list.append(torch.ones((1, dvs.shape[0]), device=self.device).T)
            # assert not (torch.isnan(valid_i_list[-1]).any() or torch.isinf(valid_i_list[-1]).any())
        return torch.nn.utils.rnn.pad_sequence(valid_i_list, batch_first=False, padding_value=1).transpose(0, 2).long()

    def _sample_n(self,
                  ordering,
                  columns,
                  operators,
                  vals,
                  valid_i_list,
                  inp):
        enc_st = time.time()
        ncols = len(columns)
        # Use the query to filter each column's domain.
        def encode_transformer(preds, op, val, natural_idx, inp, i):
            # Column i.
            val = val.unsqueeze(-1)
            # There exists a filter.
            if op[0] is None:
                self._put_samples_as_input(None, None, inp, natural_idx, sampling_order_idx=i)
            else:
                preds.zero_()
                preds[:, 0, op[0]] = 1
                if op[1] is not None:
                    preds[:, 1, op[1]] = 1
                self._put_samples_as_input(val,
                                           preds,
                                           inp,
                                           natural_idx,
                                           sampling_order_idx=i)

        def encode_made(preds, op, val, natural_idx, inp, i):
            # Column i.
            val = val.unsqueeze(-1)
            # There exists a filter.
            preds.zero_()
            preds[:, 0, op[0]] = 1
            if op[1] is not None:
                preds[:, 1, op[1]] = 1
            self._put_samples_as_input(val,
                                       preds,
                                       inp,
                                       natural_idx,
                                       sampling_order_idx=i)
        is_made = self.is_made
        for i in range(ncols):
            natural_idx = ordering[i]
            op = operators[natural_idx]
            if op[0] is not None and is_made:
                encode_made(self.preds[i], op, vals[..., natural_idx], natural_idx, inp, i)
            elif not is_made:
                encode_transformer(self.preds[i], op, vals[..., natural_idx], natural_idx, inp, i)
        enc_st = time.time() - enc_st
        st = time.time()
        logits = self.model.forward_with_encoded_input(inp.to(self.device), ordering=ordering)
        logits_ = torch.split(logits.T, list(np.diff(np.insert(self.model.output_bins_cumsum, 0, 0))), dim=0)
        logits_ = torch.softmax(
            torch.nn.utils.rnn.pad_sequence(logits_, batch_first=False, padding_value=float('-inf')).transpose(0,
                                                                                                               2),
            dim=-1)
        p_pred = torch.cumprod((logits_ * valid_i_list).sum(-1), dim=-1)[..., -1]
        st2 = time.time()
        self.encoding_dur_ms.append(enc_st * 1e3)
        self.infer_dur_ms.append((st2 - st) * 1e3)

        return p_pred.item()

    def encode_queries(self, queries):
        with torch.no_grad():
            ncols = len(queries[0][0])
            if isinstance(self.model, made.MADE):
                inp = torch.zeros((len(queries[0]), 2, np.array(self.model.input_bins_encoded).sum()),
                                  device=self.device,
                                  requires_grad=False)
            else:
                inp = torch.zeros((len(queries[0]), 2, self.model.d_model * self.model.nin), device=self.device,
                                  requires_grad=False)
            ordering = self.get_ordering(queries)
            # Use the query to filter each column's domain.
            valid_i_list = [None] * ncols  # None means all valid.

            all_columns = queries[0]
            all_operators = queries[1]
            all_vals = queries[2]
            wild_card_mask = np.zeros((inp.shape[0], 2, self.model.nin))
            for i in range(ncols):
                natural_idx = ordering[i]
                # shape:[num_queries,preds_num], dtypes=Columns
                columns = np.array(list(map(lambda x: x[natural_idx], all_columns)))
                # shape:[num_queries,preds_num], dtypes=int
                operators = np.array(list(map(lambda x: x[natural_idx], all_operators)))
                # shape:[num_queries,preds_num], dtypes=int
                vals = np.array(list(map(lambda x: x[natural_idx], all_vals)))
                # 对于两个谓词全是None的列，使用wild cards和=谓词填充
                none_ind = operators == [None, None]
                none_ind = np.bitwise_and(none_ind[:, 0], none_ind[:, 1])
                if none_ind.any():
                    # 对于所有没有谓词的部分，设置为需要替换wildcard
                    # wild_card_mask[none_ind, :, s:e-5] = 1
                    self._put_samples_as_input(None, None, inp, natural_idx, sampling_order_idx=i, batch_ind=none_ind)
                # 所有columns都一样
                dvs = columns[0].all_discretize_distinct_values_gpu
                preds = torch.zeros((len(queries[0]), 2, 5), device=self.device)
                # 对于none的就乘以1
                valid_i = torch.ones((len(queries[0]), dvs.shape[-1]), device=self.device).float()
                # 对于包含部分None的列，和训练时一样，pred全为0，val为-1
                # 根据生成workload规则，不存在[None, op]这样的operator， o1必不为None
                for o1, o2 in all_op_pairs:
                    ind = operators == [o1, o2]
                    ind = np.bitwise_and(ind[:, 0], ind[:, 1])
                    b1 = torch_OPS[o1](dvs.view(1, -1),
                                       torch.as_tensor(vals[ind, 0].astype(int), device=self.device).view(-1, 1))
                    preds[ind, 0, o1] = 1

                    if o2 is not None:
                        b2 = torch_OPS[o2](dvs.view(1, -1),
                                           torch.as_tensor(vals[ind, 1].astype(int), device=self.device).view(-1, 1))
                        valid_i[ind, :] = torch.bitwise_and(b1, b2).float()
                        preds[ind, 1, o2] = 1
                    else:
                        # 此时ind选中的是第二个谓词为空的，需要把第二个谓词对应维度mask设置为1
                        # wild_card_mask[ind, 1, s:e-5] = 1
                        valid_i[ind, :] = b1.float()
                mask = vals == None
                # assert (wild_card_mask[mask][..., s:e-5]==1).all()
                wild_card_mask[mask, i] = 1
                if self.table.columns[natural_idx].nan_ind.any():
                    valid_i[...,0] = 0.
                # nan_mask = torch.isnan(dvs)
                # valid_i[:, nan_mask] = 0.
                # (queries, cols, valid_i)
                valid_i_list[i] = valid_i
                vals[mask] = -1
                v = torch.as_tensor(vals.astype(int), device=self.device).view(-1, 2, 1)
                self._put_samples_as_input(
                    v,
                    preds,
                    inp,
                    natural_idx,
                    sampling_order_idx=i)
            wild_card_mask = torch.as_tensor(wild_card_mask.astype(bool)).pin_memory()
            return inp.cpu().pin_memory(), valid_i_list, wild_card_mask

    def _sample_n_batch(self,
                        ordering,
                        queries,
                        inp,
                        valid_i_list):
        ncols = queries[0].shape[1]
        n_samples = queries[0].shape[0]
        # assert not (torch.isnan(inp).any() or torch.isinf(inp).any())
        logits = self.model.forward_with_encoded_input(inp, ordering=ordering, decode=False)
        # assert not (torch.isnan(logits).any() or torch.isinf(logits).any())
        p_pred = torch.ones((n_samples,), device=self.device, requires_grad=True)
        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]
            valid_i = valid_i_list[i]
            probs_i = torch.softmax(self.model.logits_for_col(natural_idx, logits, is_training=True), 1)
            # assert not (torch.isnan(probs_i).any() or torch.isinf(probs_i).any())
            # assert probs_i.shape[-1] == all_columns[natural_idx].distribution_size
            probs_i = probs_i * valid_i
            p_pred = p_pred * probs_i.sum(1)
        return p_pred

    def get_ordering(self, queries):
        if hasattr(self.model, 'orderings'):
            assert len(self.model.orderings) == 1
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(queries[0][0]))
            orderings = [ordering]

        assert len(orderings) == 1
        inv_ordering = np.argsort(ordering)

        # Fast (?) path.
        ordering = orderings[0]
        return ordering if isinstance(
            self.model, transformer.Transformer) else inv_ordering

    def BatchQueryWithGrad(self, queries, inps=None, valid_i_lists=None):
        # Massages queries into natural order.
        # queries = list(map(lambda x: FillInUnqueriedColumns(self.table, *x), queries))

        # TODO: we can move these attributes to ctor.
        ordering = self.get_ordering(queries)
        p = self._sample_n_batch(
            ordering=ordering,
            queries=queries,
            inp=inps,
            valid_i_list=valid_i_lists)
        return p

    def Query(self, columns, operators, vals, valid_i_list=None):
        # Massages queries into natural order.
        if len(columns) != len(self.table.columns):
            columns, operators, vals = FillInUnqueriedColumns(
                self.table, columns, operators, vals)
        # TODO: we can move these attributes to ctor.
        if hasattr(self.model, 'orderings'):
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]

        num_orderings = len(orderings)

        with torch.no_grad():
            # Fast (?) path.
            if num_orderings == 1:
                ordering = orderings[0]
                if valid_i_list is None:
                    valid_i_list = self.ProcessQuery(
                        ordering if isinstance(
                            self.model, transformer.Transformer) else np.argsort(ordering),
                        columns,
                        operators,
                        vals)
                    self.valid_i_list_cache.append(valid_i_list)
                inp = torch.narrow(self.inp, 0, 0, 1).clone()
                self.OnStart()
                p = self._sample_n(
                    ordering if isinstance(
                        self.model, transformer.Transformer) else np.argsort(ordering),
                    columns,
                    operators,
                    vals,
                    valid_i_list,
                    inp)
                self.OnEnd()
                return np.ceil(p * self.cardinality).astype(dtype=np.int32,
                                                            copy=False)

            # Num orderings > 1.
            ps = []
            self.OnStart()
            for ordering in orderings:
                p_scalar = self._sample_n(ordering, columns, operators, vals, torch.narrow(self.inp, 0, 0, 1).clone())
                ps.append(p_scalar)
            self.OnEnd()
            return np.ceil(np.mean(ps) * self.cardinality).astype(
                dtype=np.int32, copy=False)


class SampleFromModel(CardEst):
    """Sample from an autoregressive model."""

    def __init__(self, model, table, num_samples_per_query, device=None):
        super(SampleFromModel, self).__init__()
        self.model = model
        self.table = table  # The table that MADE is trained on.
        self.num_samples_per_query = num_samples_per_query
        self.device = device  # device to use for pytorch

        doms = [c.DistributionSize() for c in table.columns]
        # Right shift by 1; put 0 at head.
        doms[1:] = doms[:-1]
        doms[0] = 0
        self.cumsum_shifted_doms = np.cumsum(doms)
        print('shifted cumsum', self.cumsum_shifted_doms)

    def __str__(self):
        return 'msample_{}'.format(self.num_samples_per_query)

    def SampleTuples(self, num):
        """Samples num tuples from the MADE model"""
        samples = self.model.sample(num,
                                    self.device).to(torch.int32).cpu().numpy()
        return samples

    def Query(self, columns, operators, vals):
        columns, operators, vals = FillInUnqueriedColumns(
            self.table, columns, operators, vals)
        self.OnStart()

        # [N, num cols].
        tuples = self.SampleTuples(self.num_samples_per_query)

        # all_valids:
        # [ (col1) T, F, F, T; (col2) F, F, T; (col3) T ]
        #
        # Samples:
        # [ [ 0, 2, 0 ];  [1, 1, 0] ]
        #
        # Then only the first sample satisfies the query.

        all_valids = []
        for col, op, val in zip(columns, operators, vals):
            if op is not None:
                valid = OPS[op](col.all_distinct_values, val)
            else:
                valid = [True] * col.DistributionSize()
            all_valids.extend(valid)
        all_valids = np.asarray(all_valids)

        # all() along column dimension: indicates whether each sample passes.
        s = all_valids.take(tuples + self.cumsum_shifted_doms).all(1).sum()
        sel = s * 1.0 / self.num_samples_per_query

        self.OnEnd()
        return np.ceil(sel * self.table.cardinality).astype(dtype=np.int32)


class Heuristic(CardEst):
    """Uses independence assumption."""

    def __init__(self, table):
        super(Heuristic, self).__init__()
        self.table = table
        self.size = self.table.cardinality

    def __str__(self):
        return 'heuristic'

    def Query(self, columns, operators, vals):
        self.OnStart()

        sels = [
            OPS[o](c.data if isinstance(c.data, np.ndarray) else c.data.values,
                   v).sum() / self.size
            for c, o, v in zip(columns, operators, vals)
        ]
        sel = np.prod(sels)

        self.OnEnd()
        return np.ceil(sel * self.size).astype(np.int32)


class Oracle(CardEst):
    """Returns true cardinalities."""

    def __init__(self, table, limit_first_n=None):
        super(Oracle, self).__init__()
        self.table = copy.deepcopy(table)
        for c in self.table.columns:
            data = util.in2d(c.data.values, c.all_distinct_values)
            c.data_gpu = torch.as_tensor(data, device=util.get_device())
            c.nan_ind = pd.isnull(c.data).values
        self.limit_first_n = limit_first_n

    def __str__(self):
        return 'oracle'

    def Query(self, columns, operators, vals, queue=None, return_masks=False):
        assert len(columns) == len(operators) == len(vals)
        self.OnStart()

        bools = None
        for c, o, v in zip(columns, operators, vals):
            if o[0] is None:
                continue
            if self.limit_first_n is None:
                inds = torch_OPS[o[0]](self.table.columns_dict[c.name].data_gpu, v[0])
            else:
                # For data shifts experiment.
                inds = torch_OPS[o[0]](self.table.columns_dict[c.name].data_gpu[:self.limit_first_n], v[0])
            if o[1] is not None:
                if self.limit_first_n is None:
                    s_inds = torch_OPS[o[1]](self.table.columns_dict[c.name].data_gpu, v[1])
                else:
                    # For data shifts experiment.
                    s_inds = torch_OPS[o[1]](self.table.columns_dict[c.name].data_gpu[:self.limit_first_n], v[1])

                inds = torch.bitwise_and(inds, s_inds)

            # this is correct since we makesure no nan value in vals when generating queries
            # inds[self.table.columns_dict[c.name].nan_ind] = False

            # o = OPS_array[o]
            # v = c.all_distinct_values[v]
            # if self.limit_first_n is None:
            #     inds_ = OPS[o](c.data, v)
            # else:
            #     # For data shifts experiment.
            #     inds_ = OPS[o](c.data[:self.limit_first_n], v)
            #
            # assert (inds_.values==inds.cpu().numpy()).all()
            if bools is None:
                bools = inds
            else:
                bools &= inds
        c = bools.sum()
        self.OnEnd()
        if queue is not None:
            queue.put(1)
        if return_masks:
            return bools
        return c.item()


class QueryRegionSize(CardEst):
    """Returns query region size including wildcards."""

    def __init__(self, table, count_wildcards=True):
        super().__init__()
        self.table = table
        self.count_wildcards = count_wildcards

    def __str__(self):
        return 'region_size_{}'.format(self.count_wildcards)

    def Query(self, columns, operators, vals, return_masks=False):
        columns, operators, vals = FillInUnqueriedColumns(
            self.table, columns, operators, vals)

        total_size = 1.0
        for c, o, v in zip(columns, operators, vals):
            if o is None:
                if self.count_wildcards:
                    domain_i_size = len(c.all_distinct_values)
                else:
                    domain_i_size = 1.0
            else:
                domain_i_size = OPS[o](c.all_distinct_values, v).sum()
            total_size *= domain_i_size
        return total_size


class Const(CardEst):
    """Returns a constant."""

    def __init__(self, const):
        super().__init__()
        self.const = const

    def __str__(self):
        return 'Const[{}]'.format(self.const)

    def Query(self, columns, operators, vals):
        self.OnStart()
        c = self.const
        self.OnEnd()
        return c


class Sampling(CardEst):
    """Keep p% of samples in memory."""

    def __init__(self, table, p):
        super(Sampling, self).__init__()
        self.table = table

        self.p = p
        self.num_samples = int(p * table.cardinality)
        self.size = table.cardinality

        # TODO: add seed for repro.
        self.tuples = table.data.sample(n=self.num_samples)

        self.name = str(self)

    def __str__(self):
        if self.p * 100 != int(self.p * 100):
            return 'sample_{:.1f}%'.format(self.p * 100)
        return 'sample_{}%'.format(int(self.p * 100))

    def Query(self, columns, operators, vals):
        assert len(columns) == len(operators) == len(vals)
        self.OnStart()

        qualifying_tuples = []
        for col, op, val in zip(columns, operators, vals):
            qualifying_tuples.append(OPS[op](self.tuples[col.name], val))
        s = np.all(qualifying_tuples, axis=0).sum()
        sel = s * 1.0 / self.num_samples

        self.OnEnd()
        return np.ceil(sel * self.table.cardinality).astype(dtype=np.int32)


class Postgres(CardEst):

    def __init__(self, database, relation, port=None):
        """Postgres estimator (i.e., EXPLAIN).  Must have the PG server live.
        E.g.,
            def MakeEstimators():
                return [Postgres('dmv', 'vehicle_reg', None), ...]
        Args:
          database: string, the database name.
          relation: string, the relation name.
          port: int, the port.
        """
        import psycopg2

        super(Postgres, self).__init__()

        self.conn = psycopg2.connect(database=database, port=port)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

        self.cursor.execute('analyze ' + relation + ';')
        self.conn.commit()

        self.database = database
        self.relation = relation

    def __str__(self):
        return 'postgres'

    def Query(self, columns, operators, vals):
        assert len(columns) == len(operators) == len(vals)
        pred = QueryToPredicate(columns, operators, vals)
        # Use json so it's easier to parse.
        query_s = 'explain(format json) select * from ' + self.relation + pred
        # print(query_s)
        self.OnStart()
        self.cursor.execute(query_s)
        res = self.cursor.fetchall()
        # print(res)
        result = res[0][0][0]['Plan']['Plan Rows']
        self.OnEnd()
        return result

    def QueryByExec(self, columns, operators, vals):
        # Runs actual query on postgres and returns true cardinality.
        assert len(columns) == len(operators) == len(vals)

        pred = QueryToPredicate(columns, operators, vals)
        query_s = 'select count(*) from ' + self.relation + pred
        self.string = query_s

        self.cursor.execute(query_s)
        result = self.cursor.fetchone()[0]

        return result

    def Close(self):
        self.cursor.close()
        self.conn.close()


class BayesianNetwork(CardEst):
    """Progressive sampling with a pomegranate bayes net."""

    def build_discrete_mapping(self, table, discretize, discretize_method):
        assert discretize_method in ["equal_size",
                                     "equal_freq"], discretize_method
        self.max_val = collections.defaultdict(lambda: None)
        if not discretize:
            return {}
        table = table.copy()
        mapping = {}
        for col_id in range(len(table[0])):
            col = table[:, col_id]
            if max(col) > discretize:
                if discretize_method == "equal_size":
                    denom = (max(col) + 1) / discretize
                    fn = lambda v: np.floor(v / denom)
                elif discretize_method == "equal_freq":
                    per_bin = len(col) // discretize
                    counts = collections.defaultdict(int)
                    for x in col:
                        counts[int(x)] += 1
                    assignments = {}
                    i = 0
                    bin_size = 0
                    for k, count in sorted(counts.items()):
                        if bin_size > 0 and bin_size + count >= per_bin:
                            bin_size = 0
                            i += 1
                        assignments[k] = i
                        self.max_val[col_id] = i
                        bin_size += count
                    assignments = np.array(
                        [assignments[i] for i in range(int(max(col) + 1))])

                    def capture(assignments):

                        def fn(v):
                            return assignments[v.astype(np.int32)]

                        return fn

                    fn = capture(assignments)
                else:
                    assert False

                mapping[col_id] = fn
        return mapping

    def apply_discrete_mapping(self, table, discrete_mapping):
        table = table.copy()
        for col_id in range(len(table[0])):
            if col_id in discrete_mapping:
                fn = discrete_mapping[col_id]
                table[:, col_id] = fn(table[:, col_id])
        return table

    def apply_discrete_mapping_to_value(self, value, col_id, discrete_mapping):
        if col_id not in discrete_mapping:
            return value
        return discrete_mapping[col_id](value)

    def __init__(self,
                 dataset,
                 num_samples,
                 algorithm="greedy",
                 max_parents=-1,
                 topological_sampling_order=True,
                 use_pgm=True,
                 discretize=None,
                 discretize_method="equal_size",
                 root=None):
        CardEst.__init__(self)

        from pomegranate import BayesianNetwork
        self.discretize = discretize
        self.discretize_method = discretize_method
        self.dataset = dataset
        self.original_table = self.dataset.tuples.numpy()
        self.algorithm = algorithm
        self.topological_sampling_order = topological_sampling_order
        self.num_samples = num_samples
        self.discrete_mapping = self.build_discrete_mapping(
            self.original_table, discretize, discretize_method)
        self.discrete_table = self.apply_discrete_mapping(
            self.original_table, self.discrete_mapping)
        print('calling BayesianNetwork.from_samples...', end='')
        t = time.time()
        self.model = BayesianNetwork.from_samples(self.discrete_table,
                                                  algorithm=self.algorithm,
                                                  max_parents=max_parents,
                                                  n_jobs=8,
                                                  root=root)
        print('done, took', time.time() - t, 'secs.')

        def size(states):
            n = 0
            for state in states:
                if "distribution" in state:
                    dist = state["distribution"]
                else:
                    dist = state
                if dist["name"] == "DiscreteDistribution":
                    for p in dist["parameters"]:
                        n += len(p)
                elif dist["name"] == "ConditionalProbabilityTable":
                    for t in dist["table"]:
                        n += len(t)
                    if "parents" in dist:
                        for parent in dist["parents"]:
                            n += size(dist["parents"])
                else:
                    assert False, dist["name"]
            return n

        self.size = 4 * size(json.loads(self.model.to_json())["states"])

        # print('json:\n', self.model.to_json())
        self.json_size = len(self.model.to_json())
        self.use_pgm = use_pgm
        #        print(self.model.to_json())

        if topological_sampling_order:
            self.sampling_order = []
            while len(self.sampling_order) < len(self.model.structure):
                for i, deps in enumerate(self.model.structure):
                    if i in self.sampling_order:
                        continue  # already ordered
                    if all(d in self.sampling_order for d in deps):
                        self.sampling_order.append(i)
                print("Building sampling order", self.sampling_order)
        else:
            self.sampling_order = list(range(len(self.model.structure)))
        print("Using sampling order", self.sampling_order, str(self))

        if use_pgm:
            from pgmpy.models import BayesianModel
            data = pd.DataFrame(self.discrete_table.astype(np.int64))
            spec = []
            orphans = []
            for i, parents in enumerate(self.model.structure):
                for p in parents:
                    spec.append((p, i))
                if not parents:
                    orphans.append(i)
            print("Model spec", spec)
            model = BayesianModel(spec)
            for o in orphans:
                model.add_node(o)
            print('calling pgm.BayesianModel.fit...', end='')
            t = time.time()
            model.fit(data)
            print('done, took', time.time() - t, 'secs.')
            self.pgm_model = model

    def __str__(self):
        return "bn-{}-{}-{}-{}-bytes-{}-{}-{}".format(
            self.algorithm,
            self.num_samples,
            "topo" if self.topological_sampling_order else "nat",
            self.size,
            # self.json_size,
            self.discretize,
            self.discretize_method if self.discretize else "na",
            "pgmpy" if self.use_pgm else "pomegranate")

    def Query(self, columns, operators, vals):
        if len(columns) != len(self.dataset.table.columns):
            columns, operators, vals = FillInUnqueriedColumns(
                self.dataset.table, columns, operators, vals)

        self.OnStart()
        ncols = len(columns)
        nrows = self.discrete_table.shape[0]
        assert ncols == self.discrete_table.shape[1], (
            ncols, self.discrete_table.shape)

        def draw_conditional_pgm(evidence, col_id):
            """PGM version of draw_conditional()"""

            if operators[col_id] is None:
                op = None
                val = None
            else:
                op = OPS[operators[col_id]]
                val = self.apply_discrete_mapping_to_value(
                    self.dataset.table.val_to_bin_funcs[col_id](vals[col_id]),
                    col_id, self.discrete_mapping)
                if self.discretize:
                    # avoid some bad cases
                    if val == 0 and operators[col_id] == "<":
                        val += 1
                    elif val == self.max_val[col_id] and operators[
                        col_id] == ">":
                        val -= 1

            def prob_match(distribution):
                if not op:
                    return 1.
                p = 0.
                for k, v in enumerate(distribution):
                    if op(k, val):
                        p += v
                return p

            from pgmpy.inference import VariableElimination
            model_inference = VariableElimination(self.pgm_model)
            xi_distribution = []
            for row in evidence:
                e = {}
                for i, v in enumerate(row):
                    if v is not None:
                        e[i] = v
                result = model_inference.query(variables=[col_id], evidence=e)
                xi_distribution.append(result[col_id].values)

            xi_marginal = [prob_match(d) for d in xi_distribution]
            filtered_distributions = []
            for d in xi_distribution:
                keys = []
                prob = []
                for k, p in enumerate(d):
                    if not op or op(k, val):
                        keys.append(k)
                        prob.append(p)
                denominator = sum(prob)
                if denominator == 0:
                    prob = [1. for _ in prob]  # doesn't matter
                    if len(prob) == 0:
                        prob = [1.]
                        keys = [0.]
                prob = np.array(prob) / sum(prob)
                filtered_distributions.append((keys, prob))
            xi_samples = [
                np.random.choice(k, p=v) for k, v in filtered_distributions
            ]

            return xi_marginal, xi_samples

        def draw_conditional(evidence, col_id):
            """Draws a new value x_i for the column, and returns P(x_i|prev).
            Arguments:
                evidence: shape [BATCH, ncols] with None for unknown cols
                col_id: index of the current column, i
            Returns:
                xi_marginal: P(x_i|x0...x_{i-1}), computed by marginalizing
                    across the range constraint
                match_rows: the subset of rows from filtered_rows that also
                    satisfy the predicate at column i.
            """

            if operators[col_id] is None:
                op = None
                val = None
            else:
                op = OPS[operators[col_id]]
                val = self.apply_discrete_mapping_to_value(
                    self.dataset.table.val_to_bin_funcs[col_id](vals[col_id]),
                    col_id, self.discrete_mapping)
                if self.discretize:
                    # avoid some bad cases
                    if val == 0 and operators[col_id] == "<":
                        val += 1
                    elif val == self.max_val[col_id] and operators[
                        col_id] == ">":
                        val -= 1

            def prob_match(distribution):
                if not op:
                    return 1.
                p = 0.
                for k, v in distribution.items():
                    if op(k, val):
                        p += v
                return p

            xi_distribution = self.model.predict_proba(evidence,
                                                       max_iterations=1,
                                                       n_jobs=-1)
            xi_marginal = [
                prob_match(d[col_id].parameters[0]) for d in xi_distribution
            ]
            filtered_distributions = []
            for d in xi_distribution:
                keys = []
                prob = []
                for k, p in d[col_id].parameters[0].items():
                    if not op or op(k, val):
                        keys.append(k)
                        prob.append(p)
                denominator = sum(prob)
                if denominator == 0:
                    prob = [1. for _ in prob]  # doesn't matter
                    if len(prob) == 0:
                        prob = [1.]
                        keys = [0.]
                prob = np.array(prob) / sum(prob)
                filtered_distributions.append((keys, prob))
            xi_samples = [
                np.random.choice(k, p=v) for k, v in filtered_distributions
            ]

            return xi_marginal, xi_samples

        p_estimates = [1. for _ in range(self.num_samples)]
        evidence = [[None] * ncols for _ in range(self.num_samples)]
        for col_id in self.sampling_order:
            if self.use_pgm:
                xi_marginal, xi_samples = draw_conditional_pgm(evidence, col_id)
            else:
                xi_marginal, xi_samples = draw_conditional(evidence, col_id)
            for ev_list, xi in zip(evidence, xi_samples):
                ev_list[col_id] = xi
            for i in range(self.num_samples):
                p_estimates[i] *= xi_marginal[i]

        self.OnEnd()
        return int(np.mean(p_estimates) * nrows)


class MaxDiffHistogram(CardEst):
    """MaxDiff n-dimensional histogram."""

    def __init__(self, table, limit):
        super(MaxDiffHistogram, self).__init__()
        self.table = table
        self.limit = limit
        self.partitions = []
        self.maxdiff = {}
        self.partition_to_maxdiff = {}
        self.num_new_partitions = 2
        # map<cid, map<bound_type, map<bound_value, list(partition id)>>>
        self.column_bound_map = {}
        for cid in range(len(self.table.columns)):
            self.column_bound_map[cid] = {}
            self.column_bound_map[cid]['l'] = {}
            self.column_bound_map[cid]['u'] = {}
        # map<cid, map<bound_type, sorted_list(bound_value)>>
        self.column_bound_index = {}
        for cid in range(len(self.table.columns)):
            self.column_bound_index[cid] = {}
            self.column_bound_index[cid]['l'] = []
            self.column_bound_index[cid]['u'] = []
        self.name = str(self)

        print('Building MaxDiff histogram, may take a while...')
        self._build_histogram()

    def __str__(self):
        return 'maxdiff[{}]'.format(self.limit)

    class Partition(object):

        def __init__(self):
            # a list of tuples (low, high)
            self.boundaries = []
            # a list of row id that belongs to this partition
            self.data_points = []
            # per-column uniform spread
            self.uniform_spreads = []
            # per-distinct value density
            self.density = None
            self.col_value_list = {}
            self.rowid_to_position = {}

        def Size(self):
            total_size = 0
            for _ in self.uniform_spreads:
                total_size += (4 * len(_))
            total_size += 4
            return total_size

    def _compute_maxdiff(self, partition):
        for col in range(len(partition.boundaries)):
            vals = partition.col_value_list[col]
            counter_start = time.time()
            counter = pd.Series(vals).value_counts().sort_index()

            # compute Diff(V, A)
            spread = counter.index[1:] - counter.index[:-1]
            spread_m_counts = spread * counter.iloc[:-1]
            maxdiff = 0
            if len(spread_m_counts) > 0:
                maxdiff = max(spread_m_counts.values.max(), 0)
            if maxdiff not in self.maxdiff:
                self.maxdiff[maxdiff] = [(partition, col)]
            else:
                self.maxdiff[maxdiff].append((partition, col))
            self.partition_to_maxdiff[partition].add(maxdiff)

    def _build_histogram(self):
        # initial partition
        p = self.Partition()
        # populate initial boundary
        for cid in range(len(self.table.columns)):
            if not self.table.columns[cid].data.dtype == 'int64':
                p.boundaries.append(
                    (0, self.table.columns[cid].distribution_size, True))
            else:
                p.boundaries.append((min(self.table.columns[cid].data),
                                     max(self.table.columns[cid].data), True))
        # include all rowids
        self.table_ds = common.TableDataset(self.table)
        num_rows = self.table.cardinality
        p.data_points = np.arange(num_rows)
        for cid in range(len(p.boundaries)):
            if not self.table.columns[cid].data.dtype == 'int64':
                p.col_value_list[cid] = self.table_ds.tuples_np[:, cid]
            else:
                p.col_value_list[cid] = self.table.columns[cid].data[:, cid]
        p.rowid_to_position = list(np.arange(num_rows))
        self.partition_to_maxdiff[p] = set()
        self._compute_maxdiff(p)
        self.partitions.append(p)

        while len(self.partitions) < self.limit:
            start_next_partition = time.time()
            (split_partition_index, split_column_index, partition_boundaries,
             global_maxdiff) = self.next_partition_candidate(
                self.partitions, len(self.table.columns), self.table,
                min(self.num_new_partitions,
                    self.limit - len(self.partitions) + 1), self.maxdiff)
            print('determining partition number ', len(self.partitions))
            if global_maxdiff == 0:
                print('maxdiff already 0 before reaching bucket limit')
                break
            start_generate_next_partition = time.time()
            new_partitions = self.generate_new_partitions(
                self.partitions[split_partition_index], split_column_index,
                partition_boundaries)
            for p in new_partitions:
                self.partition_to_maxdiff[p] = set()
                self._compute_maxdiff(p)
            for d in self.partition_to_maxdiff[
                self.partitions[split_partition_index]]:
                remove_set = set()
                for cid in range(len(self.table.columns)):
                    remove_set.add(
                        (self.partitions[split_partition_index], cid))
                for tp in remove_set:
                    if tp in self.maxdiff[d]:
                        self.maxdiff[d].remove(tp)
                if len(self.maxdiff[d]) == 0:
                    del self.maxdiff[d]
            del self.partition_to_maxdiff[
                self.partitions[split_partition_index]]
            self.partitions.pop(split_partition_index)
            self.partitions += new_partitions
        # finish partitioning, for each partition we condense its info and
        # compute uniform spread.
        total_point = 0
        for pid, partition in enumerate(self.partitions):
            total_point += len(partition.data_points)
            total = len(partition.data_points)
            total_distinct = 1
            for cid, boundary in enumerate(partition.boundaries):
                distinct = len(
                    set(self.table.columns[cid].data[rowid]
                        for rowid in partition.data_points))
                if distinct == 1:
                    if not self.table.columns[cid].data.dtype == 'int64':
                        partition.uniform_spreads.append([
                            list(
                                set(self.table_ds.
                                    tuples_np[partition.data_points, cid]))[0]
                        ])
                    else:
                        partition.uniform_spreads.append([
                            list(
                                set(self.table.columns[cid].data[
                                        partition.data_points]))[0]
                        ])
                else:
                    uniform_spread = None
                    spread_length = None
                    if boundary[2]:
                        spread_length = float(
                            (boundary[1] - boundary[0])) / (distinct - 1)
                        uniform_spread = [boundary[0]]
                    else:
                        spread_length = float(
                            (boundary[1] - boundary[0])) / (distinct)
                        uniform_spread = [boundary[0] + spread_length]
                    for _ in range(distinct - 2):
                        uniform_spread.append(uniform_spread[-1] +
                                              spread_length)
                    uniform_spread.append(boundary[1])
                    partition.uniform_spreads.append(uniform_spread)
                total_distinct = total_distinct * distinct
            partition.density = float(total) / total_distinct
        print('total number of point is ', total_point)

        # populate column bound map and list metadata
        for cid in range(len(self.table.columns)):
            for pid, partition in enumerate(self.partitions):
                if partition.boundaries[cid][0] not in self.column_bound_map[
                    cid]['l']:
                    self.column_bound_map[cid]['l'][partition.boundaries[cid]
                    [0]] = [pid]
                else:
                    self.column_bound_map[cid]['l'][partition.boundaries[cid]
                    [0]].append(pid)

                if partition.boundaries[cid][1] not in self.column_bound_map[
                    cid]['u']:
                    self.column_bound_map[cid]['u'][partition.boundaries[cid]
                    [1]] = [pid]
                else:
                    self.column_bound_map[cid]['u'][partition.boundaries[cid]
                    [1]].append(pid)

                self.column_bound_index[cid]['l'].append(
                    partition.boundaries[cid][0])
                self.column_bound_index[cid]['u'].append(
                    partition.boundaries[cid][1])
            self.column_bound_index[cid]['l'] = sorted(
                set(self.column_bound_index[cid]['l']))
            self.column_bound_index[cid]['u'] = sorted(
                set(self.column_bound_index[cid]['u']))

    def next_partition_candidate(self, partitions, column_number, table,
                                 num_new_partitions, maxdiff_map):
        global_maxdiff = max(sorted(maxdiff_map.keys()))
        partition, cid = maxdiff_map[global_maxdiff][0]
        vals = partition.col_value_list[cid]
        counter = collections.Counter(vals)
        first_key = True
        prev_key = None
        diff = []
        for key in sorted(counter.keys()):
            if first_key:
                first_key = False
                prev_key = key
            else:
                spread = key - prev_key
                diff.append((prev_key, (spread * counter[prev_key])))
                prev_key = key
        diff.append((prev_key, (0 * counter[prev_key])))
        partition_boundaries = sorted(
            list(
                tp[0]
                for tp in sorted(diff, key=operator.itemgetter(
                    1), reverse=True)[:min(num_new_partitions - 1, len(diff))]))
        return (partitions.index(partition), cid, partition_boundaries,
                global_maxdiff)

    def generate_new_partitions(self, partition, partition_column_index,
                                partition_boundaries):
        new_partitions = []
        for i in range(len(partition_boundaries) + 1):
            new_partition = self.Partition()
            for cid, boundary in enumerate(partition.boundaries):
                if not cid == partition_column_index:
                    new_partition.boundaries.append(boundary)
                else:
                    if i == 0:
                        new_partition.boundaries.append(
                            (boundary[0], partition_boundaries[i], boundary[2]))
                    elif i == len(partition_boundaries):
                        new_partition.boundaries.append(
                            (partition_boundaries[i - 1], boundary[1], False))
                    else:
                        new_partition.boundaries.append(
                            (partition_boundaries[i - 1],
                             partition_boundaries[i], False))
            new_partitions.append(new_partition)
        # distribute data points to new partitions
        for rowid in partition.data_points:
            if not self.table.columns[
                       partition_column_index].data.dtype == 'int64':
                val = self.table_ds.tuples_np[rowid, partition_column_index]
            else:
                val = self.table.columns[partition_column_index].data[rowid]
            # find the new partition that the row belongs to
            new_partitions[bisect.bisect_left(partition_boundaries,
                                              val)].data_points.append(rowid)

        # debug
        start = time.time()
        for new_partition in new_partitions:
            for cid in range(len(new_partition.boundaries)):
                new_partition.col_value_list[cid] = [
                    partition.col_value_list[cid][
                        partition.rowid_to_position[rowid]]
                    for rowid in new_partition.data_points
                ]
            pos = 0
            for rowid in new_partition.data_points:
                new_partition.rowid_to_position[rowid] = pos
                pos += 1
            if len(new_partition.data_points) == 0:
                print('found partition with no data!')
                print(sorted(
                    list(self.table.columns[partition_column_index].data[rowid]
                         for rowid in partition.data_points)))
                print(partition_boundaries)
        return new_partitions

    def _populate_column_set_map(self, c, o, v, column_set_map):
        cid = self.table.ColumnIndex(c.name)
        column_set_map[cid] = set()
        if o in ['<', '<=']:
            insert_index = None
            if o == '<':
                insert_index = bisect.bisect_left(
                    self.column_bound_index[cid]['l'], v)
                for i in range(insert_index):
                    column_set_map[cid] = column_set_map[cid].union(
                        self.column_bound_map[cid]['l'][
                            self.column_bound_index[cid]['l'][i]])
            else:
                insert_index = bisect.bisect(self.column_bound_index[cid]['l'],
                                             v)
                for i in range(insert_index):
                    if self.column_bound_index[cid]['l'][i] == v:
                        for pid in self.column_bound_map[cid]['l'][v]:
                            if self.partitions[pid].boundaries[cid][2]:
                                # add only when the lower bound is inclusive
                                column_set_map[cid].add(pid)
                    else:
                        column_set_map[cid] = column_set_map[cid].union(
                            self.column_bound_map[cid]['l'][
                                self.column_bound_index[cid]['l'][i]])

        elif o in ['>', '>=']:
            insert_index = None
            if o == '>':
                insert_index = bisect.bisect(self.column_bound_index[cid]['u'],
                                             v)
            else:
                insert_index = bisect.bisect_left(
                    self.column_bound_index[cid]['u'], v)
            for i in range(insert_index,
                           len(self.column_bound_index[cid]['u'])):
                column_set_map[cid] = column_set_map[cid].union(
                    self.column_bound_map[cid]['u'][self.column_bound_index[cid]
                    ['u'][i]])
        else:
            assert o == '=', o
            lower_bound_set = set()
            insert_index = bisect.bisect(self.column_bound_index[cid]['l'], v)
            for i in range(insert_index):
                if self.column_bound_index[cid]['l'][i] == v:
                    for pid in self.column_bound_map[cid]['l'][v]:
                        if self.partitions[pid].boundaries[cid][2]:
                            # add only when the lower bound is inclusive
                            lower_bound_set.add(pid)
                else:
                    lower_bound_set = lower_bound_set.union(
                        self.column_bound_map[cid]['l'][
                            self.column_bound_index[cid]['l'][i]])

            upper_bound_set = set()
            insert_index = bisect.bisect_left(self.column_bound_index[cid]['u'],
                                              v)
            for i in range(insert_index,
                           len(self.column_bound_index[cid]['u'])):
                upper_bound_set = upper_bound_set.union(
                    self.column_bound_map[cid]['u'][self.column_bound_index[cid]
                    ['u'][i]])
            column_set_map[cid] = lower_bound_set.intersection(upper_bound_set)

    def _estimate_cardinality_per_partition(self, partition, columns, operators,
                                            vals):
        distinct_val_covered = 1
        observed_cid = []
        for c, o, v in zip(columns, operators, vals):
            if not c.data.dtype == 'int64':
                v = c.ValToBin(v)
            cid = self.table.ColumnIndex(c.name)
            observed_cid.append(cid)
            spread = partition.uniform_spreads[cid]
            if o in ['<', '<=']:
                if o == '<':
                    distinct_val_covered = distinct_val_covered * bisect.bisect_left(
                        spread, v)
                else:
                    distinct_val_covered = distinct_val_covered * bisect.bisect(
                        spread, v)
            elif o in ['>', '>=']:
                if o == '>':
                    distinct_val_covered = distinct_val_covered * (
                            len(spread) - bisect.bisect(spread, v))
                else:
                    distinct_val_covered = distinct_val_covered * (
                            len(spread) - bisect.bisect_left(spread, v))
            else:
                assert o == '=', o
                if not v in spread:
                    distinct_val_covered = 0
        for cid in range(len(partition.uniform_spreads)):
            if not cid in observed_cid:
                distinct_val_covered = distinct_val_covered * len(
                    partition.uniform_spreads[cid])
        return distinct_val_covered * partition.density

    def Query(self, columns, operators, vals):
        self.OnStart()
        # map<cid, set(pids)>
        column_set_map = {}
        for c, o, v in zip(columns, operators, vals):
            if not c.data.dtype == 'int64':
                v = c.ValToBin(v)
            self._populate_column_set_map(c, o, v, column_set_map)
        # compute the set of pids that's relevant to this query
        relevant_pids = set()
        first = True
        for cid in column_set_map:
            if first:
                relevant_pids = column_set_map[cid]
                first = False
            else:
                relevant_pids = relevant_pids.intersection(column_set_map[cid])
        total_card = 0
        # for each pid, check full or partial coverage
        # if full coverage, just add the full count
        # otherwise, estimate the partial sum with uniform spread assumption
        for pid in relevant_pids:
            total_card += self._estimate_cardinality_per_partition(
                self.partitions[pid], columns, operators, vals)
        self.OnEnd()
        return total_card

    def Size(self):
        total_size = 15 * 2 * 4
        for p in self.partitions:
            total_size += p.Size()
        total_size += 24 * (len(self.partitions) - 1)
        return total_size
