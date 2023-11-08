import os
import re
import sys

import numpy as np
import pandas as pd
import torch

gpu_id = 0


def get_device():
    return torch.device(f'cuda:{gpu_id}') if gpu_id >= 0 and torch.cuda.is_available() else 'cpu'


def check_sample(table, tuples, new_tuples, new_preds):
    from estimators import torch_OPS
    num_samples = tuples.shape[0]
    nin = tuples.shape[1]
    eval_preds = torch.zeros((num_samples, 2, nin)).numpy().astype(object)
    for i in range(nin):
        assert (new_tuples[..., i] < table.columns[i].distribution_size).all()
        eval_preds[..., i] = torch_OPS[torch.argmax(new_preds[..., i * 5:(i + 1) * 5], dim=-1).cpu().numpy()]
    for i in range(num_samples):
        for c in range(new_tuples.shape[-1]):
            cv = tuples[i, c]
            # assert (new_tuples[i, 0, c] != -1).all()
            for j in range(2):
                if new_tuples[i, j, c] != -1:
                    assert eval_preds[i, j, c](cv, new_tuples[i, j, c])


def ConvertSQLQueryToBin(query):
    from estimators import OPS_dict
    cols = query[0]
    ops = query[1]
    vals = query[2]
    for op in ops:
        op[0] = OPS_dict[op[0]]
        if op[1] is not None:
            op[1] = OPS_dict[op[1]]

    for j, (col, val) in enumerate(zip(cols, vals)):
        vals[j][0] = np.where(col.all_distinct_values == val[0])[0].item()
        if val[1] is not None:
            vals[j][1] = np.where(col.all_distinct_values == val[1])[0].item()


def ConvertOPSBinToSQLQuery(ops):
    from estimators import OPS_array
    n = len(ops) if isinstance(ops, list) else ops.shape[-1] // 5
    new_ops = [[None, None] for _ in range(n)]
    for i in range(n):
        op = ops[:, i * 5:(i + 1) * 5]
        if op[0].max() > 0:
            new_ops[i][0] = OPS_array[op[0].argmax().item()]
            if op[1].max() > 0:
                new_ops[i][1] = OPS_array[op[1].argmax().item()]
    return new_ops


def in2d(x, all_x):
    mask = x[:, None] == all_x[None, :]
    ind = pd.isnull(x)
    nan_ind = pd.isnull(all_x)
    if nan_ind.any():
        nan_pos = np.argmax(nan_ind)
        assert np.isnan(all_x[nan_pos])
        assert (mask[ind, :].sum(1) == 0).all()
        mask[ind, nan_pos] = True
    return np.where(mask)[1]


def SampleTupleThenRandom(all_cols,
                          num_filters,
                          rng,
                          table,
                          dataset,
                          return_col_idx=False,
                          bound=False):
    s = None
    while s is None or len(s) - pd.isnull(s).astype(int).sum().item() < num_filters:
        s = table.data.iloc[rng.randint(0, table.cardinality)]
    vals = s.values

    if 'dmv' in dataset:
        # Giant hack for DMV.
        vals[6] = vals[6].to_datetime64()

    idxs = None
    while idxs is None or pd.isnull(vals[idxs]).any():
        idxs = rng.choice(len(all_cols), replace=False, size=num_filters)
    cols = np.take(all_cols, idxs)
    replace_pred = None
    replace_idx = None
    original_val = None
    if bound and table.bounded_col is not None and table.bounded_col in idxs and vals[
        table.bounded_col] not in table.bounded_distinct_value:
        replace_val = rng.choice(table.bounded_distinct_value, 1, replace=False)[0]
        replace_bin = table.columns[table.bounded_col].ValToBin(replace_val)
        original_val = vals[table.bounded_col]
        original_bin = table.columns[table.bounded_col].ValToBin(original_val)
        # 确保选择度不为0
        replace_idx = np.where(idxs==table.bounded_col)[0]
        if replace_bin > original_bin:
            replace_pred = '<='
        elif replace_bin < original_bin:
            replace_pred = '>='
        vals[table.bounded_col] = replace_val
        # idxs = idxs.tolist()
        # idxs.remove(table.bounded_col)
        # idxs = np.array(idxs)
    vals = vals[idxs]
    num_filters = len(vals)
    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)
    if replace_idx is not None:
        ops[replace_idx] = replace_pred
    is_range_op = ops != '='
    second_op = [None] * num_filters
    second_vals = [None] * num_filters
    use_second_pred = np.bitwise_and(rng.randint(2, size=(num_filters,)).astype(bool), is_range_op)
    for i in range(num_filters):
        if use_second_pred[i]:
            dvs = cols[i].all_distinct_values
            if ops[i] == '<=':
                # second_val<=col<=val, -> second_val<=val
                second_op[i] = '>='
                lower_idx = 0
                if pd.isnull(dvs[0]):
                    lower_idx = 1
                if i != replace_idx or original_val is None:
                    second_idx = rng.randint(lower_idx, np.where(dvs == vals[i])[0] + 1)[0]
                else:
                    second_idx = rng.randint(lower_idx, np.where(dvs == original_val)[0] + 1)[0]
                second_vals[i] = dvs[second_idx]
            elif ops[i] == '>=':
                second_op[i] = '<='
                # val<=col<=second_val->val<=second_val
                if i != replace_idx or original_val is None:
                    second_idx = rng.randint(np.where(dvs == vals[i])[0], len(dvs))[0]
                else:
                    second_idx = rng.randint(np.where(dvs == original_val)[0], len(dvs))[0]
                second_vals[i] = dvs[second_idx]
    final_ops = []
    for first_op, second_op in zip(ops, second_op):
        final_ops.append([first_op, second_op])

    # if num_filters == len(all_cols):
    #     final_vals = []
    #     for first_val, second_val in zip(vals, second_vals):
    #         final_vals.append([first_val, second_val])
    #     if return_col_idx:
    #         return np.arange(len(all_cols)), ops, vals
    #     return all_cols, ops, vals
    #
    # vals = vals[idxs]
    final_vals = []
    for first_val, second_val in zip(vals, second_vals):
        final_vals.append([first_val, second_val])
    if return_col_idx:
        return idxs, final_ops, final_vals

    return cols, final_ops, final_vals

def GenerateQuery(all_cols, rng, table, dataset, return_col_idx=False, num_filters=None, bound=False):
    """Generate a random query."""
    if num_filters is not None:
        num_filters = min(num_filters, len(table.columns))
    else:
        if dataset == 'dmv':
            if bound:
                num_filters = np.clip(int(rng.gamma(5, 2)), 1, 11)
            else:
                num_filters = rng.randint(5, 12)
        elif dataset == 'cup98':
            if bound:
                num_filters = np.clip(int(rng.gamma(10, 2)), 1, 100)
            else:
                # num_filters = np.clip(int(rng.normal(20, 2)), 1, 100)
                num_filters = rng.randint(5, 101)
        elif dataset == 'census':
            if bound:
                num_filters = np.clip(int(rng.gamma(7, 2)), 1, 13)
            else:
                num_filters = rng.randint(5, 14)
        else:
            num_filters = rng.randint(max(1, int(len(table.columns) * 0.3)), len(table.columns))
    cols, ops, vals = SampleTupleThenRandom(all_cols,
                                            num_filters,
                                            rng,
                                            table,
                                            dataset=dataset,
                                            return_col_idx=return_col_idx,
                                            bound=bound)
    return [cols, ops, vals]


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def train_background(args):
    res = re.match('.*--tag=([^ ]+)[ ]', args)
    assert res
    tag = res.groups()[0]
    command = "nohup " + sys.executable + " train_model.py " + args + f" >./log/{tag}.txt" " &"
    print(command)
    os.system(command)


def MakeMade(args, scale, cols_to_train, seed, DEVICE, fixed_ordering=None):
    if args.inv_order:
        print('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)
    from made import MADE
    model = MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
                     args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        multi_pred_embedding=args.multi_pred_embedding,
        use_ensemble=args.use_ensemble
    ).to(DEVICE)
    # if hasattr(torch, 'compile'):
    #     try:
    #         return torch.compile(model)
    #     except Exception as e:
    #         print(f'error during compile: {e}')
    #         return model
    return model


def MakeTransformer(args, cols_to_train, fixed_ordering, DEVICE, seed=None):
    from transformer import Transformer
    model = Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=args.heads,
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=False,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        seed=seed,
    ).to(DEVICE)
    if hasattr(torch, 'compile'):
        try:
            return torch.compile(model)
        except Exception as e:
            print(f'error during compile: {e}')
            return model
    return model


class EvalParam:
    def __init__(self, glob, blacklist=None, psample=2000, order=None, gpu_id=3, start_epoch=0, load_queries='',
                 inference_opts=False, use_oracle=False, load_cache=True,
                 full_eval=True, num_queries=20, query_seed=1234, dataset='dmv-tiny', multi_pred_embedding='mlp',
                 err_csv='results.csv',
                 fc_hiddens=128, layers=4, residual=False, direct_io=False, inv_order=False, input_encoding='binary',
                 output_encoding='one_hot',
                 heads=0, blocks=2, dmodel=32, dff=128, transformer_act='gelu', run_sampling=False, run_maxdiff=False,
                 run_bn=False, bn_samples=200,
                 bn_root=0, maxdiff_limit=30000, tag=None, end_epoch=100, result_tag=None, use_ensemble=False):
        self.glob = glob
        # only for building model parse, estimator generate ensemble even this is false
        self.use_ensemble = use_ensemble
        self.blacklist = blacklist
        self.psample = psample
        self.result_tag = result_tag
        self.order = order
        self.gpu_id = gpu_id
        self.start_epoch = start_epoch
        self.load_queries = load_queries
        self.inference_opts = inference_opts
        self.use_oracle = use_oracle
        self.load_cache = load_cache
        self.full_eval = full_eval
        self.num_queries = num_queries
        self.query_seed = query_seed
        self.dataset = dataset
        self.multi_pred_embedding = multi_pred_embedding
        self.err_csv = err_csv
        self.fc_hiddens = fc_hiddens
        self.layers = layers
        self.residual = residual
        self.direct_io = direct_io
        self.inv_order = inv_order
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        self.heads = heads
        self.blocks = blocks
        self.dmodel = dmodel
        self.dff = dff
        self.transformer_act = transformer_act
        self.run_sampling = run_sampling
        self.run_maxdiff = run_maxdiff
        self.run_bn = run_bn
        self.bn_samples = bn_samples
        self.bn_root = bn_root
        self.maxdiff_limit = maxdiff_limit
        self.tag = tag
        self.end_epoch = end_epoch


class TrainParam:
    def __init__(self, order=None, gpu_id=0, num_queries=100, query_seed=42, dataset='dmv-tiny',
                 multi_pred_embedding='mlp', err_csv='results.csv', fc_hiddens=123, layers=4, residual=False,
                 direct_io=False, inv_order=False, input_encoding='binary', output_encoding='one_hot', heads=0,
                 blocks=2, dmodel=32, dff=128, transformer_act='gelu', tag=None,
                 use_workloads=False, independent=False, bs=1024, warmups=0, data_model_warmups=0, epochs=20,
                 constant_lr=None, num_orderings=1, q_weight=1e-2, expand_factor=4, use_ensemble=False):
        self.order = order
        self.use_ensemble = use_ensemble
        self.expand_factor = expand_factor
        self.gpu_id = gpu_id
        self.num_queries = num_queries
        self.query_seed = query_seed
        self.dataset = dataset
        self.multi_pred_embedding = multi_pred_embedding
        self.err_csv = err_csv
        self.fc_hiddens = fc_hiddens
        self.layers = layers
        self.residual = residual
        self.direct_io = direct_io
        self.inv_order = inv_order
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        self.heads = heads
        self.blocks = blocks
        self.dmodel = dmodel
        self.dff = dff
        self.transformer_act = transformer_act
        self.tag = tag
        self.use_workloads = use_workloads
        self.independent = independent
        self.bs = bs
        self.warmups = warmups
        self.data_model_warmups = data_model_warmups
        self.epochs = epochs
        self.constant_lr = constant_lr
        self.num_orderings = num_orderings
        self.num_queries = num_queries
        self.q_weight = q_weight
