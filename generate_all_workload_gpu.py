import argparse
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import datasets
import estimators
import util
from util import GenerateQuery
import torch
import pickle as pkl
OPS = [torch.eq, torch.greater, torch.less, torch.greater_equal, torch.less_equal]

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='')
args = parser.parse_args()


def Query(columns, operators, vals, queue=None, return_masks=False):
    assert len(columns) == len(operators) == len(vals)

    bools = None
    for c, o, v in zip(columns, operators, vals):
        if o[0] is None:
            continue
        data = c.data
        # v = np.where(c.all_distinct_values==v)[0].item()

        inds = OPS[o[0]](data, v[0])
        if o[1] is not None:
            assert v[1] is not None
            inds = torch.bitwise_and(inds, OPS[o[1]](data, v[1]))
        if c.nan_ind.any():
            inds[c.data == 0] = False

        if bools is None:
            bools = inds
        else:
            bools &= inds
    c = bools.sum()
    if queue is not None:
        queue.put(1)
    if return_masks:
        return bools
    return c.item()


if __name__ == '__main__':
    util.gpu_id = 3
    conf = ((100000, 42), (2000, 42), (2000, 1234))
    # conf = ((100000, 42), )
    for dataset in ['dmv', 'cup98', 'census']:
        if dataset == 'dmv':
            table = datasets.LoadDmv()
        elif dataset == 'census':
            table = datasets.LoadCensus()
        elif dataset == 'cup98':
            table = datasets.LoadCup98()
        else:
            table = datasets.LoadDmv(f'{dataset}.csv')
        data_list = []
        has_cache = False
        if os.path.exists(f'./temp/{dataset}_in2d.pkl'):
            with open(f'./temp/{dataset}_in2d.pkl', 'rb') as f:
                data_list = pkl.load(f)
            has_cache = True
        for idx, c in enumerate(table.columns):
            c.nan_ind = pd.isnull(c.data)
            if has_cache:
                data = data_list[idx]
            else:
                data = util.in2d(c.data.values, c.all_distinct_values)
                data_list.append(data)
            c.data = torch.as_tensor(data, device=util.get_device())
        if not has_cache:
            if not os.path.exists(f'./temp/'):
                os.makedirs('./temp/')
            with open(f'./temp/{dataset}_in2d.pkl', 'wb') as f:
                pkl.dump(data_list, f)
        for num_queries, seed in conf:
            rng = np.random.RandomState(seed)
            true_cards = []
            queries = []
            raw_queries = []
            for i in tqdm(range(num_queries), desc='generating queries'):
                if seed == 42:
                    if table.bounded_col is None:
                        # 随机选取一列作为bounded column
                        while table.bounded_col is None or table.columns[table.bounded_col].distribution_size<min(100, np.max(np.array([x.distribution_size for x in table.columns]))):
                            table.bounded_col = rng.randint(0, len(table.columns))
                        # 选取其中1% distinct value
                        dvs = table.columns[table.bounded_col].all_distinct_values
                        n = int(np.clip(len(dvs) * 0.01, 1, len(dvs)))
                        r = int(np.clip(n/2, 1, len(dvs)))
                        center = rng.randint(r, len(dvs)-r)
                        table.bounded_distinct_value = dvs[center-r:center+r]
                #     query = GenerateInWorkloadQuery(table.columns, rng, table, dataset)
                # else:
                    query = GenerateQuery(table.columns, rng, table, dataset, bound=True)
                else:
                    query = GenerateQuery(table.columns, rng, table, dataset)
                raw_query = copy.deepcopy(query)
                util.ConvertSQLQueryToBin(query)
                # print(f'Query {i}: Q(', end='')
                # for c, o, v in zip(query[0], query[1], query[2]):
                #     if query[1][0] is not None:
                #         print('{} {} {}, '.format(c.name, o, str(v)), end='')
                # print(')')
                query = estimators.FillInUnqueriedColumns(table, *query, replace_to_torch=False)
                raw_query = estimators.FillInUnqueriedColumns(table, *raw_query, replace_to_torch=False)
                for vals in raw_query[2]:
                    for k, v in enumerate(vals):
                        if not isinstance(v, np.datetime64):
                            break
                        vals[k] = pd.Timestamp(v).to_pydatetime().strftime('%m/%d/%Y')
                queries.append(query)
                raw_queries.append([[c.name for c in raw_query[0]], raw_query[1], raw_query[2]])
            for i in tqdm(range(num_queries), desc='generating cards'):
                query = queries[i]
                true_card = Query(*query)
                assert true_card>0
                true_cards.append(true_card)
                queries[i] = [*query, true_card]
                raw_queries[i] = [*raw_queries[i], true_card]

            import pickle as pkl

            queries = list(map(lambda x: [np.array(list(map(lambda y: y.name, x[0]))), x[1], x[2], x[3]], queries))
            path = f'datasets/{dataset}-{num_queries}queries-oracle-cards-seed{seed}{args.tag}.pkl'
            print(f'output path: {path}')
            with open(path, 'wb') as f:
                pkl.dump(queries, f)
            path = f'datasets/raw_{dataset}-{num_queries}queries-oracle-cards-seed{seed}{args.tag}.pkl'
            print(f'raw output path: {path}')
            with open(path, 'wb') as f:
                pkl.dump(raw_queries, f)

            data = {
                'true_card': true_cards,
            }
            df = pd.DataFrame(data)
            df['true_card'].to_csv(f'datasets/{dataset}-{num_queries}queries-oracle-cards-seed{seed}{args.tag}.csv',
                                   index=False)
