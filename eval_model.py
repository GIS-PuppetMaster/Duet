"""Evaluate estimators (Naru or others) on queries."""
import argparse
import collections
import glob
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import common
import datasets
import estimators as estimators_lib
import made
import transformer
import util
import pickle as pkl
from util import InvertOrder, MakeMade, MakeTransformer, GenerateQuery

# For inference speed.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=200)
parser.add_argument('--load-queries', type=str, default='')
parser.add_argument('--inference-opts',
                    action='store_true',
                    help='Tracing optimization for better latency.')
parser.add_argument('--use_oracle', action='store_true')
parser.add_argument('--load_cache', action='store_true')
parser.add_argument('--full_eval', action='store_true')
parser.add_argument('--num-queries', type=int, default=20, help='# queries.')
parser.add_argument('--query-seed', type=int, default=1234, help='# query seed.')
parser.add_argument('--dataset', type=str, default='dmv-tiny', help='Dataset.')
parser.add_argument('--multi_pred_embedding',
                    type=str,
                    default='mlp')
parser.add_argument('--err-csv',
                    type=str,
                    default='results.csv',
                    help='Save result csv to what path?')
parser.add_argument('--glob',
                    type=str,
                    help='Checkpoints to glob under models/.')
parser.add_argument('--blacklist',
                    type=str,
                    help='Remove some globbed checkpoint files.')
parser.add_argument('--psample',
                    type=int,
                    default=2000,
                    help='# of progressive samples to use per query.')
# parser.add_argument(
#     '--column-masking',
#     action='store_true',
#     help='Turn on wildcard skipping.  Requires checkpoints be trained with ' \
#          'column masking.')
parser.add_argument('--order',
                    nargs='+',
                    type=int,
                    help='Use a specific order?')

# MADE.
parser.add_argument('--fc-hiddens',
                    type=int,
                    default=128,
                    help='Hidden units in FC.')
parser.add_argument('--layers', type=int, default=4, help='# layers in FC.')
parser.add_argument('--residual', action='store_true', help='ResMade?')
parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')
parser.add_argument(
    '--inv-order',
    action='store_true',
    help='Set this flag iff using MADE and specifying --order. Flag --order' \
         'lists natural indices, e.g., [0 2 1] means variable 2 appears second.' \
         'MADE, however, is implemented to take in an argument the inverse ' \
         'semantics (element i indicates the position of variable i).  Transformer' \
         ' does not have this issue and thus should not have this flag on.')
parser.add_argument(
    '--input-encoding',
    type=str,
    default='binary',
    help='Input encoding for MADE/ResMADE, {binary, one_hot, embed}.')
parser.add_argument(
    '--output-encoding',
    type=str,
    default='one_hot',
    help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
         'then input encoding should be set to embed as well.')

# Transformer.
parser.add_argument(
    '--heads',
    type=int,
    default=0,
    help='Transformer: num heads.  A non-zero value turns on Transformer' \
         ' (otherwise MADE/ResMADE).'
)
parser.add_argument('--blocks',
                    type=int,
                    default=2,
                    help='Transformer: num blocks.')
parser.add_argument('--dmodel',
                    type=int,
                    default=32,
                    help='Transformer: d_model.')
parser.add_argument('--dff', type=int, default=128, help='Transformer: d_ff.')
parser.add_argument('--transformer-act',
                    type=str,
                    default='gelu',
                    help='Transformer activation.')

# Estimators to enable.
parser.add_argument('--run-sampling',
                    action='store_true',
                    help='Run a materialized sampler?')
parser.add_argument('--run-maxdiff',
                    action='store_true',
                    help='Run the MaxDiff histogram?')
parser.add_argument('--run-bn',
                    action='store_true',
                    help='Run Bayes nets? If enabled, run BN only.')

# Bayes nets.
parser.add_argument('--bn-samples',
                    type=int,
                    default=200,
                    help='# samples for each BN inference.')
parser.add_argument('--bn-root',
                    type=int,
                    default=0,
                    help='Root variable index for chow liu tree.')
# Maxdiff
parser.add_argument(
    '--maxdiff-limit',
    type=int,
    default=30000,
    help='Maximum number of partitions of the Maxdiff histogram.')
parser.add_argument('--result-tag',type=str, default=None)
parser.add_argument('--tag',type=str, default=None)

args = parser.parse_args()
valid_i_list_cache = None

def MakeTable(build_est=True):

    if args.dataset == 'dmv':
        table = datasets.LoadDmv()
    elif args.dataset == 'census':
        table = datasets.LoadCensus()
    elif args.dataset == 'cup98':
        table = datasets.LoadCup98()
    else:
        table = datasets.LoadDmv(args.dataset + '.csv')

    if build_est:
        oracle_est = estimators_lib.Oracle(table)
    else:
        oracle_est = None
    if args.run_bn:
        return table, common.TableDataset(table), oracle_est
    return table, None, oracle_est


def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def Query(estimators,
          idx,
          do_print=True,
          oracle_card=None,
          query=None,
          table=None,
          oracle_est=None):
    assert query is not None
    cols, ops, vals = query
    # cols = list(map(lambda x:table.columns_dict[x], cols))
    ### Actually estimate the query.

    def pprint(*args, **kwargs):
        if do_print:
            print(*args, **kwargs)

    # pprint('Q(', end='')
    # for c, o, v in zip(cols, ops, vals):
    #     if o[0] is not None:
    #         pprint('{} {} {}, '.format(c.name, o, str(v)), end='')
    # pprint('): ', end='')

    # Actual.
    card = oracle_est.Query(cols, ops,
                            vals) if oracle_card is None else oracle_card
    if oracle_card is not None:
        assert card == oracle_card, (card, oracle_card)
    if card == 0:
        pprint('Q(', end='')
        for c, o, v in zip(cols, ops, vals):
            if o[0] is not None:
                pprint('{} {} {}, '.format(c.name, o, str(v)), end='')
        pprint('): ', end='')
        for est in estimators:
            if hasattr(est, 'valid_i_list_cache'):
                est.valid_i_list_cache.append(None)
        return

    # pprint('\n  actual {} ({:.3f}%) '.format(card,
    #                                          card / table.cardinality * 100),
    #        end='')
    global valid_i_list_cache
    for est in estimators:
        if valid_i_list_cache is not None:
            valid_i_list = valid_i_list_cache[idx]
        else:
            valid_i_list = None
        est_card = est.Query(cols, ops, vals, valid_i_list)
        err = ErrorMetric(est_card, card)
        est.AddError(err, est_card, card)
        # pprint('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
    pprint()


def ReportEsts(estimators, log):
    v = -1
    for est in estimators:
        max_ = np.max(est.errs)
        th99 = np.quantile(est.errs, 0.99)
        th95 = np.quantile(est.errs, 0.95)
        th75 = np.quantile(est.errs, 0.75)
        median = np.quantile(est.errs, 0.5)
        th25 = np.quantile(est.errs, 0.25)
        mean = np.mean(est.errs)
        min_ = np.min(est.errs)
        mean_cost = np.mean(est.query_dur_ms[1:])
        if isinstance(est, estimators_lib.DirectEstimator):
            encoding_cost = np.mean(est.encoding_dur_ms[1:])
            infer_cost = np.mean(est.infer_dur_ms[1:])
            v = max(v, np.max(est.errs))
        else:
            encoding_cost = None
            infer_cost = None
        print(est.name, 'max', max_, '99th',
              th99, '95th', th95, '75th', th75,
              'median', median, '25th', th25, 'min', min_, 'mean', mean, 'mean_cost', mean_cost, 'encoding_cost', encoding_cost, 'infer_cost', infer_cost)
        if log:
            log['est_name'].append(est.name)
            log['max'].append(max_)
            log['99th'].append(th99)
            log['95th'].append(th95)
            log['75th'].append(th75)
            log['median'].append(median)
            log['25th'].append(th25)
            log['min'].append(min_)
            log['mean'].append(mean)
            log['mean_cost'].append(mean_cost)
            log['encoding_cost'].append(encoding_cost)
            log['infer_cost'].append(infer_cost)
    return v


def RunN(table,
         cols,
         estimators,
         rng,
         num=20,
         log_every=50,
         num_filters=11,
         queries=None,
         oracle_cards=None,
         oracle_est=None,
         log=None):
    last_time = None
    all_max_err = float('-inf')
    if queries is not None:
        num = len(queries)
    for i in tqdm(range(num)):
        do_print = False
        # if i % log_every == 0:
            # if last_time is not None:
            #     print('{:.1f} queries/sec'.format(log_every /
            #                                       (time.time() - last_time)))
            # do_print = True
            # print('Query {}:'.format(i), end=' ')
            # last_time = time.time()
        if queries is None:
            query = GenerateQuery(cols, rng, table, args.dataset)
            util.ConvertSQLQueryToBin(query)
            oracle_card = oracle_cards[i] if (oracle_cards is not None and i < len(oracle_cards)) else None
        else:
            query = queries[i][:-1]
            oracle_card = queries[i][-1]
        # with open('./query.pkl', 'rb') as f:
        #     query = pickle.load(f)
        Query(estimators,
              i,
              do_print,
              oracle_card=oracle_card,
              query=query,
              table=table,
              oracle_est=oracle_est)
    global valid_i_list_cache
    if valid_i_list_cache is None:
        for est in estimators:
            if hasattr(est, 'valid_i_list_cache'):
                valid_i_list_cache = estimators[0].valid_i_list_cache
    max_err = ReportEsts(estimators, log)
    if log:
        if args.result_tag:
            pd.DataFrame(log).to_csv(f'./result/{args.result_tag}/summary.csv', index=False)
        else:
            pd.DataFrame(log).to_csv(f'./result/{model_name}/summary.csv', index=False)
    if max_err!=-1 and max_err > all_max_err:
        all_max_err = max_err
        for est in estimators:
            if isinstance(est, estimators_lib.DirectEstimator):
                if args.result_tag:
                    torch.save(est.model.state_dict(), f'./result/{args.result_tag}/eval_best_parameter.pt')
                else:
                    torch.save(est.model.state_dict(), f'./result/{model_name}/eval_best_parameter.pt')
                est.pool.close()
    return all_max_err


def RunNParallel(estimator_factory,
                 parallelism=2,
                 rng=None,
                 num=20,
                 num_filters=11,
                 oracle_cards=None):
    """RunN in parallel with Ray.  Useful for slow estimators e.g., BN."""
    import ray
    ray.init(redis_password='xxx')

    @ray.remote
    class Worker(object):

        def __init__(self, i):
            self.estimators, self.table, self.oracle_est = estimator_factory()
            self.columns = np.asarray(self.table.columns)
            self.i = i

        def run_query(self, query, j):
            col_idxs, ops, vals = pickle.loads(query)
            Query(self.estimators,
                  do_print=True,
                  oracle_card=oracle_cards[j]
                  if oracle_cards is not None else None,
                  query=(self.columns[col_idxs], ops, vals),
                  table=self.table,
                  oracle_est=self.oracle_est)

            print('=== Worker {}, Query {} ==='.format(self.i, j))
            for est in self.estimators:
                est.report()

        def get_stats(self):
            return [e.get_stats() for e in self.estimators]

    print('Building estimators on {} workers'.format(parallelism))
    workers = []
    for i in range(parallelism):
        workers.append(Worker.remote(i))

    print('Building estimators on driver')
    estimators, table, _ = estimator_factory()
    cols = table.columns

    if rng is None:
        rng = np.random.RandomState(args.query_seed)
    queries = []
    for i in range(num):
        col_idxs, ops, vals = GenerateQuery(cols,
                                            rng,
                                            table=table,
                                            return_col_idx=True)
        queries.append((col_idxs, ops, vals))

    cnts = 0
    for i in range(num):
        query = queries[i]
        print('Queueing execution of query', i)
        workers[i % parallelism].run_query.remote(pickle.dumps(query), i)

    print('Waiting for queries to finish')
    stats = ray.get([w.get_stats.remote() for w in workers])

    print('Merging and printing final results')
    for stat_set in stats:
        for e, s in zip(estimators, stat_set):
            e.merge_stats(s)
    time.sleep(1)

    print('=== Merged stats ===')
    for est in estimators:
        est.report()
    return estimators


def MakeBnEstimators():
    table, train_data, oracle_est = MakeTable()
    estimators = [
        estimators_lib.BayesianNetwork(train_data,
                                       args.bn_samples,
                                       'chow-liu',
                                       topological_sampling_order=True,
                                       root=args.bn_root,
                                       max_parents=2,
                                       use_pgm=False,
                                       discretize=100,
                                       discretize_method='equal_freq')
    ]

    for est in estimators:
        est.name = str(est)
    return estimators, table, oracle_est


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


def SaveEstimators(path, estimators, return_df=False):
    # name, query_dur_ms, errs, est_cards, true_cards
    results = pd.DataFrame()
    for est in estimators:
        data = {
            'est': [est.name] * len(est.errs),
            'err': est.errs,
            'est_card': est.est_cards,
            'true_card': est.true_cards,
            'query_dur_ms': est.query_dur_ms,
        }
        df = pd.DataFrame(data)
        df['true_card'].to_csv(
            f'datasets/{args.dataset}-{args.num_queries}queries-oracle-cards-seed{args.query_seed}.csv', index=False)
        results = results.append(df)
    if return_df:
        return results
    results.to_csv(path, index=False)


def LoadOracleCardinalities():
    path = f'datasets/{args.dataset}-{args.num_queries}queries-oracle-cards-seed{args.query_seed}.csv'
    if path and os.path.exists(path):
        print(f'loading true card from: {path}')
        df = pd.read_csv(path)
        return df.values.reshape(-1)
    return None


def Main(params=None):
    global valid_i_list_cache
    valid_i_list_cache = None
    torch.set_grad_enabled(False)
    global args
    if params:
        args = params
    if args.result_tag is None:
        args.result_tag = args.tag
    util.gpu_id = args.gpu_id
    global DEVICE
    DEVICE = util.get_device()
    print('Device', DEVICE)
    global model_name
    model_name = args.glob.replace('.pt', '')

    read = args.load_cache and args.use_oracle and os.path.exists(f'./temp/{args.dataset}_cache.pkl')
    # OK to load tables now
    table, train_data, oracle_est = MakeTable(build_est=not read and args.use_oracle)
    if read:
        with open(f'./temp/{args.dataset}_cache.pkl', 'rb') as f:
            oracle_est = pickle.load(f)
        for c in oracle_est.table.columns:
            c.data_gpu = c.data_gpu.to(util.get_device())
    elif args.use_oracle:
        with open(f'./temp/{args.dataset}_cache.pkl', 'wb') as f:
            pickle.dump(oracle_est, f)
    cols_to_train = table.columns
    if args.load_queries!='':
        with open(f'datasets/{args.load_queries}','rb') as f:
            queries = pkl.load(f)
        queries = list(map(lambda x: [list(map(lambda y: table.columns_dict[y], x[0])), x[1], x[2], x[3]], queries))
        for i in range(len(queries)):
            columns = queries[i][0]
            operators = queries[i][1]
            vals = queries[i][2]
            columns, operators, vals = estimators_lib.FillInUnqueriedColumns(
                table, columns, operators, vals)
            queries[i][0] = columns
            queries[i][1] = operators
            queries[i][2] = vals
        oracle_cards = None
    else:
        oracle_cards = LoadOracleCardinalities()
        queries = None
    print('Waiting for model parameters...')
    flag = True
    while flag:
        if args.tag:
            if 'best' in args.glob:
                all_ckpts = glob.glob('./models/{}/{}'.format(args.tag, args.glob+'.pt'))
            else:
                all_ckpts = glob.glob('./models/{}/{}'.format(args.tag, args.glob+f'-epoch0.pt'))
        else:
            if 'best' in args.glob:
                all_ckpts = glob.glob('./models/{}'.format(args.glob + '.pt'))
            else:
                all_ckpts = glob.glob('./models/{}'.format(args.glob + f'-epoch0.pt'))
        if args.blacklist:
            all_ckpts = [ckpt for ckpt in all_ckpts if args.blacklist not in ckpt]
        selected_ckpts = all_ckpts
        flag = len(selected_ckpts) == 0
    print('ckpts', selected_ckpts)

    Ckpt = collections.namedtuple(
        'Ckpt', 'epoch path loaded_model seed')

    assert len(selected_ckpts) == 1
    s = selected_ckpts[0]
    if args.order is None:
        z = re.match('.+-data([\d\.]+).+seed([\d\.]+).*.pt',
                     s)
    else:
        z = re.match(
            '.+-data([\d\.]+).+seed([\d\.]+)-order.*.pt', s)
    assert z
    # model_bits = float(z.group(1))
    data_bits = float(z.group(1))
    seed = int(z.group(2))
    # bits_gap = model_bits - data_bits

    order = None
    if args.order is not None:
        order = list(args.order)

    if args.heads > 0:
        model = MakeTransformer(args, cols_to_train=table.columns,
                                fixed_ordering=order,
                                DEVICE=DEVICE,
                                seed=seed)
    else:
        model = MakeMade(
            args,
            scale=args.fc_hiddens,
            cols_to_train=table.columns,
            seed=seed,
            DEVICE=DEVICE,
            fixed_ordering=order,
        )

    assert order is None or len(order) == model.nin, order
    ReportModel(model)
    last_mtime = None
    min_max_err = float('inf')
    epoch = args.start_epoch
    if epoch ==0 :
        err_log = {'est_name': [],
                   'max': [],
                   '99th': [],
                   '95th': [],
                   '75th': [],
                   'median': [],
                   '25th': [],
                   'min': [],
                   'mean': [],
                   'mean_cost': [],
                   'encoding_cost': [],
                   'infer_cost': []
                   }
    else:
        if args.result_tag:
            df = pd.read_csv(f'./result/{args.result_tag}/summary.csv', index_col=False)
        else:
            df = pd.read_csv(f'./result/{model_name}/summary.csv', index_col=False)
        err_log = df.to_dict('list')
    if args.result_tag:
        if not os.path.exists(f'./result/{args.result_tag}/by_epoch/'):
            os.makedirs(f'./result/{args.result_tag}/by_epoch/')
    else:
        if not os.path.exists(f'./result/{model_name}/by_epoch/'):
            os.makedirs(f'./result/{model_name}/by_epoch/')
    mtime=None
    have_done_best = not args.full_eval if 'best' not in args.glob else False
    best_only = False
    if 'best' in args.glob:
        best_only = True
    while epoch<args.end_epoch and not (have_done_best and 'best' in args.glob):
        if not have_done_best or 'best' in args.glob:
            s = re.sub('-epoch[0-9]+', f'-best', s)
        else:
            if 'best' in s:
                s = re.sub('-best', f'-epoch{epoch}', s)
            else:
                s = re.sub('-epoch[0-9]+', f'-epoch{epoch}', s)
        if 'best' in args.glob:
            mtime = os.path.getmtime(s)
            start_flag = last_mtime != mtime
        else:
            start_flag = os.path.exists(s)
        if start_flag:
            parsed_ckpts = []
            last_mtime = mtime
            print('Loading ckpt:', s)
            loaded = False
            while not loaded:
                try:
                    model.load_state_dict(torch.load(s, map_location=DEVICE))
                    loaded = True
                except Exception as e:
                    pass
            model.eval()

            print(s, seed)

            parsed_ckpts.append(
                Ckpt(path=s,
                     epoch=None,
                     loaded_model=model,
                     seed=seed))
        else:
            time.sleep(1)
            continue

        # Estimators to run.
        if args.run_bn:
            estimators = RunNParallel(estimator_factory=MakeBnEstimators,
                                      parallelism=50,
                                      rng=np.random.RandomState(args.query_seed),
                                      num=args.num_queries,
                                      num_filters=None,
                                      oracle_cards=oracle_cards)
        else:
            estimators = [
                estimators_lib.DirectEstimator(c.loaded_model,
                                               table,
                                               device=DEVICE)
                for c in parsed_ckpts
            ]
        for est, ckpt in zip(estimators, parsed_ckpts):
            est.name = str(est) + '_{}'.format(ckpt.seed)

        if args.inference_opts:
            print('Tracing forward_with_encoded_input()...')
            for est in estimators:
                encoded_input = est.model.EncodeInput(
                    torch.zeros(args.psample, est.model.nin, device=DEVICE))

                # NOTE: this line works with torch 1.0.1.post2 (but not 1.2).
                # The 1.2 version changes the API to
                # torch.jit.script(est.model) and requires an annotation --
                # which was found to be slower.
                est.traced_fwd = torch.jit.trace(
                    est.model.forward_with_encoded_input, encoded_input)

        if args.run_sampling:
            SAMPLE_RATIO = {'dmv': [0.0013]}  # ~1.3MB.
            for p in SAMPLE_RATIO.get(args.dataset, [0.01]):
                estimators.append(estimators_lib.Sampling(table, p=p))

        if args.run_maxdiff:
            estimators.append(
                estimators_lib.MaxDiffHistogram(table, args.maxdiff_limit))

        # Other estimators can be appended as well.
        if len(estimators):
            max_err = RunN(table,
                           cols_to_train,
                           estimators,
                           rng=np.random.RandomState(args.query_seed),
                           num=args.num_queries,
                           log_every=1,
                           num_filters=None,
                           oracle_cards=oracle_cards,
                           queries=queries,
                           oracle_est=oracle_est,
                           log=err_log)
        if max_err < min_max_err:
            min_max_err = max_err

        if args.result_tag:
            err_root =f'./result/{args.result_tag}/by_epoch/'
        else:
            err_root = f'./result/{model_name}/by_epoch/'
        os.makedirs(err_root, exist_ok=True)
        if 'best' in s:
            err_path = err_root + f'best_min_max_err_{min_max_err}.csv'
        else:
            err_path = err_root + f'epoch_{epoch}_min_max_err_{min_max_err}.csv'
        SaveEstimators(err_path, estimators)
        print('...Done, result:', err_path)
        print(f'min max_err: {min_max_err}')
        if isinstance(model, made.MADE):
            for layer in model.net:
                if type(layer) == made.MaskedLinear:
                    layer.masked_weight = None
            if estimators[0].multi_pred_embed_nn is not None:
                model.multi_pred_embed_nn = estimators[0].multi_pred_embed_nn
        if not have_done_best and 'best' in s:
            have_done_best = True
            min_max_err = float('inf')
        else:
            epoch += 1
        if have_done_best and best_only and not args.full_eval:
            break
        for est in estimators:
            est.clear_errs()


if __name__ == '__main__':
    Main()
