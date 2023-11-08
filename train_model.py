"""Model training."""
import argparse
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
import time
import pickle as pkl

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import common
import datasets
import estimators
import made
import transformer
import util
from util import MakeMade, MakeTransformer, GenerateQuery
# torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()

# Training.
parser.add_argument('--use-workloads', action='store_true')
parser.add_argument('--independent', action='store_true')
parser.add_argument('--use_ensemble', action='store_true')
parser.add_argument('--query-seed', type=int, default=42)
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--num-queries', type=int, default=100)

parser.add_argument('--dataset', type=str, default='dmv-tiny', help='Dataset.')
parser.add_argument('--bs', type=int, default=1024, help='Batch size.')
parser.add_argument('--expand-factor', type=int, default=1, help='How many samples from one tuple')
parser.add_argument(
    '--warmups',
    type=int,
    default=0,
    help='Learning rate warmup steps.  Crucial for Transformer.')
parser.add_argument(
    '--data-model-warmups',
    type=int,
    default=0,
    help='warm-up steps for the data-driven model')
parser.add_argument('--epochs',
                    type=int,
                    default=20,
                    help='Number of epochs to train for.')
parser.add_argument('--constant-lr',
                    type=float,
                    default=None,
                    help='Constant LR?')
parser.add_argument('--multi_pred_embedding',
                    type=str,
                    default='mlp')
# parser.add_argument(
#     '--column-masking',
#     action='store_true',
#     help='Column masking training, which permits wildcard skipping' \
#          ' at querying time.')

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
    help='Set this flag iff using MADE and specifying --order. Flag --order ' \
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

# Ordering.
parser.add_argument('--num-orderings',
                    type=int,
                    default=1,
                    help='Number of orderings.')
parser.add_argument(
    '--order',
    nargs='+',
    type=int,
    required=False,
    help=
    'Use a specific ordering.  ' \
    'Format: e.g., [0 2 1] means variable 2 appears second.'
)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--q-weight',
                    type=float,
                    default=1e-1,
                    help='weight of the query model.')
args = parser.parse_args()
DEVICE = None


# OPS = {
#     '>': torch.greater,
#     '<': torch.less,
#     '>=': torch.greater_equal,
#     '<=': torch.less_equal,
#     '=': torch.equal
# }
# def get_true_card(cols, ops, vals):


def Entropy(name, data, bases=None):
    import scipy.stats
    s = 'Entropy of {}:'.format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == 'e' or base is None
        e = scipy.stats.entropy(data, base=base if base != 'e' else None)
        ret.append(e)
        unit = 'nats' if (base == 'e' or base is None) else 'bits'
        s += ' {:.4f} {}'.format(e, unit)
    print(s)
    return ret


def QError(est_card, card):
    if card == 0 and est_card.item() != 0:
        return est_card
    if card != 0 and est_card.item() == 0:
        return 0
    if card == 0 and est_card == 0:
        return 0
    a = est_card / card
    b = card / est_card
    if a.item() > b.item():
        return a
    else:
        return b


def BatchedQError(actual_cards, est_cards):
    # [batch_size]
    bacth_ones = torch.ones(actual_cards.shape, dtype=torch.float32, device=DEVICE)
    fixed_actual_cards = torch.where(actual_cards == 0., bacth_ones, actual_cards)
    fixed_est_cards = torch.where(est_cards == 0., bacth_ones, est_cards)

    q_error = torch.where(actual_cards > est_cards, fixed_actual_cards / fixed_est_cards,
                          fixed_est_cards / fixed_actual_cards)

    return q_error


def train_workloads(est, queries, true_cards, inps, valid_i_lists):
    est_card = est.BatchQueryWithGrad(queries, inps=inps, valid_i_lists=valid_i_lists)
    # if torch.isnan(est_card).any():
    #     print(f'got nan in est_card{est_card}')
    true_cards = torch.as_tensor(true_cards, device=util.get_device()).view(-1, )
    true_cards.requires_grad = False
    q_error = BatchedQError(true_cards, est_card)
    return q_error


def RunEpoch(split,
             model,
             opt,
             train_data,
             est=None,
             loader=None,
             query_loader=None,
             query_opt=None,
             val_data=None,
             batch_size=100,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []
    if loader is None:
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=(split == 'train'),
                                             collate_fn=dataset.collect_fn)

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    for step, xb in enumerate(tqdm(loader)):
        if split == 'train':
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model ** -0.5) * min(
                        (global_steps ** -.5), global_steps * (t ** -1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr

        if upto and step >= upto:
            break

        xb[0] = xb[0].to(torch.float32)
        xb[1] = xb[1].to(torch.float32)
        xb[2] = xb[2].to(torch.float32)
        if args.use_workloads and not args.independent:
            queries = xb[3]
            true_cards = xb[4]
            inps = xb[5]
            valid_i_lists = xb[6]
        raw_data = xb[2]
        xb = xb[:2]
        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples
        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb[0].shape:
            if mean:
                xb = (xb * std) + mean
            loss = F.binary_cross_entropy_with_logits(
                xbhat, xb, size_average=False) / xbhat.size()[0]
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                    .sum(-1).mean()
            else:
                if num_orders_to_forward == 1:
                    loss = model.nll(xbhat, raw_data).mean()
                else:
                    # Average across orderings & then across minibatch.
                    #
                    #   p(x) = 1/N sum_i p_i(x)
                    #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                    #             = log(1/N) + logsumexp ( log p_i(x) )
                    #             = log(1/N) + logsumexp ( - nll_i (x) )
                    #
                    # Used only at test time.
                    logps = []  # [batch size, num orders]
                    assert len(model_logits) == num_orders_to_forward, len(
                        model_logits)
                    for logits in model_logits:
                        # Note the minus.
                        logps.append(-model.nll(logits, xb))
                    logps = torch.stack(logps, dim=1)
                    logps = logps.logsumexp(dim=1) + torch.log(
                        torch.tensor(1.0 / nsamples, device=logps.device))
                    loss = (-logps).mean()
        if args.use_workloads and not args.independent and split == 'train':
            # q_loss = torch.clamp(train_workloads(est, queries, true_cards, inps, valid_i_lists), max=1e8).mean()
            q_error = train_workloads(est, queries, true_cards, inps, valid_i_lists)
            # if train_data.table.name == 'cup98':
            q_error = q_error[q_error<=1e8]
            q_loss = q_error.mean()
            # bit2nat
            weighted_q_loss = torch.log2(q_loss+1)*args.q_weight
            # weighted_q_loss = q_loss * args.q_weight
            all_loss = loss + weighted_q_loss
            # if args.fade_in_beta > 0.:
            #     q_weight = args.fade_in_beta * np.exp(-10. / (epoch_num + 1))
            #     all_loss = loss + q_weight * q_loss
            # else:
            #     if epoch_num + 1 > args.data_model_warmups:
            #         q_weight = args.q_weight
            #         all_loss = loss + q_weight * q_loss
            #     else:
            #         q_weight = 0.
            #         all_loss = loss
            all_loss = all_loss.mean()
        else:
            all_loss = loss
        losses.append(all_loss.item())
        if step % log_every == 0:
            if split == 'train':
                if args.use_workloads:
                    print(
                        'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}), loss nat {:.3f}, query{:.3f}, weighted query{:.3f}, {:.5f} lr'
                        .format(epoch_num, step, split,
                                loss.item() / np.log(2) - table_bits,
                                loss.item() / np.log(2), table_bits, loss.item(), q_loss.item(), weighted_q_loss.item(), lr))
                else:
                    print(
                        'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}), {:.5f} lr'
                        .format(epoch_num, step, split,
                                loss.item() / np.log(2) - table_bits,
                                loss.item() / np.log(2), table_bits, lr))
            else:
                print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            all_loss.backward()
            if isinstance(model, made.MADE) and model.is_ensembled:
                for layer in list(model.multi_pred_embed_nn):
                    if isinstance(layer, made.EnsembleLinear):
                        layer.weight.grad *= layer.block_gradient_mask
            opt.step()
        if est is not None:
            est.inp.detach_()
        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))
    if args.use_workloads and args.independent and split == 'train':
        assert False, 'deprecated'
        for step, xb in enumerate(tqdm(query_loader)):
            if split == 'train':
                base_lr = 8e-4
                for param_group in query_opt.param_groups:
                    if args.constant_lr:
                        lr = args.constant_lr
                    elif args.warmups:
                        t = args.warmups
                        d_model = model.embed_size
                        global_steps = len(query_loader) * epoch_num + step + 1
                        lr = (d_model ** -0.5) * min(
                            (global_steps ** -.5), global_steps * (t ** -1.5))
                    else:
                        lr = 1e-2

                    param_group['lr'] = lr

            if upto and step >= upto:
                break

            queries = xb[0]
            true_cards = xb[1]
            inps = xb[2]
            valid_i_lists = xb[3]
            q_error = train_workloads(est, queries, true_cards, inps, valid_i_lists)
            if q_error > 0:
                loss = torch.log(q_error / len(queries))
            if step % log_every == 0:
                print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

            if split == 'train':
                query_opt.zero_grad()
                loss.backward()
                query_opt.step()
            if est is not None:
                est.inp.detach_()
    if return_losses:
        return losses
    return np.mean(losses)


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


def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)


def LoadOracleCardinalities():
    path = f'datasets/{args.dataset}-{args.num_queries}queries-oracle-cards-seed{args.query_seed}.csv'
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        return df.values.reshape(-1)
    return None


def TrainTask(param=None, seed=0):
    global args
    if param is not None:
        args = param
    print(args)
    util.gpu_id = args.gpu_id
    global DEVICE
    DEVICE = util.get_device()
    print('Device', DEVICE)
    torch.manual_seed(0)
    np.random.seed(0)

    # assert 'dmv' in args.dataset
    if args.dataset == 'dmv':
        table = datasets.LoadDmv()
    elif args.dataset == 'census':
        table = datasets.LoadCensus()
    elif args.dataset == 'cup98':
        table = datasets.LoadCup98()
    else:
        table = datasets.LoadDmv(f'{args.dataset}.csv')

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns
                                            ]).size(), [2])[0]
    fixed_ordering = None

    if args.order is not None:
        print('Using passed-in order:', args.order)
        fixed_ordering = args.order

    print(table.data.info())

    table_train = table

    if args.heads > 0:
        model = MakeTransformer(args, cols_to_train=table.columns,
                                fixed_ordering=fixed_ordering,
                                DEVICE=DEVICE,
                                seed=seed)
    else:
        model = MakeMade(
            args,
            scale=args.fc_hiddens,
            cols_to_train=table.columns,
            seed=seed,
            DEVICE=DEVICE,
            fixed_ordering=fixed_ordering,
        )

    mb = ReportModel(model)

    if not isinstance(model, transformer.Transformer):
        print('Applying InitWeight()')
        model.apply(InitWeight)

    def build_opt():
        if isinstance(model, transformer.Transformer):
            opt = torch.optim.Adam(
                list(model.parameters()),
                2e-4,
                betas=(0.9, 0.98),
                eps=1e-9,
            )
        else:
            opt = torch.optim.Adam(list(model.parameters()), 2e-4)
        return opt

    opt = build_opt()
    query_opt = None
    if args.use_workloads and args.independent:
        query_opt = build_opt()
    bs = args.bs
    log_every = 200
    est = estimators.DirectEstimator(model,
                                     table,
                                     device=DEVICE,
                                     requires_grad=True,
                                     batch_size=bs)
    query_data = None
    if args.use_workloads:
        with open(f'datasets/{args.dataset}-{args.num_queries}queries-oracle-cards-seed{args.query_seed}.pkl',
                  'rb') as f:
            queries = pkl.load(f)
        queries = list(map(lambda x: [list(map(lambda y: table.columns_dict[y], x[0])), x[1], x[2], x[3]], queries))
        all_columns = list(map(lambda x: x[0], queries))
        all_operators = list(map(lambda x: x[1], queries))
        all_vals = list(map(lambda x: x[2], queries))
        if len(queries[0]) == 4:
            true_cards = list(map(lambda x: x[3]/est.cardinality, queries))
        else:
            true_cards = LoadOracleCardinalities()
        queries = (all_columns, all_operators, all_vals)

        inp, valid_i_list, wild_card_mask = est.encode_queries(queries)
        if args.independent:
            assert False, 'not support yet'
            train_data = common.TableDataset(table_train, bs, args.expand_factor)
            query_data = common.WorkloadDataset(train_data, queries, true_cards, inp, valid_i_list, wild_card_mask,
                                                model)
        else:
            train_data = common.TableDataset(table_train, bs, args.expand_factor, queries, true_cards, inp,
                                             valid_i_list, wild_card_mask, model)
    else:
        train_data = common.TableDataset(table_train, bs, args.expand_factor)

    train_losses = []
    train_start = time.time()
    min_loss = float('inf')
    loader = torch.utils.data.DataLoader(train_data,
                                         batch_size=bs,
                                         shuffle=True,
                                         collate_fn=train_data.collect_fn)
    query_loader = None
    if args.use_workloads and args.independent:
        query_loader = torch.utils.data.DataLoader(query_data,
                                                   batch_size=bs,
                                                   shuffle=True,
                                                   collate_fn=query_data.collect_fn)

    # oracle_est = estimators.Oracle(table)
    if args.tag:
        root = f'models/{args.tag}/'
    else:
        root = 'models/'
    if fixed_ordering is None:
        if seed is not None:
            PATH = '{}-{:.1f}MB-data{:.3f}-{}-seed{}'.format(
                args.dataset, mb, table_bits, model.name(), seed)
        else:
            PATH = '{}-{:.1f}MB-data{:.3f}-{}-seed{}-{}'.format(
                args.dataset, mb, table_bits, model.name(), seed, time.time())
    else:
        annot = ''
        if args.inv_order:
            annot = '-invOrder'

        PATH = '{}-{:.1f}MB-data{:.3f}-{}-seed{}-order{}{}'.format(
            args.dataset, mb, table_bits, model.name(), seed,
            '_'.join(map(str, fixed_ordering)), annot)
    PATH = root + PATH
    if args.use_workloads:
        PATH = PATH + f'-use_workloads_{args.num_queries}'
    print(PATH + '-best.pt')
    log = {
        'epoch': [],
        'model_bits': [],
        'table_bits': [],
        'time_cost': []
    }
    for epoch in range(args.epochs):
        st = time.time()
        mean_epoch_train_loss = RunEpoch('train',
                                         model,
                                         opt,
                                         est=est,
                                         loader=loader,
                                         query_opt=query_opt,
                                         query_loader=query_loader,
                                         train_data=train_data,
                                         val_data=train_data,
                                         batch_size=bs,
                                         epoch_num=epoch,
                                         log_every=log_every,
                                         table_bits=table_bits)
        cost = time.time() - st
        # model_nats = np.mean(all_losses)
        model_bits = mean_epoch_train_loss / np.log(2)
        model.model_bits = model_bits
        log['epoch'].append(epoch)
        log['model_bits'].append(model_bits)
        log['table_bits'].append(table_bits)
        log['time_cost'].append(cost)
        os.makedirs(os.path.dirname(PATH + f'-epoch{epoch}.pt'), exist_ok=True)
        torch.save(model.state_dict(), PATH + f'-epoch{epoch}.pt')
        print('Saved to:')
        print(PATH)

        if mean_epoch_train_loss < min_loss:
            min_loss = mean_epoch_train_loss
            os.makedirs(os.path.dirname(PATH + '-best.pt'), exist_ok=True)
            torch.save(model.state_dict(), PATH + '-best.pt')
            print('Saved best to:')
            print(PATH + '-best.pt')
        if epoch % 1 == 0:
            print('epoch {} train loss {:.4f} nats / {:.4f} bits'.format(
                epoch, mean_epoch_train_loss,
                mean_epoch_train_loss / np.log(2)))
            since_start = time.time() - train_start
            print('time since start: {:.1f} secs'.format(since_start))

        train_losses.append(mean_epoch_train_loss)
        pd.DataFrame(log).to_csv(root + '/train_log.csv')
    print('Training done; evaluating likelihood on full data:')
    # all_losses = RunEpoch('test',
    #                       model,
    #                       train_data=train_data,
    #                       val_data=train_data,
    #                       opt=None,
    #                       batch_size=1024,
    #                       log_every=500,
    #                       table_bits=table_bits,
    #                       return_losses=True)
    # model_nats = np.mean(all_losses)
    # model_bits = model_nats / np.log(2)
    # model.model_bits = model_bits
    #
    # os.makedirs(os.path.dirname(PATH + '-final.pt'), exist_ok=True)
    # torch.save(model.state_dict(), PATH + '-final.pt')
    # print('Saved to:')
    # print(PATH)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    TrainTask()
