from eval_model import Main
from util import EvalParam as Param

if __name__ == '__main__':
    queries = 'cup98-2000queries-oracle-cards-seed1234-filter{}.pkl'
    for nfilter in [2, 4, 8, 16, 32, 64, 100]:
        p = Param(dataset='cup98', load_queries=queries.format(nfilter),
                  glob='cup98-7.3MB-data16.542-made-resmade-hidden128_128-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-mlp-seed0-use_workloads_100000-best',
                  layers=2, fc_hiddens=128, direct_io=True, residual=True, input_encoding='binary', output_encoding='one_hot',
                  multi_pred_embedding='mlp',
                  tag='cup98_mlp_binary_Workloads', result_tag=f'cup98_mlp_binary_filter{nfilter}', gpu_id=1,
                  end_epoch=50, full_eval=False)
        Main(p)
