import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train_root = './models/'
    root = './result/'
    res = {
        'method': [],
        'nfilter': [],
        'mean_costs': [],
        'encoding_costs': [],
        'infer_costs': [],
        'sampling_costs': []
    }
    name_map = {'encoding_costs': 'EncodingCost',
                'infer_costs':'InferenceCost',
                'sampling_costs': 'SamplingCost'}
    exps = [('Duet', 'cup98_mlp_binary_filter{}', 'solid', 'x'), ('Naru', 'Naru_cup98_binary_filter{}', 'dashed', 'o'), ('UAE', 'UAE_cup98_binary_filter{}', 'dashdot', '^')]
    nfilters = [2, 4, 8, 16, 32, 64, 100]
    cate_nfilters = list(map(lambda x: str(x), nfilters))
    width = 0.15
    color = ['C0', 'C1', 'C2', 'C3']
    textures = ["///" , "\\\\"  , "---"]
    plt.figure(figsize=(5, 2.5), dpi=300)
    # fig, axes = plt.subplots(nrows=1, ncols=3, sharex='col', sharey='row', squeeze=True, layout='constrained')
    for i, (method_name, exp_root, linestyle, marker) in enumerate(exps):
        methods = []
        mean_costs = []
        encoding_costs = []
        infer_costs = []
        sampling_costs = []
        for nfilter in nfilters:
            exp = exp_root.format(nfilter)
            name = exp.split('_')[1]
            df = pd.read_csv(os.path.join(root, exp, 'summary.csv'))
            mean_cost = df['mean_cost'].values.item()
            encoding_cost = df['encoding_cost'].values.item()
            infer_cost = df['infer_cost'].values.item()
            if method_name in ['Naru', 'UAE']:
                sampling_cost = df['sampling_cost'].values.item()
            else:
                sampling_cost = 0
            sampling_costs.append(sampling_cost)
            methods.append(method_name)
            mean_costs.append(mean_cost)
            encoding_costs.append(encoding_cost)
            infer_costs.append(infer_cost)
        res['method'].extend(methods)
        res['mean_costs'].extend(mean_costs)
        res['encoding_costs'].extend(encoding_costs)
        res['infer_costs'].extend(infer_costs)
        res['nfilter'].extend(nfilters)
        res['sampling_costs'].extend(sampling_costs)
    df = pd.DataFrame(res)
    df.to_csv('./result/nfilter.csv', index=None)
    metrics = ['encoding_costs', 'infer_costs', 'sampling_costs']
    have_labeled = set()
    gap = width + 0.05
    cmap = plt.get_cmap("Set2")
    for i, (method_name, exp_root, linestyle, marker) in enumerate(exps):
        method_data = df[df['method'] == method_name]
        pos = []
        for k, nfilter in enumerate(nfilters):
            position = k + gap * (1 - len(exps)) / 2 + i * gap
            pos.append(position)
            method_nfilter_data = method_data[method_data['nfilter'] == nfilter]
            data = method_nfilter_data[metrics].values.flatten()
            # arg_index = np.argsort(data)
            data_cum = np.cumsum(data)
            for j, idx in enumerate(range(len(data))):
                if j not in have_labeled:
                    plt.bar(position, data[idx], bottom=data_cum[idx-1] if idx>0 else 0, zorder=-j, color=cmap(j), width=width, label=name_map[metrics[j]], alpha=0.8, hatch=textures[j])
                    have_labeled.add(j)
                else:
                    plt.bar(position, data[idx], bottom=data_cum[idx-1] if idx>0 else 0, zorder=-j, color=cmap(j), width=width, alpha=0.8, hatch=textures[j])
        plt.plot(pos, method_data['mean_costs'], label=method_name, linestyle=linestyle, color='black', marker=marker, markersize=5)
    plt.xlabel('#involved_columns')
    plt.ylabel('cost(ms)')
    x = np.arange(0, len(cate_nfilters))
    plt.xticks(x, cate_nfilters)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./result/nfilter.png')
    plt.show()
