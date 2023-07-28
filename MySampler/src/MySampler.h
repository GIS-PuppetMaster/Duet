#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>
#include <thread>
#include <future>
#include "ThreadPool.h"

std::vector<torch::Tensor>
sample_i(int i, const torch::Tensor &tuples, torch::Tensor *new_tuples_, torch::Tensor *new_preds_, int columns_size,
         int start_idx, int end_idx, bool has_none, int num_samples, int first_pred = -1,
         const torch::Tensor *first_mask = nullptr);

void
sample(const torch::Tensor &tuples, torch::Tensor &new_tuples, torch::Tensor &new_preds,
       const std::vector<int> &columns_size, std::vector<bool> &has_nones, int num_samples);


torch::Tensor
encode_i(const torch::Tensor &val, const torch::Tensor &pred, const torch::Tensor &unk_embedding,
         const torch::Tensor &bin_as_onehot_shift);

torch::Tensor
encode(const std::vector<torch::Tensor> &vals, const std::vector<torch::Tensor> &preds,
       const std::vector<torch::Tensor> &unk_embeddings, const std::vector<torch::Tensor> pred_unk_embedding_cache,
       const std::vector<torch::Tensor> &bin_as_onehot_shifts, const std::vector<int> ordering,
       const std::vector<bool> skip_mask);

/*
std::vector<torch::Tensor>
parallel_forward(const std::vector<torch::nn::Sequential>& models, const std::vector<torch::Tensor>& inputs);
*/