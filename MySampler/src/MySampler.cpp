#include "MySampler.h"

using namespace torch::indexing;
// namespace py = pybind11;
// using Tensor = torch::Tensor;

std::vector<torch::Tensor>
sample_i(int i, const torch::Tensor &tuples, torch::Tensor *new_tuples_, torch::Tensor *new_preds_, int column_size,
         int start_idx, int end_idx, bool has_none, int num_samples, int first_pred, const torch::Tensor *first_mask) {
    int region_size = end_idx - start_idx;
    torch::TensorOptions device = torch::TensorOptions().device(tuples.device());
    // torch::Tensor pred_steps;
    torch::Tensor available_preds;
    torch::Tensor new_tuples;
    torch::Tensor new_preds;
    torch::Tensor pred_slices;
    int pred_idx;
    if (first_pred == -1) {
        torch::TensorOptions op = device.dtype(torch::kInt64).requires_grad(false);
        new_tuples = torch::zeros({num_samples, 2}, op) - 1;
        new_preds = torch::zeros({num_samples, 2, 5}, op);
        pred_idx = 0;
        int chunk_num = 5;
        available_preds = torch::randperm(5);
        int step = region_size / chunk_num;
        pred_slices = torch::tensor(
                {start_idx, start_idx + step, start_idx + 2 * step, start_idx + 3 * step, start_idx + 4 * step,
                 end_idx}, device.dtype(torch::kInt64));
        /*
        int end = region_size / 4 - 1;
        // value: [1,end-1], index: [0,end-2]
        torch::Tensor range = torch::arange(1, end);
        // select chunk_num-1 number from range without replacement, multiply by 4 and sort them
        // a random number without repeat, range [0,end-2)
        torch::Tensor range_indices = torch::randperm(end - 1);
        // take the first chunk_num-1 indices
        range_indices = range_indices.index({Slice(None, chunk_num - 1)});
        pred_steps = std::get<0>(torch::sort(range.index({range_indices}) * 4));
         */
    } else {
        new_tuples = *new_tuples_;
        new_preds = *new_preds_;
        pred_idx = 1;
        int chunk_num = 3;
        if (first_pred == 1 || first_pred == 3) {
            // permutation [2,-1,4], start from permuting [0,1,2], then add 2, then replace 3 with -1
            available_preds = torch::randperm(3) + 2;
            torch::Tensor ava_indices = available_preds == 3;
            available_preds.index_put_({ava_indices}, -1);
        } else {
            // permutation [1, -1, 3], start from permuting [0,1,2], then add 1, then replace 2 with -1
            available_preds = torch::randperm(3) + 1;
            torch::Tensor ava_indices = available_preds == 2;
            available_preds.index_put_({ava_indices}, -1);
        }
        int step = region_size / chunk_num;
        pred_slices = torch::tensor({start_idx, start_idx + step, start_idx + 2 * step, end_idx},
                                    device.dtype(torch::kInt64));
        /*
        int end = region_size - 1;
        // value: [1,end-1], index: [0,end-2]
        torch::Tensor range = torch::arange(1, end);
        // select chunk_num-1 number from range without replacement, and sort them
        // a random number without repeat, range [0,end-2]
        torch::Tensor range_indices = torch::randperm(end - 1);
        // take the first chunk_num-1 indices
        range_indices = range_indices.index({Slice(None, chunk_num - 1)});
        pred_steps = std::get<0>(torch::sort(range.index({range_indices})));
         */
    }
    // pred_steps += start_idx;
    // torch::Tensor pred_slices = torch::cat({torch::tensor({start_idx}), pred_steps, torch::tensor({end_idx})});
    torch::Tensor put_masks = torch::zeros({region_size,}, device.dtype(torch::kBool));
    torch::Tensor lower_bounds = torch::zeros({region_size,}, device);
    torch::Tensor higher_bounds = torch::zeros({region_size,}, device);
    for (int j = 0; j < available_preds.sizes()[0]; ++j) {
        int pred = available_preds.index({j}).item<int>();
        if (pred == -1) {
            continue;
        }
        int s = pred_slices.index({j}).item<int>();
        int e = pred_slices.index({j + 1}).item<int>();
        int sub_region_size = e - s;
        Slice ls = Slice(s, e);
        torch::Tensor local_tuples = tuples.index({ls});
        torch::Tensor local_new_tuples = new_tuples.index({ls, pred_idx});
        torch::Tensor local_new_preds = new_preds.index({ls, pred_idx, pred});
        torch::Tensor local_new_equal_preds = new_preds.index({ls, pred_idx, 0});
        if (pred == 0) {
            new_tuples.index_put_({ls, pred_idx}, local_tuples);
            local_new_preds.index_put_({"..."}, 1);
        } else {
            torch::Tensor lower_bound;
            torch::Tensor higher_bound;
            switch (pred) {
                case 1:
                case 3:
                    // cv >/>= pred
                    // assume a column has at least one non-NULL value
                    if (has_none) {
                        lower_bound = torch::ones({sub_region_size,}, device);
                    } else {
                        lower_bound = torch::zeros({sub_region_size,}, device);
                    }
                    if (pred == 1) {
                        higher_bound = local_tuples;
                        //std::cout<<"line 94, column_size:"<<column_size<<" high_bound.max():"<<higher_bound.max()<<std::endl;
                    } else {
                        higher_bound = local_tuples + 1;
                        //std::cout<<"line 97, column_size:"<<column_size<<" high_bound.max():"<<higher_bound.max()<<std::endl;
                    }
                    break;
                case 2:
                case 4:
                    if (pred == 2) {
                        lower_bound = local_tuples + 1;
                    } else {
                        lower_bound = local_tuples;
                        if (has_none) {
                            lower_bound.index_put_({lower_bound == 0}, 1);
                        }
                    }
                    // assume a column has at least one non-NULL value
                    higher_bound = torch::zeros({sub_region_size,}, device) + column_size;
                    //std::cout<<"line 112, column_size:"<<column_size<<" high_bound.max():"<<higher_bound.max()<<std::endl;
                    break;
            }
            torch::Tensor put_mask = lower_bound < higher_bound;
            Slice gs = Slice(s - start_idx, e - start_idx);
            if (first_mask != nullptr) {
                torch::bitwise_and_out(put_mask, put_mask,
                                       first_mask->index({Slice(s - start_idx, e - start_idx)}));
            }
            lower_bounds.index_put_({gs}, lower_bound);
            higher_bounds.index_put_({gs}, higher_bound);
            put_masks.index_put_({gs}, put_mask);
            local_new_preds.index_put_({put_mask}, 1);
            torch::Tensor un_mask = torch::logical_not(put_mask);
            if (first_pred == -1) {
                local_new_equal_preds.index_put_({un_mask}, 1);
                // local_new_tuples.index_put_({un_mask}, local_tuples.index({un_mask}));
                if (put_mask.any().item().toBool()) {
                    sample_i(i, tuples, &new_tuples, &new_preds, column_size, s, e, has_none, num_samples, pred,
                             &put_mask);
                }
            }
        }
    }
    Slice slice = Slice(start_idx, end_idx);
    torch::Tensor samples =
            (higher_bounds - lower_bounds) * torch::rand({region_size,}, device.dtype(torch::kFloat64)) + lower_bounds;
    torch::Tensor local_tuples = tuples.index({slice});
    torch::Tensor not_put_masks = torch::logical_not(put_masks);
    torch::Tensor local_new_tuples = new_tuples.index({slice, pred_idx});
    local_new_tuples.index_put_({put_masks}, torch::floor(samples.index({put_masks})).to(
            torch::TensorOptions().dtype(torch::kInt64)));
    if (first_pred == -1) {
        local_new_tuples.index_put_({not_put_masks}, local_tuples.index({not_put_masks}));
        // std::vector<torch::Tensor>&& tmp_ret = std::vector<torch::Tensor>{torch::tensor({i}), torch::unsqueeze(new_tuples, -1), new_preds};
        // return tmp_ret;
        return std::vector<torch::Tensor>{torch::tensor({i}), torch::unsqueeze(new_tuples, -1), new_preds};
    }
    return std::vector<torch::Tensor>{};
}

void
sample(const torch::Tensor &tuples, torch::Tensor &new_tuples, torch::Tensor &new_preds,
       const std::vector<int> &columns_size, std::vector<bool> &has_nones, int num_samples) {
    unsigned long long col_num = columns_size.size();
    std::vector<std::future<std::vector<torch::Tensor>>> thread_list;
    std::vector<torch::Tensor> new_tuples_list(col_num);
    std::vector<torch::Tensor> new_preds_list(col_num);
    for (int i = 0; i < col_num; ++i) {
        std::future<std::vector<torch::Tensor>> t = std::async(std::launch::async, sample_i, i,
                                                               tuples.index({"...", i}),
                                                               nullptr, nullptr, columns_size[i], 0, num_samples,
                                                               has_nones[i], num_samples, -1, nullptr);
        thread_list.push_back(std::move(t));
    }

    for (int i = 0; i < col_num; ++i) {
        std::vector<torch::Tensor> ret = thread_list[i].get();
        int col_idx = ret[0].item<int>();
        new_tuples_list[col_idx] = ret[1];
        new_preds_list[col_idx] = ret[2];
    }
    torch::cat_out(new_tuples, at::TensorList(new_tuples_list), -1);
    torch::cat_out(new_preds, at::TensorList(new_preds_list), -1);
}

torch::Tensor
encode_i(const torch::Tensor &val, const torch::Tensor &pred, const torch::Tensor &unk_embedding,
         const torch::Tensor &bin_as_onehot_shift) {
    torch::Tensor none_pred_mask = val.eq(-1).to(torch::kFloat32);
    torch::Tensor tmp = none_pred_mask * unk_embedding + (1.0 - none_pred_mask) * torch::greater(
            torch::bitwise_and(val, bin_as_onehot_shift).to(torch::kFloat32), 0).to(torch::kFloat32);
    return torch::cat({tmp, pred}, -1);
}

torch::Tensor
encode(const std::vector<torch::Tensor> &vals, const std::vector<torch::Tensor> &preds,
       const std::vector<torch::Tensor> &unk_embeddings, const std::vector<torch::Tensor> pred_unk_embedding_cache,
       const std::vector<torch::Tensor> &bin_as_onehot_shifts, const std::vector<int> ordering,
       const std::vector<bool> skip_mask) {
    unsigned long long col_num = vals.size();
    ThreadPool pool(std::thread::hardware_concurrency());
    std::vector<std::future<torch::Tensor>> thread_list;
    std::vector<torch::Tensor> new_emb_list(col_num);
    std::vector<torch::Tensor> res(col_num);
    for (int i = 0; i < col_num; ++i) {
        int nat_idx = ordering[i];
        if (skip_mask[i]) {
            res[i] = pred_unk_embedding_cache[nat_idx];
        }
        std::future<torch::Tensor> t = pool.enqueue(encode_i, vals[nat_idx], preds[nat_idx], unk_embeddings[nat_idx],
                                                    bin_as_onehot_shifts[nat_idx]);
        thread_list.push_back(std::move(t));
    }
    for (int i = 0; i < col_num; ++i) {
        if (!skip_mask[i]) {
            res[i] = thread_list[i].get();
        }
    }
    return torch::cat(res, -1);
}

/*
std::vector<torch::Tensor>
parallel_forward(const std::vector<torch::nn::Sequential>& models, const std::vector<torch::Tensor>& inputs){
    unsigned long long model_num = models.size();
    std::vector<std::future<std::vector<torch::Tensor>>> thread_list;
    std::vector<torch::Tensor> new_tuples_list(model_num);
    for (int i=0; i < model_num; ++i){
        std::future<std::vector<torch::Tensor>>t = std::async(std::launch::async, models[i]->forward, inputs[i]);
        thread_list.push_back(std::move(t));
    }
}
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample", &sample, "do sample", py::arg("tuples"), py::arg("new_tuples"), py::arg("new_preds"),
          py::arg("columns_size"), py::arg("has_nones"), py::arg("num_samples"));
    m.def("encode", &encode, "do binary encoding for MADE", py::arg("vals"), py::arg("preds"),
          py::arg("unk_embeddings"), py::arg("pred_unk_embedding_cache"), py::arg("bin_as_onehot_shifts"),
          py::arg("ordering"), py::arg("skip_mask"));
}