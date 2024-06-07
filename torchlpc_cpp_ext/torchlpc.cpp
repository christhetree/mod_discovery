#include <torch/extension.h>
#include <torch/script.h>
#include "/opt/homebrew/opt/libomp/include/omp.h"

torch::Tensor torchlpc_forward(torch::Tensor x, torch::Tensor a, torch::Tensor zi) {
    // Ensure input dimensions are correct
    TORCH_CHECK(x.dim() == 2, "x must be 2-dimensional");
    TORCH_CHECK(a.dim() == 3, "a must be 3-dimensional");
    TORCH_CHECK(x.size(0) == a.size(0), "Batch size of x and a must match");
    TORCH_CHECK(x.size(1) == a.size(1), "Time dimension of x and a must match");

    // Get the dimensions
    const auto B = a.size(0);
    const auto T = a.size(1);
    const auto order = a.size(2);

    // Ensure the zi tensor is the correct size
    TORCH_CHECK(zi.sizes() == torch::IntArrayRef({B, order}), "zi must have shape (B, order)");

    // Flip zi and a to match scipy.signal.lfilter
    zi = torch::flip(zi, {1});
    // This is not needed when doing the nested for loop approach
//    a = torch::flip(a, {2});

    // Concatenate zi and x along the time dimension
    auto padded_y = torch::cat({zi, x}, 1);
//    int result = 0;

    // Perform the computation for each time step
//    omp_set_num_threads(10);
//    #pragma omp parallel for reduction(-:padded_y)
//    #pragma omp parallel for reduction(+:result)
//    for (int64_t t = 0; t < T; ++t) {
//        auto a_slice = a.slice(1, t, t + 1);
//        auto y_slice = padded_y.slice(1, t, t + order).unsqueeze(2);
//        auto prod = torch::matmul(a_slice, y_slice).squeeze(2);
//        padded_y.slice(1, t + order, t + order + 1) -= prod;
//    }
//    at::parallel_for(0, T * 40000, 100, [&](int64_t begin, int64_t end) {
//        for (auto t = begin; t < end; ++t) {
////            result += 1;
//            auto a_slice = a.slice(1, t, t + 1);
//            auto y_slice = padded_y.slice(1, t, t + order).unsqueeze(2);
//            auto prod = torch::matmul(a_slice, y_slice).squeeze(2);
//            padded_y.slice(1, t + order, t + order + 1) -= prod;
//        }
//    });
//    assert(result != 0);

//    for (int64_t b = 0; b < B; ++b) {
//        for (int64_t t = 0; t < T; ++t) {
//            auto ref = padded_y.index({b, t + order});
//            for (int64_t i = 0; i < order; ++i) {
//                auto a_val = a.index({b, t, i});
//                auto y_val = padded_y.index({b, t + order - i - 1});
//                auto prod = a_val * y_val;
//                ref -= prod;
//            }
//            padded_y.index_put_({b, t + order}, ref);
//        }
//    }
    at::parallel_for(0, B, 1, [&](int64_t begin_b, int64_t end_b) {
        for (auto b = begin_b; b < end_b; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                auto ref = padded_y.index({b, t + order});
                at::parallel_for(0, order, 1, [&](int64_t begin_i, int64_t end_i) {
                    for (auto i = begin_i; i < end_i; ++i) {
                        auto a_val = a.index({b, t, i});
                        auto y_val = padded_y.index({b, t + order - i - 1});
                        auto prod = a_val * y_val;
                        ref -= prod;
                    }
                });
                padded_y.index_put_({b, t + order}, ref);
            }
        }
    });

    // Remove the padding and return the result
    auto y = padded_y.slice(1, order, T + order);
    return y;
}

TORCH_LIBRARY(torchlpc, m) {
  m.def("forward", torchlpc_forward);
}
