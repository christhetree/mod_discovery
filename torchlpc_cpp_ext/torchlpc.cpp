#include <torch/extension.h>
#include <torch/script.h>

torch::Tensor torchlpc_forward(torch::Tensor x, torch::Tensor a, torch::Tensor zi) {
  // tmp
  auto s = torch::sigmoid(x);
  return (1 - s) * s;
}

// Registor for torch extensions, not compatible with torchscript
// import torchlpc_cpp
// torchlpc_cpp.forward()
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &torchlpc_forward, "torchlpc forward");
}

// Registor for torchscript
// torch.ops.custom_ts_ops.torchlpc_forward()
TORCH_LIBRARY(custom_ts_ops, m) {
  m.def("torchlpc_forward", torchlpc_forward);
}
