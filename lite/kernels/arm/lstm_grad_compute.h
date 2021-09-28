// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <algorithm>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/backends/arm/math/sgemm.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <PrecisionType Ptype>
class LstmGradCompute : public KernelLite<TARGET(kARM), Ptype> {
 public:
  void Run() override;

  virtual ~LstmGradCompute() = default;
};

namespace math {

template<typename T>
struct SetConstant {
    void operator()(Tensor* tensor, T value) {
        auto t = tensor->mutable_data<T>();
        size_t size = tensor->numel();
        for (int i = 0; i < size; ++i) {
            *(t + i) = value;
        }
    }
};

struct TensorMatMul {
    TensorMatMul(ARMContext* ctx) : _ctx(ctx) {}

    void operator()(const Tensor &mat_a, bool trans_a,
                const Tensor &mat_b, bool trans_b,
                float alpha, Tensor *mat_out,
                float beta) {
        auto dim_a = mat_a.dims();
        auto dim_b = mat_b.dims();
        auto dim_out = mat_out->dims();
        
        int M = dim_out[0];
        int N = dim_out[1];
        int K = !trans_a ? dim_a[1] : dim_a[0];

        operators::ActivationParam act_param;
        act_param.has_active = false;

        lite::arm::math::sgemm(trans_a,
                               trans_b,
                               M,
                               N,
                               K,
                               1,
                               mat_a.data<float>(),
                               K,
                               mat_b.data<float>(),
                               N,
                               1,
                               mat_out->mutable_data<float>(),
                               N,
                               nullptr,
                               false,
                               act_param,
                               _ctx);
    }
    ARMContext* _ctx;
};

template <typename T>
class ColwiseSum {
 public:
  void operator()(const lite::TensorLite& input,
                  lite::TensorLite* out) {
    auto& in_dims = input.dims();
    auto height = in_dims[0];
    auto size = in_dims[1];

    T* out_buf = out->template mutable_data<T>();
    const T* in_buf = input.data<T>();

    for (size_t i = 0; i < static_cast<size_t>(height); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(size); ++j) {
        if (i == 0) {
          out_buf[j] = in_buf[i * size + j];
        } else {
          out_buf[j] += in_buf[i * size + j];
        }
      }
    }
  }
};

}  // namespace math

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
