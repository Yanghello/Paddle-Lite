// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/lookup_table_grad_compute.h"
#include "lite/kernels/arm/lookup_table_compute.h"
#include <gtest/gtest.h>
#include "lite/core/op_registry.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

using param_t = operators::LookupTableParam;
using grad_param_t = operators::LookupTableGradParam;
using kernel_t = LookupTableCompute;
using grad_kernel_t = LookupTableGradCompute;

class LookupTableGradTester {
 public:
  explicit LookupTableGradTester(DDim table_dims, int64_t ids_size)
           : w_dims_(table_dims) {
             ids_dims_ = DDim({ids_size, 1});
             out_dims_ = DDim({ids_size, table_dims[1]});
           }

  void prepare_kernel() {
    std::unique_ptr<KernelContext> ctx1(new KernelContext);
    ctx1->As<ARMContext>();
    kernel_.SetContext(std::move(ctx1));

    std::unique_ptr<KernelContext> ctx2(new KernelContext);
    ctx2->As<ARMContext>();
    delta_kernel_.SetContext(std::move(ctx2));

    std::unique_ptr<KernelContext> ctx3(new KernelContext);
    ctx3->As<ARMContext>();
    grad_kernel_.SetContext(std::move(ctx3));
  }

  void run_forward(param_t* param,
                   kernel_t* kernel,
                   const std::vector<float>& w_vec,
                   const std::vector<int64_t>& ids_vec,
                   float* out_vec) {
    Tensor w;
    Tensor ids;
    Tensor output;
    w.Resize(w_dims_);
    ids.Resize(ids_dims_);
    output.Resize(out_dims_);
    auto* w_data = w.mutable_data<float>();
    auto* ids_data = ids.mutable_data<int64_t>();
    for (int i = 0; i < w_dims_.production(); i++) {
      w_data[i] = w_vec[i];
    }
    for (int i = 0; i < ids_dims_.production(); i++) {
      ids_data[i] = ids_vec[i];
    }

    param->W = &w;
    param->Ids = &ids;
    param->Out = &output;
    kernel->SetParam(*param);
    kernel->Launch();

    auto* output_data = output.mutable_data<float>();
    for (int i = 0; i < out_dims_.production(); i++) {
      out_vec[i] = output_data[i];
    }
  }

  void run_backward(grad_param_t* param,
                    grad_kernel_t* kernel,
                    const std::vector<float>& w_vec,
                    const std::vector<int64_t>& ids_vec,
                    const std::vector<float>& out_grad_vec,
                    float* w_grad_vec) {
    Tensor w;
    Tensor w_grad;
    Tensor ids;
    Tensor out_grad;
    w.Resize(w_dims_);
    w_grad.Resize(w_dims_);
    ids.Resize(ids_dims_);
    out_grad.Resize(out_dims_);
    auto* w_data = w.mutable_data<float>();
    auto* ids_data = ids.mutable_data<int64_t>();
    auto* out_grad_data = out_grad.mutable_data<float>();
    for (int i = 0; i < w_dims_.production(); i++) {
      w_data[i] = w_vec[i];
    }
    for (int i = 0; i < ids_dims_.production(); i++) {
      ids_data[i] = ids_vec[i];
    }
    for (int i = 0; i < out_dims_.production(); i++) {
      out_grad_data[i] = out_grad_vec[i];
    }

    param->W = &w;
    param->W_Grad = &w_grad;
    param->Ids = &ids;
    param->Out_Grad = &out_grad;
    kernel->SetParam(*param);
    kernel->Launch();

    auto* w_grad_data = w_grad.mutable_data<float>();
    for (int i = 0; i < w_dims_.production(); i++) {
      w_grad_vec[i] = w_grad_data[i];
    }
  }

  void check_grad() {

    // forward
    std::vector<float> w(w_dims_.production());
    std::vector<int64_t> ids(ids_dims_.production());
    std::vector<float> out(out_dims_.production());
    fill_data_rand(w.data(), -1.f, 1.f, w_dims_.production());
    fill_data_rand<int64_t>(ids.data(), 0, w_dims_.size(), ids_dims_.production());
    this->run_forward(&param_, &kernel_, w, ids, out.data());

    // backward
    std::vector<float> out_grad(out_dims_.production());
    std::vector<float> w_grad(w_dims_.production());
    for (int i = 0; i < out_dims_.production(); i++) {
      out_grad[i] = 1.0;
    }
    this->run_backward(&grad_param_,
                       &grad_kernel_,
                       w,
                       ids,
                       out_grad,
                       w_grad.data());

    // get numeric gradient
    std::vector<float> w_delta(w_dims_.production());
    std::vector<float> out_delta(out_dims_.production());

    float delta = 0.001;
    float max_grad_delta = 0.0055;
    for (int i = 0; i < w_dims_.production(); i++) {
      for (int j = 0; j < w_dims_.production(); j++) {
        if (i == j) {
          w_delta[j] = w[j] + delta;
        } else {
          w_delta[j] = w[j];
        }
      }
      this->run_forward(
          &delta_param_, &delta_kernel_, w_delta, ids, out_delta.data());

      float sum = 0;
      for (int j = 0; j < out_dims_.production(); j++) {
        sum += (out_delta[j] - out[j]);
      }

      EXPECT_NEAR(w_grad[i], sum / delta, max_grad_delta);
    }

  }

 private:
  DDim w_dims_;
  DDim ids_dims_;
  DDim out_dims_;
 
  kernel_t kernel_;
  kernel_t delta_kernel_;
  grad_kernel_t grad_kernel_;
  param_t param_;
  param_t delta_param_;
  grad_param_t grad_param_;
};

void TestNormalCase(const std::vector<int64_t>& table_dims,
                    uint64_t ids_size) {
  std::unique_ptr<LookupTableGradTester> tester(new LookupTableGradTester(
      DDim(table_dims), ids_size));

  tester->prepare_kernel();

  tester->check_grad();
}

TEST(lookup_table_grad_arm, compute) {
  LOG(INFO) << "Test Lookup Table grad";
  DeviceInfo::Init();
  TestNormalCase({17, 31}, 4);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(lookup_table, kARM, kAny, kNCHW, def);
USE_LITE_KERNEL(lookup_table_grad, kARM, kAny, kNCHW, def);
