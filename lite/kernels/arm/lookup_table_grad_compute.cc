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

#include "lite/kernels/arm/lookup_table_grad_compute.h"
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void LookupTableGradCompute::Run() {
  auto& param = this->Param<param_t>();

  auto *table_var = param.W;
  DDim table_dim = table_var->dims();

  int64_t padding_idx = param.padding_idx;

  auto* ids = param.Ids;
  auto* d_output = param.Out_Grad;
  auto* d_table = param.W_Grad;

  auto *ids_data = ids->data<int64_t>();

  int64_t N = table_dim[0];
  int64_t D = table_dim[1];

  auto *d_output_data = d_output->data<float>();
  auto *d_table_data = d_table->mutable_data<float>();

  memset(d_table_data, 0, d_table->numel() * sizeof(float));

  for (int64_t i = 0; i < ids->numel(); ++i) {
    CHECK_LT(ids_data[i], N) <<
        "Variable value (input) of OP(fluid.layers.embedding) "
        "expected >= 0 and < " << N <<", but got " << ids_data[i] <<". Please check input value.";
    CHECK_GE(ids_data[i], 0) <<
        "Variable value (input) of OP(fluid.layers.embedding) "
        "expected >= 0 and < " << N <<", but got " << ids_data[i] << ". Please check input value.";
    for (int j = 0; j < D; ++j) {
      d_table_data[ids_data[i] * D + j] += d_output_data[i * D + j];
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(lookup_table_grad,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::LookupTableGradCompute,
                     def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Out_Grad", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("W_Grad", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(lookup_table_v2_grad,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::LookupTableGradCompute,
                     def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Out_Grad", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("W_Grad", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("lookup_table_v2_grad", 1)
    .Finalize();
