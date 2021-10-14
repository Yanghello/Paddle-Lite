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

#include "lite/kernels/arm/accuracy_compute.h"
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/api/paddle_place.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void AccuracyCompute::Run() {
  auto& param = this->Param<param_t>();
  auto* inference = param.Out;
  auto* indices = param.Indices;
  auto* label = param.Label;
  auto* accuracy = param.Accuracy;
  auto* correct = param.Correct;
  auto* total = param.Total;

  int* correct_data = correct->mutable_data<int>();
  int* total_data = total->mutable_data<int>();
  float* accuracy_data = accuracy->mutable_data<float>();

  const int64_t* indices_data = indices->data<int64_t>();
  const int64_t* label_data = label->data<int64_t>();

  size_t num_samples = inference->dims()[0];
  size_t class_dim = inference->dims()[1];
  *accuracy_data = 0.0f;

  if (num_samples == 0) {
    return;
  }

  int num_correct = 0;
  // assume inference is already the topk of the output
  for (size_t i = 0; i < num_samples; ++i) {
    CHECK_GE(label_data[i], 0) << "label of AccuracyOp must >= 0, But received label[" <<
            i << "] is " << label_data[i];
    for (size_t j = 0; j < class_dim; ++j) {
      if (indices_data[i * class_dim + j] == label_data[i]) {
        ++num_correct;
        break;
      }
    }
  }

  *correct_data = num_correct;
  *total_data = num_samples;
  *accuracy_data =
      static_cast<float>(num_correct) / static_cast<float>(num_samples);
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    accuracy, kARM, kAny, kNCHW, paddle::lite::kernels::arm::AccuracyCompute, def)
    .BindInput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Indices", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Label", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Accuracy", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Correct", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Total", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();
