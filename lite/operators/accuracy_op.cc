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

#include "lite/operators/accuracy_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool AccuracyOp::CheckShape() const {
  CHECK_OR_FALSE(param_.Out);
  CHECK_OR_FALSE(param_.Indices);
  CHECK_OR_FALSE(param_.Label)
  CHECK_OR_FALSE(param_.Accuracy);
  CHECK_OR_FALSE(param_.Correct);
  CHECK_OR_FALSE(param_.Total);
  return true;
}

bool AccuracyOp::InferShapeImpl() const {
  auto inference_dim = param_.Out->dims();
  auto label_dim = param_.Label->dims();

  CHECK_EQ(label_dim.size(), 2) << 
            "ShapeError: label's dimensions of AccuracyOp must be 2. "
            "But received label's dimensions = " << label_dim.size() <<
            ", label's shape = [" << label_dim << "]";
  CHECK_EQ(inference_dim[0], label_dim[0]) << 
            "ShapeError: the output's num_rows of AccuracyOp must be"
              " the same as label's num_rows. But received output's "
              "shape = [" << inference_dim << "], label's shape = ["
              << label_dim << "], output's num_rows = " << inference_dim[0] << ", "
              "label's "
              "num_rows = " << label_dim[0];
  
  param_.Accuracy->Resize({1});
  param_.Correct->Resize({1});
  param_.Total->Resize({1});

  return true;
}

bool AccuracyOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.Out =
      scope->FindVar(opdesc.Input("Out").front())->GetMutable<lite::Tensor>();
  param_.Label = scope->FindVar(opdesc.Input("Label").front())
                      ->GetMutable<lite::Tensor>();
  param_.Indices =
      scope->FindVar(opdesc.Input("Indices").front())->GetMutable<lite::Tensor>();
  param_.Accuracy = scope->FindVar(opdesc.Output("Accuracy").front())
                      ->GetMutable<lite::Tensor>();
  param_.Correct =
      scope->FindVar(opdesc.Output("Correct").front())->GetMutable<lite::Tensor>();
  param_.Total = scope->FindVar(opdesc.Output("Total").front())
                         ->GetMutable<lite::Tensor>();
  
  CHECK(param_.Out);
  CHECK(param_.Label);
  CHECK(param_.Indices);

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(accuracy, paddle::lite::operators::AccuracyOp);
