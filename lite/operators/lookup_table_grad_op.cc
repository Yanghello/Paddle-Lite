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

#include "lite/operators/lookup_table_grad_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool LookupTableGradOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.W)
  CHECK_OR_FALSE(param_.Ids)

  CHECK_OR_FALSE(param_.W_Grad)
  CHECK_OR_FALSE(param_.Out_Grad)

  const auto& table_dims = param_.W->dims();
  const auto& ids_dims = param_.Ids->dims();

  int ids_rank = ids_dims.size();

  CHECK_EQ_OR_FALSE(table_dims.size(), 2)
  CHECK_EQ_OR_FALSE(ids_dims[ids_rank - 1], 1)

  return true;
}

bool LookupTableGradOpLite::InferShapeImpl() const {
  const auto& table_dims = param_.W->dims();
  param_.W_Grad->Resize(table_dims);
  return true;
}

bool LookupTableGradOpLite::AttachImpl(const cpp::OpDesc& op_desc,
                                   lite::Scope* scope) {
  auto input = op_desc.Input("W").front();
  auto ids = op_desc.Input("Ids").front();
  auto out_g = op_desc.Input("Out@GRAD").front();
   auto w_g = op_desc.Output("W@GRAD").front();

  param_.W = scope->FindTensor(input);
  param_.Ids = scope->FindTensor(ids);
  param_.Out_Grad = scope->FindTensor(out_g);
  param_.W_Grad = scope->FindMutableTensor(w_g);

  param_.padding_idx = op_desc.GetAttr<int64_t>("padding_idx");
  if (op_desc.HasAttr("is_test")) {
    param_.is_test = op_desc.GetAttr<bool>("is_test");
  }
  if (op_desc.HasAttr("entry_config")) {
    param_.entry_config = op_desc.GetAttr<std::string>("entry_config");
  }
  if (op_desc.HasAttr("entry")) {
    param_.entry = op_desc.GetAttr<std::string>("entry");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(lookup_table_grad, paddle::lite::operators::LookupTableGradOpLite)
