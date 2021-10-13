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

#include "lite/operators/lstm_grad_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {
inline lite_api::ActivationType GetActivationType(const std::string &type) {
  if (type == "sigmoid") {
    return lite_api::ActivationType::kSigmoid;
  } else if (type == "sigmoid_v2") {
    return lite_api::ActivationType::kSigmoid_v2;
  } else if (type == "relu") {
    return lite_api::ActivationType::kRelu;
  } else if (type == "tanh") {
    return lite_api::ActivationType::kTanh;
  } else if (type == "tanh_v2") {
    return lite_api::ActivationType::kTanh_v2;
  } else if (type == "identity" || type == "") {
    return lite_api::ActivationType::kIndentity;
  }
  LOG(FATAL) << "The input type is not supported: " << type;
  return lite_api::ActivationType::kIndentity;
}

bool LstmGradOp::InferShapeImpl() const {
  param_.Input_Grad->Resize(param_.Input->dims());
  param_.Weight_Grad->Resize(param_.Weight->dims());
  param_.Bias_Grad->Resize(param_.Bias->dims());

  if (param_.H0) {
      param_.H0_Grad->Resize(param_.H0->dims());
  }
  if (param_.C0) {
      param_.C0_Grad->Resize(param_.C0->dims());
  }

  return true;
}

bool LstmGradOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.Input =
      scope->FindVar(opdesc.Input("Input").front())->GetMutable<lite::Tensor>();
  param_.Weight = scope->FindVar(opdesc.Input("Weight").front())
                      ->GetMutable<lite::Tensor>();
  param_.Bias =
      scope->FindVar(opdesc.Input("Bias").front())->GetMutable<lite::Tensor>();
  param_.Hidden = scope->FindVar(opdesc.Input("Hidden").front())
                      ->GetMutable<lite::Tensor>();
  param_.Cell =
      scope->FindVar(opdesc.Input("Cell").front())->GetMutable<lite::Tensor>();
  param_.BatchGate = scope->FindVar(opdesc.Input("BatchGate").front())
                         ->GetMutable<lite::Tensor>();
  param_.BatchCellPreAct =
      scope->FindVar(opdesc.Input("BatchCellPreAct").front())
          ->GetMutable<lite::Tensor>();
  CHECK(param_.Input);
  CHECK(param_.Weight);
  CHECK(param_.Bias);
  if (opdesc.Input("C0").size()) {
    param_.C0 =
        scope->FindVar(opdesc.Input("C0").front())->GetMutable<lite::Tensor>();
  }
  if (opdesc.Input("H0").size()) {
    param_.H0 =
        scope->FindVar(opdesc.Input("H0").front())->GetMutable<lite::Tensor>();
  }
  param_.use_peepholes = opdesc.GetAttr<bool>("use_peepholes");
  param_.is_reverse = opdesc.GetAttr<bool>("is_reverse");
  param_.gate_activation =
      GetActivationType(opdesc.GetAttr<std::string>("gate_activation"));
  param_.cell_activation =
      GetActivationType(opdesc.GetAttr<std::string>("cell_activation"));
  param_.candidate_activation =
      GetActivationType(opdesc.GetAttr<std::string>("candidate_activation"));

  param_.Input_Grad =
      scope->FindVar(opdesc.Output("Input@GRAD").front())->GetMutable<lite::Tensor>();
  param_.Weight_Grad = scope->FindVar(opdesc.Output("Weight@GRAD").front())
                      ->GetMutable<lite::Tensor>();
  param_.Bias_Grad =
      scope->FindVar(opdesc.Output("Bias@GRAD").front())->GetMutable<lite::Tensor>();
  param_.Hidden_Grad = scope->FindVar(opdesc.Input("Hidden@GRAD").front())
                      ->GetMutable<lite::Tensor>();

  CHECK(param_.Input_Grad);
  CHECK(param_.Weight_Grad);
  CHECK(param_.Bias_Grad);
  CHECK(param_.Hidden_Grad);
  if (opdesc.Input("C0").size()) {
    param_.C0_Grad =
        scope->FindVar(opdesc.Input("C0@GRAD").front())->GetMutable<lite::Tensor>();
  }
  if (opdesc.Input("H0").size()) {
    param_.H0_Grad =
        scope->FindVar(opdesc.Input("H0@GRAD").front())->GetMutable<lite::Tensor>();
  }

  // For int8
  const OpInfo *op_info = dynamic_cast<const OpInfo *>(&opdesc);
  if (op_info != nullptr && op_info->HasAttr("enable_int8") &&
      op_info->GetAttr<bool>("enable_int8")) {
    param_.enable_int8 = true;
    param_.bit_length = opdesc.GetAttr<int>("bit_length");
    std::string weight_scale_name = "Weight0_scale";
    if (op_info->HasInputScale(weight_scale_name, true)) {
      param_.weight_scale = op_info->GetInputScale(weight_scale_name, true);
    }
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(lstm_grad, paddle::lite::operators::LstmGradOp);
