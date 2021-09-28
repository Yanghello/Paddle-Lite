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

#include "lite/kernels/arm/lstm_grad_compute.h"
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/lstm.h"
#include "lite/backends/arm/math/sequence2batch.h"
#include "lite/backends/arm/math/sgemm.h"
#include "lite/backends/arm/math/lstm_grad.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

using LoDTensor = lite::Tensor;

void LSTMGradComputeRun(const operators::LstmGradParam& param,
                    ARMContext* ctx,
                    bool enable_int8) {
  auto input = param.Input;
  auto weight = param.Weight;
  auto bias = param.Bias;
  auto hidden_out = param.Hidden;
  auto cell_out = param.Cell;

  //auto hidden_t0 = param.H0;
  //auto cell_t0 = param.C0;
  auto batch_gate = param.BatchGate;
  auto batch_cell_pre_act = param.BatchCellPreAct;

  auto hidden_g = param.Hidden_Grad;
  auto in_g = param.Input_Grad;
  auto weight_g = param.Weight_Grad;
  auto bias_g = param.Bias_Grad;

  auto h0 = param.H0;
  auto c0 = param.C0;
  auto h0_g = param.H0_Grad;
  auto c0_g = param.C0_Grad;

  math::SetConstant<float> zero;

  if (weight_g) {
    weight_g->mutable_data<float>();
    zero(weight_g, 0.0f);
  }

  // ordered_h0/c0 is the reordered hidden/cell initialization.
  // ordered_h0_g/c0_g is the reordered gradient of hidden/cell
  // initialization.
  Tensor ordered_h0, ordered_c0, ordered_h0_g, ordered_c0_g;
  std::vector<uint64_t> order(batch_gate->lod()[2]);

  if (c0) {
    lite::arm::math::ReorderInitState<float>(*c0, order, &ordered_c0, true);
  }

  if (c0 && c0_g) {
    ordered_c0_g.mutable_data<float>();
    ordered_c0_g.Resize(c0_g->dims());
  }

  auto in_dims = input->dims();
  auto out_dims = hidden_g->dims();
  int frame_size = static_cast<int>(in_dims[1] / 4);

  lite::arm::math::LstmMetaValue<float> lstm_value;
  if (bias && param.use_peepholes) {
    float* bias_data = const_cast<float*>(bias->data<float>());
    lstm_value.check_ig = bias_data + 4 * frame_size;
    lstm_value.check_fg = lstm_value.check_ig + frame_size;
    lstm_value.check_og = lstm_value.check_fg + frame_size;
  } else {
    lstm_value.check_ig = nullptr;
    lstm_value.check_fg = nullptr;
    lstm_value.check_og = nullptr;
  }

  lite::arm::math::LstmMetaGrad<float> lstm_grad;

  if (bias && bias_g) {
    bias_g->mutable_data<float>();
    zero(bias_g, 0.0f);
  }
  if (bias && bias_g && param.use_peepholes) {
    float* bias_g_data = bias_g->mutable_data<float>();
    lstm_grad.check_ig_grad = bias_g_data + 4 * frame_size;
    lstm_grad.check_fg_grad = lstm_grad.check_ig_grad + frame_size;
    lstm_grad.check_og_grad = lstm_grad.check_fg_grad + frame_size;
  } else {
    lstm_grad.check_ig_grad = nullptr;
    lstm_grad.check_fg_grad = nullptr;
    lstm_grad.check_og_grad = nullptr;
  }

  lite::arm::math::LoDTensor2BatchFunctor<float> to_batch;

  auto ToBatch = [&batch_gate, &to_batch](
      const LoDTensor& src,
      const paddle::lite::DDimLite& dims, LoDTensor& dst) {
    dst.mutable_data<float>();
    dst.Resize(dims);
    dst.set_lod(batch_gate->lod());
    to_batch(src, &dst, false);
  };

  LoDTensor batch_hidden, batch_hidden_g, batch_cell;
  ToBatch(*hidden_out, out_dims, batch_hidden);
  ToBatch(*hidden_g, out_dims, batch_hidden_g);
  ToBatch(*cell_out, out_dims, batch_cell);

  LoDTensor batch_cell_g, batch_gate_g;
  batch_cell_g.mutable_data<float>();
  batch_cell_g.Resize(out_dims);
  // TODO(qingqing) support the case output cell has gradient.
  // to_batch(device_ctx, *cell_g, batch_cell_g, false);
  zero(&batch_cell_g, static_cast<float>(0.0));
  batch_gate_g.mutable_data<float>();
  batch_gate_g.Resize(batch_gate->dims());
  batch_gate_g.set_lod(batch_gate->lod());

  auto gate_act = lite::arm::math::detail::GetActivationType(param.gate_activation);
  auto cell_act = lite::arm::math::detail::GetActivationType(param.cell_activation);
  auto cand_act = lite::arm::math::detail::GetActivationType(param.candidate_activation);

  auto batch_starts = batch_gate->lod()[0];
  size_t num_batch = batch_starts.size() - 1;
  //auto blas = math::GetBlas<DeviceContext, T>(device_ctx);
  for (int n = static_cast<int>(num_batch) - 1; n >= 0; n--) {
    int bstart = static_cast<int>(batch_starts[n]);
    int bend = static_cast<int>(batch_starts[n + 1]);

    Tensor gate = batch_gate->Slice<float>(bstart, bend);
    Tensor cell = batch_cell.Slice<float>(bstart, bend);
    Tensor cell_pre_act = batch_cell_pre_act->Slice<float>(bstart, bend);
    lstm_value.gate_value = gate.mutable_data<float>();
    lstm_value.state_value = cell.mutable_data<float>();
    lstm_value.state_active_value = cell_pre_act.mutable_data<float>();

    Tensor out_g = batch_hidden_g.Slice<float>(bstart, bend);
    Tensor gate_g = batch_gate_g.Slice<float>(bstart, bend);
    Tensor cell_g = batch_cell_g.Slice<float>(bstart, bend);
    lstm_grad.state_grad = cell_g.mutable_data<float>();
    lstm_grad.gate_grad = gate_g.mutable_data<float>();
    lstm_grad.output_grad = out_g.mutable_data<float>();

    if (n > 0) {
      int bstart_pre = static_cast<int>(batch_starts[n - 1]);
      Tensor cell_pre = batch_cell.Slice<float>(bstart_pre, bstart);
      Tensor cell_pre_g = batch_cell_g.Slice<float>(bstart_pre, bstart);
      lstm_value.prev_state_value = cell_pre.mutable_data<float>();
      lstm_grad.prev_state_grad = cell_pre_g.mutable_data<float>();
    } else {
      lstm_value.prev_state_value = c0 ? ordered_c0.mutable_data<float>() : nullptr;
      lstm_grad.prev_state_grad = c0_g ? ordered_c0_g.mutable_data<float>() : nullptr;
    }

    // lstm_value.output_value not used in bp, set to nullptr
    // lstm_grad.state_active_grad not used in bp, set to nullptr
    lstm_value.output_value = nullptr;
    lstm_grad.state_active_grad = nullptr;
    int cur_batch_size = bend - bstart;
    float cell_clip = 0.0;
    lite::arm::math::LstmUnitGradFunctor<float>::compute(
        lstm_value, lstm_grad, frame_size, cur_batch_size,
        cell_clip, gate_act, cell_act, cand_act);

    std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
    auto& ctx_ = ctx1->As<paddle::lite::ARMContext>();
    math::TensorMatMul mat_mul(&ctx_);
    if (n > 0) {
      int pre_h_start = static_cast<int>(batch_starts[n - 1]);
      int pre_h_end = pre_h_start + cur_batch_size;
      auto pre_hidden_g = batch_hidden_g.Slice<float>(pre_h_start, pre_h_end);
      mat_mul(gate_g, false, *weight, true, static_cast<float>(1.0),
                  &pre_hidden_g, static_cast<float>(1.0));
      
      if (weight_g) {
        /* backward weight */
        auto pre_hidden = batch_hidden.Slice<float>(pre_h_start, pre_h_end);
        pre_hidden.mutable_data<float>();
        mat_mul(pre_hidden, true, gate_g, false, static_cast<float>(1.0),
                    weight_g, static_cast<float>(1.0));
      }
    } else {
      if (h0 && weight_g) {
        lite::arm::math::ReorderInitState<float>(*h0, order,
                                            &ordered_h0, true);
        mat_mul(ordered_h0, true, gate_g, false, static_cast<float>(1.0),
                    weight_g, static_cast<float>(1.0));
      }
      if (h0 && h0_g) {
        ordered_h0_g.mutable_data<float>();
        ordered_h0_g.Resize(h0_g->dims());
        mat_mul(gate_g, false, *weight, true, static_cast<float>(1.0),
                    &ordered_h0_g, static_cast<float>(0.0));
      }
    }
  }
  
  lite::arm::math::Batch2LoDTensorFunctor<float> to_seq;
  if (in_g) {
    /* backward data */
    in_g->mutable_data<float>();
    to_seq(batch_gate_g, in_g);
  }
  if (bias && bias_g) {
    /* backward bias */
    Tensor b_g = *bias_g;
    b_g.Resize({bias_g->numel(), 1});
    Tensor gate_bias_g = b_g.Slice<float>(0, 4 * frame_size);
    math::ColwiseSum<float> col_sum;
    col_sum(batch_gate_g, &gate_bias_g);
  }

  if (h0 && h0_g) {
    lite::arm::math::ReorderInitState<float>(ordered_h0_g, order, h0_g,
                                        false);
  }
  if (c0 && c0_g) {
    lite::arm::math::ReorderInitState<float>(ordered_c0_g, order, c0_g,
                                        false);
  }
  /*
  std::vector<float> weight_scale{};
  int bit_length{};
  if (enable_int8) {
    CHECK(param.enable_int8);
    CHECK_EQ(weight->dims().size(), 2);
    CHECK_EQ(param.weight_scale.size(), weight->dims()[1]);
    weight_scale = param.weight_scale;
    bit_length = param.bit_length;
  }

  batch_gate->mutable_data<float>();
  hidden_out->mutable_data<float>();
  cell_out->mutable_data<float>();

  bool is_reverse = param.is_reverse;
  lite::arm::math::LoDTensor2BatchFunctor<float> to_batch;
  to_batch(*input, batch_gate, true, is_reverse);

  auto in_dims = input->dims();
  int frame_size = static_cast<int>(in_dims[1] / 4);
  DDimLite dims(std::vector<int64_t>{in_dims[0], frame_size});

  if (bias) {
    // checkpoint1
    lite::arm::math::add_bias_rowwise(batch_gate, bias, 0, 4 * frame_size);
  }

  lite::arm::math::LstmMetaValue<float> lstm_value;
  if (bias && param.use_peepholes) {
    float* bias_data = const_cast<float*>(bias->data<float>());
    // the code style in LstmMetaValue will be updated later.
    lstm_value.check_ig = bias_data + 4 * frame_size;
    lstm_value.check_fg = lstm_value.check_ig + frame_size;
    lstm_value.check_og = lstm_value.check_fg + frame_size;
  } else {
    lstm_value.check_ig = nullptr;
    lstm_value.check_fg = nullptr;
    lstm_value.check_og = nullptr;
  }
  lstm_value.prev_state_value = nullptr;
  Tensor ordered_c0;

  std::vector<uint64_t> order(batch_gate->lod()[2]);

  if (cell_t0) {
    // Since the batch computing for LSTM reorders the input sequence
    // according to their length. The initialized cell state also needs
    // to reorder.
    lite::arm::math::ReorderInitState<float>(
        *cell_t0, order, &ordered_c0, true);
    lstm_value.prev_state_value = ordered_c0.mutable_data<float>();
  }
  // Use the local variable as here.
  Tensor batch_hidden, batch_cell;
  batch_hidden.Resize(dims);
  batch_cell.Resize(dims);
  batch_cell_pre_act->Resize(dims);
  batch_hidden.mutable_data<float>();
  batch_cell.mutable_data<float>();
  batch_cell_pre_act->mutable_data<float>();

  auto batch_starts = batch_gate->lod()[0];
  size_t num_batch = batch_starts.size() - 1;

  lite_api::ActivationType gate_act = param.gate_activation;
  lite_api::ActivationType cell_act = param.cell_activation;
  lite_api::ActivationType cand_act = param.candidate_activation;

  int matrix_width = batch_gate->numel() / in_dims[0];
  for (size_t n = 0; n < num_batch; n++) {
    int bstart = static_cast<int>(batch_starts[n]);
    int bend = static_cast<int>(batch_starts[n + 1]);
    auto gate_t = lite::arm::math::row_offset(*batch_gate, bstart);
    auto out_t = lite::arm::math::row_offset(batch_hidden, bstart);
    auto cell_t = lite::arm::math::row_offset(batch_cell, bstart);
    auto cell_pre_act_t =
        lite::arm::math::row_offset(*batch_cell_pre_act, bstart);

    int cur_batch_size = bend - bstart;
    operators::ActivationParam act_param;
    act_param.has_active = false;
    lite_api::ActivationType act_type;

    if (n > 0) {
      int pre_h_start = static_cast<int>(batch_starts[n - 1]);
      int pre_h_end = pre_h_start + cur_batch_size;

      auto pre_hidden_t =
          lite::arm::math::row_offset(batch_hidden, pre_h_start);
      int M = pre_h_end - pre_h_start;
      int N = matrix_width;
      int K = frame_size;

      if (enable_int8) {
        // quantize Ht-1
        int pre_hidden_size = M * K;
        float threshold =
            lite::arm::math::FindAbsMax(pre_hidden_t, pre_hidden_size);
        float pre_hidden_scale =
            lite::arm::math::GetScale(threshold, bit_length);
        std::unique_ptr<int8_t[]> pre_hidden_int8(new int8_t[pre_hidden_size]);
        lite::arm::math::QuantizeTensor(pre_hidden_t,
                                        pre_hidden_int8.get(),
                                        pre_hidden_size,
                                        pre_hidden_scale);
        // update scales
        std::vector<float> scales(M, weight_scale[0]);
        for (auto&& x : scales) {
          x *= pre_hidden_scale;
        }

        operators::ActivationParam act_param;
        act_param.has_active = false;

        std::unique_ptr<float[]> o_data(new float[M * N]);
        lite::arm::math::gemm_s8(false,
                                 false,
                                 M,
                                 N,
                                 K,
                                 pre_hidden_int8.get(),
                                 weight->data<int8_t>(),
                                 o_data.get(),
                                 nullptr,
                                 false,
                                 scales.data(),
                                 act_param,
                                 ctx);

        for (int i = 0; i < M * N; i++) {
          gate_t[i] += o_data[i];
        }
      } else {
        lite::arm::math::sgemm(false,
                               false,
                               M,
                               N,
                               K,
                               1,
                               pre_hidden_t,
                               K,
                               weight->data<float>(),
                               N,
                               1,
                               gate_t,
                               N,
                               nullptr,
                               false,
                               act_param,
                               ctx);
      }
    } else if (hidden_t0) {
      // If n == 0 and there is no initialized hidden state, that is to say
      // the H0 is zeros, the calculation W_h * H0 will be skiped.
      // If n == 0 and there is initialized hidden state, calculate W_h * H0.
      // Since the batch computing for LSTM reorders the input sequence
      // according to their length. The initialized hidden state also needs
      // to reorder.
      Tensor ordered_h0;
      lite::arm::math::ReorderInitState<float>(
          *hidden_t0, order, &ordered_h0, true);
      int M = ordered_h0.dims()[0];
      int N = matrix_width;
      int K = frame_size;
      lite::arm::math::sgemm(false,
                             false,
                             M,
                             N,
                             K,
                             1,
                             ordered_h0.data<float>(),
                             K,
                             weight->data<float>(),
                             N,
                             1,
                             gate_t,
                             N,
                             nullptr,
                             false,
                             act_param,
                             ctx);
    }

    lstm_value.gate_value = gate_t;
    lstm_value.output_value = out_t;
    lstm_value.state_value = cell_t;
    lstm_value.state_active_value = cell_pre_act_t;
    float cell_clip = 0.0;
    // checkpoint
    lite::arm::math::LstmUnitFunctor<float>::compute(lstm_value,
                                                     frame_size,
                                                     cur_batch_size,
                                                     cell_clip,
                                                     cand_act,
                                                     gate_act,
                                                     cell_act,
                                                     ctx->threads());
    lstm_value.prev_state_value = lstm_value.state_value;
  }

  lite::arm::math::Batch2LoDTensorFunctor<float> to_seq;
  auto* lod_hidden = batch_hidden.mutable_lod();
  *lod_hidden = batch_gate->lod();
  to_seq(batch_hidden, hidden_out);
  auto* lod_cell = batch_cell.mutable_lod();
  *lod_cell = batch_gate->lod();
  to_seq(batch_cell, cell_out);
  */
}

template <>
void LstmGradCompute<PRECISION(kFloat)>::Run() {
  auto& param = this->Param<operators::LstmGradParam>();
  auto& ctx = this->ctx_->As<ARMContext>();
  LSTMGradComputeRun(param, &ctx, false);
}

/*template <>
void LstmGradCompute<PRECISION(kInt8)>::Run() {
  auto& param = this->Param<operators::LstmGradParam>();
  auto& ctx = this->ctx_->As<ARMContext>();
  LSTMGradComputeRun(param, &ctx, true);
}*/

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(lstm_grad,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::LstmGradCompute<PRECISION(kFloat)>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("C0", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Hidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Cell", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("BatchGate", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("BatchCellPreAct", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Input_Grad", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Weight_Grad", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Bias_Grad", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("C0_Grad", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("H0_Grad", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Hidden_Grad", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

/*REGISTER_LITE_KERNEL(lstm,
                     kARM,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::arm::LstmCompute<PRECISION(kInt8)>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Weight",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("C0", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Cell", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("BatchCellPreAct", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
*/