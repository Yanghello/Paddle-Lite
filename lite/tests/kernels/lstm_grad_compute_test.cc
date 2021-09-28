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

#include "lite/kernels/arm/lstm_grad_compute.h"
#include <gtest/gtest.h>
#include "lite/core/op_registry.h"
#include "lite/kernels/arm/lstm_compute.h"
#include "lite/tests/utils/fill_data.h"
#include "lite/api/paddle_place.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

using param_t = operators::LstmParam;
using grad_param_t = operators::LstmGradParam;
using kernel_t = LstmCompute<PrecisionType::kFloat>;
using grad_kernel_t = LstmGradCompute<PrecisionType::kFloat>;

inline void init_tensor_from_vector(Tensor& dst, std::vector<float> src) {
  float* data = dst.mutable_data<float>();
  for (int i = 0; i < src.size(); ++i) {
    data[i] = src[i];
  }
}

inline void init_vector_from_tesnor(float* dst, const Tensor& src) {
  //dst.resize(src.numel());
  const float* data = src.data<float>();
  for (int i = 0; i < src.numel(); ++i) {
    dst[i] = data[i];
  }
}

class LstmGradTester {
 public:
  explicit LstmGradTester(const LoD& lod,
                         const size_t D,
                         lite_api::ActivationType act_gate = lite_api::ActivationType::kSigmoid,
                         lite_api::ActivationType act_cell = lite_api::ActivationType::kTanh,
                         lite_api::ActivationType act_cand = lite_api::ActivationType::kTanh,
                         bool has_initial_state = false,
                         bool is_reverse = false,
                         bool use_peepholes = true)
      : lod_(lod),
        D_(D),
        act_gate_(act_gate),
        act_cell_(act_cell),
        act_cand_(act_gate),
        has_initial_state_(has_initial_state),
        is_reverse_(is_reverse),
        use_peepholes_(use_peepholes) {
          T_ = lod_[0].back();
          N_ = lod_[0].size() - 1;
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
                   const std::vector<float>& input_vec,
                   const std::vector<float>& w_vec,
                   const std::vector<float>& h0_vec,
                   const std::vector<float>& c0_vec,
                   const std::vector<float>& b_vec,
                   float* out_hidden_vec,
                   float* out_cell_vec) {
    Tensor input;
    Tensor h0;
    Tensor c0;
    Tensor w;
    Tensor b;
    Tensor output_hidden;
    Tensor output_cell;
    Tensor batch_gate;
    Tensor batch_cell_pre_act;
    DDim input_dim({T_, 4 * D_});
    DDim h0_dim({N_, D_});
    DDim c0_dim({N_, D_});
    DDim w_dim({D_, 4 * D_});
    DDim out_dim({T_, D_});
    DDim b_dim;
    DDim batch_gate_dim({T_, 4 * D_});
    DDim batch_cell_pre_act_dim({T_, D_});
    if (use_peepholes_) {
      b_dim = DDim({1, 7 * D_});
    } else {
      b_dim = DDim({1, 4 * D_});
    }

    input.Resize(input_dim);
    h0.Resize(h0_dim);
    c0.Resize(c0_dim);
    w.Resize(w_dim);
    b.Resize(b_dim);
    output_hidden.Resize(out_dim);
    output_cell.Resize(out_dim);
    batch_cell_pre_act.Resize(batch_cell_pre_act_dim);
    batch_gate.Resize(batch_gate_dim);

    init_tensor_from_vector(h0, h0_vec);

    init_tensor_from_vector(c0, c0_vec);

    init_tensor_from_vector(b, b_vec);
  
    init_tensor_from_vector(input, input_vec);

    init_tensor_from_vector(w, w_vec);

    input.set_lod(lod_);
    output_hidden.set_lod(lod_);
    output_cell.set_lod(lod_);
    //batch_gate.set_lod(lod_);

    param->Input = &input;
    param->Weight = &w;
    param->Bias = &b;
    param->Hidden = &output_hidden;
    param->Cell = &output_cell;
    param->H0 = &h0;
    param->C0 = &c0;
    param->BatchGate = &batch_gate;
    param->BatchCellPreAct = & batch_cell_pre_act;
    param->use_peepholes = use_peepholes_;
    param->is_reverse = is_reverse_;
    param->gate_activation = act_gate_;
    param->cell_activation = act_cell_;
    param->candidate_activation = act_cand_;
    kernel->SetParam(*param);
    kernel->Launch();

    init_vector_from_tesnor(out_hidden_vec, output_hidden);
    init_vector_from_tesnor(out_cell_vec, output_cell);
    batch_lod_ = batch_gate.lod();
  }

  void run_backward(grad_param_t* param,
                    grad_kernel_t* kernel,
                    const std::vector<float>& input_vec,
                    const std::vector<float>& w_vec,
                    const std::vector<float>& b_vec,
                    const std::vector<float>& hidden_vec,
                    const std::vector<float>& cell_vec,
                    const std::vector<float>& h0_vec,
                    const std::vector<float>& c0_vec,
                    const std::vector<float>& out_hidden_grad_vec,
                    //const std::vector<float>& out_cell_grad_vec,
                    float* input_grad_vec,
                    float* w_grad_vec,
                    float* b_grad_vec,
                    float* h0_grad_vec,
                    float* c0_grad_vec) {
    Tensor input;
    Tensor h0;
    Tensor c0;
    Tensor w;
    Tensor b;
    Tensor hidden;
    Tensor cell;
    Tensor hidden_grad;
    //Tensor cell_grad;
    Tensor out_input_grad;
    Tensor out_w_grad;
    Tensor out_b_grad;
    Tensor out_h0_grad;
    Tensor out_c0_grad;

    Tensor batch_gate;
    Tensor batch_cell_pre_act;
    
    DDim input_dim({T_, 4 * D_});
    DDim h0_dim({N_, D_});
    DDim c0_dim({N_, D_});
    DDim w_dim({D_, 4 * D_});
    DDim out_dim({T_, D_});
    DDim b_dim;
    if (use_peepholes_) {
      b_dim = DDim({1, 7 * D_});
    } else {
      b_dim = DDim({1, 4 * D_});
    }

    DDim batch_gate_dim({T_, 4 * D_});
    DDim batch_cell_pre_act_dim({T_, D_});

    input.Resize(input_dim);
    h0.Resize(h0_dim);
    c0.Resize(c0_dim);
    w.Resize(w_dim);
    b.Resize(b_dim);
    hidden.Resize(out_dim);
    cell.Resize(out_dim);
    hidden_grad.Resize(out_dim);
    //cell_grad.Resize(out_dim);
    out_input_grad.Resize(input_dim);
    out_w_grad.Resize(w_dim);
    out_b_grad.Resize(b_dim);
    out_h0_grad.Resize(h0_dim);
    out_c0_grad.Resize(c0_dim);

    batch_cell_pre_act.Resize(batch_cell_pre_act_dim);
    batch_gate.Resize(batch_gate_dim);

    init_tensor_from_vector(input, input_vec);
    init_tensor_from_vector(h0, h0_vec);
    init_tensor_from_vector(c0, c0_vec);
    init_tensor_from_vector(w, w_vec);
    init_tensor_from_vector(b, b_vec);
    init_tensor_from_vector(cell, cell_vec);
    init_tensor_from_vector(hidden, hidden_vec);
    init_tensor_from_vector(hidden_grad, out_hidden_grad_vec);
    //init_tensor_from_vector(cell_grad, out_cell_grad_vec);

    input.set_lod(lod_);
    hidden.set_lod(lod_);
    cell.set_lod(lod_);
    batch_gate.set_lod(batch_lod_);

    param->Input = &input;
    param->Weight = &w;
    param->Bias = &b;
    param->Hidden = &hidden;
    param->Cell = &cell;
    param->H0 = &h0;
    param->C0 = &c0;
    param->Hidden_Grad = &hidden_grad;
    param->Input_Grad = &out_input_grad;
    param->Weight_Grad = &out_w_grad;
    param->Bias_Grad = &out_b_grad;
    param->H0_Grad = &out_h0_grad;
    param->C0_Grad = & out_c0_grad;
    param->BatchGate = &batch_gate;
    param->BatchCellPreAct = & batch_cell_pre_act;
    param->use_peepholes = use_peepholes_;
    param->is_reverse = is_reverse_;
    param->gate_activation = act_gate_;
    param->cell_activation = act_cell_;
    param->candidate_activation = act_cand_;
    kernel->SetParam(*param);
    kernel->Launch();

    init_vector_from_tesnor(input_grad_vec, out_input_grad);
    init_vector_from_tesnor(w_grad_vec, out_w_grad);
    init_vector_from_tesnor(b_grad_vec, out_b_grad);
    init_vector_from_tesnor(h0_grad_vec, out_h0_grad);
    init_vector_from_tesnor(c0_grad_vec, out_c0_grad);
  }

  void check_grad() {
    DDim input_dim({T_, 4 * D_});
    DDim h0_dim({N_, D_});
    DDim c0_dim({N_, D_});
    DDim w_dim({D_, 4 * D_});
    DDim out_dim({T_, D_});
    DDim b_dim;
    if (use_peepholes_) {
      b_dim = DDim({1, 7 * D_});
    } else {
      b_dim = DDim({1, 4 * D_});
    }

    // forward
    std::vector<float> input(input_dim.production());
    std::vector<float> w(w_dim.production());
    std::vector<float> h0(h0_dim.production());
    std::vector<float> c0(c0_dim.production());
    std::vector<float> b(b_dim.production());
    std::vector<float> out_hidden(out_dim.production());
    std::vector<float> out_cell(out_dim.production());
    fill_data_rand(input.data(), -1.f, 1.f, input_dim.production());
    fill_data_rand(w.data(), -1.f, 1.f, w_dim.production());
    if (has_initial_state_) {
      fill_data_rand(h0.data(), -1.f, 1.f, h0_dim.production());
      fill_data_rand(c0.data(), -1.f, 1.f, c0_dim.production());
    } else {
      fill_data_const(h0.data(), 0.0f, h0_dim.production());
      fill_data_const(c0.data(), 0.0f, c0_dim.production());
    }

    if (use_peepholes_) {
      fill_data_rand(b.data(), -1.f, 1.f, b_dim.production());
    } else {
      fill_data_rand(b.data(), -1.f, 1.f, b_dim.production());
    }
    
    this->run_forward(&param_, &kernel_, input, w, h0, c0, b, out_hidden.data(), out_cell.data());

    // backward
    std::vector<float> out_hidden_grad(out_dim.production());
    std::vector<float> input_grad(input_dim.production());
    std::vector<float> w_grad(w_dim.production());
    std::vector<float> h0_grad(h0_dim.production());
    std::vector<float> c0_grad(c0_dim.production());
    std::vector<float> b_grad(b_dim.production());
    for (int i = 0; i < out_dim.production(); i++) {
      out_hidden_grad[i] = 1.0;
    }
    this->run_backward(&grad_param_,
                       &grad_kernel_,
                       input,
                       w,
                       b,
                       out_hidden,
                       out_cell,
                       h0,
                       c0,
                       out_hidden_grad,
                       input_grad.data(),
                       w_grad.data(),
                       b_grad.data(),
                       h0_grad.data(),
                       c0_grad.data());

    // get numeric gradient
    std::vector<float> input_delta(input_dim.production());
    std::vector<float> w_delta(w_dim.production());
    std::vector<float> b_delta(b_dim.production());
    std::vector<float> h0_delta(h0_dim.production());
    std::vector<float> c0_delta(c0_dim.production());

    float delta = 0.001;
    float max_grad_delta = 0.0055;

    // check input grad
    std::vector<float> out_hidden_delta(out_hidden.size());
    std::vector<float> out_cell_delta(out_cell.size());
    for (int i = 0; i < input_dim.production(); i++) {
      for (int j = 0; j < input_dim.production(); j++) {
        if (i == j) {
          input_delta[j] = input[j] + delta;
        } else {
          input_delta[j] = input[j];
        }
      }
      this->run_forward(
          &delta_param_, &delta_kernel_, input_delta, w, h0, c0, b,
          out_hidden_delta.data(), out_cell_delta.data());

      float sum = 0;
      for (int j = 0; j < out_dim.production(); j++) {
        sum += (out_hidden_delta[j] - out_hidden[j]);
      }

      EXPECT_NEAR(input_grad[i], sum / delta, max_grad_delta);
    }

    // check w grad
    for (int i = 0; i < w_dim.production(); i++) {
      for (int j = 0; j < w_dim.production(); j++) {
        if (i == j) {
          w_delta[j] = w[j] + delta;
        } else {
          w_delta[j] = w[j];
        }
      }
      this->run_forward(
          &delta_param_, &delta_kernel_, input, w_delta, h0, c0, b,
          out_hidden_delta.data(), out_cell_delta.data());

      float sum = 0;
      for (int j = 0; j < out_dim.production(); j++) {
        sum += (out_hidden_delta[j] - out_hidden[j]);
      }

      EXPECT_NEAR(w_grad[i], sum / delta, max_grad_delta);
    }

    // check h0 grad
    for (int i = 0; i < h0_dim.production(); i++) {
      for (int j = 0; j < h0_dim.production(); j++) {
        if (i == j) {
          h0_delta[j] = h0[j] + delta;
        } else {
          h0_delta[j] = h0[j];
        }
      }
      this->run_forward(
          &delta_param_, &delta_kernel_, input, w, h0_delta, c0, b,
          out_hidden_delta.data(), out_cell_delta.data());

      float sum = 0;
      for (int j = 0; j < out_dim.production(); j++) {
        sum += (out_hidden_delta[j] - out_hidden[j]);
      }

      EXPECT_NEAR(h0_grad[i], sum / delta, max_grad_delta);
    }

    // check c0 grad
    for (int i = 0; i < c0_dim.production(); i++) {
      for (int j = 0; j < c0_dim.production(); j++) {
        if (i == j) {
          c0_delta[j] = c0[j] + delta;
        } else {
          c0_delta[j] = c0[j];
        }
      }
      this->run_forward(
          &delta_param_, &delta_kernel_, input, w, h0, c0_delta, b,
          out_hidden_delta.data(), out_cell_delta.data());

      float sum = 0;
      for (int j = 0; j < out_dim.production(); j++) {
        sum += (out_hidden_delta[j] - out_hidden[j]);
      }

      EXPECT_NEAR(c0_grad[i], sum / delta, max_grad_delta);
    }

    // check b grad
    for (int i = 0; i < b_dim.production(); i++) {
      for (int j = 0; j < b_dim.production(); j++) {
        if (i == j) {
          b_delta[j] = b[j] + delta;
        } else {
          b_delta[j] = b[j];
        }
      }
      this->run_forward(
          &delta_param_, &delta_kernel_, input, w, h0, c0, b_delta,
          out_hidden_delta.data(), out_cell_delta.data());

      float sum = 0;
      for (int j = 0; j < out_dim.production(); j++) {
        sum += (out_hidden_delta[j] - out_hidden[j]);
      }

      EXPECT_NEAR(b_grad[i], sum / delta, max_grad_delta);
    }

  }

 private:
  LoD lod_;
  LoD batch_lod_;
  int64_t D_;
  int64_t T_;
  int64_t N_;
  lite_api::ActivationType act_gate_;
  lite_api::ActivationType act_cell_;
  lite_api::ActivationType act_cand_;
  bool has_initial_state_;
  bool is_reverse_;
  bool use_peepholes_;
  kernel_t kernel_;
  kernel_t delta_kernel_;
  grad_kernel_t grad_kernel_;
  param_t param_;
  param_t delta_param_;
  grad_param_t grad_param_;
};

void TestNormalCase(const std::vector<std::vector<uint64_t>>& lod,
                    const int64_t D = 16,
                    lite_api::ActivationType act_gate = lite_api::ActivationType::kSigmoid,
                    lite_api::ActivationType act_cell = lite_api::ActivationType::kTanh,
                    lite_api::ActivationType act_cand = lite_api::ActivationType::kTanh,
                    bool has_initial_state = false,
                    bool is_reverse = false,
                    bool use_peepholes = true) {
  std::unique_ptr<LstmGradTester> tester(new LstmGradTester(
        lod,
        D,
        act_gate,
        act_cell,
        act_cand,
        has_initial_state,
        is_reverse,
        use_peepholes));

  tester->prepare_kernel();

  tester->check_grad();
}

TEST(lstm_grad_arm, compute) {
  LOG(INFO) << "Test Lstm grad";
  DeviceInfo::Init();
  TestNormalCase({{0, 3, 5}});
  //TestNormalCase({{0, 3, 3}});
  //TestNormalCase({{0, 2, 2, 6}});
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(lstm, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(lstm_grad, kARM, kFloat, kNCHW, def);
