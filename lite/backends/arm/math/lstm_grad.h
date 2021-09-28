/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "lite/backends/arm/math/activation_functions.h"
#include "lite/core/context.h"
#include "lite/utils/cp_logging.h"
#include "lstm.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {


template <class T>
struct LstmMetaGrad {
  T *gate_grad;
  T *prev_state_grad;
  T *state_grad;
  T *state_active_grad;
  T *output_grad;
  T *check_ig_grad;
  T *check_fg_grad;
  T *check_og_grad;
};


template <typename T>
class LstmUnitGradFunctor {
 public:
  static void compute(LstmMetaValue<T> value,
                      LstmMetaGrad<T> grad,
                      int frame_size,
                      int batch_size,
                      T cell_clip,
                      const detail::ActivationType &gate_act,
                      const detail::ActivationType &cell_act,
                      const detail::ActivationType &cand_act);
};


namespace detail {


template <class T, class Op>
void naive_lstm_backward_one_sequence(Op op,
                                      LstmMetaValue<T> value,
                                      LstmMetaGrad<T> grad,
                                      int frame_size,
                                      T cell_clip,
                                      ActivationType active_node,
                                      ActivationType active_gate,
                                      ActivationType active_state) {
  T r_value_in;
  T r_value_ig;
  T r_value_fg;
  T r_value_og;
  T r_grad_in;
  T r_grad_ig;
  T r_grad_fg;
  T r_grad_og;
  T r_prev_state = 0;
  T r_prev_state_grad;
  T r_state;
  T r_state_grad;
  T r_state_atv;
  T r_output_grad;
  T r_checkI;
  T r_checkF;
  T r_checkO;
  T r_checkIGrad;
  T r_checkFGrad;
  T r_checkOGrad;

  T *value_in = value.gate_value;
  T *value_ig = value.gate_value + frame_size;
  T *value_fg = value.gate_value + frame_size * 2;
  T *value_og = value.gate_value + frame_size * 3;
  T *grad_in = grad.gate_grad;
  T *grad_ig = grad.gate_grad + frame_size;
  T *grad_fg = grad.gate_grad + frame_size * 2;
  T *grad_og = grad.gate_grad + frame_size * 3;

  for (int i = 0; i < frame_size; i++) {
    r_value_in = value_in[i];
    r_value_ig = value_ig[i];
    r_value_fg = value_fg[i];
    r_value_og = value_og[i];
    r_checkI = value.check_ig ? value.check_ig[i] : 0;
    r_checkF = value.check_fg ? value.check_fg[i] : 0;
    r_checkO = value.check_og ? value.check_og[i] : 0;
    r_state = value.state_value[i];
    r_state_atv = value.state_active_value[i];
    r_output_grad = grad.output_grad[i];
    r_state_grad = grad.state_grad[i];
    if (value.prev_state_value) {
      r_prev_state = value.prev_state_value[i];
    }

    op(&r_value_in,
       &r_value_ig,
       &r_value_fg,
       &r_value_og,
       &r_grad_in,
       &r_grad_ig,
       &r_grad_fg,
       &r_grad_og,
       &r_prev_state,
       &r_prev_state_grad,
       &r_state,
       &r_state_grad,
       &r_state_atv,
       &r_output_grad,
       &r_checkI,
       &r_checkF,
       &r_checkO,
       &r_checkIGrad,
       &r_checkFGrad,
       &r_checkOGrad,
       &cell_clip,
       active_node,
       active_gate,
       active_state);

    grad_in[i] = r_grad_in;
    grad_ig[i] = r_grad_ig;
    grad_fg[i] = r_grad_fg;
    grad_og[i] = r_grad_og;
    grad.state_grad[i] = r_state_grad;

    if (grad.prev_state_grad) grad.prev_state_grad[i] = r_prev_state_grad;
    if (value.prev_state_value) {
      if (grad.check_ig_grad) grad.check_ig_grad[i] += r_checkIGrad;
      if (grad.check_fg_grad) grad.check_fg_grad[i] += r_checkFGrad;
    }
    if (grad.check_og_grad) grad.check_og_grad[i] += r_checkOGrad;
  }
}

template <class T, class Op>
void cpu_lstm_backward(Op op,
                       LstmMetaValue<T> value,
                       LstmMetaGrad<T> grad,
                       int frame_size,
                       T cell_clip,
                       ActivationType active_node,
                       ActivationType active_gate,
                       ActivationType active_state) {
    naive_lstm_backward_one_sequence<T>(op,
                                        value,
                                        grad,
                                        frame_size,
                                        cell_clip,
                                        active_node,
                                        active_gate,
                                        active_state);
  //}
}

namespace backward {

template <class T>
class lstm {
 public:
  HOSTDEVICE void operator()(T *value_in,
                             T *value_ig,
                             T *value_fg,
                             T *value_og,
                             T *grad_in,
                             T *grad_ig,
                             T *grad_fg,
                             T *grad_og,
                             T *prev_state,
                             T *prev_state_grad,
                             T *state,
                             T *state_grad,
                             T *state_atv,
                             T *output_grad,
                             T *checkI,
                             T *checkF,
                             T *checkO,
                             T *checkIGrad,
                             T *checkFGrad,
                             T *checkOGrad,
                             T *cell_clip,
                             ActivationType active_node,
                             ActivationType active_gate,
                             ActivationType active_state) {
    *grad_og =
        activation((*output_grad) * (*state_atv), *value_og, active_gate);
    if (*cell_clip > 0.0f) {
      if (*state >= (*cell_clip) || *state <= (0.0f - (*cell_clip))) {
        *state_grad = 0.0f;
      } else {
        *state_grad +=
            activation((*output_grad) * (*value_og), *state_atv, active_state) +
            (*grad_og) * (*checkO);
      }
    } else {
      *state_grad +=
          activation((*output_grad) * (*value_og), *state_atv, active_state) +
          (*grad_og) * (*checkO);
    }

    *grad_in = activation((*state_grad) * (*value_ig), *value_in, active_node);
    *grad_ig = activation((*state_grad) * (*value_in), *value_ig, active_gate);
    *grad_fg =
        activation((*state_grad) * (*prev_state), *value_fg, active_gate);
    *prev_state_grad = (*grad_ig) * (*checkI) + (*grad_fg) * (*checkF) +
                       (*state_grad) * (*value_fg);
    *checkIGrad = (*grad_ig) * (*prev_state);
    *checkFGrad = (*grad_fg) * (*prev_state);
    *checkOGrad = (*grad_og) * (*state);
  }
};

}  // namespace backward
}  // namespace detail

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
