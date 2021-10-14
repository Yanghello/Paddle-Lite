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

#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class AccuracyComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string type_ = "accuracy";
  std::string out_ = "out";
  std::string label_ = "label";
  std::string indices_ = "indices";
  std::string accuracy_ = "accuracy";
  std::string correct_ = "correct";
  std::string total_ = "total";
  int n_ = 8192;
  DDim infer_dims_{{n_, 1}};
  DDim indices_dims_{{n_, 1}};
  DDim label_dims_{{n_, 1}};
  DDim out_dims_{{1}};

 public:
  AccuracyComputeTester(const Place& place,
                        const std::string& alias)
            : TestCase(place, alias) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->FindTensor(out_);
    auto* label = scope->FindTensor(label_);
    auto* indices = scope->FindTensor(indices_);

    auto* accuracy = scope->NewTensor(accuracy_);
    auto* correct = scope->NewTensor(correct_);
    auto* total = scope->NewTensor(total_);
    CHECK(accuracy);
    CHECK(correct);
    CHECK(total);
    
    accuracy->Resize(out_dims_);
    correct->Resize(out_dims_);
    total->Resize(out_dims_);

    auto indices_data = indices->data<int64_t>();
    auto label_data = label->data<int64_t>();
    auto* accuracy_data = accuracy->mutable_data<float>();
    auto* correct_data = correct->mutable_data<int32_t>();
    auto* total_data = total->mutable_data<int32_t>();

    int64_t num_correct = 0;
    for (int rowid = 0; rowid < n_; ++rowid) {
      auto ele = indices_data[rowid];
      if (ele == label_data[rowid]) {
        num_correct++;
      }
    }
    *accuracy_data = float(num_correct) / n_;
    *correct_data = num_correct;
    *total_data = n_;
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(type_);
    op_desc->SetInput("Out", {out_});
    op_desc->SetInput("Indices", {indices_});
    op_desc->SetInput("Label", {label_});
    op_desc->SetOutput("Accuracy", {accuracy_});
    op_desc->SetOutput("Correct", {correct_});
    op_desc->SetOutput("Total", {total_});
  }

  void PrepareData() override {
    std::vector<float> infer(infer_dims_.production());
    fill_data_rand(infer.data(), 0.f, 1.f, infer_dims_.production());
    SetCommonTensor(out_, infer_dims_, infer.data());

    std::vector<int64_t> indices(indices_dims_.production());
    fill_data_rand<int64_t>(indices.data(), 0, 2, indices_dims_.production());
    SetCommonTensor<int64_t>(indices_, indices_dims_, indices.data());

    std::vector<int64_t> label(label_dims_.production());
    fill_data_rand<int64_t>(label.data(), 0, 2, label_dims_.production());
    SetCommonTensor(label_, label_dims_, label.data());
  }
};

void TestAccuracy(float abs_error) {
  LOG(INFO) << "run test arm";
  Place place;
  place = {TARGET(kARM), PRECISION(kAny)};
  std::unique_ptr<arena::TestCase> tester(new AccuracyComputeTester(place, "def"));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(Accuracy, precision) {
  LOG(INFO) << "test accuracy op";
  float abs_error = 2e-5;
  TestAccuracy(abs_error);
}

}  // namespace lite
}  // namespace paddle
