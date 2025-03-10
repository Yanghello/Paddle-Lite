/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/backends/opencl/cl_utility.h"
#include "lite/backends/opencl/cl_wrapper.h"
#include "lite/utils/io.h"

typedef enum {
  UNKNOWN = 0,
  QUALCOMM_ADRENO = 1,
  ARM_MALI = 2,
  IMAGINATION_POWERVR = 3,
  OTHERS = 4,
} GpuType;

typedef enum {
  CL_VER_UNKNOWN = 0,
  CL_VER_1_0 = 1,
  CL_VER_1_1 = 2,
  CL_VER_1_2 = 3,
  CL_VER_2_0 = 4,
  CL_VER_2_1 = 5
} OpenCLVersion;

typedef enum {
  PERF_DEFAULT = 0,
  PERF_LOW = 1,
  PERF_NORMAL = 2,
  PERF_HIGH = 3
} GPUPerfMode;

typedef enum {
  PRIORITY_DEFAULT = 0,
  PRIORITY_LOW = 1,
  PRIORITY_NORMAL = 2,
  PRIORITY_HIGH = 3
} GPUPriorityLevel;

// Adreno extensions
// Adreno performance hints
typedef cl_uint cl_perf_hint;
#define CL_CONTEXT_PERF_MODE_QCOM 0x40C2
#define CL_PERF_MODE_HIGH_QCOM 0x40C3
#define CL_PERF_MODE_NORMAL_QCOM 0x40C4
#define CL_PERF_MODE_LOW_QCOM 0x40C5

// Adreno priority hints
typedef cl_uint cl_priority_hint;

#define CL_PRIORITY_HINT_NONE_QCOM 0
#define CL_CONTEXT_PRIORITY_LEVEL_QCOM 0x40C9
#define CL_PRIORITY_HINT_HIGH_QCOM 0x40CA
#define CL_PRIORITY_HINT_NORMAL_QCOM 0x40CB
#define CL_PRIORITY_HINT_LOW_QCOM 0x40CC

namespace paddle {
namespace lite {

extern const std::map<std::string, std::vector<unsigned char>>
    opencl_kernels_files;

class CLRuntime {
 public:
  static CLRuntime* Global();

  bool support_half() {
    return static_cast<bool>(device_info_["CL_DEVICE_EXTENSIONS_FP16"]);
  }

  bool OpenCLAvaliableForDevice(bool check_fp16_valid = false) {
// note(ysh329): entered this func means:
//  1. opencl_lib_found must be true
//  2. dlsym_success must be true
#ifdef LITE_WITH_LOG
    LOG(INFO) << "check_fp16_valid:" << check_fp16_valid;
#endif
    if (!paddle::lite::CLWrapper::Global()->OpenclLibFound() ||
        !paddle::lite::CLWrapper::Global()->DlsymSuccess()) {
      LOG(ERROR) << "Invalid opencl device, OpenclLibFound:"
                 << paddle::lite::CLWrapper::Global()->OpenclLibFound()
                 << ", DlsymSuccess:"
                 << paddle::lite::CLWrapper::Global()->DlsymSuccess();
      return false;
    }
    if (device_info_.count("CL_DEVICE_TYPE") == 0) {
      LOG(ERROR) << "Invalid opencl device, CL_DEVICE_TYPE is None.";
      return false;
    }

    bool support_fp16 = support_half();
    is_device_avaliable_for_opencl_ =
        check_fp16_valid ? support_fp16 : is_device_avaliable_for_opencl_;
    return is_device_avaliable_for_opencl_;
  }

  void set_auto_tune(lite_api::CLTuneMode tune_mode,
                     const std::string& path,
                     const std::string& name,
                     size_t lws_repeats = 4);

  lite_api::CLTuneMode auto_tune() { return auto_tune_; }

  size_t lws_repeats() { return lws_repeats_; }
  bool tune_file_flag() { return have_tune_file_flag_; }
  void set_del_flag() { del_tune_bin_flag_ = true; }

  void set_precision(
      lite_api::CLPrecisionType p = lite_api::CL_PRECISION_AUTO) {
    // CL_PRECISION_AUTO: 0
    // CL_PRECISION_FP32: 1
    // CL_PRECISION_FP16: 2
    if ((lite_api::CL_PRECISION_AUTO == p ||
         lite_api::CL_PRECISION_FP16 == p) &&
        support_half()) {
      precision_ = lite_api::CL_PRECISION_FP16;
    } else if (lite_api::CL_PRECISION_AUTO == p ||
               lite_api::CL_PRECISION_FP32 == p) {
      precision_ = lite_api::CL_PRECISION_FP32;
    } else {
      LOG(FATAL) << "unsupported precision for opencl:"
                 << static_cast<size_t>(p);
    }
  }

  lite_api::CLPrecisionType get_precision() { return precision_; }

  void SetBinaryPathName(const std::string& path, const std::string& name) {
    binary_path_name_.clear();
    binary_path_name_.push_back(path);
    binary_path_name_.push_back(name);
  }

  std::vector<std::string> GetBinaryPathName() const {
    return binary_path_name_;
  }

  void Flush(const int index);

  bool Init();

  cl::Platform& platform();

  cl::Context& context();

  cl::Device& device();

  std::map<std::string, std::unique_ptr<cl::Program>>& program_map();

  cl::CommandQueue& command_queue();

  cl::Program& GetProgram(const std::string& file_name,
                          const std::string& options);

  std::unique_ptr<cl::Program> CreateProgramFromSource(
      const cl::Context& context, std::string file_name);

  bool CheckFromCache(const std::string& program_key);

  bool CheckFromPrecompiledBinary(const std::string& program_key,
                                  const std::string& build_option);

  bool CheckFromSource(const std::string& file_name,
                       const std::string& program_key,
                       const std::string& build_option);

  void SaveProgram();

  void SaveTuned();

  std::unique_ptr<cl::UserEvent> CreateEvent(const cl::Context& context);

  bool BuildProgram(cl::Program* program, const std::string& options = "");

  bool IsInitSuccess() { return is_platform_device_init_success_; }

  std::string cl_path() { return cl_path_; }

  void set_cl_path(std::string cl_path) { cl_path_ = cl_path; }

  std::map<std::string, size_t>& GetDeviceInfo();

  GpuType& GetGpuType();

  uint32_t DeviceComputeUnits() const {
    return device_->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  }

  // Query the maximum work-group size that can be used to execute a kernel on a
  // specific device
  uint64_t GetMaxWorkGroupSize(const cl::Kernel& kernel);

  double GetCommandTime(const cl::Event& event);

  double GetQueuedTime(const cl::Event& event);

  double GetSubmitTime(const cl::Event& event);

  bool HasTunedLocalWorkSizeMap(const std::string& key,
                                std::vector<int>* tuned_value);

  void SetTunedLocalWorkSizeMap(const std::string& key,
                                const std::vector<int>& tune_vct);

 private:
  CLRuntime() { Init(); }
  CLRuntime(const CLRuntime&) = delete;
  CLRuntime(const CLRuntime&&) = delete;
  CLRuntime& operator=(const CLRuntime&) = delete;
  CLRuntime& operator=(const CLRuntime&&) = delete;
  ~CLRuntime();

  bool InitializePlatform();

  bool InitializeDevice();

  void GetAdrenoContextProperties(
      std::vector<cl_context_properties>* properties,
      GPUPerfMode gpu_perf_mode,
      GPUPriorityLevel gpu_priority_level);

  std::string GetSN(const std::string options);

  std::shared_ptr<cl::Context> CreateContext() {
    // note(ysh329): gpu perf mode and priority level of adreno gpu referred
    // from xiaomi/mace.
    // However, no performance gain after `PERF_HIGH` and `PRIORITY_HIGH` set.
    auto perf_mode = GPUPerfMode::PERF_HIGH;
    auto priority_level = GPUPriorityLevel::PRIORITY_HIGH;
    std::vector<cl_context_properties> context_properties;
    if (gpu_type_ == GpuType::QUALCOMM_ADRENO &&
        device_info_["CL_DEVICE_VERSION"] >= OpenCLVersion::CL_VER_2_0) {
      GetAdrenoContextProperties(
          &context_properties, perf_mode, priority_level);
    }
    auto context =
        std::make_shared<cl::Context>(std::vector<cl::Device>{device()},
                                      context_properties.data(),
                                      nullptr,
                                      nullptr,
                                      &status_);
    // use in is opencl valid check, do not exit here when release.
    CL_CHECK_ERROR(status_);
    return context;
  }

  std::shared_ptr<cl::CommandQueue> CreateCommandQueue(
      const cl::Context& context) {
    cl_command_queue_properties properties = 0;

#ifdef LITE_WITH_PROFILE
    properties |= CL_QUEUE_PROFILING_ENABLE;
#endif  // LITE_WITH_PROFILE
    if (auto_tune_ > 0) {
      properties |= CL_QUEUE_PROFILING_ENABLE;
    }

    auto queue = std::make_shared<cl::CommandQueue>(
        context, device(), properties, &status_);
    // use in is opencl valid check, do not exit here when release.
    CL_CHECK_ERROR(status_);
    return queue;
  }

  OpenCLVersion ParseDeviceVersion(const std::string& device_version);

  GpuType ParseGpuTypeFromDeviceName(std::string device_name);

  // binary
  bool Serialize(const std::string file_name,
                 const std::map<std::string, cl::Program::Binaries>& map_data);

  bool Deserialize(const std::string file_name,
                   std::map<std::string, cl::Program::Binaries>* map_ptr);

  // tuned param
  bool Serialize(const std::string file_name,
                 const std::map<std::string, std::vector<int>>& map_data);

  bool Deserialize(const std::string file_name,
                   std::map<std::string, std::vector<int>>* map_ptr);

  std::map<std::string, size_t> device_info_;

  GpuType gpu_type_{GpuType::UNKNOWN};

  std::string cl_path_;

  std::shared_ptr<cl::Platform> platform_{nullptr};

  std::shared_ptr<cl::Context> context_{nullptr};

  std::shared_ptr<cl::Device> device_{nullptr};

  std::shared_ptr<cl::CommandQueue> command_queue_{nullptr};

  cl_int status_{CL_SUCCESS};

  bool is_device_avaliable_for_opencl_{true};

  bool is_cl_runtime_initialized_{false};

  bool is_platform_device_init_success_{false};

  // CLTuneMode
  // 0 - None
  // 1 - Rapid
  // 2 - Normal
  // 3 - Exhaustive
  lite_api::CLTuneMode auto_tune_{lite_api::CL_TUNE_NONE};

  size_t lws_repeats_{0};

  // CLPrecisionType
  // 0 - AUTO, 1 - fp32, 2 - fp16
  lite_api::CLPrecisionType precision_{lite_api::CL_PRECISION_AUTO};

  std::map<std::string, std::unique_ptr<cl::Program>> programs_;
  std::map<std::string, cl::Program::Binaries> programs_precompiled_binary_;
  std::map<std::string, std::vector<int>> tuned_lwss_map_;
  std::vector<std::string> binary_path_name_;
  std::vector<std::string> tuned_path_name_;
  // magic number for precompiled binary
  const std::string sn_key_{"lite_opencl_precompiled_binary_identifier"};
  bool gotten_bin_flag_{false};
  bool del_tune_bin_flag_{false};
  bool have_tune_file_flag_{false};
  // magic number for cl flush judgement
  const int opencl_flush_period_ = 10;
};

}  // namespace lite
}  // namespace paddle
