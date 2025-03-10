# PaddleLite使用Amlogic NPU预测部署

Paddle Lite已支持Amlogic NPU的预测部署。
其接入原理是与之前华为Kirin NPU、瑞芯微Rockchip NPU等类似，即加载并分析Paddle模型，首先将Paddle算子转成NNAdapter标准算子，其次再转换为Amlogic NPU组网API进行网络构建，在线生成并执行模型。

## 支持现状

### 已支持的芯片

- C308X
- A311D
- S905D3(Android版本)

### 已支持的Paddle模型

#### 模型
- [mobilenet_v1_int8_224_per_layer](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_int8_224_per_layer.tar.gz)

#### 性能
- 测试环境
  - 编译环境
    - Ubuntu 16.04，GCC 5.4 for ARMLinux armhf and aarch64

  - 硬件环境
    - C308X
      - CPU：2 x ARM Cortex-55
      - NPU：4 TOPs for INT8

    - A311D
      - CPU：4 x ARM Cortex-A73 \+  2 x ARM Cortex-A53
      - NPU：5 TOPs for INT8
    - S905D3(Android版本)
      - CPU：2 x ARM Cortex-55
      - NPU：1.2 TOPs for INT8
  
- 测试方法
  - warmup=1, repeats=5，统计平均时间，单位是ms
  - 线程数为1，```paddle::lite_api::PowerMode CPU_POWER_MODE```设置为``` paddle::lite_api::PowerMode::LITE_POWER_HIGH ```
  - 分类模型的输入图像维度是{1，3，224，224}
  
- 测试结果

  |模型 |C308X||A311D||S905D3(Android版本)||
  |---|---|---|---|---|---|---|
  |  |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |CPU(ms) | NPU(ms) |
  |mobilenet_v1_int8_224_per_layer| 167.6996 |  6.982800| 81.632133 | 5.607733 | 280.465997 | 13.411600 |

### 已支持（或部分支持）NNAdapter的Paddle算子
可以通过访问[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/bridges/paddle_use_bridges.h](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/kernels/nnadapter/bridges/paddle_use_bridges.h)获得最新的算子支持列表。

## 参考示例演示

### 测试设备

- C308X开发板

  <img src="https://paddlelite-demo.bj.bcebos.com/devices/amlogic/C308X.jpg" alt="C380X" style="zoom: 33%;" />

  

- A311D开发板

   <img src="https://paddlelite-demo.bj.bcebos.com/devices/amlogic/A311D.jpg" alt="A311D" style="zoom: 33%;" />

### 准备设备环境

- C308X

  - 需要驱动版本为6.4.4.3（下载驱动请联系开发版厂商）。
  - 注意是64位系统。
  - 将MicroUSB线插入到设备的MicroUSB OTG口，就可以使用Android的adb命令进行设备的交互，当然也提供了网络连接SSH登录的方式。

    - 可通过dmesg | grep -r Galcore查询系统版本：

  ```shell
    $ dmesg | grep -rsn Galcore
    [   23.599566] Galcore version 6.4.4.3.310723AAA
  ```

- A311D

  - 需要驱动版本为6.4.4.3（下载驱动请联系开发版厂商）。

  - 注意是64位系统。

  - 将MicroUSB线插入到设备的MicroUSB OTG口，就可以使用Android的adb命令进行设备的交互，当然也提供了网络连接SSH登录的方式。

    - 可通过dmesg | grep -r Galcore查询系统版本：

    ```shell
    $ dmesg | grep -rsn Galcore
    [   24.140820] Galcore version 6.4.4.3.310723AAA
    ```

- S905D3(Android版本)

   - 需要驱动版本为6.4.4.3（下载驱动请联系开发版厂商）：
   - adb root + adb remount以获得修改系统库的权限。
   
    ```shell
    # dmesg | grep version
    [    9.020108] <4>[    9.020108@0] npu_version: 3
    [    9.020168] <6>[    9.020168@0] Galcore version 6.4.4.3.310723a
    ```
   
   - 示例程序和PaddleLite库的编译需要采用交叉编译方式，通过adb进行设备的交互和示例程序的运行。
   

### 准备交叉编译环境

- 为了保证编译环境一致，建议参考[编译环境准备](../source_compile/compile_env)中的Docker开发环境进行配置；
- 由于有些设备只提供网络访问方式（根据开发版的实际情况），需要通过scp和ssh命令将交叉编译生成的PaddleLite库和示例程序传输到设备上执行，因此，在进入Docker容器后还需要安装如下软件：

  ```
  # apt-get install openssh-client sshpass
  ```

### 运行图像分类示例程序

- 下载PaddleLite通用示例程序[PaddleLite-generic-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/devices/generic/PaddleLite-generic-demo.tar.gz)，解压后目录主体结构如下：

  ```shell
    - PaddleLite-generic-demo
      - image_classification_demo
        - assets
          - images
            - tabby_cat.jpg # 测试图片
            - tabby_cat.raw # 经过convert_to_raw_image.py处理后的RGB Raw图像
          - labels
            - synset_words.txt # 1000分类label文件
          - models
            - mobilenet_v1_int8_224_per_layer
              - __model__ # Paddle fluid模型组网文件，可使用netron查看网络结构
              — conv1_weights # Paddle fluid模型参数文件
              - batch_norm_0.tmp_2.quant_dequant.scale # Paddle fluid模型量化参数文件
              — subgraph_partition_config_file.txt # 自定义子图分割配置文件
              ...
        - shell
          - CMakeLists.txt # 示例程序CMake脚本
          - build.linux.arm64 # arm64编译工作目录
            - image_classification_demo # 已编译好的，适用于arm64的示例程序
          - build.linux.armhf # armhf编译工作目录
            - image_classification_demo # 已编译好的，适用于armhf的示例程序
          ...
          - image_classification_demo.cc # 示例程序源码
          - build.sh # 示例程序编译脚本
          - run.sh # 示例程序本地运行脚本
          - run_with_ssh.sh # 示例程序ssh运行脚本
          - run_with_adb.sh # 示例程序adb运行脚本
      - libs
        - PaddleLite
          - linux
            - arm64 # Linux 64位系统
              - include # PaddleLite头文件
              - lib # PaddleLite库文件
                - amlogic_npu # Amlogic NPU DDK、NNAdapter运行时库、device HAL库
                  - libnnadapter.so # NNAdapter运行时库
                  - libamlogic_npu.so # NNAdapter device HAL库
                  - libamlnpu_ddk.so # 晶晨NPU DDK
                  - libGAL.so # 芯原DDK
                  - libVSC.so # 芯原DDK
                  - libOpenVX.so # 芯原DDK
                  - libarchmodelSw.so # 芯原DDK
                  - libNNArchPerf.so # 芯原DDK
                  - libOvx12VXCBinary.so # 芯原DDK
                  - libNNVXCBinary.so # 芯原DDK
                  - libOpenVXU.so # 芯原DDK
                  - libNNGPUBinary.so # 芯原DDK
                  - libovxlib.so # 芯原DDK
                  - libOpenCL.so # OpenCL
                  - libnnrt.so # amlogic DDK依赖库
                  - libnnsdk_lite.so # amlogic DDK依赖库
                  - libgomp.so.1 # gnuomp库
                - libpaddle_full_api_shared.so # 预编译PaddleLite full api库
                - libpaddle_light_api_shared.so # 预编译PaddleLite light api库
            ...
          - android
           - armeabi-v7a # Android 32位系统
              - include # PaddleLite头文件
              - lib # PaddleLite库文件
                - amlogic_npu # Amlogic NPU DDK、NNAdapter运行时库、device HAL库
                  - libnnadapter.so # NNAdapter运行时库
                  - libamlogic_npu.so # NNAdapter device HAL库
                  - libamlnpu_ddk.so # 晶晨NPU DDK
                  - libGAL.so # 芯原DDK
                  - libVSC.so # 芯原DDK
                  - libOpenVX.so # 芯原DDK
                  - libarchmodelSw.so # 芯原DDK
                  - libNNArchPerf.so # 芯原DDK
                  - libOvx12VXCBinary.so # 芯原DDK
                  - libNNVXCBinary.so # 芯原DDK
                  - libOpenVXU.so # 芯原DDK
                  - libNNGPUBinary.so # 芯原DDK
                  - libovxlib.so # 芯原DDK
                  - libOpenCL.so # OpenCL
                  - libnnrt.so # amlogic DDK依赖库
                  - libnnsdk_lite.so # amlogic DDK依赖库
                  - libc++_shared.so
                - libpaddle_full_api_shared.so # 预编译PaddleLite full api库
                - libpaddle_light_api_shared.so # 预编译PaddleLite light api库
        - OpenCV # OpenCV预编译库
      - ssd_detection_demo # 基于ssd的目标检测示例程序
  ```

- 按照以下命令分别运行转换后的ARM CPU模型和Amlogic NPU模型，比较它们的性能和结果；

  ```shell
  注意：
  1）run_with_adb.sh不能在docker环境执行，否则可能无法找到设备，也不能在设备上运行。
  2）run_with_ssh.sh不能在设备上运行，且执行前需要配置目标设备的IP地址、SSH账号和密码。
  3）build.sh根据入参生成针对不同操作系统、体系结构的二进制程序，需查阅注释信息配置正确的参数值。
  4）run_with_adb.sh入参包括模型名称、操作系统、体系结构、目标设备、设备序列号等，需查阅注释信息配置正确的参数值。
  5）run_with_ssh.sh入参包括模型名称、操作系统、体系结构、目标设备、ip地址、用户名、用户密码等，需查阅注释信息配置正确的参数值。
  
  在ARM CPU上运行mobilenet_v1_int8_224_per_layer全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  
  For C308X
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu 
    (C308X)
    warmup: 1 repeat: 5, average: 167.6916 ms, max: 207.458000 ms, min: 159.823239 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 3.423000 ms
    Prediction time: 167.6996 ms
    Postprocess time: 0.542000 ms
  
  For A311D
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 cpu 
    (A311D)
    warmup: 1 repeat: 15, average: 81.678067 ms, max: 81.945999 ms, min: 81.591003 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 1.352000 ms
    Prediction time: 81.678067 ms
    Postprocess time: 0.407000 ms
  
  For S905D3(Android版)
  $ ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer android armeabi-v7a cpu
    (S905D3(Android版))
    warmup: 1 repeat: 5, average: 280.465997 ms, max: 358.815002 ms, min: 268.549812 ms
    results: 3
    Top0  Egyptian cat - 0.512545
    Top1  tabby, tabby cat - 0.402567
    Top2  tiger cat - 0.067904
    Preprocess time: 3.199000 ms
    Prediction time: 280.465997 ms
    Postprocess time: 0.596000 ms
  
  ------------------------------
  
  在Amlogic NPU上运行mobilenet_v1_int8_224_per_layer全量化模型
  $ cd PaddleLite-generic-demo/image_classification_demo/shell
  
  For C308X
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 amlogic_npu
    (C308X)
    warmup: 1 repeat: 5, average: 6.982800 ms, max: 7.045000 ms, min: 6.951000 ms
    results: 3
    Top0  Egyptian cat - 0.508929
    Top1  tabby, tabby cat - 0.415333
    Top2  tiger cat - 0.064347
    Preprocess time: 2.417000 ms
    Prediction time: 6.982800 ms
    Postprocess time: 0.509000 ms
  
  For A311D
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer linux arm64 amlogic_npu
    ( A311D)
    warmup: 1 repeat: 15, average: 5.567867 ms, max: 5.723000 ms, min: 5.461000 ms
    results: 3
    Top0  Egyptian cat - 0.508929
    Top1  tabby, tabby cat - 0.415333
    Top2  tiger cat - 0.064347
    Preprocess time: 1.356000 ms
    Prediction time: 5.567867 ms
    Postprocess time: 0.411000 ms
  
  For S905D3(Android版)
  $ ./run_with_adb.sh mobilenet_v1_int8_224_per_layer android armeabi-v7a amlogic_npu
    (S905D3(Android版))
    warmup: 1 repeat: 5, average: 13.4116 ms, max: 15.751210 ms, min: 12.433400 ms
    results: 3
    Top0  Egyptian cat - 0.508929
    Top1  tabby, tabby cat - 0.415333
    Top2  tiger cat - 0.064347
    Preprocess time: 3.170000 ms
    Prediction time: 13.4116 ms
    Postprocess time: 0.634000 ms
  ```
  
- 如果需要更改测试图片，可将图片拷贝到PaddleLite-generic-demo/image_classification_demo/assets/images目录下，然后调用convert_to_raw_image.py生成相应的RGB Raw图像，最后修改run_with_adb.sh、run_with_ssh.sh的IMAGE_NAME变量即可；
- 重新编译示例程序：  
  ```shell
  注意：
  1）请根据buid.sh配置正确的参数值。
  2）需在docker环境中编译。
  
  # 对于C308X，A311D
  ./build.sh linux arm64
  
  # 对于S905D3(Android版)
  ./build.sh android armeabi-v7a
  ```

### 更新模型
- 通过Paddle训练或X2Paddle转换得到MobileNetv1 foat32模型[mobilenet_v1_fp32_224](https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_fluid.tar.gz)
- 通过Paddle+PaddleSlim后量化方式，生成[mobilenet_v1_int8_224_per_layer量化模型](https://paddlelite-demo.bj.bcebos.com/devices/rockchip/mobilenet_v1_int8_224_fluid.tar.gz)
- 下载[PaddleSlim-quant-demo.tar.gz](https://paddlelite-demo.bj.bcebos.com/tools/PaddleSlim-quant-demo.tar.gz)，解压后清单如下：
    ```shell
    - PaddleSlim-quant-demo
      - image_classification_demo
        - quant_post # 后量化
          - quant_post_rockchip_npu.sh # Rockchip NPU 一键量化脚本，Amlogic和瑞芯微底层都使用芯原的NPU，所以通用
          - README.md # 环境配置说明，涉及PaddlePaddle、PaddleSlim的版本选择、编译和安装步骤
          - datasets # 量化所需要的校准数据集合
            - ILSVRC2012_val_100 # 从ImageNet2012验证集挑选的100张图片
          - inputs # 待量化的fp32模型
            - mobilenet_v1
            - resnet50
          - outputs # 产出的全量化模型
          - scripts # 后量化内置脚本
    ```
- 查看README.md完成PaddlePaddle和PaddleSlim的安装
- 直接执行./quant_post_rockchip_npu.sh即可在outputs目录下生成mobilenet_v1_int8_224_per_layer量化模型
  ```shell
  -----------  Configuration Arguments -----------
  activation_bits: 8
  activation_quantize_type: moving_average_abs_max
  algo: KL
  batch_nums: 10
  batch_size: 10
  data_dir: ../dataset/ILSVRC2012_val_100
  is_full_quantize: 1
  is_use_cache_file: 0
  model_path: ../models/mobilenet_v1
  optimize_model: 1
  output_path: ../outputs/mobilenet_v1
  quantizable_op_type: conv2d,depthwise_conv2d,mul
  use_gpu: 0
  use_slim: 1
  weight_bits: 8
  weight_quantize_type: abs_max
  ------------------------------------------------
  quantizable_op_type:['conv2d', 'depthwise_conv2d', 'mul']
  2021-08-30 05:52:10,048-INFO: Load model and set data loader ...
  2021-08-30 05:52:10,129-INFO: Optimize FP32 model ...
  I0830 05:52:10.139564 14447 graph_pattern_detector.cc:91] ---  detected 14 subgraphs
  I0830 05:52:10.148236 14447 graph_pattern_detector.cc:91] ---  detected 13 subgraphs
  2021-08-30 05:52:10,167-INFO: Collect quantized variable names ...
  2021-08-30 05:52:10,168-WARNING: feed is not supported for quantization.
  2021-08-30 05:52:10,169-WARNING: fetch is not supported for quantization.
  2021-08-30 05:52:10,170-INFO: Preparation stage ...
  2021-08-30 05:52:11,853-INFO: Run batch: 0
  2021-08-30 05:52:16,963-INFO: Run batch: 5
  2021-08-30 05:52:21,037-INFO: Finish preparation stage, all batch:10
  2021-08-30 05:52:21,048-INFO: Sampling stage ...
  2021-08-30 05:52:31,800-INFO: Run batch: 0
  2021-08-30 05:53:23,443-INFO: Run batch: 5
  2021-08-30 05:54:03,773-INFO: Finish sampling stage, all batch: 10
  2021-08-30 05:54:03,774-INFO: Calculate KL threshold ...
  2021-08-30 05:54:28,580-INFO: Update the program ...
  2021-08-30 05:54:29,194-INFO: The quantized model is saved in ../outputs/mobilenet_v1
  post training quantization finish, and it takes 139.42292165756226.
  
  -----------  Configuration Arguments -----------
  batch_size: 20
  class_dim: 1000
  data_dir: ../dataset/ILSVRC2012_val_100
  image_shape: 3,224,224
  inference_model: ../outputs/mobilenet_v1
  input_img_save_path: ./img_txt
  save_input_img: False
  test_samples: -1
  use_gpu: 0
  ------------------------------------------------
  Testbatch 0, acc1 0.8, acc5 1.0, time 1.63 sec
  End test: test_acc1 0.76, test_acc5 0.92
  --------finish eval int8 model: mobilenet_v1-------------
  ```
  - 参考[模型转化方法](../user_guides/model_optimize_tool)，利用opt工具转换生成Amlogic NPU模型，仅需要将valid_targets设置为amlogic_npu,arm即可。
  ```shell
  $ ./opt --model_dir=mobilenet_v1_int8_224_per_layer \
      --optimize_out_type=naive_buffer \
      --optimize_out=opt_model \
      --valid_targets=amlogic_npu,arm
  ```
### 更新支持Amlogic NPU的Paddle Lite库

- 下载PaddleLite源码和Amlogic NPU DDK

  ```shell
  $ git clone https://github.com/PaddlePaddle/Paddle-Lite.git
  $ cd Paddle-Lite
  $ git checkout <release-version-tag>
  ###C308X、A311D
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/amlogic/Linux-arm64/amlnpu_ddk.zip
  ###S905D3 Android版本
  $ wget https://paddlelite-demo.bj.bcebos.com/devices/amlogic/Android-armeabi-v7a/amlnpu_ddk.zip
  $ unzip amlnpu_ddk.zip
  ```

- 编译并生成PaddleLite+AmlogicNPU的部署库

  - For C308X and A311D
    - tiny_publish编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_amlogic_npu=ON --nnadapter_amlogic_npu_sdk_root=$(pwd)/amlnpu_ddk
      
      ```
    - full_publish编译方式
      ```shell
      $ ./lite/tools/build_linux.sh --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_amlogic_npu=ON --nnadapter_amlogic_npu_sdk_root=$(pwd)/amlnpu_ddk full_publish
      
      ```
    - 替换头文件和库
      ```shell
      # 替换include目录
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/include/
      # 替换NNAdapter运行时库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/amlogic_npu/
      # 替换NNAdapter device HAL库
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libamlogic_npu.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/amlogic_npu/
      # 替换libpaddle_light_api_shared.so
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      # 替换libpaddle_full_api_shared.so(仅在full_publish编译方式下)
      $ cp -rf build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/linux/arm64/lib/
      ```

  - S905D3(Android版)
    - tiny_publish编译方式
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_amlogic_npu=ON --nnadapter_amlogic_npu_sdk_root=$(pwd)/amlnpu_ddk
      ```

    - full_publish编译方式
      ```shell
      $ ./lite/tools/build_android.sh --arch=armv7 --toolchain=clang --android_stl=c++_shared --with_extra=ON --with_log=ON --with_nnadapter=ON --nnadapter_with_amlogic_npu=ON --nnadapter_amlogic_npu_sdk_root=$(pwd)/amlnpu_ddk full_publish
      ```
    - 替换头文件和库
      ```shell
      # 替换include目录
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/include/ PaddleLite-generic-demo/libs/PaddleLite/linux/armhf/include/
      # 替换NNAdapter运行时库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libnnadapter.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/amlogic_npu/
      # 替换NNAdapter device HAL库
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libamlogic_npu.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/amlogic_npu/
      # 替换libpaddle_light_api_shared.so
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_light_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      # 替换libpaddle_full_api_shared.so(仅在full_publish编译方式下)
      $ cp -rf build.lite.android.armv7.clang/inference_lite_lib.android.armv7.nnadapter/cxx/lib/libpaddle_full_api_shared.so PaddleLite-generic-demo/libs/PaddleLite/android/armeabi-v7a/lib/
      ```
  
- 替换头文件后需要重新编译示例程序

## 其它说明

- Amlogic和PaddleLite研发团队正在持续增加用于适配Paddle算子的bridge/converter，以便适配更多Paddle模型。
