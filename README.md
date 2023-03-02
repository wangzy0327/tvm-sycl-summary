# tvm-sycl测试验证

### 概述

TVM中添加SYCL设备代码支持，相关tvm-sycl代码详见https://github.com/RELOAD22/tvm的sycl分支

依赖sycl分支详见 http://10.18.127.29:8888/wangziyang/intel-llvm-new 的v2022-12-local-deps分支

目前sycl后端设备已经支持Nvidia、AMD、Hygon三种硬件设备的大多数网络模型。

![tvm-sycl框架图](imgs/tvm-sycl-structure.png)

### 支持网络模型

以下的测试的tvm版本为v.0.10 Release，cuda版本为11.2，hip版本为5.2，hygon版本为5.2，SYCL版本为2022-09-release

cuda平台下测试设备为Tesla V100-32GB，hip平台下测试设备为AMD Radeon (TM) Pro-16GB-gfx-906，hygon平台下测试设备为Hygon-Z100-33GB-gfx916 . 

cuda Tesla V100 测试日志文件 [详见](tvm-cuda-V100-sycl-test-result/cuda-V100-networksummary.xlsx)

| network      | cuda | SYCL-Nvidia   | rocm-AMD | SYCL-AMD | rocm-Hygon | SYCL-Hygon         |
| ------------ | ---- | ------------- | -------- | -------- | ---------- | ------------------ |
| mnist        | √    | √（mnist-1×） | ×        | √        | ×          | √（Log_Error?）    |
| alexnet      | √    | √             | ×        | √        | ×          | √（same as above） |
| caffenet     | √    | √             | ×        | √        | ×          | √（same as above） |
| densenet     | √    | √             | ×        | √        | ×          | √（same as above） |
| efficientnet | √    | √             | ×        | √        | ×          | √（same as above） |
| inception    | √    | √             | ×        | √        | ×          | ×                  |
| googlenet    | √    | √             | ×        | √        | ×          | ×                  |
| mobilenet    | √    | √             | ×        | √        | ×          | √（same as above） |
| rcnn         | √    | √             | ×        | √        | ×          | √（same as above） |
| resnet       | √    | √             | ×        | √        | ×          | √（same as above） |
| shufflenet   | √    | √             | ×        | √        | ×          | √（same as above） |
| squeezenet   | √    | √             | ×        | √        | ×          | √（same as above） |
| vgg          | √    | √             | ×        | √        | ×          | √（same as above） |
| zfnet        | √    | √             | ×        | √        | ×          | √（same as above） |

tvm-sycl开发测试过程中遇到的bug

| network                                                      | platform       | bug                                                          | progress                                          |
| ------------------------------------------------------------ | -------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| bvlcalexnet-7(alexnet)                                       | cuda           | cuda_piextUSMFree(pi_context, void*): Assertion `type == CU_MEMORYTYPE_DEVICE | fix（fix SYCL plugin USMFree interface）          |
| resnet50-caffe2-v1-6（resnet-caffe）&& mnist-1(mnist) && ... | cuda/hip/hygon | Assertion `KSIdMap[EntriesIt->name] == KSIdIt->second && "Kernel sets are not disjoint"' failed | fix（fix SYCL program manager kernel sets check） |
| any                                                          | cuda/hip/hygon | warning: linking module ''[-Wlinker-warnings]                | fix（fix in 2022-12-release）                     |
| any                                                          | cuda/hip/hygon | warning: linked binaries do not contain expected [-Wsycl-target] | fix（fix in 2022-12-release）                     |
| **any**(some time)                                           | cuda/hip/hygon | clang-offload-bundler:error ‘/tmp/libsycl-complex-fp65-complex-fp64-11cc6d.cubin’:permission denied | undo                                              |
| **any**(all time)                                            | hygon          | [LOG_ERROR]: cannot find the function _ZTSZZ39tvmgen_default_ | undo                                              |
| inception & googlenet                                        | hygon          | **nan**                                                      | undo                                              |

### 自动优化

测试auto-tuning的tvm版本为v.0.10 Release，cuda版本为11.2，SYCL版本为2022-12-release

cuda平台下测试设备为Tesla V100-32GB

目前测试遇到的问题

| network | platform  | bug                                                          | progress              |
| ------- | --------- | ------------------------------------------------------------ | --------------------- |
| mnist-1 | sycl/cuda | [compute number accuracy]()                                  | undo(2022-12-release) |
| any     | sycl/cuda | PI CUDA ERROR 700 an illegal memory access was encountered(sycl/plugins/cuda/pi_cuda.cpp) | undo                  |
|         |           |                                                              |                       |

##### bug-1

correct result(sycl-2022-09)

![correct-result.png](imgs/auto-tune-mnist-1-correct.png)

wrong result(sycl-2022-12)

<img src="imgs/auto-tune-mnist-1-wrong.png" alt="wrong-result.png" style="zoom: 70%;" />

##### bug-2

PI CUDA ERROR

![PI_CUDA_ERROR.png](imgs/PI_CUDA_ERROR.png)