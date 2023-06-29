

# tvm-sycl测试验证

### 概述

TVM中添加SYCL设备代码支持，相关tvm-sycl代码详见https://github.com/RELOAD22/tvm 的sycl分支

依赖sycl分支详见 http://10.18.127.29:8888/wangziyang/intel-llvm-new 的v2022-12-local-deps分支

目前sycl后端设备已经支持Nvidia、AMD、Hygon三种硬件设备的大多数网络模型。

![tvm-sycl框架图](imgs/tvm-sycl-structure.png)

### 支持网络模型

以下的测试的tvm版本为v.0.10 Release，cuda版本为11.2，hip版本为5.4.3，hygon版本为5.2，SYCL版本为2022-12-release

Nvidia硬件平台下测试设备为Tesla V100-32GB，AMD硬件平台下测试设备为AMD Radeon (TM) Pro-16GB-gfx-906，Intel平台下测试设备为Intel Arctic Sound-P-16G 300W，hygon平台下测试设备为Hygon-Z100-33GB-gfx916 . 

Nvidia Tesla V100 测试统计结果 [详见](tvm-nvidia-V100-sycl-test-result/cuda-V100-network-summary-new.xlsx)

Nvidia Tesla V100 测试日志 [详见](tvm-nvidia-V100-sycl-test-result/error_tvm_V100_nvidia_sycl_new.log)

AMD Radeon MI50 测试统计结果 [详见](tvm-amd-MI50-sycl-test-result/rocm-MI50-network-summary-new.xlsx)

AMD Radeon MI50 测试日志 [详见](tvm-amd-MI50-sycl-test-result/error_tvm_MI50_rocm_sycl.log)

Intel Arctic Sound-P 300W测试统计结果 [详见](tvm-intel-ASP300-sycl-test-result/sycl-ASP300-network-summary-new.xlsx)

Intel Arctic Sound-P 300W测试日志 [详见](tvm-intel-ASP300-sycl-test-result/error_tvm_Intel_sycl.log)

Hygon Z100 测试统计结果 [详见](tvm-hygon-Z100-sycl-test-result/Z100-rocm-summary.xlsx)

Hygon Z100测试日志 [详见](tvm-hygon-Z100-sycl-test-result/error_tvm_Z100_hygon_sycl.log)

| network      | cuda-Nvidia | SYCL-Nvidia                                                  | OpenCL-Nvidia | rocm-AMD                                                     | SYCL-AMD                                          | OpenCL-AMD | rocm-Hygon | SYCL-Hygon         | OpenCL-Hygon | SYCL-Intel         | OpenCL-Intel |
| ------------ | ----------- | ------------------------------------------------------------ | ------------- | ------------------------------------------------------------ | ------------------------------------------------- | ---------- | ---------- | ------------------ | ------------ | ------------------ | ------------ |
| mnist        | √           | √（mnist-1×）                                                | √             | <font color=green>**√** </font>                              | √ (mnist-1 ×)                                     | √          | ×          | √（Log_Error?）    | √            | √                  | √            |
| alexnet      | √           | √                                                            | √             | <font color=green>**√**</font>                               | √                                                 | √          | ×          | √（same as above） | √            | √                  | √            |
| caffenet     | √           | √(caffenet-3 ×)<br />(caffenet-9 ×)                          | √             | <font color=green>**√**  </font>                             | √√ (caffenet-3 ×)<br />(caffenet-9 ×)             | √          | ×          | √（same as above） | √            | √                  | √            |
| densenet     | √           | √                                                            | √             | <font color=green>**√**  </font>                             | √                                                 | √          | ×          | √（same as above） | √            | √                  | √            |
| efficientnet | √           | √                                                            | √             | <font color=green>**√** </font>                              | √                                                 | √          | ×          | √（same as above） | √            | √                  | √            |
| inception    | √           | √                                                            | √             | <font color=green>**√** </font>                              | √                                                 | √          | ×          | ×                  | √            | √                  | √            |
| googlenet    | √           | √                                                            | √             | <font color=green>**√** </font>                              | √（googlenet-3 ×）                                | √          | ×          | ×                  | √            | √                  | √            |
| mobilenet    | √           | √                                                            | √             | <font color=green>**√** </font>                              | √                                                 | √          | ×          | √（same as above） | √            | √                  | √            |
| rcnn         | √           | √                                                            | √             | <font color=green>**√** </font>                              | √                                                 | √          | ×          | √（same as above） | √            | √                  | √            |
| resnet       | √           | √(resnet50-v1-7 ×)<br />(resnet50-caffe2-v1-6 ×)<br/>(resnet101-v2-7 ×) | √             | <font color=green>**√<br/>(resnet152-v1-7 ×)<br />(resnet50-caffe2-v1-3 ×)** </font> | √ (resnet50-v1-7 ×)<br />(resnet50-caffe2-v1-6 ×) | √          | ×          | √（same as above） | √            | √(resnet50-v1-7 ×) | √            |
| shufflenet   | √           | √                                                            | √             | <font color=green>**√** </font>                              | √                                                 | √          | ×          | √（same as above） | √            | √                  | √            |
| squeezenet   | √           | √                                                            | √             | <font color=green>**√** </font>                              | √                                                 | √          | ×          | √（same as above） | √            | √                  | √            |
| vgg          | √           | √(vgg19-caffe2-6 ×)                                          | √             | <font color=green>**√（vgg16-bn-7 ×）** </font>              | √（vgg16-bn-7 ×）(vgg19-caffe2-6 ×)               | √          | ×          | √（same as above） | √            | √                  | √            |
| zfnet        | √           | √                                                            | √             | <font color=green>**√** </font>                              | √                                                 | √          | ×          | √（same as above） | √            | √                  | √            |

tvm-sycl开发测试过程中遇到的bug

| network                                                      | platform       | bug                                                          | progress                                                     |
| ------------------------------------------------------------ | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| bvlcalexnet-7(alexnet)                                       | cuda           | cuda_piextUSMFree(pi_context, void*): Assertion `type == CU_MEMORYTYPE_DEVICE | fix（fix SYCL plugin USMFree interface）                     |
| any                                                          | hip            | rocm compute result is not correct                           | fix (rocm5.4 add rocm fix on python/tvm/relay/op/strategy/cuda.p) |
| **any**(some time)                                           | cuda/hip/hygon | clang-offload-bundler:error ‘/tmp/libsycl-complex-fp65-complex-fp64-11cc6d.cubin’:permission denied | fix(add CAP_SYS_ADMIN privilege and NVreg_RestrictProfilingToAdminUsers=0) |
| resnet50-caffe2-v1-6（resnet-caffe）&& mnist-1(mnist) && ... | cuda/hip/hygon | Assertion `KSIdMap[EntriesIt->name] == KSIdIt->second && "Kernel sets are not disjoint"' failed | fix（fix SYCL program manager kernel sets check）            |
| any                                                          | cuda/hip/hygon | warning: linking module ''[-Wlinker-warnings]                | fix（fix in 2022-12-release）                                |
| any                                                          | cuda/hip/hygon | warning: linked binaries do not contain expected [-Wsycl-target] | fix（fix in 2022-12-release）                                |
| shuffulenet/efficientnet                                     | hip            | warning:Warning: Unroll hint get ignore at CodeGenLLVM backend,  consider set unroll_explicit=True<br />tvm/tvm/src/target/llvm/codegen_llvm.cc:1503 | unfix                                                        |
| vgg16-7(vgg)（some time）                                    | cuda           | CUDA_ERROR_ILLEGAL_ADDRESS：an illegal memory access was encountered | unfix                                                        |
| vgg16-7                                                      | hip            | PI HIP ERROR： hipErrorNotFound<br />Function:        hip_piKernelCreate<br />/home/wzy/sycl_workspace/llvm/sycl/plugins/hip/pi_hip.cpp:2717 | unfix                                                        |
| googlenet-3                                                  | hip            | Segmentation fault (core dumped)                             | unfix                                                        |
| **any**(some time)<br />resnet152-v1-7<br />resnet50-caffe2-v1-3 | cuda/hip/hygon | Check failed: (e.code() == sycl::errc::success) is false: SYCL Error, code=sycl:13: kERNEL NOT SUPPORTED | unfix                                                        |
| **any**(all time)                                            | hygon          | [LOG_ERROR]: cannot find the function _ZTSZZ39tvmgen_default_ | unfix                                                        |
| inception & googlenet                                        | hygon          | **nan**                                                      | unfix                                                        |

SYCL平台性能

sycl在Nvidia、AMD、Hygon、Intel硬件平台网络模型执行性能。

![resnet50-sycl-platform](imgs/resnet50-sycl-platform.png)

### 自动优化

测试auto-tuning的tvm版本为v.0.10 Release，cuda版本为11.2，SYCL版本为2022-12-release

cuda平台下测试设备为Tesla V100-32GB

目前测试遇到的问题

| network  | platform  | bug                                                          | progress                                    |
| -------- | --------- | ------------------------------------------------------------ | ------------------------------------------- |
| any      | rocm      | ROCM HIP：invalid device ordinal                             | <font color=green>fix(ROCMDeviceAPI)</font> |
| mnist-1  | sycl/cuda | compute number accuracy                                      | undo(2022-12-release)                       |
| any      | sycl/cuda | PI CUDA ERROR 700 an illegal memory access was encountered(sycl/plugins/cuda/pi_cuda.cpp) | undo                                        |
| google-3 | sycl/hip  | Memory access fault by GPU node-2 (Agent handle: 0x56534ffb3250) on address 0x7facb0600000. Reason: Page not present or supervisor privilege. | undo                                        |

##### bug-1

correct result(sycl-2022-09)

![correct-result.png](imgs/auto-tune-mnist-1-correct.png)

wrong result(sycl-2022-12)

<img src="imgs/auto-tune-mnist-1-wrong.png" alt="wrong-result.png" style="zoom: 70%;" />

##### bug-2

PI CUDA ERROR

![PI_CUDA_ERROR.png](imgs/PI_CUDA_ERROR.png)

**bug-3**

PI HIP ERROR

<img src="imgs/PI_HIP_ERROR.png" alt="PI_HIP_ERROR.png" style="zoom: 45%;" />

**bug_4**

/tmp/libsycl-fallback-xxx.cubin : Permission denied.

<img src="imgs/permission_error.png" alt="permission-error.png" style="zoom:55%;" />

**bug-5**

Memory access fault by GPU node-2 (Agent handle: 0x56534ffb3250) on address 0x7facb0600000. Reason: Page not present or supervisor privilege.

<img src="imgs/AMD-googlenet-3-error.png" alt="AMD-googlenet-3-error.png" style="zoom:67%;" />

<img src="imgs/AMD-memory-access-fault.png" alt="AMD-memory-access-fault.png" style="zoom:67%;" />

**bug-6**

```cpp
pi_result cuda_piKernelCreate(pi_program program, const char *kernel_name,
                              pi_kernel *kernel) {
  assert(kernel != nullptr);
  assert(program != nullptr);

  pi_result retErr = PI_SUCCESS;
  std::unique_ptr<_pi_kernel> retKernel{nullptr};

  try {
    ScopedContext active(program->get_context());

    CUfunction cuFunc;
    retErr = PI_CHECK_ERROR(
        cuModuleGetFunction(&cuFunc, program->get(), kernel_name));

    std::string kernel_name_woffset = std::string(kernel_name) + "_with_offset";
    CUfunction cuFuncWithOffsetParam;
    CUresult offsetRes = cuModuleGetFunction(
        &cuFuncWithOffsetParam, program->get(), kernel_name_woffset.c_str());

    // If there is no kernel with global offset parameter we mark it as missing
    if (offsetRes == CUDA_ERROR_NOT_FOUND) {
      cuFuncWithOffsetParam = nullptr;
    } else {
      retErr = PI_CHECK_ERROR(offsetRes);
    }

    retKernel = std::unique_ptr<_pi_kernel>(
        new _pi_kernel{cuFunc, cuFuncWithOffsetParam, kernel_name, program,
                       program->get_context()});
  } catch (pi_result err) {
    retErr = err;
  } catch (...) {
    retErr = PI_ERROR_OUT_OF_HOST_MEMORY;
  }

  *kernel = retKernel.release();
  return retErr;
}
```

<img src="imgs/hipErrorNotFound.png" alt="hipErrorNotFound" style="zoom: 80%;" />

tvm的rocm容器执行网络模型

```shell
docker run -it --device=/dev/dri --device=/dev/kfd --network=host --group-add=render \
-v /home/wzy:/home/wzy rocm-tvm-0.7:5.4.2 /bin/bash


docker run -it --device=/dev/dri --device=/dev/kfd --network=host --group-add=render \
-v /home/wzy:/home/wzy mevermeulen/rocm-tvm:5.4.2 /bin/bash
```

tvm的rocm容器Dockerfile

```dockerfile
FROM rocm/dev-ubuntu-20.04:5.2.3
RUN sed -e 's/debian/5.2.3/g' /etc/apt/sources.list.d/rocm.list > /etc/apt/sources.list.d/rocm5.2.list
RUN rm /etc/apt/sources.list.d/rocm.list
ENV PATH=/opt/rocm/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git wget libz3-dev libxml2-dev openssl libssl-dev libtinfo-dev libprotobuf-dev protobuf-compiler
RUN mkdir /src && cd /src && wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz && tar xf cmake-3.17.3.tar.gz && cd cmake-3.17.3 && ./configure && make && make install
RUN apt update && apt install -y rocm-libs miopen-hip
RUN apt update && apt install -y python python-dev python-setuptools gcc libtinfo-dev zlib1g-dev build-essential python3 python3-pip python3-setuptools python3-numpy
# a34731 - ok
# c2eb51 - not ok
RUN cd /src && git clone --recursive https://github.com/mvermeulen/tvm && cd tvm && git checkout rocm-5.2-test
RUN mkdir /src/tvm/build
RUN cd /src/tvm/build && sed -e 's/USE_ROCM OFF/USE_ROCM ON/g' -e 's?USE_LLVM OFF?USE_LLVM /opt/rocm/llvm/bin/llvm-config?g' -e 's/USE_MIOPEN OFF/USE_MIOPEN ON/g' -e 's/USE_ROCBLAS OFF/USE_ROCBLAS ON/g' ../cmake/config.cmake > config.cmake
RUN cd /src/tvm/build && cmake -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" .. && make
#RUN pip3 install os-sys
RUN pip3 install -U numpy
RUN cd /src/tvm/python && python3 setup.py install
RUN cd /src && git clone https://github.com/mvermeulen/rocm-tvm
RUN pip3 install scipy psutil xgboost tornado pytest
RUN apt update && apt install -y libomp-dev graphviz rccl libopenblas-dev pciutils
RUN export CMAKE_ARGS=-DONNX_USE_PROTOBUF_SHARED_LIBS=ON
RUN pip3 install jupyter transformers antlr4-python3-runtime graphviz onnx pillow
RUN pip3 install tensorflow
WORKDIR /src/rocm-tvm
```

**bug-7**

TVMError: LLVM module verification failed with the following errors: 

```shell
14.tvm/python/tvm/relay/build_module.py lib = relay.build(mod, target=target, params=params)
13.tvm/python/tvm/relay/build_module.py graph_json, runtime_mod, params = bld_mod.build(mod=ir_mod,
            target=raw_targets,
            params=params,
            executor=executor,
            runtime=runtime,
            workspace_memory_pools=workspace_memory_pools,
            constant_memory_pools=constant_memory_pools,
            mod_name=mod_name,
        )
12.tvm/python/tvm/relay/build_module.py class BuildModule def build(...) : self._build()
11.tvm/python/tvm/_ffi/_ctypes/packed_func.py PackedFuncBase::__call__()
10.src/relay/backend/build_module.cc {} tvm {} relay {} PackedFunc RelayBuildModule::GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self){}
9.src/relay/backend/build_module.cc {} tvm {} relay {} void RelayBuildModule:: Build(IRModule mod, const Array<Target>& raw_targets, const tvm::Target& target_host,
             const Executor& executor, const Runtime& runtime,
             const WorkspaceMemoryPools& workspace_memory_pools,
             const ConstantMemoryPools& constant_memory_pools, const String mod_name)
8.src/relay/backend/build_module.cc {} tvm {} relay {} backend  void RelayBuildModule::BuildRelay(IRModule relay_module, const String& mod_name) {}
7.src/driver/driver_api.cc {} tvm runtime::Module TIRToRuntime(const Map<Target, IRModule>& inputs_arg,const Target& target_host_arg) {}
6.src/target/codegen.cc {} tvm {} codegen Build(IRModule,Target)
5.include/tvm/runtime/packed_func.h {} tvm {} runtime template <typename R, typename... Args> template <typename FType>
inline void TypedPackedFunc<R(Args...)>::AssignTypedLambda(FType flambda, std::string name){}
4.src/target/llvm/codegen_amdgpu.cc {} tvm {} codegen BuildAMDGPU(IRModule,Target) / 3.src/target/opt/build_cuda_on.cc {} tvm {} codegen BuildCUDA(IRModule,Target)
2.src/target/llvm/codegen_llvm.cc {} tvm {} codegen CodeGenLLVM::Finish()
1.src/target/llvm/codegen_llvm.cc {} tvm {} codegen CodeGenLLVM::Verify() const
```



```
    raise get_last_ffi_error()
tvm._ffi.base.TVMError: Traceback (most recent call last):
  11: TVMFuncCall
  10: tvm::relay::backend::RelayBuildModule::GetFunction(tvm::runtime::String const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
  9: tvm::relay::backend::RelayBuildModule::Build(tvm::IRModule, tvm::runtime::Array<tvm::Target, void> const&, tvm::Target const&, tvm::relay::Executor const&, tvm::relay::Runtime const&, tvm::WorkspaceMemoryPools const&, tvm::ConstantMemoryPools const&, tvm::runtime::String)
  8: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
  7: tvm::TIRToRuntime(tvm::runtime::Map<tvm::Target, tvm::IRModule, void, void> const&, tvm::Target const&)
  6: tvm::codegen::Build(tvm::IRModule, tvm::Target)
  5: _ZN3tvm7runtime13Pac
  4: tvm::runtime::TypedPackedFunc<tvm::runtime::Module (tvm::IRModule, tvm::Target)>::AssignTypedLambda<tvm::runtime::Module (*)(tvm::IRModule, tvm::Target)>(tvm::runtime::Module (*)(tvm::IRModule, tvm::Target), std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*) const
  3: tvm::codegen::BuildAMDGPU(tvm::IRModule, tvm::Target)
  2: tvm::codegen::CodeGenLLVM::Finish()
  1: tvm::codegen::CodeGenLLVM::Verify() const
  0: _ZN3tvm7runtime6detail
  File "/home/wzy/native-tvm/tvm/src/target/llvm/codegen_llvm.cc", line 354
  
TVMError: LLVM module verification failed with the following errors: 
Calling convention requires void return type
i32 (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*)* @tvmgen_default_fused_nn_conv2d_add_nn_relu_1_kernel
Function return type does not match operand type of return inst!
  ret void
 i32Calling convention requires void return type  
```



### auto-scheduler过程

##### 定义矩阵乘法

```python
@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te.placeholder((N, M), name="C", dtype=dtype)
	return [A, B, C, out]
```

##### 创建搜索任务

```python
target = tvm.target.Target("llvm")
N = L = M = 1024
task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(N, L, M, "float32"), target=target)

# 检查计算图
print("Computational DAG:")
print(task.compute_dag)
```

##### 设置auto-sheduler参数

```python
log_file = "matmul.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
```

##### 开始搜索

```python
# 运行 auto-tuning（搜索）
task.tune(tune_option)
# 应用最佳 schedule
sch, args = task.apply_best(log_file)
```

##### 检查优化的schedule

```python
print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))
```

##### 检查正确性并评估性能

```python
# Check results
np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000)
)
```

##### 使用记录文件

```python
print("Equivalent python schedule:")
print(task.print_best(log_file))
```



### sycl的auto-scheduler代码修改：

```
1.python/tvm/auto_scheduler/utils.py  
2.python/tvm/autotvm/tophub.py
3.src/auto_scheduler/search_policy/utils.h
4.src/auto_scheduler/search_task.cc
```

sycl的auto-scheduler代码的 traceback

```
Traceback (most recent call last):
  File "/home/wangziyang/tvm-sycl/git-tvm-sycl/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 81, in cfun
    rv = local_pyfunc(*pyargs)
  File "/home/wangziyang/tvm-sycl/git-tvm-sycl/tvm/python/tvm/auto_scheduler/cost_model/cost_model.py", line 98, in predict_func
    array_wrapper[:] = self.predict(task, states)
  File "/home/wangziyang/tvm-sycl/git-tvm-sycl/tvm/python/tvm/auto_scheduler/cost_model/xgb_model.py", line 232, in predict
    features = get_per_store_features_from_states(states, task)
  File "/home/wangziyang/tvm-sycl/git-tvm-sycl/tvm/python/tvm/auto_scheduler/feature.py", line 236, in get_per_store_features_from_states
    byte_arr = _ffi_api.GetPerStoreFeaturesFromStates(
  File "/home/wangziyang/tvm-sycl/git-tvm-sycl/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 227, in __call__
    _LIB.TVMFuncCall(
KeyboardInterrupt: 
Traceback (most recent call last):
  File "compile_optimizing_schedule.py", line 142, in <module>
    tuner.tune(tune_option)
  File "/home/wangziyang/tvm-sycl/git-tvm-sycl/tvm/python/tvm/auto_scheduler/task_scheduler.py", line 357, in tune
    self._tune_task(idx)
  File "/home/wangziyang/tvm-sycl/git-tvm-sycl/tvm/python/tvm/auto_scheduler/task_scheduler.py", line 452, in _tune_task
    measure_inputs, measure_results = self.search_policies[task_idx].continue_search_one_round(
  File "/home/wangziyang/tvm-sycl/git-tvm-sycl/tvm/python/tvm/auto_scheduler/search_policy.py", line 119, in continue_search_one_round
    return _ffi_api.SearchPolicyContinueSearchOneRound(self, num_measure, measurer)
  File "/home/wangziyang/tvm-sycl/git-tvm-sycl/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    raise get_last_ffi_error()
AttributeError: Traceback (most recent call last):
  6: TVMFuncCall
  5: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::runtime::Array<tvm::runtime::ObjectRef, void> (tvm::auto_scheduler::SearchPolicy, int, tvm::auto_scheduler::ProgramMeasurer)>::AssignTypedLambda<tvm::auto_scheduler::$_1>(tvm::auto_scheduler::$_1, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
  4: tvm::auto_scheduler::SketchPolicyNode::ContinueSearchOneRound(int, tvm::auto_scheduler::ProgramMeasurer)
  3: tvm::auto_scheduler::SketchPolicyNode::SearchOneRound(int, tvm::runtime::Array<tvm::auto_scheduler::State, void>*)
  2: tvm::auto_scheduler::SketchPolicyNode::EvolutionarySearch(tvm::runtime::Array<tvm::auto_scheduler::State, void> const&, int)
  1: tvm::auto_scheduler::PythonBasedModelNode::Predict(tvm::auto_scheduler::SearchTask const&, tvm::runtime::Array<tvm::auto_scheduler::State, void> const&, std::vector<float, std::allocator<float> >*)
  0: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<TVMFuncCreateFromCFunc::$_1> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
  2: TVMFuncCall
  1: tvm::NodeGetAttr(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
  0: tvm::ReflectionVTable::GetAttr(tvm::runtime::Object*, tvm::runtime::String const&) const
  File "/home/wangziyang/tvm-sycl/git-tvm-sycl/tvm/src/node/reflection.cc", line 109
AttributeError: tir.IterVar object has no attributed __array__
```

bug-1  

```
1.sycl/plugins/cuda/pi_cuda.cpp  pi_result cuda_piQueueFinish(pi_queue command_queue)
2.sycl/plugins/cuda/pi_cuda.cpp  _PI_CL(piQueueFinish, cuda_piQueueFinish)
3.sycl/source/detail/queue_impl.cpp void queue_impl::wait(const detail::code_location &CodeLoc)
4.sycl/include/sycl/queue.hpp  wait(){wait_proxy();}
5.sycl/source/sycl/queue.cpp wait_proxy{impl->wait();}

PI CUDA ERROR:
        Value:           700
        Name:            CUDA_ERROR_ILLEGAL_ADDRESS
        Description:     an illegal memory access was encountered
        Function:        operator()
        Source Location: /home/wangziyang/sycl_workspace/intel-llvm-new/sycl/plugins/cuda/pi_cuda.cpp:2645
```

bug-2

```
1.sycl/plugins/cuda/pi_cuda.cpp  pi_result cuda_piEnqueueKernelLaunch(
    pi_queue command_queue, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) 
    
2.sycl/plugins/cuda/pi_cuda.cpp   _PI_CL(piEnqueueKernelLaunch, cuda_piEnqueueKernelLaunch)

3.sycl/include/sycl/handler.hpp  event finalize();
4.sycl/source/handler.cpp  event handler::finalize();

PI CUDA ERROR:
        Value:           701
        Name:            CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
        Description:     too many resources requested for launch
        Function:        cuda_piEnqueueKernelLaunch
        Source Location: /home/wangziyang/sycl_workspace/intel-llvm-new/sycl/plugins/cuda/pi_cuda.cpp:3179
```

bug-3

```cpp
PI CUDA ERROR:
        Value:           1
        Name:            CUDA_ERROR_INVALID_VALUE
        Description:     invalid argument
        Function:        cuda_piEnqueueKernelLaunch
        Source Location: /home/wangziyang/sycl_workspace/intel-llvm-new/sycl/plugins/cuda/pi_cuda.cpp:3179
```

bug-4

```cpp
In file included from /tmp/tvm_sycl/sycl_4083715_1.cc:2:
In file included from /home/wangziyang/sycl_workspace/build-cuda-2022-12/bin/../include/sycl/CL/sycl.hpp:11:
In file included from /home/wangziyang/sycl_workspace/build-cuda-2022-12/bin/../include/sycl/sycl.hpp:11:
In file included from /home/wangziyang/sycl_workspace/build-cuda-2022-12/bin/../include/sycl/accessor.hpp:28:
In file included from /home/wangziyang/sycl_workspace/build-cuda-2022-12/bin/../include/sycl/image.hpp:18:
/home/wangziyang/sycl_workspace/build-cuda-2022-12/bin/../include/sycl/types.hpp:870:18: error: no matching function for call to 'convertImpl'
                 detail::convertImpl<vec_data_t<DataT>, vec_data_t<convertT>,
                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/tmp/tvm_sycl/sycl_4083715_1.cc:40:630: note: in instantiation of function template specialization 'sycl::vec<int, 4>::convert<bool, sycl::rounding_mode::automatic>' requested here
  ...2, 2, 2})) - (vec<int, 4>{1, 1, 1, 1})), (((int4){(0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3)}) / (vec<int, 4>{2, 2, 2, 2})), (((((((vec<int, 4>{2, 2, 2, 2}) >= (vec<int, 4>{0, 0, 0, 0})).convert<bool>()) && (((((int4){(0)+(1*...
```

##### traceback auto-scheduler

python

```python
1.auto_scheduler.extract_tasks(mod["main"], target=target, params=params)
2.tvm/python/auto_scheduler/relay_integration.py
def extract_tasks(mod,params,target,...):
    ...
    tasks.append(SearchTask(...))
    weights.append(int(weight))
    ...
    return tasks,weights
3.tvm/python/tvm/auto_scheduler/search_task.py 
@tvm._ffi.register_object("auto_scheduler.SearchTask")
class SearchTask(Object):
    ...
    self.__init_handle_by_constructor__(
            _ffi_api.SearchTask,
            compute_dag,
            workload_key,
            target,
            target_host,
            hardware_params,
            layout_rewrite_option,
            task_input_names,
            desc,
        )

```

cpp

```cpp
4.tvm/incldue/tvm/auto_scheduler/search_task.h 
class SearchTaskNode : public Object {
  ...
  static constexpr const char* _type_key = "auto_scheduler.SearchTask";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchTaskNode, Object);
}

5.tvm/src/auto_sheduler/search_task.cc
TVM_REGISTER_GLOBAL("auto_scheduler.SearchTask")
    .set_body_typed([](ComputeDAG compute_dag, String workload_key, Target target,
                       Target target_host, Optional<HardwareParams> hardware_params,
                       int layout_rewrite_option, Array<String> task_input_names, String desc) {
      CheckAndUpdateHostConsistency(&target, &target_host);
      return SearchTask(compute_dag, workload_key, target, target_host, hardware_params,
                        LayoutRewriteOption(layout_rewrite_option), task_input_names, desc);
    });    

6.tvm/src/auto_sheduler/search_task.cc
SearchTask::SearchTask(ComputeDAG compute_dag, String workload_key, Target target,
                       Target target_host, Optional<HardwareParams> hardware_params,
                       LayoutRewriteOption layout_rewrite_option, Array<String> task_input_names,
                       String desc) {
    ...
    auto node = make_object<SearchTaskNode>();
    ...
        if (hardware_params) {
            node->hardware_params = hardware_params.value();
        } else {
            node->hardware_params =
                HardwareParamsNode::GetDefaultHardwareParams(node->target, node->target_host);
        }  
    ...        
}    
7.tvm/src/auto_sheduler/search_task.cc
HardwareParams HardwareParamsNode::GetDefaultHardwareParams(const Target& target,
                                                            const Target& target_host){
    ...
    return HardWareParams;
}        
```

python

```python
1.tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
2.tvm/python/auto_scheduler/task_sheduler.py 
class TaskScheduler:
    """
    Allocate the time resources when tuning multiple tasks together.
    This implements two strategies: "round-robin" and "gradient".  
    """
    # Build similarity groups
    self.task_tags = []  # task_id -> tag
    self.tag_to_group_id = {}  # tag -> group_id
    self.group_task_ids = []  # group_id -> all task ids in this group
    self.flop_cts = []  # task_id -> the number of floating ops
    for i, task in enumerate(self.tasks):
        tag = derive_similarity_tag(task.compute_dag)
        self.task_tags.append(tag)
        self.flop_cts.append(task.compute_dag.flop_ct)
        if not tag:
            continue

        if tag not in self.tag_to_group_id:
           self.tag_to_group_id[tag] = len(self.tag_to_group_id)
           self.group_task_ids.append([])
        self.group_task_ids[self.tag_to_group_id[tag]].append(i)
```

python

```python
1.tuner.tune(tune_option)
2.tvm/python/auto_scheduler/task_sheduler.py 
def tune(
        self,
        tune_option,
        search_policy="default",
        search_policy_params=None,
        adaptive_training=False,
        per_task_early_stopping=None,
    ):
    """Tune a batch of tasks together."""
        # use the specific strategy to choose workload to tune
        task_idx = -1
        while self.ct < tune_option.num_measure_trials and len(self.dead_tasks) < len(self.tasks):    
            if self.strategy == "round-robin":
                ...
            elif self.strategy == "gradient":
                ...
            else:
                raise ValueError("Invalid strategy: "+self.strategy)
            self._tune_task(task_idx)
            self._adjust_similarity_group(task_idx)
            
3. tvm/python/auto_scheduler/task_sheduler.py 
def _tune_task(self, task_idx):
    """Tune the select task for one round"""
    # Run pre-tune callbacks
    for callback in self.callbacks:
        callback.pre_tune(self, task_idx)

     measure_inputs, measure_results = 		   self.search_policies[task_idx].continue_search_one_round(
            self.num_measures_per_round, self.measurer
        )
4.tvm/python/tvm/auto_scheduler/search_policy.py
@tvm._ffi.register_object("auto_scheduler.SearchPolicy")
class SearchPolicy(Object):
    """The base class of search policies."""

    def continue_search_one_round(self, num_measure, measurer):
        """
        Continue the search by doing an additional search round.
        """
        return _ffi_api.SearchPolicyContinueSearchOneRound(self, num_measure, measurer)
```

cpp

```cpp
5.include/tvm/auto_scheduler/search_policy.cc
TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyContinueSearchOneRound")
    .set_body_typed([](SearchPolicy policy, int num_measure, ProgramMeasurer measurer) {
      auto [inputs, results] = policy->ContinueSearchOneRound(num_measure, measurer);
      return Array<ObjectRef>{inputs, results};
    });

6.src/auto_scheduler/search_policy/empty_policy.cc
std::pair<Array<MeasureInput>, Array<MeasureResult>> EmptyPolicyNode::ContinueSearchOneRound() {
    // Search one round to get promising states
  PrintTitle("Search", verbose);
  best_states = SearchOneRound();

  // Measure these states
  PrintTitle("Measure", verbose);
  for (const auto& state : best_states) {
    inputs.push_back(MeasureInput(search_task, state));
  }
  results = measurer->Measure(search_task, GetRef<SearchPolicy>(this), inputs);
}   

7.src/auto_sheduler/measure.cc
Array<MeasureResult> ProgramMeasurerNode::Measure(const SearchTask& task,
                                                  const SearchPolicy& policy,
                                                  const Array<MeasureInput>& inputs,
                                                  int batch_size) {
    ...
    // build and run
    SilentMeasure(task, input_batch, &result_batch);
    ...
}

8.src/auto_scheduler/measure.cc
void ProgramMeasurerNode::SilentMeasure(const SearchTask& task, const Array<MeasureInput>& inputs,
                                        Array<MeasureResult>* results) {
  // Call builder and runner
  Array<BuildResult> build_res_batch = builder->Build(inputs, verbose);
  Array<MeasureResult> result_batch = runner->Run(inputs, build_res_batch, verbose);
}    

9.src/auto_scheduler/measure.cc
Array<BuildResult> LocalBuilderNode::Build(const Array<MeasureInput>& inputs, int verbose) {
    if (const auto* f = runtime::Registry::Get("auto_scheduler.local_builder.build")) {
    Array<BuildResult> results = (*f)(inputs, timeout, n_parallel, build_func, verbose);
    return results;
  }
}  

12.src/auto_scheduler/measure.cc
Array<MeasureResult> LocalRunnerNode::Run(const Array<MeasureInput>& inputs,
                                          const Array<BuildResult>& build_results, int verbose) {
      if (const auto* f = runtime::Registry::Get("auto_scheduler.local_runner.run")) {
    Array<MeasureResult> results =
        (*f)(inputs, build_results, timeout, number, repeat, min_repeat_ms, cooldown_interval,
             enable_cpu_cache_flush, verbose, device);
    return results;
  }
}    
    

```

python

```python
10.python/tvm/auto_scheduler/measuer.py

@tvm._ffi.register_func("auto_scheduler.local_builder.build")
def local_builder_build(inputs, timeout, n_parallel, build_func="default", verbose=1):
    """
    Build function of LocalBuilder to build the MeasureInputs to runnable modules.
    """
    ...
        executor = PopenPoolExecutor(
        n_parallel, timeout, reset_global_scope, (AutotvmGlobalScope.current,)
    )
    ...

11.python/tvm/contrib/popen_pool.py
class PopenPoolExecutor:
    """An parallel executor backed by Popen processes.
    """
    if max_workers is None:
       max_workers = os.cpu_count()
    # Use an internal thread pool to send to popen workers
    self._threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    

13.python/tvm/auto_scheduler/measure.py
@tvm._ffi.register_func("auto_scheduler.local_runner.run")
def local_run(
    inputs,
    build_results,
    timeout=10,
    number=3,
    repeat=1,
    min_repeat_ms=0,
    cooldown_interval=0,
    enable_cpu_cache_flush=False,
    verbose=1,
    device=0,
):
        """
    Run function of LocalRunner to test the performance of the input BuildResults."""
	worker = PopenWorker()
    for inp, build_res in zip(inputs, build_results):  
        if build_res.error_no != 0:
            ...
        else:
            args = prepare_runner_args(inp, build_res)
            res = call_func_with_timeout(
                worker,
                timeout,
                _timed_eval_func,
                args=(
                    inp.serialize(),
                    build_res,
                    args,
                    number,
                    repeat,
                    min_repeat_ms,
                    cooldown_interval,
                    enable_cpu_cache_flush,
                    verbose,
                    device,
                ),
            ) 
            
14.python/tvm/auto_scheduler/utils.py
def call_func_with_timeout(
    worker, timeout, func, args=(), kwargs=None
):  # pylint: disable=unused-argument
    """Call a function with timeout"""
    worker.send(func, args, kwargs, timeout)
    try:
        res = worker.recv()
    except Exception:  # pylint: disable=broad-except
        res = Exception(make_traceback_info())

    return res
```

