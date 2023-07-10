## IRModule

```python
1.lib = relay.build(mod, target=target, params=params)
2.python/tvm/relay/build_module.py
def build(
    ir_mod,
    target=None,
    target_host=None,
    executor=Executor("graph"),
    runtime=Runtime("cpp"),
    workspace_memory_pools=None,
    constant_memory_pools=None,
    params=None,
    mod_name="default",
):
"""Helper function that builds a Relay function to run on TVM graph executor.

    Parameters
    ----------
    ir_mod : :py:class:`~tvm.IRModule`
        The IR module to build. Using relay.Function is deprecated.

    target : None, or any multi-target like object, see Target.canon_multi_target
        For homogeneous compilation, the unique build target.
        For heterogeneous compilation, a dictionary or list of possible build targets.
        Defaults to the current target in the environment if None.
    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.        
	Returns
    -------
    factory_module : tvm.relay.backend.executor_factory.ExecutorFactoryModule
            The runtime factory for the TVM graph executor.
    """
with tophub_context:
        bld_mod = BuildModule()
        graph_json, runtime_mod, params = bld_mod.build(
            mod=ir_mod,
            target=raw_targets,
            params=params,
            executor=executor,
            runtime=runtime,
            workspace_memory_pools=workspace_memory_pools,
            constant_memory_pools=constant_memory_pools,
            mod_name=mod_name,
        )
        func_metadata = bld_mod.get_function_metadata()
        devices = bld_mod.get_devices()
        lowered_ir_mods = bld_mod.get_irmodule()
        executor_codegen_metadata = bld_mod.get_executor_codegen_metadata()
if executor.name == "aot":        
    ...
elif executor.name == "graph":
            executor_factory = _executor_factory.GraphExecutorFactoryModule(
                ir_mod,
                raw_targets,
                executor,
                graph_json,
                runtime_mod,
                mod_name,
                params,
                func_metadata,
            )  
else:
    ...
return executor_factory  

3.python/tvm/relay/build_module.py
class BuildModule(object):
    """Build an IR module to run on TVM graph executor. This class is used
    to expose the `RelayBuildModule` APIs implemented in C++.
    """
    def __init__(self):
        # 4.python/tvm/relay/_build_module.py
        self.mod = _build_module._BuildModule()
        self._get_graph_json = self.mod["get_graph_json"]
        self._get_module = self.mod["get_module"]
        self._build = self.mod["build"]
        self._optimize = self.mod["optimize"]
        self._set_params_func = self.mod["set_params"]
        self._get_params_func = self.mod["get_params"]
        self._get_function_metadata = self.mod["get_function_metadata"]
        self._get_executor_codegen_metadata = self.mod["get_executor_codegen_metadata"]
        self._get_devices = self.mod["get_devices"]
    
    def build(
        self,
        mod,
        target=None,
        target_host=None,
        executor=Executor("graph"),
        runtime=Runtime("cpp"),
        workspace_memory_pools=None,
        constant_memory_pools=None,
        params=None,
        mod_name=None,
    ):
        """
        Parameters
        ----------
        mod : :py:class:`~tvm.IRModule`
            The IRModule to build.

        target : any multi-target like object, see Target.canon_multi_target
            For homogeneous compilation, the unique build target.
            For heterogeneous compilation, a dictionary or list of possible build targets.

        target_host : None, or any target-like object, see Target.canon_target
            Host compilation target, if target is device.
            When TVM compiles device specific program such as CUDA,
            we also need host(CPU) side code to interact with the driver
            to setup the dimensions and parameters correctly.
            target_host is used to specify the host side codegen target.
            By default, llvm is used if it is enabled,
            otherwise a stackvm interpreter is used.

        executor : Optional[Executor]
            The executor configuration with which to build the model.
            Defaults to "graph" if no executor specified.

        runtime : Optional[Runtime]
            Runtime configuration to use when building the model.
            Defaults to "cpp" if no runtime specified.
		params : dict of str to NDArray
            Input parameters to the graph that do not change
            during inference time. Used for constant folding.

        mod_name: Optional[str]
            The module name we will build

        Returns
        -------
        graph_json : str
            The json string that can be accepted by graph executor.

        mod : tvm.Module
            The module containing necessary libraries.

        params : dict
            The parameters of the final graph.
        """            
		...
        mod_name = mangle_module_name(mod_name)

        self._build(
            mod,
            target,
            target_host,
            executor,
            runtime,
            workspace_memory_pools,
            constant_memory_pools,
            mod_name,
        )
        # mod_name = "tvmgen_default"
        # target = [sycl -keys=sycl,gpu -max_num_threads=1024 -thread_warp_size=32]
        # runtime = Runtime("cpp")
        # executor = graph{"link-params":(bool)0}
        ...
        # Get artifacts
        mod = self.get_module()
        params = self.get_params()
        executor_config = self.get_graph_json() if executor.name == "graph" else None

        return executor_config, mod, params
    
4.python/tvm/relay/_build_module.py
tvm._ffi._init_api("relay.build_module", __name__)
"""
src/relay/backend/build_module.cc

TVM_REGISTER_GLOBAL("relay.build_module._BuildModule").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = RelayBuildCreate();
});
"""

```

cpp

```cpp
5.src/relay/backend/build_module.cc
TVM_REGISTER_GLOBAL("relay.build_module._BuildModule").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = RelayBuildCreate();
});

/*!
 * \brief Relay build module
 *
 */
class RelayBuildModule : public runtime::ModuleNode {
 public:
  RelayBuildModule() = default;

    /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
      if (name == "get_graph_json") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetGraphJSON(); });
    } else if (name == "get_module") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetModule(); });
    } else if (name == "build") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 8);
        this->Build(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
      });
	} else if (name == "get_params") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetParams(); });
    } else if (name == "set_params") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Map<String, Constant> params = args[0];
        for (const auto& kv : params) {
          this->SetParam(kv.first, kv.second->data);
        }
      });
    } else if (name == "get_devices") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->executor_codegen_->ListDevices();
      });
    } else if (name == "get_irmodule") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->executor_codegen_->GetIRModule();
      });
    }else{}
      ...
  }
/*!
   * \brief Build relay IRModule for graph executor
   *
   * \param mod Relay IRModule
   * \param raw_targets List of available targets for kernels.
   * \param executor Executor to target
   * \param runtime Runtime to codegen for
   * \param mod_name Name of the module
   */
  void Build(IRModule mod, const Array<Target>& raw_targets, const tvm::Target& target_host,
             const Executor& executor, const Runtime& runtime,
             const WorkspaceMemoryPools& workspace_memory_pools,
             const ConstantMemoryPools& constant_memory_pools, const String mod_name) {
    /*
    
    */ 
    VLOG_CONTEXT << "Build";
    executor_ = executor;
    runtime_ = runtime;
    workspace_memory_pools_ = workspace_memory_pools;
    constant_memory_pools_ = constant_memory_pools;
    config_ = CompilationConfig(PassContext::Current(), raw_targets);
    VLOG(1) << "Using compilation config:" << std::endl << config_;
    BuildRelay(std::move(mod), mod_name);
  }
/*!
   * \brief Compile a Relay IR module to runtime module.
   *
   * \param relay_module The Relay IR module.
   * \param params The parameters.
   */
  void BuildRelay(IRModule relay_module, const String& mod_name) {
    // Relay IRModule -> IRModule optimizations.
    IRModule module = WithAttrs(
        relay_module, {{tvm::attr::kExecutor, executor_}, {tvm::attr::kRuntime, runtime_}});
    relay_module = OptimizeImpl(std::move(module));

    // Get the updated function and new IRModule to build.
    // Instead of recreating the IRModule, we should look at the differences between this and the
    // incoming IRModule to see if we can just pass (IRModule, Function) to the code generator.
    Function func = Downcast<Function>(relay_module->Lookup("main"));
    IRModule func_module = WithAttrs(IRModule::FromExpr(func),
                                     {{tvm::attr::kExecutor, executor_},
                                      {tvm::attr::kRuntime, runtime_},
                                      {tvm::attr::kWorkspaceMemoryPools, workspace_memory_pools_},
                                      {tvm::attr::kConstantMemoryPools, constant_memory_pools_}});

    // Generate code for the updated function.
    executor_codegen_ = MakeExecutorCodegen(executor_->name);
    // -> 6  src/relay/backend/build_module.cc
    executor_codegen_->Init(nullptr, config_->primitive_targets);
    // -> 6  src/relay/backend/build_module.cc
    executor_codegen_->Codegen(func_module, func, mod_name);
    // -> 6 src/relay/backend/build_module.cc UpdateOutput -> GetGraphJSON
    executor_codegen_->UpdateOutput(&ret_);
    // -> 6 src/relay/backend/build_module.cc GetParams -> CallFunc(list_params_name) & CallFunc(get_param_by_name)
    ret_.params = executor_codegen_->GetParams();

    auto lowered_funcs = executor_codegen_->GetIRModule();

    // No need to build for external functions.
    Target ext_dev("ext_dev");
    if (lowered_funcs.find(ext_dev) != lowered_funcs.end()) {
      lowered_funcs.Set(ext_dev, IRModule());
    }

    const Target& host_target = config_->host_virtual_device->target;
    const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
    // When there is no lowered_funcs due to reasons such as optimization.
    if (lowered_funcs.size() == 0) {
      if (host_target->kind->name == "llvm") {
        CHECK(pf != nullptr) << "Unable to create empty module for llvm without llvm codegen.";
        // If we can decide the target is LLVM, we then create an empty LLVM module.
        ret_.mod = (*pf)(host_target->str(), "empty_module");
      } else {
        // If we cannot decide the target is LLVM, we create an empty CSourceModule.
        // The code content is initialized with ";" to prevent complaining
        // from CSourceModuleNode::SaveToFile.
        ret_.mod = tvm::codegen::CSourceModuleCreate(";", "", Array<String>{});
      }
    } else {
      // call
      ret_.mod = tvm::TIRToRuntime(lowered_funcs, host_target);
    }

    ...
  }    

6.src/relay/backend/build_module.cc
struct ExecutorCodegen {
  void Init(runtime::Module* m, const Array<Target>& raw_targets) {
    // callFunc
    CallFunc("init", m, raw_targets);
    //return -> 5.src/relay/backend/build_module.cc
  }

  void Codegen(IRModule mod, const Function& func, String mod_name) {
    // -> 7.2 include
    CallFunc("codegen", mod, func, mod_name);
  } 
    
  std::unordered_map<std::string, tvm::runtime::NDArray> GetParams() {
    std::unordered_map<std::string, tvm::runtime::NDArray> ret;
    auto names = CallFunc<Array<runtime::String>>("list_params_name", nullptr);
    for (const auto& expr : names) {
      // Implicit cast from runtime::String to std::string
      std::string key = expr;
      ret[key] = CallFunc<runtime::NDArray>("get_param_by_name", key);
    }
    return ret;
  }
    
 protected:
  tvm::runtime::Module mod;
  template <typename R, typename... Args>
  R CallFunc(const std::string& name, Args... args) {
    auto pf = mod.GetFunction(name, false);
    return pf(std::forward<Args>(args)...);
  }
  template <typename... Args>
  void CallFunc(const std::string& name, Args... args) {
    // -> 7.include/tvm/runtime/packed_func.h
    auto pf = mod.GetFunction(name, false);
    pf(std::forward<Args>(args)...);
    return;
  }
}
    
/*!
 * \brief GraphCodegen module wrapper
 *
 */
struct GraphCodegen : ExecutorCodegen {
  GraphCodegen() {
    auto pf = GetPackedFunc("relay.build_module._GraphExecutorCodegen");
    mod = (*pf)();
  }
  void UpdateOutput(BuildOutput* ret) override { ret->graph_json = GetGraphJSON(); }

  std::string GetGraphJSON() { return CallFunc<std::string>("get_graph_json", nullptr); }

  ~GraphCodegen() {}
};    
  
    
7. include/tvm/runtime/packed_func.h

inline PackedFunc Module::GetFunction(const std::string& name, bool query_imports) {
  return (*this)->GetFunction(name, query_imports);
  //return 6.src/relay/backend/build_module.cc
  //return ModuleNode    
} 
template <typename... Args>
inline TVMRetValue PackedFunc::operator()(Args&&... args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  TVMValue values[kArraySize];
  int type_codes[kArraySize];
  detail::for_each(TVMArgsSetter(values, type_codes), std::forward<Args>(args)...);
  TVMRetValue rv;
  (static_cast<PackedFuncObj*>(data_.get()))
      ->CallPacked(TVMArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}  
    

    
8.include/tvm/runtime/module.h

    /*!
 * \brief Base container of module.
 *
 * Please subclass ModuleNode to create a specific runtime module.
 *
 * \code
 *
 *  class MyModuleNode : public ModuleNode {
 *   public:
 *    // implement the interface
 *  };
 *
 *  // use make_object to create a specific
 *  // instace of MyModuleNode.
 *  Module CreateMyModule() {
 *    ObjectPtr<MyModuleNode> n =
 *      tvm::runtime::make_object<MyModuleNode>();
 *    return Module(n);
 *  }
 *
 * \endcode
 */
class TVM_DLL ModuleNode : public Object {
    /*!
   * \brief Get a PackedFunc from module.
   *
   *  The PackedFunc may not be fully initialized,
   *  there might still be first time running overhead when
   *  executing the function on certain devices.
   *  For benchmarking, use prepare to eliminate
   */
  virtual PackedFunc GetFunction(const std::string& name,
                                 const ObjectPtr<Object>& sptr_to_self) = 0;
    /*!
   * \brief Save the module to file.
   * \param file_name The file to be saved to.
   * \param format The format of the file.
   */  
  virtual void SaveToFile(const std::string& file_name, const std::string& format);  
  /*!
   * \brief Save the module to binary stream.
   * \param stream The binary stream to save to.
   * \note It is recommended to implement this for device modules,
   *   but not necessarily host modules.
   *   We can use this to do AOT loading of bundled device functions.
   */    
  virtual void SaveToBinary(dmlc::Stream* stream);    
}    
inline ModuleNode* Module::operator->() { return static_cast<ModuleNode*>(get_mutable()); }

9.include/tvm/runtime/object.h
/*! \brief Base class of all object reference */
class ObjectRef {
  protected:
      /*! \return return a mutable internal ptr, can be used by sub-classes. */
  Object* get_mutable() const { return data_.get(); }
}

10.include/tvm/runtime/object.h    
/*!
 * \brief A custom smart pointer for Object.
 * \tparam T the content data type.
 * \sa make_object
 */
template <typename T>
class ObjectPtr {
  /*!
   * \return Get the content of the pointer
   */
  T* get() const { return static_cast<T*>(data_); }
}    
    
11. src/runtime/module.cc
    
PackedFunc ModuleNode::GetFunction(const std::string& name, bool query_imports) {
  ModuleNode* self = this;
  PackedFunc pf = self->GetFunction(name, GetObjectPtr<Object>(this));
  //GraphExecutorCodegenModule : public runtime::ModuleNode
  // return -> 7.include/tvm/runtime/packed_func.h 
  if (pf != nullptr) return pf;
  if (query_imports) {
    for (Module& m : self->imports_) {
      pf = m.operator->()->GetFunction(name, query_imports);
      if (pf != nullptr) {
        return pf;
      }
    }
  }
  return pf;
}

12.src/relay/backend/graph_executor_codegen.cc
class GraphExecutorCodegenModule : public runtime::ModuleNode {
 public:
  GraphExecutorCodegenModule() {}
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "init") {
      //init
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 2) << "The expected of arguments are: "
                                    << "runtime::Module mod and Array<Target> targets";
        void* mod = args[0];
        Array<Target> targets = args[1];
        codegen_ = std::make_shared<GraphExecutorCodegen>(reinterpret_cast<runtime::Module*>(mod),
                                                          std::move(targets));
      });
      // -> 11. src/runtime/module.cc
    }else if(name == "codegen"){
      ...  
    }else{
       ...   
    }
  }
    
13.src/driver/driver_api.cc
    
runtime::Module TIRToRuntime(const Map<Target, IRModule>& inputs_arg,
                             const Target& target_host_arg) {
  auto pass_ctx = transform::PassContext::Current();

  std::vector<runtime::Module> device_modules;
  Map<Target, IRModule> inputs = inputs_arg;
  ...
  for (const auto& it : inputs) {
      if (it.second.defined()) {
      ...
      // We don't want library modules going back into host codegen
      // unless they're supposed to. Here if we overrode the target host
      // to allow lowering previously we check that it's meant to be placed
      // back into the host Module.
      bool overrides_host_target = target->kind->device_type == target_host->kind->device_type;
      bool non_host_target_kind = target->kind != target_host->kind;
      if (overrides_host_target && non_host_target_kind) {
        device_modules.push_back(codegen::Build(host_mod, it.first));
      } else {
        mhost_all->Update(host_mod);
      }

      if (device_mod->functions.size() != 0) {
        // call 15.src/target/codegen.cc Build(IRModule mod, Target target)
        device_modules.push_back(codegen::Build(device_mod, it.first));
      }
      }
  }      
}    

14.src/driver/internel_driver_api.h

/*!
 * \brief Build a device and host module for a specific target from a map
 * contains target to IRModule. This function is used
 * for heterogeneous build.
 * \param input The map contains target to an IRModule.
 * \param target_host The target for building host code. To use the default,
 *        pass Target().
 * \return The built module that contains code for different processors.
 */
runtime::Module TIRToRuntime(const Map<Target, IRModule>& input, const Target& target_host);  
    
15.src/target/codegen.cc
runtime::Module Build(IRModule mod, Target target) {
  ...
  auto target_attr_map = tvm::TargetKind::GetAttrMap<FTVMTIRToRuntime>("TIRToRuntime");
  if (target_attr_map.count(target->kind)) {
    return target_attr_map[target->kind](mod, target);
  }

  // the build function.
  std::string build_f_name = "target.build." + target->kind->name;
  //call 16. src/target/source/codegen_sycl.cc 
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  ICHECK(bf != nullptr) << build_f_name << " is not enabled";
  return (*bf)(mod, target);
} 
    
16. src/target/source/codegen_sycl.cc
TVM_REGISTER_GLOBAL("target.build.sycl").set_body_typed(BuildSYCL); 

//生成codegen SYCL源码然后执行编译    
runtime::Module BuildSYCL(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  std::stringstream code;
  code << "// tvm target: " << target->str() << "\n";
  code << "#include <CL/sycl.hpp>\n";
  code << "using namespace sycl;\n";

  const auto* fpostproc = Registry::Get("tvm_callback_opencl_postproc");
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenSYCL: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenSYCL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    CodeGenSYCL cg;
    // call 
    cg.Init(output_ssa);
    // call src/target/source/codegen_sycl.cc  AddFunction
    cg.AddFunction(f);
    std::string fsource = cg.Finish();
    // Debug for SYCL
    VLOG(0) << "BuildSYCL: code:\n" << fsource;
    if (fpostproc) {
      fsource = (*fpostproc)(fsource).operator std::string();
    }
    code << fsource;
  }
  //call 18 src/runtime/sycl/sycl_module.cc
  return SYCLModuleCreate(code.str(), "sycl", ExtractFuncInfo(mod), code.str());
}    
    

17.src/runtime/registry.cc
const PackedFunc* Registry::Get(const std::string& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) return nullptr;
  return &(it->second->func_);
} 
    
18.src/rutime/sycl/sycl_module.cc
Module SYCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  auto n = make_object<SYCLModuleNode>(data, fmt, fmap, source);
  // call 20 src/runtime/sycl/sycl_module.cc
  n->Init();
  return Module(n);
} 

19.src/runtime/sycl/sycl_module.h
/*!
 * \brief create a sycl module for GPU devices from data.
 *
 * \param data The module data.
 * \param fmt The format of the data, can be "so"
 * \param fmap The map function information map of each function.
 */
Module SYCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source);
    
    
20.src/runtime/sycl/sycl_module.cc
void SYCLModuleNode::Init(){
  workspace_ = GetGlobalWorkspace();
  workspace_->Init();
  ...
  // sycl kernel source code
  //sycl生成的源码编译
  std::ofstream kernels_file;
  kernels_file.open(this->lib_compiler.source_file_path);
  kernels_file << GetSource("sycl");
  kernels_file.close();
  // compile kernel source code to share libary
  std::cout<<"[SYCL] Compile kernels source code(" + this->lib_compiler.source_file_path + ") to share library."<<std::endl;
}
    
21.PackedFunc SYCLModuleNode::GetFunction(const std::string& name,
                                         const ObjectPtr<Object>& sptr_to_self) {
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  SYCLWrappedFunc f;
  std::vector<size_t> arg_size(info.arg_types.size());  
  ...
  VLOG(0) << "SYCLModuleNode::begin initialize the wrapped func:";
  // initialize the wrapped func.
  f.Init(this, name, info.arg_types.size(), info.launch_param_tags, so_handler_);
  VLOG(0) << "SYCLModuleNode::finish initialize the wrapped func:";
  //调用SYCLWrappedFunc去执行
  return PackFuncVoidAddr(f, info.arg_types);      
}    
    
22.
class SYCLWrappedFunc {
   public:
  // initialize the SYCL function.
  void Init(SYCLModuleNode* m, std::string func_name, \
    size_t num_void_args, const std::vector<std::string>& launch_param_tags, void *so_handler) {
    w_ = m->GetGlobalWorkspace();
    func_name_ = func_name;
    so_handler_ = so_handler;
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
      ICHECK(w_->devices.size() != 0) << "No SYCL device";
    // get kernel
    void (*kernel_func)(sycl::queue &Q, sycl::range<3> k0_dimGrid, sycl::range<3> k0_dimBlock, void** void_args) = (void (*)(sycl::queue &Q, sycl::range<3> k0_dimGrid, sycl::range<3> k0_dimBlock, void** void_args))dlsym(so_handler_, func_name_.c_str());
    // get thread dimension
    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    sycl::range<3> k0_dimGrid(wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2));
    sycl::range<3> k0_dimBlock(wl.block_dim(2), wl.block_dim(1), wl.block_dim(0));  
    ...
    syclT::SYCLThreadEntry* t = w_->GetThreadEntry();
    sycl::queue Queue = w_->GetQueue(t->device);  
    //真正调用执行kernel func
    SYCL_CALL(kernel_func(Queue, k0_dimGrid, k0_dimBlock, void_args));      
  }
}    
    
```






