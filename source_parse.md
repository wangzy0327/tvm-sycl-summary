tvm源到源代码生成

python

```python
1.with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
2. python/tvm/relay/build_module.py

from .backend import executor_factory as _executor_factory
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
        if executor.name == "aot":
            ...
	    elif executor.name = "graph":
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
            assert False, "Executor " + executor + " not supported"
        return executor_factory        

3.python/tvm/relay/backend/executor_factory.py

class GraphExecutorFactoryModule(ExecutorFactoryModule):
    def __init__(
        self,
        ir_mod,
        target,
        executor,
        graph_json_str,
        libmod,
        libmod_name,
        params,
        function_metadata,
    ):
        ...
        fcreate = get_global_func("tvm.graph_executor_factory.create")
        self.module = fcreate(graph_json_str, libmod, libmod_name, *args)
        ...

4.python/tvm/_ffi/registry.py

def get_global_func(name, allow_missing=False):
    return _get_global_func(name, allow_missing)

5.python/tvm/_ffi/_ctypes/packed_func.py

PackedFuncHandle = ctypes.c_void_p

def _get_global_func(name, allow_missing=False):
    handle = PackedFuncHandle()
    check_call(_LIB.TVMFuncGetGlobal(c_str(name), ctypes.byref(handle)))
    if handle.value:
        return _make_packed_func(handle, False)

6.    
```

cpp

```cpp
6. src/runtime/graph_executor/graph_executor_factory.cc
TVM_REGISTER_GLOBAL("tvm.graph_executor_factory.create")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ...
      std::unordered_map<std::string, tvm::runtime::NDArray> params;
      for (size_t i = 3; i < static_cast<size_t>(args.size()); i += 2) {
        std::string name = args[i].operator String();
        params[name] = args[i + 1].operator tvm::runtime::NDArray();
      }
      auto exec = make_object<GraphExecutorFactory>(args[0], params, args[2]);
      exec->Import(args[1]);
      *rv = Module(exec);
    });
7.src/runtime/graph_executor/graph_executor_factory.h
 class TVM_DLL GraphExecutorFactory : public runtime::ModuleNode {
      public:
  /*!
   * \brief Construct the GraphExecutorFactory.
   * \param graph_json The execution graph.
   * \param params The params of graph.
   * \param module_name The module name of graph.
   */
  GraphExecutorFactory(const std::string& graph_json,
                       const std::unordered_map<std::string, tvm::runtime::NDArray>& params,
                       const std::string& module_name = "default");
 }   
8.
```

python

```python
1.dev = tvm.device(str(target), 1)
  module = graph_executor.GraphModule(lib["default"](dev))
2.python/tvm/contrib/graph_executor.py
class GraphModule(object):
	 """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions"""
    def __init__(self, module):
        self.module = module
        self._set_input = module["set_input"]
        self._run = module["run"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        self._get_num_outputs = module["get_num_outputs"]
        self._get_input_index = module["get_input_index"]
        self._get_input_info = module["get_input_info"]
        self._get_num_inputs = module["get_num_inputs"]
        self._load_params = module["load_params"]
        self._share_params = module["share_params"]    
```

