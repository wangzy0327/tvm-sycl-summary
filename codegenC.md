## CodeGenC

```cpp
/*CodeGenC does not aim at generating C codes consumed by MSVC or GCC,
 * Rather, it's providing infrastructural abstraction for C variants like CUDA
 * and OpenCL-C. You might find some odd variant features, e.g., type `int3` for
 * a vector of 3 `int`s. For native C code generator, see `CodeGenLLVM`.*/
class CodeGenC : public ExprFunctor<void(const PrimExpr&, std::ostream&)>,
                 public StmtFunctor<void(const Stmt&)>,
                 public CodeGenSourceBase {
public:
    /*!
   * \brief Initialize the code generator.
   * \param output_ssa Whether output SSA.
   */
  void Init(bool output_ssa);                   
/*!
   * \brief Add the function to the generated module.
   * \param f The function to be compiled.
   * \param whether to append return 0 in the end.
   */
  void AddFunction(const PrimFunc& f);
  /*!
   * \brief Finalize the compilation and return the code.
   * \return The code.
   */
  std::string Finish();
  /*!
   * \brief Print the Stmt n to CodeGenC->stream
   * \param n The statement to be printed.
   */
  void PrintStmt(const Stmt& n) { VisitStmt(n); }
  /*!
   * \brief Print the expression n(or its ssa id if in ssa mode) into os
   * \param n The expression to be printed.
   * \param os The output stream
   */
  void PrintExpr(const PrimExpr& n, std::ostream& os);
  /*!
   * \brief Same as PrintExpr, but simply returns result string
   * \param n The expression to be printed.
   */
  std::string PrintExpr(const PrimExpr& n) {
    std::ostringstream os;
    PrintExpr(n, os);
    return os.str();
  }  
// The following parts are overloadable print operations.
  /*!
   * \brief Print the function header before the argument list
   *
   *  Example: stream << "void";
   */
  virtual void PrintFuncPrefix();  // NOLINT(*)
  /*!
   * \brief Print extra function attributes
   *
   *  Example: __launch_bounds__(256) for CUDA functions
   */
  virtual void PrintExtraAttrs(const PrimFunc& f);
  /*!
   * \brief Print the final return at the end the function.
   */
  virtual void PrintFinalReturn();  // NOLINT(*)
  /*!
   * \brief Insert statement before function body.
   * \param f The function to be compiled.
   */
  virtual void PreFunctionBody(const PrimFunc& f) {}
  /*!
   * \brief Initialize codegen state for generating f.
   * \param f The function to be compiled.
   */
  virtual void InitFuncState(const PrimFunc& f);     
                     
                     

 /*!
   * \brief Print expr representing the thread tag
   * \param IterVar iv The thread index to be binded;
   */
  virtual void BindThreadIndex(const IterVar& iv);                             // NOLINT(*)
  virtual void PrintStorageScope(const std::string& scope, std::ostream& os);  // NOLINT(*)
  virtual void PrintStorageSync(const CallNode* op);                           // NOLINT(*)
  // Binary vector op.
  virtual void PrintVecBinaryOp(const std::string& op, DataType op_type, PrimExpr lhs, PrimExpr rhs,
                                std::ostream& os);  // NOLINT(*)
  // print vector load
  virtual std::string GetVecLoad(DataType t, const BufferNode* buffer, PrimExpr base);
  // print vector store
  virtual void PrintVecStore(const BufferNode* buffer, DataType t, PrimExpr base,
                             const std::string& value);  // NOLINT(*)
  // print load of single element
  virtual void PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                std::ostream& os);  // NOLINT(*)
  // print store of single element.
  virtual void PrintVecElemStore(const std::string& vec, DataType t, int i,
                                 const std::string& value);
  // Get a cast type from to
  virtual std::string CastFromTo(std::string value, DataType from, DataType target);
  // Get load of single element with expression
  virtual void PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os);
  // Print restrict keyword for a given Var if applicable
  virtual void PrintRestrict(const Var& v, std::ostream& os);

  virtual void SetConstantsByteAlignment(Integer constants_byte_alignment) {
    constants_byte_alignment_ = constants_byte_alignment;
  }                     
                 }
```

