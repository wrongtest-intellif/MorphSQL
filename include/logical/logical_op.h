#ifndef LOGICAL_LOGICAL_OP_H
#define LOGICAL_LOGICAL_OP_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Function.h"

namespace morph {
namespace logical {

using namespace mlir;  // NOLINT

#define GET_OP_CLASSES
#include "logical/LogicalOp.h.inc"

namespace LogicalResultTypes {
enum Kind {
	Table = 1000000
};
}  // namespace LogicalResultTypes


class LogicalDialect : public ::mlir::Dialect {
  public:
    explicit LogicalDialect(::mlir::MLIRContext *ctx);
    static ::llvm::StringRef getDialectNamespace() { return "logical"; }
  
    void printType(mlir::Type type,
                   mlir::DialectAsmPrinter &printer) const override;
};

class LogicalPlan {
 public:
 	mlir::FuncOp GetMLIRFunc() { return logical_func_; }

 private:
 	mlir::FuncOp logical_func_;
};

class LogicalResultType : public Type {
 public:
 	using ImplType = DefaultTypeStorage;
  	using Type::Type;
};

class NNType : public Type::TypeBase<NNType, Type, TypeStorage> {
public:
  using Base::Base;

  /// Get an instance of the NoneType.
  static NNType get(MLIRContext *context) {
  	return Base::get(context, LogicalResultTypes::Table);
  }

  static bool kindof(unsigned kind) { return kind == LogicalResultTypes::Table; }
};

class LogicalTableType : public Type::TypeBase<LogicalTableType, Type, DefaultTypeStorage> {
 public:
 	using Base::Base;

 	static LogicalTableType get(MLIRContext* mlir_ctx) {
 		return Base::get(mlir_ctx, LogicalResultTypes::Table);
 	}

 	static bool kindof(unsigned kind) {
    	return kind == LogicalResultTypes::Table;
  	}
};

} // namespace logical
} // namespace morph

#endif  // LOGICAL_LOGICAL_OP_H