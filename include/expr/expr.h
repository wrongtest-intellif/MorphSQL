#ifndef EXPR_EXPR_H
#define EXPR_EXPR_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace morph {
namespace expr {

using namespace mlir;  // NOLINT

#define GET_OP_CLASSES
#include "expr/ExprDialect.h.inc"
#include "expr/Expr.h.inc"


} // namespace expr
} // namespace morph

#endif  // EXPR_EXPR_H