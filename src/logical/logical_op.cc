#include "logical/logical_op.h"

namespace morph {
namespace logical {

#define GET_OP_CLASSES
#include "logical/LogicalOp.cpp.inc"

LogicalDialect::LogicalDialect(::mlir::MLIRContext *ctx): mlir::Dialect("logical", ctx) {
	this->addOperations<
		#define GET_OP_LIST
		#include "logical/LogicalOp.cpp.inc"
    >();
	this->addTypes<NNType>();
	printf("???\n");
}

void LogicalDialect::printType(mlir::Type type,
                               mlir::DialectAsmPrinter &printer) const {
	
}

void TableOp::build(OpBuilder &b, OperationState &state) {

}

} // namespace logical
} // namespace morph
