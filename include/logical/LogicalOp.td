#ifndef LOGICAL_OPS
#define LOGICAL_OPS

include "LogicalDialect.td"

def TableOp : LogicalOp<"table", []> {
    let summary = "Logical input table";
    let description = [{}];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let builders = [
    	OpBuilder<"OpBuilder &b, OperationState &state, Value lhs">
  	];
}

#endif // LOGICAL_OPS
