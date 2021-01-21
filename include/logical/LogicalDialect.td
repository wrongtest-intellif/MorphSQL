#ifndef LOGICAL_DIALECT
#define LOGICAL_DIALECT

include "mlir/IR/OpBase.td"


def LogicalDialect : Dialect {
    let name = "logical";
    let summary = "Logical plan MLIR dialect.";
    let description = [{
        
    }];
    let cppNamespace = "logical";
}

class LogicalOp<string mnemonic, list<OpTrait> traits = []> :
               Op<LogicalDialect, mnemonic, traits>;

#endif  // LOGICAL_DIALECT
