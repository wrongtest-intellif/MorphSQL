#ifndef EXPR_DIALECT
#define EXPR_DIALECT

include "mlir/IR/OpBase.td"


def ExprDialect : Dialect {
    let name = "expr";
    let summary = "Expression MLIR dialect.";
    let description = [{
        
    }];
    let cppNamespace = "expr";
}

class Expr<string mnemonic, list<OpTrait> traits = []> :
        Op<ExprDialect, mnemonic, traits>;

#endif  // EXPR_DIALECT
