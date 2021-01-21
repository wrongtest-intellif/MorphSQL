#ifndef EXPR_OPS
#define EXPR_OPS

include "ExprDialect.td"

def AddOp : Expr<"add", []> {
    let summary = "Arithmetic add";
    let description = [{}];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // EXPR_OPS
