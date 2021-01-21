add_mlir_dialect_library(MorphExpr
    expr.cc
    ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR}/include/expr
    DEPENDS MLIRExprIncGen
    LINK_LIBS PUBLIC MLIRIR)