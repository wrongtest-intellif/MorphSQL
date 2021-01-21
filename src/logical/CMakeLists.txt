add_mlir_dialect_library(MorphLogical
    logical_op.cc
    ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR}/include/logical
    DEPENDS MLIRLogicalOpIncGen
    LINK_LIBS PUBLIC MLIRIR)