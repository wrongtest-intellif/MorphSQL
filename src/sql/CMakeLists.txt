find_package(ANTLR REQUIRED)
antlr_target(SqlBase SqlBase.g4)

include_directories(${ANTLR_DIR}/include/antlr4-runtime)

add_library(MorphSQLParser ${ANTLR_SqlBase_CXX_OUTPUTS})
target_link_options(MorphSQLParser PUBLIC -lantlr4-runtime -L${ANTLR_DIR}/lib)