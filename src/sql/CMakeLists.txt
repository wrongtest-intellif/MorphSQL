find_package(ANTLR REQUIRED)
antlr_target(SqlBase SqlBase.g4 LEXER PARSER VISITOR)

include_directories(${ANTLR_DIR}/include/antlr4-runtime)

add_library(MorphSQLParser sql_parser.cc ${ANTLR_SqlBase_CXX_OUTPUTS})
target_link_options(MorphSQLParser PUBLIC -lantlr4-runtime -L${ANTLR_DIR}/lib)
target_include_directories(MorphSQLParser PUBLIC ${ANTLR_SqlBase_OUTPUT_DIR})

add_executable(morph_sql_parser sql_parser_main.cc)
target_link_libraries(morph_sql_parser MorphSQLParser)