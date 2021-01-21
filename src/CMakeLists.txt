llvm_map_components_to_libnames(LLVM_LIBS support core support)
set(MLIR_LIBS MLIRIR)

add_subdirectory(sql)
add_subdirectory(logical)
add_subdirectory(expr)