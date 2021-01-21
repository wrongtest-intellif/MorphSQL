cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(MorphSQL LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

set(ANTLR_DIR /usr/local)
set(MLIR_DIR /Users/wrongtest/projects/llvm-project/build/lib/cmake/mlir)
set(LLVM_DIR /Users/wrongtest/projects/llvm-project/build/lib/cmake/llvm)

include(cmake/LLVM.cmake)
include(cmake/MLIR.cmake)

include_directories(include)
add_subdirectory(include)
add_subdirectory(src)

#add_subdirectory(lib)
#add_subdirectory(test)
#add_subdirectory(standalone-opt)
#add_subdirectory(standalone-translate)
