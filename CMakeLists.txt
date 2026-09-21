cmake_minimum_required(VERSION 3.24)

# On this machine the CommandLineTools SDK ships .tbd stubs declaring an
# arm64e.x1 slice that the installed linker cannot parse, so /usr/bin/c++ fails
# its own compiler check. Ask xcrun for the active SDK instead. Must happen
# before project(). No effect anywhere but macOS.
if(APPLE AND NOT CMAKE_OSX_SYSROOT)
  execute_process(COMMAND xcrun --sdk macosx --show-sdk-path
                  OUTPUT_VARIABLE GD_MACOS_SDK
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  if(GD_MACOS_SDK AND IS_DIRECTORY "${GD_MACOS_SDK}")
    set(CMAKE_OSX_SYSROOT "${GD_MACOS_SDK}" CACHE PATH "macOS SDK" FORCE)
  endif()
endif()

project(guandan_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(GD_SANITIZE "Build with ASan and UBSan" OFF)
if(GD_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

option(GD_BUILD_PYTHON "Build the pybind11 module" ON)

add_subdirectory(cpp)

option(GD_BUILD_BENCH "Build the throughput benchmarks" OFF)
if(GD_BUILD_BENCH)
  add_subdirectory(bench)
endif()

option(GD_BUILD_FUZZ "Build the property fuzzer" OFF)
if(GD_BUILD_FUZZ)
  add_subdirectory(cpp/fuzz)
endif()

if(GD_BUILD_PYTHON AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/python/bindings.cpp)
  find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED PATHS ${PYBIND11_CMAKE_DIR} NO_DEFAULT_PATH)
  add_subdirectory(python)
endif()

enable_testing()
