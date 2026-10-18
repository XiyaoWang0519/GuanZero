if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
  # pybind11 otherwise adds LTO to this wrapper on every engine relink.
  # The hot loops live in gd_core; keep edits cheap, with an explicit opt-in.
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()
pybind11_add_module(_gd_core ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp)
target_link_libraries(_gd_core PRIVATE gd_core)
# GD_PYTHON_OUTPUT_DIR builds the module elsewhere (a side package directory),
# so a rebuild never replaces the extension under running processes.
set(GD_PYTHON_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gd" CACHE PATH "Where _gd_core is written")
set_target_properties(_gd_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${GD_PYTHON_OUTPUT_DIR})
