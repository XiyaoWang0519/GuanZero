add_executable(gd_fuzz ${CMAKE_CURRENT_SOURCE_DIR}/fuzz_main.cpp)
target_link_libraries(gd_fuzz PRIVATE gd_core)
