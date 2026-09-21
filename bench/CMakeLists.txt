add_executable(gd_bench ${CMAKE_CURRENT_SOURCE_DIR}/bench_engine.cpp)
target_link_libraries(gd_bench PRIVATE gd_core)
