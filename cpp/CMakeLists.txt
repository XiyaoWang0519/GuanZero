file(GLOB GD_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(gd_core STATIC ${GD_CORE_SOURCES})
target_include_directories(gd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(gd_core PRIVATE -Wall -Wextra -Wpedantic)
find_package(Threads REQUIRED)
target_link_libraries(gd_core PUBLIC Threads::Threads)

add_subdirectory(tests)
