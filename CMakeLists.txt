cmake_minimum_required(VERSION 3.14)
project(tdle)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

Set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_PROGRESS TRUE GIT_SHALLOW TRUE GIT_TAG v1.11.0
    USES_TERMINAL_DOWNLOAD TRUE)
FetchContent_MakeAvailable(spdlog)

add_library(tdle STATIC graph.cpp tensor.cpp optimizer.cpp)
target_link_libraries(tdle spdlog)
add_executable(tdle_main main.cpp)
target_link_libraries(tdle_main tdle)
