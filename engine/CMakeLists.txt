cmake_minimum_required(VERSION 3.16)

project(ChessHacksEngine LANGUAGES CXX)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# On MSVC, use static runtime (optional but often nicer for single exe)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# List all engine source files explicitly
set(ENGINE_SOURCES
    src/board.cpp
    src/move.cpp
    src/movegen.cpp
    src/repetition.cpp
    src/time_manager.cpp
    src/nn_infer.cpp
    src/search.cpp
    src/uci.cpp
    src/main.cpp
)

add_executable(chess_engine ${ENGINE_SOURCES})

target_include_directories(chess_engine
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
