cmake_minimum_required(VERSION 3.16)
project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/core
  ${PROJECT_SOURCE_DIR}/include/math ${PROJECT_SOURCE_DIR}/include/solver
  ${PROJECT_SOURCE_DIR}/include/writer)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS_RELEASE
      "-O3 -march=native -flto -funroll-loops -fomit-frame-pointer -ffast-math -fno-rtti"
  )
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

endif()

file(
  GLOB_RECURSE
  SOURCES
  src/*.cpp
  src/core/*.cpp
  src/math/*.cpp
  src/solver/*.cpp
  src/writer/*.cpp
  src/logger/*.cpp)

add_executable(main ${SOURCES})
