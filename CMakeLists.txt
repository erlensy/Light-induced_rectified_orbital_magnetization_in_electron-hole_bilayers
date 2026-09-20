cmake_minimum_required(VERSION 3.10)

project(ife)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)

file(GLOB SOURCES src/*.cpp)

add_executable(ife ${SOURCES})

target_include_directories(ife PRIVATE src ${ARMADILLO_INCLUDE_DIRS})

target_link_libraries(ife PRIVATE ${ARMADILLO_LIBRARIES})
