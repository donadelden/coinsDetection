cmake_minimum_required(VERSION 3.10.2)
project(coinsDetection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )

#include_directories(${OpenCV_INCLUDE_DIRS})
include_directories( include )

set(SOURCE_FILES src/main.cpp src/coinRecognize.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )