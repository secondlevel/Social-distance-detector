cmake_minimum_required(VERSION 3.8.12)
project(camera_calibrate_utils)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) 

add_subdirectory(pybind11)

SET(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/camera_calibrate_utils.cpp
) 

pybind11_add_module(camera_calibrate_utils ${SOURCES})

target_link_libraries(camera_calibrate_utils PRIVATE ${OpenCV_LIBS})