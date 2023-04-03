cmake_minimum_required(VERSION 3.5)

project(detect_YOLOv8_IR VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(OpenVINO PATHS /opt/intel/openvino_2022.3.0/runtime/)
# !OpenCV

set(PROJECT_SOURCES
    detect_YOLOv8_IR.cpp
)

add_executable(detect_YOLOv8_IR ${PROJECT_SOURCES})
target_link_libraries(detect_YOLOv8_IR PRIVATE openvino::runtime ${OpenCV_LIBS})