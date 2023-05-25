cmake_minimum_required(VERSION 3.5)

project(detect_YOLOv8_IR VERSION 0.1)

set (CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(OpenVino_INCLUDE_DIRS
    "/opt/intel/openvino_2021.4.582/inference_engine/include" 
    "/opt/intel/openvino_2021.4.582/inference_engine/samples/cpp/common/utils/include"
    "/opt/intel/openvino_2021.4.582/deployment_tools/ngraph/include"
  )

# eigen
find_package(Torch REQUIRED PATHS /home/ss21mipt/libtorch/)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)

find_package(OpenVINO PATHS /opt/intel/openvino_2021/inference_engine)

include_directories(${OpenVino_INCLUDE_DIRS})
# !OpenCV

set(PROJECT_SOURCES
    detect_YOLOv8_IR.cpp
)

add_executable(detect_YOLOv8_IR ${PROJECT_SOURCES})
target_link_libraries(detect_YOLOv8_IR ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES} "${TORCH_LIBRARIES}" ${OpenCV_LIBS})