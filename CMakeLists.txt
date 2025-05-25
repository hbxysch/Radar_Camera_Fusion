cmake_minimum_required(VERSION 3.16)
project(radar_camera_fusion)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

find_package(OpenCV REQUIRED)
#find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${YAML_CPP_INCLUDE_DIRS})

#add_definitions(${PCL_DEFINITIONS})

file(GLOB_RECURSE SRC_FILES
    src/fusion/*.hpp
    main.cpp
)

add_executable(radar_camera_fusion ${SRC_FILES})

target_link_libraries(radar_camera_fusion
    ${OpenCV_LIBS}
#    ${PCL_LIBRARIES}
    Eigen3::Eigen
    yaml-cpp
)
