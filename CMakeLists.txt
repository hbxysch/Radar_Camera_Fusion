cmake_minimum_required(VERSION 3.16)
project(radar_camera_fusion)

set(CMAKE_CXX_STANDARD 17)

# include path
include_directories(include)

# ===============================
# OpenMP configuration（macOS + Homebrew）
# ===============================
set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/include")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/include")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/opt/homebrew/lib/libomp.dylib")
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "✅ OpenMP found, enabling support")
    add_compile_options(${OpenMP_CXX_FLAGS})
    link_directories(/opt/homebrew/lib)
    link_libraries(omp)
else()
    message(WARNING "❌ OpenMP not found, features will be disabled")
endif()

# ===============================
# third party libraries
# ===============================
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(JsonCpp REQUIRED)

# ===============================
# Include directories
# ===============================
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${YAML_CPP_INCLUDE_DIRS}
)

# ===============================
# source files
# ===============================
file(GLOB_RECURSE SRC_FILES
        src/fusion/*.hpp
        src/fusion/*.cpp
        main.cpp
)

# ===============================
# executable and linking
# ===============================
add_executable(radar_camera_fusion ${SRC_FILES})

target_link_libraries(radar_camera_fusion
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        Eigen3::Eigen
        yaml-cpp
        JsonCpp::JsonCpp
)