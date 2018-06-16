cmake_minimum_required(VERSION 2.8.3)
project(vi_ekf)


set(CMAKE_CXX_FLAGS "-std=c++17 -mavx -mfma")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)


include_directories(
  inc
  ${EIGEN3_INCLUDE_DIRS}
  )
  
add_executable(vi_ekf_cvtest
    src/cvtest.cpp
    inc/quat.h
    inc/math_helper.h
)
target_link_libraries(vi_ekf_cvtest ${OpenCV_LIBS})
