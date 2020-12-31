cmake_minimum_required(VERSION 3.18)
project(ba)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophos_INCLUDE_DIRS}
        "/usr/include/eigen3"
)
add_executable(ba main_ba.cpp)
target_link_libraries(ba
        ${OpenCV_LIBS}
        )

add_executable(pose_estimation_2d_2d pose_estimation_2d_2d.cpp)
target_link_libraries(pose_estimation_2d_2d
        ${OpenCV_LIBS}
)

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d
        ${OpenCV_LIBS}
        ${G2O_LIBS}
        g2o_core g2o_stuff
)