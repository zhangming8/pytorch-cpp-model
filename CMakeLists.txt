cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(predict-app predict-app.cpp)
add_executable(predict-app1 predict-app1.cpp)
target_link_libraries(predict-app "${TORCH_LIBRARIES}" ${OpenCV_LIBS} stdc++fs)
target_link_libraries(predict-app1 "${TORCH_LIBRARIES}" ${OpenCV_LIBS} stdc++fs)
set_property(TARGET predict-app PROPERTY CXX_STANDARD 11)
set_property(TARGET predict-app1 PROPERTY CXX_STANDARD 11)
