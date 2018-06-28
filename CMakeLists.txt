cmake_minimum_required(VERSION 2.8)

project(mobilefacenet_ncnn C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

include_directories(/include
                    ${CMAKE_CURRENT_LIST_DIR}/src)


list(APPEND MBF_LINKER_LIBS opencv_core opencv_highgui opencv_imgproc opencv_video ncnn)

file(GLOB MBF_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.h
                    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
set(MBF_COMPILE_CODE ${MBF_SRC})

add_executable(mobilefacenet_ncnn ${MBF_COMPILE_CODE})
# add_library(mobilefacenet_ncnn SHARED ${MBF_COMPILE_CODE})

target_link_libraries(mobilefacenet_ncnn ${MBF_LINKER_LIBS})
