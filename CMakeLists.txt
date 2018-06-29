cmake_minimum_required(VERSION 2.8)

project(mobilefacenet_ncnn C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

FIND_PACKAGE(Boost COMPONENTS python)
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})
include_directories(/include
	            ${CMAKE_CURRENT_LIST_DIR}/src)


list(APPEND LINKER_LIBS opencv_imgcodecs opencv_core opencv_highgui opencv_imgproc opencv_video ncnn)

file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.h
              ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

#add_executable(mobilefacenet_ncnn ${MBF_COMPILE_CODE})
add_library(facerecognition SHARED ${SRC})
set_target_properties(facerecognition PROPERTIES PREFIX "")
target_link_libraries(facerecognition ${LINKER_LIBS} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
