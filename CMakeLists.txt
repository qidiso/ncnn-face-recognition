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


list(APPEND LINKER_LIBS opencv_videoio opencv_video opencv_imgcodecs opencv_core opencv_highgui opencv_imgproc ncnn)

file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.h
              ${CMAKE_CURRENT_LIST_DIR}/src/featuredb.cpp
	      ${CMAKE_CURRENT_LIST_DIR}/src/mobilefacenet.cpp
	      ${CMAKE_CURRENT_LIST_DIR}/src/mtcnn.cpp)

file(GLOB BINSRC ${CMAKE_CURRENT_LIST_DIR}/src/tcpsender.cpp)
add_executable(tcpsender ${SRC} ${BINSRC})
set_target_properties(tcpsender PROPERTIES PREFIX "")
target_link_libraries(tcpsender ${LINKER_LIBS})

file(GLOB LIBSRC ${CMAKE_CURRENT_LIST_DIR}/src/facerecognition.cpp)
add_library(facerecognition SHARED ${SRC} ${LIBSRC})
set_target_properties(facerecognition PROPERTIES PREFIX "")
target_link_libraries(facerecognition ${LINKER_LIBS} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
