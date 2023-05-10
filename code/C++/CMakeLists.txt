cmake_minimum_required(VERSION 2.8)

project(Fruitse)

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} source/main.cpp)

if(UNIX)
    find_package(OpenCV REQUIRED)
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/source)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
else()
    include_directories(${PROJECT_NAME} PUBLIC $ENV{OPENCV_DIR}\\..\\..\\include)
    target_link_directories(${PROJECT_NAME} PUBLIC $ENV{OPENCV_DIR}\\lib)
    target_link_libraries(${PROJECT_NAME} opencv_core470 opencv_highgui470 opencv_imgproc470 opencv_imgcodecs470 opencv_videoio470)
endif()



