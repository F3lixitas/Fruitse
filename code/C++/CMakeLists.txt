cmake_minimum_required(VERSION 2.8)

project(Fruitse)

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} source/main.cpp source/LBP.hpp source/LBP.cpp)

if(UNIX)
    find_package(OpenCV REQUIRED)
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/source)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
else()
    include_directories(${PROJECT_NAME} PUBLIC C:/ton/chemin/ici)
    target_link_libraries(${PROJECT_NAME} C:/ton/chemin/ici)
endif()



