cmake_minimum_required(VERSION 3.10)
project(CVServer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add pthread for Windows (MinGW)
if(MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/operations)

# Source files
set(SOURCES
    main.cpp
    operations/utils.cpp
    operations/harris.cpp
    operations/sift.cpp
    operations/matcher.cpp
)

# Executable
add_executable(cv_server ${SOURCES})

# Link libraries
if(UNIX)
    target_link_libraries(cv_server pthread)
elseif(WIN32)
    target_link_libraries(cv_server ws2_32)
endif()