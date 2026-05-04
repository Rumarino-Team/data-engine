cmake_minimum_required(VERSION 3.16)
project(DataEngine VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find required packages
find_package(CURL REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Add source and header directories
include_directories(include)
include_directories(${CURL_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(src)