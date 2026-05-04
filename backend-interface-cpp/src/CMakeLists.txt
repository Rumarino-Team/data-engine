# Add library for backend interface
add_library(backend_interface backend_interface.cpp)
target_link_libraries(backend_interface PRIVATE CURL::libcurl nlohmann_json::nlohmann_json)

# Main demo executable
add_executable(DataEngine main.cpp)
target_link_libraries(DataEngine PRIVATE backend_interface CURL::libcurl nlohmann_json::nlohmann_json)

# Test executable that mirrors backend/tests/tester.py using the C++ interface
add_executable(BackendInterfaceTests tests.cpp)
target_link_libraries(BackendInterfaceTests PRIVATE backend_interface CURL::libcurl nlohmann_json::nlohmann_json)