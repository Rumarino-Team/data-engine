#include <chrono>
#include <iostream>
#include <thread>

#include "backend_interface.hpp"

int main() {
  std::cout << "Data Engine go brr" << std::endl;
  BackendInterface backend;
  backend.launchBackend();
  auto status = backend.getStatus();
  std::cout << "Backend status: " << status.status << std::endl;
  backend.stopBackend();

  return 0;
}
