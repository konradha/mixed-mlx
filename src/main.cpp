#include "application.hpp"

#include <Foundation/Foundation.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char **argv) {
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  int status = EXIT_FAILURE;
  try {
    status = mixed_sqrt::runApplication(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
  }
  pool->release();
  return status;
}
