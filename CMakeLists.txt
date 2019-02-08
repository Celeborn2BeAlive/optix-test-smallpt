add_subdirectory(glfw)

file(GLOB SRC_FILES *.h *.cpp)

OPTIX_add_sample_executable( smallptOptix
  ${SRC_FILES}
)

target_include_directories(smallptOptix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} glfw/include)

target_link_libraries( smallptOptix
  optix_prime
  ${CUDA_LIBRARIES}
  glfw
)