OPTIX_add_sample_executable( smallptOptix
  smallpt.cpp
  ThreadUtils.h
)

target_include_directories(smallptOptix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries( smallptOptix
  optix_prime
  ${CUDA_LIBRARIES}
)