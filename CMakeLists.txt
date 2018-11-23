OPTIX_add_sample_executable( smallptOptix
  smallpt.cpp
  ThreadUtils.h
)

target_link_libraries( smallptOptix
  optix_prime
  ${CUDA_LIBRARIES}
)