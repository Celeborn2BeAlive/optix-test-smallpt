find_package(OpenGL REQUIRED)

add_subdirectory(glfw)

file(GLOB SRC_FILES *.h *.cpp glad/src/glad.c)

OPTIX_add_sample_executable( smallptOptix
  ${SRC_FILES}
)

target_include_directories(smallptOptix 
  PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OPENGL_INCLUDE_DIRS}
  glfw/include
  glad/include
)

target_link_libraries( smallptOptix
  optix_prime
  ${CUDA_LIBRARIES}
  glfw
  ${OPENGL_LIBRARIES}
)

target_compile_definitions(smallptOptix 
  PRIVATE 
  NOMINMAX
)