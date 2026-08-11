cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_STANDARD 11)

project(SimulacionFluidos LANGUAGES C CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(program
    main.cu
    screen_config.cu
    solve_edos.cu
    glad.c
    
)

target_include_directories(program PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/glad
)

set_target_properties(program PROPERTIES
    CUDA_SEPARATE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_LINK_PHASE ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(program PRIVATE
    CUDA::cudart
    CUDA::cudart_static
    OpenGL::GL
    glfw
    m
)

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env
        __NV_PRIME_RENDER_OFFLOAD=1
        __GLX_VENDOR_LIBRARY_NAME=nvidia
        $<TARGET_FILE:program>
    DEPENDS program
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Ejecutando en la GPU dedicada"
)