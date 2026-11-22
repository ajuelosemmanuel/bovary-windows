cmake_minimum_required(VERSION 3.22)
project(bovary_inkling_preview LANGUAGES C CXX)

if(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "This preview requires x64 Windows with MSVC.")
endif()
if(NOT EXISTS "${INKLING_ENGINE_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "Prepare the verified Inkling engine source first.")
endif()

foreach(option LLAMA_BUILD_COMMON LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS
               LLAMA_BUILD_EXAMPLES LLAMA_BUILD_MTMD LLAMA_BUILD_APP
               LLAMA_BUILD_SERVER LLAMA_BUILD_UI GGML_BUILD_TESTS GGML_BUILD_EXAMPLES
               GGML_CUDA GGML_NATIVE GGML_CPU_ALL_VARIANTS)
    set(${option} OFF CACHE BOOL "" FORCE)
endforeach()
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_subdirectory("${INKLING_ENGINE_SOURCE}" engine)

add_executable(bovary-infer infer.cpp)
target_compile_features(bovary-infer PRIVATE cxx_std_17)
target_link_libraries(bovary-infer PRIVATE llama)
