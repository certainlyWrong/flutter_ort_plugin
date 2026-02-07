cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()

set(PROJECT_NAME "flutter_ort_plugin")
project(${PROJECT_NAME} LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  onnxruntime_linux
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.24.1/onnxruntime-linux-x64-1.24.1.tgz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime_linux)

set(ORT_ROOT "${onnxruntime_linux_SOURCE_DIR}")
set(ORT_LIB_DIR "${ORT_ROOT}/lib")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")

# O arquivo real baixado
set(ORT_SO_REAL "${ORT_LIB_DIR}/libonnxruntime.so.1.24.1")
# O nome que o Dart quer
set(ORT_SO_LINK "${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.so")

# Cria um link simbólico ou cópia para o Dart encontrar
add_custom_command(
    OUTPUT "${ORT_SO_LINK}"
    COMMAND ${CMAKE_COMMAND} -E copy "${ORT_SO_REAL}" "${ORT_SO_LINK}"
    DEPENDS "${ORT_SO_REAL}"
)
add_custom_target(onnx_lib_target ALL DEPENDS "${ORT_SO_LINK}")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src" "${CMAKE_CURRENT_BINARY_DIR}/shared")

# Define as bibliotecas que o Flutter deve empacotar
# IMPORTANTE: Use o ORT_SO_LINK (o nome limpo .so)
set(flutter_ort_plugin_bundled_libraries
  "$<TARGET_FILE:flutter_ort_plugin>"
  "${ORT_SO_LINK}"
  PARENT_SCOPE
)

target_link_libraries(flutter_ort_plugin PRIVATE "${ORT_SO_REAL}")
target_include_directories(flutter_ort_plugin PRIVATE "${ORT_INCLUDE_DIR}")

# RPATH para o executável achar a lib na pasta bundle/lib
set_target_properties(flutter_ort_plugin PROPERTIES 
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)