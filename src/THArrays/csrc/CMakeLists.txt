cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(TorchCAPI CXX)

# 设置libtorch路径（假设在当前目录）
get_filename_component(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch" ABSOLUTE)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

# 自动探测CUDA工具链
function(find_latest_cuda_version)
    # 首先检查环境变量
    if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
        set(LATEST_CUDA_PATH $ENV{CUDA_HOME} PARENT_SCOPE)
        execute_process(
            COMMAND "$ENV{CUDA_HOME}/bin/nvcc" --version
            OUTPUT_VARIABLE NVCC_OUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        string(REGEX MATCH "V([0-9]+[.][0-9]+)" VERSION_MATCH "${NVCC_OUT}")
        if(VERSION_MATCH)
            set(LATEST_CUDA_VERSION "${CMAKE_MATCH_1}" PARENT_SCOPE)
            return()
        endif()
    endif()

    set(CUDA_PATHS
        "/usr/local/cuda"
        "/usr/local/cuda-*"
        "/usr/cuda"
        "/usr/cuda-*"
        "/opt/cuda"
        "/opt/cuda-*"
        "/usr/lib/nvidia-cuda-toolkit"
    )
    
    file(GLOB CUDA_INSTALLATIONS ${CUDA_PATHS})
    
    set(VALID_CUDA_PATHS "")
    set(CUDA_VERSIONS "")
    foreach(CUDA_PATH ${CUDA_INSTALLATIONS})
        if(IS_DIRECTORY "${CUDA_PATH}" AND EXISTS "${CUDA_PATH}/bin/nvcc")
            string(REGEX MATCH "cuda-([0-9]+[.][0-9]+)$" VERSION_MATCH "${CUDA_PATH}")
            if(VERSION_MATCH)
                list(APPEND VALID_CUDA_PATHS "${CUDA_PATH}")
                list(APPEND CUDA_VERSIONS "${CMAKE_MATCH_1}")
            else()
                execute_process(
                    COMMAND "${CUDA_PATH}/bin/nvcc" --version
                    OUTPUT_VARIABLE NVCC_OUT
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                )
                string(REGEX MATCH "V([0-9]+[.][0-9]+)" VERSION_MATCH "${NVCC_OUT}")
                if(VERSION_MATCH)
                    list(APPEND VALID_CUDA_PATHS "${CUDA_PATH}")
                    list(APPEND CUDA_VERSIONS "${CMAKE_MATCH_1}")
                endif()
            endif()
        endif()
    endforeach()
    
    if(CUDA_VERSIONS)
        list(LENGTH CUDA_VERSIONS NUM_VERSIONS)
        math(EXPR LAST_INDEX "${NUM_VERSIONS} - 1")
        foreach(IDX RANGE ${LAST_INDEX})
            list(GET CUDA_VERSIONS ${IDX} VERSION)
            list(GET VALID_CUDA_PATHS ${IDX} PATH)
            if(NOT DEFINED LATEST_VERSION OR VERSION VERSION_GREATER LATEST_VERSION)
                set(LATEST_VERSION ${VERSION})
                set(SELECTED_PATH ${PATH})
            endif()
        endforeach()
        
        set(LATEST_CUDA_PATH "${SELECTED_PATH}" PARENT_SCOPE)
        set(LATEST_CUDA_VERSION "${LATEST_VERSION}" PARENT_SCOPE)
    endif()
endfunction()

find_latest_cuda_version()

if(DEFINED LATEST_CUDA_PATH)
    message(STATUS "Found latest CUDA installation: ${LATEST_CUDA_PATH} (version ${LATEST_CUDA_VERSION})")
    set(CUDA_TOOLKIT_ROOT_DIR ${LATEST_CUDA_PATH})
    set(CMAKE_CUDA_COMPILER "${LATEST_CUDA_PATH}/bin/nvcc")
    
    set(ENV{CUDA_HOME} ${LATEST_CUDA_PATH})
    list(APPEND CMAKE_PREFIX_PATH ${LATEST_CUDA_PATH})
    
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    message(WARNING "No CUDA installation found")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 尝试查找Torch
find_package(Torch REQUIRED)
if(NOT TORCH_FOUND)
    message(FATAL_ERROR "PyTorch not found. Please check if libtorch is in the correct location: ${LIBTORCH_DIR}")
endif()

message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch CXX flags: ${TORCH_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TORCH_CAPI_SOURCES
    torch_capi_scalar.cpp
    torch_capi_tensor.cpp
    torch_capi_script.cpp
    torch_api.cpp
)

set(TORCH_CAPI_HEADERS
    torch_capi.h
    torch_capi_scalar.h
    torch_capi_tensor.h
    torch_capi_script.h
    torch_api.h
)

add_library(torch_capi SHARED 
    ${TORCH_CAPI_SOURCES}
    ${TORCH_CAPI_HEADERS}
)

target_link_libraries(torch_capi PRIVATE ${TORCH_LIBRARIES})
target_include_directories(torch_capi PRIVATE ${TORCH_INCLUDE_DIRS})

add_executable(example-app example_app.cpp)
target_link_libraries(example-app PRIVATE torch_capi ${TORCH_LIBRARIES})
target_include_directories(example-app PRIVATE ${TORCH_INCLUDE_DIRS})

install(TARGETS torch_capi example-app
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${TORCH_CAPI_HEADERS}
    DESTINATION include
)