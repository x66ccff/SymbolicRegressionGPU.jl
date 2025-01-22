cmake_minimum_required(VERSION 3.5...3.27)
project(TorchCAPI CXX)

# 设置libtorch路径
get_filename_component(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch" ABSOLUTE)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

# 改进的CUDA检测函数
function(find_cuda_12_plus)
    # 检查常见的CUDA安装路径
    set(CUDA_SEARCH_PATHS
        ENV CUDA_HOME
        ENV CUDA_PATH
        "/usr/local/cuda"
        "/usr/lib/cuda"
        "/usr/lib/nvidia-cuda-toolkit"
        "/usr/lib/nvidia"
        "/usr/local"
        "/usr"
        "/opt/cuda"
    )

    set(CUDA_FOUND FALSE)
    set(SELECTED_CUDA_PATH "")
    set(SELECTED_CUDA_VERSION "")

    # 遍历所有可能的路径
    foreach(base_path ${CUDA_SEARCH_PATHS})
        file(GLOB cuda_paths
            "${base_path}/cuda-[0-9]*.[0-9]*"
            "${base_path}/cuda"
        )
        list(APPEND cuda_paths "${base_path}")

        foreach(cuda_path ${cuda_paths})
            # 检查cuda_runtime.h的存在
            if(EXISTS "${cuda_path}/include/cuda_runtime.h")
                # 获取CUDA版本
                if(EXISTS "${cuda_path}/bin/nvcc")
                    execute_process(
                        COMMAND "${cuda_path}/bin/nvcc" --version
                        OUTPUT_VARIABLE NVCC_OUT
                        ERROR_QUIET
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                    )
                    
                    string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" VERSION_MATCH "${NVCC_OUT}")
                    if(VERSION_MATCH)
                        set(CUDA_VERSION ${CMAKE_MATCH_1})
                        if(CUDA_VERSION VERSION_GREATER_EQUAL "12.0")
                            if(NOT CUDA_FOUND OR CUDA_VERSION VERSION_GREATER SELECTED_CUDA_VERSION)
                                set(CUDA_FOUND TRUE)
                                set(SELECTED_CUDA_PATH "${cuda_path}")
                                set(SELECTED_CUDA_VERSION "${CUDA_VERSION}")
                            endif()
                        endif()
                    endif()
                endif()
            endif()
        endforeach()
    endforeach()

    if(CUDA_FOUND)
        set(CUDA_PATH "${SELECTED_CUDA_PATH}" PARENT_SCOPE)
        set(CUDA_VERSION "${SELECTED_CUDA_VERSION}" PARENT_SCOPE)
        set(CUDA_FOUND TRUE PARENT_SCOPE)
        
        # 设置额外的CUDA相关变量
        set(CUDAToolkit_ROOT "${SELECTED_CUDA_PATH}" PARENT_SCOPE)
        set(CMAKE_CUDA_COMPILER "${SELECTED_CUDA_PATH}/bin/nvcc" PARENT_SCOPE)
        set(CUDAToolkit_INCLUDE_DIR "${SELECTED_CUDA_PATH}/include" PARENT_SCOPE)
        set(CUDA_TOOLKIT_ROOT_DIR "${SELECTED_CUDA_PATH}" PARENT_SCOPE)
    endif()
endfunction()

# 查找CUDA
find_cuda_12_plus()

if(CUDA_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION} at: ${CUDA_PATH}")
    message(STATUS "CUDA include dir: ${CUDAToolkit_INCLUDE_DIR}")
    
    # 设置CUDA相关变量
    set(ENV{CUDA_HOME} ${CUDA_PATH})
    list(APPEND CMAKE_PREFIX_PATH ${CUDA_PATH})
    
    # 启用CUDA支持
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # 添加CUDA包含路径
    include_directories(
        ${CUDA_PATH}/include
        ${CUDAToolkit_INCLUDE_DIR}
        /usr/include
    )

    # 设置CUDA库路径
    link_directories(
        ${CUDA_PATH}/lib64
        ${CUDA_PATH}/lib
    )
    
    # 设置CUDA编译标志
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    
    # 检查常见的CUDA依赖库
    set(CUDA_LIBS
        cuda
        cudart
        cublas
        curand
        cusolver
        cudnn
        nvrtc
    )
    
    foreach(lib ${CUDA_LIBS})
        find_library(${lib}_LIBRARY
            NAMES ${lib}
            PATHS 
                ${CUDA_PATH}/lib64 
                ${CUDA_PATH}/lib
                /usr/lib/x86_64-linux-gnu
                /usr/local/cuda/lib64
            NO_DEFAULT_PATH
        )
        if(${lib}_LIBRARY)
            list(APPEND CUDA_LIBRARIES ${${lib}_LIBRARY})
        endif()
    endforeach()
else()
    message(FATAL_ERROR "No CUDA 12.0+ installation found!")
endif()

# 设置基本编译选项
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找Torch包
find_package(Torch REQUIRED)
if(NOT TORCH_FOUND)
    message(FATAL_ERROR "PyTorch not found. Please check if libtorch is in the correct location: ${LIBTORCH_DIR}")
endif()

message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch CXX flags: ${TORCH_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 设置源文件和头文件
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

# 添加库目标
add_library(torch_capi SHARED 
    ${TORCH_CAPI_SOURCES}
    ${TORCH_CAPI_HEADERS}
)

# 设置库的链接和包含
target_link_libraries(torch_capi PRIVATE 
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
)

target_include_directories(torch_capi PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIR}
)

# 添加可执行文件目标
add_executable(example-app example_app.cpp)
target_link_libraries(example-app PRIVATE 
    torch_capi 
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
)

target_include_directories(example-app PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIR}
)

# 安装规则
install(TARGETS torch_capi example-app
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${TORCH_CAPI_HEADERS}
    DESTINATION include
)