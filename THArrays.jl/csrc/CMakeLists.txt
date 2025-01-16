cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(TorchCAPI CXX)

# 自动探测CUDA工具链
function(find_latest_cuda_version)
    set(CUDA_PATHS
        "/usr/local/cuda"
        "/usr/local/cuda-*"
    )
    
    # 查找所有CUDA安装
    file(GLOB CUDA_INSTALLATIONS ${CUDA_PATHS})
    
    # 提取版本号并排序
    set(CUDA_VERSIONS "")
    foreach(CUDA_PATH ${CUDA_INSTALLATIONS})
        if(IS_DIRECTORY "${CUDA_PATH}")
            string(REGEX MATCH "cuda-([0-9]+[.][0-9]+)$" VERSION_MATCH "${CUDA_PATH}")
            if(VERSION_MATCH)
                list(APPEND CUDA_VERSIONS "${CMAKE_MATCH_1}")
            endif()
        endif()
    endforeach()
    
    # 排序版本号，获取最新版本
    if(CUDA_VERSIONS)
        list(SORT CUDA_VERSIONS COMPARE NATURAL ORDER DESCENDING)
        list(GET CUDA_VERSIONS 0 LATEST_VERSION)
        set(LATEST_CUDA_PATH "/usr/local/cuda-${LATEST_VERSION}" PARENT_SCOPE)
        set(LATEST_CUDA_VERSION "${LATEST_VERSION}" PARENT_SCOPE)
    endif()
endfunction()

# 调用函数找到最新CUDA
find_latest_cuda_version()

if(DEFINED LATEST_CUDA_PATH)
    message(STATUS "Found latest CUDA installation: ${LATEST_CUDA_PATH} (version ${LATEST_CUDA_VERSION})")
    set(CUDA_TOOLKIT_ROOT_DIR ${LATEST_CUDA_PATH})
    set(CMAKE_CUDA_COMPILER "${LATEST_CUDA_PATH}/bin/nvcc")
    
    # 设置CUDA相关环境
    set(ENV{CUDA_HOME} ${LATEST_CUDA_PATH})
    list(APPEND CMAKE_PREFIX_PATH ${LATEST_CUDA_PATH})
    
    # 启用CUDA语言支持
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    message(WARNING "No CUDA installation found in /usr/local/cuda*")
endif()

# 设置编译选项
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# PyTorch查找
find_package(Torch REQUIRED)
if(NOT TORCH_FOUND)
    message(FATAL_ERROR "PyTorch not found. Please set Torch_DIR.")
endif()

message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch CXX flags: ${TORCH_CXX_FLAGS}")

# 添加编译标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 源文件列表
set(TORCH_CAPI_SOURCES
    torch_capi_scalar.cpp
    torch_capi_tensor.cpp
    torch_capi_script.cpp
    torch_api.cpp
)

# 头文件列表
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

# 设置库的属性
target_link_libraries(torch_capi PRIVATE ${TORCH_LIBRARIES})
target_include_directories(torch_capi PRIVATE ${TORCH_INCLUDE_DIRS})

# 添加可执行文件目标
add_executable(example-app example_app.cpp)
target_link_libraries(example-app PRIVATE torch_capi ${TORCH_LIBRARIES})
target_include_directories(example-app PRIVATE ${TORCH_INCLUDE_DIRS})

# 安装规则
install(TARGETS torch_capi example-app
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${TORCH_CAPI_HEADERS}
    DESTINATION include
)