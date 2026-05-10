cmake_minimum_required(VERSION 3.20)
option(CUDA_ON "Enable CUDA support" OFF)
option(PRUNE_ON "Enable PRUNE support" ON)
option(UNIT_TEST_BUILD "Enable unit_tests build" ON)
option(TEST_BUILD "Enable tests build" ON)
option(APP_BUILD "Enable app build" ON)
option(SPANN_AUG_BUILD "Enable SpannAugmentor (index builder) build" ON)

project(MiniHyperVec VERSION 1.0 LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CUDA_ON)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA enabled")
else()
    message(STATUS "CUDA disabled")
endif()

add_compile_options(
  -fpermissive
  -pthread
  -march=native
)
add_link_options(
  -static-libstdc++ 
  -static-libgcc
  -pthread
  -Wl,--warn-unresolved-symbols
)
add_compile_options(
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-O3>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-g>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g>
)
if (CUDA_ON)
  add_compile_options(
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-g>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g>
  )
endif()
if(CUDA_ON)
  add_compile_definitions(CUDA_ON) 
endif()


find_package(PkgConfig REQUIRED)
set(SPDK_INCLUDE_DIR)
set(SPDK_LIB)
pkg_check_modules(SPDK_NVME REQUIRED spdk_nvme)
pkg_check_modules(SPDK_ENV_DPDK REQUIRED spdk_env_dpdk)
pkg_check_modules(SPDK_SYSLIBS REQUIRED spdk_syslibs)
pkg_check_modules(SPDK_VMD REQUIRED spdk_vmd)
list(APPEND SPDK_INCLUDE_DIR
${SPDK_NVME_INCLUDE_DIRS} 
${SPDK_ENV_DPDK_INCLUDE_DIRS} 
${SPDK_SYSLIBS_INCLUDE_DIRS} 
${SPDK_VMD_INCLUDE_DIRS}
)
list(APPEND SPDK_LIB
${SPDK_NVME_LINK_LIBRARIES} 
${SPDK_ENV_DPDK_LINK_LIBRARIES}   
${SPDK_VMD_LINK_LIBRARIES} 
${SPDK_SYSLIBS_LINK_LIBRARIES}
uuid
)
set(SPDK_CFLAGS
${SPDK_NVME_CFLAGS_OTHER} 
${SPDK_ENV_DPDK_CFLAGS_OTHER} 
${SPDK_VMD_CFLAGS_OTHER} 
${SPDK_SYSLIBS_CFLAGS_OTHER}
)

set(TBB_INCLUDE_DIR)
set(TBB_LIB)
set(TBB_LIB_DIR)
set(TBB_ROOT_DIR /mnt/service/3rd_party/TBB_Path)
list(APPEND TBB_INCLUDE_DIR 
${TBB_ROOT_DIR}/include/tbb/
${TBB_ROOT_DIR}/include/)
list(APPEND TBB_LIB_DIR ${TBB_ROOT_DIR}/lib64)
list(APPEND TBB_LIB libtbb.so)

set(LIGHTGBM_INCLUDE_DIR)
set(LIGHTGBM_LIB_DIR)
set(LIGHTGBM_LIB)
set(ONNXRT_INCLUDE_DIR)
set(ONNXRT_LIB_DIR)      
set(ONNXRT_LIB)     
if(PRUNE_ON)
  set(LIGHTGBM_ROOT_DIR /mnt/service/3rd_party/LightGBM_Path) 
  list(APPEND LIGHTGBM_INCLUDE_DIR ${LIGHTGBM_ROOT_DIR}/include)
  list(APPEND LIGHTGBM_LIB_DIR ${LIGHTGBM_ROOT_DIR}/lib)
  list(APPEND LIGHTGBM_LIB lib_lightgbm.so)
  set(ONNXRT_ROOT_DIR /mnt/service/3rd_party/ONNXRT_Path) 
  list(APPEND ONNXRT_INCLUDE_DIR ${ONNXRT_ROOT_DIR}/include)
  list(APPEND ONNXRT_LIB_DIR ${ONNXRT_ROOT_DIR}/lib)
  list(APPEND ONNXRT_LIB onnxruntime)
endif()

set(LIB_ALL)
list(APPEND LIB_ALL
 stdc++fs
 ${TBB_LIB}
 ${SPDK_LIB}
 $<IF:$<BOOL:${PRUNE_ON}>,${LIGHTGBM_LIB},>
 $<IF:$<BOOL:${PRUNE_ON}>,${ONNXRT_LIB},>
)
if (CUDA_ON)
endif()

set(INCLUDE_DIR)
list(APPEND INCLUDE_DIR
 ${PROJECT_SOURCE_DIR}/include
 ${TBB_INCLUDE_DIR}
 ${SPDK_INCLUDE_DIR}
 $<IF:$<BOOL:${PRUNE_ON}>,${LIGHTGBM_INCLUDE_DIR},>
 $<IF:$<BOOL:${PRUNE_ON}>,${ONNXRT_INCLUDE_DIR},>
)
if  (CUDA_ON)
endif()
include_directories(${INCLUDE_DIR})

set(LIB_DIR)
list(APPEND LIB_DIR
 ${TBB_LIB_DIR}
 $<IF:$<BOOL:${PRUNE_ON}>,${LIGHTGBM_LIB_DIR},>
 $<IF:$<BOOL:${PRUNE_ON}>,${ONNXRT_LIB_DIR},>
)
if (CUDA_ON)
endif()
link_directories(${LIB_DIR})

if (SPANN_AUG_BUILD)
  add_subdirectory(third_party/spann_aug)
endif()

file(GLOB_RECURSE sources_cpp ${PROJECT_SOURCE_DIR}/source/*.cpp)
file(GLOB_RECURSE sources_app_cpp ${PROJECT_SOURCE_DIR}/app/*.cpp)
file(GLOB_RECURSE sources_test_cpp ${PROJECT_SOURCE_DIR}/test/*.cpp)
file(GLOB_RECURSE sources_unit_test_cpp ${PROJECT_SOURCE_DIR}/unit_test/*.cpp)
if (CUDA_ON)
  file(GLOB_RECURSE sources_cuda ${PROJECT_SOURCE_DIR}/source/*.cu)
  file(GLOB_RECURSE sources_app_cuda ${PROJECT_SOURCE_DIR}/app/*.cu)
  file(GLOB_RECURSE sources_test_cuda ${PROJECT_SOURCE_DIR}/test/*.cu)
  file(GLOB_RECURSE sources_unit_test_cuda ${PROJECT_SOURCE_DIR}/unit_test/*.cu)
endif()

add_library(MiniHyperVec SHARED ${sources_cpp} ${sources_cuda})
target_link_libraries(MiniHyperVec 
  PUBLIC ${LIB_ALL} 
)
target_compile_options(MiniHyperVec PUBLIC ${SPDK_CFLAGS})
if (CUDA_ON)
  target_compile_options(MiniHyperVec PUBLIC 
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  target_compile_options(MiniHyperVec PUBLIC 
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

if (UNIT_TEST_BUILD)
  set(UNIT_TEST_EXES)
  foreach(unit_test_exe IN LISTS sources_unit_test_cpp)
      get_filename_component(unit_test_exe_name ${unit_test_exe} NAME_WE)
      add_executable(${unit_test_exe_name} ${unit_test_exe})
      list(APPEND UNIT_TEST_EXES ${unit_test_exe_name})
      target_link_libraries(${unit_test_exe_name} 
        PRIVATE MiniHyperVec
      )
  endforeach()
  if (CUDA_ON)
    foreach(unit_test_exe IN LISTS sources_unit_test_cuda)
        get_filename_component(unit_test_exe_name ${unit_test_exe} NAME_WE)
        add_executable(${unit_test_exe_name} ${unit_test_exe})
        list(APPEND UNIT_TEST_EXES ${unit_test_exe_name})
        target_link_libraries(${unit_test_exe_name} 
          PRIVATE MiniHyperVec
        )
    endforeach()
  endif()
  set_target_properties(
  ${UNIT_TEST_EXES}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test
  )
endif()

if (TEST_BUILD)
  set(TEST_EXES)
  foreach(test_exe IN LISTS sources_test_cpp)
      get_filename_component(test_exe_name ${test_exe} NAME_WE)
      add_executable(${test_exe_name} ${test_exe})
      list(APPEND TEST_EXES ${test_exe_name}) 
      target_link_libraries(${test_exe_name} 
        PRIVATE MiniHyperVec
      )
  endforeach()
  if (CUDA_ON)
    foreach(test_exe IN LISTS sources_test_cuda)
        get_filename_component(test_exe_name ${test_exe} NAME_WE)
        add_executable(${test_exe_name} ${test_exe})
        list(APPEND TEST_EXES ${test_exe_name}) 
        target_link_libraries(${test_exe_name} 
          PRIVATE MiniHyperVec
        )
    endforeach()
  endif()
  set_target_properties(
  ${TEST_EXES}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test
  )
endif()


if (APP_BUILD)
  set(SPANN_AUG_APP_SOURCES
    ${PROJECT_SOURCE_DIR}/app/spann_build_index.cpp
  )

  set(APP_EXES)
  foreach(app_exe IN LISTS sources_app_cpp)
      get_filename_component(app_exe_name ${app_exe} NAME_WE)
      if (SPANN_AUG_BUILD AND app_exe IN_LIST SPANN_AUG_APP_SOURCES)
          continue()
      endif()

      add_executable(${app_exe_name} ${app_exe})
      list(APPEND APP_EXES ${app_exe_name})
      target_link_libraries(${app_exe_name}
        PRIVATE MiniHyperVec
      )
  endforeach()
  if (CUDA_ON)
    foreach(app_exe IN LISTS sources_app_cuda)
        get_filename_component(app_exe_name ${app_exe} NAME_WE)
        add_executable(${app_exe_name} ${app_exe})
        list(APPEND APP_EXES ${app_exe_name})
        target_link_libraries(${app_exe_name}
          PRIVATE MiniHyperVec
        )
    endforeach()
  endif()
  set_target_properties(
  ${APP_EXES}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/app
  )

  if (SPANN_AUG_BUILD)
    foreach(spann_app IN LISTS SPANN_AUG_APP_SOURCES)
        get_filename_component(spann_app_name ${spann_app} NAME_WE)
        add_executable(${spann_app_name} ${spann_app})
        target_link_libraries(${spann_app_name} PRIVATE spann_core)
        set_target_properties(${spann_app_name}
          PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/app
        )
    endforeach()
  endif()
endif()

include(GNUInstallDirs)
install(TARGETS MiniHyperVec
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}   
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}  
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.hh" PATTERN "*.hxx"  PATTERN "*.cuh"
)

if (APP_BUILD)
  install(TARGETS ${APP_EXES}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/app
  )
endif()

if (TEST_BUILD)
  install(TARGETS ${TEST_EXES}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/test
  )
endif()


