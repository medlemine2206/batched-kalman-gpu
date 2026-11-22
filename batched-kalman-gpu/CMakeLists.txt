cmake_minimum_required(VERSION 3.24)
project(batched_kalman_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- options -------------------------------------------------------------------
set(KF_ARCH "75" CACHE STRING "GPU compute capability without dot (75=T4, 80=A100, 86=RTX30xx, 89=L4/RTX40xx, 90=H100)")
set(MATHDX_INCLUDE "" CACHE PATH "MathDx include dir (the one containing cublasdx.hpp)")
set(CUTLASS_INCLUDE "" CACHE PATH "CUTLASS include dir (shipped with MathDx under external/cutlass/include)")
option(KF_WITH_CUSOLVERDX "Use cuSolverDx POSV in the Dx kernel (needs the full MathDx package)" OFF)
set(MATHDX_LIB_DIR "" CACHE PATH "Directory containing libcusolverdx.a / libcusolverdx.fatbin")

# Auto-detect MathDx installed with:  pip install nvidia-mathdx
if(NOT MATHDX_INCLUDE)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/find_mathdx.py
      OUTPUT_VARIABLE _mdx OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _mdx_rc)
    if(_mdx_rc EQUAL 0 AND _mdx)
      set(MATHDX_INCLUDE "${_mdx}/include")
      set(CUTLASS_INCLUDE "${_mdx}/external/cutlass/include")
    endif()
  endif()
endif()
if(NOT EXISTS "${MATHDX_INCLUDE}/cublasdx.hpp")
  message(FATAL_ERROR "cublasdx.hpp not found. Run `pip install nvidia-mathdx` or pass -DMATHDX_INCLUDE=...")
endif()
message(STATUS "MathDx include : ${MATHDX_INCLUDE}")
message(STATUS "CUTLASS include: ${CUTLASS_INCLUDE}")
message(STATUS "GPU arch       : sm_${KF_ARCH}")

find_package(CUDAToolkit REQUIRED)

# ---- benchmark -----------------------------------------------------------------
add_executable(kf_bench src/main.cu)
set_target_properties(kf_bench PROPERTIES CUDA_ARCHITECTURES ${KF_ARCH})
target_compile_definitions(kf_bench PRIVATE KF_SM=${KF_ARCH}0)
target_include_directories(kf_bench PRIVATE include src ${MATHDX_INCLUDE} ${CUTLASS_INCLUDE})
target_compile_options(kf_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xptxas=-v>)
target_link_libraries(kf_bench PRIVATE CUDA::cublas CUDA::cudart)

if(KF_WITH_CUSOLVERDX)
  # cuSolverDx ships as LTO-IR: relocatable device code + device link-time optimisation
  target_compile_definitions(kf_bench PRIVATE KF_WITH_CUSOLVERDX)
  set_target_properties(kf_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON INTERPROCEDURAL_OPTIMIZATION ON)
  if(EXISTS "${MATHDX_LIB_DIR}/libcusolverdx.fatbin" AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    target_link_options(kf_bench PRIVATE $<DEVICE_LINK:${MATHDX_LIB_DIR}/libcusolverdx.fatbin>)
  else()
    target_link_libraries(kf_bench PRIVATE ${MATHDX_LIB_DIR}/libcusolverdx.a)
  endif()
  target_link_options(kf_bench PRIVATE $<DEVICE_LINK:-dlto>)
endif()

# ---- CPU tests (no GPU needed) -------------------------------------------------
add_executable(test_cpu tests/test_cpu.cpp)
target_include_directories(test_cpu PRIVATE include)
add_executable(test_layout_logic tests/test_layout_logic.cpp)
target_include_directories(test_layout_logic PRIVATE include)

enable_testing()
add_test(NAME cpu_reference COMMAND test_cpu 2000 200 ${CMAKE_BINARY_DIR}/cpu_rmse.csv)
add_test(NAME layout_logic COMMAND test_layout_logic)
add_test(NAME gpu_correctness COMMAND kf_bench --filters 4100 --steps 50 --repeats 1
         --csv ${CMAKE_BINARY_DIR}/ctest_bench.csv)
set_tests_properties(gpu_correctness PROPERTIES FAIL_REGULAR_EXPRESSION "MISMATCH")
