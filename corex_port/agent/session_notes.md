# amgcl → Iluvatar CoreX (ivcore11) 迁移记录

## 一、来源
- 仓库：https://github.com/ddemidov/amgcl.git
- 分支/commit：master @ `3f7970bdd5ab07c4e14eca1689b6bf5fe077a6a4`（提交日期 2026-07-18T07:42:56+03:00）
- 子模块：`pyamgcl/pybind11`（递归 init，本次未涉及 Python 绑定构建）
- 性质：header-only C++ 代数多重网格（AMG）求解库。AMG 层级在 CPU（builtin/Eigen 后端）构建，求解阶段可切到多种后端（OpenMP / VexCL / ViennaCL / **CUDA** / HIP）。

## 二、CUDA 使用性质
- CUDA 后端为 `amgcl/backend/cuda.hpp`：向量运算用 **thrust**，稀疏矩阵向量乘（SpMV）用 **cuSPARSE**；直接求解器把粗层拷回 host 用 skyline_lu。ILU0 光滑子在 `amgcl/relaxation/cusparse_ilu0.hpp`。
- 关键点：backend 的 `cuda_matrix` 按 `CUDART_VERSION` 二选一——`>=11000` 用现代 generic API（`cusparseCreateCsr`+`cusparseSpMV`），`<11000` 用 **legacy Hyb**（`cusparseScsr2hyb`+`cusparse{S,D}hybmv`）。
- 所有示例硬编码 `amgcl::backend::cuda<double>`（科学计算求解，重度依赖 double）。
- 上游 CUDA 示例经 `cuda_add_executable()`（nvcc 的 FindCUDA 模块）构建；仓库**没有任何 CUDA 版 ctest 用例**，CUDA 后端原本只由 examples/tutorial 驱动。

## 三、环境
- SDK：CoreX 4.5.0.20260630（`clang version 22.1.0git (4.5.0.20260630 ...)`）
- 编译器：`/usr/local/corex/bin/clang++`（前端 clang，`-x ivcore --cuda-gpu-arch=ivcore11`）
- 运行时报告：`ixsmi` 显示 IX-ML 4.4.0 / Driver 4.5.0 / **CUDA Version 10.2**；`CUDART_VERSION=10020`、`CUSPARSE_VERSION=10301`
- 硬件：2× Iluvatar BI-V150（32GiB），预检时 GPU-Util 0%、无残留进程
- 依赖：CMake 3.31.8(corex)、Boost 1.83（program_options/serialization/unit_test_framework）、Eigen 可用、OpenMP 未装（builtin 串行运行不影响正确性）

## 四、适配内容（均为平台门控、非侵入，NVIDIA 默认路径不变）
1. **构建系统（CMakeLists.txt）**：新增 `option(USE_COREX)`，依据 `/usr/local/corex` 自动探测。CoreX 上以 INTERFACE 库 `cuda_target` 直接链接 CoreX 的 `libcusparse/libcudart/libcublas`，并定义
   `AMGCL_COREX_CUDA_FLAGS = -x ivcore --cuda-path=/usr/local/corex --cuda-gpu-arch=ivcore11 -std=c++17`
   （`-std=c++17` 是 ixthrust 硬性要求）。上游 `find_package(CUDA)`+nvcc 分支移到 `elseif`（`USE_COREX=OFF` 时逐字节不变）。
2. **示例/教程（examples/CMakeLists.txt、tutorial/1.poisson3Db/CMakeLists.txt）**：新增 `USE_COREX` 分支，用 `clang++ -x ivcore` 直接编译 `.cpp/.cu`，取代 `cuda_add_executable`。`examples/mpi` 的 MPI+CUDA 分布式示例（需多 rank + 输入矩阵）按 ≤2GPU 约束在 CoreX 下跳过。
3. **CUDA 后端（amgcl/backend/cuda.hpp）**：新增宏 `AMGCL_CUDA_COREX`。定义时 `AMGCL_CUDA_USE_GENERIC_SPMV=1`，强制走现代 generic `cusparseSpMV`（即便 CUDART 报 10.2），并把 11.0 才有的 `CUSPARSE_SPMV_CSR_ALG1` 映射为 CoreX 10.2 的 `CUSPARSE_MV_ALG_DEFAULT`。未定义该宏时等价于原 `CUDART_VERSION>=11000` 判断。
4. **精度门控（examples/solver.cpp）**：新增宏 `AMGCL_CUDA_COREX_FP32`，把 CUDA 后端 value_type 由 `double` 降为 `float`（守卫式，默认 double 分支保留）。
5. **测试注册（cmake/corex_check_convergence.cmake）**：把 `solver_cuda` 在 3D Poisson 上的求解包装成收敛判定，注册为 ctest 用例 `test_solver_cuda`，使 `ctest` 全量覆盖迁移后的 GPU 求解路径。

## 五、结果
- **主机侧 ctest（12 用例）**：零源码改动，全部通过（builtin/complex/block_crs/ns_builtin/skyline_lu/qr/io/static_matrix/eigen 系列）。x86 主机有真 FP64，double 在 CPU 上正常。
- **CUDA 后端**：4 个 CUDA 示例（solver/schur_pressure_correction/cpr_drs/poisson3Db）均经 CMake 用 `clang++ -x ivcore` 编译成功。启用 `AMGCL_CUDA_COREX`（generic SpMV）+ `AMGCL_CUDA_COREX_FP32`（float）后，`solver_cuda` 在 3D Poisson(32³=32768 未知量) 上收敛（6~7 步，残差 ~1e-7~1e-9）。
- **ctest 全量**：13/13 通过（12 CPU + `test_solver_cuda`）。
- `overall_status=migrated`，`compile_status=success`，`test_status=passed`。

## 六、Failure Gate（真实复现，判 verdict/status）
逐条真实运行复现，日志落 `corex_port/build/cuda/`：
1. **nvcc 构建不可用**（build / workaround / fixed）：`cuda_add_executable` 未定义 → 改平台门控 clang++ `-x ivcore` 构建。已实测编译产物可用。
2. **legacy cuSPARSE Hyb SpMV 静默 no-op**（api / workaround / fixed）：CUDART 报 10.2 使 amgcl 走 Hyb 路径；最小复现 `spmv_probe`（A=diag(2),x=1 期望 2）实测 **float/double 都返回全 0**；对照 `generic_spmv_probe` 现代 API **float 得全 2**。改用平台门控 generic SpMV 后求解收敛。判 workaround/fixed。
3. **FP64/double 无真硬件支持**（precision / **corex-issue** / **open**）：`fp64_probe` 实测 `thrust::inner_product<float>` 正确、`<double>` 抛 `ixErrorSymbolNotFound`（device 双精度归约符号缺失）；`generic_spmv_probe` 现代 `cusparseSpMV<double>` 返回 `NOT_SUPPORTED`(err 10)；端到端 double 版 `solver_cuda` 在双精度 SpMV 处抛 err 10。已尝试：①强制 generic API 仍 NOT_SUPPORTED（排除只是 Hyb 的问题）；②按红线 3/5 做平台门控 FP32 降级——降级后可用。double 本身不改 `/usr/local/corex` 无解，判 corex-issue/open。

## 七、红线遵守
- 未修改/安装 `/usr/local/corex` 任何内容；未用 nvcc、未造 nvcc 假 stub；double/FP64 用平台宏门控降级而非删除上游代码路径；MPI+CUDA 示例限制在 ≤2 GPU（直接跳过分布式多 rank）；所有 CoreX 改动都在 `USE_COREX`/`AMGCL_CUDA_COREX`/`AMGCL_CUDA_COREX_FP32` 开关后，NVIDIA 默认路径零回归。

## 八、工具链细节（不入 sdk_version）
- `clang++ --version`: clang 22.1.0git (4.5.0.20260630)
- CUDA 编译：`clang++ -x ivcore --cuda-path=/usr/local/corex --cuda-gpu-arch=ivcore11 -std=c++17`
- 链接库：`/usr/local/corex/lib/{libcusparse,libcudart,libcublas}.so`
- CMake 3.31.8+corex.4.5.0.20260630；Boost 1.83
