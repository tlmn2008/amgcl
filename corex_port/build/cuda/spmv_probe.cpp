// Probe the cuSPARSE Hyb SpMV path (CUDART 10.2 legacy) that amgcl's cuda
// backend uses on CoreX, for float and double. A = diag(2) on 8 rows, x = 1,
// so y = A*x should be all 2.
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <amgcl/backend/cuda.hpp>

template <class real>
void probe(const char *tag) {
    const int n = 8;
    std::vector<ptrdiff_t> ptr(n+1), col(n);
    std::vector<real> val(n);
    for (int i = 0; i < n; ++i) { ptr[i] = i; col[i] = i; val[i] = real(2); }
    ptr[n] = n;

    cusparseHandle_t h; cusparseCreate(&h);
    amgcl::backend::cuda_matrix<real> A(n, n, ptr.data(), col.data(), val.data(), h);

    thrust::device_vector<real> x(n, real(1)), y(n, real(0));
    A.spmv(real(1), x, real(0), y);
    cudaDeviceSynchronize();

    std::vector<real> hy(n);
    thrust::copy(y.begin(), y.end(), hy.begin());
    std::cout << "[" << tag << "] y = ";
    for (int i = 0; i < n; ++i) std::cout << (double)hy[i] << " ";
    std::cout << " (expected all 2)" << std::endl;
}

int main() {
    try { probe<float>("SpMV float "); }
    catch (std::exception &e) { std::cout << "[SpMV float ] THREW: " << e.what() << std::endl; }
    try { probe<double>("SpMV double"); }
    catch (std::exception &e) { std::cout << "[SpMV double] THREW: " << e.what() << std::endl; }
    return 0;
}
