// Probe the MODERN generic cuSPARSE SpMV API (cusparseCreateCsr + cusparseSpMV),
// which amgcl guards behind CUDART_VERSION>=11000. CoreX reports CUDART 10.2 so
// amgcl falls back to the broken legacy Hyb path; here we call the generic API
// directly to check whether it works on ivcore11. A = diag(2), x = 1 -> y = 2.
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <cusparse_v2.h>

#define CK(x) do { auto rc=(x); if(rc){ std::cout<<"cusparse/cuda err "<<rc<<" @"<<__LINE__<<"\n"; } } while(0)

template <class real>
void probe(const char *tag, cudaDataType dt) {
    const int n = 8, nnz = 8;
    thrust::device_vector<int> ptr(n+1), col(n);
    thrust::device_vector<real> val(n), x(n, real(1)), y(n, real(0));
    { std::vector<int> hp(n+1), hc(n); std::vector<real> hv(n);
      for(int i=0;i<n;++i){hp[i]=i;hc[i]=i;hv[i]=real(2);} hp[n]=n;
      thrust::copy(hp.begin(),hp.end(),ptr.begin());
      thrust::copy(hc.begin(),hc.end(),col.begin());
      thrust::copy(hv.begin(),hv.end(),val.begin()); }

    cusparseHandle_t h; cusparseCreate(&h);
    cusparseSpMatDescr_t A; cusparseDnVecDescr_t X, Y;
    CK(cusparseCreateCsr(&A, n, n, nnz,
        thrust::raw_pointer_cast(ptr.data()),
        thrust::raw_pointer_cast(col.data()),
        thrust::raw_pointer_cast(val.data()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dt));
    CK(cusparseCreateDnVec(&X, n, thrust::raw_pointer_cast(x.data()), dt));
    CK(cusparseCreateDnVec(&Y, n, thrust::raw_pointer_cast(y.data()), dt));

    real alpha = real(1), beta = real(0);
    size_t bufsz = 0;
    CK(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A, X, &beta, Y, dt, CUSPARSE_MV_ALG_DEFAULT, &bufsz));
    thrust::device_vector<char> buf(bufsz > 0 ? bufsz : 1);
    CK(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A, X, &beta, Y, dt, CUSPARSE_MV_ALG_DEFAULT,
        thrust::raw_pointer_cast(buf.data())));
    cudaDeviceSynchronize();

    std::vector<real> hy(n); thrust::copy(y.begin(), y.end(), hy.begin());
    std::cout << "[" << tag << "] y = ";
    for (int i = 0; i < n; ++i) std::cout << (double)hy[i] << " ";
    std::cout << " (expected all 2)" << std::endl;
}

int main() {
    probe<float>("generic SpMV float ", CUDA_R_32F);
    probe<double>("generic SpMV double", CUDA_R_64F);
    return 0;
}
