// Root-cause probe for amgcl CUDA backend on ivcore11.
// 1) thrust::inner_product over double vs float (the reduction amgcl relies on).
// 2) amgcl AMG+BiCGStab solve of the same Poisson problem in float and double.
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include <amgcl/backend/cuda.hpp>
#include <amgcl/relaxation/cusparse_ilu0.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include "sample_problem.hpp"

template <class real>
void solve_and_report(const char *tag,
        size_t rows,
        std::vector<ptrdiff_t> const &ptr,
        std::vector<ptrdiff_t> const &col,
        std::vector<double>    const &val_d,
        std::vector<double>    const &rhs_d)
{
    typedef amgcl::backend::cuda<real> Backend;
    typename Backend::params bprm;
    cusparseCreate(&bprm.cusparse_handle);

    std::vector<real> val(val_d.begin(), val_d.end());
    std::vector<real> rhs(rhs_d.begin(), rhs_d.end());

    typedef amgcl::make_solver<
        amgcl::amg<Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::bicgstab<Backend>
        > Solver;

    typename Solver::params prm;
    Solver solve(std::tie(rows, ptr, col, val), prm, bprm);

    thrust::device_vector<real> f(rhs.begin(), rhs.end());
    thrust::device_vector<real> x(rows, real(0));

    size_t iters; double error;
    std::tie(iters, error) = solve(f, x);
    std::cout << "[" << tag << "] iters=" << iters << " error=" << error << std::endl;
}

int main() {
    // thrust::inner_product probe (all-ones dot product == N)
    const int N = 100000;
    try {
        thrust::device_vector<float> a(N, 1.0f), b(N, 1.0f);
        float dot = thrust::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
        std::cout << "[probe] thrust::inner_product<float>  ones dot = " << dot
                  << " (expected " << (float)N << ")" << std::endl;
    } catch (std::exception &e) {
        std::cout << "[probe] thrust::inner_product<float>  THREW: " << e.what() << std::endl;
    }
    try {
        thrust::device_vector<double> a(N, 1.0), b(N, 1.0);
        double dot = thrust::inner_product(a.begin(), a.end(), b.begin(), 0.0);
        std::cout << "[probe] thrust::inner_product<double> ones dot = " << dot
                  << " (expected " << (double)N << ")" << std::endl;
    } catch (std::exception &e) {
        std::cout << "[probe] thrust::inner_product<double> THREW: " << e.what() << std::endl;
    }

    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, rhs;
    size_t rows = sample_problem(32, val, col, ptr, rhs, 1.0);

    try { solve_and_report<float>("cuda<float> ", rows, ptr, col, val, rhs); }
    catch (std::exception &e) { std::cout << "[cuda<float> ] THREW: " << e.what() << std::endl; }
    try { solve_and_report<double>("cuda<double>", rows, ptr, col, val, rhs); }
    catch (std::exception &e) { std::cout << "[cuda<double>] THREW: " << e.what() << std::endl; }
    return 0;
}
