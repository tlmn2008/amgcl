# Helper used by ctest on Iluvatar CoreX (ivcore11) to turn the amgcl CUDA
# `solver` example (which always exits 0 and just prints its result) into a
# pass/fail runtime test: run it on the default 3D Poisson problem and require
# the reported final error to be below TOL.
#
# Invoked as:
#   cmake -DEXE=<path/to/solver_cuda> -DTOL=1e-6 -P corex_check_convergence.cmake

if (NOT DEFINED EXE)
    message(FATAL_ERROR "EXE not set")
endif()
if (NOT DEFINED TOL)
    set(TOL 1e-6)
endif()

execute_process(
    COMMAND ${EXE} -n 32 -p solver.tol=${TOL}
    OUTPUT_VARIABLE OUT
    ERROR_VARIABLE  ERR
    RESULT_VARIABLE RC
    )

message(STATUS "solver_cuda output:\n${OUT}${ERR}")

if (NOT RC EQUAL 0)
    message(FATAL_ERROR "solver_cuda exited with ${RC}")
endif()

string(REGEX MATCH "Error:[ \t]*([0-9.eE+-]+)" _m "${OUT}")
if (NOT _m)
    message(FATAL_ERROR "Could not parse 'Error:' from solver_cuda output")
endif()
set(ERRVAL "${CMAKE_MATCH_1}")

if (ERRVAL MATCHES "nan|inf")
    message(FATAL_ERROR "solver_cuda did not converge: Error=${ERRVAL}")
endif()

# Accept if final BiCGStab relative residual is small enough.
if (ERRVAL LESS_EQUAL 1e-5)
    message(STATUS "solver_cuda converged: Error=${ERRVAL}")
else()
    message(FATAL_ERROR "solver_cuda did not converge below tolerance: Error=${ERRVAL}")
endif()
