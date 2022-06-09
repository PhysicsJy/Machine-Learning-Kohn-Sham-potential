/* This file defines the settings of the code repository and the system*/
#ifndef __common_h__
#define __common_h__
#include <complex>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/math/special_functions/rsqrt.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/math/constants/constants.hpp>
#include <fftw3.h>
//
// ublas includes
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/hermitian.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/lu.hpp>

// mkl includes
#include <stdlib.h>
#include <stdio.h>
#include <lapacke.h>
#include <omp.h>

namespace ublas = boost::numeric::ublas;
using real = double;
using complex = std::complex<real>;
using vector_complex = ublas::vector<complex>;
using vector_real = ublas::vector<real>;
using complex_sparse_matrix = ublas::compressed_matrix<complex>;
using real_sparse_matrix = ublas::compressed_matrix<real>;
using matrix_complex = ublas::matrix<complex>;
using matrix_real = ublas::matrix<real>;

#define m_pi boost::math::constants::pi<real>()
#define MKL_INT int
constexpr real dt = 0.5e-3;
constexpr real xsmall = 1.e-6;
constexpr real lmbd = 1.e-6;
constexpr real sys_xmin = -60.0;
constexpr real sys_xmax = 60.0;
constexpr int BC = 2;
constexpr int sys_size = 512;

constexpr int num_threads = 8;
#endif
