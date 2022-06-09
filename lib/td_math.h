#ifndef __TD_MATH_H__
#define __TD_MATH_H__
#include <assert.h>
#include <functional>
#include "common.h"
namespace td_math
{
  /**
   * @brief
   *
   *  The routine computes the minimum-norm solution to a real linear least
      squares problem: minimize ||b - A*x|| using the singular value
      decomposition (SVD) of A. A is an m-by-n matrix which may be rank-deficient.

      Several right hand side vectors b and solution vectors x can be handled
      in a single call; they are stored as the columns of the m-by-nrhs right
      hand side matrix B and the n-by-nrhs solution matrix X.

      The effective rank of A is determined by treating as zero those singular
      values which are less than rcond times the largest singular value.
   *
   * @param M number of rows in A
   * @param N number of columns(rows) in A(b)
   * @param vect x
   * @param amat A
   * @param bvect b
   * @param rcond is used to determine the effective rank of matrix A. Singular values
   *              of si (rcond)(si) are treated as zero.
                  If rcond is less than or equal to zero or rcond is greater than or
                  equal to one, then an rcond value of ε is used, where ε is the
                  machine precision.
   */
  static void dgelsd_lsquare(int M, int N, vector_real &vect, matrix_real &amat, vector_real &bvect, real rcond)
  {
    int NRHS = 1;
    MKL_INT m = M, n = N, nrhs = NRHS, info, rank;
    MKL_INT lda = std::max(m, n), ldb = nrhs;
    int l = std::max(1, std::min(m, n));
    matrix_real &a = amat;
    vector_real s(l);
    vector_real &b = bvect;
    // b = bvect;
    info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, m, n, nrhs, &a(0, 0), lda, &b[0], ldb,
                          &s[0], rcond, &rank);
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
      vect[i] = b[i];
    }
  }

  static void zheev_diag(int n, matrix_complex &amat, vector_real &eigval, matrix_complex &eigvec)
  {
    int info;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < amat.size1(); ++i)
    {
      for (int j = 0; j < amat.size2(); ++j)
      {
        eigvec(i, j) = amat(i, j);
      }
    }
    info = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', n, reinterpret_cast<__complex__ double *>(&eigvec(0, 0)), n, &eigval(0));
  }

  inline complex dotc(vector_complex v1, vector_complex v2)
  {
    return inner_prod(conj(v1), v2);
  }

  // worth trying QR decomposition here
  static void SchmidtOrthonormalization(std::vector<vector_complex> &vec_set)
  {
    for (size_t i = 0; i < vec_set.size(); ++i)
    {
      vector_complex &v_i = vec_set[i];
      for (int j = 0; j < i; ++j)
      {
        vector_complex &v_j = vec_set[j];
        complex coef = dotc(v_j, v_i);
        v_i -= coef * v_j;
      }
      v_i /= norm_2(v_i);
    }
  }

  template <class T>
  void FourierTransformation(ublas::vector<T> &x, bool forward)
  {
    int size = x.size();
    ublas::vector<T> y(size, complex(0.0, 0.0));
    fftw_plan p;

    fftw_complex *in = reinterpret_cast<fftw_complex *>(&x[0]);
    fftw_complex *out = reinterpret_cast<fftw_complex *>(&y[0]);
    p = fftw_plan_dft_1d(size, in, out, (forward ? FFTW_FORWARD : FFTW_BACKWARD), FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
      x[i] = y[i] * boost::math::rsqrt(static_cast<real>(size));
    }
  }

  template <class T>
  void FourierTransformation2D(ublas::matrix<T> &m, bool forward)
  {
    int size1 = m.size1();
    int size2 = m.size2();
    ublas::matrix<T> fm(size1, size2);
    fftw_plan p;

    fftw_complex *in = reinterpret_cast<fftw_complex *>(&m(0, 0));
    fftw_complex *out = reinterpret_cast<fftw_complex *>(&fm(0, 0));
    p = fftw_plan_dft_2d(size1, size2, in, out, (forward ? FFTW_FORWARD : FFTW_BACKWARD), FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    int size_square = size1 * size2;
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < size1; ++i)
    {
      for (size_t j = 0; j < size2; ++j)
      {
        m(i, j) = fm(i, j) * boost::math::rsqrt(static_cast<real>(size_square));
      }
    }
  }

  static complex
  GaussianWavePacket(real x, real center, real alpha, real p)
  {
    return std::pow(2 * alpha / m_pi, 0.25) *
           std::exp(-alpha * (x - center) * (x - center) + complex(0., 1.) * p * (x - center));
  }

  template <class T>
  T Deriv_1_7p_central(ublas::vector<T> &arr, int i, real dx)
  {
    T deriv = (-1. / 60.) * arr[i + 0] + (3.0 / 20.0) * arr[i + 1] + (-3.0 / 4.0) * arr[i + 2] +
              (3. / 4.) * arr[i + 4] + (-3.0 / 20.0) * arr[i + 5] + (1.0 / 60.0) * arr[i + 6];
    return deriv / dx;
  }

  template <class T>
  T Deriv_1_7p_left(ublas::vector<T> &arr, int i, real dx)
  {
    T deriv = (-49. / 20.) * arr[i + 0] + (6.) * arr[i + 1] + (-15.0 / 2.0) * arr[i + 2] + (20. / 3.) * arr[i + 3] +
              (-15.0 / 4.0) * arr[i + 4] + (6.0 / 5.0) * arr[i + 5] + (-1. / 6.) * arr[i + 6];
    return deriv / dx;
  }

  template <class T>
  T Deriv_1_7p_right(ublas::vector<T> &arr, int i, real dx)
  {
    T deriv = (1. / 6.) * arr[i + 0] + (-6. / 5.) * arr[i + 1] + (15.0 / 4.0) * arr[i + 2] + (-20. / 3.) * arr[i + 3] +
              (15.0 / 2.0) * arr[i + 4] + (-6.) * arr[i + 5] + (49. / 20.) * arr[i + 6];
    return deriv / dx;
  }

  template <class T>
  void Gradient_1d_7p(ublas::vector<T> &arr, ublas::vector<T> &grad_arr, real dx)
  {
    assert(arr.size() > 7);

    for (int i = 0; i < 4; ++i)
    {
      grad_arr[i] = Deriv_1_7p_left(arr, i, dx);
    }

    for (int i = 4; i < arr.size() - 3; ++i)
    {
      grad_arr[i] = Deriv_1_7p_central(arr, i - 3, dx);
    }

    for (int i = arr.size() - 3; i < arr.size(); ++i)
    {
      grad_arr[i] = Deriv_1_7p_right(arr, i - 6, dx);
    }
  }

  template <class T>
  T Deriv_2_8p_central(ublas::vector<T> &arr, int start, real dx)
  {
    T deriv = (1.0 / 90.0) * arr[start + 0] + (-3.0 / 20.0) * arr[start + 1] + (3.0 / 2.0) * arr[start + 2] +
              (-49.0 / 18.0) * arr[start + 3] + (3.0 / 2.0) * arr[start + 4] + (-3.0 / 20.0) * arr[start + 5] +
              (1.0 / 90.0) * arr[start + 6];
    return deriv / (dx * dx);
  }

  template <class T>
  T Deriv_2_8p_left(ublas::vector<T> &arr, int start, real dx)
  {
    T deriv = (469.0 / 90.0) * arr[start + 0] + (-223.0 / 10.0) * arr[start + 1] + (879.0 / 20.0) * arr[start + 2] +
              (-949.0 / 18.0) * arr[start + 3] + (41.0) * arr[start + 4] + (-201.0 / 10.0) * arr[start + 5] +
              (1019.0 / 180.0) * arr[start + 6] + (-7.0 / 10.0) * arr[start + 7];
    return deriv / (dx * dx);
  }

  template <class T>
  T Deriv_2_8p_right(ublas::vector<T> &arr, int start, real dx)
  {
    T deriv = (469.0 / 90.0) * arr[start + 7] + (-223.0 / 10.0) * arr[start + 6] + (879.0 / 20.0) * arr[start + 5] +
              (-949.0 / 18.0) * arr[start + 4] + (41.0) * arr[start + 3] + (-201.0 / 10.0) * arr[start + 2] +
              (1019.0 / 180.0) * arr[start + 1] + (-7.0 / 10.0) * arr[start + 0];
    return deriv / (dx * dx);
  }

  template <class T>
  void Laplacian_1d_8p(ublas::vector<T> &arr, ublas::vector<T> &laplacian_arr, real dx)
  {
    assert(arr.size() > 8);

    for (int i = 0; i < 3; ++i)
    {
      laplacian_arr[i] = Deriv_2_8p_left(arr, i, dx);
    }

    for (int i = 3; i < arr.size() - 3; ++i)
    {
      laplacian_arr[i] = Deriv_2_8p_central(arr, i - 3, dx);
    }

    for (int i = arr.size() - 3; i < arr.size(); ++i)
    {
      laplacian_arr[i] = Deriv_2_8p_right(arr, i - 7, dx);
    }
  }

  template <class T>
  T Partial_y_central(ublas::matrix<T> &mat, int i, int j, real dx)
  {
    T deriv = (-1. / 60.) * mat(i, j + 0) + (3.0 / 20.0) * mat(i, j + 1) + (-3.0 / 4.0) * mat(i, j + 2) +
              (3. / 4.) * mat(i, j + 4) + (-3.0 / 20.0) * mat(i, j + 5) + (1.0 / 60.0) * mat(i, j + 6);
    return deriv / dx;
  }

  template <class T>
  T Partial_y_left(ublas::matrix<T> &mat, int i, int j, real dx)
  {
    T deriv = (-49. / 20.) * mat(i, j + 0) + (6.) * mat(i, j + 1) + (-15.0 / 2.0) * mat(i, j + 2) +
              (20. / 3.) * mat(i, j + 3) + (-15.0 / 4.0) * mat(i, j + 4) + (6.0 / 5.0) * mat(i, j + 5) +
              (-1. / 6.) * mat(i, j + 6);
    return deriv / dx;
  }

  template <class T>
  T Partial_y_right(ublas::matrix<T> &mat, int i, int j, real dx)
  {
    T deriv = (1. / 6.) * mat(i, j + 0) + (-6. / 5.) * mat(i, j + 1) + (15.0 / 4.0) * mat(i, j + 2) +
              (-20. / 3.) * mat(i, j + 3) + (15.0 / 2.0) * mat(i, j + 4) + (-6.) * mat(i, j + 5) +
              (49. / 20.) * mat(i, j + 6);
    return deriv / dx;
  }

  template <class T>
  void Partial_y_2d(ublas::matrix<T> &mat, ublas::matrix<T> &partial_y_mat, real dx)
  {
    assert(mat.size2() > 7);
    for (int i = 0; i < mat.size1(); ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        partial_y_mat(i, j) = Partial_y_left(mat, i, j, dx);
      }

      for (int j = 3; j < mat.size2() - 3; ++j)
      {
        partial_y_mat(i, j) = Partial_y_central(mat, i, j - 3, dx);
      }

      for (int j = mat.size2() - 3; j < mat.size2(); ++j)
      {
        partial_y_mat(i, j) = Partial_y_right(mat, i, j - 6, dx);
      }
    }
  }

  template <class T>
  void Partial_x_2d(ublas::matrix<T> &mat, ublas::matrix<T> &partial_x_mat, real dx)
  {
    assert(mat.size1() > 7);

    ublas::matrix<T> mat_trans = ublas::trans(mat);
    partial_x_mat = ublas::trans(partial_x_mat);

    Partial_y_2d(mat_trans, partial_x_mat, dx);

    partial_x_mat = ublas::trans(partial_x_mat);
  }

  template <class T>
  T Partial_yy_central(ublas::matrix<T> &mat, int i, int j, real dx)
  {
    T deriv = (1. / 90.) * mat(i, j + 0) + (-3.0 / 20.0) * mat(i, j + 1) + (3.0 / 2.0) * mat(i, j + 2) +
              (-49.0 / 18.0) * mat(i, j + 3) + (3.0 / 2.0) * mat(i, j + 4) + (-3.0 / 20.0) * mat(i, j + 5) +
              (1.0 / 90.0) * mat(i, j + 6);
    return deriv / (dx * dx);
  }

  template <class T>
  T Partial_yy_left(ublas::matrix<T> &mat, int i, int j, real dx)
  {
    T deriv = (469.0 / 90.0) * mat(i, j + 0) + (-223.0 / 10.0) * mat(i, j + 1) + (879.0 / 20.0) * mat(i, j + 2) +
              (-949.0 / 18.0) * mat(i, j + 3) + (41.0) * mat(i, j + 4) + (-201.0 / 10.0) * mat(i, j + 5) +
              (1019.0 / 180.0) * mat(i, j + 6) + (-7.0 / 10.0) * mat(i, j + 7);
    return deriv / (dx * dx);
  }

  template <class T>
  T Partial_yy_right(ublas::matrix<T> &mat, int i, int j, real dx)
  {
    T deriv = (469.0 / 90.0) * mat(i, j + 7) + (-223.0 / 10.0) * mat(i, j + 6) + (879.0 / 20.0) * mat(i, j + 5) +
              (-949.0 / 18.0) * mat(i, j + 4) + (41.0) * mat(i, j + 3) + (-201.0 / 10.0) * mat(i, j + 2) +
              (1019.0 / 180.0) * mat(i, j + 1) + (-7.0 / 10.0) * mat(i, j + 0);
    return deriv / (dx * dx);
  }

  template <class T>
  void Partial_yy_2d(ublas::matrix<T> &mat, ublas::matrix<T> &partial_yy_mat, real dx)
  {
    assert(mat.size2() > 8);

    for (int i = 0; i < mat.size1(); ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        partial_yy_mat(i, j) = Partial_yy_left(mat, i, j, dx);
      }

      for (int j = 3; j < mat.size2() - 4; ++j)
      {
        partial_yy_mat(i, j) = Partial_yy_central(mat, i, j - 3, dx);
      }

      for (int j = mat.size2() - 4; j < mat.size2(); ++j)
      {
        partial_yy_mat(i, j) = Partial_yy_right(mat, i, j - 7, dx);
      }
    }
  }

  template <class T>
  void Partial_xx_2d(ublas::matrix<T> &mat, ublas::matrix<T> &partial_xx_mat, real dx)
  {
    assert(mat.size1() > 8);
    ublas::matrix<T> mat_trans = ublas::trans(mat);
    for (int i = 0; i < mat_trans.size1(); ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        partial_xx_mat(i, j) = Partial_yy_left(mat_trans, i, j, dx);
      }

      for (int j = 3; j < mat_trans.size2() - 4; ++j)
      {
        partial_xx_mat(i, j) = Partial_yy_central(mat_trans, i, j - 3, dx);
      }

      for (int j = mat_trans.size2() - 4; j < mat_trans.size2(); ++j)
      {
        partial_xx_mat(i, j) = Partial_yy_right(mat_trans, i, j - 7, dx);
      }
    }

    partial_xx_mat = ublas::trans(partial_xx_mat);
  }

  inline int modulo(int i, int n)
  {
    assert(n > 0);
    return (i % n + n) % n;
  }

  /**
   * @brief Helper function for handling the boundary value of a 1d array
   *
   * @tparam T real or complex
   * @param boundary_condition type of boundary conditions:
   *                           3. open boundary condition (value vanishes out of the boundary)
   *                           2. periodic boundary condition
   *                           1. symmetirc boundary condition
   *                          -1. anti-symmetric boundary condition
   * @param v input vector
   * @param i the index of the value to be modified
   * @return T the modified value
   */
  template <class T>
  T Rval_1d(int boundary_condition, ublas::vector<T> &v, int i)
  {
    int v_len = v.size();
    if (boundary_condition == 3)
    {
      if (i >= 0 && i < v_len)
        return v[i];
      return 0;
    }
    else if (boundary_condition == 2)
    {
      int l = modulo(i, v_len);
      return v[l];
    }
    int l = modulo(i, 2 * v_len);
    if (l < v_len)
    {
      return v[l];
    }
    l = v_len - 1 - modulo(l, v_len);
    return (real)boundary_condition * v[l];
  }

  template <class T>
  T RightForwardIntgrate(int boundary_condition, ublas::vector<T> arr, int i, real dx)
  {
    // can also try some other methods with higher formulas
    T result = 0.5 * (Rval_1d(boundary_condition, arr, i - 1) + Rval_1d(boundary_condition, arr, i)) * dx;

    // T result = (11.0 * Rval_1d(boundary_condition, arr, i - 3) -
    //             93.0 * Rval_1d(boundary_condition, arr, i - 2) +
    //             802.0 * Rval_1d(boundary_condition, arr, i - 1) +
    //             802.0 * Rval_1d(boundary_condition, arr, i) -
    //             93.0 * Rval_1d(boundary_condition, arr, i + 1) +
    //             11.0 * Rval_1d(boundary_condition, arr, i + 2)) *
    //            dx / 1440.0;

    // T result = (2497.0 * Rval_1d(boundary_condition, arr, i - 5) -
    //             28939.0 * Rval_1d(boundary_condition, arr, i - 4) +
    //             162680.0 * Rval_1d(boundary_condition, arr, i - 3) -
    //             641776.0 * Rval_1d(boundary_condition, arr, i - 2) +
    //             4134338.0 * Rval_1d(boundary_condition, arr, i - 1) +
    //             4134338.0 * Rval_1d(boundary_condition, arr, i) -
    //             641776.0 * Rval_1d(boundary_condition, arr, i + 1) +
    //             162680.0 * Rval_1d(boundary_condition, arr, i + 2) -
    //             28939.0 * Rval_1d(boundary_condition, arr, i + 3) +
    //             2497.0 * Rval_1d(boundary_condition, arr, i + 4)) *
    //            dx / 7257600.0;

    return result;
  }

  template <class T>
  void Antideriv_r(int boundary_condition, ublas::vector<T> &inarr, ublas::vector<T> &outarr, real dx)
  {
    assert(inarr.size() > 6);

    for (int i = 0; i < inarr.size(); ++i)
    {
      outarr[i] = Rval_1d(boundary_condition, outarr, i - 1) + RightForwardIntgrate(boundary_condition, inarr, i, dx);
    }
  }

  template <class T>
  void Cumtrapz(ublas::matrix<T>& inarr, ublas::matrix<T>& outarr, real dx) {
      for (int i = 0; i < outarr.size1(); ++i) {
          outarr(i, 0) = 0.0;
          for (int j = 1; j < outarr.size2(); ++j) {
              outarr(i, j) = outarr(i, j-1) + 0.5 * (inarr(i, j) + inarr(i, j-1)) * dx;
          }
      }
  }

  /**
   * @brief Helper function for handling the boundary value of a 2d matrix
   *
   * @tparam T real or complex
   * @param boundary_condition type of boundary conditions
   *                           3. open boundary condition (value vanishes out of the boundary)
   *                           2. periodic boundary condition
   *                           1. symmetirc boundary condition
   *                          -1. anti-symmetric boundary condition
   * @param mat input matrix
   * @param i row index
   * @param j col index
   * @return T modified value
   */
  template <class T>
  T Zval_2d(int boundary_condition, ublas::matrix<T> &mat, int i, int j)
  {
    int mm = mat.size1();
    int ll = mat.size2();
    if (abs(boundary_condition) == 3)
    {
      if (i < mm && i >= 0 && j < ll && j >= 0)
      {
        return mat(i, j);
      }
      else
      {
        return 0;
      }
    }
    else if (abs(boundary_condition) == 2)
    {
      int m = modulo(i, mm);
      int l = modulo(j, ll);
      return mat(m, l);
    }
    else if (abs(boundary_condition) <= 1)
    {
      int m = modulo(i, 2 * mm);
      real sgn = 1.0;

      if (m < mm)
      {
        sgn = 1.0;
      }
      else
      {
        m = mm - 1 - modulo(m, mm);
        sgn = (real)boundary_condition;
      }

      int l = modulo(j, 2 * ll);

      if (l < ll)
      {
        sgn = 1.0;
      }
      else
      {
        l = ll - 1 - modulo(l, ll);
        sgn = (real)boundary_condition;
      }
      return sgn * mat(m, l);
    }

    return -1;
  }

  /**
   * @brief Calculate ∂^2_x of a matrix ....
   *
   * @tparam T complex or real type
   * @param boundary_condition type of boundary conditions:
   *                           3. open boundary condition (value vanishes out of the boundary)
   *                           2. periodic boundary condition
   *                           1. symmetirc boundary condition
   *                          -1. anti-symmetric boundary condition
   * @param inmatrix input matrix M
   * @param outmatrix output matrix ∂^2_x M
   */
  template <class T>
  void BC_Partial_xx_2d(int boundary_condition, ublas::matrix<T> &inmatrix, ublas::matrix<T> &outmatrix, real dx)
  {
    ublas::vector<T> v_temp(7);

    for (int i = 0; i < inmatrix.size1(); ++i)
    {
      for (int j = 0; j < inmatrix.size2(); ++j)
      {
        for (int k = 0; k < 7; ++k)
        {
          v_temp[k] = Zval_2d(boundary_condition, inmatrix, i - 3 + k, j);
        }
        outmatrix(i, j) = Deriv_2_8p_central(v_temp, 0, dx);
      }
    }
  }

/**
 * @brief Calculate ∂^2_x of a matrix ....
 *
 * @tparam T complex or real type
 * @param boundary_condition type of boundary conditions:
 *                           3. open boundary condition (value vanishes out of the boundary)
 *                           2. periodic boundary condition
 *                           1. symmetirc boundary condition
 *                          -1. anti-symmetric boundary condition
 * @param invec input vector M
 * @param outvec output vector ∂^2_x M
 */
    template <class T>
    void BC_Partial_xx_1d(int boundary_condition, ublas::vector<T> &invec, ublas::vector<T> &outvec, real dx)
    {
        ublas::vector<T> v_temp(7);
        for (int j = 0; j < invec.size(); ++j)
        {
          for (int k = 0; k < 7; ++k)
          {
            v_temp[k] = Rval_1d(boundary_condition, invec, j - 3 + k);
          }
          outvec(j) = Deriv_2_8p_central(v_temp, 0, dx);
        }
    }

  /**
   * @brief Calculate ∂^2_y of a matrix ....
   *
   * @tparam T complex or real type
   * @param boundary_condition type of boundary conditions:
   *                           3. open boundary condition (value vanishes out of the boundary)
   *                           2. periodic boundary condition
   *                           1. symmetirc boundary condition
   *                          -1. anti-symmetric boundary condition
   * @param inmatrix input matrix M
   * @param outmatrix output matrix ∂^2_y M
   */
  template <class T>
  void BC_Partial_yy_2d(int boundary_condition, ublas::matrix<T> &inmatrix, ublas::matrix<T> &outmatrix, real dx)
  {
    ublas::vector<T> v_temp(7);

    for (int i = 0; i < inmatrix.size1(); ++i)
    {
      for (int j = 0; j < inmatrix.size2(); ++j)
      {
        for (int k = 0; k < 7; ++k)
        {
          v_temp[k] = Zval_2d(boundary_condition, inmatrix, i, j - 3 + k);
        }
        outmatrix(i, j) = Deriv_2_8p_central(v_temp, 0, dx);
      }
    }
  }

  template <class T>
  T Dot_Matrix(ublas::matrix<T> &a, ublas::matrix<T> &b)
  {
    T sum = 0;
    int n = a.size1();
#pragma omp parallel for reduction(+ \
                                   : sum)
    for (int i = 0; i < n; ++i)
    {
      for (int j = 0; j < n; ++j)
      {
        sum += conj(a(i, j)) * b(i, j);
      }
    }
    return sum;
  }

  static void Biconjugate(int mm, int ll, matrix_complex &vect, const std::function<void(int, int, matrix_complex &, matrix_complex &)> &mult, matrix_complex &bvect, real limit, int nmax)
  {
    matrix_complex r(mm, ll);
    matrix_complex r0(mm, ll);
    matrix_complex p(mm, ll, 0.0);
    matrix_complex v(mm, ll, 0.0), t(mm, ll), s(mm, ll), x(mm, ll);

    real e;
    mult(mm, ll, vect, r);
// r = bvect - r;
// r0 = r;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < mm; ++i)
    {
      for (int j = 0; j < ll; ++j)
      {
        r(i, j) = bvect(i, j) - r(i, j);
        r0(i, j) = r(i, j);
      }
    }

    complex alpha = 1;
    complex rhp = 1;
    complex rhn = 1;
    complex ww = 1;

    int itr = 0;

    e = std::real(Dot_Matrix(r, r) / Dot_Matrix(vect, vect));

    while (e > limit && itr < nmax)
    {
      rhn = Dot_Matrix(r0, r);
      complex beta = (rhn / rhp) * (alpha / ww);
// p = r + beta * (p - ww * v);
#pragma omp parallel for collapse(2)
      for (int i = 0; i < mm; ++i)
      {
        for (int j = 0; j < ll; ++j)
        {
          p(i, j) = r(i, j) + beta * (p(i, j) - ww * v(i, j));
        }
      }
      mult(mm, ll, p, v);
      alpha = rhn / Dot_Matrix(r0, v);
// s = r - alpha * v;
#pragma omp parallel for collapse(2)
      for (int i = 0; i < mm; ++i)
      {
        for (int j = 0; j < ll; ++j)
        {
          s(i, j) = r(i, j) - alpha * v(i, j);
        }
      }
      mult(mm, ll, s, t);
      ww = Dot_Matrix(t, s) / Dot_Matrix(t, t);
// vect += alpha * p + ww * s;
// r = s - ww * t;
#pragma omp parallel for collapse(2)
      for (int i = 0; i < mm; ++i)
      {
        for (int j = 0; j < ll; ++j)
        {
          vect(i, j) += alpha * p(i, j) + ww * s(i, j);
          r(i, j) = s(i, j) - ww * t(i, j);
        }
      }

      rhp = rhn;

      e = std::real(Dot_Matrix(r, r) / Dot_Matrix(vect, vect));
      ++itr;
    }
  }

static void Biconjugate1d(int mm, vector_complex &vect, const std::function<void(int, vector_complex &, vector_complex &)> &mult, vector_complex &bvect, real limit, int nmax)
{
    vector_complex r(mm);
    vector_complex r0(mm);
    vector_complex p(mm, 0.0);
    vector_complex v(mm, 0.0), t(mm), s(mm), x(mm);

  real e;
  mult(mm, vect, r);
// r = bvect - r;
// r0 = r;
#pragma omp parallel for
    for (int i = 0; i < mm; ++i)
    {
      r(i) = bvect(i) - r(i);
      r0(i) = r(i);
    }

  complex alpha = 1;
  complex rhp = 1;
  complex rhn = 1;
  complex ww = 1;

  int itr = 0;

  e = std::real(dotc(r, r) / dotc(vect, vect));

  while (e > limit && itr < nmax)
  {
    rhn = dotc(r0, r);
    complex beta = (rhn / rhp) * (alpha / ww);
// p = r + beta * (p - ww * v);
#pragma omp parallel for
    for (int i = 0; i < mm; ++i)
    {
        p(i) = r(i) + beta * (p(i) - ww * v(i));
    }
    mult(mm, p, v);
    alpha = rhn / dotc(r0, v);
// s = r - alpha * v;
#pragma omp parallel for
    for (int i = 0; i < mm; ++i)
    {
        s(i) = r(i) - alpha * v(i);
    }
    mult(mm, s, t);
    ww = dotc(t, s) / dotc(t, t);
// vect += alpha * p + ww * s;
// r = s - ww * t;
#pragma omp parallel for
    for (int i = 0; i < mm; ++i)
    {
        vect(i) += alpha * p(i) + ww * s(i);
        r(i) = s(i) - ww * t(i);
    }

    rhp = rhn;

    e = std::real(dotc(r, r) / dotc(vect, vect));
    ++itr;
  }
}
} // namespace td_math

#endif
