#ifndef __propagation_h__
#define __propagation_h__

#ifndef NDEBUG
#define BOOST_UBLAS_NDEBUG
#endif

// Must be set if you want to use ViennaCL algorithms on ublas objects
#define VIENNACL_WITH_UBLAS 1
//
// ViennaCL includes
//
#include "hamiltonian.h"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/linalg/ilu.hpp"
// 1. Construct the left and right matrices A and B
//    wf_t: Wavefucntion at time t
//    A * wf_t+1 = B * wf_t
// 2. Calculate the rhs B * wf_t
// 3. Call BiCG-STAB to calculate wf_t+1

class Propagator : public Hamiltonian
{
protected:
  // We use five points discretization method to compute the
  // laplacian matrix. To enforce the boundary conditions, we
  // need I(i, j) to calculate the periodic indices of the
  // position (i, j)
  real_sparse_matrix identity_;
  real_sparse_matrix laplacian_;
  real_sparse_matrix potential_;

  inline int I(int i, int j)
  {
    // i = -2, -1, 0, 1, ... , num_grid_points_ - 1, num_grid_points_,
    // num_grid_points_ + 1 j = -2, -1, 0, 1, ... , num_grid_points_ - 1,
    // num_grid_points_, num_grid_points_ + 1
    i = (i < 0) ? i + num_grid_points_
                : ((i >= num_grid_points_) ? i - num_grid_points_ : i);
    j = (j < 0) ? j + num_grid_points_
                : ((j >= num_grid_points_) ? j - num_grid_points_ : j);
    int ind = i * num_grid_points_ + j;
    return ind;
  }

  // Construct the Laplacian operator. Five points method is used to compute the
  // second order derivatives
  /*
                        [-60 +16     ... +16         ...][   F_0   ]
                        |+16 -60 +16 ...     +16     ...||   F_1   |
     ∆F = 1/(12 dx^2) * |    +16 -60 +16 ...     +16 ...||   F_2   |
                        |        ...                    ||...      |
                        [                            -60][F_(n^2-1)]
  */

  void LaplacianConstructor()
  {
    laplacian_.resize(num_sampling_points_, num_sampling_points_, false);
    for (int i = 0; i < num_grid_points_; ++i)
    {
      for (int j = 0; j < num_grid_points_; ++j)
      {
        // Check eight pairs (i, j+1), (i, j-1), (i, j-2), (i, j+2)
        //                   (i-1, j), (i+1, j), (i-2, j), (i+2, j)
        int ind = I(i, j);

        laplacian_(ind, ind) = -60.;
        laplacian_(ind, I(i - 1, j)) = 16.;
        laplacian_(ind, I(i + 1, j)) = 16.;
        laplacian_(ind, I(i, j - 1)) = 16.;
        laplacian_(ind, I(i, j + 1)) = 16.;

        laplacian_(ind, I(i - 2, j)) = -1.;
        laplacian_(ind, I(i + 2, j)) = -1.;
        laplacian_(ind, I(i, j - 2)) = -1.;
        laplacian_(ind, I(i, j + 2)) = -1.;
      }
    }

    laplacian_ /= (12 * Grid::Dx() * Grid::Dx());
  }

  // Construct the potential operator. Five points method is used to compute the
  // second order derivatives
  /*
          [V(I(0,0))                                  ][   F_0   ]
          |          V(I(0,1))                        ||   F_1   |
    VF =  |                    V(I(0,2))              ||   F_2   |
          |        ...                                ||...      |
          [                             V(I(n-1, n-1))][F_(n^2-1)]
  */

  void PotentialConstructor()
  {
    potential_.resize(num_sampling_points_, num_sampling_points_, false);
    for (int i = 0; i < num_grid_points_; ++i)
    {
      for (int j = 0; j < num_grid_points_; ++j)
      {
        real V = Hamiltonian::PotentialExternal(i) +
                 Hamiltonian::PotentialExternal(j) +
                 Hamiltonian::PotentialInteraction(i, j);
        int diag_ind = I(i, j);
        potential_(diag_ind, diag_ind) = V;
      }
    }
  }

public:
  Propagator()
  {
    identity_.resize(2 * num_sampling_points_, 2 * num_sampling_points_, false);
    for (int i = 0; i < 2 * num_sampling_points_; ++i)
    {
      identity_(i, i) = 1.;
    }
    LaplacianConstructor();
    PotentialConstructor();
  }
};

//////////////////////////////////////////////////////////////////////////
// wave_function_real_ : the real and imaginary part of the wave_function_
// e.g. wave_function_ = (1+i, 2+2i, 3+3i)
//      wave_function_real_ = (1, 2, 3, 1, 2, 3)
//
// CrankNicolsonPropagator: (1 - i*dt / 2 H) \psi(t+dt) = (1 + i dt / 2 H)
// \psi(t)
//                          lhs_ = (1 - i*dt / 2 H)
//                          rhs_ = (1 + i dt / 2 H)
//
// In order to utilize the ViennaCL package to solve the linear system, we need
// to convert the complex Crank-Nicolson propagator to its real form.
//                          psi_real = [Re(psi) Im(psi)]^T
// The real form Crank-Nicolson propagator is:
//                          [1  -dt/2 H][Re(psi(t+dt))]   [1    dt/2 H][Re(psi(t))]
//                          |          ||             | = |           ||          |
//                          [dt/2 H   1][Im(psi(t+dt))]   [-dt/2 H   1][Im(psi(t))]
//////////////////////////////////////////////////////////////////////////

class CrankNicolsonPropagator : public Propagator
{
private:
  real_sparse_matrix rhs_;
  real_sparse_matrix lhs_;
  viennacl::linalg::ilut_precond<real_sparse_matrix> *ublas_ilut;

public:
  CrankNicolsonPropagator(real delta_t)
  {
    rhs_.resize(2 * num_sampling_points_, 2 * num_sampling_points_, false);
    lhs_.resize(2 * num_sampling_points_, 2 * num_sampling_points_, false);

    real_sparse_matrix H = laplacian_ + potential_;
    for (int i = 0; i < num_sampling_points_; ++i)
    {
      for (int j = 0; j < num_sampling_points_; ++j)
      {
        lhs_(i, j + num_sampling_points_) = -0.5 * delta_t * H(i, j);
        lhs_(i + num_sampling_points_, j) = 0.5 * delta_t * H(i, j);

        rhs_(i, j + num_sampling_points_) = 0.5 * delta_t * H(i, j);
        rhs_(i + num_sampling_points_, j) = -0.5 * delta_t * H(i, j);
      }
    }

    lhs_ += identity_;
    rhs_ += identity_;

    ublas_ilut = new viennacl::linalg::ilut_precond<real_sparse_matrix>(
        lhs_, viennacl::linalg::ilut_tag());
  }

  ~CrankNicolsonPropagator() { delete ublas_ilut; }

  void step()
  {
    vector_real rhs = prod(rhs_, wave_function_real_);
    wave_function_real_ = viennacl::linalg::solve(
        lhs_, rhs, viennacl::linalg::bicgstab_tag(1e-6, 20), *ublas_ilut);

    wave_function_real_ /= norm_2(wave_function_real_);
    int n = wave_function_.size();
    for(int i = 0; i < n; ++i) {
      wave_function_[i] = complex(wave_function_real_[i], wave_function_real_[i+n]);
    }
  }
};

//////////////////////////////////////////////////////////////////////////
// wave_function_real_ : the real and imaginary part of the wave_function_
// e.g. wave_function_ = (1+i, 2+2i, 3+3i)
//      wave_function_real_ = (1, 2, 3, 1, 2, 3)
//
// RungeKuttaPropagator: (1 - i*dt / 2 H - 1 / 12 dt^2 H^2) \psi(t+dt) = (1 + i*dt / 2 H - 1 / 12 dt^2 H^2) \psi(t)
//                          lhs_ = (1 - i*dt / 2 H - 1 / 12 dt^2 H^2)
//                          rhs_ = (1 + i dt / 2 H - 1 / 12 dt^2 H^2)
//
// In order to utilize the ViennaCL package to solve the linear system, we need
// to convert the complex Runge-Kutta propagator to its real form.
//                          psi_real = [Re(psi) Im(psi)]^T
// The real form Runge-Kutta propagator is:
//                          [1 - 1 / 12 dt^2 H^2  -dt/2 H][Re(psi(t+dt))]   [1 - 1 / 12 dt^2 H^2    dt/2 H][Re(psi(t))]
//                          |                            ||             | = |                             ||          |
//                          [dt/2 H   1 - 1 / 12 dt^2 H^2][Im(psi(t+dt))]   [-dt/2 H   1 - 1 / 12 dt^2 H^2][Im(psi(t))]
//////////////////////////////////////////////////////////////////////////

class RungeKuttaPropagator : public Propagator
{
private:
  real_sparse_matrix rhs_;
  real_sparse_matrix lhs_;
  viennacl::linalg::ilut_precond<real_sparse_matrix> *ublas_ilut;

public:
  RungeKuttaPropagator() {
    init(0.5 * dt);
  }

  RungeKuttaPropagator(real delta_t) {
    init(delta_t);
  }
  void init(real delta_t)
  {
    rhs_.resize(2 * num_sampling_points_,
                2 * num_sampling_points_, false);
    lhs_.resize(2 * num_sampling_points_,
                2 * num_sampling_points_, false);

    real_sparse_matrix H = laplacian_ + potential_;
    real_sparse_matrix H_squared = prod(H, H);

    for (int i = 0; i < num_sampling_points_; ++i)
    {
      for (int j = 0; j < num_sampling_points_; ++j)
      {
        // if (i == j)
        lhs_(i, j) =  - 1 / 12 * delta_t * delta_t * H_squared(i, j);
        lhs_(i + num_sampling_points_, j + num_sampling_points_) =
           - 1 / 12 * delta_t * delta_t * H_squared(i, j);

        lhs_(i, j + num_sampling_points_) = -0.5 * delta_t * H(i, j);
        lhs_(i + num_sampling_points_, j) = 0.5 * delta_t * H(i, j);

        rhs_(i, j + num_sampling_points_) = 0.5 * delta_t * H(i, j);
        rhs_(i + num_sampling_points_, j) = -0.5 * delta_t * H(i, j);

        rhs_(i, j) =  - 1 / 12 * delta_t * delta_t * H_squared(i, j);
        rhs_(i + num_sampling_points_, j + num_sampling_points_) = 
         - 1 / 12 * delta_t * delta_t * H_squared(i, j);
      }
    }

    lhs_ += identity_;
    rhs_ += identity_;

    // std::cout << lhs_ << std::endl;
    ublas_ilut = new viennacl::linalg::ilut_precond<real_sparse_matrix>(
        lhs_, viennacl::linalg::ilut_tag());
  }

  ~RungeKuttaPropagator() { delete ublas_ilut; }

  void step()
  {
    vector_real rhs = prod(rhs_, wave_function_real_);
    // std::cout << rhs << std::endl;

    wave_function_real_ = viennacl::linalg::solve(
        lhs_, rhs, viennacl::linalg::bicgstab_tag(1e-6, 20), *ublas_ilut);

    wave_function_real_ /= norm_2(wave_function_real_);
    int n = wave_function_.size();
    for(int i = 0; i < n; ++i) {
      wave_function_[i] = complex(wave_function_real_[i], wave_function_real_[i+n]);
    }
  }
};

#endif