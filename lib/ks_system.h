#ifndef __ks_system__
#define __ks_system__

#include "propagation.h"

class KSSystem : public RungeKuttaPropagator, public CrankNicolsonPropagator {
 public:
  KSSystem() {
    num_orbitals_ = 2;

    ks_potential_.resize(num_grid_points_);
    ks_hartree_fock_potential_.resize(num_grid_points_);

    ks_density_.resize(num_grid_points_);
    ks_density_current_.resize(num_grid_points_);
    ks_density_matrix_.resize(num_grid_points_, num_grid_points_);

    for (int i = 0; i < num_orbitals_; ++i) {
      vector_complex orb(num_grid_points_);
      orbitals_.emplace_back(orb);
      occupation_.push_back(1);
    }
  }

  void Density() {
    for (int i = 0; i < num_orbitals_; ++i) {
      for (int j = 0; j < num_grid_points_; ++j) {
        ks_density_[j] +=
            occupation_[i] * std::real(orbitals_[i][j] * conj(orbitals_[i][j]));
      }
    }
  }

  void Current() {
    for (int i = 0; i < num_orbitals_; ++i) {
      vector_complex deriv_t_orb(num_grid_points_);
      td_math::Gradient_1d_7p(orbitals_[i], deriv_t_orb, Dx());
      for (int j = 0; j < num_grid_points_; ++j) {
        ks_density_current_[j] +=
            occupation_[i] * std::imag(conj(orbitals_[i][j]) * deriv_t_orb);
      }
    }
  }

  void DensityMatrix() {
    ublas::hermitian_adaptor << matrix_complex >,
        ublas::upper > hau(ks_density_matrix_);

    for (int i = 0; i < num_grid_points_; ++i) {
      for (int k = 0; k < num_orbitals_; ++k) {
        hau(i, i) += occupation_[k] * orbitals_[k][i] * conj(orbitals_[k][i]);
        for (int j = i + 1; j < num_grid_points_; ++j) {
          hau(i, j) += occupation_[k] * orbitals_[k][i] * conj(orbitals_[k][j]);
        }
      }
    }
  }

  void DensityMatrixXCH() {
    ublas::hermitian_adaptor << matrix_complex >,
        ublas::upper > hau(ks_density_matrix_xch_);

    for (int i = 0; i < num_grid_points_; ++i) {
      for (int j = i + 1; j < num_grid_points_; ++j) {
        real dens_2rdm_abs = abs(
            orbitals_[0][j] * orbitals_[0][j]);  // This line should be replaced
                                                 // for more general calulation
        hau(i, j) =
            2 * dens_2rdm_abs * dens_2rdm_abs / (ks_density_[j] + xsamll) -
            ks_density_[i];
      }
    }
  }

  void CalculateHartreeFockPotential() {
    for (int i = 0; i < num_grid_points_; ++i) {
      for (int j = 0; j < num_grid_points_; ++j) {
        ks_hartree_fock_potential_[i] +=
            PotentialInteraction(i, j) * ks_density_[j] * Dx();
      }
    }
  }


  void CalculatePotentialWXC() {
    int dx_square = Dx() * Dx();
    matrix_complex ks_density_matrix_xch_trans =
        ublas::transpose(rf_density_matrix_xch_);
    for (int i = 0; i < num_grid_points_; ++i) {
      rf_wxc_potential_(i) = rf_wxc_potential_(std::max(i - 1, 0));
      for (int j = 0; j < num_grid_points_; ++j) {
        rf_wxc_potential_(i) += 0.5 * ks_density_matrix_xch_trans(i, j) *
                                PartialXjPotentialIntraction(j, i) * dx_square;
        if (i > 0) {
          rf_wxc_potential_(i) += 0.5 * ks_density_matrix_xch_trans(i - 1, j) *
                                  PartialXjPotentialIntraction(j, i - 1) *
                                  dx_square;
        }
      }
    }
  }

 private:
  int num_orbitals_;
  std::vector<vector_complex> orbitals_;
  std::vector<int> occupation_;

  vector_real ks_potential_;
  vector_real ks_density_;
  vector_complex ks_density_current_;

  vector_real ks_hartree_fock_potential_;

  matrix_complex ks_density_matrix_;
  matrix_complex ks_density_matrix_xch_;

 public:
  vector_real& KSDensity() { return ks_density_; }

  vector_complex& KSDensityCurrent() { return ks_density_current_; }

  vector_real& KSHartreeFockPotential() { return ks_hartree_fock_potential_; }

  matrix_complex& KSDensityMatrix() { return ks_density_matrix_; }

  matrix_complex& KSDensityMatrixXCH() { return ks_density_matrix_xch_; }
};

#endif