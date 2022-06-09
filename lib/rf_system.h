#ifndef __rf_system__
#define __rf_system__

#include "propagation.h"

class RfSystem : public RungeKuttaPropagator, public CrankNicolsonPropagator {
 public:
  RfSystem() {
    rf_density_.resize(num_grid_points_);
    rf_hartree_fock_potential_.resize(num_grid_points_);
    rf_density_current_.resize(num_grid_points_);
    wave_function_2d_.resize(num_grid_points_, num_grid_points_);
    rf_density_matrix_.resize(num_grid_points_, num_grid_points_);
    rf_density_matrix_xch_.resize(num_grid_points_, num_grid_points_);
  }

  void CalculateDensity() {
    for (int j = 0; j < num_grid_points_; ++j) {
      for (int k = 0; k < num_grid_points_; ++k) {
        rf_density_[j] += 2 * std::real(wave_function_2d_(j, k) *
                                        conj(wave_function_2d_(j, k)));
      }
    }
  }

  void CalculateCurrent() {
    matrix_complex partial_x_wf(num_grid_points_, num_grid_points_);
    td_math::Partial_x_2d(wave_function, partial_x_wf, Dx());
    for (int i = 0; i < num_grid_points_; ++i) {
      for (int j = 0; j < num_grid_points_; ++j) {
        rf_density_current_[i] +=
            std::imag(conj(wave_function(i, j)) * partial_x_wf(i, j)) * Dx();
      }
    }
  }

  void CalculateDensityMatrix() {
    ublas::hermitian_adaptor<matrix_complex,
        ublas::upper > hau(rf_density_matrix_);

    for (int i = 0; i < num_grid_points_; ++i) {
      for (int j = i + 1; j < num_grid_points_; ++j) {
        for (int k = 0; k < num_grid_points_; ++k) {
          hau(i, j) += num_electrons_ * wave_function_2d_(i, k) *
                       conj(wave_function_2d_(j, k)) * Dx();
        }
      }
    }
  }

  void CalculateDensityMatrixXCH() {
    ublas::hermitian_adaptor <matrix_complex,
        ublas::upper > hau(rf_density_matrix_xch_);

    for (int i = 0; i < num_grid_points_; ++i) {
      for (int j = i + 1; j < num_grid_points_; ++j) {
        complex psi = wave_function_2d_(i, j);
        hau(i, j) =
            2 * psi * conj(psi) / (rf_density_[j] + xsmall) - rf_density_[i];
      }
    }
  }

  void CalculateHartreeFockPotential() {
    for (int i = 0; i < num_grid_points_; ++i) {
      for (int j = 0; j < num_grid_points_; ++j) {
        rf_hartree_fock_potential_[i] +=
            PotentialInteraction(i, j) * rf_density_[j] * Dx();
      }
    }
  }

  void CalculatePotentialWXC() {
    real dx_square = Dx() * Dx();
    matrix_complex rf_density_matrix_xch_trans =
        ublas::transpose(rf_density_matrix_xch_);
    for (int i = 0; i < num_grid_points_; ++i) {
      rf_wxc_potential_(i) = rf_wxc_potential_(std::max(i - 1, 0));
      for (int j = 0; j < num_grid_points_; ++j) {
        rf_wxc_potential_(i) += 0.5 * rf_density_matrix_xch_trans(i, j) *
                                PartialXjPotentialIntraction(j, i) * dx_square;
        if (i > 0) {
          rf_wxc_potential_(i) += 0.5 * rf_density_matrix_xch_trans(i - 1, j) *
                                  PartialXjPotentialIntraction(j, i - 1) *
                                  dx_square;
        }
      }
    }
  }

  void CalculateHartreeXC() {
    for (int i = 0; i < num_grid_points_; ++i) {
      for (int j = 0; j < num_grid_points_; ++j) {
        rf_hartree_fock_potential_xc_[i] +=
            PotentialInteraction(i, j) * rf_density_matrix_xch_*(i, j) * Dx();
      }
    }
  }

 private:
  vector_real rf_density_;
  vector_complex rf_density_current_;

  vector_real rf_hartree_fock_potential_;
  vector_real rf_wxc_potential_;
  vector_real rf_hartree_fock_potential_xc_;

  matrix_complex wave_function_2d_;
  matrix_complex rf_density_matrix_;
  matrix_complex rf_density_matrix_xch_;

 public:
  vector_real& RfDensity() { return rf_density_; }

  vector_complex& RfDensityCurrent() { return rf_density_current_; }

  vector_real& RfHartreeFockPotential() { return rf_hartree_fock_potential_; }

  matrix_complex& RfDensityMatrix() { return rf_density_matrix_; }

  matrix_complex& RfDensityMatrixXCH() { return rf_density_matrix_xch_; }
};

#endif