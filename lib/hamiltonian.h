#ifndef __hamiltonian_h__
#define __hamiltonian_h__
#include "common.h"
#include "grid.h"
#include "td_math.h"
class Hamiltonian : public Grid {
protected:
    int num_electrons_;
    int num_sampling_points_;
    
    real soft_coulomb_;
    real x_ion_;
    vector_real potential_real_space_;
    vector_complex kinetic_fourier_space_;
    vector_complex wave_function_;
    
    vector_real wave_function_real_;
    matrix_complex wave_function_2e_;
public:
    Hamiltonian() {
        init_1e(sys_size, sys_xmin, sys_xmax, 1);
    }
    
    void init_1e(int num_grid_points, real x_min, real x_max, int num_electrons) {
        Grid::init(num_grid_points, x_min, x_max);
        soft_coulomb_ = 1.;
        x_ion_ = -10.;
        num_electrons_ = num_electrons;
        
        potential_real_space_.resize(num_grid_points_);
        kinetic_fourier_space_.resize(num_grid_points_);
        
        num_sampling_points_ = 1;
        for (int i = 0; i < num_electrons_; ++i) {
            num_sampling_points_ *= num_grid_points_;
        }
        wave_function_.resize(num_sampling_points_);
        wave_function_real_.resize(2 * num_sampling_points_);
        
        for (int i = 0; i < num_grid_points_; ++i) {
            potential_real_space_[i] = PotentialExternal(i);
            kinetic_fourier_space_[i] = 0.5 * FourierSpace(i) * FourierSpace(i);
        }
        
        for (int i = 0; i < num_sampling_points_; ++i) {
            wave_function_[i] = complex(1., 0.);
        }
        real norm = norm_2(wave_function_);
        wave_function_ /= norm;
        
        real rsqrt_dx = boost::math::rsqrt(dx_);
        for (int i = 0; i < num_electrons_; ++i) {
            wave_function_ *= rsqrt_dx;
        }
    }
    
    void init_2e() {
        wave_function_2e_.resize(num_grid_points_, num_grid_points_);
        for (int i = 0; i < num_grid_points_; ++i) {
            for (int j = 0; j < num_grid_points_; ++j) {
                wave_function_2e_(i, j) = 1.0;
            }
        }
        
        real norm = norm_frobenius(wave_function_2e_);
        wave_function_2e_ /= norm;
        wave_function_2e_ *= 1 / dx_;
    }
    
    void SetXIon(real x_ion) { x_ion_ = x_ion; }
    
    int NumOfElectrons() { return num_electrons_; }
    
    void SetNumOfElectrons(int num_electrons) { num_electrons_ = num_electrons; }
    
    real PotentialInteraction(int i, int j) {
        real xi = Coord(i);
        real xj = Coord(j);
        return boost::math::rsqrt((xi - xj) * (xi - xj) + soft_coulomb_);
    }
    
    real PartialXjPotentialIntraction(int i, int j) {
        real xi = Coord(i);
        real xj = Coord(j);
        
        real r = boost::math::rsqrt((xi - xj) * (xi - xj) + soft_coulomb_);
        
        return (xi - xj) * r * r * r;
    }
    
    real PotentialExternal(int i) {
        real xi = Coord(i);
        return -boost::math::rsqrt((xi - x_ion_) * (xi - x_ion_) + soft_coulomb_);
    }
    
    
    vector_complex& WaveFunction() { return wave_function_; }
    matrix_complex& WaveFunction2e() {return wave_function_2e_;}
    
    void SeparateWaveFunctionReal() {
        for (int i = 0; i < wave_function_.size(); ++i) {
            wave_function_real_(i) = std::real(wave_function_(i));
            wave_function_real_(i + num_sampling_points_) =
            std::imag(wave_function_(i));
        }
    }
    
    vector_real& WaveFunctionReal() { return wave_function_real_; }
    
    vector_real& PotentialRealSpace() { return potential_real_space_; }
    
    vector_complex& KineticFourierSpace() { return kinetic_fourier_space_; }
    
    void GaussianInitialization(complex center) {
        for (int i = 0; i < num_grid_points_; ++i) {
            real xi = Coord(i);
            complex x = complex(xi, 0) - center;
            wave_function_[i] = std::exp(-x * x);
        }
        real norm = norm_2(wave_function_);
        real rsqrt_dx = boost::math::rsqrt(dx_);
        wave_function_ /= norm;
        wave_function_ *= rsqrt_dx;
    }
    
    /**
     * @brief Calculate the result of H * \Psi, the result is stored in H_psi
     *
     * @param H_psi = H*\Psi
     */
    void HamiltonianWf(matrix_complex& psi, matrix_complex& H_psi) {
        int n = num_grid_points_;
        if (BC == 3) {
            H_psi = psi;
            td_math::FourierTransformation2D(H_psi, true);
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                complex ki = kinetic_fourier_space_[i];
                for (int j = 0; j < n; ++j) {
                    complex kj = kinetic_fourier_space_[j];
                    H_psi(i, j) = 0.5 * H_psi(i, j) * (ki * ki + kj * kj);
                }
            }
            td_math::FourierTransformation2D(H_psi, false);
        }
        else {
            matrix_complex twf(n, n);
            
            td_math::BC_Partial_xx_2d(BC, psi, twf, dx_);
            td_math::BC_Partial_yy_2d(BC, psi, H_psi, dx_);
            
#pragma omp parallel for collapse(2)
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    H_psi(i, j) = -0.5 * (twf(i, j) + H_psi(i, j));
                }
            }
        }
        
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            real vi = PotentialRealSpace()[i];
            for (int j = 0; j < n; ++j) {
                real vj = PotentialRealSpace()[j];
                H_psi(i, j) += (vi + vj) * psi(i, j)
                + PotentialInteraction(i, j) * psi(i, j);
            }
        }
    }

    /**
     * @brief Calculate the result of H * \Psi in 1d system, the result is stored in H_psi
     *
     * @param H_psi = H*\Psi
     */
    void HamiltonianWf1d(vector_complex& psi, vector_complex& H_psi) {
        int n = num_grid_points_;
        if (BC == 3) {
            H_psi = psi;
            td_math::FourierTransformation(H_psi, true);
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                complex ki = kinetic_fourier_space_[i];
                H_psi(i) = 0.5 * H_psi(i) * (ki * ki);
            }
            td_math::FourierTransformation(H_psi, false);
        }
        else {
            td_math::BC_Partial_xx_1d(BC, psi, H_psi, dx_);
            
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                H_psi(i) = -0.5 * H_psi(i);
            }
        }
        
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            real vi = PotentialRealSpace()[i];
            H_psi(i) += vi * psi(i);
        }
    }
};

#endif
