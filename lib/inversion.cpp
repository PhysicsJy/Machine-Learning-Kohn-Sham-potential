#include "inversion.h"
/**
 * @brief KS potential inversion
 * @ref Density-Matrix Coupled Time-Dependent Exchange-Correlation Functional Approximations
 * @cite J. Chem. Theory Comput. 2019, 15, 1672−1678
 */

/**
 * @brief System initialization
 *
 */
void Inversion::InitRfSystem()
{
    rf_density_.resize(num_grid_points_);
    rf_hartree_fock_potential_.resize(num_grid_points_);
    rf_density_current_.resize(num_grid_points_);
    // wave_function_2e_.resize(num_grid_points_, num_grid_points_);
    rf_density_matrix_.resize(num_grid_points_, num_grid_points_);
    rf_density_matrix_xch_.resize(num_grid_points_, num_grid_points_);
}

/**
 * @brief Caculate the exact density in the system
 *
 */
void Inversion::CalculateRfDensity()
{
    for (int j = 0; j < num_grid_points_; ++j)
    {
        rf_density_[j] = 0.0;
        for (int k = 0; k < num_grid_points_; ++k)
        {
            rf_density_[j] +=
                2.0 * std::real(wave_function_2e_(j, k) * conj(wave_function_2e_(j, k))) * Dx();
        }
    }
}

/**
 * @brief Caculate the exact density current in the system
 *
 */
void Inversion::CalculateRfCurrent()
{
    matrix_complex partial_x_wf(num_grid_points_, num_grid_points_);
    td_math::Partial_x_2d(wave_function_2e_, partial_x_wf, Dx());

    for (int i = 0; i < num_grid_points_; ++i)
    {
        complex sum = 0;
#pragma omp parallel for reduction(+ \
                                   : sum)
        for (int j = 0; j < num_grid_points_; ++j)
        {
            sum +=
                std::imag(conj(wave_function_2e_(i, j)) * partial_x_wf(i, j)) * Dx();
        }

        rf_density_current_[i] = 2.0 * sum;
    }
}

/**
 * @brief Caculate the exact density matrix (1rdm) in the system
 *
 */
void Inversion::CalculateRfDensityMatrix()
{
    ublas::hermitian_adaptor<matrix_complex, ublas::upper> hau(rf_density_matrix_);

    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = i + 1; j < num_grid_points_; ++j)
        {
            for (int k = 0; k < num_grid_points_; ++k)
            {
                hau(i, j) += (real)num_electrons_ * wave_function_2e_(i, k) *
                             conj(wave_function_2e_(j, k)) * Dx();
            }
        }
    }
}

/**
 * @brief Caculate \rho_xch
 *
 */
void Inversion::CalculateRfDensityMatrixXCH()
{
    ublas::hermitian_adaptor<matrix_complex, ublas::upper> hau(rf_density_matrix_xch_);

    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = i + 1; j < num_grid_points_; ++j)
        {
            complex psi = wave_function_2e_(i, j);
            hau(i, j) = 2. * psi * conj(psi) / (rf_density_[j] + xsmall) - rf_density_[i];
        }
    }
}

void Inversion::CalculateRfHartreeFockPotential()
{
    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            rf_hartree_fock_potential_[i] += PotentialInteraction(i, j) * rf_density_[j] * Dx();
        }
    }
}

void Inversion::CalculateRfHartreeXC()
{
    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            rf_hartree_fock_potential_xc_[i] +=
                std::real(PotentialInteraction(i, j) * rf_density_matrix_xch_(i, j) * Dx());
        }
    }
}

void Inversion::InitKSSystem()
{
    num_orbitals_ = 2;

    ks_potential_.resize(num_grid_points_);
    ks_potential_xc_.resize(num_grid_points_);
    ks_hartree_fock_potential_.resize(num_grid_points_);

    ks_density_.resize(num_grid_points_);
    ks_density_current_.resize(num_grid_points_);
    ks_density_matrix_.resize(num_grid_points_, num_grid_points_);
    ks_density_matrix_xch_.resize(num_grid_points_, num_grid_points_);
    
    orbitals_.resize(2);
    occupation_.resize(2);
    for (int i = 0; i < num_orbitals_; ++i)
    {
        vector_complex orb(num_grid_points_);
        orbitals_[i] = orb;
        occupation_[i] = 1;
    }
}

void Inversion::CalculateKSDensity()
{
    for (int i = 0; i < num_orbitals_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            ks_density_[j] += occupation_[i] * std::real(orbitals_[i][j] * conj(orbitals_[i][j]));
        }
    }
}

void Inversion::CalculateKSCurrent()
{
    for (int i = 0; i < num_orbitals_; ++i)
    {
        vector_complex deriv_t_orb(num_grid_points_);
        td_math::Gradient_1d_7p(orbitals_[i], deriv_t_orb, Dx());
        for (int j = 0; j < num_grid_points_; ++j)
        {
            ks_density_current_[j] +=
                occupation_[i] * std::imag(conj(orbitals_[i][j]) * deriv_t_orb[j]);
        }
    }
}

void Inversion::CalculateKSDensityMatrix()
{
    ublas::hermitian_adaptor<matrix_complex, ublas::upper> hau(ks_density_matrix_);

    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int k = 0; k < num_orbitals_; ++k)
        {
            hau(i, i) += occupation_[k] * orbitals_[k][i] * conj(orbitals_[k][i]);
            for (int j = i + 1; j < num_grid_points_; ++j)
            {
                hau(i, j) += occupation_[k] * orbitals_[k][i] * conj(orbitals_[k][j]);
            }
        }
    }
}

void Inversion::CalculateKSDensityMatrixXCH()
{
//    ublas::hermitian_adaptor<matrix_complex, ublas::upper> hau(ks_density_matrix_xch_);

    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            real dens_2rdm_abs = std::real(orbitals_[0][j] * conj(orbitals_[0][j])); // This line should
            // be replaced for
            // more general
            // calulation
            ks_density_matrix_xch_(i, j) =
                2 * dens_2rdm_abs * dens_2rdm_abs / (ks_density_[j] + xsmall) - ks_density_[i];
        }
    }
}

void Inversion::CalculateKSHartreeFockPotential()
{
    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            ks_hartree_fock_potential_[i] += PotentialInteraction(i, j) * ks_density_[j] * Dx();
        }
    }
}

void Inversion::CalculateKSPotentialWXC()
{
    real dx_square = Dx() * Dx();
    matrix_complex ks_density_matrix_xch_trans = ublas::trans(rf_density_matrix_xch_);
    for (int i = 0; i < num_grid_points_; ++i)
    {
        potential_xc_interaction_component_[i] =
            potential_xc_interaction_component_[std::max(i - 1, 0)];
        for (int j = 0; j < num_grid_points_; ++j)
        {
            potential_xc_interaction_component_[i] +=
                std::real(0.5 * ks_density_matrix_xch_trans(i, j) *
                          PartialXjPotentialIntraction(j, i) * dx_square);
            if (i > 0)
            {
                potential_xc_interaction_component_[i] +=
                    std::real(0.5 * ks_density_matrix_xch_trans(i - 1, j) *
                              PartialXjPotentialIntraction(j, i - 1) * dx_square);
            }
        }
    }
}

void Inversion::InitMixedSystem()
{
    potential_xc_kinetic_component_.resize(num_grid_points_);
    potential_xc_interaction_component_.resize(num_grid_points_);
    potential_hxc_.resize(num_grid_points_);
    deriv_potential_xc_interaction_component_.resize(num_grid_points_);
    potential_xc_kinetic_component_rf_.resize(num_grid_points_);
    potential_xc_kinetic_component_ks_.resize(num_grid_points_);
    vs_it_.resize(num_grid_points_);
}

void Inversion::CalculatePotentialWXC()
{
    real dx_square = Dx() * Dx();
    matrix_complex rf_density_matrix_xch_trans = ublas::trans(rf_density_matrix_xch_);
    for (int i = 0; i < num_grid_points_; ++i)
    {
        potential_xc_interaction_component_[i] =
            potential_xc_interaction_component_[std::max(i - 1, 0)];
        for (int j = 0; j < num_grid_points_; ++j)
        {
            potential_xc_interaction_component_[i] +=
                std::real(0.5 * rf_density_matrix_xch_trans(i, j) *
                          PartialXjPotentialIntraction(j, i) * dx_square);
            if (i > 0)
            {
                potential_xc_interaction_component_[i] +=
                    std::real(0.5 * rf_density_matrix_xch_trans(i - 1, j) *
                              PartialXjPotentialIntraction(j, i - 1) * dx_square);
            }
        }
    }
}

void Inversion::InitMidRef()
{
    real cc = 0.0;
    for (int i = 0; i < num_grid_points_; ++i)
    {
        if (abs(PotentialExternal(i)) > cc)
        {
            cc = abs(PotentialExternal(i));
            midref = i;
        }
    }
}

void Inversion::CalculatePotentialTC()
{
    matrix_complex D_dmat = rf_density_matrix_ - ks_density_matrix_;

    matrix_complex D_dmat_x_temp(D_dmat.size1(), D_dmat.size2());
    matrix_complex D_dmat_y_temp(D_dmat.size1(), D_dmat.size2());

    td_math::Partial_xx_2d(D_dmat, D_dmat_x_temp, Dx());
    td_math::Partial_yy_2d(D_dmat, D_dmat_y_temp, Dx());

    D_dmat = D_dmat_x_temp - D_dmat_y_temp;

    td_math::Partial_x_2d(D_dmat, D_dmat_x_temp, Dx());
    td_math::Partial_y_2d(D_dmat, D_dmat_y_temp, Dx());

    D_dmat = -D_dmat_x_temp + D_dmat_y_temp;

    vector_real force_kinetic_component(num_grid_points_);

    for (int i = 0; i < num_grid_points_; ++i)
    {
        real f = std::real(D_dmat(i, i));
        // smooth
        // the force
        f = 0.5 * (1.0 + tanh((f - lmbd) / (0.001 * lmbd))) * f;
        force_kinetic_component[i] = 1.0 / (4.0 * RfDensity()[i] + xsmall) * f;
    }

    td_math::Antideriv_r(3, force_kinetic_component, potential_xc_kinetic_component_, Dx());

    for (int i = 0; i < num_grid_points_; ++i)
    {
        potential_xc_kinetic_component_[i] -= potential_xc_kinetic_component_[midref];
    }
}

void Inversion::CalculateKSPotentialTC()
{
    matrix_complex D_dmat = -ks_density_matrix_;

    matrix_complex D_dmat_x_temp(D_dmat.size1(), D_dmat.size2());
    matrix_complex D_dmat_y_temp(D_dmat.size1(), D_dmat.size2());

    td_math::Partial_xx_2d(D_dmat, D_dmat_x_temp, Dx());
    td_math::Partial_yy_2d(D_dmat, D_dmat_y_temp, Dx());

    D_dmat = D_dmat_x_temp - D_dmat_y_temp;

    td_math::Partial_x_2d(D_dmat, D_dmat_x_temp, Dx());
    td_math::Partial_y_2d(D_dmat, D_dmat_y_temp, Dx());

    D_dmat = -D_dmat_x_temp + D_dmat_y_temp;

    vector_real force_kinetic_component(num_grid_points_);

    // modify the
    // loop index
    for (int i = 0; i < num_grid_points_; ++i)
    {
        real f = std::real(D_dmat(i, i));
        // smooth
        // the force
        f = 0.5 * (1.0 + tanh((f - lmbd) / (0.001 * lmbd))) * f;
        force_kinetic_component[i] = 1.0 / (4.0 * RfDensity()[i] + xsmall) * f;
    }

    td_math::Antideriv_r(3, force_kinetic_component, potential_xc_kinetic_component_ks_, Dx());

    for (int i = 0; i < num_grid_points_; ++i)
    {
        potential_xc_kinetic_component_ks_[i] -= potential_xc_kinetic_component_ks_[midref];
    }
}

/**
 * @brief Caculate V^T_C
 *
 */
void Inversion::CalculateRfPotentialTC()
{
    matrix_complex D_dmat = rf_density_matrix_;

    matrix_complex D_dmat_x_temp(D_dmat.size1(), D_dmat.size2());
    matrix_complex D_dmat_y_temp(D_dmat.size1(), D_dmat.size2());

    td_math::Partial_xx_2d(D_dmat, D_dmat_x_temp, Dx());
    td_math::Partial_yy_2d(D_dmat, D_dmat_y_temp, Dx());

    D_dmat = D_dmat_x_temp - D_dmat_y_temp;

    td_math::Partial_x_2d(D_dmat, D_dmat_x_temp, Dx());
    td_math::Partial_y_2d(D_dmat, D_dmat_y_temp, Dx());

    D_dmat = -D_dmat_x_temp + D_dmat_y_temp;

    vector_real force_kinetic_component(num_grid_points_);

    // need modify
    // the loop
    // index
    for (int i = 0; i < num_grid_points_; ++i)
    {
        real f = std::real(D_dmat(i, i));
        // smooth
        // the force
        f = 0.5 * (1.0 + tanh((f - lmbd) / (0.001 * lmbd))) * f;
        force_kinetic_component[i] = 1.0 / (4.0 * RfDensity()[i] + xsmall) * f;
    }

    td_math::Antideriv_r(3, force_kinetic_component, potential_xc_kinetic_component_rf_, Dx());

    for (int i = 0; i < num_grid_points_; ++i)
    {
        potential_xc_kinetic_component_rf_[i] -= potential_xc_kinetic_component_rf_[midref];
    }
}

/**
 * @brief Calculate V_Hxc using the n_xch
 *
 * Results are stored in @param potential_hxc_
 */

void Inversion::CalculatePotentialHXC()
{
    for (int i = 0; i < num_grid_points_; ++i)
    {
        potential_hxc_[i] = 0.0;
    }

    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            potential_hxc_[j] += std::real(rf_density_matrix_xch_(i, j) * PotentialInteraction(i, j) * Dx());
        }
    }
}

/**
 * @brief Calculate the derivative of V^W_XC
 * results are stored in @param deriv_potential_xc_interaction_component_
 *
 */

void Inversion::CalculateDerivPotentialWXC()
{
    // set all elements of deriv_potential_xc_interaction_component_ to zero

    deriv_potential_xc_interaction_component_ = vector_real(num_grid_points_, 0.0);

    matrix_complex dnxch_dj = rf_density_matrix_xch_;

    td_math::FourierTransformation2D(dnxch_dj, true);

    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            dnxch_dj(i, j) *= complex(0.0, fourier_space_[j]);
        }
    }

    td_math::FourierTransformation2D(dnxch_dj, false);
    real dx_square = Dx() * Dx();
    for (int j = 0; j < num_grid_points_; ++j)
    {
        deriv_potential_xc_interaction_component_[0] -= std::real(dnxch_dj(j, 0) * PotentialInteraction(j, 0) * dx_square);
    }

    for (int j = 0; j < num_grid_points_; ++j)
    {
        for (int i = 1; i < num_grid_points_; ++i)
        {
            deriv_potential_xc_interaction_component_[i] -= std::real(dnxch_dj(j, i) * PotentialInteraction(j, i) * dx_square);
        }
    }
}

/**
 * @brief Calibrate the reference point
 *
 */
void Inversion::Calibrate()
{
    for (int i = 0; i < num_grid_points_; ++i)
    {
        ks_potential_[i] = potential_real_space_[i];
    }

    ks_potential_ += rf_hartree_fock_potential_ + potential_xc_interaction_component_ + potential_xc_kinetic_component_;
    potential_xc_kinetic_component_ = /*ks_hartree_fock_potential_ +*/ potential_xc_kinetic_component_ + potential_xc_interaction_component_;

    real cc = ks_potential_[midref];

    for (auto &vks : ks_potential_)
    {
        vks -= cc;
    }

    cc = rf_hartree_fock_potential_[midref];

    for (auto &vhf : rf_hartree_fock_potential_)
    {
        vhf -= cc;
    }

    cc = potential_xc_interaction_component_[midref];

    for (auto &vwxc : potential_xc_interaction_component_)
    {
        vwxc -= cc;
    }

    cc = potential_hxc_[midref];

    for (auto &vhxc : potential_hxc_)
    {
        vhxc -= cc;
    }

    cc = deriv_potential_xc_interaction_component_[midref];

    for (auto &dvwxc : deriv_potential_xc_interaction_component_)
    {
        dvwxc -= cc;
    }

    cc = potential_xc_kinetic_component_[midref];

    for (auto &vkinc : potential_xc_kinetic_component_)
    {
        vkinc -= cc;
    }

    cc = potential_xc_kinetic_component_rf_[midref];

    for (auto &vkincrf : potential_xc_interaction_component_)
    {
        vkincrf -= cc;
    }

    cc = potential_xc_kinetic_component_ks_[midref];

    for (auto &vkincks : potential_xc_kinetic_component_ks_)
    {
        vkincks -= cc;
    }

    cc = vs_it_[midref];

    for (auto &vsit : vs_it_)
    {
        vsit -= cc;
    }
}

/**
 * @brief Caclulate the matrix element of 2rdm at (i, j, l, m)
 *
 * @param i
 * @param j
 * @param l
 * @param m
 * @return complex
 */
complex Inversion::RDM2(int i, int j, int l, int m)
{
    return 2.0 * conj(wave_function_2e_(l, m)) * wave_function_2e_(i, j);
}

/**
 * @brief Caculate the integral of 2rdm and interaction potential
 *
 * @param dmat2int output matrix of the integration
 */
void Inversion::CalculateRDM2Int(matrix_complex &dmat2int)
{
    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            dmat2int(i, j) = complex(0.0, 0.0);
            for (int l = 0; l < num_grid_points_; ++l)
            {
                dmat2int(i, j) += (PotentialInteraction(i, l) - PotentialInteraction(j, l)) * RDM2(i, l, j, l) * Dx();
            }
        }
    }
}

/**
 * @brief Calculate the Liouvil term in the expression of 1rdm's time derivative
 *
 * @param v external potential
 * @param ldmat Liouvil term
 */
void Inversion::Liouvil(vector_real &v, matrix_complex &ldmat)
{
    matrix_complex D_dmat = ldmat;

    matrix_complex D_dmat_x_temp(D_dmat.size1(), D_dmat.size2());
    matrix_complex D_dmat_y_temp(D_dmat.size1(), D_dmat.size2());

    td_math::BC_Partial_xx_2d(BC, D_dmat, D_dmat_x_temp, Dx());
    td_math::BC_Partial_yy_2d(BC, D_dmat, D_dmat_y_temp, Dx());

    D_dmat = -0.5 * (D_dmat_x_temp - D_dmat_y_temp);

    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            ldmat(i, j) = D_dmat(i, j) + (v[i] - v[j]) * ldmat(i, j);
        }
    }

    ldmat = 0.5 * (ldmat - conj(ublas::trans(ldmat)));
}

/**
 * @brief Calculate the time derivative of 1rdm
 *
 * @param v external potential
 * @param dmat2int the integration of 2rdm and interaction potential
 * @param dmat 1rdm
 * @param output_mat time derivative of 1rdm
 */
void Inversion::DerivDensityMatrix(vector_real &v, matrix_complex &dmat2int, matrix_complex &dmat, matrix_complex &output_mat)
{
    matrix_complex ldmat = dmat;

    Liouvil(v, ldmat);

    output_mat = -complex(0.0, 1.0) * (ldmat + dmat2int);
}

/**
 * @brief Caculate V^W_XCH from KS orbitals
 *
 * @param psi KS orbitals
 * @param vwxch
 */
void Inversion::WaveFunctionToVWXCH(matrix_complex &psi, vector_real &vwxch)
{
    vector_real narr(num_grid_points_);
    matrix_complex paird(num_grid_points_, num_grid_points_);
    WaveFunctionToDens(psi, narr);
    WaveFunctionToPaird(psi, paird);
    CaculateVWXCH(narr, paird, vwxch);
}

/**
 * @brief Helper function for @ref WaveFunctionToVWXCH
 *
 * @param psi
 * @param dens
 */
void Inversion::WaveFunctionToDens(matrix_complex &psi, vector_real &dens)
{
    dens = vector_real(num_grid_points_, 0);
    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int l = 0; l < num_grid_points_; ++l)
        {
            dens[i] += 2.0 * std::real(psi(i, l) * conj(psi(i, l))) * Dx();
        }
    }
}

/**
 * @brief Helper function for @ref WaveFunctionToVWXCH
 *
 * @param psi
 * @param dens
 */
void Inversion::WaveFunctionToPaird(matrix_complex &psi, matrix_complex &paird)
{
    paird = matrix_complex(num_grid_points_, num_grid_points_, complex(0.0, 0.0));
    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            paird(i, j) = 2.0 * conj(psi(j, i)) * psi(i, j);
        }
    }
}

/**
 * @brief Helper function for @ref WaveFunctionToVWXCH
 *
 * @param psi
 * @param dens
 */
void Inversion::CaculateVWXCH(vector_real &dens, matrix_complex &paird, vector_real &vwxch)
{
    vwxch = vector_real(num_grid_points_, 0.0);
    vector_real tvwxch = vector_real(num_grid_points_, 0.0);
    matrix_complex tmat = matrix_complex(num_grid_points_, num_grid_points_, complex(0.0, 0.0));

    for (int j = 0; j < num_grid_points_; ++j)
    {
        for (int i = 0; i < num_grid_points_; ++i)
        {
            tmat(i, j) = PartialXjPotentialIntraction(i, j) * paird(j, i);
        }
    }

    for (int j = 0; j < num_grid_points_; ++j)
    {
        for (int i = 0; i < num_grid_points_; ++i)
        {
            tvwxch(i) += std::real(tmat(j, i)) * Dx();
        }
    }

    for (int i = 0; i < num_grid_points_; ++i)
    {
        tvwxch(i) /= (dens(i) + xsmall);
    }

    td_math::Antideriv_r(BC, tvwxch, vwxch, Dx());
}

/**
 * @brief Find the Derivative of density matrix in the KS system
 *
 * @param fdmat
 * @param dmat_ks
 * @param fdmat_ks
 * @param vin
 * @param vs
 */
void Inversion::DerivDensityMatrixKS(matrix_complex &fdmat, matrix_complex &dmat_ks, matrix_complex &fdmat_ks, vector_real &vin, vector_real &vs)
{
    matrix_complex tddmat(num_grid_points_, num_grid_points_);
    matrix_complex kddmat(num_grid_points_, num_grid_points_);
    vector_real divjdot(num_grid_points_);
    DerivDensityMatrixKin(dmat_ks, fdmat_ks);
    DerivDensityMatrixV(vin, dmat_ks, tddmat);
    fdmat_ks += tddmat;
    tddmat = fdmat - fdmat_ks;
    CalculateFKinDiag(tddmat, divjdot);
    CalculateKSPotential(divjdot, dmat_ks, vs, xpes, 1.e-15);
    DerivDensityMatrixV(vs, dmat_ks, tddmat);
    vs += vin;
    fdmat_ks += tddmat;
}

void Inversion::DerivDensityMatrixKSRef(vector_real &ref, matrix_complex &fdmat, matrix_complex &dmat_ks, matrix_complex &fdmat_ks, vector_real &vin, vector_real &vs)
{
    matrix_complex tddmat(num_grid_points_, num_grid_points_);
    matrix_complex kddmat(num_grid_points_, num_grid_points_);
    vector_real divjdot(num_grid_points_);
    DerivDensityMatrixKin(dmat_ks, fdmat_ks);
    DerivDensityMatrixV(vin, dmat_ks, tddmat);
    fdmat_ks += tddmat;
    tddmat = fdmat - fdmat_ks;
    CalculateFKinDiag(tddmat, divjdot);
    divjdot += ref;
    CalculateKSPotential(divjdot, dmat_ks, vs, xpes, 1.e-15);
    DerivDensityMatrixV(vs, dmat_ks, tddmat);
    vs += vin;
    fdmat_ks += tddmat;
}

void Inversion::DerivDensityMatrixKin(matrix_complex &dmat, matrix_complex &fdmat)
{
    fdmat = dmat;
    matrix_complex tdmat(num_grid_points_, num_grid_points_);

    td_math::BC_Partial_xx_2d(BC, dmat, fdmat, Dx());
    td_math::BC_Partial_yy_2d(BC, dmat, tdmat, Dx());

    fdmat = -0.5 * (fdmat - tdmat);

    tdmat = fdmat;

    fdmat = -complex(0.0, 0.5) * (tdmat - conj(ublas::trans(tdmat)));
}

void Inversion::DerivDensityMatrixV(vector_real &vv, matrix_complex &dmat, matrix_complex &fdmat)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            fdmat(i, j) = -complex(0.0, 1.0) * (vv(i) - vv(j)) * dmat(i, j);
        }
    }
}

void Inversion::CalculateFKinDiag(matrix_complex &dmat, vector_real &fkindiag)
{
    matrix_complex fdmat(num_grid_points_, num_grid_points_);

    DerivDensityMatrixKin(dmat, fdmat);
#pragma omp parallel for
    for (int i = 0; i < num_grid_points_; ++i)
    {
        fkindiag(i) = std::real(fdmat(i, i));
    }
}

void Inversion::CalculateKSPotential(vector_real &divjdot, matrix_complex &dmatks, vector_real &vs, real eps, real rcond)
{
    matrix_real vkinmat(num_grid_points_, num_grid_points_);
    matrix_real amat(2 * num_grid_points_, num_grid_points_, 0.0);
    vector_real bvect(2 * num_grid_points_);
    CalculateVKinMat2(dmatks, vkinmat);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            amat(i, j) = vkinmat(i, j);
        }
    }
#pragma omp parallel for
    for (int i = 0; i < num_grid_points_ - 1; ++i)
    {
        amat(num_grid_points_ + i, i) = -eps / Dx();
        amat(num_grid_points_ + i, i + 1) = eps / Dx();
    }

    amat(2 * num_grid_points_ - 1, num_grid_points_ - 2) = eps / Dx();
    amat(2 * num_grid_points_ - 1, num_grid_points_ - 1) = -eps / Dx();

#pragma omp parallel for
    for (int i = 0; i < num_grid_points_; ++i)
    {
        bvect[i] = divjdot[i];
        bvect[i + num_grid_points_] = 0.0;
    }

    td_math::dgelsd_lsquare(2 * num_grid_points_, num_grid_points_, vs, amat, bvect, rcond);
}

void Inversion::InitDeltaMat()
{
    delta_mat_.resize(num_grid_points_, num_grid_points_);
    ublas::matrix<real> idmat(num_grid_points_, num_grid_points_);
    idmat = ublas::identity_matrix<real>(num_grid_points_);

    td_math::BC_Partial_xx_2d(BC, idmat, delta_mat_, Dx());
    delta_mat_ = ublas::real(delta_mat_);
}

void Inversion::CalculateVKinMat2(matrix_complex &dmat, matrix_real &vkinmat)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            vkinmat(i, j) = delta_mat_(i, j) * std::real(dmat(i, j));
        }
    }

    for (int i = 0; i < num_grid_points_; ++i)
    {
        real cc = 0.0;
        for (int j = 0; j < num_grid_points_; ++j)
        {
            cc += delta_mat_(i, j) * std::real(dmat(i, j));
        }
        vkinmat(i, i) -= cc;
    }
}

void Inversion::DmatToOrbitals(matrix_complex &admatks, std::vector<vector_complex> &orbitals, std::vector<real> &occupation)
{
    vector_real eigval(num_grid_points_);
    matrix_complex eigvec(num_grid_points_, num_grid_points_);
    td_math::zheev_diag(num_grid_points_, admatks, eigval, eigvec);

    for (int i = 0; i < num_orbitals_; ++i)
    {
        real one_divided_dr = 1 / sqrt(Dx());
        int l = num_grid_points_ - i - 1;
#pragma omp parallel for
        for (int j = 0; j < num_grid_points_; ++j)
        {
            orbitals[i][j] = eigvec(j, l) * one_divided_dr;
            occupation[i] = eigval(l) * Dx();
        }
    }
}

void Inversion::Step(real dt_)
{
    matrix_complex H_psi(num_grid_points_, num_grid_points_);
    HamiltonianWf(WaveFunction2e(), H_psi);
    complex ihalfdt = complex(0, 0.5 * dt_);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_grid_points_; ++i)
    {
        for (int j = 0; j < num_grid_points_; ++j)
        {
            H_psi(i, j) = WaveFunction2e()(i, j) - ihalfdt * H_psi(i, j);
        }
    }

    auto mult = [&](int mm, int ll, matrix_complex &in, matrix_complex &out)
    {
        complex ihalfdt = complex(0, 0.5 * dt_);
        HamiltonianWf(in, out);
#pragma omp parallel for collapse(2)
        for (int i = 0; i < mm; ++i)
        {
            for (int j = 0; j < ll; ++j)
            {
                out(i, j) = ihalfdt * out(i, j) + in(i, j);
            }
        }
    };

    td_math::Biconjugate(num_grid_points_, num_grid_points_, WaveFunction2e(), mult, H_psi, 1e-20, 5000);
}

void Inversion::RG4Step()
{
    CalculateRfDensityMatrix();
    matrix_complex &admat = rf_density_matrix_;
    vector_real vv(num_grid_points_);

#pragma omp parallel for
    for (int i = 0; i < num_grid_points_; ++i)
    {
        vv[i] = PotentialRealSpace()[i];
    }

    matrix_complex dmat2int(num_grid_points_, num_grid_points_);
    matrix_complex k1(num_grid_points_, num_grid_points_);
    vector_real vin(num_grid_points_);
    matrix_complex &admatks = ks_density_matrix_;
    matrix_complex k1ks(num_grid_points_, num_grid_points_);
    vector_real tvs(num_grid_points_);
    matrix_complex tdmat(num_grid_points_, num_grid_points_);
    matrix_complex tdmatks(num_grid_points_, num_grid_points_);

    CalculateRDM2Int(dmat2int);
    DerivDensityMatrix(vv, dmat2int, admat, k1);
    vin = vv;
    DerivDensityMatrixKS(k1, admatks, k1ks, vin, tvs);
    vs_it_ = tvs;

    tdmat = admat + 0.5 * dt * k1;
    tdmatks = admatks + 0.5 * dt * k1ks;

    Step(0.5 * dt);

    CalculateRDM2Int(dmat2int);
    matrix_complex k2(num_grid_points_, num_grid_points_);
    matrix_complex k2ks(num_grid_points_, num_grid_points_);
    DerivDensityMatrix(vv, dmat2int, tdmat, k2);
    DerivDensityMatrixKS(k2, tdmatks, k2ks, vin, tvs);

    vs_it_ = vs_it_ + 2.0 * tvs;
    tdmat = admat + 0.5 * dt * k2;
    tdmatks = admatks + 0.5 * dt * k2ks;

    // maybe some mask functions
    matrix_complex k3(num_grid_points_, num_grid_points_);
    matrix_complex k3ks(num_grid_points_, num_grid_points_);
    DerivDensityMatrix(vv, dmat2int, tdmat, k3);
    DerivDensityMatrixKS(k3, tdmatks, k3ks, vin, tvs);

    vs_it_ = vs_it_ + 2.0 * tvs;
    tdmat = admat + 0.5 * dt * k3;
    tdmatks = admatks + 0.5 * dt * k3ks;

    matrix_complex tmat(num_grid_points_, num_grid_points_);
    tmat = tdmatks - tdmat;

    vector_real dk4(num_grid_points_);
    CalculateFKinDiag(tmat, dk4);

    vector_real dn(num_grid_points_);

#pragma omp parallel for
    for (int i = 0; i < num_grid_points_; ++i)
    {
        dn[i] = std::real(admatks(i, i) - admat(i, i) + (dt / 6) * (k1ks(i, i) - k1(i, i)) + 2.0 * (k2ks(i, i) - k2(i, i)) + 2.0 * (k3ks(i, i) - k3(i, i)) + dk4[i]);
    }

    tmat = admatks - admat;
    tmat = 6.0 / dt * tmat + (k1ks - k1) + 2.0 * (k2ks - k2) + 2.0 * (k3ks - k3);

    vector_real df(num_grid_points_);
    CalculateFKinDiag(tmat, df);

    vector_real ddj(num_grid_points_);
    ddj = -dn / (dt * dt) - df;

    Step(0.5 * dt);

    CalculateRDM2Int(dmat2int);

    matrix_complex k4(num_grid_points_, num_grid_points_);
    matrix_complex k4ks(num_grid_points_, num_grid_points_);
    DerivDensityMatrix(vv, dmat2int, tdmat, k4);
    DerivDensityMatrixKSRef(ddj, k4, tdmatks, k4ks, vin, tvs);
    vs_it_ = vs_it_ + tvs;

    admatks = admatks + (dt / 6.0) * (k1ks + 2.0 * k2ks + 2.0 * k3ks + k4ks);

    vs_it_ = vs_it_ / 6.0;
}
