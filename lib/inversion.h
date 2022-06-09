#ifndef __inversion_h__
#define __inversion_h__
#include "common.h"
#include "hamiltonian.h"
#include "td_math.h"

class Inversion : public Hamiltonian
{
private:
    void InitRfSystem();
    
    // Reference system
    vector_real rf_density_;
    vector_complex rf_density_current_;
    vector_real rf_hartree_fock_potential_;
    vector_real rf_hartree_fock_potential_xc_;
    
    // matrix_complex wave_function_2d_;
    matrix_complex rf_density_matrix_;
    matrix_complex rf_density_matrix_xch_;
    
public:
    Inversion()
    {
        InitKSSystem();
        InitRfSystem();
        InitMixedSystem();
        InitMidRef();
        InitDeltaMat();
    };
    
public:
    // Reference system
    void CalculateRfDensity();
    
    void CalculateRfCurrent();
    
    void CalculateRfDensityMatrix();
    
    void CalculateRfDensityMatrixXCH();
    
    void CalculateRfHartreeFockPotential();
    
    void CalculateRfHartreeXC();
    
    void CalculateRfPotentialTC();
    
public:
    // Reference system
    vector_real &RfDensity()
    {
        return rf_density_;
    }
    
    vector_complex &RfDensityCurrent()
    {
        return rf_density_current_;
    }
    
    vector_real &RfHartreeFockPotential()
    {
        return rf_hartree_fock_potential_;
    }
    
    matrix_complex &RfDensityMatrix()
    {
        return rf_density_matrix_;
    }
    
    matrix_complex &RfDensityMatrixXCH()
    {
        return rf_density_matrix_xch_;
    }
    
private:
    // KS system
    int num_orbitals_;
    std::vector<vector_complex> orbitals_;
    std::vector<real> occupation_;
    
    vector_real ks_potential_;
    vector_real ks_density_;
    vector_complex ks_density_current_;
    vector_real ks_hartree_fock_potential_;
    vector_real ks_potential_xc_;
    
    matrix_complex ks_density_matrix_;
    matrix_complex ks_density_matrix_xch_;
    
public:
    // KS system
    void InitKSSystem();
    
    void CalculateKSDensity();
    
    void CalculateKSCurrent();
    
    void CalculateKSDensityMatrix();
    
    void CalculateKSDensityMatrixXCH();
    
    void CalculateKSHartreeFockPotential();
    
    void CalculateKSPotentialWXC();
    
    void CalculateKSPotentialTC();
    
public:
    vector_real &KSDensity()
    {
        return ks_density_;
    }
    
    vector_complex &KSDensityCurrent()
    {
        return ks_density_current_;
    }
    
    vector_real &KSHartreeFockPotential()
    {
        return ks_hartree_fock_potential_;
    }
    
    matrix_complex &KSDensityMatrix()
    {
        return ks_density_matrix_;
    }
    
    matrix_complex &KSDensityMatrixXCH()
    {
        return ks_density_matrix_xch_;
    }
    
private:
    // mix of KS and Reference systems
    // vector_real potential_hartree_fock_;
    
    int midref;
    
    vector_real potential_xc_kinetic_component_; // V_C^T
    
    vector_real potential_xc_interaction_component_; // V_XC^W
    
    vector_real deriv_potential_xc_interaction_component_; // d/dx V_XC^W
    
    vector_real potential_hxc_;
    
    vector_real potential_xc_kinetic_component_rf_;
    
    vector_real potential_xc_kinetic_component_ks_;
    
    vector_real vs_it_;
    
public:
    int MidRef() {
        return midref;
    }
    
    vector_real& VSit() {
        return vs_it_;
    }
    
    vector_real& VWXC() {
        return potential_xc_interaction_component_;
    }
    
    vector_real& DVWXC() {
        return deriv_potential_xc_interaction_component_;
    }
    
    vector_real& VHXC() {
        return potential_hxc_;
    }
    
    vector_real& VKINCKS() {
        return potential_xc_kinetic_component_ks_;
    }
    
    vector_real& VKINCRef() {
        return potential_xc_kinetic_component_rf_;
    }
    
    vector_real& VKINC() {
        return potential_xc_kinetic_component_;
    }
    
    void InitMidRef();
    
    void InitDeltaMat();
    
    void InitMixedSystem();
    
    void CalculatePotentialWXC();
    
    void CalculatePotentialTC();
    
    void CalculatePotentialHXC();
    
    void CalculateDerivPotentialWXC();
    
private:
    real xpes = 1.0e-3;
    
    matrix_real delta_mat_;
    
public:
    void Calibrate();
    
    void Step(real dt_);
    
    void RG4Step();
    
    // helper functions for RG4Step
    complex RDM2(int i, int j, int l, int m);
    
    void CalculateRDM2Int(matrix_complex &dmat2int);
    
    void Liouvil(vector_real &v, matrix_complex &dmat);
    
    void DerivDensityMatrix(vector_real &v, matrix_complex &dmat2int, matrix_complex &dmat, matrix_complex &output_mat);
    
    void WaveFunctionToVWXCH(matrix_complex &psi, vector_real &vwxch);
    
    void WaveFunctionToDens(matrix_complex &psi, vector_real &dens);
    
    void WaveFunctionToPaird(matrix_complex &psi, matrix_complex &paird);
    
    void CaculateVWXCH(vector_real &dens, matrix_complex &paird, vector_real &vwxch);
    
    void DerivDensityMatrixKS(matrix_complex &fdmat, matrix_complex &dmat_ks, matrix_complex &fdmat_ks, vector_real &vin, vector_real &vs);
    
    void DerivDensityMatrixKSRef(vector_real &ref, matrix_complex &fdmat, matrix_complex &dmat_ks, matrix_complex &fdmat_ks, vector_real &vin, vector_real &vs);
    
    void DerivDensityMatrixKin(matrix_complex &dmat, matrix_complex &fdmat);
    
    void DerivDensityMatrixV(vector_real &vv, matrix_complex &dmat, matrix_complex &fdmat);
    
    void CalculateFKinDiag(matrix_complex &dmat, vector_real &fkindiag);
    
    void CalculateKSPotential(vector_real &divjdot, matrix_complex &dmatks, vector_real &vs, real eps, real rcond);
    
    void CalculateVKinMat2(matrix_complex &dmat, matrix_real &vkinmat);
    
    void DmatToOrbitals(matrix_complex &admatks, std::vector<vector_complex>& orbitals, std::vector<real>& occupation);
};
#endif
