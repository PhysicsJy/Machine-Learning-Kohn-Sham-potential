#include "../lib/ground_state.h"
#include "../lib/inversion.h"

#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>
#include "../lib/td_math.h"

int main() {
    bool ground_state = false;
    bool propagation = false;
    bool read_file = true;
    bool calc_1d = false;
    
    if (ground_state) {
        //    Ground state
        //   initialize Hydrogen
        Hamiltonian hydrogen = Hamiltonian();
        vector_real density (hydrogen.NumOfGridPoints());
        hydrogen.GaussianInitialization(complex(10,0));
        
        evolution_imaginary(hydrogen, 100000, 1e-25);
        for (int i = 0; i < hydrogen.NumOfGridPoints(); ++i) {
            real abs_wf = abs(hydrogen.WaveFunction()[i]);
            density[i] = abs_wf * abs_wf;
        }
        
        std::stringstream density_stream;
        density_stream << "output.dat";
        
        std::ofstream fstream = std::ofstream(density_stream.str());
        
        if (fstream) {
            for (int i = 0; i < hydrogen.NumOfGridPoints(); ++i) {
                std::stringstream data_stream;
                
                data_stream << i << "\t" << density[i] << "\n";
                
                fstream.write(data_stream.str().c_str(), data_stream.str().length());
            }
        }
    }
    //   Propagation
    if (propagation) {
        Inversion H;
        H.init_2e();
        H.GaussianInitialization(complex(-10.0,0));
        evolution_imaginary(H, 100000, 1e-25);
        
        int n = H.NumOfGridPoints();
        real p = -1.5;
        
        vector_complex gwp(n);
        for (int i = 0; i < n; ++i) {
            gwp[i] = td_math::GaussianWavePacket(H.Coord(i), 10, 0.1, p);
        }
        
        real norm_gwp = norm_2(gwp);
        gwp /= norm_gwp;
        gwp *= 1 / std::sqrt(H.Dx());
        
        for (int i = 0; i < n; ++i)
        {
            complex gpi = gwp[i];
            for (int j = 0; j < n; ++j)
            {
                complex gpj = gwp[j];
                H.WaveFunction2e()(i, j) = 1 / std::sqrt(2) * (H.WaveFunction()[i] * gpj + H.WaveFunction()[j] * gpi);
            }
        }
        
        matrix_complex H_psi(n, n);
        
        auto mult = [&](int mm, int ll, matrix_complex &in, matrix_complex &out)
        {
            complex ihalfdt = complex(0, 0.5 * dt);
            H.HamiltonianWf(in, out);
#pragma omp parallel for collapse(2)
            for (int i = 0; i < mm; ++i) {
                for (int j = 0; j < ll; ++j) {
                    out(i, j) = ihalfdt * out(i, j) + in(i, j);
                }
            }
        };
        std::stringstream wf_stream;
        wf_stream << "output" + std::to_string(p) + ".dat";
        std::ofstream fstream = std::ofstream(wf_stream.str());
        auto start=std::chrono::system_clock::now();
#pragma unroll
        for (int i = 0; i < 30000; ++i)
        {
            H.CalculateRfDensity();
            H.CalculateRfCurrent();
            
            if (fstream)
            {
                for (int j = 0; j < n; ++j)
                {
                    real dens_j = H.RfDensity()[j];
                    real current_j = std::real(H.RfDensityCurrent()[j]);
                    
                    fstream << std::setprecision(17) << i << "\t" << j << "\t" << dens_j << "\t" << current_j << "\n";
                }
            }
            std::cout << "density" <<std::endl;
            H.HamiltonianWf(H.WaveFunction2e(), H_psi);
            std::cout << "Ha" <<std::endl;
            complex ihalfdt = complex(0, 0.5 * dt);
#pragma omp parallel for collapse(2)
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    H_psi(i, j) = H.WaveFunction2e()(i, j) - ihalfdt * H_psi(i, j);
                }
            }
            std::cout << "H_psi" <<std::endl;
            
            td_math::Biconjugate(n, n, H.WaveFunction2e(), mult, H_psi, 1e-20, 10000);
            std::cout << i << std::endl;
        }
        
        fstream.close();
        auto end=std::chrono::system_clock::now();
        std::chrono::duration<double> t=end-start;
        std::cout<<"run time: "<<t.count()*1.<<" s."<<std::endl;
    }
    
    if (read_file) {
        Inversion H;
        int midref = H.MidRef();
        std::ifstream inputFile;
        inputFile.open("/Users/junyang/computationalPhysics/TDDFTcopy2/Release/output-1.5.dat");
        int n = sys_size;
        int ind_t;
        int ind_x;
        int T = 30000;
        real dx = (sys_xmax - sys_xmin) / sys_size;
        matrix_real dens(T, n);
        matrix_real current(T, n);
        std::cout << "reading density and current" << std::endl;
        if (inputFile) {
            for (int i = 0; i < T; ++i) {
                for (int j = 0; j < n; ++j) {
                    inputFile >> ind_t >> ind_x >> dens(i, j) >> current(i, j);
                }
            }
        }
        
        std::cout << "calculating g" << std::endl;
        matrix_real g(T, n);
#pragma omp parallel for collapse(2)
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < n; ++j) {
                g(i, j) = 0.5 + 0.5 * std::tanh(1000 * (dens(i, j) - xsmall) / xsmall);
            }
        }
        
        std::cout << "calculating u" << std::endl;
        // u = current / dens
        matrix_real u(T, n);
#pragma omp parallel for collapse(2)
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < n; ++j) {
                u(i, j) = current(i, j) / (dens(i, j) + xsmall);
            }
        }
        
        std::cout << "calculating partial_u" << std::endl;
        // partial_t_u
        matrix_real partial_t_u(T, n);
        matrix_real int_partial_t_u(T, n);
        td_math::Partial_x_2d(u, partial_t_u, dt);
        td_math::Cumtrapz(partial_t_u, int_partial_t_u, dx);
        
        matrix_real partial_yy_sqrt_dens(T, n);
        matrix_real sqrt_dens(T, n);
#pragma omp parallel for collapse(2)
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < n; ++j) {
                sqrt_dens(i, j) = std::sqrt(dens(i, j) + xsmall);
            }
        }
        td_math::Partial_yy_2d(sqrt_dens, partial_yy_sqrt_dens, dx);
        
        matrix_real vs(T, n);
        matrix_real vs_res(T,n);
#pragma omp parallel for collapse(2)
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < n; ++j) {
                vs(i, j) = 0.5 * ((partial_yy_sqrt_dens(i, j) / sqrt_dens(i, j)) - u(i, j) * u(i, j)) - int_partial_t_u(i, j);
            }
        }
        
#pragma omp parallel for collapse(2)
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < n; ++j) {
                vs_res(i, j) = vs(i, j) - vs(i, midref);
            }
        }
        
        matrix_real vh(T, n);
        matrix_real vext(T, n);
        
#pragma omp parallel for collapse(2)
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < n; ++j) {
                vext(i, j) = H.PotentialRealSpace()[j];
            }
        }
        
        
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < n; ++i) {
                real sum = 0;
#pragma omp parallel for reduction(+ : sum)
                for (int j = 0; j < n; ++j) {
                    sum += H.PotentialInteraction(i, j) * dens(t, j) * H.Dx();
                }
                vh(t, i) = sum;
            }
        }
        
#pragma omp parallel for collapse(2)
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < n; ++j) {
                vs_res(i, j) = vs_res(i, j) - vh(i, j) - vext(i, j);
            }
        }
        
        matrix_real real_part(T, n);
        matrix_real imag_part(T, n);
#pragma omp parallel for collapse(2)
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < n; ++j) {
                real_part(i, j) = std::sqrt(0.5 * dens(i, j)) * std::cos(int_partial_t_u(i, j));
                imag_part(i, j) = std::sqrt(0.5 * dens(i, j)) * std::sin(int_partial_t_u(i, j));
            }
        }
        
        std::stringstream wf_stream;
        wf_stream << "vs-1.5.dat";
        std::ofstream fstream = std::ofstream(wf_stream.str());
        
        if (fstream)
        {
            for (int i = 0; i < T; ++i) {
                for (int j = 0; j < n; ++j)
                {
                    fstream << std::setprecision(17) << vs_res(i, j) << '\t';
                }
                fstream << "\n";
            }
        }
        fstream.close();
        
        std::string filename = "orbitals-1.5.dat";
        std::ofstream orbital_stream = std::ofstream(filename);
        
        if (orbital_stream) {
            for (int i = 0; i < T; ++i) {
                for (int j = 0; j < n; ++j)
                {
                    orbital_stream << std::setprecision(17) << real_part(i, j) << '\t' << imag_part(i, j) << '\t';
                }
                orbital_stream << "\n";
            }
        }
        orbital_stream.close();
        matrix_real real_part_dt(T, n);
        matrix_real imag_part_dt(T, n);
        td_math::Partial_x_2d(real_part, real_part_dt, dt);
        td_math::Partial_x_2d(imag_part, imag_part_dt, dt);
        
        filename = "deriv_orbitals-1.5.dat";
        std::ofstream deriv_orbital_stream = std::ofstream(filename);
        
        if (deriv_orbital_stream) {
            for (int i = 0; i < T; ++i) {
                for (int j = 0; j < n; ++j)
                {
                    deriv_orbital_stream << std::setprecision(17) << real_part_dt(i, j) << '\t' << imag_part_dt(i, j) << '\t';
                }
                deriv_orbital_stream << "\n";
            }
        }
        deriv_orbital_stream.close();
    }
}
