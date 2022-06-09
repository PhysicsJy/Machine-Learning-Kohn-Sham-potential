#ifndef __ground_state_h__
#define __ground_state_h__
#include "hamiltonian.h"
#include "td_math.h"

// Split operator method in imaginary time to find the ground state wavefunction
void step(Hamiltonian& ham) {
  // This part could be parallized
  for (int j = 0; j < ham.NumOfGridPoints(); ++j) {
    ham.WaveFunction()[j] *= std::exp(-0.5 * dt * ham.PotentialRealSpace()[j]);
  }
  td_math::FourierTransformation(ham.WaveFunction(), true);

  for (int j = 0; j < ham.NumOfGridPoints(); ++j) {
    ham.WaveFunction()[j] *= std::exp(-dt * ham.KineticFourierSpace()[j]);
  }

  td_math::FourierTransformation(ham.WaveFunction(), false);

  for (int j = 0; j < ham.NumOfGridPoints(); ++j) {
    ham.WaveFunction()[j] *= std::exp(-0.5 * dt * ham.PotentialRealSpace()[j]);
  }

  ham.WaveFunction() /= norm_2(ham.WaveFunction());
  ham.WaveFunction() *= boost::math::rsqrt(ham.Dx());
}

void evolution_imaginary(Hamiltonian& ham, int steps, real etot) {
  real err = 1;
  int i = 0;
  vector_complex wf = ham.WaveFunction();
  while (i < steps && err > etot) {
    step(ham);
    err = std::sqrt(norm_2(ham.WaveFunction() - wf));
    wf = ham.WaveFunction();
    i++;
  }
  // std::cout << err << std::endl;
}

#endif
