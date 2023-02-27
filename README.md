# Learning Kohn-Sham potential from dynamics in a time-dependent Kohn-Sham system

This work is the official implementation of the method mentioned in the paper [*Machine-learning Kohn-Sham potential from dynamics in time-dependent Kohn-Sham systems*](https://arxiv.org/abs/2207.00687).

This work is supported by the U.S. Department of Energy, Office of Science, and Office of Advanced Scientific Computing Research, under the Quantum
Computing Application Teams program (Award 1979657), NSF (Grant 1820747) and additional funding from the DOE (Award A053685). 

## How to cite

```
@misc{yang2022machinelearning,
    title={Machine-learning Kohn-Sham potential from dynamics in time-dependent Kohn-Sham systems},
    author={Jun Yang and James D Whitfield},
    year={2022},
    eprint={2207.00687},
    archivePrefix={arXiv},
    primaryClass={quant-ph}
}
```

===============
## 0. Setup

- Pytorch
- numpy
- scipy
- Openblas
- Boost
- fftw3

## 1. NN sturcture and propagating pipeline
------------

- The structure of our neural network. $q_i = \sqrt{2} \Re c_i$'s are the scaled real part of the coefficients, $p_i = \sqrt{2} \Im c_i$'s are the scaled imaginary part of the coefficients. The scaled real and imaginary parts of the coefficients at time $t$ are fed into the neural network as our inputs. There are two hidden layers. Neurons are connected non-linear activation functions (in the numerical experiments we carried out, softplus and tanh were chosen). The output layer is the energy functional of the Kohn-Sham system. The loss function relies on the gradient (shown in the last layer in the figure) of the energy functional, we compute it with auto differentiation functionality in Pytorch.

<p align="center">
  <img  width= 500 height= 250 src=./plots/nn_structure.png>
</p>


- Pipeline of propagating the initial state with our neural network
<p align="center">
  <img src=./plots/pipeline.png>
</p>

## 2. Example: Harmonic oscillator
------------
- Generate the traning set for the harmonic oscillator model: `mkdir build && cmake .. && make && ./build/tddft`
- Train the model for harmonic oscillator: `python3 train_ho.py --verbose`
- Visualize and analyze the results: `make_plots.ipynb`

#### 2.1 ML energy level and ML potential
- The left panel is the figure of energy level. We manually shift the ground state energy to 0 for comparison. The neural network generates the correct level spacing for the harmonic oscillator test. 
- The right panel is the machine learned Kohn-Sham potential(black dots) versus the actual potential(red line).
<p align="center">
  <img width=320 height=320 src=./plots/lvls_15eigen_xmax6_150pts.png>
  <img width=320 height=320 src=./plots/V_15eigen_xmax6_150pts.png>
</p>

#### 2.2 ML potential scaling vs the size of the training set
- Comparison of the machine learned Kohn-Sham potential among different data sets. From left to right, each plot shows the results of 15, 10, and 5 eigenstates in the data set.

<p align="center">
  <img width= 280 height= 280 src=./plots/V_15eigen_xmax6_150pts.png>
  <img width= 280 height= 280 src=./plots/V_10eigen_xmax4_100pts.png>
  <img width= 280 height= 280 src=./plots/V_5eigen_xmax4_100pts.png>
</p>

#### 2.3 ML density evolution

- The machine learned  density and the exact density at different timestamps. The blue solid line is the machine learned density, the red dotted line is the exact density.
<p align="center">
  <img width= 280 height= 280 src=./plots/evolution.gif>
</p>

## 3. Example: 2 electron test
------------
- Generate the traning set for the 2electron model: `python3 make_dataset.py`
- Train the model for harmonic oscillator: `python3 train_2e.py --verbose`
- Visualize and analyze the results: `make_plots.ipynb`

#### 3.1 ML density evolution
- The machine learned  density and the exact density at different timestamps. The black dashed line is the machine learned density, the red solid line is the exact density. The three panels correspond to the result result at timestamp $t = 1.0 (a.u.)$ (left), $t = 4.0 (a.u.)$ (middle) and $t = 15.0 (a.u.)$ (right)

<p align="center">
  <img width= 280 height= 280 src=./plots/2e_evo_t5.0.png>
  <img width= 280 height= 280 src=./plots/2e_evo_t8.5.png>
  <img width= 280 height= 280 src=./plots/2e_evo_t13.0.png>
</p>

#### 3.2 ML Kohn-Sham potential
- The machine learned Kohn-Sham potential, the exact Kohn-Sham potential and the ALDA potential at different timestamps in the two electron test. The black dashed line is the machine learned density, the red solid line is the exact density, and the blue dashed line is the ALDA potential.

<p align="center">
  <img width= 280 height= 280 src=./plots/2e_5.0.png>
  <img width= 280 height= 280 src=./plots/2e_8.5.png>
  <img width= 280 height= 280 src=./plots/2e_13.0.png>
</p>
