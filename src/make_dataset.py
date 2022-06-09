import numpy as np
import numpy.polynomial.hermite as Herm
import math

m=1.
w=1.
hbar=1.
#Discretized space
x_lim = 6
x = np.linspace(-x_lim, x_lim, 150)
dx = x[1] - x[0]

def hermite(x, n):
    xi = np.sqrt(m*w/hbar)*x
    herm_coeffs = np.zeros(n+1)
    herm_coeffs[n] = 1
    return Herm.hermval(xi, herm_coeffs)
  
def stationary_state(x,n):
    xi = np.sqrt(m*w/hbar)*x
    prefactor = 1./math.sqrt(2.**n * math.factorial(n)) * (m*w/(np.pi*hbar))**(0.25)
    psi = prefactor * np.exp(- xi**2 / 2) * hermite(x,n)
    return psi

wf_t = []
wf_dt = []


def E_n(n):
    return (n + 1/2)*w

num_eigns = 15
En = [E_n(n) for n in range(num_eigns)]
psi = np.array([stationary_state(x,n) for n in range(num_eigns)])
repetitions = 50
En = np.transpose([En] * repetitions)
time_total = 2000
end_time = 4 * np.pi
dt = 4 * np.pi / (1 + time_total)
for n in range(num_eigns):
    E = (n + 0.5) * w
    end_time = 2 * np.pi / E
    psi_0 = stationary_state(x,n)
    T = np.linspace(0, end_time, time_total)
    for t in T:
        wf_t.append(psi_0 * np.exp(-1j * E * t))
        wf_dt.append(-1j * E * psi_0 * np.exp(-1j * E * t))

wf_t = np.array(wf_t)
wf_dt = np.array(wf_dt)

np.savetxt("./data/ho_wf_15_2.csv", wf_t)
np.savetxt("./data/ho_wf_dt_15_2.csv", wf_dt)