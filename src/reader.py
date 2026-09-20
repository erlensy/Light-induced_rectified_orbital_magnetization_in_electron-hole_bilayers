from matplotlib import pyplot as plt
import numpy as np

plt.rcParams["axes.formatter.limits"] = (0, 0)
plt.rcParams["axes.formatter.use_mathtext"] = True

def wavefunctions():
    energies = np.fromfile(f"../data/wavefunctions_energies.bin", dtype = np.float64)
    n = 1000
    for state in range(6):
        psi = np.fromfile(f"../data/psi_{state}.bin", dtype = np.complex128).reshape(n, n)
        plt.imshow(np.abs(psi) ** 2)
        plt.title(f"$\\varepsilon_{state}$ = {np.round(energies[state], 2)} [meV]")
        plt.show()

def binding_energy():
    n_d = 50
    for kappa in range(1, 5):
        data = np.fromfile(f"../data/binding_energy_kappa={kappa}.bin", dtype = np.float64).reshape(2, n_d)
        plt.plot(data[0, :], data[1, :], label = f"$\\kappa = {kappa}$")
    plt.legend(); plt.xlabel("d [m]"); plt.ylabel("$\\varepsilon_0 [meV]$")
    plt.show()

def mag_of_t():
    labels = ["LHCP", "LP", "RHCP"]
    for i in range(3):
        orbital_mag = np.fromfile(f"../data/mag_of_t_{labels[i]}.bin", dtype = np.complex128)
        plt.plot(np.arange(0, len(orbital_mag) - 1, 1) / (0.1 * len(orbital_mag)), np.real(orbital_mag[:-1]), label = labels[i])
    plt.xlabel("t / T"); plt.ylabel("$M / \\mu_B$"); plt.legend()
    plt.show()

def mag_of_d():
    n_d = 50
    for kappa in range(1, 5):
        data = np.fromfile(f"../data/mag_of_d_kappa={kappa}.bin", dtype = np.float64).reshape(2, n_d)
        plt.plot(data[0, :], data[1, :], label = f"$\\kappa = {kappa}$")
    plt.legend(); plt.xlabel("d [m]"); plt.ylabel("$\langle M \\rangle_{7T} / \mu_B$"); plt.gca().invert_xaxis(); plt.gca().set_yscale("log")
    plt.show()

wavefunctions()
binding_energy()
mag_of_t()
mag_of_d()
