import os
import numpy as np


def get_atom_coords(mol):
    atoms = []
    coords = []

    for i in range(mol.natm):
        atoms.append(mol.atom_symbol(i))
        coords.append(mol.atom_coord(i))

    return atoms, np.array(coords)


def write_xyz(f, atoms, r):
    Å2B = 1.8897261245650618
    r = r / Å2B
    f.write(f"{len(atoms)}\n\n")
    for a, xyz in zip(atoms, r):
        f.write(f"{a} {xyz[0]} {xyz[1]} {xyz[2]}\n")


def write_vib_xyz(filename, atoms, r, dr, amp, nframes):
    with open(filename, "w") as f:
        for t in np.linspace(0, 2 * np.pi, nframes):
            write_xyz(f, atoms, r + amp * np.sin(t) * dr)


def animate_vibs(name, mol, harmonic_results, time_factor=1.0, amp_factor=1.0):
    m_p = 1836.1526734311

    atoms, r = get_atom_coords(mol)

    modes = harmonic_results["norm_mode"]
    wavenumbers = harmonic_results["freq_wavenumber"]
    e_au = harmonic_results["freq_au"]
    k_au = harmonic_results["force_const_au"]

    os.makedirs(name, exist_ok=True)

    for i, (mode, e_cm, e, k) in enumerate(zip(modes, wavenumbers, e_au, k_au)):
        amp = np.sqrt(e / k / np.sqrt(m_p)) * amp_factor
        nframes = int(np.ceil(25 * 2000 / e_cm * time_factor))
        write_vib_xyz(f"{name}/{i + 1}.xyz", atoms, r, mode, amp, nframes)
