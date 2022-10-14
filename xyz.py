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


def write_xyz(mol, filename):
    atoms, r = get_atom_coords(mol)
    with open(filename, "w") as f:
        write_xyz(f, atoms, r)
