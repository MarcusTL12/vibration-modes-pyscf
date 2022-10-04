import pyscf
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian.thermo import harmonic_analysis

import vibration

mol = pyscf.gto.M(atom="N 0 0 0; H 1 0 0; H 0 1 0; H 0 0 1", basis="sto3g")

mol_opt = optimize(mol.RHF())

h = mol_opt.Hessian().hess()

h_res = harmonic_analysis(mol_opt, h)

atoms, r = vibration.get_atom_coords(mol_opt)

vibration.animate_vibs("nh3", mol, h_res)
