function get_mol_opt(atom; basis1="sto3g", basis2="ccpvdz", prec2=1e-9)
    mol = pyscf.gto.M(atom=atom; basis=basis1)
    hf = mol.RHF()

    mol_opt = geomopt.optimize(hf)
    mol_opt.basis = basis2
    mol_opt.build()
    hf = mol_opt.RHF()

    geomopt.optimize(hf;
        convergence_energy=prec2,
        convergence_grms=prec2,
        convergence_gmax=prec2,
        convergence_drms=prec2 * 100,
        convergence_dmax=prec2 * 100
    )
end

function get_nh3_opt()
    get_mol_opt("N 0 0 0; H 1 0 0; H 0 1 0; H 0 0 1")
end

function get_h2o_opt(basis="ccpvdz")
    get_mol_opt("O 0 0 0; H 1 0 0; H 0 1 0"; basis2=basis)
end

function get_ethene_opt()
    get_mol_opt("""
C 0  0 0
C 1  0 0
H 0  1 0
H 0 -1 0
H 1  1 0
H 1 -1 0
"""; prec2=1e-7)
end

function get_ethane_opt()
    get_mol_opt("""
C 0  0 0
C 1  0 0
H 0  1 0
H 0 -1 0
H 0  0 1
H 1  1 0
H 1 -1 0
H 1 -1 1
"""; prec2=1e-7)
end

function get_HCl_opt()
    get_mol_opt("Cl 0 0 0; H 1 0 0")
end

function get_CO2_opt()
    get_mol_opt("C 0 0 0; O -1 0 0; O 1 0 0")
end

function get_CO_opt()
    get_mol_opt("C 0 0 0; O 1 0 0")
end

function get_2h2o_opt()
    get_mol_opt("""
O 0 0 0
H 1 0 0
H 0 1 0
O -2 0 0
H -1 0 0
H -2 0 1
"""; prec2=1e-7)
end

function get_H2_opt()
    get_mol_opt("H 0 0 0; H 1 0 0"; basis2="ccpvqz")
end

function get_benzene_opt()
    get_mol_opt("""
C 1 2 0
C 3 1 0
C 5 2 0
C 5 4 0
C 3 5 0
C 1 4 0
H 0 1 0
H 3 0 0
H 6 1 0
H 6 5 0
H 3 6 0
H 0 5 0
""")
end

function get_tetrahedrane_opt()
    get_mol_opt("""
C 0   0   0
C 1   0   0
C 0.5 1   0
C 0.5 0.5 0.5
H -0.5 -0.5 -0.5
H  1.5 -0.5 -0.5
H  0.5  1.5 -0.5
H  0.5 0.5 1
""")
end