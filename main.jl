using PyCall
using LinearAlgebra
using OhMyREPL

pyscf = pyimport("pyscf")
geomopt = pyimport("pyscf.geomopt.geometric_solver")

include("writexyz.jl")

function get_atom_coords(mol)
    atoms = String[]
    r = Float64[]

    for i in 1:mol.natm
        push!(atoms, mol.atom_symbol(i - 1))
        append!(r, mol.atom_coord(i - 1))
    end

    atoms, reshape(r, 3, length(r) ÷ 3)
end

function get_nh3_opt()
    mol = pyscf.gto.M(atom="N 0 0 0; H 1 0 0; H 0 1 0; H 0 0 1"; basis="sto3g")
    hf = mol.RHF()

    geomopt.optimize(hf;
        convergence_energy=1e-10,
        convergence_grms=1e-10,
        convergence_gmax=1e-10,
        convergence_drms=1e-10,
        convergence_dmax=1e-10
    )
end

function get_h2o_opt()
    mol = pyscf.gto.M(atom="O 0 0 0; H 1 0 0; H 0 1 0"; basis="sto3g")
    hf = mol.RHF()

    mol_opt = geomopt.optimize(hf;
        convergence_energy=1e-5,
        convergence_grms=1e-5,
        convergence_gmax=1e-5,
        convergence_drms=1e-5,
        convergence_dmax=1e-5
    )
    mol_opt.basis = "ccpvdz"
    mol_opt.build()
    hf = mol_opt.RHF()

    geomopt.optimize(hf;
        convergence_energy=1e-10,
        convergence_grms=1e-10,
        convergence_gmax=1e-10,
        convergence_drms=1e-10,
        convergence_dmax=1e-10
    )
end

function get_rhf_hessian(mol)
    mol.RHF().run().Hessian().hess()
end

function get_hessian_eigen(h)
    hp = PermutedDimsArray(h, (3, 1, 4, 2))
    hm = reshape(hp, size(hp, 1) * size(hp, 2), size(hp, 3) * size(hp, 4))

    e, v = eigen(hm)

    e, reshape(v, 3, size(v, 1) ÷ 3, size(v, 2))
end

function animate_vib(filename, atoms, r, dr, amp, n_frames)
    buf = copy(r)

    open(filename, "w") do io
        for θ in range(0, 2π, length=n_frames)
            copyto!(buf, r)
            axpy!(amp * sin(θ), dr, buf)

            writexyz(io, atoms, buf)
        end
    end
end
