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

function get_mol_opt(atom; basis1="sto3g", basis2="ccpvdz", prec2=1e-10)
    mol = pyscf.gto.M(atom=atom; basis="sto3g")
    hf = mol.RHF()

    mol_opt = geomopt.optimize(hf)
    mol_opt.basis = "ccpvdz"
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

function get_h2o_opt()
    get_mol_opt("O 0 0 0; H 1 0 0; H 0 1 0")
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

function get_rhf_hessian(mol)
    mol.RHF().run().Hessian().hess()
end

function get_mass_weighted_hessian(h, mol)
    masslist = mol.atom_mass_list() * 1836.1526734311

    h = copy(h)

    for i in eachindex(masslist), j in eachindex(masslist)
        h[i, j, :, :] ./= √(masslist[i] * masslist[j])
    end

    h
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

function make_vib_anims(name, mol; newbasis=nothing)
    if !isnothing(newbasis)
        mol.basis = newbasis
        mol.build()
    end

    h = get_rhf_hessian(mol)
    hm = get_mass_weighted_hessian(h, mol)

    e, v = get_hessian_eigen(hm)

    atoms, r = get_atom_coords(mol)

    ks = [v[:]' * reshape(h, 3 * length(atoms), 3 * length(atoms)) * v[:] for v in eachslice(v, dims=3)]

    mkpath(name)
    for i in axes(v, 3)
        e_au = √(abs(e[i]))
        e_cm = e_au * 220000
        println("ħω $i: ", round(e_au, sigdigits=3), " => ",
            round(e_cm, sigdigits=3), " cm⁻¹")
        animate_vib("$name/$i.xyz", atoms, r, v[:, :, i], 0.5, 25)
    end

    display(ks)
end

