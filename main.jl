using PyCall
using LinearAlgebra
using OhMyREPL

pyscf = pyimport("pyscf")
geomopt = pyimport("pyscf.geomopt.geometric_solver")
thermo = pyimport("pyscf.hessian.thermo")

include("writexyz.jl")
include("num_hess.jl")
include("mol_opt.jl")

const mp::Float64 = 1836.1526734311

function get_atom_coords(mol)
    atoms = String[]
    r = Float64[]

    for i in 1:mol.natm
        push!(atoms, mol.atom_symbol(i - 1))
        append!(r, mol.atom_coord(i - 1))
    end

    atoms, reshape(r, 3, length(r) ÷ 3)
end

function make_mol(atoms, r, basis)
    pyscf.gto.M(
        atom=[(string(a), r[1], r[2], r[3])
              for (a, r) in zip(atoms, eachcol(r))],
        basis=basis,
        unit="bohr"
    )
end

function make_rhf_ef(atoms, basis)
    function ef(r)
        make_mol(atoms, r, basis).RHF().kernel()
    end
end

function make_ccsd_ef(atoms, basis)
    function ef(r)
        make_mol(atoms, r, basis).RHF().run().CCSD().run().e_tot
    end
end

function make_ccsd_t_ef(atoms, basis)
    function ef(r)
        ccsd = make_mol(atoms, r, basis).RHF().run().CCSD().run()
        ccsd.e_tot + ccsd.ccsd_t()
    end
end

function make_hess_func_2(ef, h)
    function hess_func(r)
        get_num_hessian_2(ef, r, h)
    end
end

function get_rhf_hessian(mol)
    mol.RHF().run().Hessian().hess()
end

function make_rhf_hess_func(atoms, basis)
    function hess_func(r)
        make_mol(atoms, r, basis).RHF().run().Hessian().hess()
    end
end

function get_mass_weighted_hessian(h, mol)
    masslist = mol.atom_mass_list() * mp

    h = copy(h)

    for i in eachindex(masslist), j in eachindex(masslist)
        h[i, j, :, :] ./= √(masslist[i] * masslist[j])
    end

    h
end

function get_hessian_eigen(h)
    hp = PermutedDimsArray(h, (3, 1, 4, 2))
    hm = reshape(hp, size(hp, 1) * size(hp, 2), size(hp, 3) * size(hp, 4))

    e, v = eigen(Symmetric(hm))

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

function find_ks(h, v)
    hp = PermutedDimsArray(h, (3, 1, 4, 2))
    hm = reshape(hp, size(hp, 1) * size(hp, 2), size(hp, 3) * size(hp, 4))

    [
        begin
            vv = @view v[:]
            vv' * hm * vv
        end for v in eachslice(v, dims=3)
    ]
end

function make_vib_anims(name, mol, hess_func=nothing; linear=false, time_factor=0.5)
    atoms, r = get_atom_coords(mol)

    if isnothing(hess_func)
        hess_func = make_rhf_hess_func(atoms, mol.basis)
    end

    h = hess_func(r)
    hm = get_mass_weighted_hessian(h, mol)

    e, v = get_hessian_eigen(hm)

    ks = find_ks(h, v)

    time_factor *= 25 * √maximum(abs, e)

    mkpath(name)
    n_skip = linear ? 5 : 6
    for i in axes(v, 3)[n_skip+1:end]
        e_au = √(abs(e[i]))
        e_cm = e_au * 219474.6
        m_eff = (ks[i] / e[i]) / mp
        println("ħω $i:\t=> ",
            round(e_cm, sigdigits=5), " cm⁻¹, \t",
            "k = ", round(ks[i] * 1556.8931028304864; sigdigits=5), " ᴺ/ₘ, \t",
            "m = ", round(m_eff; sigdigits=3), " mₚ")
        println(e_au)
        animate_vib("$name/$i.xyz", atoms, r, v[:, :, i], √(2 * e_au / ks[i]),
            ceil(Int, time_factor / e_au))
    end
end
