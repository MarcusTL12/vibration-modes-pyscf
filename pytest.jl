using PyCall
using LinearAlgebra
using OhMyREPL

pyscf = pyimport("pyscf")
geomopt = pyimport("pyscf.geomopt.geometric_solver")
thermo = pyimport("pyscf.hessian.thermo")

include("mol_opt.jl")

pushfirst!(pyimport("sys")."path", ".")

vibration = pyimport("vibration")

function test_h2o()
    mol = get_h2o_opt()

    h = mol.RHF().run().Hessian().hess()

    vib = thermo.harmonic_analysis(mol, h)

    display(vib)

    vibration.animate_vibs("anims/h2o", mol, vib)
end
