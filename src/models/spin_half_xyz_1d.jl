using ITensors: SiteType, @SiteType_str

"""
    SpinHalfXYZ1D(; Jx=1.0, Jy=1.0, Jz=1.0, site=SiteType("S=1/2"))

1D anisotropic XYZ Heisenberg chain on spin-1/2:

    H = Jx sum_i Sx_i Sx_{i+1} + Jy sum_i Sy_i Sy_{i+1} + Jz sum_i Sz_i Sz_{i+1}

The most general SU(2)-breaking nearest-neighbour two-body spin-1/2
Hamiltonian. Special cases: Jx=Jy=Jz Heisenberg (XXX), Jx=Jy XXZ,
Jx=Jy/Jz=0 XY. Exactly solvable by Bethe ansatz (XXZ) / Baxter (XYZ).
"""
Base.@kwdef struct SpinHalfXYZ1D <: AbstractLatticeModel
    Jx::Float64 = 1.0
    Jy::Float64 = 1.0
    Jz::Float64 = 1.0
    site::SiteType = SiteType("S=1/2")
end

site_type(m::SpinHalfXYZ1D) = m.site

function bond_term(m::SpinHalfXYZ1D, i::Int, j::Int)
    H = OpSum()
    H += m.Jx, "Sx", i, "Sx", j
    H += m.Jy, "Sy", i, "Sy", j
    H += m.Jz, "Sz", i, "Sz", j
    return H
end

boundary_patch(::SpinHalfXYZ1D, ::Int) = OpSum()
bond_coupling_term(m::SpinHalfXYZ1D, i::Int, j::Int) = bond_term(m, i, j)
onsite_term(::SpinHalfXYZ1D, ::Int) = OpSum()

function onsite_observable_op(::SpinHalfXYZ1D, name::Symbol)
    name === :sx && return "Sx"
    name === :sy && return "Sy"
    name === :sz && return "Sz"
    return error("SpinHalfXYZ1D: unsupported onsite observable $name")
end
