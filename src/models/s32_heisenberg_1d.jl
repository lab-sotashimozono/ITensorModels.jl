using ITensors: SiteType, @SiteType_str

"""
    S32Heisenberg1D(; J=1.0, site=SiteType("S=3/2"))

S=3/2 isotropic antiferromagnetic Heisenberg chain:

    H = J sum_i Si . S_{i+1}

with SiteType("S=3/2"). By the Lieb-Mattis theorem and Haldane
conjecture, half-integer spin chains are gapless (algebraic long-range
order) in contrast to the gapped integer-spin Haldane phase.

Useful benchmark for half-integer-spin DMRG / imaginary-time evolution.
"""
Base.@kwdef struct S32Heisenberg1D <: AbstractLatticeModel
    J::Float64 = 1.0
    site::SiteType = SiteType("S=3/2")
end

site_type(m::S32Heisenberg1D) = m.site

function bond_term(m::S32Heisenberg1D, i::Int, j::Int)
    H = OpSum()
    H += m.J, "Sx", i, "Sx", j
    H += m.J, "Sy", i, "Sy", j
    H += m.J, "Sz", i, "Sz", j
    return H
end

boundary_patch(::S32Heisenberg1D, ::Int) = OpSum()
bond_coupling_term(m::S32Heisenberg1D, i::Int, j::Int) = bond_term(m, i, j)
onsite_term(::S32Heisenberg1D, ::Int) = OpSum()

function onsite_observable_op(::S32Heisenberg1D, name::Symbol)
    name === :sx && return "Sx"
    name === :sy && return "Sy"
    name === :sz && return "Sz"
    return error("S32Heisenberg1D: unsupported onsite observable $name")
end
