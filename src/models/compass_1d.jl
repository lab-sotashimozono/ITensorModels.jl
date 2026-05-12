using ITensors: SiteType, @SiteType_str

"""
    Compass1D(; Jx=1.0, Jy=1.0, site=SiteType("S=1/2"))

1D compass / alternating-bond model on a spin-1/2 chain:

    H = Jx Σ_{i odd}  S^x_i S^x_{i+1}  +  Jy Σ_{i even} S^y_i S^y_{i+1}

i.e. odd (1-2, 3-4, …) bonds carry `XX`, even (2-3, 4-5, …) bonds carry
`YY`. This is the 1D restriction of the 2D Kugel-Khomskii / quantum
compass model (Brzezicki, Dziarmaga, Oleś 2007; Nussinov & van den Brink
RMP 2015) — the simplest setting in which bond-direction-dependent
two-body interactions compete on a single chain.

Non-standard bond Hamiltonian: parity of the *bond* matters, so this
file overrides [`local_ham_terms`](@ref) rather than going through the
default bond / on-site split.
"""
Base.@kwdef struct Compass1D <: AbstractLatticeModel
    Jx::Float64 = 1.0
    Jy::Float64 = 1.0
    site::SiteType = SiteType("S=1/2")
end

site_type(m::Compass1D) = m.site

bond_term(::Compass1D, ::Int, ::Int) = OpSum()
bond_coupling_term(::Compass1D, ::Int, ::Int) = OpSum()
onsite_term(::Compass1D, ::Int) = OpSum()

function onsite_observable_op(::Compass1D, name::Symbol)
    name === :sx && return "Sx"
    name === :sy && return "Sy"
    name === :sz && return "Sz"
    return error("Compass1D: unsupported onsite observable $name")
end

function local_ham_terms(m::Compass1D, phys_sites; boundary::Symbol=:bulk_half_edge)
    phys = collect(phys_sites)
    N = length(phys)
    terms = OpSum[]
    for i in 1:(N - 1)
        H = OpSum()
        if isodd(i)
            H += m.Jx, "Sx", phys[i], "Sx", phys[i + 1]
        else
            H += m.Jy, "Sy", phys[i], "Sy", phys[i + 1]
        end
        push!(terms, H)
    end
    return terms
end
