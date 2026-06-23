using ITensors: SiteType, @SiteType_str
using LinearAlgebra: norm, dot, Diagonal

"""
    AndersonImpurity1D(; εd, U, eon, thop, site=SiteType("Electron"))

Single-impurity Anderson model (SIAM) in **chain geometry** — an interacting
impurity at site 1 coupled to a non-interacting tight-binding bath chain:

    H = εd Σ_σ n_{1σ} + U n_{1↑} n_{1↓}                          (impurity)
      + Σ_{k≥2} eon[k-1] Σ_σ n_{kσ}                              (bath on-site)
      - Σ_{i,σ} thop[i] (c†_{iσ} c_{i+1,σ} + h.c.)               (hoppings)

on `SiteType("Electron")` sites (local dim 4: |0⟩,|↑⟩,|↓⟩,|↑↓⟩). The chain has
`N = length(thop) + 1` sites: site 1 is the impurity, sites 2…N the bath.
`thop[1]` is the impurity↔bath hybridization `V`; `thop[2:end]` are the
bath-chain hoppings; `eon` (length `N-1`) are the bath on-site energies.

Placing the impurity at the chain end means `c†_{1σ}` carries **no**
Jordan–Wigner string (nothing to its left), so the local Green's function
`⟨ψ_g| c_{1σ} e^{-i(H-E_g)t} c†_{1σ} |ψ_g⟩` is obtained with a bare local
operator — convenient for time-dependent impurity spectroscopy.

Build a SIAM whose bath reproduces a semielliptic hybridization with
[`semielliptic_anderson`](@ref); map an arbitrary discretized bath
`{ε_b, v_b}` (star geometry) to this chain with [`star_to_chain`](@ref).
"""
struct AndersonImpurity1D <: AbstractLatticeModel
    εd::Float64
    U::Float64
    eon::Vector{Float64}
    thop::Vector{Float64}
    site::SiteType
    function AndersonImpurity1D(εd, U, eon, thop, site)
        length(eon) == length(thop) || error(
            "AndersonImpurity1D: length(eon)=$(length(eon)) must equal " *
            "length(thop)=$(length(thop)) (= number of bath sites Nb).",
        )
        return new(Float64(εd), Float64(U), Float64.(eon), Float64.(thop), site)
    end
end

function AndersonImpurity1D(; εd, U, eon, thop, site=SiteType("Electron"))
    return AndersonImpurity1D(εd, U, eon, thop, site)
end

site_type(m::AndersonImpurity1D) = m.site

# on-site energy at chain position k (1 = impurity, ≥2 = bath)
_onsite_energy(m::AndersonImpurity1D, k::Int) = k == 1 ? m.εd : m.eon[k - 1]

function bond_term(m::AndersonImpurity1D, i::Int, j::Int)
    H = OpSum()
    t = m.thop[min(i, j)]                       # hopping on bond (i, i+1)
    H += -t, "Cdagup", i, "Cup", j
    H += -t, "Cdagup", j, "Cup", i
    H += -t, "Cdagdn", i, "Cdn", j
    H += -t, "Cdagdn", j, "Cdn", i
    for k in (i, j)                             # half-weight on-site (bulk_half_edge)
        ε = _onsite_energy(m, k)
        H += (ε / 2), "Nup", k
        H += (ε / 2), "Ndn", k
        if k == 1                               # interaction lives only on the impurity
            H += (m.U / 2), "Nupdn", k
        end
    end
    return H
end

function boundary_patch(m::AndersonImpurity1D, k::Int)
    H = OpSum()
    ε = _onsite_energy(m, k)
    H += (ε / 2), "Nup", k
    H += (ε / 2), "Ndn", k
    if k == 1
        H += (m.U / 2), "Nupdn", k
    end
    return H
end

function bond_coupling_term(m::AndersonImpurity1D, i::Int, j::Int)
    H = OpSum()
    t = m.thop[min(i, j)]
    H += -t, "Cdagup", i, "Cup", j
    H += -t, "Cdagup", j, "Cup", i
    H += -t, "Cdagdn", i, "Cdn", j
    H += -t, "Cdagdn", j, "Cdn", i
    return H
end

function onsite_term(m::AndersonImpurity1D, k::Int)
    H = OpSum()
    ε = _onsite_energy(m, k)
    H += ε, "Nup", k
    H += ε, "Ndn", k
    if k == 1
        H += m.U, "Nupdn", k
    end
    return H
end

function onsite_observable_op(::AndersonImpurity1D, name::Symbol)
    name === :nup && return "Nup"
    name === :ndn && return "Ndn"
    name === :n && return "Ntot"
    name === :nupdn && return "Nupdn"
    name === :sz && return "Sz"
    return error("AndersonImpurity1D: unsupported onsite observable $name")
end

"""
    semielliptic_anderson(; Nb, D=1.0, U=2.0, εd=-U/2, V=D/2) -> AndersonImpurity1D

SIAM whose bath reproduces a **semielliptic** hybridization of half-bandwidth
`D`, `-1/π Im Δ(ω) = (2/πD)√(1-(ω/D)²)`. A semielliptic spectrum maps *exactly*
(no discretization error) onto a uniform tight-binding chain: bath on-site
energies `0`, bath-chain hopping `D/2`. The impurity couples to the first bath
site with hybridization `V`. The chain is truncated to `Nb` bath sites
(`N = Nb+1` total); a longer chain pushes back the boundary-reflection time,
extending the accessible evolution time. Defaults are particle–hole symmetric
(`εd = -U/2`). This is the benchmark setup of Cao et al. (arXiv:2311.10909) and
Grundner et al. (arXiv:2312.11705).
"""
function semielliptic_anderson(;
    Nb::Int, D::Float64=1.0, U::Float64=2.0, εd::Float64=(-U / 2), V::Float64=D / 2
)
    Nb ≥ 1 || error("semielliptic_anderson: need Nb ≥ 1 bath sites")
    eon = zeros(Float64, Nb)
    thop = vcat(V, fill(D / 2, Nb - 1))
    return AndersonImpurity1D(; εd, U, eon, thop)
end

"""
    star_to_chain(εb, vb) -> (eon, thop)

Map a discretized bath in **star geometry** — levels `εb` each hybridizing with
the impurity by `vb` — onto the **chain geometry** consumed by
[`AndersonImpurity1D`](@ref), via Lanczos tridiagonalization of `Diagonal(εb)`
with starting vector `vb/‖vb‖` (full reorthogonalization). Returns bath on-site
energies `eon` (length `Nb`) and hoppings `thop` (length `Nb`), where
`thop[1] = ‖vb‖` is the impurity↔bath hybridization and `thop[2:end]` the
bath-chain hoppings. The two geometries give identical impurity dynamics; the
chain is the MPS-friendly form.
"""
function star_to_chain(εb::AbstractVector, vb::AbstractVector)
    Nb = length(εb)
    Nb == length(vb) || error("star_to_chain: εb and vb must have equal length")
    A = Diagonal(Float64.(εb))
    Q = zeros(Float64, Nb, Nb)
    a = zeros(Float64, Nb)
    b = zeros(Float64, max(Nb - 1, 0))
    nv = norm(vb)
    nv > 0 || error("star_to_chain: ‖vb‖ must be > 0")
    q = Float64.(vb) ./ nv
    Q[:, 1] = q
    r = A * q
    a[1] = dot(q, r)
    r = r .- a[1] .* q
    for k in 2:Nb
        r = r .- Q[:, 1:(k - 1)] * (Q[:, 1:(k - 1)]' * r)   # reorthogonalize
        b[k - 1] = norm(r)
        b[k - 1] < 1e-14 && break
        q = r ./ b[k - 1]
        Q[:, k] = q
        r = A * q
        a[k] = dot(q, r)
        r = r .- a[k] .* q .- b[k - 1] .* Q[:, k - 1]
    end
    return a, vcat(nv, b)
end
