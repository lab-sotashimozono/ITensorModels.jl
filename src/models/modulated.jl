"""
    ModulatedModel(base::M, L::Int, modulation::Mod)
    modulated(base; L, modulation=SSD())

Apply a spatial envelope ([`AbstractModulation`](@ref)) to a base model
that implements the split protocol ([`bond_coupling_term`](@ref),
[`onsite_term`](@ref)). The resulting Hamiltonian on an open 1D chain
of length `L` is

```
H = Σ_{i=1}^{L-1} bond_weight(mod, i, L) · bond_coupling(base, i, i+1)
    + Σ_{i=1}^{L}   site_weight(mod, i, L) · onsite(base, i)
```

The wrapper plugs into the existing `bond_term` / `boundary_patch`
pipeline by half-distributing the on-site weight across the two
adjacent bond terms and adding the leftover halves at the chain ends
via [`boundary_patch`](@ref). Pair with `build_opsum(..., boundary=:full)`.

# Examples

```julia
ssd_tfim   = modulated(TFIM(J=1.0, h=1.0); L=60)                          # SSD by default
sin4_tfim  = modulated(TFIM(); L=60, modulation=SinPower{4}())            # Hotta-Shibata sin^4
sbc_tfim   = modulated(TFIM(); L=60, modulation=SmoothBoundary(10))       # Vekic-White
uniform    = modulated(TFIM(); L=60, modulation=Uniform())                # sanity: == bare TFIM
custom     = modulated(TFIM(); L=60, modulation=Tabulated(fs, fb))        # arbitrary envelope
```
"""
struct ModulatedModel{M<:AbstractLatticeModel,Mod<:AbstractModulation} <:
       AbstractLatticeModel
    base::M
    L::Int
    modulation::Mod

    function ModulatedModel(
        base::M, L::Int, modulation::Mod
    ) where {M<:AbstractLatticeModel,Mod<:AbstractModulation}
        L >= 2 || error("ModulatedModel: chain length L=$L must be >= 2.")
        return new{M,Mod}(base, L, modulation)
    end
end

"""
    modulated(base; L::Int, modulation::AbstractModulation=SSD())

Convenience constructor. Defaults to [`SSD()`](@ref) — the most common
modulation in the literature.
"""
function modulated(base::AbstractLatticeModel; L::Int, modulation::AbstractModulation=SSD())
    return ModulatedModel(base, L, modulation)
end

site_type(m::ModulatedModel) = site_type(m.base)

# The wrapper consumes the split protocol on the base model. Each bond
# term carries the full bond coupling weighted by `bond_weight(mod, i, L)`
# plus *half* of the on-site weight at each endpoint (so two adjacent
# bonds sum to the full on-site term in the bulk). The leftover halves
# at the two ends are added by [`boundary_patch`](@ref) when callers
# pass `boundary = :full` to [`build_opsum`](@ref).
function bond_term(m::ModulatedModel, i::Int, j::Int)
    j == i + 1 || error(
        "ModulatedModel currently supports only nearest-neighbour bonds " *
        "on a 1D chain (got bond i=$i, j=$j). 2D modulation will need a " *
        "separate wrapper.",
    )
    fb = bond_weight(m.modulation, i, m.L)
    fi = site_weight(m.modulation, i, m.L)
    fj = site_weight(m.modulation, j, m.L)
    opsum = fb * bond_coupling_term(m.base, i, j)
    opsum += (fi / 2) * onsite_term(m.base, i)
    opsum += (fj / 2) * onsite_term(m.base, j)
    return opsum
end

function boundary_patch(m::ModulatedModel, k::Int)
    fk = site_weight(m.modulation, k, m.L)
    return (fk / 2) * onsite_term(m.base, k)
end

# Forward onsite observable lookup to the base model — the modulation
# only rescales energetics, not which operator measures "magnetisation".
function onsite_observable_op(m::ModulatedModel, name::Symbol)
    return onsite_observable_op(m.base, name)
end
