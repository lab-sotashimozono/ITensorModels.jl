ENV["GKSwstype"] = "100"

using ITensorModels, Test
using TestShards

# ── Shared namespace (load-order coupling; see below) ────────────────
#
# Every test file is `include`d into `Main`, so in a FULL run `Main` ends up holding the
# UNION of every file's `using` lines — and the suite silently came to depend on that.
# Concretely: 33 files call `site_type(...)` or construct `SSD()`, but only THREE ever
# import `site_type` and NOT ONE imports `SSD`. The other 30 worked purely because some
# alphabetically-earlier file had already pulled the name into `Main`.
#
# Under sharding a file can be the ONLY file its job runs, so that leakage evaporates and
# the shard dies with `UndefVarError`. Hoist the shared namespace here so every shard sees
# exactly what a full run sees. This is strictly MORE binding than before (it can only add
# names, never remove them), so it cannot change any assertion.
#
# ── The name clash the hoist must disarm ─────────────────────────────
#
# `ITensorModels` and `LatticeCore` (hence its re-exporters `Lattice2D` / `QuasiCrystal`)
# BOTH export three names — `SSD`, `site_type`, `bond_weight` — bound to DIFFERENT objects
# (`ITensorModels.SSD` is an `AbstractModulation`; `LatticeCore.SSD` is an
# `AbstractBoundaryModifier`). Two bare `using`s that each export the same name make that
# name AMBIGUOUS in `Main`: it is not imported at all, and the first use throws
# `UndefVarError: SSD not defined in Main`.
#
# In the old full run this stayed hidden by pure luck of ordering: an alphabetically early
# file used `SSD` (resolving `Main.SSD` → `ITensorModels.SSD`) BEFORE any file ran a bare
# `using LatticeCore`, so the later `using` could no longer clobber it. Hoisting the
# `using`s to the top removes that accident — so the resolution must be made EXPLICIT.
# `using ITensorModels: …` is an explicit binding and takes precedence over the implicit
# ambiguity, which is exactly the meaning every test file relies on (every `SSD()` in the
# suite is a modulation; every `bond_weight`/`site_weight` call is `ITensorModels.`-qualified).
using Random
using LinearAlgebra
using Statistics
using ITensors
using ITensorMPS
using LatticeCore
using Lattice2D
using QuasiCrystal
using ITensorModels:
    SSD, site_type, site_weight, bond_weight, bond_term, build_opsum, local_ham_terms

# ── The suite ────────────────────────────────────────────────────────
#
# Every `test_*.jl` under `test/`, in a deterministic order, each one its own shardable unit.
# `@shard` shadows `include` inside the block, so a unit is whatever this loop includes — a new
# file, or a whole new directory, is picked up BY BEING ON DISK. That is what the old
# `test/ci/universe.jl` completeness guard existed to enforce, and it could only ever ERROR,
# because the wiring lived in a second list that could disagree with the tree.
#
# The `using` block above stays OUTSIDE this block, and must. It is the shared namespace every
# unit needs, and an `include` inside the block becomes a unit of its own — it would land on ONE
# shard and every file on the other shards would go back to the `UndefVarError` the hoist was
# written to prevent.
#
# A bare `Pkg.test()` with nothing set in the environment runs all of it, in this order. Run one
# shard locally with `TESTSHARDS_ID=s3 TESTSHARDS_N=8 julia --project -e 'using Pkg; Pkg.test()'`.
TestShards.@shard begin
    for (dir, _, files) in sort!(collect(walkdir(@__DIR__)); by=first)
        for f in sort(files)
            startswith(f, "test_") && endswith(f, ".jl") || continue
            include(joinpath(dir, f))
        end
    end
end
