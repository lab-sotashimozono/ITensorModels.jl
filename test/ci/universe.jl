# test/ci/universe.jl — canonical test-file universe (single source of truth).
#
# Included by BOTH test/runtests.jl and test/ci/plan_shards.jl. Pure stdlib, NO
# `using ITensorModels`, so the shard planner stays cheap (it must not precompile the
# ITensors stack just to decide how to split the suite).
#
# This file alone decides WHAT the suite is. The timing history only decides HOW to
# split it — it can never add or drop a file.
#
# Ported from QAtlas.jl's test/ci/universe.jl.

const TEST_ROOT = dirname(@__DIR__)   # test/ci/ → test/

const ALL_DIRS = ["base/"]

_is_test_file(f) = startswith(f, "test_") && endswith(f, ".jl")

# ── Completeness guard ───────────────────────────────────────────────
# Every on-disk directory holding a `test_*.jl` MUST be enumerated in ALL_DIRS, and
# every ALL_DIRS entry must exist and be non-empty. Runs wherever this file is
# included (every shard + the planner) and fails loudly — so a new test directory
# can never be added without being wired in, silently running zero tests.
let
    enumerated = Set(ALL_DIRS)
    discovered = Set{String}()
    for (d, _, files) in walkdir(TEST_ROOT)
        any(_is_test_file, files) || continue
        rel = replace(relpath(d, TEST_ROOT), '\\' => '/')
        rel == "." && continue          # test/ root holds runtests.jl only
        startswith(rel, "ci") && continue
        push!(discovered, rel * "/")
    end
    leaked = sort(collect(setdiff(discovered, enumerated)))
    isempty(leaked) || error(
        "universe.jl completeness guard: these on-disk test directories hold " *
        "test_*.jl files but are NOT in ALL_DIRS and would never run — add them " *
        "to ALL_DIRS: $(leaked)",
    )
    for d in ALL_DIRS
        p = joinpath(TEST_ROOT, d)
        (isdir(p) && any(_is_test_file, readdir(p))) || error(
            "universe.jl completeness guard: ALL_DIRS entry $(repr(d)) is missing " *
            "on disk or contains no test_*.jl files.",
        )
    end
end

# ── Canonical, deterministic global test-file universe ───────────────
# ALL_DIRS order × lexically-sorted files. Every selection mode picks a SUBSET of
# this list, so the union of all shards is exactly this set.
const ALL_TEST_FILES = let acc = Tuple{String,String}[]
    for d in ALL_DIRS
        for f in sort(filter(_is_test_file, readdir(joinpath(TEST_ROOT, d))))
            push!(acc, (d, f))
        end
    end
    acc
end

# Stable "d/f" key used by the timing history and ITENSORMODELS_TEST_FILES.
test_file_key(d, f) = d * f
