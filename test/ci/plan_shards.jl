# test/ci/plan_shards.jl — emit a balanced GitHub-matrix shard plan.
#
#   julia test/ci/plan_shards.jl <N> [timings.tsv]
#
# Prints (stdout, last line) a JSON array for `matrix: include:` —
#   [{"sid":"s1","files":"base/test_a.jl,base/test_b.jl","aqua":"1"}, …]
#
# WHAT to run is the canonical universe (universe.jl — the single source of truth
# + completeness guard). Timings only decide HOW to split:
#   * timings.tsv present → Longest-Processing-Time bin-packing, so every shard
#                           finishes at roughly the same wall-clock.
#   * absent / unreadable → deterministic round-robin (identical to the
#                           ITENSORMODELS_TEST_SHARD k/N fallback). Never a leak,
#                           just not yet time-optimal.
#
# A new / unknown file gets a PESSIMISTIC estimate (P90 of the known times) so a
# surprise-heavy new test is isolated rather than piled onto an already-heavy shard.
#
# The "aqua" field is emitted for parity with the sibling repos' planner (it marks the
# lightest shard, the one that would carry a whole-package one-shot check). ITensorModels
# has no Aqua suite today, so runtests.jl simply ignores it — wiring Aqua in later needs
# no planner change.
#
# Ported from QAtlas.jl's test/ci/plan_shards.jl.

include(joinpath(@__DIR__, "universe.jl"))   # ALL_TEST_FILES, test_file_key

const N = let a = get(ARGS, 1, "")
    n = tryparse(Int, a)
    (n !== nothing && n >= 1) ||
        error("plan_shards.jl: arg 1 must be N>=1; got $(repr(a))")
    n
end
const TIMINGS_PATH = get(ARGS, 2, "")

const KEYS = [test_file_key(d, f) for (d, f) in ALL_TEST_FILES]

# Load "key\tseconds"; silently ignore missing/garbage rows — the timing plane is
# advisory and must DEGRADE, never abort planning.
function load_timings(path)
    t = Dict{String,Float64}()
    (isempty(path) || !isfile(path)) && return t
    for ln in eachline(path)
        parts = split(strip(ln), '\t')
        length(parts) == 2 || continue
        v = tryparse(Float64, parts[2])
        v === nothing && continue
        t[String(parts[1])] = v
    end
    return t
end
const TIMES = load_timings(TIMINGS_PATH)

const DEFAULT_T = if isempty(TIMES)
    1.0
else
    s = sort(collect(values(TIMES)))
    s[clamp(ceil(Int, 0.9 * length(s)), 1, length(s))]   # P90 of the known times
end

est(k) = get(TIMES, k, DEFAULT_T)

bins = [String[] for _ in 1:N]
loads = zeros(Float64, N)

if isempty(TIMES)
    for (i, k) in enumerate(KEYS)
        b = ((i - 1) % N) + 1
        push!(bins[b], k)
        loads[b] += est(k)
    end
    mode = "round-robin (no timing history)"
else
    for k in sort(KEYS; by=est, rev=true)      # longest first
        b = argmin(loads)                       # → least-loaded bin
        push!(bins[b], k)
        loads[b] += est(k)
    end
    mode = "LPT bin-packing"
end

# Drop empty bins. N is a REQUEST, not a promise: a suite with fewer files than N
# would otherwise emit shards that run zero tests (and, with the SHARD k/N fallback,
# silently pass). The emitted plan always has min(N, #files) shards.
const KEEP = [b for b in 1:N if !isempty(bins[b])]
bins = bins[KEEP]
loads = loads[KEEP]
const NSHARDS = length(bins)

# Aqua is a one-shot whole-package check: give it to the shard that finishes
# EARLIEST, so it never lands on the critical path.
const AQUA_BIN = argmin(loads)

io = IOBuffer()                                 # ASCII paths ⇒ no escaping needed
print(io, "[")
for b in 1:NSHARDS
    b == 1 || print(io, ",")
    sid = "s" * string(b)
    print(io, "{\"sid\":\"", sid, "\",\"files\":\"", join(bins[b], ","), "\",")
    print(io, "\"aqua\":\"", b == AQUA_BIN ? "1" : "0", "\"}")
end
print(io, "]")
plan_json = String(take!(io))

# Human-readable summary → stderr (must NOT pollute the JSON on stdout).
println(
    stderr,
    "plan_shards: N=$N → $NSHARDS shards  mode=$mode  files=$(length(KEYS))  " *
    "default_t=$(round(DEFAULT_T; digits=3))s  aqua→s$(AQUA_BIN)",
)
for b in 1:NSHARDS
    println(
        stderr,
        "  s$(b): $(length(bins[b])) files  est=$(round(loads[b]; digits=1))s" *
        (b == AQUA_BIN ? "  +aqua" : ""),
    )
end

println(plan_json)
