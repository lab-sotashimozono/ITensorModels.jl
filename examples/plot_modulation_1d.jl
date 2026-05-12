# 1D modulation envelopes — visual sanity check of site_weight / bond_weight.
# Output: out/*.png next to this script.
using ITensorModels
using Plots

const L = 60

modulations = [
    ("Uniform", Uniform()),
    ("SSD", SSD()),
    ("SinPower{4}", SinPower{4}()),
    ("SinPower{6}", SinPower{6}()),
    ("SinPower{8}", SinPower{8}()),
    ("SmoothBoundary(5)", SmoothBoundary(5)),
    ("SmoothBoundary(10)", SmoothBoundary(10)),
    ("SmoothBoundary(20)", SmoothBoundary(20)),
]

site_x = 1:L
bond_x = 1:(L - 1)

# ---- site_weight ----
p_site = plot(;
    xlabel="site index i",
    ylabel="site_weight(mod, i, L=$L)",
    title="site weight envelopes (1D, L=$L)",
    legend=:outerright,
    size=(900, 420),
    framestyle=:box,
    grid=true,
)
for (name, m) in modulations
    ys = [site_weight(m, i, L) for i in site_x]
    plot!(p_site, site_x, ys; label=name, lw=2)
end

# ---- bond_weight ----
p_bond = plot(;
    xlabel="bond index i (bond connects i and i+1)",
    ylabel="bond_weight(mod, i, L=$L)",
    title="bond weight envelopes (1D, L=$L)",
    legend=:outerright,
    size=(900, 420),
    framestyle=:box,
    grid=true,
)
for (name, m) in modulations
    ys = [bond_weight(m, i, L) for i in bond_x]
    plot!(p_bond, bond_x, ys; label=name, lw=2)
end

# Combined panel.
p_all = plot(p_site, p_bond; layout=(2, 1), size=(1000, 800))

outdir = joinpath(@__DIR__, "out")
mkpath(outdir)
savefig(p_site, joinpath(outdir, "site_weight.png"))
savefig(p_bond, joinpath(outdir, "bond_weight.png"))
savefig(p_all, joinpath(outdir, "combined.png"))
println("Wrote: ", readdir(outdir))
