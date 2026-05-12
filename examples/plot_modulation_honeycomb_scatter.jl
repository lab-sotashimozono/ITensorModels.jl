# Honeycomb modulation envelope scatter.
#
# Sample each ND envelope (sphere / cylinder / rectangle, plus a
# higher-N sphere) on a real honeycomb(L, L) lattice and scatter the
# sites coloured by site_weight. The bond_weight at every bond's
# midpoint is overlaid as a thin line segment coloured the same way,
# so the geometry of each envelope is immediately legible.
using ITensorModels
using LatticeCore
using LatticeCore: num_sites, position, positions, bonds
using Lattice2D
using Plots

const L = 8

lat = Lattice2D.honeycomb(L, L; boundary=OpenAxis())
ps = collect(positions(lat))
xs = [p[1] for p in ps]
ys = [p[2] for p in ps]

panels = [
    ("spherical_ssd (inscribed)", spherical_ssd(lat; radius=:inscribed)),
    ("spherical_ssd (circumscribed)", spherical_ssd(lat; radius=:circumscribed)),
    ("spherical_ssd (N=4)", spherical_ssd(lat; radius=:circumscribed, N=4)),
    ("cylindrical_ssd (axis=1)", cylindrical_ssd(lat; axis=1)),
    ("cylindrical_ssd (axis=2)", cylindrical_ssd(lat; axis=2)),
    ("rectangular_ssd", rectangular_ssd(lat)),
]

plts = []
for (title, env) in panels
    ws = [ITensorModels.site_weight(env, lat, k) for k in 1:num_sites(lat)]
    p = scatter(
        xs,
        ys;
        marker_z=ws,
        c=:viridis,
        ms=8,
        msw=0.4,
        clims=(0.0, 1.0),
        aspect_ratio=:equal,
        title=title,
        xlabel="x",
        ylabel="y",
        framestyle=:box,
        legend=false,
        colorbar=true,
    )
    # Overlay bond midpoints coloured by bond_weight for visual continuity.
    for b in bonds(lat)
        ri = position(lat, b.i)
        rj = position(lat, b.j)
        wb = ITensorModels.bond_weight(env, lat, b.i, b.j)
        col = cgrad(:viridis, [0.0, 1.0])[wb]
        plot!(p, [ri[1], rj[1]], [ri[2], rj[2]]; color=col, lw=1.0, label="")
    end
    push!(plts, p)
end

p_all = plot(
    plts...;
    layout=(3, 2),
    size=(1100, 1500),
    plot_title="ND modulation envelopes on honeycomb($L, $L) — site_weight scatter",
)

outdir = joinpath(@__DIR__, "out")
mkpath(outdir)
savefig(p_all, joinpath(outdir, "honeycomb_envelopes.png"))
println("Wrote: ", joinpath("out", "honeycomb_envelopes.png"))
