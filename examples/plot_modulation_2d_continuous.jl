# 2D RadialEnvelope shapes — continuous-function heatmaps.
# Visualises (rectangular | cylindrical | spherical) × (SinSquare | SinPower{N} | SmoothBoundary)
# before committing the design to src/. No ITensorModels dependency yet
# (the envelope formulas are inlined here precisely so we can iterate on
# the geometry independently of the type machinery).
using Plots

const Lx = 60
const Ly = 60

# Center at the geometric midpoint of a chain of length L (matches the
# 1D convention site_weight(SSD, i, L) = sin²(π (i - 1/2) / L), whose
# zeros sit at i = 0 and i = L, so the center is L/2 + 0.5 in continuous
# coordinates running over [0.5, L+0.5]).
center(L) = (L + 1) / 2

# 1-axis SSD envelope, continuous version (matches site_weight at integer i).
ssd_axis(x, L) = sin(pi * (x - 0.5) / L)^2

# 1-axis SinPower{N} envelope: cos^N(π d / (2R)).
sin_power_axis(x, L, N) = sin(pi * (x - 0.5) / L)^N

# Radial profile (1D, distance-based). Returns 1 at d=0 and 0 at d=R.
function profile_sin_square(d, R)
    d >= R && return 0.0
    return cos(pi * d / (2R))^2
end

function profile_sin_power(d, R, N)
    d >= R && return 0.0
    return cos(pi * d / (2R))^N
end

# Smooth-boundary profile: 1 for d <= R - edge, half-cosine ramp from
# R - edge to R (drops to 0 at d = R).
function profile_smooth(d, R, edge)
    d >= R && return 0.0
    d <= R - edge && return 1.0
    return (1 - cos(pi * (R - d) / edge)) / 2
end

# ---- Envelope realisations (D=2) ------------------------------------

# (1) Rectangular SSD = axis product (≡ LatticeCore feat/ssd-weight).
env_rect(x, y) = ssd_axis(x, Lx) * ssd_axis(y, Ly)

# (2) Cylindrical SSD: SSD along the x-axis, uniform along y. (Useful
# when y is the periodic / "compact" direction of a cylinder.)
env_cyl_x(x, y) = ssd_axis(x, Lx)

# (3) Spherical SSD with the inscribed circle R = min(Lx, Ly) / 2:
# the envelope reaches 0 on the inscribed circle and is identically
# zero outside it (corners are dark).
function env_sphere_inscribed(x, y)
    cx = center(Lx);
    cy = center(Ly)
    d = sqrt((x - cx)^2 + (y - cy)^2)
    R = min(Lx, Ly) / 2
    return profile_sin_square(d, R)
end

# (4) Spherical SSD with the circumscribed circle R = √((Lx/2)² + (Ly/2)²):
# the envelope only reaches 0 at the corners; the four edge midpoints
# carry a finite weight. This is what "spherical, no zero rim" looks like.
function env_sphere_circumscribed(x, y)
    cx = center(Lx);
    cy = center(Ly)
    d = sqrt((x - cx)^2 + (y - cy)^2)
    R = sqrt((Lx / 2)^2 + (Ly / 2)^2)
    return profile_sin_square(d, R)
end

# (5) Spherical SinPower{4}: sharper roll-off at the boundary, broader
# central plateau (Hotta-Shibata flavour on a disk).
function env_sphere_sinpow4(x, y)
    cx = center(Lx);
    cy = center(Ly)
    d = sqrt((x - cx)^2 + (y - cy)^2)
    R = min(Lx, Ly) / 2
    return profile_sin_power(d, R, 4)
end

# (6) Spherical SmoothBoundary: flat-1 plateau in the bulk + half-cosine
# ramp on the rim (Vekic-White on a disk).
function env_sphere_smooth(x, y)
    cx = center(Lx);
    cy = center(Ly)
    d = sqrt((x - cx)^2 + (y - cy)^2)
    R = min(Lx, Ly) / 2
    return profile_smooth(d, R, 8)
end

# ---- Build heatmaps -------------------------------------------------

xs = range(0.5, Lx + 0.5; length=200)
ys = range(0.5, Ly + 0.5; length=200)

panels = [
    ("Rectangular SSD\n(axis product)", env_rect),
    ("Cylindrical SSD (x-axial)\nuniform in y", env_cyl_x),
    ("Spherical SSD\ninscribed R = min(Lx,Ly)/2", env_sphere_inscribed),
    ("Spherical SSD\ncircumscribed R = ‖(Lx,Ly)‖/2", env_sphere_circumscribed),
    ("Spherical SinPower{4}\nR = min/2", env_sphere_sinpow4),
    ("Spherical SmoothBoundary\nR = min/2, edge = 8", env_sphere_smooth),
]

plts = []
for (title, f) in panels
    Z = [f(x, y) for y in ys, x in xs]   # Z[j, i] = f(xs[i], ys[j])
    p = heatmap(
        xs,
        ys,
        Z;
        title=title,
        xlabel="x",
        ylabel="y",
        aspect_ratio=:equal,
        c=:viridis,
        clims=(0.0, 1.0),
        colorbar=false,
        framestyle=:box,
    )
    # Overlay a 0.5-level contour to make the falloff region visible.
    contour!(p, xs, ys, Z; levels=[0.5], lc=:white, lw=1.2)
    push!(plts, p)
end

p_all = plot(
    plts...;
    layout=(3, 2),
    size=(1100, 1500),
    plot_title="2D Modulation Envelopes (Lx=Ly=$Lx)",
)

outdir = joinpath(@__DIR__, "out")
mkpath(outdir)
savefig(p_all, joinpath(outdir, "2d_continuous.png"))
println("Wrote: ", joinpath("out", "2d_continuous.png"))

# Also dump a 1D radial cut through the disk envelopes so we can compare
# the radial profiles directly.
rs = range(0.0, min(Lx, Ly) / 2; length=200)
p_radial = plot(;
    xlabel="distance from center r",
    ylabel="envelope(r)",
    title="Radial profile cuts (R = min(Lx,Ly)/2 = $(min(Lx,Ly) ÷ 2))",
    legend=:outerright,
    framestyle=:box,
    size=(900, 420),
)
plot!(
    p_radial,
    rs,
    [profile_sin_square(r, min(Lx, Ly) / 2) for r in rs];
    label="SinSquare",
    lw=2,
)
plot!(
    p_radial,
    rs,
    [profile_sin_power(r, min(Lx, Ly) / 2, 4) for r in rs];
    label="SinPower{4}",
    lw=2,
)
plot!(
    p_radial,
    rs,
    [profile_sin_power(r, min(Lx, Ly) / 2, 8) for r in rs];
    label="SinPower{8}",
    lw=2,
)
plot!(
    p_radial,
    rs,
    [profile_smooth(r, min(Lx, Ly) / 2, 4) for r in rs];
    label="SmoothBoundary edge=4",
    lw=2,
)
plot!(
    p_radial,
    rs,
    [profile_smooth(r, min(Lx, Ly) / 2, 8) for r in rs];
    label="SmoothBoundary edge=8",
    lw=2,
)
plot!(
    p_radial,
    rs,
    [profile_smooth(r, min(Lx, Ly) / 2, 12) for r in rs];
    label="SmoothBoundary edge=12",
    lw=2,
)
savefig(p_radial, joinpath(outdir, "2d_radial_profile.png"))
println("Wrote: ", joinpath("out", "2d_radial_profile.png"))
