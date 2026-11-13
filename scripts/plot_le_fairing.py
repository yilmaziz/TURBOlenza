"""
plot_le_fairing.py
==================
Diagnostics of the robust C2 leading edge (geometry.py, "The C2 leading edge"): the legacy C1
LE, the previous C2 (fit_quintic_surface, which kept the legacy LE fair), and the robust C2
(monotone-curvature fair from the nose circle to an adaptive anchor), on SS and PS.

    Figure 1  LE geometry overlay, equal aspect, with close-ups (default, thin, high camber)
    Figure 2  turning angle of the wall from the apex tangent against s / L
    Figure 3  curvature kappa c against s / L
    Figure 4  d(tan beta)/dm = d2y/dm2 in the cascade frame, exact parametric derivatives
    Figure 5  normal deviation from the legacy C1 surface against s / L
    Figure 6  (--continuity) anchor, departure, LE deviation and fairness against cl0 and t/c

Every curve quantity comes from exact derivatives: the analytic C1 reference pieces and the
B-splines. Run from the repository root:

    python scripts/plot_le_fairing.py [--variant emery_base] [--save DIR] [--no-show] [--continuity]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turbolenza.b2b import geometry as g  # noqa: E402

# --- settings ------------------------------------------------------------------------------
SETTINGS = {
    "blades": [                       # (label, cl0, t_max_over_c)
        ("NACA 65-(12A10)10", 1.2, 0.10),
        ("thin 65-(12A10)04", 1.2, 0.04),
        ("high camber 65-(16A10)10", 1.6, 0.10),
    ],
    "stagger_deg": 33.0,
    "solidity": 1.0,
    "variant": "kovach_sandercock",
    "le_window_u": 0.035,             # s / L shown in figures 2-5
    "samples": 6001,
    "closeup_half_width_radii": 0.15, # smallest close-up half width, in LE radii
    "closeup_deviation_factor": 8.0,  # close-up half width: this times the largest LE change
    "singular_cos_beta": 0.05,        # figure 4 shades |cos(beta)| below this
    "continuity_t": (0.04, 0.10, 0.002),
    "continuity_cl0": (0.0, 1.6, 0.04),
    "colors": {"C1": "black", "old C2": "tab:orange", "robust C2": "tab:blue"},
    "dpi": 130,
}
# -------------------------------------------------------------------------------------------


THICKNESS = g.DEFAULT_THICKNESS_RECONSTRUCTION["C2"]


def build(cl0: float, t: float, variant: str):
    """Legacy reference surfaces, the previous C2 fits and the robust C2 fits of one blade, all
    with the C2 default thickness reconstruction."""
    references = g.naca65_reference_surfaces(cl0, t, profile_variant=variant, thickness_reconstruction=THICKNESS)
    old = tuple(g.fit_quintic_surface(reference) for reference in references)
    new = g._naca65_quintic_fits(cl0, t, variant, THICKNESS)
    return references, old, new


def legacy_trace(reference: g.ReferenceSurface, u_max: float, n: int):
    """C1 points, unit tangents, curvature and s / L from the analytic pieces."""
    sample = reference.sample(max(41, n // len(reference.pieces)))
    keep = sample.s / reference.length <= u_max
    return (sample.points[keep], sample.tangent[keep], sample.second[keep], sample.kappa[keep],
            sample.s[keep] / reference.length)


def curve_trace(curve: g.ParametricBladeCurve, u_max: float, n: int):
    u = np.linspace(0.0, u_max, n)
    d1 = curve.derivative(u, 1)
    return curve.evaluate(u), d1 / np.hypot(*d1.T)[:, None], curve.derivative(u, 2), curve.curvature(u), u


def turning(tangent: np.ndarray, sense: float) -> np.ndarray:
    """Turning angle from the first tangent, positive in the apex curvature's sense (deg)."""
    angle = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    return np.degrees(sense * (angle - angle[0]))


def figure_geometry(plt, data, settings):
    rows = len(data)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.4 * rows), squeeze=False)
    colors = settings["colors"]
    for row, (label, references, old, new) in enumerate(data):
        ax = axes[row, 0]
        for k, (reference, fit_old, fit_new) in enumerate(zip(references, old, new)):
            u_max = 2.0 * fit_new.anchor_u
            p1 = legacy_trace(reference, u_max, settings["samples"])[0]
            p_old = fit_old.curve.evaluate(np.linspace(0, u_max, settings["samples"]))
            p_new = fit_new.curve.evaluate(np.linspace(0, u_max, settings["samples"]))
            ax.plot(*p1.T, color=colors["C1"], lw=1.6, label="legacy C1" if k == 0 else None)
            ax.plot(*p_old.T, color=colors["old C2"], lw=1.0, ls="--", label="old C2" if k == 0 else None)
            ax.plot(*p_new.T, color=colors["robust C2"], lw=1.0, label="robust C2" if k == 0 else None)
            le = fit_new.leading_edge
            for phi, marker, name in ((le.departure_angle, "o", "departure"), (le.anchor_angle, "s", "anchor")):
                point = le.points(phi)[0]
                ax.plot(*point, marker, color=colors["robust C2"], ms=5, mfc="white",
                        label=name if (k == 0 and row == 0) else None)
        nose = new[0].leading_edge.nose
        circle = nose.center + nose.radius * np.column_stack((np.cos(np.linspace(0, 2 * np.pi, 361)),
                                                              np.sin(np.linspace(0, 2 * np.pi, 361))))
        ax.plot(*circle.T, color="0.6", lw=0.6, ls=":", label="LE circle")
        ax.set_aspect("equal")
        ax.set_title(f"{label}: LE region (blade frame)")
        ax.set_xlabel("x / c"); ax.set_ylabel("y / c")
        ax.legend(fontsize=7, loc="upper right")
        for col, (name, k) in enumerate((("suction", 0), ("pressure", 1)), start=1):
            ax = axes[row, col]
            reference, fit_old, fit_new = references[k], old[k], new[k]
            le = fit_new.leading_edge
            sample = reference.sample(401)
            distance = fit_new.curve.project(sample.points, fit_new.parameter(sample.s))[1]
            zone = np.flatnonzero(sample.s <= le.anchor)
            worst = zone[np.argmax(np.abs(distance[zone]))]
            centre = sample.points[worst]
            span = max(settings["closeup_deviation_factor"] * abs(distance[worst]),
                       settings["closeup_half_width_radii"] * le.nose.radius)
            u_max = 1.5 * fit_new.anchor_u
            p1 = legacy_trace(reference, u_max, settings["samples"])[0]
            ax.plot(*p1.T, color=colors["C1"], lw=1.6, label="legacy C1")
            ax.plot(*fit_old.curve.evaluate(np.linspace(0, u_max, settings["samples"])).T, color=colors["old C2"],
                    lw=1.0, ls="--", label="old C2")
            ax.plot(*fit_new.curve.evaluate(np.linspace(0, u_max, settings["samples"])).T, color=colors["robust C2"],
                    lw=1.0, label="robust C2")
            for phi, marker in ((le.departure_angle, "o"), (le.anchor_angle, "s")):
                ax.plot(*le.points(phi)[0], marker, color=colors["robust C2"], ms=5, mfc="white")
            ax.set_xlim(centre[0] - span, centre[0] + span); ax.set_ylim(centre[1] - span, centre[1] + span)
            ax.set_aspect("equal")
            ax.set_title(f"{name}: close-up at the largest change, {fit_new.le_max_deviation:.2e} c")
            ax.set_xlabel("x / c"); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def figure_turning(plt, data, settings):
    fig, axes = plt.subplots(len(data), 2, figsize=(13, 3.6 * len(data)), squeeze=False)
    colors = settings["colors"]
    for row, (label, references, old, new) in enumerate(data):
        for col, (name, k, sense) in enumerate((("suction", 0, -1.0), ("pressure", 1, 1.0))):
            ax = axes[row, col]
            reference, fit = references[k], new[k]
            window = max(settings["le_window_u"], 1.5 * fit.anchor_u)
            _, t1, _, _, u1 = legacy_trace(reference, window, settings["samples"])
            _, t2, _, _, u2 = curve_trace(fit.curve, window, settings["samples"])
            ax.plot(u1, turning(t1, sense), color=colors["C1"], lw=1.4, label="legacy C1")
            ax.plot(u2, turning(t2, sense), color=colors["robust C2"], lw=1.0, label="robust C2")
            for value, style, text in ((fit.departure_u, ":", "departure"), (fit.anchor_u, "--", "anchor")):
                ax.axvline(value, color="0.4", ls=style, lw=0.8)
                ax.text(value, ax.get_ylim()[0], f" {text}", rotation=90, va="bottom", fontsize=7, color="0.3")
            ax.set_title(f"{label}, {name}: turning from the apex tangent")
            ax.set_xlabel("s / L"); ax.set_ylabel("turning angle (deg)")
            ax.set_xlim(0, window); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def figure_curvature(plt, data, settings):
    fig, axes = plt.subplots(len(data), 2, figsize=(13, 3.6 * len(data)), squeeze=False)
    colors = settings["colors"]
    for row, (label, references, old, new) in enumerate(data):
        for col, (name, k) in enumerate((("suction", 0), ("pressure", 1))):
            ax = axes[row, col]
            reference, fit = references[k], new[k]
            window = max(settings["le_window_u"], 1.5 * fit.anchor_u)
            *_, k1, u1 = legacy_trace(reference, window, settings["samples"])
            *_, k2, u2 = curve_trace(fit.curve, window, settings["samples"])
            ax.plot(u1, k1, color=colors["C1"], lw=1.4, label="legacy C1")
            ax.plot(u2, k2, color=colors["robust C2"], lw=1.0, label="robust C2")
            radius = fit.leading_edge.nose.radius
            for level in (1.0 / radius, -1.0 / radius):
                ax.axhline(level, color="0.7", lw=0.6, ls=":")
            ax.axhline(0.0, color="0.5", lw=0.5)
            for value, style in ((fit.departure_u, ":"), (fit.anchor_u, "--")):
                ax.axvline(value, color="0.4", ls=style, lw=0.8)
            ax.set_title(f"{label}, {name}: curvature (dotted: +-1/R_LE; departure, anchor)")
            ax.set_xlabel("s / L"); ax.set_ylabel("kappa c"); ax.set_xlim(0, window)
            ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def figure_slope(plt, data, settings, stagger):
    fig, axes = plt.subplots(len(data), 2, figsize=(13, 3.6 * len(data)), squeeze=False)
    colors = settings["colors"]
    for row, (label, references, old, new) in enumerate(data):
        for col, (name, k) in enumerate((("suction", 0), ("pressure", 1))):
            ax = axes[row, col]
            reference, fit = references[k], new[k]
            window = max(settings["le_window_u"], 1.5 * fit.anchor_u)
            _, t1, s1, _, u1 = legacy_trace(reference, window, settings["samples"])
            c, s = math.cos(stagger), math.sin(stagger)
            rotate = np.array([[c, -s], [s, c]])
            _, legacy = g.wall_slopes(t1 @ rotate.T, s1 @ rotate.T)
            curve = fit.curve.rotated(stagger)
            u2 = np.linspace(0.0, window, settings["samples"])
            _, robust = curve.wall_slope(u2)
            d1 = curve.derivative(u2, 1)
            cos_beta = np.abs(d1[:, 0]) / np.hypot(*d1.T)
            ax.plot(u1, legacy, color=colors["C1"], lw=1.4, label="legacy C1")
            ax.plot(u2, robust, color=colors["robust C2"], lw=1.0, label="robust C2")
            near = cos_beta < settings["singular_cos_beta"]
            if np.any(near):
                ax.fill_between(u2, 0, 1, where=near, transform=ax.get_xaxis_transform(), color="0.85",
                                label=f"|cos beta| < {settings['singular_cos_beta']}")
            for value, style in ((fit.departure_u, ":"), (fit.anchor_u, "--")):
                ax.axvline(value, color="0.4", ls=style, lw=0.8)
            ax.set_yscale("symlog", linthresh=10.0)
            ax.set_title(f"{label}, {name}: d(tan beta)/dm, cascade frame (stagger {math.degrees(stagger):g} deg)")
            ax.set_xlabel("s / L"); ax.set_ylabel("d2y/dm2  c"); ax.set_xlim(0, window)
            ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def figure_deviation(plt, data, settings):
    fig, axes = plt.subplots(len(data), 2, figsize=(13, 3.4 * len(data)), squeeze=False)
    colors = settings["colors"]
    for row, (label, references, old, new) in enumerate(data):
        for col, (name, k) in enumerate((("suction", 0), ("pressure", 1))):
            ax = axes[row, col]
            reference, fit_old, fit_new = references[k], old[k], new[k]
            sample = reference.sample(401)
            u_ref = sample.s / reference.length
            d_old = fit_old.curve.project(sample.points, u_ref)[1]
            d_new = fit_new.curve.project(sample.points, fit_new.parameter(sample.s))[1]
            ax.semilogy(u_ref, np.abs(d_old) + 1e-12, color=colors["old C2"], lw=0.8, label="old C2")
            ax.semilogy(u_ref, np.abs(d_new) + 1e-12, color=colors["robust C2"], lw=1.0, label="robust C2")
            ax.axvspan(0.0, fit_new.leading_edge.anchor / reference.length, color="tab:blue", alpha=0.08,
                       label="LE zone (redesigned)")
            ax.axhline(g.C2_DEVIATION_TARGET, color="0.5", lw=0.6, ls=":")
            ax.set_xscale("symlog", linthresh=0.02)
            ax.set_xlim(0.0, 1.0); ax.set_ylim(1e-11, 3e-3)
            ax.set_title(f"{label}, {name}: |normal distance to C1| / c")
            ax.text(0.98, 0.95, f"LE zone: max {fit_new.le_max_deviation:.2e}, RMS {fit_new.le_rms_deviation:.1e}\n"
                    f"downstream: max {fit_new.max_deviation:.2e}, RMS {fit_new.rms_deviation:.1e}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="0.8"))
            ax.set_xlabel("s / L (symlog)"); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def figure_continuity(plt, settings, variant):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for row, (name, values, fixed) in enumerate((("t/c", np.arange(*settings["continuity_t"]) , 1.2),
                                                 ("cl0", np.arange(*settings["continuity_cl0"]), 0.10))):
        records = {"ss": [], "ps": []}
        for value in values:
            cl0, t = (fixed, float(value)) if name == "t/c" else (float(value), fixed)
            fits = g._naca65_quintic_fits(round(cl0, 6), round(t, 6), variant, THICKNESS)
            for key, fit in (("ss", fits[0]), ("ps", fits[1])):
                le = fit.leading_edge
                records[key].append((value, le.anchor - fit.source.start, math.degrees(le.departure_angle),
                                     fit.le_max_deviation, le.fairness, le.earliest_anchor - fit.source.start))
        for key, style in (("ss", "-o"), ("ps", "-s")):
            data = np.array(records[key])
            for col, (index, ylabel) in enumerate(((1, "anchor past station 0.005 (c)"), (2, "departure (deg)"),
                                                   (3, "max LE deviation (c)"), (4, "R^3 int kappa_s^2 ds"))):
                axes[row, col].plot(data[:, 0], data[:, index], style, ms=2.5, lw=0.9, label=key.upper())
                if index == 1:
                    axes[row, col].plot(data[:, 0], data[:, 5], ":", lw=0.8, label=f"{key.upper()} earliest anchor")
                axes[row, col].set_xlabel(name + (f" (cl0 = {fixed})" if name == "t/c" else f" (t/c = {fixed})"))
                axes[row, col].set_ylabel(ylabel)
        for ax in axes[row]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--variant", default=SETTINGS["variant"], choices=g.PROFILE_VARIANTS)
    parser.add_argument("--save", type=Path, default=None, help="directory for the PNG files")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--continuity", action="store_true", help="also figure 6 (a parameter sweep, slow)")
    args = parser.parse_args()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = []
    print(f"{'blade':28s} {'side':4s} {'R_LE/c':>8s} {'departure':>10s} {'anchor+':>9s} {'F':>6s} "
          f"{'LE max dev':>11s} {'LE rms':>9s} {'down max':>10s}")
    for label, cl0, t in SETTINGS["blades"]:
        references, old, new = build(cl0, t, args.variant)
        data.append((label, references, old, new))
        for name, fit in zip(("SS", "PS"), new):
            le = fit.leading_edge
            print(f"{label:28s} {name:4s} {le.nose.radius:8.5f} {math.degrees(le.departure_angle):9.2f}d "
                  f"{le.anchor - fit.source.start:9.5f} {le.fairness:6.3f} {fit.le_max_deviation:11.3e} "
                  f"{fit.le_rms_deviation:9.2e} {fit.max_deviation:10.3e}")
    stagger = math.radians(SETTINGS["stagger_deg"])
    figures = {
        "le_fairing_1_geometry": figure_geometry(plt, data, SETTINGS),
        "le_fairing_2_turning": figure_turning(plt, data, SETTINGS),
        "le_fairing_3_curvature": figure_curvature(plt, data, SETTINGS),
        "le_fairing_4_dtanbeta_dm": figure_slope(plt, data, SETTINGS, stagger),
        "le_fairing_5_deviation": figure_deviation(plt, data, SETTINGS),
    }
    if args.continuity:
        figures["le_fairing_6_continuity"] = figure_continuity(plt, SETTINGS, args.variant)
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            path = args.save / f"{name}.png"
            fig.savefig(path, dpi=SETTINGS["dpi"])
            print("saved", path)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
