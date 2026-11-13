# NACA 65-(12A10)10 cascade

The validation case of the blade-to-blade solver: the cascade that Aungier (2003) shows in
Fig. 5-6, measured by Dunavant et al. (1955), NACA RM L55I08 (β₁ 45°, σ 1.0, α 12.0°,
M₁ 0.650).

| File | Content | Used by |
|---|---|---|
| `Dunavant_Aungier_Fix.csv` | surface Mach number against x/c, digitized: suction and pressure side of the experiment and of Aungier's potential-flow curve. The corrected M₁ 0.65 data (2026-11-13) | `scripts/compare_dunavant_aungier.py` |
| `Dunavant_Aungier_Data.csv` | the earlier digitization in the same layout, superseded by the file above; kept for the record | no script |

Layout of both files: a header row naming four series, a header row `X,Y` for each, then the
points; a series may end before the others.
