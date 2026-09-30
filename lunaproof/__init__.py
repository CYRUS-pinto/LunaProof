"""
LunaProof — Moon image registration under extreme illumination variation.
Multi-modal, sun-angle and scale-invariant correspondence for Chandrayaan-2 (SIH26166).
Team Maximus2 (ID 185903) | PS SIH26166

Modules
-------
physics      : Lommel–Seeliger renderer + cast-shadow ray marcher
phasecong    : 2D Log-Gabor Phase Congruency (Kovesi, 4 scales × 6 orientations)
match        : SIFT / PC-SIFT / LoFTR matcher, MAGSAC++ verifier, Gini gate v2
cascade      : Bounded 3-Hop Scale Cascade (NMI+PhaseCorr / PC-SIFT / ECC)
dem_io       : LOLA LDEM_64 tile fetcher + downsampler
gis_export   : Cloud-Optimised GeoTIFF + tie-point CSV + JSON telemetry
spice_bridge : PDS4 XML geometry reader + SpiceyPy ephemeris (Tier 2)
"""

__version__ = "3.0.0"
