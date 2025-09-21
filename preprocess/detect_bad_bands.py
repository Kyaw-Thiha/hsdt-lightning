import numpy as np
from typing import Optional, Tuple, Dict


def bad_band_mask(
    cube: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,  # shape (C,), nm or µm
    bbl: Optional[np.ndarray] = None,  # shape (C,), 1=good, 0=bad
    *,
    snr_frac: float = 0.4,  # Rule 2: band SNR must be at least this fraction of median SNR
    floor_frac: float = 0.7,  # Rule 3: fraction of pixels at/below global 1st pct to flag
    corr_abs: float = 0.5,  # Rule 4: min neighbor correlation
    d2_z: float = 3.5,  # Rule 5: z-score threshold on 2nd derivative trough
    deep_frac: float = 0.5,  # Rule 5: band median must also be < deep_frac * global median
    votes_needed: int = 2,  # Voting threshold to label a band as bad
    max_drop_frac: float = 0.30,  # Safety rail: max fraction of bands to drop
    sample_pixels: int = 200_000,  # Subsample for robust, fast statistics
    edge_k: int = 4,  # Rule 7: first/last k bands get “edge boost” if already suspicious
    rng_seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Identify water-absorption / low-quality bands *without* hard-coded band indices.

    INPUTS
    ------
    cube : np.ndarray (H, W, C)
        Hyperspectral image cube (reflectance or scaled reflectance).
    wavelengths : Optional[np.ndarray] (C,)
        Band centers (nm or µm). Used for a soft-gating "water window" bonus.
    bbl : Optional[np.ndarray] (C,)
        ENVI bad-band list; 1=good, 0=bad. If provided, it's treated as one vote.

    DECISION MECHANISM (all rules cast votes; a band is "bad" if votes >= votes_needed)
    -----------------------------------------------------------------------------------
    Rule 1) Header-flag (if available):                bbl == 0
        Rationale: Atmospheric-correction tools mark unusable bands; trust but verify.

    Rule 2) Low-SNR band (global SNR drop):            SNR_b = mean_b / std_b
        Flag if SNR_b < snr_frac * median(SNR).
        Rationale: In water windows, transmittance plummets → low mean, relatively higher noise.

    Rule 3) Near-floor dominance:                      f0_b = P[x ≤ global 1st percentile]
        Flag if f0_b > floor_frac.
        Rationale: A very large fraction of pixels collapse near the global floor in bad bands.

    Rule 4) Poor spectral smoothness (neighbor corr):  r_b = mean(corr(b,b-1), corr(b,b+1))
        Flag if r_b < corr_abs (or missing neighbors use the one present).
        Rationale: Water bands break spectral continuity; correlation with neighbors drops.

    Rule 5) Median-spectrum trough (2nd derivative):   z2_b = |Δ² median_spectrum| / MAD
        After a light 5-pt smoothing, compute the discrete 2nd derivative of the per-band median.
        Flag if z2_b > d2_z AND band median < deep_frac * global median.
        Rationale: Water windows create sharp troughs in the median spectrum at low signal levels.

    Rule 6) Wavelength soft-gating (bonus vote):
        If wavelengths are given (nm or µm) and band lies in classic H₂O ranges
        (~930–970, 1340–1460, 1790–1960 nm), add +1 vote *if* the band already has ≥1 vote.
        Rationale: Physics prior boosts confidence but never acts alone.

    Rule 7) Sensor edge boost:
        First/last edge_k bands often have poor SNR; if such a band already has ≥1 vote, add +1.
        Rationale: Sensor response decays at edges.

    POST-PROCESSING
    ---------------
    • Merge consecutive bad bands; revert singletons unless they are very strong (votes ≥ 4).
    • Safety rail: cap total deletions to max_drop_frac of bands; if exceeded, keep only the
      highest-vote bands up to the cap (AND they must have been voted bad).

    RETURNS
    -------
    bad : np.ndarray (C,), dtype=bool
        True means “drop this band”.
    metrics : dict of np.ndarray
        Diagnostics per band: votes, snr, f0, corr, z2, median, wavelength mask, etc.

    TUNING TIPS
    -----------
    • If you see false positives → raise votes_needed to 3 or increase corr_abs slightly.
    • If you miss obvious water bands → lower corr_abs to 0.4 or increase the wavelength bonus
      by widening the nm windows.
    • Always log `metrics` and visually inspect a few spectra while calibrating thresholds.
    """
    assert cube.ndim == 3, "Image must be H x W x C"
    H, W, C = cube.shape
    eps = 1e-12

    # Sample pixels for robust global stats
    rng = np.random.default_rng(rng_seed)
    N_total = H * W
    n = min(sample_pixels, N_total)
    flat = cube.reshape(-1, C)
    idx = rng.choice(N_total, size=n, replace=False)
    X = flat[idx]  # (n, C)

    votes = np.zeros(C, dtype=int)

    # Rule 1: Header flag (bbl)
    if bbl is not None:
        bbl = np.asarray(bbl).astype(float)
        votes += (bbl < 0.5).astype(int)

    # Robust per-band stats
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0) + eps
    snr = mean / std
    snr_med = np.nanmedian(snr)

    # Rule 2: Low SNR
    rule_snr = snr < snr_frac * snr_med
    votes += rule_snr.astype(int)

    # Global floor percentile (1st pct)
    q01_all = np.nanpercentile(X, 1.0)
    f0 = np.mean(X <= q01_all, axis=0)

    # Rule 3: Near-floor dominance
    rule_floor = f0 > floor_frac
    votes += rule_floor.astype(int)

    # Rule 4: Neighbor correlation
    # Compute correlation with neighbors efficiently
    # (center each column, then dot with shifted columns)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    denom = np.sqrt((np.sum(Xc * Xc, axis=0) + eps))
    corr_prev = np.full(C, np.nan)
    corr_next = np.full(C, np.nan)
    # prev
    num_prev = np.sum(Xc[:, 1:] * Xc[:, :-1], axis=0)
    denom_prev = denom[1:] * denom[:-1]
    corr_prev[1:] = num_prev / (denom_prev + eps)
    # next
    num_next = np.sum(Xc[:, :-1] * Xc[:, 1:], axis=0)
    denom_next = denom[:-1] * denom[1:]
    corr_next[:-1] = num_next / (denom_next + eps)

    # average available neighbor corrs
    corr_band = np.nanmean(np.vstack([corr_prev, corr_next]), axis=0)
    rule_corr = corr_band < corr_abs
    votes += rule_corr.astype(int)

    # Rule 5: Median-spectrum trough (2nd derivative)
    m = np.nanmedian(X, axis=0)
    # light smoothing with 5-pt moving average (reflect padding)
    k = 5
    pad = k // 2
    m_pad = np.pad(m, (pad, pad), mode="reflect")
    kernel = np.ones(k) / k
    m_s = np.convolve(m_pad, kernel, mode="valid")
    # discrete second derivative
    d2 = np.empty(C)
    d2[:] = np.nan
    if C >= 3:
        d2[1:-1] = m_s[2:] - 2 * m_s[1:-1] + m_s[:-2]
    # robust z-score via MAD
    d2_valid = d2[1:-1] if C >= 3 else np.array([np.nan])
    med_d2 = np.nanmedian(d2_valid)
    mad_d2 = np.nanmedian(np.abs(d2_valid - med_d2)) + eps
    z2 = np.abs(d2 - med_d2) / mad_d2
    rule_d2 = (z2 > d2_z) & (m < (deep_frac * np.nanmedian(m)))
    votes += rule_d2.astype(int)

    # Rule 6: Wavelength soft-gating (bonus vote)
    wl_mask = np.zeros(C, dtype=bool)
    if wavelengths is not None:
        wl = np.asarray(wavelengths).astype(float)
        wl_nm = wl * (1000.0 if wl.max() < 100.0 else 1.0)
        win = ((wl_nm > 930) & (wl_nm < 970)) | ((wl_nm > 1340) & (wl_nm < 1460)) | ((wl_nm > 1790) & (wl_nm < 1960))
        wl_mask = win
        # add a bonus *only where some suspicion already exists*
        votes += (win & (votes > 0)).astype(int)

    # Rule 7: Edge-band boost
    edge = np.zeros(C, dtype=bool)
    edge[: min(edge_k, C)] = True
    edge[-min(edge_k, C) :] = True
    votes += (edge & (votes > 0)).astype(int)

    # Initial bad mask by votes
    bad = votes >= votes_needed

    # Post-processing: revert singleton bad bands unless very strong (votes >= 4)
    strong = votes >= 4
    # find runs of True
    marks = np.r_[False, bad, False].astype(int)
    starts = np.where(np.diff(marks) == 1)[0]
    ends = np.where(np.diff(marks) == -1)[0]
    for s, e in zip(starts, ends):
        length = e - s
        if length == 1 and not strong[s]:
            bad[s] = False

    # Safety rail: cap maximum drop fraction
    max_drop = int(np.floor(max_drop_frac * C))
    if bad.sum() > max_drop:
        # Keep only highest-vote bands among those marked bad, up to the cap
        bad_indices = np.where(bad)[0]
        order = bad_indices[np.argsort(-votes[bad_indices])]
        keep = np.zeros_like(bad)
        keep[order[:max_drop]] = True
        bad = bad & keep

    metrics = dict(
        votes=votes,
        snr=snr,
        snr_med=np.array([snr_med]),
        floor_frac_per_band=f0,
        corr_with_neighbors=corr_band,
        median_spectrum=m,
        d2=d2,
        z2=z2,
        wavelength_mask=wl_mask,
        edge_mask=edge,
        initial_bad_by_votes=(votes >= votes_needed),
        final_bad=bad,
    )
    return bad.astype(bool), metrics


# --- Example usage ---
if __name__ == "__main__":
    # Fake demo cube: 100x100 pixels, 200 bands
    H, W, C = 100, 100, 200
    rng = np.random.default_rng(42)
    base = np.linspace(0.2, 0.6, C)
    cube = rng.normal(loc=base, scale=0.02, size=(H, W, C)).astype(np.float32)

    # Inject a synthetic “water window” trough around bands 120–135
    trough = slice(120, 136)
    cube[..., trough] *= 0.1
    cube[..., trough] += rng.normal(0, 0.03, size=(H, W, trough.stop - trough.start))

    wavelengths = np.linspace(400, 2400, C)  # nm

    bad, metrics = bad_band_mask(cube, wavelengths=wavelengths)
    print(f"Bad bands: {np.where(bad)[0].tolist()}")
