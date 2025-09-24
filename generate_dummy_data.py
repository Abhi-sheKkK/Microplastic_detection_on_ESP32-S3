"""
generate_microplastics_dataset.py

Generates synthetic photodiode 1-second-waveform rows (high-res -> downsampled to 200 samples),
adds peaks/noise/drift, extracts engineered features, and saves as CSV.

Dependencies:
  pip install numpy pandas scipy

Usage:
  python generate_microplastics_dataset.py
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew
import os

# -----------------------
# Config (tweak these)
# -----------------------
WINDOW_S = 1.0                # seconds per row
HR_SAMPLING_HZ = 2000         # high-res sampling frequency used for waveform generation (Hz)
DOWNSAMPLED_POINTS = 200      # samples per row after downsampling (ESP32-friendly)
MAX_PARTICLES = 8             # maximum number of microplastics per window (label)
N_ROWS = 3000                 # number of rows to generate
NOISE_STD = 0.02              # background Gaussian noise (on high-res signal)
BASELINE_DRIFT = 0.02         # amplitude of slow baseline drift
MIN_PULSE_WIDTH_MS = 5        # min pulse width in ms (realistic small pulses)
MAX_PULSE_WIDTH_MS = 50       # max pulse width in ms
MIN_PULSE_AMP = 0.2           # min pulse amplitude
MAX_PULSE_AMP = 1.0           # max pulse amplitude
PEAK_DETECT_PROM = 0.08       # min prominence for peak detection (applied to downsampled signal)
OUTPUT_CSV = "synthetic_microplastics_1s_200pts.csv"

# Derived config
HR_NSAMPLES = int(HR_SAMPLING_HZ * WINDOW_S)
DOWN_FACTOR = HR_NSAMPLES // DOWNSAMPLED_POINTS
assert HR_NSAMPLES % DOWNSAMPLED_POINTS == 0, "HR samples must be divisible by downsample points"
TIME_HR = np.linspace(0, WINDOW_S, HR_NSAMPLES, endpoint=False)
DOWNSAMPLED_MS_PER_SAMPLE = (WINDOW_S / DOWNSAMPLED_POINTS) * 1000.0  # ms per downsampled sample

# -----------------------
# Helper functions
# -----------------------
def generate_highres_waveform(num_particles):
    """Generate a high-res photodiode waveform with num_particles Gaussian pulses plus noise and drift."""
    t = TIME_HR
    signal = np.ones_like(t) * 1.0  # baseline ~1.0

    # slow baseline drift (sinusoidal low-freq)
    drift = BASELINE_DRIFT * np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz drift
    signal += drift

    # place pulses at random distinct times
    if num_particles > 0:
        # choose particle centers randomly but allow some overlap
        centers = np.random.uniform(0.02, WINDOW_S - 0.02, size=num_particles)
        for c in centers:
            # pulse width in seconds
            width_ms = np.random.uniform(MIN_PULSE_WIDTH_MS, MAX_PULSE_WIDTH_MS)
            width_s = width_ms / 1000.0
            amp = np.random.uniform(MIN_PULSE_AMP, MAX_PULSE_AMP)
            # Gaussian pulse
            pulse = amp * np.exp(-0.5 * ((t - c) / (width_s / 2.0))**2)
            # optionally add small asymmetric shoulder sometimes
            if np.random.rand() < 0.25:
                pulse += 0.1 * amp * np.exp(-0.5 * ((t - (c + width_s*0.6)) / (width_s))**2)
            signal += pulse

    # additive Gaussian noise
    signal += np.random.normal(0, NOISE_STD, size=signal.shape)

    return signal

def downsample_block_average(hr_signal):
    """Simple block average downsampling from HR to DOWNSAMPLED_POINTS."""
    # reshape to (DOWNSAMPLED_POINTS, DOWN_FACTOR)
    s = hr_signal.reshape(DOWNSAMPLED_POINTS, DOWN_FACTOR)
    return s.mean(axis=1)

def extract_features_from_downsampled(ds_signal):
    """Return engineered features extracted from downsampled signal."""
    # Total energy (area under curve)
    total_energy = np.sum(ds_signal)

    # Peak detection on downsampled signal
    # Use prominence-based detection to avoid tiny noise peaks.
    peaks, props = find_peaks(ds_signal, prominence=PEAK_DETECT_PROM)
    peak_count = len(peaks)

    widths_ms_list = []
    energy_list = []
    skew_list = []
    rise_ms_list = []
    fall_ms_list = []

    if peak_count > 0:
        # compute widths (in samples) using half-prominence width estimate
        results_half = peak_widths(ds_signal, peaks, rel_height=0.5)
        widths_samples = results_half[0]  # widths in samples
        for i, p in enumerate(peaks):
            w_samples = widths_samples[i]
            # convert to ms
            widths_ms_list.append(w_samples * DOWNSAMPLED_MS_PER_SAMPLE)

            # energy around peak: sum in +/- width samples (clamped)
            half_w = max(1, int(round(w_samples // 2)))
            left = max(0, p - half_w)
            right = min(len(ds_signal) - 1, p + half_w)
            energy_list.append(np.sum(ds_signal[left:right+1]))

            # skewness of the neighborhood (peak window)
            neigh = ds_signal[left:right+1]
            if neigh.size >= 3:
                skew_list.append(float(skew(neigh)))
            else:
                skew_list.append(0.0)

            # rise time: samples from 10% to peak
            peak_val = ds_signal[p]
            ten_pct = 0.1 * peak_val
            ninety_pct = 0.9 * peak_val
            # find first index before p where value <= 10%
            left_idx = None
            for idx in range(p, -1, -1):
                if ds_signal[idx] <= ten_pct:
                    left_idx = idx
                    break
            if left_idx is None:
                left_idx = max(0, p - int(np.ceil(w_samples*2)))
            # find index where value rises above 90% (between left_idx and p)
            rise_idx = left_idx
            for idx in range(left_idx, p+1):
                if ds_signal[idx] >= ninety_pct:
                    rise_idx = idx
                    break
            rise_ms_list.append((p - rise_idx) * DOWNSAMPLED_MS_PER_SAMPLE)

            # fall time: samples from peak down to 10%
            right_idx = None
            for idx in range(p, len(ds_signal)):
                if ds_signal[idx] <= ten_pct:
                    right_idx = idx
                    break
            if right_idx is None:
                right_idx = min(len(ds_signal)-1, p + int(np.ceil(w_samples*2)))
            # find index where value falls below 90% after peak
            fall_idx = p
            for idx in range(p, right_idx+1):
                if ds_signal[idx] <= ninety_pct:
                    fall_idx = idx
                    break
            fall_ms_list.append((fall_idx - p) * DOWNSAMPLED_MS_PER_SAMPLE)
    else:
        # No peaks: fill zeros
        widths_ms_list = []
        energy_list = []
        skew_list = []
        rise_ms_list = []
        fall_ms_list = []

    # Aggregate statistics (use 0 when no peaks)
    mean_peak_width_ms = float(np.mean(widths_ms_list)) if len(widths_ms_list) > 0 else 0.0
    mean_peak_energy = float(np.mean(energy_list)) if len(energy_list) > 0 else 0.0
    mean_peak_skew = float(np.mean(skew_list)) if len(skew_list) > 0 else 0.0
    mean_rise_ms = float(np.mean(rise_ms_list)) if len(rise_ms_list) > 0 else 0.0
    mean_fall_ms = float(np.mean(fall_ms_list)) if len(fall_ms_list) > 0 else 0.0

    return {
        "peak_count": int(peak_count),
        "mean_peak_width_ms": mean_peak_width_ms,
        "mean_peak_energy": mean_peak_energy,
        "mean_peak_symmetry": mean_peak_skew,
        "mean_rise_ms": mean_rise_ms,
        "mean_fall_ms": mean_fall_ms,
        "total_energy": float(total_energy)
    }

# -----------------------
# Main loop: generate dataset
# -----------------------
rows = []
for i in range(N_ROWS):
    # sample true number of particles (label) per 1s window
    label = np.random.randint(0, MAX_PARTICLES + 1)

    # generate high-res waveform and downsample
    hr = generate_highres_waveform(label)
    ds = downsample_block_average(hr)  # length DOWNSAMPLED_POINTS

    feats = extract_features_from_downsampled(ds)

    # Build row: downsampled samples + engineered features + label
    row = list(ds.astype(np.float32)) + [
        feats["peak_count"],
        feats["mean_peak_width_ms"],
        feats["mean_peak_energy"],
        feats["mean_peak_symmetry"],
        feats["mean_rise_ms"],
        feats["mean_fall_ms"],
        feats["total_energy"],
        label
    ]
    rows.append(row)

# Column names
v_cols = [f"v{i}" for i in range(DOWNSAMPLED_POINTS)]
feat_cols = [
    "peak_count",
    "mean_peak_width_ms",
    "mean_peak_energy",
    "mean_peak_symmetry",
    "mean_rise_ms",
    "mean_fall_ms",
    "total_energy",
    "label"
]
cols = v_cols + feat_cols

# Save to CSV
df = pd.DataFrame(rows, columns=cols)
os.makedirs("out", exist_ok=True)
outpath = os.path.join("out", OUTPUT_CSV)
df.to_csv(outpath, index=False)
print(f"✅ Dataset saved to: {outpath}")
print("Dataset shape:", df.shape)
print(df[["peak_count", "mean_peak_width_ms", "mean_peak_energy", "mean_peak_symmetry", "mean_rise_ms", "mean_fall_ms", "total_energy", "label"]].describe().T)
