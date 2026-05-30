"""
extract_features.py
Extracts voice biomarkers from a WAV file that match the parkinsons.data dataset features.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False


def extract_voice_features(audio_path: str) -> dict:
    """
    Extract parkinsons-relevant voice features from a WAV file.

    Returns a dict matching the original dataset column names:
        MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz),
        MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ,
        Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB),
        Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA,
        NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa is not installed. Run: pip install librosa")
    if not PARSELMOUTH_AVAILABLE:
        raise RuntimeError("praat-parselmouth is not installed. Run: pip install praat-parselmouth")

    # Load audio
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    if duration < 1.0:
        raise ValueError("Recording too short (< 1 second). Please record at least 3 seconds of sustained vowel sound.")

    # --- Praat-based features ---
    snd = parselmouth.Sound(audio_path)

    # Pitch analysis
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]  # voiced frames only

    if len(pitch_values) == 0:
        raise ValueError("No voiced frames detected. Please record a sustained vowel (e.g., 'ahhh').")

    fo   = float(np.mean(pitch_values))
    fhi  = float(np.max(pitch_values))
    flo  = float(np.min(pitch_values))

    # Jitter measures (local, absolute, rap, ppq5)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)

    jitter_local   = call(point_process, "Get jitter (local)",         0, 0, 0.0001, 0.02, 1.3)
    jitter_abs     = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap     = call(point_process, "Get jitter (rap)",            0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5    = call(point_process, "Get jitter (ppq5)",           0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp     = jitter_rap * 3  # DDP = 3 * RAP by definition

    # Shimmer measures
    shimmer_local  = call([snd, point_process], "Get shimmer (local)",         0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db     = call([snd, point_process], "Get shimmer (local_dB)",      0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3   = call([snd, point_process], "Get shimmer (apq3)",          0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5   = call([snd, point_process], "Get shimmer (apq5)",          0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11  = call([snd, point_process], "Get shimmer (apq11)",         0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda    = shimmer_apq3 * 3

    # Harmonics-to-Noise Ratio
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    nhr = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 0.5  # approximate NHR from HNR

    # --- Librosa-based nonlinear features ---
    # RPDE: Recurrence Period Density Entropy (approximated via spectral entropy)
    S = np.abs(librosa.stft(y))
    S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-10)
    spectral_entropy = float(-np.mean(np.sum(S_norm * np.log(S_norm + 1e-10), axis=0)))
    rpde = np.clip(spectral_entropy / 10.0, 0.0, 1.0)  # normalise to [0,1]

    # DFA: Detrended Fluctuation Analysis
    dfa = _compute_dfa(y)

    # Spread1, Spread2, D2, PPE from pitch series
    spread1, spread2, d2, ppe = _compute_nonlinear_pitch_features(pitch_values)

    features = {
        "MDVP:Fo(Hz)":        fo,
        "MDVP:Fhi(Hz)":       fhi,
        "MDVP:Flo(Hz)":       flo,
        "MDVP:Jitter(%)":     jitter_local * 100,
        "MDVP:Jitter(Abs)":   jitter_abs,
        "MDVP:RAP":           jitter_rap,
        "MDVP:PPQ":           jitter_ppq5,
        "Jitter:DDP":         jitter_ddp,
        "MDVP:Shimmer":       shimmer_local,
        "MDVP:Shimmer(dB)":   shimmer_db,
        "Shimmer:APQ3":       shimmer_apq3,
        "Shimmer:APQ5":       shimmer_apq5,
        "MDVP:APQ":           shimmer_apq11,
        "Shimmer:DDA":        shimmer_dda,
        "NHR":                nhr,
        "HNR":                hnr,
        "RPDE":               rpde,
        "DFA":                dfa,
        "spread1":            spread1,
        "spread2":            spread2,
        "D2":                 d2,
        "PPE":                ppe,
    }

    return features


# ---------------------------------------------------------------------------
# Helper: Detrended Fluctuation Analysis
# ---------------------------------------------------------------------------
def _compute_dfa(signal: np.ndarray, scales: int = 16) -> float:
    """Simplified DFA returning the scaling exponent α."""
    N = len(signal)
    if N < 64:
        return 0.6  # fallback

    # Cumulative sum
    x = np.cumsum(signal - np.mean(signal))

    scale_sizes = np.logspace(np.log10(16), np.log10(N // 4), scales).astype(int)
    scale_sizes = np.unique(scale_sizes)

    flucts = []
    for s in scale_sizes:
        if s < 4:
            continue
        segments = N // s
        if segments == 0:
            continue
        rms_list = []
        for k in range(segments):
            seg = x[k * s:(k + 1) * s]
            t = np.arange(len(seg))
            coef = np.polyfit(t, seg, 1)
            trend = np.polyval(coef, t)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        flucts.append(np.mean(rms_list))

    if len(flucts) < 2:
        return 0.6

    log_s = np.log(scale_sizes[:len(flucts)])
    log_f = np.log(np.array(flucts) + 1e-10)
    alpha = float(np.polyfit(log_s, log_f, 1)[0])
    return float(np.clip(alpha, 0.0, 2.0))


# ---------------------------------------------------------------------------
# Helper: Nonlinear pitch-based features
# ---------------------------------------------------------------------------
def _compute_nonlinear_pitch_features(pitch_values: np.ndarray):
    """
    Compute spread1, spread2, D2, PPE from the voiced pitch series.
    Formulas chosen to closely match the UCI Parkinson's dataset value ranges:
        spread1: -7.96 to -2.43  (log of pitch coefficient of variation)
        spread2:  0.006 to 0.527 (normalised frame-to-frame pitch variation)
        D2:       1.42  to 3.67  (sample entropy proxy of log-pitch)
        PPE:      0.044 to 0.527 (log-ratio pitch period entropy)
    """
    if len(pitch_values) < 10:
        return -5.0, 0.2, 2.0, 0.2

    mean_pitch = np.mean(pitch_values) + 1e-10

    # spread1: log of the coefficient of variation of pitch
    # Healthy voices have tighter pitch → more negative (e.g. -6 to -4)
    # PD voices have looser pitch → less negative (e.g. -4 to -2)
    pitch_norm = pitch_values / mean_pitch
    spread1 = float(np.log(np.std(pitch_norm) + 1e-10))

    # spread2: std of normalised frame-to-frame pitch differences
    # Related to rapid pitch fluctuations
    spread2 = float(np.std(np.diff(pitch_values) / mean_pitch))

    # D2: sample entropy of log-pitch (correlation dimension proxy)
    log_pitch = np.log(pitch_values + 1e-10)
    d2 = _sample_entropy(log_pitch)

    # PPE: Pitch Period Entropy — entropy of log-ratios of consecutive periods
    # More faithful to Little et al. 2009 definition
    periods = 1.0 / pitch_values
    if len(periods) > 1:
        log_ratios = np.abs(np.log(periods[1:] / (periods[:-1] + 1e-10) + 1e-10))
        hist, _ = np.histogram(log_ratios, bins=30, density=True)
        hist = hist / (hist.sum() + 1e-10)
        ppe = float(-np.sum(hist * np.log(hist + 1e-10)) / np.log(30))
    else:
        ppe = 0.2

    return (
        float(np.clip(spread1, -10.0,  0.0)),
        float(np.clip(spread2,   0.0,  1.0)),
        float(np.clip(d2,        0.0,  5.0)),
        float(np.clip(ppe,       0.0,  1.0)),
    )


def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Approximate sample entropy via explicit pairwise comparison."""
    N = len(x)
    if N < 10:
        return 2.0
    # Subsample to keep O(n²) loop fast for long pitch series
    if N > 80:
        step = N // 80
        x = x[::step]
        N = len(x)
    r_abs = r * np.std(x)
    if r_abs == 0:
        return 2.0
    B = 0
    A = 0
    for i in range(N - m - 1):
        for j in range(i + 1, N - m):
            if np.max(np.abs(x[i:i+m] - x[j:j+m])) <= r_abs:
                B += 1
            if np.max(np.abs(x[i:i+m+1] - x[j:j+m+1])) <= r_abs:
                A += 1
    if B == 0 or A == 0:
        return 2.0
    return float(-np.log(A / B))
