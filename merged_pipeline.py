"""
merged_pipeline.py (v1.0)
========================
Project: Quant Trader Lab - Merged Quant Feature Lab (HHT + Cross-Checks + Regimes)

This is a research/analysis pipeline that starts from TRUE HHT (EMD -> IMFs),
adds Hilbert features (phase/frequency) and then merges additional quant
tooling from the repo:

- Per-IMF Hilbert features (dominant cycle IMF)
- Cycle quality/confidence score
  - Rolling FFT cross-check (dominant period + spectral concentration)
  - Wavelet cross-check (time-frequency power concentration)
- OU half-life estimation (adaptive windows / regime hints)
- Chaos / criticality gates (PSR analogue distance, sandpile, Ising)
- Multi-asset context (RMT-cleaned correlations + MST stress proxy)

Outputs are designed to be console logs (and optional video frames).
"""

import os
import sys
import shutil
import time
import argparse
import logging
import warnings
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List, Union
import itertools

# -------------------------------
# EMD backend (REQUIRED for true HHT)
# -------------------------------
EMD_IMPORT_ERROR = None
try:
    import emd  # pip package that exposes `emd.sift.sift`
    EMD_LIB = "emd"
except Exception as e_emd:
    try:
        from PyEMD import EMD as PyEMD_EMD  # pip package PyEMD
        EMD_LIB = "PyEMD"
    except Exception as e_pyemd:
        EMD_LIB = None
        # keep the most relevant error for diagnostics
        EMD_IMPORT_ERROR = e_pyemd if e_pyemd is not None else e_emd


def require_emd_backend() -> None:
    """
    Enforce TRUE EMD/HHT execution.
    If missing, we FAIL FAST (no EMA/EWM fallback allowed).
    """
    if EMD_LIB is None:
        raise ImportError(
            "No EMD backend found. This pipeline runs TRUE HHT only.\n"
            "Install one of the following and retry:\n"
            "  - pip install emd\n"
            "  - pip install PyEMD\n"
            "Then rerun `python hht_pipeline_refactored.py --self-test`.\n"
        ) from EMD_IMPORT_ERROR


import subprocess

# -------------------------------
# SciPy (Hilbert + math utilities)
# -------------------------------
SCIPY_IMPORT_ERROR = None
try:
    from scipy.signal import hilbert as scipy_hilbert
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import minimum_spanning_tree
    SCIPY_OK = True
except Exception as e_scipy:
    SCIPY_OK = False
    SCIPY_IMPORT_ERROR = e_scipy


def require_hilbert_transform() -> None:
    """
    Enforce SciPy-dependent merged features (Hilbert phase/frequency, chaos gate, market context MST).
    """
    if not SCIPY_OK:
        raise ImportError(
            "scipy is required for merged HHT features (Hilbert phase/frequency, chaos gate, market context).\n"
            "Install and retry:\n"
            "  - pip install scipy\n"
        ) from SCIPY_IMPORT_ERROR


# -------------------------------
# Optional deps (wavelets)
# -------------------------------
PYWT_IMPORT_ERROR = None
try:
    import pywt
    PYWT_OK = True
except Exception as e_pywt:
    PYWT_OK = False
    PYWT_IMPORT_ERROR = e_pywt


def require_wavelets() -> None:
    if not PYWT_OK:
        raise ImportError(
            "pywavelets is required for wavelet cross-check features.\n"
            "Install and retry:\n"
            "  - pip install pywavelets\n"
        ) from PYWT_IMPORT_ERROR


def apply_optional_feature_fallbacks(cfg: "Config") -> None:
    """
    Make the pipeline runnable even if optional packages are missing.
    """
    if getattr(cfg, "enable_wavelet_crosscheck", False) and not PYWT_OK:
        logger.warning(
            "pywavelets not installed; disabling wavelet cross-check. "
            "Install with: pip install pywavelets"
        )
        cfg.enable_wavelet_crosscheck = False

# -------------------------------
# Optional deps (video)
# -------------------------------
try:
    import imageio_ffmpeg
    IMAGEIO_FFMPEG_OK = True
except Exception:
    IMAGEIO_FFMPEG_OK = False

try:
    # Try v1 import (active in moviepy < 2.0)
    from moviepy.editor import ImageSequenceClip
    MOVIEPY_OK = True
except ImportError:
    try:
        # Try v2 import (moviepy >= 2.0)
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        MOVIEPY_OK = True
    except ImportError:
        MOVIEPY_OK = False
except Exception:
    MOVIEPY_OK = False


warnings.filterwarnings("ignore")

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("merged_pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    asset: str = "EURUSD=X"
    start_date: str = "2024-06-01"
    end_date: Optional[str] = None
    timeframe: str = "1h"  # '1h' or '4h'
    input_transform: str = "price"  # 'price' or 'log_price'

    # Execution
    execution_price_type: str = "next_open"  # 'next_open', 'next_close'

    # Adaptive Decomposition
    adaptive_imf_split: bool = True
    noise_period_threshold: float = 12.0
    trend_period_threshold: float = 100.0
    amplitude_gate_fraction: float = 0.05
    bucket_stability_lookback: int = 50
    max_bucket_flips: int = 10

    # Hilbert (HHT features)
    use_cycle_phase: bool = True
    hilbert_window: int = 256  # SAFE mode: window length for cycle phase/frequency
    cycle_amp_lookback: int = 200
    cycle_amp_min_quantile: float = 0.30
    long_phase_min: float = -np.pi / 2.0
    long_phase_max: float = 0.0
    short_phase_min: float = np.pi / 2.0
    short_phase_max: float = np.pi

    # Per-IMF Hilbert + confidence (HHT "make it real")
    use_per_imf_hilbert: bool = True
    enable_cycle_confidence: bool = True
    min_cycle_confidence: float = 0.35  # hard trade gate (0..1)

    # Rolling FFT cross-check (cycle quality)
    enable_fft_crosscheck: bool = True
    fft_window: int = 256
    fft_top_n_components: int = 10
    fft_match_tolerance: float = 0.50  # relative freq mismatch tolerance (higher = looser)

    # Wavelet cross-check (cycle quality)
    enable_wavelet_crosscheck: bool = True
    wavelet_window: int = 256
    wavelet_scales_max: int = 64
    wavelet_name: str = "cmor1.5-1.0"
    wavelet_match_tolerance: float = 0.20  # min ratio power@target / total_power

    # OU half-life (adaptive windows)
    enable_ou_adaptive_windows: bool = True
    ou_lookback: int = 600
    ou_half_life_min: int = 20
    ou_half_life_max: int = 2000
    ou_adapt_zscore_mult: float = 2.0
    ou_adapt_amp_mult: float = 3.0
    ou_adapt_slope_mult: float = 1.0

    # Chaos / criticality gates (regime filters)
    enable_chaos_gate: bool = False
    chaos_compute_every_n_bars: int = 20
    psr_dim: int = 3
    psr_tau: int = 12
    psr_lookback_vectors: int = 500
    chaos_distance_threshold: float = 0.15

    enable_sandpile_gate: bool = False
    sandpile_grid_size: int = 30
    sandpile_critical_mass: int = 4
    sandpile_grain_scale: int = 1500
    sandpile_gate_lookback: int = 800
    sandpile_gate_quantile: float = 0.95

    enable_ising_gate: bool = False
    ising_grid_size: int = 20
    ising_steps_per_bar: int = 1
    ising_vol_window: int = 200
    ising_temp_min: float = 0.5
    ising_temp_max: float = 5.0
    ising_gate_lookback: int = 800
    ising_gate_quantile: float = 0.95

    # Multi-asset context (RMT + MST)
    enable_market_context: bool = True
    market_context_window: int = 1200
    market_context_step: int = 200
    market_context_gate_quantile: float = 0.90
    market_context_min_assets: int = 3

    # Backtest Safety
    backtest_safe: bool = False
    rolling_window: int = 600
    endpoint_method: str = "standard"  # standard causal T -> T+1
    warmup_bars: int = 100
    decompose_every_n_bars: int = 1

    # Signal Engine
    trend_slope_lookback: int = 20
    deviation_zscore_lookback: int = 20
    entry_zscore: float = 2.0
    exit_zscore: float = 0.0
    stop_loss_zscore: float = 4.0
    spread_pips: float = 1.0
    pip_size: float = 0.0001
    auto_pip_size: bool = True

    # Probabilistic signal model (optional; default keeps legacy rules)
    signal_model: str = "rules"  # "rules" | "bayes-logit"

    # Bayesian logistic (MAP with L2 prior) on next-bar direction
    bayes_l2: float = 10.0
    bayes_max_iter: int = 25
    bayes_tol: float = 1e-6
    bayes_min_train_samples: int = 200
    bayes_fit_only_when_enabled: bool = False
    bayes_fit_start: Optional[int] = None  # set internally by walk-forward (slice index)
    bayes_fit_end: Optional[int] = None    # set internally by walk-forward (slice index)

    # Probability-to-trade mapping (hysteresis)
    prob_enter_edge: float = 0.10  # enter long if p_up >= 0.5+edge; short if <= 0.5-edge
    prob_exit_edge: float = 0.02   # exit if p_up crosses back inside 0.5±edge
    prob_use_posterior_conf: bool = True
    prob_min_direction_conf: float = 0.55  # in [0.5, 1.0]; uses Laplace approx on logit sign
    prob_use_phase_windows: bool = False   # if True, also enforce hard phase windows (like rules)

    # Stability Filters
    noise_volatility_percentile: float = 90.0
    volatility_quantile_mode: str = "rolling_proxy"
    rolling_proxy_window: int = 2000
    max_reconstruction_error: float = 1e-3
    exit_on_instability: bool = True

    # Video
    fps: int = 30
    duration_sec: int = 20
    theme_dpi: int = 100
    temp_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "temp_frames_merged"))
    output_file: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "Merged_Pipeline_Output.mp4"))

    resample_label: str = "right"
    resample_closed: str = "right"

    theme: Dict[str, str] = field(default_factory=lambda: {
        "BG": "black", "GRID": "#333333", "PRICE": "#FFFFFF",
        "TREND": "#FF9800", "CYCLE": "#00FF00", "NOISE": "#00BFFF",
        "TEXT": "#B0B0B0", "FONT": "sans-serif"
    })


# -------------------------------
# Data Loader
# -------------------------------
class DataLoader:
    @staticmethod
    def _sanitize_intraday_range(
        start_date: str,
        end_date: Optional[str],
        max_days: int = 729
    ) -> Tuple[str, Optional[str]]:
        """
        Yahoo intraday (1h) is limited to ~730 days.
        Make the request safe by:
        - normalizing to tz-aware UTC for arithmetic (prevents tz-naive/tz-aware errors),
        - forcing end_date=None if end is too old (so we fetch "up to now"),
        - clamping start_date to max_days before the effective end,
        - fixing reversed ranges (start >= end) by dropping end_date.
        """
        now_utc = pd.Timestamp.now(tz="UTC")

        def to_utc(ts_like) -> pd.Timestamp:
            ts = pd.Timestamp(ts_like)
            return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

        # If end_date is None => use now
        end_ts = now_utc if end_date is None else to_utc(end_date)
        start_ts = to_utc(start_date)

        # If user provided end_date in the future, cap to now (keeps API stable)
        if end_ts > now_utc:
            logger.warning(f"end_date {end_date} is in the future; capping to now.")
            end_ts = now_utc
            end_date = None  # yfinance: end=None => up to now

        # If end is too old for intraday, force end=None => fetch up to now
        if (now_utc - end_ts) > pd.Timedelta(days=max_days):
            logger.warning(
                f"Intraday end_date {end_date} older than Yahoo intraday limit. "
                f"Forcing end_date=None (download up to now)."
            )
            end_ts = now_utc
            end_date = None

        # If start >= end, drop end_date (otherwise yfinance may return empty)
        if start_ts >= end_ts:
            logger.warning("start_date >= end_date; forcing end_date=None.")
            return start_date, None

        # Clamp start to last max_days relative to effective end_ts
        min_start = end_ts - pd.Timedelta(days=max_days)
        if start_ts < min_start:
            start_date = min_start.date().isoformat()
            logger.warning(f"Clamping start_date to {start_date} due to Yahoo intraday limit.")

        return start_date, end_date

    def _tf_to_pandas_freq(tf: str) -> str:
        # pandas >= 3.0 removed uppercase offset aliases like "H"
        return {"1h": "1h", "4h": "4h"}.get(tf, tf)

    @staticmethod
    def fetch_data(config: Config) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """
        Returns (raw_close, raw_open, transformed_close)
        """
        # IMPORTANT: We always download 1h, even if we resample to 4h.
        download_interval = "1h"

        # Clamp to Yahoo intraday limit
        # Sanitize intraday range to satisfy Yahoo 1h constraints
        safe_start, safe_end = DataLoader._sanitize_intraday_range(
            config.start_date,
            config.end_date,
            max_days=729
        )

        logger.info(f"Fetching {config.asset} ({download_interval} -> {config.timeframe}) from {safe_start}...")
        try:
            df = yf.download(
                config.asset,
                start=safe_start,
                end=safe_end,
                interval=download_interval,
                progress=False,
            )

            if df is None or df.empty:
                logger.error("No data returned from yfinance.")
                return None, None, None

            # ---- Column normalization (handles yfinance MultiIndex reliably) ----
            if isinstance(df.columns, pd.MultiIndex):
                lvl0 = df.columns.get_level_values(0)
                lvl1 = df.columns.get_level_values(1)
                set0, set1 = set(lvl0), set(lvl1)

                # yfinance can include "Adj Close"; we ignore it later anyway
                ohlc = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

                # Case A: (field, ticker)
                if (len(ohlc.intersection(set0)) >= 4) and (config.asset in set1):
                    df = df.xs(config.asset, level=1, axis=1)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    else:
                        df.columns = pd.Index(df.columns)

                # Case B: (ticker, field)
                elif (len(ohlc.intersection(set1)) >= 4) and (config.asset in set0):
                    df = df.xs(config.asset, level=0, axis=1)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(1)
                    else:
                        df.columns = pd.Index(df.columns)

                else:
                    # Fallback flatten (rare layouts)
                    df.columns = ["_".join(map(str, c)) for c in df.columns.to_list()]
            else:
                df.columns = pd.Index(df.columns)

            # Keep only the fields we care about (FX often has no Volume)
            keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[keep_cols].copy()



            # Ensure timezone normalization
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            
            # Deduplicate / sort (yfinance can occasionally return duplicate timestamps)
            df = df[~df.index.duplicated(keep="last")].sort_index()


            # ---- Resample if needed (FX-safe + completeness filter) ----
            tf_freq = DataLoader._tf_to_pandas_freq(config.timeframe) if hasattr(DataLoader, "_tf_to_pandas_freq") else config.timeframe
            if config.timeframe == "4h":
                logger.info("Resampling 1h -> 4h...")

                rule = "4h"
                agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                agg = {k: v for k, v in agg.items() if k in df.columns}

                # Must have OHLC to build candles
                if not {"Open", "High", "Low", "Close"}.issubset(set(df.columns)):
                    logger.error(f"Cannot resample: OHLC missing. Columns={list(df.columns)}")
                    return None, None, None

                df_4h = df.resample(
                    rule,
                    label=config.resample_label,
                    closed=config.resample_closed,
                    origin="start_day",
                ).agg(agg)

                # Completeness filter: require enough 1h samples inside each 4h bin
                expected_per_bin = 4
                close_counts = df["Close"].resample(
                    rule,
                    label=config.resample_label,
                    closed=config.resample_closed,
                    origin="start_day",
                ).count()

                min_required = max(1, int(np.floor(0.75 * expected_per_bin)))  # 3/4 hours
                df_4h = df_4h[close_counts >= min_required]

                # Drop only if OHLC missing (do NOT drop because Volume is NaN)
                df_4h = df_4h.dropna(subset=["Open", "High", "Low", "Close"])

                df = df_4h




            # Minimal column safety
            if "Close" not in df.columns or "Open" not in df.columns:
                logger.error(f"Missing required columns. Have: {list(df.columns)}")
                return None, None, None

            # Missing bars: weekday-only approximation (Mon-Fri)
            actual_weekdays = df[df.index.weekday < 5]
            freq = DataLoader._tf_to_pandas_freq(config.timeframe) if hasattr(DataLoader, "_tf_to_pandas_freq") else config.timeframe
            full_range = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
            expected_weekdays = full_range[full_range.weekday < 5]
            missing_count = len(expected_weekdays) - len(actual_weekdays)
            if len(expected_weekdays) > 0 and (missing_count / len(expected_weekdays) > 0.01):
                logger.warning(f"Missing weekday bars: {missing_count} ({(missing_count/len(expected_weekdays))*100:.1f}%)")

            raw_close = df["Close"].dropna()
            raw_open = df["Open"].dropna()

            common_idx = raw_close.index.intersection(raw_open.index)
            raw_close = raw_close.loc[common_idx].sort_index()
            raw_open = raw_open.loc[common_idx].sort_index()

            if config.input_transform == "log_price":
                transformed = np.log(raw_close)
            else:
                transformed = raw_close.copy()

            logger.info(f"Data ready: {len(raw_close)} bars.")
            return raw_close, raw_open, transformed

        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return None, None, None


# -------------------------------
# Feature Engines (FFT / Wavelet / OU / Regimes / Context)
# -------------------------------
def infer_pip_size(asset: str, default: float = 0.0001) -> float:
    """
    Best-effort pip size inference for Yahoo FX tickers.
    Examples:
      - EURUSD=X -> 0.0001
      - USDJPY=X -> 0.01
    """
    a = (asset or "").upper()
    if "JPY" in a:
        return 0.01
    return float(default)


def _clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(x))))


def _clean_1d(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def compute_fft_metrics(window: np.ndarray, top_n_components: int = 10) -> Dict[str, float]:
    """
    Rolling FFT metrics on a 1D window (detrended):
      - dom_freq (cycles/bar), dom_period (bars)
      - peak_ratio (spectral concentration)
      - spectral_entropy (0..1, lower=more concentrated)
    """
    x = _clean_1d(window)
    n = len(x)
    if n < 16:
        return {"dom_freq": float("nan"), "dom_period": float("nan"), "peak_ratio": 0.0, "entropy": 1.0}

    # detrend (linear)
    t = np.arange(n, dtype=float)
    a, b = np.polyfit(t, x, 1)
    x_d = x - (a * t + b)

    coeffs = np.fft.rfft(x_d)
    freqs = np.fft.rfftfreq(n, d=1.0)
    amps = np.abs(coeffs)

    # exclude DC
    if len(amps) <= 1:
        return {"dom_freq": float("nan"), "dom_period": float("nan"), "peak_ratio": 0.0, "entropy": 1.0}

    amps0 = amps.copy()
    amps0[0] = 0.0
    order = np.argsort(amps0)[::-1]

    dom_freq = float("nan")
    for idx in order[: max(1, int(top_n_components))]:
        f = float(freqs[idx])
        if f > 0:
            dom_freq = f
            break

    dom_period = (1.0 / dom_freq) if (dom_freq == dom_freq and dom_freq > 0) else float("nan")

    s = float(np.sum(amps0))
    peak_ratio = float(np.max(amps0) / s) if s > 0 else 0.0

    p = amps0**2
    ps = float(np.sum(p))
    if ps <= 0:
        entropy = 1.0
    else:
        p = p / ps
        p = p[p > 0]
        ent = -float(np.sum(p * np.log(p)))
        entropy = float(ent / np.log(len(p))) if len(p) > 1 else 0.0

    return {"dom_freq": float(dom_freq), "dom_period": float(dom_period), "peak_ratio": float(peak_ratio), "entropy": float(entropy)}


def compute_wavelet_metrics(
    window: np.ndarray,
    wavelet_name: str,
    scales_max: int,
    target_freq: float,
) -> Dict[str, float]:
    """
    CWT power concentration at target_freq (cycles/bar) on a past-only window.
    Returns:
      - power_ratio: power@target / total_power_at_last_time
      - total_power: sum(power[:, -1])
      - target_freq_used: nearest CWT frequency
    """
    require_wavelets()
    x = _clean_1d(window)
    n = len(x)
    if n < 32:
        return {"power_ratio": 0.0, "total_power": 0.0, "target_freq_used": float("nan")}

    # detrend (linear)
    t = np.arange(n, dtype=float)
    a, b = np.polyfit(t, x, 1)
    x_d = x - (a * t + b)

    scales = np.arange(1, int(scales_max) + 1)
    cwt, freqs = pywt.cwt(x_d, scales, wavelet_name, sampling_period=1.0)
    power = np.abs(cwt) ** 2
    last = power[:, -1]
    total = float(np.sum(last))
    if total <= 0:
        return {"power_ratio": 0.0, "total_power": 0.0, "target_freq_used": float("nan")}

    freqs = np.asarray(freqs, dtype=float)
    if target_freq == target_freq and target_freq > 0:
        k = int(np.argmin(np.abs(freqs - float(target_freq))))
    else:
        k = int(np.argmax(last))

    return {
        "power_ratio": float(last[k] / total),
        "total_power": float(total),
        "target_freq_used": float(freqs[k]),
    }


def estimate_ou_half_life_bars(x: np.ndarray) -> float:
    """
    Estimate OU half-life (in bars) from AR(1) coefficient on a window.
    Uses: phi = cov(x_t, x_{t+1}) / var(x_t), half_life = ln(2)/(-ln(phi))
    """
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 30:
        return float("nan")

    x0 = y[:-1]
    x1 = y[1:]
    x0c = x0 - float(np.mean(x0))
    x1c = x1 - float(np.mean(x1))

    v = float(np.sum(x0c**2))
    if v <= 0:
        return float("nan")

    phi = float(np.sum(x0c * x1c) / v)
    if not (0.0 < phi < 1.0):
        return float("nan")

    return float(math.log(2.0) / (-math.log(phi)))


def compute_ou_half_life_series(
    x: np.ndarray,
    lookback: int,
    every_n: int,
    hl_min: int,
    hl_max: int,
) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    n = len(y)
    out = np.full(n, np.nan, dtype=float)
    L = int(max(30, lookback))
    step = int(max(1, every_n))

    last = np.nan
    for i in range(L - 1, n, step):
        w = y[i - L + 1 : i + 1]
        hl = estimate_ou_half_life_bars(w)
        if hl == hl and np.isfinite(hl):
            last = float(max(hl_min, min(hl_max, hl)))
        out[i] = last

    # forward fill
    s = pd.Series(out).ffill()
    return s.values.astype(float)


def _embed_time_delay(series: np.ndarray, dim: int, tau: int) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    N = len(x)
    M = N - (dim - 1) * tau
    if M <= 0:
        return np.zeros((0, dim), dtype=float)
    emb = np.zeros((M, dim), dtype=float)
    for d in range(dim):
        st = d * tau
        emb[:, d] = x[st : st + M]
    return emb


def psr_analogue_distance(window: np.ndarray, dim: int, tau: int, lookback_vectors: int) -> float:
    """
    "Lyapunov pipeline" style analogue distance:
      - time-delay embed
      - find nearest historical neighbor to the current trajectory start
    Returns NN distance (lower = more "predictable"/repeatable regime).
    """
    x = np.asarray(window, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 200:
        return float("nan")

    # Normalize to focus on shape
    mu = float(np.mean(x))
    sd = float(np.std(x)) + 1e-9
    x = (x - mu) / sd

    vectors = _embed_time_delay(x, int(dim), int(tau))
    if len(vectors) < int(lookback_vectors) * 3:
        return float("nan")

    lookback = int(lookback_vectors)
    subject = vectors[-lookback:]
    safety_buffer = lookback
    search_end = len(vectors) - lookback - safety_buffer
    if search_end < 100:
        return float("nan")

    search_space = vectors[:search_end]
    query = subject[0]

    tree = cKDTree(search_space)
    dist, _ = tree.query(query, k=1)
    return float(dist)


class SandpileSystem:
    def __init__(self, N: int, critical_mass: int = 4):
        self.N = int(N)
        self.critical_mass = int(critical_mass)
        self.grid = np.zeros((self.N, self.N), dtype=int)

    def add_sand(self, amount: int, rng: np.random.Generator) -> None:
        a = int(amount)
        if a <= 0:
            return
        xs = rng.integers(0, self.N, size=a)
        ys = rng.integers(0, self.N, size=a)
        np.add.at(self.grid, (xs, ys), 1)

    def step(self, max_sub_steps: int = 100) -> int:
        avalanche = 0
        for _ in range(int(max_sub_steps)):
            unstable = self.grid >= self.critical_mass
            if not np.any(unstable):
                break
            topple_count = int(np.sum(unstable))
            avalanche += topple_count
            self.grid[unstable] -= self.critical_mass
            mask = unstable.astype(int)
            # open boundaries (sinks)
            self.grid[:-1, :] += mask[1:, :]
            self.grid[1:, :] += mask[:-1, :]
            self.grid[:, :-1] += mask[:, 1:]
            self.grid[:, 1:] += mask[:, :-1]
        return int(avalanche)

    def energy(self) -> int:
        return int(np.sum(self.grid))


class Ising2DSystem:
    def __init__(self, N: int, seed: int = 42):
        self.N = int(N)
        self.rng = np.random.default_rng(int(seed))
        self.lattice = self.rng.choice([-1, 1], size=(self.N, self.N))

    def _neighbor_sum(self, i: int, j: int) -> int:
        N = self.N
        lat = self.lattice
        return int(
            lat[(i + 1) % N, j]
            + lat[(i - 1) % N, j]
            + lat[i, (j + 1) % N]
            + lat[i, (j - 1) % N]
        )

    def metropolis_sweep(self, temp: float, sweeps: int = 1) -> None:
        T = float(max(1e-6, temp))
        N = self.N
        lat = self.lattice
        steps = int(max(1, sweeps) * N * N)
        for _ in range(steps):
            i = int(self.rng.integers(0, N))
            j = int(self.rng.integers(0, N))
            s = int(lat[i, j])
            nb = self._neighbor_sum(i, j)
            dE = 2.0 * s * nb
            if dE <= 0.0 or float(self.rng.random()) < math.exp(-dE / T):
                lat[i, j] = -s

    def magnetization(self) -> float:
        return float(np.mean(self.lattice))


def rmt_denoise_correlation(corr: np.ndarray, T: int) -> Tuple[np.ndarray, float, float]:
    """
    Marchenko–Pastur style eigenvalue filtering.
    Returns (clean_corr, lambda_plus, largest_eig).
    """
    c = np.asarray(corr, dtype=float)
    N = int(c.shape[0])
    if N < 2:
        return c, float("nan"), float("nan")

    T_ = int(max(N + 1, T))
    Q = float(T_ / N)
    lambda_plus = (1.0 + math.sqrt(1.0 / Q)) ** 2 if Q > 1.0 else float("inf")

    evals, evecs = np.linalg.eigh(c)
    largest = float(np.max(evals)) if len(evals) else float("nan")

    noise = evals < lambda_plus
    if np.any(noise):
        avg_noise = float(np.mean(evals[noise]))
        evals = evals.copy()
        evals[noise] = avg_noise

    clean = (evecs @ np.diag(evals) @ evecs.T).astype(float)
    np.fill_diagonal(clean, 1.0)
    np.clip(clean, -1.0, 1.0, out=clean)
    return clean, float(lambda_plus), float(largest)


def mst_metrics_from_corr(corr: np.ndarray) -> Dict[str, float]:
    c = np.asarray(corr, dtype=float)
    N = int(c.shape[0])
    if N < 2:
        return {"mst_total_weight": 0.0, "mst_max_degree": 0.0, "avg_corr": 0.0}

    # distance from correlation
    dist = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - c)))
    np.fill_diagonal(dist, 0.0)
    dist = np.nan_to_num(dist, nan=2.0, posinf=2.0, neginf=2.0)

    mst = minimum_spanning_tree(dist).toarray().astype(float)
    adj = mst + mst.T
    deg = np.count_nonzero(adj > 0, axis=1).astype(float)
    max_deg = float(np.max(deg)) if len(deg) else 0.0
    total_w = float(np.sum(adj) / 2.0)

    off = c[np.triu_indices_from(c, k=1)]
    avg_corr = float(np.mean(off)) if len(off) else 0.0

    return {"mst_total_weight": float(total_w), "mst_max_degree": float(max_deg), "avg_corr": float(avg_corr)}


def compute_regime_features(
    dev: np.ndarray,
    raw_close: np.ndarray,
    index: pd.DatetimeIndex,
    cfg: Config,
    seed: int = 123,
) -> Dict[str, np.ndarray]:
    """
    Computes causal-ish regime features used as trade gates.
    Designed to be computed ONCE per dataset (not per candidate in grid-search).
    """
    n = len(dev)
    out: Dict[str, np.ndarray] = {}

    # --- Chaos gate (PSR analogue distance) ---
    if cfg.enable_chaos_gate:
        dist = np.full(n, np.nan, dtype=float)
        gate = np.full(n, True, dtype=bool)

        # window length must support embedding + history
        min_win = int((cfg.psr_dim - 1) * cfg.psr_tau + cfg.psr_lookback_vectors * 3 + 100)
        win = int(max(400, min_win))
        step = int(max(1, cfg.chaos_compute_every_n_bars))

        last_d = np.nan
        last_g = True
        for i in range(win - 1, n, step):
            w = dev[i - win + 1 : i + 1]
            d = psr_analogue_distance(w, cfg.psr_dim, cfg.psr_tau, cfg.psr_lookback_vectors)
            if d == d and np.isfinite(d):
                last_d = float(d)
                last_g = bool(d <= cfg.chaos_distance_threshold)
            dist[i] = last_d
            gate[i] = last_g

        dist_s = pd.Series(dist, index=index).ffill()
        gate_s = pd.Series(gate, index=index).ffill().fillna(True)
        out["chaos_distance"] = dist_s.values.astype(float)
        out["chaos_gate"] = gate_s.values.astype(bool)

    # --- Sandpile criticality gate ---
    if cfg.enable_sandpile_gate:
        price = np.asarray(raw_close, dtype=float)
        rets = np.diff(np.log(np.maximum(price, 1e-12)), prepend=np.log(max(price[0], 1e-12)))
        stress = np.abs(rets)

        rng = np.random.default_rng(int(seed))
        sim = SandpileSystem(cfg.sandpile_grid_size, cfg.sandpile_critical_mass)
        avalanche = np.zeros(n, dtype=float)
        energy = np.zeros(n, dtype=float)
        grains = np.zeros(n, dtype=float)
        for i in range(n):
            g = int(max(1, float(stress[i]) * float(cfg.sandpile_grain_scale)))
            grains[i] = float(g)
            sim.add_sand(g, rng)
            avalanche[i] = float(sim.step(max_sub_steps=100))
            energy[i] = float(sim.energy())

        aval_s = pd.Series(avalanche, index=index)
        thr = aval_s.rolling(int(max(50, cfg.sandpile_gate_lookback)), min_periods=50).quantile(float(cfg.sandpile_gate_quantile))
        gate = (aval_s <= thr).fillna(True)

        out["sandpile_avalanche"] = avalanche.astype(float)
        out["sandpile_energy"] = energy.astype(float)
        out["sandpile_grains"] = grains.astype(float)
        out["sandpile_gate"] = gate.values.astype(bool)

    # --- Ising susceptibility gate (2D Ising toy regime proxy) ---
    if cfg.enable_ising_gate:
        price = np.asarray(raw_close, dtype=float)
        rets = np.diff(np.log(np.maximum(price, 1e-12)), prepend=np.log(max(price[0], 1e-12)))
        rets_s = pd.Series(rets, index=index)
        vol = rets_s.rolling(int(max(20, cfg.ising_vol_window)), min_periods=20).std().fillna(method="bfill")
        vol_med = float(np.nanmedian(np.abs(vol.values))) if len(vol) else 1.0
        vol_med = vol_med if vol_med > 0 else 1.0
        temp = np.clip((vol.values / vol_med), float(cfg.ising_temp_min), float(cfg.ising_temp_max))

        ising = Ising2DSystem(cfg.ising_grid_size, seed=int(seed))
        mag = np.full(n, np.nan, dtype=float)
        for i in range(n):
            if np.isfinite(temp[i]):
                ising.metropolis_sweep(float(temp[i]), sweeps=int(max(1, cfg.ising_steps_per_bar)))
                mag[i] = float(ising.magnetization())

        mag_s = pd.Series(mag, index=index).ffill()
        look = int(max(50, cfg.ising_gate_lookback))
        susc = (mag_s.rolling(look, min_periods=50).var() / pd.Series(temp, index=index)).replace([np.inf, -np.inf], np.nan)
        thr = susc.rolling(look, min_periods=50).quantile(float(cfg.ising_gate_quantile))
        gate = (susc <= thr).fillna(True)

        out["ising_temp"] = temp.astype(float)
        out["ising_magnetization"] = mag_s.values.astype(float)
        out["ising_susceptibility"] = susc.fillna(method="ffill").fillna(0.0).values.astype(float)
        out["ising_gate"] = gate.values.astype(bool)

    return out


def fetch_multi_close(
    assets: List[str],
    cfg: Config,
    start: Optional[str],
    end: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Fetch multi-asset close data aligned on a shared time index.
    Intended for market context (RMT/MST) features.
    """
    if not assets:
        return None

    # Yahoo intraday safety (always download 1h then resample if needed)
    safe_start, safe_end = DataLoader._sanitize_intraday_range(
        start_date=start or cfg.start_date,
        end_date=end or cfg.end_date,
        max_days=729,
    )

    df = yf.download(
        tickers=assets,
        start=safe_start,
        end=safe_end,
        interval="1h",
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        # common layout: (field, ticker)
        if "Close" in set(df.columns.get_level_values(0)):
            close = df["Close"].copy()
        elif "Close" in set(df.columns.get_level_values(1)):
            close = df.xs("Close", level=1, axis=1).copy()
        else:
            return None
    else:
        # single column fallback
        if "Close" in df.columns:
            close = df[["Close"]].copy()
            close.columns = [assets[0]]
        else:
            return None

    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_convert("UTC").tz_localize(None)
    close = close[~close.index.duplicated(keep="last")].sort_index()

    if cfg.timeframe == "4h":
        close = close.resample(
            "4h",
            label=cfg.resample_label,
            closed=cfg.resample_closed,
            origin="start_day",
        ).last()

    # keep columns in same order as requested
    cols = [a for a in assets if a in close.columns]
    close = close[cols].dropna(how="all")
    return close


def compute_market_context_gate(
    close_df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Computes RMT-cleaned correlation + MST stress proxies on a rolling window.
    Returns a DataFrame indexed like close_df with:
      - market_stress_score
      - market_ok (bool)
      - avg_corr, mst_max_degree, mst_total_weight, lambda_plus, largest_eig
    """
    close = close_df.dropna()
    if close.shape[1] < int(cfg.market_context_min_assets) or len(close) < int(cfg.market_context_window) + 50:
        idx = close_df.index
        return pd.DataFrame(
            {
                "market_stress_score": np.full(len(idx), np.nan),
                "market_ok": np.full(len(idx), True, dtype=bool),
                "avg_corr": np.full(len(idx), np.nan),
                "mst_max_degree": np.full(len(idx), np.nan),
                "mst_total_weight": np.full(len(idx), np.nan),
                "lambda_plus": np.full(len(idx), np.nan),
                "largest_eig": np.full(len(idx), np.nan),
            },
            index=idx,
        )

    rets = np.log(close / close.shift(1)).dropna()
    idx = close_df.index

    score = pd.Series(np.nan, index=rets.index, dtype=float)
    avg_corr_s = pd.Series(np.nan, index=rets.index, dtype=float)
    mst_deg_s = pd.Series(np.nan, index=rets.index, dtype=float)
    mst_w_s = pd.Series(np.nan, index=rets.index, dtype=float)
    lp_s = pd.Series(np.nan, index=rets.index, dtype=float)
    le_s = pd.Series(np.nan, index=rets.index, dtype=float)

    W = int(cfg.market_context_window)
    step = int(max(10, cfg.market_context_step))
    N = int(rets.shape[1])

    for i in range(W, len(rets) + 1, step):
        sub = rets.iloc[i - W : i].dropna()
        if len(sub) < int(0.7 * W):
            continue

        corr = sub.corr().values.astype(float)
        clean, lambda_plus, largest = rmt_denoise_correlation(corr, T=len(sub))
        mst_m = mst_metrics_from_corr(clean)

        # stress score: correlation + topology concentration
        deg_norm = float(mst_m["mst_max_degree"] / max(1.0, (N - 1.0)))
        stress = float(mst_m["avg_corr"] * (1.0 + deg_norm))

        t_end = sub.index[-1]
        score.loc[t_end] = stress
        avg_corr_s.loc[t_end] = float(mst_m["avg_corr"])
        mst_deg_s.loc[t_end] = float(mst_m["mst_max_degree"])
        mst_w_s.loc[t_end] = float(mst_m["mst_total_weight"])
        lp_s.loc[t_end] = float(lambda_plus)
        le_s.loc[t_end] = float(largest)

    score = score.ffill()
    thr = score.expanding(min_periods=10).quantile(float(cfg.market_context_gate_quantile))
    ok = (score <= thr).fillna(True)

    out = pd.DataFrame(
        {
            "market_stress_score": score.reindex(idx, method="ffill").values.astype(float),
            "market_ok": ok.reindex(idx, method="ffill").fillna(True).values.astype(bool),
            "avg_corr": avg_corr_s.ffill().reindex(idx, method="ffill").values.astype(float),
            "mst_max_degree": mst_deg_s.ffill().reindex(idx, method="ffill").values.astype(float),
            "mst_total_weight": mst_w_s.ffill().reindex(idx, method="ffill").values.astype(float),
            "lambda_plus": lp_s.ffill().reindex(idx, method="ffill").values.astype(float),
            "largest_eig": le_s.ffill().reindex(idx, method="ffill").values.astype(float),
        },
        index=idx,
    )
    return out


def enrich_components_with_extras(
    raw_close: pd.Series,
    transformed: pd.Series,
    components: Dict[str, np.ndarray],
    cfg: Config,
    seed: int = 123,
    market_context: Optional[pd.DataFrame] = None,
) -> Dict[str, np.ndarray]:
    """
    Attach extra (merged) features/gates to an existing SAFE/GLOBAL decomposition dict.
    Intended to be called ONCE per dataset (not inside candidate grid loops).
    """
    n = len(transformed)
    if n == 0:
        return components
    if "trend" not in components:
        return components

    trend = np.asarray(components["trend"], dtype=float)
    x = np.asarray(transformed.values, dtype=float)
    if len(trend) != n:
        return components

    dev = x - trend

    if cfg.enable_ou_adaptive_windows and n >= int(max(31, cfg.ou_lookback)):
        components["ou_half_life"] = compute_ou_half_life_series(
            dev,
            lookback=int(cfg.ou_lookback),
            every_n=int(max(1, cfg.decompose_every_n_bars, 5)),
            hl_min=int(cfg.ou_half_life_min),
            hl_max=int(cfg.ou_half_life_max),
        )

    if cfg.enable_chaos_gate or cfg.enable_sandpile_gate or cfg.enable_ising_gate:
        reg = compute_regime_features(
            dev=dev,
            raw_close=np.asarray(raw_close.values, dtype=float),
            index=raw_close.index,
            cfg=cfg,
            seed=int(seed),
        )
        # ensure correct length
        for k, v in reg.items():
            if isinstance(v, np.ndarray) and len(v) == n:
                components[k] = v

    if cfg.enable_market_context and market_context is not None and not market_context.empty:
        ctx = market_context.reindex(raw_close.index).ffill()
        if "market_ok" in ctx.columns:
            components["market_ok"] = ctx["market_ok"].fillna(True).values.astype(bool)
        if "market_stress_score" in ctx.columns:
            components["market_stress_score"] = ctx["market_stress_score"].values.astype(float)

    return components


# -------------------------------
# Decomposer
# -------------------------------
class HHTDecomposer:
    def __init__(self, config: Config):
        self.config = config

    def _hilbert_cycle_features(self, cycle: np.ndarray) -> Dict[str, np.ndarray]:
        require_hilbert_transform()

        cycle = np.asarray(cycle, dtype=float)
        cycle_filled = np.nan_to_num(cycle, nan=0.0, posinf=0.0, neginf=0.0)

        analytic = scipy_hilbert(cycle_filled)
        phase = np.angle(analytic)
        amp = np.abs(analytic)

        # instantaneous frequency (cycles per bar)
        unwrapped = np.unwrap(phase)
        inst_freq = np.full_like(phase, np.nan, dtype=float)
        if len(unwrapped) >= 2:
            inst_freq[1:] = np.diff(unwrapped) / (2.0 * np.pi)

        # respect missing cycle values
        if np.isnan(cycle).any():
            mask = np.isnan(cycle)
            phase = phase.astype(float)
            amp = amp.astype(float)
            phase[mask] = np.nan
            amp[mask] = np.nan
            inst_freq[mask] = np.nan

        return {"cycle_phase": phase, "cycle_amp": amp, "cycle_inst_freq": inst_freq}

    def _hilbert_cycle_features_from_imfs(self, imfs: np.ndarray, cycle_idxs: List[int]) -> Dict[str, np.ndarray]:
        """
        Compute per-IMF Hilbert features for the cycle bucket and return dominant-IMF features.
        This is closer to "true HHT" than applying Hilbert to a multi-frequency sum.
        """
        require_hilbert_transform()

        n_imfs, n = imfs.shape[0], imfs.shape[1]
        if not cycle_idxs:
            nan = np.full(n, np.nan, dtype=float)
            return {
                "cycle_phase": nan.copy(),
                "cycle_amp": nan.copy(),
                "cycle_inst_freq": nan.copy(),
                "cycle_imf_concentration": nan.copy(),
                "cycle_dom_imf_index": np.full(n, -1.0, dtype=float),
            }

        idxs = [i for i in cycle_idxs if 0 <= int(i) < n_imfs]
        if not idxs:
            nan = np.full(n, np.nan, dtype=float)
            return {
                "cycle_phase": nan.copy(),
                "cycle_amp": nan.copy(),
                "cycle_inst_freq": nan.copy(),
                "cycle_imf_concentration": nan.copy(),
                "cycle_dom_imf_index": np.full(n, -1.0, dtype=float),
            }

        cycle_imfs = np.asarray(imfs[idxs], dtype=float)
        cycle_imfs_filled = np.nan_to_num(cycle_imfs, nan=0.0, posinf=0.0, neginf=0.0)

        analytic = scipy_hilbert(cycle_imfs_filled, axis=1)
        phase = np.angle(analytic)
        amp = np.abs(analytic)

        unwrapped = np.unwrap(phase, axis=1)
        inst_freq = np.full_like(phase, np.nan, dtype=float)
        if n >= 2:
            inst_freq[:, 1:] = np.diff(unwrapped, axis=1) / (2.0 * np.pi)

        energy = amp**2
        denom = np.sum(energy, axis=0)
        denom = np.where(denom <= 0.0, np.nan, denom)
        conc = np.nanmax(energy, axis=0) / denom

        dom_local = np.nanargmax(energy, axis=0)
        cols = np.arange(n, dtype=int)
        phase_dom = phase[dom_local, cols].astype(float)
        amp_dom = amp[dom_local, cols].astype(float)
        freq_dom = inst_freq[dom_local, cols].astype(float)
        dom_global = np.asarray([idxs[int(j)] for j in dom_local], dtype=float)

        return {
            "cycle_phase": phase_dom,
            "cycle_amp": amp_dom,
            "cycle_inst_freq": freq_dom,
            "cycle_imf_concentration": conc.astype(float),
            "cycle_dom_imf_index": dom_global.astype(float),
        }

    def _run_emd(self, data: np.ndarray) -> np.ndarray:
        if EMD_LIB == "emd":
            return emd.sift.sift(data).T
        elif EMD_LIB == "PyEMD":
            return PyEMD_EMD().emd(data)
        else:
            raise ValueError("No EMD library found. Please install 'emd' (preferred) or 'PyEMD'.")



    def _classify_imfs_by_speed(self, imfs: np.ndarray, data_len: int) -> Dict[str, List[int]]:
        periods: List[float] = []
        for imf in imfs:
            std_val = float(np.std(imf))
            gate = std_val * self.config.amplitude_gate_fraction

            gated = np.where(np.abs(imf) < gate, 0.0, imf)
            signed = np.sign(gated)  # -1, 0, 1
            # propagate 0 as "last sign"
            signed_prop = pd.Series(signed).replace(0, np.nan).ffill().fillna(0).values
            z_count = len(np.nonzero(np.diff(signed_prop))[0])

            p = data_len / (z_count / 2.0) if z_count > 0 else float(data_len)
            periods.append(p)

        noise, cycle, trend = [], [], []
        for k, p in enumerate(periods):
            if p <= self.config.noise_period_threshold:
                noise.append(k)
            elif p >= self.config.trend_period_threshold:
                trend.append(k)
            else:
                cycle.append(k)

        # guarantees
        if not trend:
            slowest = int(np.argsort(periods)[::-1][0])
            trend = [slowest]
            if slowest in cycle: cycle.remove(slowest)
            if slowest in noise: noise.remove(slowest)

        if not noise and imfs.shape[0] > 1:
            fastest = int(np.argsort(periods)[0])
            noise = [fastest]
            if fastest in cycle: cycle.remove(fastest)
            if fastest in trend: trend.remove(fastest)

        return {"noise": noise, "cycle": cycle, "trend": trend}

    def _extract_components(self, original: np.ndarray, imfs: np.ndarray) -> Dict[str, Union[np.ndarray, Tuple]]:
        n = len(original)
        buckets = self._classify_imfs_by_speed(imfs, n)

        def s(idxs: List[int]) -> np.ndarray:
            return np.sum(imfs[idxs], axis=0) if idxs else np.zeros(n)

        noise = s(buckets["noise"])
        cycle = s(buckets["cycle"])
        trend = s(buckets["trend"])

        recon = noise + cycle + trend
        
        # Treat residual as part of trend (standard EMD practice)
        residual = original - recon
        trend = trend + residual

        # Now this is true numerical reconstruction error (should be ~0)
        error = original - (noise + cycle + trend)

        bucket_sig = (tuple(sorted(buckets["noise"])), tuple(sorted(buckets["cycle"])), tuple(sorted(buckets["trend"])))
        return {
            "noise": noise,
            "cycle": cycle,
            "trend": trend,
            "residual": residual,
            "error": error,
            "buckets": bucket_sig,
            "bucket_idxs": buckets,
        }

    def decompose_global(self, series: pd.Series) -> Dict[str, np.ndarray]:
        data = series.values
        imfs = self._run_emd(data)
        comps = self._extract_components(data, imfs)
        if self.config.use_cycle_phase:
            if self.config.use_per_imf_hilbert:
                cycle_idxs = comps.get("bucket_idxs", {}).get("cycle", [])
                comps.update(self._hilbert_cycle_features_from_imfs(imfs, cycle_idxs))
            else:
                comps.update(self._hilbert_cycle_features(comps["cycle"]))

        # Cycle confidence (rolling, past-only windows; ffill between compute points)
        if self.config.enable_cycle_confidence and self.config.use_cycle_phase:
            n = len(data)
            conf = np.full(n, np.nan, dtype=float)
            fft_dom_period = np.full(n, np.nan, dtype=float)
            fft_peak_ratio = np.full(n, np.nan, dtype=float)
            fft_entropy = np.full(n, np.nan, dtype=float)
            wv_ratio = np.full(n, np.nan, dtype=float)
            wv_total = np.full(n, np.nan, dtype=float)

            cycle_s = np.asarray(comps["cycle"], dtype=float)
            freq_s = np.asarray(comps.get("cycle_inst_freq", np.full(n, np.nan)), dtype=float)
            conc_s = np.asarray(comps.get("cycle_imf_concentration", np.full(n, np.nan)), dtype=float)

            step = int(max(1, self.config.decompose_every_n_bars))
            start_i = int(max(self.config.fft_window, self.config.wavelet_window, 32) - 1)

            last_vals = {
                "conf": np.nan,
                "fft_dom_period": np.nan,
                "fft_peak_ratio": np.nan,
                "fft_entropy": np.nan,
                "wv_ratio": np.nan,
                "wv_total": np.nan,
            }

            for i in range(start_i, n, step):
                hf = float(freq_s[i]) if np.isfinite(freq_s[i]) else float("nan")
                if self.config.enable_fft_crosscheck:
                    cy_win = cycle_s[max(0, i - self.config.fft_window + 1) : i + 1]
                    fft_m = compute_fft_metrics(cy_win, top_n_components=self.config.fft_top_n_components)

                    df = float(fft_m["dom_freq"]) if np.isfinite(fft_m["dom_freq"]) else float("nan")
                    if hf == hf and df == df and df > 0:
                        rel_err = abs(hf - df) / max(df, 1e-12)
                        fft_score = float(math.exp(-rel_err / max(1e-6, float(self.config.fft_match_tolerance))))
                    else:
                        fft_score = 0.5
                else:
                    fft_m = {"dom_freq": float("nan"), "dom_period": float("nan"), "peak_ratio": float("nan"), "entropy": float("nan")}
                    fft_score = 0.5

                if self.config.enable_wavelet_crosscheck:
                    wv_win = cycle_s[max(0, i - self.config.wavelet_window + 1) : i + 1]
                    wv_m = compute_wavelet_metrics(
                        wv_win,
                        wavelet_name=self.config.wavelet_name,
                        scales_max=self.config.wavelet_scales_max,
                        target_freq=hf,
                    )
                    wv_ratio_i = float(wv_m["power_ratio"])
                    wv_total_i = float(wv_m["total_power"])
                    wv_score = float(max(0.0, min(1.0, wv_ratio_i / max(1e-9, float(self.config.wavelet_match_tolerance)))))
                else:
                    wv_ratio_i = float("nan")
                    wv_total_i = float("nan")
                    wv_score = 0.5

                conc = float(conc_s[i]) if np.isfinite(conc_s[i]) else float("nan")
                conc_score = float(max(0.0, min(1.0, conc))) if conc == conc else 0.5

                conf_i = float(0.4 * conc_score + 0.3 * fft_score + 0.3 * wv_score)

                conf[i] = conf_i
                fft_dom_period[i] = float(fft_m["dom_period"])
                fft_peak_ratio[i] = float(fft_m["peak_ratio"])
                fft_entropy[i] = float(fft_m["entropy"])
                wv_ratio[i] = wv_ratio_i
                wv_total[i] = wv_total_i

                last_vals = {
                    "conf": conf_i,
                    "fft_dom_period": fft_dom_period[i],
                    "fft_peak_ratio": fft_peak_ratio[i],
                    "fft_entropy": fft_entropy[i],
                    "wv_ratio": wv_ratio[i],
                    "wv_total": wv_total[i],
                }

            # forward fill
            conf = pd.Series(conf).ffill().values.astype(float)
            fft_dom_period = pd.Series(fft_dom_period).ffill().values.astype(float)
            fft_peak_ratio = pd.Series(fft_peak_ratio).ffill().values.astype(float)
            fft_entropy = pd.Series(fft_entropy).ffill().values.astype(float)
            wv_ratio = pd.Series(wv_ratio).ffill().values.astype(float)
            wv_total = pd.Series(wv_total).ffill().values.astype(float)

            comps.update(
                {
                    "cycle_confidence": conf,
                    "fft_dom_period": fft_dom_period,
                    "fft_peak_ratio": fft_peak_ratio,
                    "fft_entropy": fft_entropy,
                    "wavelet_power_ratio": wv_ratio,
                    "wavelet_total_power": wv_total,
                }
            )
        comps["stability"] = np.ones(len(data), dtype=bool)
        comps["bucket_signature"] = [comps["buckets"]] * len(data)
        return comps

    def decompose_backtest_safe(self, series: pd.Series) -> Dict[str, np.ndarray]:
        logger.info(f"Decomposing SAFE MODE (every {self.config.decompose_every_n_bars} bars)...")
        n = len(series)
        noise = np.full(n, np.nan)
        trend = np.full(n, np.nan)
        cycle = np.full(n, np.nan)
        cycle_phase = np.full(n, np.nan)
        cycle_amp = np.full(n, np.nan)
        cycle_inst_freq = np.full(n, np.nan)
        cycle_imf_concentration = np.full(n, np.nan)
        cycle_dom_imf_index = np.full(n, np.nan)
        cycle_confidence = np.full(n, np.nan)
        fft_dom_period = np.full(n, np.nan)
        fft_peak_ratio = np.full(n, np.nan)
        fft_entropy = np.full(n, np.nan)
        wavelet_power_ratio = np.full(n, np.nan)
        wavelet_total_power = np.full(n, np.nan)
        residual = np.full(n, np.nan)
        error = np.full(n, np.nan)
        stability = np.ones(n, dtype=bool)
        bucket_sigs = [None] * n

        data = series.values
        warmup = max(self.config.warmup_bars, 50)

        last_vals = {
            "noise": np.nan,
            "trend": np.nan,
            "cycle": np.nan,
            "cycle_phase": np.nan,
            "cycle_amp": np.nan,
            "cycle_inst_freq": np.nan,
            "cycle_imf_concentration": np.nan,
            "cycle_dom_imf_index": np.nan,
            "cycle_confidence": np.nan,
            "fft_dom_period": np.nan,
            "fft_peak_ratio": np.nan,
            "fft_entropy": np.nan,
            "wavelet_power_ratio": np.nan,
            "wavelet_total_power": np.nan,
            "residual": np.nan,
            "error": np.nan,
            "sig": None,
            "stable": True,
        }
        bucket_history: List[Tuple] = []

        t0 = time.time()
        for i in range(warmup, n):
            if (i - warmup) % self.config.decompose_every_n_bars != 0:
                noise[i] = last_vals["noise"]
                trend[i] = last_vals["trend"]
                cycle[i] = last_vals["cycle"]
                cycle_phase[i] = last_vals["cycle_phase"]
                cycle_amp[i] = last_vals["cycle_amp"]
                cycle_inst_freq[i] = last_vals["cycle_inst_freq"]
                cycle_imf_concentration[i] = last_vals["cycle_imf_concentration"]
                cycle_dom_imf_index[i] = last_vals["cycle_dom_imf_index"]
                cycle_confidence[i] = last_vals["cycle_confidence"]
                fft_dom_period[i] = last_vals["fft_dom_period"]
                fft_peak_ratio[i] = last_vals["fft_peak_ratio"]
                fft_entropy[i] = last_vals["fft_entropy"]
                wavelet_power_ratio[i] = last_vals["wavelet_power_ratio"]
                wavelet_total_power[i] = last_vals["wavelet_total_power"]
                residual[i] = last_vals["residual"]
                error[i] = last_vals["error"]
                bucket_sigs[i] = last_vals["sig"]
                stability[i] = last_vals["stable"]
                continue

            window_start = max(0, i - self.config.rolling_window + 1)
            window = data[window_start : i + 1]

            imfs = self._run_emd(window)
            comps = self._extract_components(window, imfs)

            noise[i] = comps["noise"][-1]
            trend[i] = comps["trend"][-1]
            cycle[i] = comps["cycle"][-1]
            residual[i] = comps["residual"][-1]
            error[i] = comps["error"][-1]
            bucket_sigs[i] = comps["buckets"]

            if self.config.use_cycle_phase:
                # Causal-ish phase/frequency: compute from a past-only window ending at i.
                # Use a capped window length for speed + reduced edge effects.
                if self.config.use_per_imf_hilbert:
                    feats = self._hilbert_cycle_features_from_imfs(imfs, comps.get("bucket_idxs", {}).get("cycle", []))
                    cycle_phase[i] = float(feats["cycle_phase"][-1])
                    cycle_amp[i] = float(feats["cycle_amp"][-1])
                    cycle_inst_freq[i] = float(feats["cycle_inst_freq"][-1])
                    cycle_imf_concentration[i] = float(feats["cycle_imf_concentration"][-1])
                    cycle_dom_imf_index[i] = float(feats["cycle_dom_imf_index"][-1])
                else:
                    w = int(max(8, self.config.hilbert_window))
                    cycle_win = comps["cycle"][-min(len(comps["cycle"]), w):]
                    feats = self._hilbert_cycle_features(cycle_win)
                    cycle_phase[i] = float(feats["cycle_phase"][-1])
                    cycle_amp[i] = float(feats["cycle_amp"][-1])
                    cycle_inst_freq[i] = float(feats["cycle_inst_freq"][-1])

                # Cross-checks -> confidence score (computed on the same past-only window)
                if self.config.enable_cycle_confidence:
                    hf = float(cycle_inst_freq[i]) if np.isfinite(cycle_inst_freq[i]) else float("nan")

                    if self.config.enable_fft_crosscheck:
                        fft_win = comps["cycle"][-min(len(comps["cycle"]), int(max(32, self.config.fft_window))) :]
                        fft_m = compute_fft_metrics(fft_win, top_n_components=self.config.fft_top_n_components)
                        fft_dom_period[i] = float(fft_m["dom_period"])
                        fft_peak_ratio[i] = float(fft_m["peak_ratio"])
                        fft_entropy[i] = float(fft_m["entropy"])

                        df = float(fft_m["dom_freq"]) if np.isfinite(fft_m["dom_freq"]) else float("nan")
                        if hf == hf and df == df and df > 0:
                            rel_err = abs(hf - df) / max(df, 1e-12)
                            fft_score = float(math.exp(-rel_err / max(1e-6, float(self.config.fft_match_tolerance))))
                        else:
                            fft_score = 0.5
                    else:
                        fft_score = 0.5

                    if self.config.enable_wavelet_crosscheck:
                        wv_win = comps["cycle"][-min(len(comps["cycle"]), int(max(32, self.config.wavelet_window))) :]
                        wv_m = compute_wavelet_metrics(
                            wv_win,
                            wavelet_name=self.config.wavelet_name,
                            scales_max=self.config.wavelet_scales_max,
                            target_freq=hf,
                        )
                        wavelet_power_ratio[i] = float(wv_m["power_ratio"])
                        wavelet_total_power[i] = float(wv_m["total_power"])
                        wv_score = float(
                            max(
                                0.0,
                                min(
                                    1.0,
                                    wavelet_power_ratio[i] / max(1e-9, float(self.config.wavelet_match_tolerance)),
                                ),
                            )
                        )
                    else:
                        wv_score = 0.5

                    conc = float(cycle_imf_concentration[i]) if np.isfinite(cycle_imf_concentration[i]) else float("nan")
                    conc_score = float(max(0.0, min(1.0, conc))) if conc == conc else 0.5
                    cycle_confidence[i] = float(0.4 * conc_score + 0.3 * fft_score + 0.3 * wv_score)

            bucket_history.append(comps["buckets"])
            if len(bucket_history) > self.config.bucket_stability_lookback:
                bucket_history.pop(0)

            flips = sum(bucket_history[k] != bucket_history[k-1] for k in range(1, len(bucket_history)))
            is_stable = flips <= self.config.max_bucket_flips
            stability[i] = is_stable

            last_vals = {
                "noise": noise[i],
                "trend": trend[i],
                "cycle": cycle[i],
                "cycle_phase": cycle_phase[i],
                "cycle_amp": cycle_amp[i],
                "cycle_inst_freq": cycle_inst_freq[i],
                "cycle_imf_concentration": cycle_imf_concentration[i],
                "cycle_dom_imf_index": cycle_dom_imf_index[i],
                "cycle_confidence": cycle_confidence[i],
                "fft_dom_period": fft_dom_period[i],
                "fft_peak_ratio": fft_peak_ratio[i],
                "fft_entropy": fft_entropy[i],
                "wavelet_power_ratio": wavelet_power_ratio[i],
                "wavelet_total_power": wavelet_total_power[i],
                "residual": residual[i],
                "error": error[i],
                "sig": comps["buckets"],
                "stable": is_stable,
            }

            if i % 200 == 0:
                logger.info(f"SafeMode progress: {i}/{n}")

        logger.info(f"Safe mode done in {time.time()-t0:.1f}s")
        out = {
            "noise": noise,
            "trend": trend,
            "cycle": cycle,
            "residual": residual,
            "error": error,
            "stability": stability,
            "bucket_signature": bucket_sigs,
        }
        if self.config.use_cycle_phase:
            out.update(
                {
                    "cycle_phase": cycle_phase,
                    "cycle_amp": cycle_amp,
                    "cycle_inst_freq": cycle_inst_freq,
                    "cycle_imf_concentration": cycle_imf_concentration,
                    "cycle_dom_imf_index": cycle_dom_imf_index,
                }
            )
        if self.config.enable_cycle_confidence:
            out.update(
                {
                    "cycle_confidence": cycle_confidence,
                    "fft_dom_period": fft_dom_period,
                    "fft_peak_ratio": fft_peak_ratio,
                    "fft_entropy": fft_entropy,
                    "wavelet_power_ratio": wavelet_power_ratio,
                    "wavelet_total_power": wavelet_total_power,
                }
            )
        return out


# -------------------------------
# Trading Engine
# -------------------------------
def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _norm_cdf_approx(x: np.ndarray) -> np.ndarray:
    """
    Fast approximation to N(0,1) CDF (Abramowitz-Stegun / Hart approximation).
    Accurate enough for confidence gating; avoids SciPy dependency.
    """
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    xa = np.abs(x)

    t = 1.0 / (1.0 + 0.2316419 * xa)
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    poly = (((a5 * t + a4) * t + a3) * t + a2) * t + a1
    phi = np.exp(-0.5 * xa * xa) / math.sqrt(2.0 * math.pi)
    cdf_pos = 1.0 - phi * poly * t
    return np.where(sign >= 0, cdf_pos, 1.0 - cdf_pos)


def _fit_bayes_logit_map(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 10.0,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Bayesian logistic regression via MAP (ridge/L2 prior) using IRLS.

    Returns:
      - w: (p+1,) weights on standardized features with intercept
      - mu: (p,) feature mean
      - sd: (p,) feature std (>=1e-12)
      - cov: (p+1,p+1) Laplace covariance approx at MAP
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(y) != X.shape[0]:
        raise ValueError("y length must match X rows")

    n, p = X.shape
    if n < 2 or p < 1:
        raise ValueError("Insufficient data for logistic fit")

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0).astype(float)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0).astype(float)

    Xs = (X - mu) / sd
    Xd = np.concatenate([np.ones((n, 1), dtype=float), Xs], axis=1)

    l2 = float(max(0.0, l2))
    pen = np.full(p + 1, l2, dtype=float)
    pen[0] = 0.0  # do not penalize intercept

    w = np.zeros(p + 1, dtype=float)
    H = None
    for _ in range(int(max_iter)):
        eta = Xd @ w
        p_up = _sigmoid_stable(eta)
        W = p_up * (1.0 - p_up)
        W = np.clip(W, 1e-9, None)

        z = eta + (y - p_up) / W
        XtW = Xd.T * W
        H = XtW @ Xd
        H.flat[:: (p + 2)] += pen
        b = XtW @ z

        # jitter for numerical stability
        H.flat[:: (p + 2)] += 1e-10
        w_new = np.linalg.solve(H, b)

        if np.linalg.norm(w_new - w) <= float(tol) * (1.0 + np.linalg.norm(w)):
            w = w_new
            break
        w = w_new

    if H is None:
        raise ValueError("IRLS failed to produce Hessian")

    cov = np.linalg.inv(H)
    return {"w": w, "mu": mu, "sd": sd, "cov": cov}


class TradingSignalEngine:
    @staticmethod
    def generate_signals(
        raw_close: np.ndarray,
        raw_open: np.ndarray,
        transformed_close: np.ndarray,
        components: Dict[str, np.ndarray],
        config: Config,
        index_timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        n = len(raw_close)

        trend = components["trend"]
        noise = components["noise"]
        error = components["error"]
        stability = components["stability"]
        cycle_phase = components.get("cycle_phase", None)
        cycle_amp = components.get("cycle_amp", None)
        cycle_confidence = components.get("cycle_confidence", None)
        ou_half_life = components.get("ou_half_life", None)
        chaos_gate = components.get("chaos_gate", None)
        sandpile_gate = components.get("sandpile_gate", None)
        ising_gate = components.get("ising_gate", None)
        market_ok = components.get("market_ok", None)

        # Adaptive windows (OU half-life)
        W = int(config.trend_slope_lookback)
        amp_lb = int(config.cycle_amp_lookback)
        if config.enable_ou_adaptive_windows and ou_half_life is not None:
            hl = np.asarray(ou_half_life, dtype=float)
            hl_last = float(pd.Series(hl).dropna().iloc[-1]) if np.isfinite(hl).any() else float("nan")
            if hl_last == hl_last and np.isfinite(hl_last):
                W = _clamp_int(config.ou_adapt_slope_mult * hl_last, 10, 200)
                amp_lb = _clamp_int(config.ou_adapt_amp_mult * hl_last, 50, 5000)

        # slope via convolution (valid windows only)
        W = int(max(5, min(W, n - 1)))  # cap to data length
        x = np.arange(W) - (W - 1) / 2.0
        denom = float(np.sum(x**2))
        kernel = x[::-1] / denom

        valid = (~np.isnan(trend)).astype(float)
        valid_count = np.convolve(valid, np.ones(W), mode="valid")
        slope_raw = np.convolve(np.nan_to_num(trend), kernel, mode="valid")

        slope = np.full(n, np.nan)
        ok = np.where(valid_count == W)[0]
        slope[ok + W - 1] = slope_raw[ok]

        # z-score on transformed deviation
        dev = np.asarray(transformed_close - trend, dtype=float)
        dev_s = pd.Series(dev, index=index_timestamps)

        if config.enable_ou_adaptive_windows and ou_half_life is not None:
            hl = np.asarray(ou_half_life, dtype=float)
            if len(hl) != n:
                hl = np.full(n, np.nan, dtype=float)

            alpha = np.where(
                np.isfinite(hl) & (hl > 1.0),
                1.0 - np.exp(-math.log(2.0) / hl),
                np.nan,
            )
            fallback_alpha = 2.0 / (max(5, int(config.deviation_zscore_lookback)) + 1.0)
            alpha = np.nan_to_num(alpha, nan=float(fallback_alpha), posinf=float(fallback_alpha), neginf=float(fallback_alpha)).astype(float)

            ewm_mean = np.full(n, np.nan, dtype=float)
            ewm_var = np.full(n, np.nan, dtype=float)
            last_m = np.nan
            last_v = np.nan
            for t in range(n):
                x_t = float(dev[t]) if np.isfinite(dev[t]) else float("nan")
                if not np.isfinite(x_t):
                    ewm_mean[t] = last_m
                    ewm_var[t] = last_v
                    continue

                a_t = float(alpha[t])
                if not np.isfinite(last_m):
                    last_m = x_t
                    last_v = 0.0
                else:
                    last_m = (1.0 - a_t) * last_m + a_t * x_t
                    err = x_t - last_m
                    last_v = (1.0 - a_t) * (last_v if np.isfinite(last_v) else 0.0) + a_t * (err * err)

                ewm_mean[t] = last_m
                ewm_var[t] = last_v

            denom = np.sqrt(np.maximum(ewm_var, 1e-12))
            z_arr = (dev - ewm_mean) / denom
        else:
            z = (dev_s - dev_s.rolling(config.deviation_zscore_lookback).mean()) / dev_s.rolling(config.deviation_zscore_lookback).std()
            z_arr = z.values

        # causal-ish volatility gate
        noise_s = pd.Series(noise, index=index_timestamps)
        noise_std = noise_s.rolling(30).std()

        if config.volatility_quantile_mode == "expanding":
            vol_thr = noise_std.expanding(min_periods=200).quantile(config.noise_volatility_percentile / 100.0)
        else:
            vol_thr = noise_std.rolling(config.rolling_proxy_window, min_periods=200).quantile(config.noise_volatility_percentile / 100.0)


        # Gate on RESIDUAL *instability* (endpoint behavior), NOT residual level
        residual = components.get("residual", np.zeros_like(error))
        residual_jump = pd.Series(residual, index=index_timestamps).diff().abs()

        # Interpret max_reconstruction_error as max acceptable residual jump in price units
        residual_jump_ok = residual_jump <= config.max_reconstruction_error

        stability_s = pd.Series(stability, index=index_timestamps)
        trade_enabled = (noise_std <= vol_thr) & residual_jump_ok & stability_s

        # --- Extra gates from merged pipelines ---
        if config.enable_cycle_confidence and (cycle_confidence is not None) and (len(cycle_confidence) == n):
            conf_s = pd.Series(np.asarray(cycle_confidence, dtype=float), index=index_timestamps)
            trade_enabled = trade_enabled & (conf_s >= float(config.min_cycle_confidence))

        if config.enable_chaos_gate and (chaos_gate is not None) and (len(chaos_gate) == n):
            trade_enabled = trade_enabled & pd.Series(np.asarray(chaos_gate, dtype=bool), index=index_timestamps)

        if config.enable_sandpile_gate and (sandpile_gate is not None) and (len(sandpile_gate) == n):
            trade_enabled = trade_enabled & pd.Series(np.asarray(sandpile_gate, dtype=bool), index=index_timestamps)

        if config.enable_ising_gate and (ising_gate is not None) and (len(ising_gate) == n):
            trade_enabled = trade_enabled & pd.Series(np.asarray(ising_gate, dtype=bool), index=index_timestamps)

        if config.enable_market_context and (market_ok is not None) and (len(market_ok) == n):
            trade_enabled = trade_enabled & pd.Series(np.asarray(market_ok, dtype=bool), index=index_timestamps)

        enabled_vals = trade_enabled.fillna(False).values

        phase_long_ok = np.ones(n, dtype=bool)
        phase_short_ok = np.ones(n, dtype=bool)
        phase_amp_ok = np.ones(n, dtype=bool)
        if config.use_cycle_phase:
            if cycle_phase is None or cycle_amp is None:
                raise ValueError("use_cycle_phase=True but cycle_phase/cycle_amp not found in components.")

            phase = np.asarray(cycle_phase, dtype=float)
            amp = np.asarray(cycle_amp, dtype=float)

            phase_long_ok = (phase >= config.long_phase_min) & (phase <= config.long_phase_max)
            phase_short_ok = (phase >= config.short_phase_min) & (phase <= config.short_phase_max)

            amp_s = pd.Series(amp, index=index_timestamps)
            amp_thr = amp_s.rolling(int(amp_lb), min_periods=max(50, int(amp_lb) // 4)).quantile(
                config.cycle_amp_min_quantile
            )
        phase_amp_ok = (amp_s >= amp_thr).fillna(False).values


        # execution price series
        exec_price = raw_open if config.execution_price_type == "next_open" else raw_close

        position = np.zeros(n)
        entry_sig = np.zeros(n)
        exit_sig = np.zeros(n)
        forced_exit = np.zeros(n)
        prob_up = np.full(n, np.nan, dtype=float)
        prob_dir_conf = np.full(n, np.nan, dtype=float)

        curr_pos = 0
        start_idx = max(config.warmup_bars, W, config.deviation_zscore_lookback)

        if str(getattr(config, "signal_model", "rules")).lower() == "bayes-logit":
            # --- Bayesian logistic model on next-bar direction (fit on training slice if provided) ---
            next_ret = np.full(n, np.nan, dtype=float)
            if n >= 2:
                next_ret[:-1] = np.diff(exec_price.astype(float))

            # Features at time t (predict y[t] = 1{ret[t]>0})
            phase = np.asarray(cycle_phase, dtype=float) if cycle_phase is not None else np.full(n, np.nan, dtype=float)
            amp = np.asarray(cycle_amp, dtype=float) if cycle_amp is not None else np.full(n, np.nan, dtype=float)
            conf = np.asarray(cycle_confidence, dtype=float) if cycle_confidence is not None else np.full(n, np.nan, dtype=float)

            z_feat = np.clip(np.asarray(z_arr, dtype=float), -8.0, 8.0)
            slope_feat = np.asarray(slope, dtype=float)
            sin_p = np.sin(phase)
            cos_p = np.cos(phase)
            amp_feat = np.log1p(np.abs(amp))
            conf_feat = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

            X_full = np.column_stack([z_feat, slope_feat, sin_p, cos_p, amp_feat, conf_feat]).astype(float)
            y_full = (next_ret > 0.0).astype(float)

            fit_start = int(config.bayes_fit_start) if config.bayes_fit_start is not None else int(start_idx)
            fit_end = int(config.bayes_fit_end) if config.bayes_fit_end is not None else int(n - 1)
            fit_start = int(max(start_idx, min(fit_start, n - 2)))
            fit_end = int(max(fit_start + 1, min(fit_end, n - 1)))

            tr_idx = np.arange(fit_start, fit_end, dtype=int)
            valid = np.isfinite(y_full[tr_idx]) & np.isfinite(X_full[tr_idx]).all(axis=1)
            if bool(config.bayes_fit_only_when_enabled):
                valid = valid & np.asarray(enabled_vals[tr_idx], dtype=bool)
            tr_idx = tr_idx[valid]

            model = None
            if len(tr_idx) >= int(max(50, config.bayes_min_train_samples)):
                try:
                    model = _fit_bayes_logit_map(
                        X_full[tr_idx],
                        y_full[tr_idx],
                        l2=float(config.bayes_l2),
                        max_iter=int(config.bayes_max_iter),
                        tol=float(config.bayes_tol),
                    )
                except Exception as e:
                    logger.warning(f"bayes-logit: fit failed; falling back to rules. err={e}")
                    model = None
            else:
                logger.warning(
                    f"bayes-logit: too few training samples ({len(tr_idx)}) "
                    f"for bayes_min_train_samples={int(config.bayes_min_train_samples)}; falling back to rules."
                )

            if model is not None:
                w = model["w"]
                mu = model["mu"]
                sd = model["sd"]
                cov = model["cov"]

                Xs = (X_full - mu) / sd
                Xd = np.concatenate([np.ones((n, 1), dtype=float), Xs], axis=1)
                m = Xd @ w
                prob_up = _sigmoid_stable(m)

                if bool(config.prob_use_posterior_conf):
                    v = np.einsum("ij,jk,ik->i", Xd, cov, Xd)
                    v = np.clip(v, 1e-12, None)
                    zc = np.abs(m) / np.sqrt(v)
                    prob_dir_conf = _norm_cdf_approx(zc)
                else:
                    prob_dir_conf = np.ones(n, dtype=float)

                enter_edge = float(max(0.0, min(0.49, config.prob_enter_edge)))
                exit_edge = float(max(0.0, min(0.49, config.prob_exit_edge)))
                min_conf = float(max(0.5, min(1.0, config.prob_min_direction_conf)))

                for t in range(start_idx, n - 1):
                    target = curr_pos
                    is_en = bool(enabled_vals[t])
                    if config.exit_on_instability and (not is_en) and curr_pos != 0:
                        target = 0
                        forced_exit[t] = 1
                        exit_sig[t] = 1
                    else:
                        if not is_en:
                            # no new entries when disabled
                            target = curr_pos
                        else:
                            p_t = float(prob_up[t]) if np.isfinite(prob_up[t]) else 0.5
                            c_t = float(prob_dir_conf[t]) if np.isfinite(prob_dir_conf[t]) else 0.5

                            # Optional hard phase windows (defaults off; phase already in features)
                            long_ok = True
                            short_ok = True
                            if bool(config.prob_use_phase_windows):
                                long_ok = bool(phase_long_ok[t])
                                short_ok = bool(phase_short_ok[t])

                            amp_ok = bool(phase_amp_ok[t])

                            if curr_pos == 0:
                                if (c_t >= min_conf) and amp_ok and long_ok and (p_t >= 0.5 + enter_edge):
                                    target = 1
                                    entry_sig[t] = 1
                                elif (c_t >= min_conf) and amp_ok and short_ok and (p_t <= 0.5 - enter_edge):
                                    target = -1
                                    entry_sig[t] = -1
                            elif curr_pos == 1:
                                if p_t <= 0.5 + exit_edge:
                                    target = 0
                                    exit_sig[t] = 1
                                elif np.isfinite(z_arr[t]) and (z_arr[t] < -config.stop_loss_zscore):
                                    target = 0
                                    exit_sig[t] = -1
                            elif curr_pos == -1:
                                if p_t >= 0.5 - exit_edge:
                                    target = 0
                                    exit_sig[t] = 1
                                elif np.isfinite(z_arr[t]) and (z_arr[t] > config.stop_loss_zscore):
                                    target = 0
                                    exit_sig[t] = -1

                    position[t + 1] = target
                    curr_pos = target

                # Fall through to PnL + frame/reporting
            else:
                # If fit fails, continue with legacy rules below.
                pass

        use_rules = (str(getattr(config, "signal_model", "rules")).lower() != "bayes-logit") or (not np.isfinite(prob_up).any())
        if use_rules:
            for t in range(start_idx, n - 1):
                target = curr_pos
                if not (np.isnan(z_arr[t]) or np.isnan(slope[t])):
                    is_en = enabled_vals[t]
                    if config.exit_on_instability and (not is_en) and curr_pos != 0:
                        target = 0
                        forced_exit[t] = 1
                        exit_sig[t] = 1
                    else:
                        if curr_pos == 0 and is_en:
                            if slope[t] > 0 and z_arr[t] < -config.entry_zscore and phase_long_ok[t] and phase_amp_ok[t]:
                                target = 1; entry_sig[t] = 1
                            elif slope[t] < 0 and z_arr[t] > config.entry_zscore and phase_short_ok[t] and phase_amp_ok[t]:
                                target = -1; entry_sig[t] = -1
                        elif curr_pos == 1:
                            if z_arr[t] >= config.exit_zscore:
                                target = 0; exit_sig[t] = 1
                            elif z_arr[t] < -config.stop_loss_zscore:
                                target = 0; exit_sig[t] = -1
                        elif curr_pos == -1:
                            if z_arr[t] <= -config.exit_zscore:
                                target = 0; exit_sig[t] = 1
                            elif z_arr[t] > config.stop_loss_zscore:
                                target = 0; exit_sig[t] = -1

                position[t + 1] = target
                curr_pos = target

                # Keep these columns defined for consistency in outputs.
                if not np.isfinite(prob_up[t]):
                    prob_up[t] = np.nan
                if not np.isfinite(prob_dir_conf[t]):
                    prob_dir_conf[t] = np.nan

        # MTM PnL in execution-price space
        pnl_gross = np.zeros(n)
        pnl_gross[1:] = position[:-1] * np.diff(exec_price)

        half_spread = (config.spread_pips * config.pip_size) / 2.0
        pos_change = np.abs(np.diff(position, prepend=0.0))
        txn_cost = pos_change * half_spread

        pnl_net = pnl_gross - txn_cost
        equity = np.cumsum(pnl_net)

        return pd.DataFrame({
            "raw_close": raw_close,
            "raw_open": raw_open,
            "transformed_close": transformed_close,
            "slope": slope,
            "zscore": z_arr,
            "position": position,
            "entry_signal": entry_sig,
            "exit_signal": exit_sig,
            "forced_exit": forced_exit,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "transaction_cost": txn_cost,
            "equity_curve": equity,
            "trade_enabled": enabled_vals,
            "execution_price": exec_price,
            "noise_volatility": noise_std.values,
            "vol_threshold": vol_thr.values,
            "bucket_signature": components.get("bucket_signature", [None]*n),
            "stability_mask": stability_s.values,
            "cycle_phase": (np.asarray(cycle_phase, dtype=float) if cycle_phase is not None else np.full(n, np.nan)),
            "cycle_amp": (np.asarray(cycle_amp, dtype=float) if cycle_amp is not None else np.full(n, np.nan)),
            "cycle_inst_freq": np.asarray(components.get("cycle_inst_freq", np.full(n, np.nan)), dtype=float),
            "cycle_imf_concentration": np.asarray(components.get("cycle_imf_concentration", np.full(n, np.nan)), dtype=float),
            "cycle_dom_imf_index": np.asarray(components.get("cycle_dom_imf_index", np.full(n, np.nan)), dtype=float),
            "cycle_confidence": np.asarray(components.get("cycle_confidence", np.full(n, np.nan)), dtype=float),
            "prob_up": np.asarray(prob_up, dtype=float),
            "prob_dir_conf": np.asarray(prob_dir_conf, dtype=float),
            "fft_dom_period": np.asarray(components.get("fft_dom_period", np.full(n, np.nan)), dtype=float),
            "fft_peak_ratio": np.asarray(components.get("fft_peak_ratio", np.full(n, np.nan)), dtype=float),
            "fft_entropy": np.asarray(components.get("fft_entropy", np.full(n, np.nan)), dtype=float),
            "wavelet_power_ratio": np.asarray(components.get("wavelet_power_ratio", np.full(n, np.nan)), dtype=float),
            "wavelet_total_power": np.asarray(components.get("wavelet_total_power", np.full(n, np.nan)), dtype=float),
            "ou_half_life": np.asarray(components.get("ou_half_life", np.full(n, np.nan)), dtype=float),
            "chaos_distance": np.asarray(components.get("chaos_distance", np.full(n, np.nan)), dtype=float),
            "chaos_gate": np.asarray(components.get("chaos_gate", np.full(n, True)), dtype=bool),
            "sandpile_avalanche": np.asarray(components.get("sandpile_avalanche", np.full(n, np.nan)), dtype=float),
            "sandpile_gate": np.asarray(components.get("sandpile_gate", np.full(n, True)), dtype=bool),
            "ising_susceptibility": np.asarray(components.get("ising_susceptibility", np.full(n, np.nan)), dtype=float),
            "ising_gate": np.asarray(components.get("ising_gate", np.full(n, True)), dtype=bool),
            "market_stress_score": np.asarray(components.get("market_stress_score", np.full(n, np.nan)), dtype=float),
            "market_ok": np.asarray(components.get("market_ok", np.full(n, True)), dtype=bool),
        }, index=index_timestamps)


# -------------------------------
# Validation / Metrics
# -------------------------------
def _bars_per_year(timeframe: str) -> int:
    # Approximation (weekday-only trading days); good enough for relative comparisons.
    if timeframe == "1h":
        return 24 * 252
    if timeframe == "4h":
        return 6 * 252
    return 252


def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(np.nanmax(dd)) if np.isfinite(dd).any() else 0.0


def _extract_trades(position: np.ndarray, pnl_net: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(position, dtype=float)
    pnl = np.asarray(pnl_net, dtype=float)
    if len(pos) != len(pnl):
        raise ValueError("position and pnl_net must have same length")

    trades = []
    holds = []
    in_trade = False
    entry_idx = None
    for i in range(len(pos)):
        prev = pos[i - 1] if i > 0 else 0.0
        curr = pos[i]

        if (not in_trade) and (prev == 0.0) and (curr != 0.0):
            in_trade = True
            entry_idx = i

        if in_trade and (prev != 0.0) and (curr == 0.0):
            exit_idx = i
            trade_pnl = float(np.nansum(pnl[entry_idx : exit_idx + 1]))
            trades.append(trade_pnl)
            holds.append(exit_idx - entry_idx)
            in_trade = False
            entry_idx = None

    return np.asarray(trades, dtype=float), np.asarray(holds, dtype=float)


def compute_performance_metrics(signals: pd.DataFrame, timeframe: str) -> Dict[str, float]:
    pnl = signals["pnl_net"].fillna(0.0).values.astype(float)
    # Rebase equity within the slice (signals["equity_curve"] may include prior history).
    equity = np.cumsum(pnl).astype(float)
    position = signals["position"].fillna(0.0).values.astype(float)

    bpy = _bars_per_year(timeframe)
    mu = float(np.mean(pnl))
    sig = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = (np.sqrt(bpy) * mu / sig) if sig > 0 else 0.0

    neg = pnl[pnl < 0]
    neg_std = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0
    sortino = (np.sqrt(bpy) * mu / neg_std) if neg_std > 0 else 0.0

    mdd = _max_drawdown(equity)

    dpos = np.diff(position, prepend=0.0)
    turnover_events = int(np.sum(np.abs(dpos) > 0))

    trades_pnl, holds = _extract_trades(position, pnl)
    n_trades = int(len(trades_pnl))
    hit_rate = float(np.mean(trades_pnl > 0)) if n_trades > 0 else 0.0
    pos_sum = float(np.sum(trades_pnl[trades_pnl > 0])) if n_trades > 0 else 0.0
    neg_sum = float(np.sum(trades_pnl[trades_pnl < 0])) if n_trades > 0 else 0.0
    profit_factor = (pos_sum / abs(neg_sum)) if neg_sum < 0 else float("inf") if pos_sum > 0 else 0.0
    avg_trade = float(np.mean(trades_pnl)) if n_trades > 0 else 0.0
    avg_hold = float(np.mean(holds)) if len(holds) > 0 else 0.0

    return {
        "bars": float(len(pnl)),
        "equity_final": _safe_float(equity[-1] if len(equity) else 0.0),
        "sharpe": _safe_float(sharpe),
        "sortino": _safe_float(sortino),
        "max_dd": _safe_float(mdd),
        "turnover_events": float(turnover_events),
        "trades": float(n_trades),
        "hit_rate": _safe_float(hit_rate),
        "profit_factor": _safe_float(profit_factor),
        "avg_trade": _safe_float(avg_trade),
        "avg_hold_bars": _safe_float(avg_hold),
        "mean_pnl_per_bar": _safe_float(mu),
        "std_pnl_per_bar": _safe_float(sig),
    }


def _log_metrics(prefix: str, m: Dict[str, float]) -> None:
    logger.info(
        f"{prefix} | bars={int(m['bars'])} equity={m['equity_final']:.6f} "
        f"Sharpe={m['sharpe']:.2f} Sortino={m['sortino']:.2f} MaxDD={m['max_dd']:.6f} "
        f"trades={int(m['trades'])} hit={m['hit_rate']*100:.1f}% PF={m['profit_factor']:.2f} "
        f"avg_trade={m['avg_trade']:.6g} avg_hold={m['avg_hold_bars']:.1f} turnover_events={int(m['turnover_events'])}"
    )


def _slice_signals(signals: pd.DataFrame, start_i: int, end_i: int) -> pd.DataFrame:
    start_i = int(max(0, start_i))
    end_i = int(min(len(signals), end_i))
    if end_i <= start_i:
        return signals.iloc[0:0].copy()
    return signals.iloc[start_i:end_i].copy()


def _grid_candidates() -> List[Dict[str, float]]:
    # Small, low-variance grid (avoid overfitting by default).
    entry = [1.5, 2.0, 2.5]
    exit_ = [0.0]
    stop = [3.0, 4.0, 5.0]
    amp_q = [0.10, 0.20, 0.30]
    out = []
    for e, x, s, q in itertools.product(entry, exit_, stop, amp_q):
        out.append(
            {
                "entry_zscore": float(e),
                "exit_zscore": float(x),
                "stop_loss_zscore": float(s),
                "cycle_amp_min_quantile": float(q),
            }
        )
    return out


def _grid_candidates_bayes_logit() -> List[Dict[str, float]]:
    # Small grid for bayes-logit hyperparams + probability-to-trade mapping.
    # Keep it compact to avoid overfitting and keep runtime bounded.
    l2 = [0.5, 1.0, 2.0, 5.0, 10.0]
    enter_edge = [0.05, 0.10, 0.15]
    exit_edge = [0.01, 0.02, 0.05]
    min_conf = [0.50, 0.55, 0.60]

    out: List[Dict[str, float]] = []
    for a, b, c, d in itertools.product(l2, enter_edge, exit_edge, min_conf):
        out.append(
            {
                "signal_model": "bayes-logit",
                "bayes_l2": float(a),
                "prob_enter_edge": float(b),
                "prob_exit_edge": float(c),
                "prob_min_direction_conf": float(d),
            }
        )
    return out


def _copy_cfg(cfg: Config, overrides: Dict[str, float]) -> Config:
    d = cfg.__dict__.copy()
    d.update(overrides)
    return Config(**d)


def run_walk_forward_validation(
    raw_close: pd.Series,
    raw_open: pd.Series,
    transformed: pd.Series,
    components: Dict[str, np.ndarray],
    base_cfg: Config,
    timeframe: str,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    min_trades_train: int,
    seed: int,
) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
    """
    Walk-forward:
      - choose params on rolling train window by Sharpe
      - apply to the next test window
      - stitch OOS pnl across windows
    Output: logs only + returns stitched OOS signals (for bootstrap/permutation)
    """
    n = len(transformed)
    if n < (train_bars + test_bars + base_cfg.warmup_bars + 10):
        raise ValueError("Not enough bars for requested walk-forward windows.")

    # Ensure validation is causal.
    if not base_cfg.backtest_safe:
        logger.warning("Validation should use backtest_safe=True. Forcing it on.")
    base_cfg = _copy_cfg(base_cfg, {"backtest_safe": True})

    # Precompute signals for each candidate? Too expensive across folds + candidates.
    # Instead: per fold, run candidate signals on a bounded slice that includes history.
    # Only need enough history to stabilize rolling stats; do NOT require full rolling_proxy_window.
    history = int(
        max(
            base_cfg.warmup_bars,
            base_cfg.trend_slope_lookback,
            base_cfg.deviation_zscore_lookback,
            30,
            250,  # matches min_periods=200 used in volatility thresholding
            base_cfg.cycle_amp_lookback,
            base_cfg.hilbert_window if base_cfg.use_cycle_phase else 0,
        )
        + 10
    )

    is_bayes = str(getattr(base_cfg, "signal_model", "rules")).lower() == "bayes-logit"
    candidates = _grid_candidates_bayes_logit() if is_bayes else _grid_candidates()
    rng = np.random.default_rng(int(seed))

    # OOS stitched container + coverage mask
    oos = pd.DataFrame(index=raw_close.index)
    for col in [
        "pnl_net",
        "equity_curve",
        "position",
        "entry_signal",
        "exit_signal",
        "forced_exit",
        "trade_enabled",
        "execution_price",
        "cycle_phase",
        "cycle_amp",
        "cycle_inst_freq",
        "prob_up",
        "prob_dir_conf",
    ]:
        oos[col] = np.nan
    oos["equity_curve"] = np.nan
    covered = np.zeros(n, dtype=bool)

    fold = 0
    oos_equity = 0.0
    start_anchor = int(max(base_cfg.warmup_bars + history, train_bars + history))
    last_test_end = None

    for anchor in range(start_anchor, n - test_bars - 1, step_bars):
        train_end = anchor
        train_start = train_end - train_bars
        test_start = train_end
        test_end = min(n, test_start + test_bars)

        slice_start = max(0, train_start - history)
        slice_end = test_end

        # local slices
        rc = raw_close.values[slice_start:slice_end]
        ro = raw_open.values[slice_start:slice_end]
        tr = transformed.values[slice_start:slice_end]
        idx = raw_close.index[slice_start:slice_end]
        comps: Dict[str, Union[np.ndarray, List]] = {}
        for k, v in components.items():
            if isinstance(v, np.ndarray):
                comps[k] = v[slice_start:slice_end]
            elif isinstance(v, list):
                # bucket_signature is a list of nested tuples (ragged); keep it as a list
                comps[k] = v[slice_start:slice_end]

        # indexes within slice
        tr_s = train_start - slice_start
        tr_e = train_end - slice_start
        te_s = test_start - slice_start
        te_e = test_end - slice_start

        best = None
        best_score = -1e9
        best_train_metrics = None
        best_cfg = None

        for cand in candidates:
            cfg = _copy_cfg(base_cfg, cand)
            if is_bayes:
                # Fit the probabilistic model on TRAIN only (avoid leakage into the test window).
                cfg.bayes_fit_start = int(tr_s)
                cfg.bayes_fit_end = int(max(tr_s + 1, tr_e - 1))
            sig = TradingSignalEngine.generate_signals(rc, ro, tr, comps, cfg, idx)
            sig_train = _slice_signals(sig, tr_s, tr_e)
            m_train = compute_performance_metrics(sig_train, timeframe)
            if int(m_train["trades"]) < int(min_trades_train):
                continue
            score = float(m_train["sharpe"])
            if score > best_score:
                best_score = score
                best = sig
                best_train_metrics = m_train
                best_cfg = cfg

        if best is None:
            # fallback: choose a random candidate (keeps walk-forward moving)
            cand = candidates[int(rng.integers(0, len(candidates)))]
            best_cfg = _copy_cfg(base_cfg, cand)
            if is_bayes:
                best_cfg.bayes_fit_start = int(tr_s)
                best_cfg.bayes_fit_end = int(max(tr_s + 1, tr_e - 1))
            best = TradingSignalEngine.generate_signals(rc, ro, tr, comps, best_cfg, idx)
            best_train_metrics = compute_performance_metrics(_slice_signals(best, tr_s, tr_e), timeframe)
            logger.warning(f"WF fold {fold}: no candidate met min_trades; using fallback {cand}.")

        sig_test = _slice_signals(best, te_s, te_e)
        m_test = compute_performance_metrics(sig_test, timeframe)

        train_range = f"{raw_close.index[train_start].date()}->{raw_close.index[train_end-1].date()}"
        test_range = f"{raw_close.index[test_start].date()}->{raw_close.index[test_end-1].date()}"
        if is_bayes:
            logger.info(
                f"WF fold {fold} | train[{train_range}] test[{test_range}] | "
                f"best={{l2={best_cfg.bayes_l2}, enter_edge={best_cfg.prob_enter_edge}, exit_edge={best_cfg.prob_exit_edge}, min_conf={best_cfg.prob_min_direction_conf}}}"
            )
        else:
            logger.info(
                f"WF fold {fold} | train[{train_range}] test[{test_range}] | "
                f"best={{entry={best_cfg.entry_zscore}, exit={best_cfg.exit_zscore}, stop={best_cfg.stop_loss_zscore}, amp_q={best_cfg.cycle_amp_min_quantile}}}"
            )
        _log_metrics(f"WF fold {fold} TRAIN", best_train_metrics)
        _log_metrics(f"WF fold {fold} TEST ", m_test)

        # stitch OOS pnl into global series (additive equity)
        if last_test_end is not None and test_start < last_test_end:
            logger.warning("WF windows overlap; stitching may double-count. Consider step_bars >= test_bars.")
        last_test_end = test_end

        oos.loc[sig_test.index, "pnl_net"] = sig_test["pnl_net"].values
        oos.loc[sig_test.index, "position"] = sig_test["position"].values
        oos.loc[sig_test.index, "entry_signal"] = sig_test["entry_signal"].values
        oos.loc[sig_test.index, "exit_signal"] = sig_test["exit_signal"].values
        oos.loc[sig_test.index, "forced_exit"] = sig_test["forced_exit"].values
        oos.loc[sig_test.index, "trade_enabled"] = sig_test["trade_enabled"].values.astype(float)
        oos.loc[sig_test.index, "execution_price"] = sig_test["execution_price"].values
        oos.loc[sig_test.index, "cycle_phase"] = sig_test["cycle_phase"].values
        oos.loc[sig_test.index, "cycle_amp"] = sig_test["cycle_amp"].values
        oos.loc[sig_test.index, "cycle_inst_freq"] = sig_test["cycle_inst_freq"].values
        if "prob_up" in sig_test.columns:
            oos.loc[sig_test.index, "prob_up"] = sig_test["prob_up"].values
        if "prob_dir_conf" in sig_test.columns:
            oos.loc[sig_test.index, "prob_dir_conf"] = sig_test["prob_dir_conf"].values
        covered[test_start:test_end] = True

        # update stitched equity
        pnl_seg = sig_test["pnl_net"].fillna(0.0).values.astype(float)
        eq_seg = oos_equity + np.cumsum(pnl_seg)
        oos.loc[sig_test.index, "equity_curve"] = eq_seg
        oos_equity = float(eq_seg[-1]) if len(eq_seg) else oos_equity

        fold += 1

    # evaluate only on covered OOS regions
    if not covered.any():
        raise ValueError("Walk-forward produced no OOS windows; reduce train/test/history or fetch more data.")

    oos_eval = oos.iloc[np.where(covered)[0]].copy()
    # rebuild equity just for reporting consistency
    oos_eval["equity_curve"] = np.cumsum(oos_eval["pnl_net"].fillna(0.0).values.astype(float))

    oos_m = compute_performance_metrics(oos_eval, timeframe)
    _log_metrics("WALK-FORWARD OOS", oos_m)

    return {"oos_signals": oos, "oos_eval": oos_eval, "oos_metrics": oos_m}


def bootstrap_sharpe_pvalue(
    pnl: np.ndarray,
    timeframe: str,
    reps: int = 500,
    block_len: int = 24,
    seed: int = 123,
) -> Dict[str, float]:
    """
    Moving-block bootstrap on per-bar pnl to estimate Sharpe distribution.
    Returns one-sided p-value for Sharpe > 0 and a 95% CI.
    """
    x = np.asarray(pnl, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(x)
    if n < 10:
        return {"sharpe_obs": 0.0, "p_sharpe_gt0": 1.0, "sharpe_ci_lo": 0.0, "sharpe_ci_hi": 0.0}

    bpy = _bars_per_year(timeframe)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    sharpe_obs = (np.sqrt(bpy) * mu / sd) if sd > 0 else 0.0

    b = int(max(2, min(block_len, n)))
    rng = np.random.default_rng(int(seed))
    sharpes = np.zeros(int(reps), dtype=float)

    starts = np.arange(0, n - b + 1)
    for r in range(int(reps)):
        sample = []
        while len(sample) < n:
            s = int(rng.choice(starts))
            sample.extend(x[s : s + b].tolist())
        samp = np.asarray(sample[:n], dtype=float)
        mu_r = float(np.mean(samp))
        sd_r = float(np.std(samp, ddof=1)) if n > 1 else 0.0
        sharpes[r] = (np.sqrt(bpy) * mu_r / sd_r) if sd_r > 0 else 0.0

    p_gt0 = float(np.mean(sharpes <= 0.0))
    lo, hi = np.quantile(sharpes, [0.025, 0.975])
    return {
        "sharpe_obs": float(sharpe_obs),
        "p_sharpe_gt0": float(1.0 - p_gt0),
        "sharpe_ci_lo": float(lo),
        "sharpe_ci_hi": float(hi),
    }


def permutation_position_alignment_test(
    execution_price: np.ndarray,
    position: np.ndarray,
    spread_pips: float,
    pip_size: float,
    timeframe: str,
    reps: int = 500,
    seed: int = 123,
) -> Dict[str, float]:
    """
    Circularly shift positions relative to returns (destroys timing) and compare Sharpe.
    One-sided p-value: P(Sharpe_perm >= Sharpe_obs).
    """
    price = np.asarray(execution_price, dtype=float)
    pos = np.asarray(position, dtype=float)
    n = min(len(price), len(pos))
    if n < 10:
        return {"sharpe_obs": 0.0, "p_perm_ge_obs": 1.0}

    price = price[:n]
    pos = pos[:n]
    ret = np.diff(price, prepend=price[0])

    def sharpe_from_pos(p: np.ndarray) -> float:
        pnl_gross = np.zeros(n, dtype=float)
        pnl_gross[1:] = p[:-1] * np.diff(price)
        half_spread = (spread_pips * pip_size) / 2.0
        txn_cost = np.abs(np.diff(p, prepend=0.0)) * half_spread
        pnl = pnl_gross - txn_cost
        bpy = _bars_per_year(timeframe)
        mu = float(np.mean(pnl))
        sd = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
        return (np.sqrt(bpy) * mu / sd) if sd > 0 else 0.0

    sharpe_obs = sharpe_from_pos(pos)

    rng = np.random.default_rng(int(seed))
    sharpes = np.zeros(int(reps), dtype=float)
    for r in range(int(reps)):
        lag = int(rng.integers(1, n))
        p_shift = np.roll(pos, lag)
        sharpes[r] = sharpe_from_pos(p_shift)

    p_val = float(np.mean(sharpes >= sharpe_obs))
    return {"sharpe_obs": float(sharpe_obs), "p_perm_ge_obs": float(p_val)}


def run_validation_suite(
    assets: List[str],
    timeframes: List[str],
    base_cfg: Config,
    start: Optional[str],
    end: Optional[str],
    train_bars: int,
    test_bars: int,
    step_bars: int,
    decompose_every: int,
    rolling_window: int,
    min_trades_train: int,
    bootstrap_reps: int,
    bootstrap_block: int,
    perm_reps: int,
    seed: int,
) -> int:
    """
    Runs:
      - SAFE decomposition (causal-ish)
      - walk-forward parameter selection
      - OOS performance report
      - bootstrap + permutation sanity checks
    Logs only, returns exit code.
    """
    require_emd_backend()
    if base_cfg.use_cycle_phase or base_cfg.enable_chaos_gate or base_cfg.enable_market_context:
        require_hilbert_transform()
    apply_optional_feature_fallbacks(base_cfg)

    rc_global = 0

    for tf in timeframes:
        market_ctx = None
        if base_cfg.enable_market_context and len(assets) >= int(base_cfg.market_context_min_assets):
            try:
                ctx_cfg = _copy_cfg(base_cfg, {"timeframe": tf})
                if start:
                    ctx_cfg.start_date = start
                if end:
                    ctx_cfg.end_date = end
                close_df = fetch_multi_close(assets, ctx_cfg, start=start, end=end)
                if close_df is None or close_df.empty:
                    logger.warning("Market context: multi-asset close fetch failed; continuing without context gate.")
                else:
                    market_ctx = compute_market_context_gate(close_df, ctx_cfg)
                    ok_pct = float(pd.Series(market_ctx["market_ok"]).mean() * 100.0)
                    logger.info(f"Market context | timeframe={tf} assets={close_df.shape[1]} ok%={ok_pct:.1f}%")
            except Exception as e:
                logger.warning(f"Market context failed: {e}")
                market_ctx = None

        for asset in assets:
            cfg = _copy_cfg(
                base_cfg,
                {
                    "asset": asset,
                    "timeframe": tf,
                    "backtest_safe": True,
                    "decompose_every_n_bars": int(decompose_every),
                    "rolling_window": int(rolling_window),
                },
            )
            if start:
                cfg.start_date = start
            if end:
                cfg.end_date = end
            if cfg.auto_pip_size:
                cfg.pip_size = infer_pip_size(cfg.asset, cfg.pip_size)

            logger.info("=" * 80)
            logger.info(f"VALIDATION START | asset={asset} timeframe={tf} | pandas={pd.__version__} numpy={np.__version__}")
            logger.info(f"SAFE decompose_every={cfg.decompose_every_n_bars} rolling_window={cfg.rolling_window} hilbert_window={cfg.hilbert_window}")

            dl = DataLoader()
            raw_close, raw_open, transformed = dl.fetch_data(cfg)
            if raw_close is None:
                logger.error("Validation: data load failed.")
                rc_global = 1
                continue

            decomp = HHTDecomposer(cfg)
            comps = decomp.decompose_backtest_safe(transformed)
            enrich_components_with_extras(raw_close, transformed, comps, cfg, seed=int(seed), market_context=market_ctx)

            # walk-forward
            try:
                wf = run_walk_forward_validation(
                    raw_close=raw_close,
                    raw_open=raw_open,
                    transformed=transformed,
                    components=comps,
                    base_cfg=cfg,
                    timeframe=tf,
                    train_bars=int(train_bars),
                    test_bars=int(test_bars),
                    step_bars=int(step_bars),
                    min_trades_train=int(min_trades_train),
                    seed=int(seed),
                )
            except Exception as e:
                logger.error(f"Walk-forward failed: {e}")
                rc_global = 1
                continue

            oos = wf["oos_signals"]
            oos_eval = wf["oos_eval"]
            oos_m = wf["oos_metrics"]

            # bootstrap Sharpe on OOS pnl
            boot = bootstrap_sharpe_pvalue(
                pnl=oos_eval["pnl_net"].values,
                timeframe=tf,
                reps=int(bootstrap_reps),
                block_len=int(bootstrap_block),
                seed=int(seed) + 11,
            )
            logger.info(
                f"BOOTSTRAP Sharpe | obs={boot['sharpe_obs']:.2f} 95%CI=[{boot['sharpe_ci_lo']:.2f},{boot['sharpe_ci_hi']:.2f}] "
                f"p(Sharpe>0)={boot['p_sharpe_gt0']:.3f} reps={bootstrap_reps} block={bootstrap_block}"
            )

            # permutation / alignment
            perm = permutation_position_alignment_test(
                execution_price=oos_eval["execution_price"].values,
                position=oos_eval["position"].values,
                spread_pips=cfg.spread_pips,
                pip_size=cfg.pip_size,
                timeframe=tf,
                reps=int(perm_reps),
                seed=int(seed) + 29,
            )
            logger.info(f"PERMUTE timing | Sharpe_obs={perm['sharpe_obs']:.2f} p_perm_ge_obs={perm['p_perm_ge_obs']:.3f} reps={perm_reps}")

            logger.info(f"VALIDATION END | asset={asset} timeframe={tf} | OOS Sharpe={oos_m['sharpe']:.2f} MaxDD={oos_m['max_dd']:.6f}")

    return int(rc_global)


# -------------------------------
# Rendering + Video
# -------------------------------
class FrameRenderer:
    @staticmethod
    def render_worker(args: Tuple) -> bool:
        f_idx, cut_idx, full_price, comps, cfg, limits, signals_df = args
        try:
            dates = full_price.index[:cut_idx]
            price = full_price.values[:cut_idx]
            trend = comps["trend"][:cut_idx]
            noise = comps["noise"][:cut_idx]
            cycle = comps["cycle"][:cut_idx]
            sig = signals_df.iloc[:cut_idx]

            plt.style.use("dark_background")
            plt.rcParams["font.family"] = cfg.theme["FONT"]

            fig = plt.figure(figsize=(12, 12), facecolor=cfg.theme["BG"])
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.15)
            ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]
            theme = cfg.theme

            for ax in (ax1, ax2, ax3):
                ax.set_facecolor(theme["BG"])
                ax.grid(True, color=theme["GRID"], ls=":", lw=0.5)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.tick_params(colors=theme["TEXT"])
                if ax is not ax3:
                    plt.setp(ax.get_xticklabels(), visible=False)

            ax1.set_xlim(full_price.index[0], full_price.index[-1])
            ax1.set_ylim(limits["p_min"], limits["p_max"])
            ax2.set_ylim(limits["n_min"], limits["n_max"])
            ax3.set_ylim(limits["c_min"], limits["c_max"])

            ax1.plot(dates, price, color=theme["PRICE"], lw=1)
            ax1.plot(dates, trend, color=theme["TREND"], lw=2)


            price_s = pd.Series(price, index=dates)

            longs_idx = sig.index[sig["entry_signal"] == 1]
            shorts_idx = sig.index[sig["entry_signal"] == -1]

            if len(longs_idx) > 0:
                x = longs_idx.intersection(price_s.index)
                ax1.scatter(x, price_s.loc[x].values, marker="^", c="#00FF00", s=80, zorder=5)

            if len(shorts_idx) > 0:
                x = shorts_idx.intersection(price_s.index)
                ax1.scatter(x, price_s.loc[x].values, marker="v", c="#FF0000", s=80, zorder=5)


            eq = float(sig["equity_curve"].iloc[-1]) if len(sig) else 0.0
            mode = "SAFE" if cfg.backtest_safe else "GLOBAL"
            ax1.text(0.02, 0.92, f"{cfg.asset} | {mode} | Equity: {eq:.4f}",
                     transform=ax1.transAxes, color="white", fontsize=12, fontweight="bold")

            ax2.plot(dates, noise, color=theme["NOISE"], lw=0.8)
            ax2.text(0.02, 0.88, "NOISE", transform=ax2.transAxes, color=theme["NOISE"], fontweight="bold")

            ax3.plot(dates, cycle, color=theme["CYCLE"], lw=1.5)
            ax3.text(0.02, 0.88, "CYCLES", transform=ax3.transAxes, color=theme["CYCLE"], fontweight="bold")
            ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator()))

            out_path = os.path.join(cfg.temp_dir, f"frame_{f_idx:04d}.png")
            fig.savefig(out_path, dpi=cfg.theme_dpi, facecolor=theme["BG"])
            plt.close(fig)
            return True
        except Exception as e:
            logger.error(f"Render Error frame {f_idx}: {e}")
            return False


class RenderManager:
    @staticmethod
    def run(price_series: pd.Series, components: Dict, signals_df: pd.DataFrame, config: Config):
        if os.path.exists(config.temp_dir):
            shutil.rmtree(config.temp_dir)
        os.makedirs(config.temp_dir, exist_ok=True)

        def get_lims(arr: np.ndarray) -> Tuple[float, float]:
            if np.all(np.isnan(arr)):
                return -1.0, 1.0
            mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
            pad = (mx - mn) * 0.1 if mx > mn else 1.0
            return mn - pad, mx + pad


        p_min, p_max = get_lims(price_series.values)
        n_min, n_max = get_lims(components["noise"])
        c_min, c_max = get_lims(components["cycle"])

        limits = {"p_min": p_min, "p_max": p_max, "n_min": n_min, "n_max": n_max, "c_min": c_min, "c_max": c_max}

        total_frames = int(config.duration_sec * config.fps)
        if len(price_series) < (config.warmup_bars + 60):
            logger.warning("Dataset short vs warmup; reducing frames.")
            start = max(0, int(len(price_series) * 0.2))
        else:
            start = min(len(price_series) - 2, config.warmup_bars + 50)

        end = max(start + 1, len(price_series) - 1)
        idxs = np.linspace(start, end, total_frames, dtype=int)

        tasks = [(i, idx, price_series, components, config, limits, signals_df) for i, idx in enumerate(idxs)]
        logger.info(f"Rendering {len(tasks)} frames...")
        with Pool(processes=max(1, cpu_count() - 2)) as pool:
            pool.map(FrameRenderer.render_worker, tasks)


class VideoCompiler:
    @staticmethod
    def compile(config: Config):
        if not MOVIEPY_OK:
            logger.warning("moviepy not available; skipping video compilation.")
            return

        frames = sorted([os.path.join(config.temp_dir, f) for f in os.listdir(config.temp_dir) if f.endswith(".png")])
        if not frames:
            logger.error("No frames found to compile.")
            return

        logger.info("Compiling video...")
        clip = ImageSequenceClip(frames, fps=config.fps)
        clip.write_videofile(config.output_file, codec="libx264", bitrate="8000k", audio=False, logger=None)
        logger.info(f"Saved: {config.output_file}")


# -------------------------------
# Self-test
# -------------------------------
def run_self_test(cfg: Config) -> int:
    logger.info("=== SELF TEST START ===")
    logger.info(f"Script: {os.path.abspath(__file__)}")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"EMD_LIB: {EMD_LIB}")
    logger.info(f"moviepy: {'OK' if MOVIEPY_OK else 'MISSING'}")

    try:
        require_emd_backend()
        if cfg.use_cycle_phase or cfg.enable_chaos_gate or cfg.enable_market_context:
            require_hilbert_transform()
    except ImportError as e:
        logger.error(str(e).rstrip())
        return 1
    apply_optional_feature_fallbacks(cfg)

    # force start within last 30 days (always safe for intraday)
    now = pd.Timestamp.utcnow()
    cfg.start_date = (now - pd.Timedelta(days=30)).date().isoformat()
    cfg.end_date = None
    cfg.timeframe = "1h"
    cfg.backtest_safe = False

    dl = DataLoader()
    raw_close, raw_open, transformed = dl.fetch_data(cfg)
    if raw_close is None:
        logger.error("Self-test failed: cannot fetch data.")
        return 1

    # small slice for decomposition sanity
    slice_len = min(len(transformed), 400)
    series_slice = transformed.iloc[-slice_len:]
    raw_close_slice = raw_close.iloc[-slice_len:]
    raw_open_slice = raw_open.iloc[-slice_len:]
    decomp = HHTDecomposer(cfg)
    comps = decomp.decompose_global(series_slice)
    enrich_components_with_extras(raw_close_slice, series_slice, comps, cfg, seed=123, market_context=None)

    max_err = float(np.nanmax(np.abs(comps["error"])))
    logger.info(f"Decomposition recon max error: {max_err:.6g}")

    eng = TradingSignalEngine()
    if cfg.auto_pip_size:
        cfg.pip_size = infer_pip_size(cfg.asset, cfg.pip_size)
    sig = eng.generate_signals(
        raw_close_slice.values,
        raw_open_slice.values,
        series_slice.values,
        comps,
        cfg,
        series_slice.index
    )

    logger.info(f"Signals rows: {len(sig)} | Final equity: {float(sig['equity_curve'].iloc[-1]):.6f}")
    logger.info("=== SELF TEST PASS ===")
    return 0


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="EURUSD=X")
    parser.add_argument("--timeframe", default="1h", choices=["1h", "4h"])
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (optional)")
    parser.add_argument("--no-wavelet", action="store_true", help="Disable wavelet cross-check (pywavelets).")
    parser.add_argument("--no-fft", action="store_true", help="Disable FFT cross-check used in cycle confidence.")
    parser.add_argument("--no-ou", action="store_true", help="Disable OU half-life adaptive windows.")
    parser.add_argument("--chaos", action="store_true", help="Enable chaos (PSR analogue) gate.")
    parser.add_argument("--sandpile", action="store_true", help="Enable sandpile criticality gate.")
    parser.add_argument("--ising", action="store_true", help="Enable Ising criticality gate.")
    parser.add_argument("--no-chaos", action="store_true", help="Disable chaos (PSR analogue) gate.")
    parser.add_argument("--no-sandpile", action="store_true", help="Disable sandpile criticality gate.")
    parser.add_argument("--no-ising", action="store_true", help="Disable Ising criticality gate.")
    parser.add_argument("--no-market-context", action="store_true", help="Disable RMT+MST market context gate.")
    parser.add_argument("--context-assets", default=None, help="Comma-separated assets for market context in non-validate runs.")
    parser.add_argument("--min-cycle-confidence", type=float, default=None, help="Override cycle confidence trade gate (0..1).")
    parser.add_argument("--no-auto-pip", action="store_true", help="Disable pip-size inference for FX tickers.")
    parser.add_argument("--self-test", action="store_true", help="Run dependency + data sanity checks and exit")
    parser.add_argument("--validate", action="store_true", help="Run professional walk-forward + stats validation and exit")
    parser.add_argument("--assets", default=None, help="Comma-separated assets for --validate (e.g., EURUSD=X,GBPUSD=X)")
    parser.add_argument("--timeframes", default=None, help="Comma-separated timeframes for --validate (e.g., 1h,4h)")
    parser.add_argument("--train-bars", type=int, default=2000, help="Walk-forward train window (bars)")
    parser.add_argument("--test-bars", type=int, default=500, help="Walk-forward test window (bars)")
    parser.add_argument("--step-bars", type=int, default=500, help="Walk-forward step size (bars)")
    parser.add_argument("--decompose-every", type=int, default=20, help="SAFE decomposition frequency (bars). Higher=faster.")
    parser.add_argument("--rolling-window", type=int, default=600, help="SAFE decomposition rolling window length (bars)")
    parser.add_argument("--min-trades-train", type=int, default=5, help="Minimum trades required in train window")
    parser.add_argument("--bootstrap-reps", type=int, default=300, help="Bootstrap repetitions")
    parser.add_argument("--bootstrap-block", type=int, default=24, help="Bootstrap block length (bars)")
    parser.add_argument("--perm-reps", type=int, default=300, help="Permutation repetitions")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--signal-model",
        default="rules",
        choices=["rules", "bayes-logit"],
        help="Signal model: legacy rules or bayes-logit (MAP logistic + probability hysteresis).",
    )
    parser.add_argument("--bayes-l2", type=float, default=None, help="bayes-logit: L2 prior precision (smaller=more flexible).")
    parser.add_argument("--bayes-min-train-samples", type=int, default=None, help="bayes-logit: minimum samples required to fit.")
    parser.add_argument("--bayes-fit-only-enabled", action="store_true", help="bayes-logit: fit only on bars where trade_enabled=True.")
    parser.add_argument("--prob-enter-edge", type=float, default=None, help="bayes-logit: enter if p deviates from 0.5 by this edge.")
    parser.add_argument("--prob-exit-edge", type=float, default=None, help="bayes-logit: exit band edge around 0.5 (hysteresis).")
    parser.add_argument("--prob-min-dir-conf", type=float, default=None, help="bayes-logit: minimum posterior direction confidence (0.5..1).")
    parser.add_argument("--no-posterior-conf", action="store_true", help="bayes-logit: disable posterior confidence gating.")
    parser.add_argument("--prob-use-phase-windows", action="store_true", help="bayes-logit: also enforce hard phase windows (like rules).")
    args = parser.parse_args()

    cfg = Config(
        asset=args.asset,
        timeframe=args.timeframe,
        backtest_safe=args.safe_mode,
        duration_sec=args.duration,
    )
    if args.start:
        cfg.start_date = args.start
    if args.end:
        cfg.end_date = args.end
    if args.no_wavelet:
        cfg.enable_wavelet_crosscheck = False
    if args.no_fft:
        cfg.enable_fft_crosscheck = False
    if args.no_ou:
        cfg.enable_ou_adaptive_windows = False
    if args.chaos:
        cfg.enable_chaos_gate = True
    if args.sandpile:
        cfg.enable_sandpile_gate = True
    if args.ising:
        cfg.enable_ising_gate = True
    if args.no_chaos:
        cfg.enable_chaos_gate = False
    if args.no_sandpile:
        cfg.enable_sandpile_gate = False
    if args.no_ising:
        cfg.enable_ising_gate = False
    if args.no_market_context:
        cfg.enable_market_context = False
    if args.min_cycle_confidence is not None:
        cfg.min_cycle_confidence = float(args.min_cycle_confidence)
    if args.no_auto_pip:
        cfg.auto_pip_size = False
    if args.signal_model:
        cfg.signal_model = str(args.signal_model)
    if args.bayes_l2 is not None:
        cfg.bayes_l2 = float(args.bayes_l2)
    if args.bayes_min_train_samples is not None:
        cfg.bayes_min_train_samples = int(args.bayes_min_train_samples)
    if args.bayes_fit_only_enabled:
        cfg.bayes_fit_only_when_enabled = True
    if args.prob_enter_edge is not None:
        cfg.prob_enter_edge = float(args.prob_enter_edge)
    if args.prob_exit_edge is not None:
        cfg.prob_exit_edge = float(args.prob_exit_edge)
    if args.prob_min_dir_conf is not None:
        cfg.prob_min_direction_conf = float(args.prob_min_dir_conf)
    if args.no_posterior_conf:
        cfg.prob_use_posterior_conf = False
    if args.prob_use_phase_windows:
        cfg.prob_use_phase_windows = True

    if args.self_test:
        sys.exit(run_self_test(cfg))

    if args.validate:
        assets = [a.strip() for a in (args.assets.split(",") if args.assets else [cfg.asset]) if a.strip()]
        timeframes = [t.strip() for t in (args.timeframes.split(",") if args.timeframes else [cfg.timeframe]) if t.strip()]
        rc = run_validation_suite(
            assets=assets,
            timeframes=timeframes,
            base_cfg=cfg,
            start=args.start,
            end=args.end,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            decompose_every=args.decompose_every,
            rolling_window=args.rolling_window,
            min_trades_train=args.min_trades_train,
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_block=args.bootstrap_block,
            perm_reps=args.perm_reps,
            seed=args.seed,
        )
        sys.exit(int(rc))

    apply_optional_feature_fallbacks(cfg)

    try:
        require_emd_backend()
        if cfg.use_cycle_phase or cfg.enable_chaos_gate or cfg.enable_market_context:
            require_hilbert_transform()
    except ImportError as e:
        logger.error(str(e).rstrip())
        sys.exit(1)

    dl = DataLoader()
    raw_close, raw_open, transformed = dl.fetch_data(cfg)
    if raw_close is None:
        return

    if str(getattr(cfg, "signal_model", "rules")).lower() == "bayes-logit" and (not bool(cfg.backtest_safe)):
        logger.warning(
            "bayes-logit in GLOBAL mode is NON-CAUSAL (trained on the full series). "
            "Use --safe-mode or --validate for any backtest/statistical evaluation."
        )

    if cfg.auto_pip_size:
        cfg.pip_size = infer_pip_size(cfg.asset, cfg.pip_size)

    market_ctx = None
    if cfg.enable_market_context and args.context_assets:
        ctx_assets = [a.strip() for a in str(args.context_assets).split(",") if a.strip()]
        if ctx_assets:
            if cfg.asset not in ctx_assets:
                ctx_assets = [cfg.asset] + ctx_assets
            # de-dup while preserving order
            ctx_assets = list(dict.fromkeys(ctx_assets))

            close_df = fetch_multi_close(ctx_assets, cfg, start=cfg.start_date, end=cfg.end_date)
            if close_df is None or close_df.empty:
                logger.warning("Market context: no multi-asset close data returned; skipping.")
            else:
                market_ctx = compute_market_context_gate(close_df, cfg)
                try:
                    ok_pct = float(pd.Series(market_ctx["market_ok"]).mean() * 100.0)
                    logger.info(f"Market context: assets={close_df.shape[1]} ok%={ok_pct:.1f}%")
                except Exception:
                    logger.info(f"Market context: assets={close_df.shape[1]}")

    decomp = HHTDecomposer(cfg)
    if cfg.backtest_safe:
        comps = decomp.decompose_backtest_safe(transformed)
    else:
        comps = decomp.decompose_global(transformed)
        logger.warning(
            "GLOBAL mode is NON-CAUSAL (future leakage). "
            "Use --safe-mode for any backtest/PnL evaluation."
        )

    enrich_components_with_extras(raw_close, transformed, comps, cfg, seed=int(args.seed), market_context=market_ctx)

    eng = TradingSignalEngine()
    signals = eng.generate_signals(raw_close.values, raw_open.values, transformed.values, comps, cfg, raw_close.index)

    logger.info(f"Final Equity: {signals['equity_curve'].iloc[-1]:.6f}")

    # --- Diagnostics ---
    logger.info(f"Gross PnL: {signals['pnl_gross'].sum():.6f}")
    logger.info(f"Costs:     {signals['transaction_cost'].sum():.6f}")
    logger.info(f"Net PnL:   {signals['pnl_net'].sum():.6f}")

    trades = (signals['transaction_cost'] > 0).sum()
    logger.info(f"Trades count (events): {trades}")

    pnl = signals['pnl_net']
    logger.info(f"Mean pnl/bar: {pnl.mean():.8f}, Std pnl/bar: {pnl.std():.8f}")

    logger.info(f"Trade enabled %: {signals['trade_enabled'].mean()*100:.1f}%")
    logger.info(f"Time in position %: {(signals['position']!=0).mean()*100:.1f}%")

    RenderManager.run(transformed, comps, signals, cfg)
    VideoCompiler.compile(cfg)


if __name__ == "__main__":
    main()
