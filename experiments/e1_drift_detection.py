"""
E1: Drift Detection Experiment (Optimized)
==========================================
Compares C1 (entropy-weighted DI) vs B4 (uniform DI) vs B5 (ADWIN)
on online change detection with K=3 heterogeneous streams.

Protocol:
- Single shift per episode at t=300
- shift_stream = 0 (low-entropy, informative)
- 3 severity levels × 40 seeds
- Threshold calibrated at matched FPR = 5%
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as scipy_entropy

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["e1"]
EXP = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["experiment"]

SEEDS = EXP["seeds"]
BASE_SEED = EXP["rng_base_seed"]
HORIZON = CFG["horizon"]
SHIFT_T = CFG["shift_time"]
K = CFG["K"]
STREAMS = CFG["streams"]
SEVERITIES = CFG["severities"]
SHIFT_STREAM = CFG["shift_stream"]
FPR_TARGET = CFG["fpr_target"]
CAL_WINDOW = CFG["calibration_window"]
ADWIN_DELTA = CFG["adwin_delta"]
WINDOW = 50
N_BINS = 10


def make_ref_dist(entropy_level, n_bins, rng):
    alpha = np.exp(entropy_level) * np.ones(n_bins)
    return rng.dirichlet(alpha)


def generate_observations(ref_dist, T, shift_mag, shift_t, rng):
    n_bins = len(ref_dist)
    if shift_mag > 0:
        shift_vec = rng.dirichlet(np.ones(n_bins))
        shifted = (1 - shift_mag) * ref_dist + shift_mag * shift_vec
        shifted /= shifted.sum()
    else:
        shifted = ref_dist

    obs = np.empty(T, dtype=int)
    obs[:shift_t] = rng.choice(n_bins, size=shift_t, p=ref_dist)
    obs[shift_t:] = rng.choice(n_bins, size=T - shift_t, p=shifted)
    return obs


def capped_simplex_projection(w_raw, w_max):
    w = w_raw.copy()
    for _ in range(30):
        w = np.clip(w, 0, w_max)
        s = w.sum()
        if abs(s - 1.0) < 1e-10:
            break
        capped = w >= w_max - 1e-10
        uncapped = ~capped
        if uncapped.sum() == 0:
            w = np.ones(len(w)) / len(w)
            break
        w[uncapped] += (1.0 - s) / uncapped.sum()
    w = np.clip(w, 0, w_max)
    return w / w.sum()


def compute_all_di(obs_list, ref_hists, method, window=WINDOW,
                   w_max=0.6, epsilon=0.1):
    """Vectorized DI computation over all timesteps."""
    T = len(obs_list[0])
    K = len(obs_list)
    di_all = np.zeros(T)

    for t in range(T):
        start = max(0, t - window + 1)
        ds = np.zeros(K)
        entropies = np.zeros(K)

        for i in range(K):
            seg = obs_list[i][start:t + 1]
            hist_c = np.bincount(seg, minlength=N_BINS).astype(float) + 1e-10
            p = hist_c / hist_c.sum()
            q = ref_hists[i] / ref_hists[i].sum()
            ds[i] = jensenshannon(p, q) ** 2
            entropies[i] = scipy_entropy(p, base=2)

        if method == "entropy_weighted":
            raw = 1.0 / (entropies + epsilon)
            raw_n = raw / raw.sum()
            weights = capped_simplex_projection(raw_n, w_max)
        else:
            weights = np.ones(K) / K

        di_all[t] = np.dot(weights, ds)

    return di_all


class ADWIN:
    def __init__(self, delta=0.002):
        self.delta = delta
        self.window = []
        self.sum_w = 0.0
        self.n = 0

    def update(self, val):
        self.window.append(val)
        self.sum_w += val
        self.n += 1

        if self.n < 10:
            return False

        # Check a few strategic splits only (for speed)
        n = self.n
        checks = [n // 4, n // 3, n // 2, 2 * n // 3]
        for split in checks:
            if split < 5 or split > n - 5:
                continue
            w0 = self.window[:split]
            w1 = self.window[split:]
            n0, n1 = len(w0), len(w1)
            m0, m1 = np.mean(w0), np.mean(w1)
            eps_cut = np.sqrt(0.5 * np.log(4 * n / self.delta) *
                              (1.0 / n0 + 1.0 / n1))
            if abs(m0 - m1) >= eps_cut:
                self.window = self.window[split:]
                self.sum_w = sum(self.window)
                self.n = len(self.window)
                return True
        return False


class CUSUM:
    """Standard cumulative sum (CUSUM) change detector.

    Monitors the cumulative deviation from a reference mean.
    Raises alarm when cumulative sum exceeds threshold h.
    """

    def __init__(self, h: float = 5.0, drift: float = 0.0):
        self.h = h
        self.drift = drift
        self.reset()

    def reset(self):
        self.s_pos = 0.0
        self.s_neg = 0.0
        self._ref_mean = None
        self._calibrating = True
        self._cal_vals = []

    def update(self, val: float) -> bool:
        if self._calibrating:
            self._cal_vals.append(val)
            if len(self._cal_vals) >= 50:
                self._ref_mean = np.mean(self._cal_vals)
                self._calibrating = False
            return False

        z = val - self._ref_mean - self.drift
        self.s_pos = max(0, self.s_pos + z)
        self.s_neg = min(0, self.s_neg + z)

        if self.s_pos > self.h or abs(self.s_neg) > self.h:
            self.s_pos = 0.0
            self.s_neg = 0.0
            return True
        return False


class PageHinkley:
    """Page-Hinkley change detector.

    Monitors the cumulative deviation from the running mean.
    Raises alarm when the maximum deviation exceeds threshold lambda_.
    """

    def __init__(self, lambda_: float = 50.0, alpha: float = 0.005):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.reset()

    def reset(self):
        self._n = 0
        self._sum = 0.0
        self._mean = 0.0
        self._cum_sum = 0.0
        self._min_cum = float("inf")

    def update(self, val: float) -> bool:
        self._n += 1
        self._sum += val
        self._mean = self._sum / self._n
        self._cum_sum += val - self._mean - self.alpha
        self._min_cum = min(self._min_cum, self._cum_sum)

        if self._n < 30:
            return False

        return (self._cum_sum - self._min_cum) > self.lambda_


def run_streaming_detector(obs_list, ref_hists, detector_type, window=WINDOW):
    """Run a streaming change detector (ADWIN, CUSUM, or Page-Hinkley).

    Returns (alarms_array, di_values_array).
    """
    T = len(obs_list[0])
    K = len(obs_list)
    alarms = np.zeros(T, dtype=bool)
    di_vals = np.zeros(T)

    if detector_type == "adwin":
        detectors = [ADWIN(ADWIN_DELTA) for _ in range(K)]
    elif detector_type == "cusum":
        detectors = [CUSUM(h=5.0) for _ in range(K)]
    elif detector_type == "page_hinkley":
        detectors = [PageHinkley(lambda_=50.0) for _ in range(K)]
    else:
        raise ValueError(f"Unknown detector: {detector_type}")

    for t in range(T):
        start = max(0, t - window + 1)
        max_jsd = 0.0
        for i in range(K):
            seg = obs_list[i][start:t + 1]
            hist_c = np.bincount(seg, minlength=N_BINS).astype(float) + 1e-10
            p = hist_c / hist_c.sum()
            q = ref_hists[i] / ref_hists[i].sum()
            jsd = jensenshannon(p, q) ** 2
            max_jsd = max(max_jsd, jsd)
            if detectors[i].update(jsd):
                alarms[t] = True
        di_vals[t] = max_jsd

    return alarms, di_vals


def run_e1():
    results = []
    ts_rows = []

    for sev_name, sev_mag in SEVERITIES.items():
        for s_idx in range(SEEDS):
            seed = BASE_SEED + s_idx
            rng = np.random.default_rng(seed)

            ref_dists = [make_ref_dist(st["entropy_level"], N_BINS, rng)
                         for st in STREAMS]
            ref_hists = [rd * 1000 for rd in ref_dists]

            obs_list = []
            for i in range(K):
                mag = sev_mag if i == SHIFT_STREAM else 0.0
                obs_list.append(generate_observations(
                    ref_dists[i], HORIZON, mag, SHIFT_T, rng))

            for det in ["entropy_weighted", "uniform", "adwin",
                        "cusum", "page_hinkley"]:
                if det in ("entropy_weighted", "uniform"):
                    di_all = compute_all_di(obs_list, ref_hists, det)
                    cal = di_all[:CAL_WINDOW]
                    tau = np.quantile(cal, 1.0 - FPR_TARGET)
                    alarms = di_all > tau
                    di_vals = di_all
                else:
                    alarms, di_vals = run_streaming_detector(
                        obs_list, ref_hists, det)

                pre_fpr = alarms[:SHIFT_T].sum() / SHIFT_T
                post_a = np.where(alarms[SHIFT_T:])[0]
                delay = int(post_a[0]) if len(post_a) > 0 else HORIZON - SHIFT_T

                if alarms[:SHIFT_T].any():
                    arl0 = int(np.argmax(alarms[:SHIFT_T]))
                else:
                    arl0 = SHIFT_T

                results.append({
                    "seed": seed, "severity": sev_name, "detector": det,
                    "delay": delay, "pre_fpr": pre_fpr, "arl0": arl0,
                })

                if s_idx < 3:
                    for t in range(HORIZON):
                        ts_rows.append({
                            "seed": seed, "severity": sev_name, "t": t,
                            "detector": det,
                            "DI_t": float(di_vals[t]) if t < len(di_vals) else 0.0,
                            "alarm": bool(alarms[t]),
                        })

            if (s_idx + 1) % 5 == 0:
                print(f"  [{sev_name}] seed {s_idx + 1}/{SEEDS} done")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    df_m = pd.DataFrame(results)
    df_m.to_csv(out / "e1_metrics.csv", index=False)
    print(f"\n[E1] Saved: {len(df_m)} rows → e1_metrics.csv")

    df_t = pd.DataFrame(ts_rows)
    df_t.to_parquet(out / "e1_timeseries.parquet", index=False)
    print(f"[E1] Saved: {len(df_t)} rows → e1_timeseries.parquet")

    print("\n=== E1 Summary ===")
    agg = df_m.groupby(["severity", "detector"]).agg(
        delay_mean=("delay", "mean"), delay_std=("delay", "std"),
        pre_fpr_mean=("pre_fpr", "mean"),
    ).reset_index()
    print(agg.to_string(index=False))
    return df_m


if __name__ == "__main__":
    run_e1()
