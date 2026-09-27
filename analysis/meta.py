"""Random-effects pooling shared by the revision analyses.

dl() is the same estimator as marjieh_summary.dl (DerSimonian-Laird tau^2,
95% prediction interval on k - 2 df) and adds nothing new except that it takes
arrays. The reported interval in the revised manuscript is the
Hartung-Knapp(-Sidik-Jonkman) one, 'hk_ci' (t on k - 1 df with the
Hartung-Knapp variance); the DerSimonian-Laird normal interval is kept as
'dl_ci' for the supplement."""
import numpy as np
from scipy.stats import t as tdist


def dl(y, se):
    y = np.asarray(y, float)
    v = np.asarray(se, float) ** 2
    k = len(y)
    w = 1 / v
    fe = (w * y).sum() / w.sum()
    Q = float((w * (y - fe) ** 2).sum())
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (v + tau2)
    m = float((wr * y).sum() / wr.sum())
    s = float(np.sqrt(1 / wr.sum()))
    q = float((wr * (y - m) ** 2).sum() / (k - 1))
    s_hk = float(np.sqrt(q / wr.sum()))
    t1 = float(tdist.ppf(0.975, k - 1))
    t2 = float(tdist.ppf(0.975, k - 2)) if k > 2 else float("nan")
    half = t2 * np.sqrt(tau2 + s ** 2)
    return {"est": m, "se": s, "dl_ci": [m - 1.96 * s, m + 1.96 * s], "hk_se": s_hk,
            "hk_ci": [m - t1 * s_hk, m + t1 * s_hk], "tau": float(np.sqrt(tau2)), "pi": [m - half, m + half],
            "Q": Q, "k": k, "I2": max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0}
