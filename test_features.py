"""test_features.py -- one runnable check for features.py (run: .venv/Scripts/python.exe test_features.py).

PURPOSE
    Fails loudly if the PLV/feature code breaks the properties the project depends on.
WORKFLOW CONNECTION
    Guards Part B before Part C builds on it. Also runs the features on one real cleaned subject and
    writes a short evidence file, logs/part_b_check.txt, which PROGRESS.md points to.
"""
from pathlib import Path

import numpy as np

import features as f
from clean_subject import CH19

rng = np.random.default_rng(0)
lines = []


def log(msg):
    print(msg)
    lines.append(msg)


# 1. Feature counts fall out of the electrode list (ED: 684 / 144 / 40 / 12).
for k, expected in [(19, 684), (9, 144), (5, 40), (3, 12)]:
    assert len(f.feature_names(CH19[:k])) == expected, k
log("PASS counts: 19->684, 9->144, 5->40, 3->12")

# 2. PLV sanity (6 channels; a 2-channel CAR would force PLV=1, so a larger montage is used):
#    ch0 & ch1 are a phase-locked 10 Hz pair, ch2-5 are unrelated noise.
t = np.arange(500) / 250
x = 5.0 * rng.standard_normal((20, 6, 500))   # strong noise: CAR leaks the sine into every channel,
                                              # so noise must dominate in the alpha band for 'unrelated' to be unrelated
x[:, 0] = np.sin(2 * np.pi * 10 * t) + 0.05 * rng.standard_normal((20, 500))
x[:, 1] = np.sin(2 * np.pi * 10 * t + 0.7) + 0.05 * rng.standard_normal((20, 500))
alpha = f.plv_features(f.analytic_bands(x), list(range(6))).reshape(20, 4, 15)[:, 2, :]  # alpha band
alpha_locked, alpha_noise = alpha[:, 0].mean(), alpha[:, 1].mean()   # pair (0,1) vs pair (0,2)
assert alpha_locked > 0.8 and alpha_noise < alpha_locked - 0.3, (alpha_locked, alpha_noise)
log(f"PASS sanity: locked alpha PLV={alpha_locked:.2f}, unrelated={alpha_noise:.2f}")

# 3. CAR equivalence: CAR in time domain on the subset, then analytic == analytic, then CAR in complex domain.
d = rng.standard_normal((4, 19, 500))
keep = [0, 3, 7, 12, 18]
car_first = d[:, keep] - d[:, keep].mean(axis=1, keepdims=True)
a1 = f.analytic_bands(car_first)
a2 = f.analytic_bands(d)[:, :, keep]
a2 = a2 - a2.mean(axis=2, keepdims=True)
assert np.allclose(a1, a2, atol=1e-4)
log("PASS CAR equivalence: time-domain montage CAR == complex-domain montage CAR")

# 4. Leak check: reusing the 19-channel CAR for a smaller montage (the wrong way: no re-centering over the
#    retained channels) gives DIFFERENT PLV than the correct per-montage CAR.
#    (Note: re-centering AFTER a 19-ch CAR would cancel it exactly, so plv_features is safe either way.)
za = f.analytic_bands(d)
correct = f.plv_features(za, keep)
full_car = za - za.mean(axis=2, keepdims=True)
wrong = f.plv_of_signals(full_car[:, :, keep].astype(np.complex128))
assert not np.allclose(correct, wrong, atol=1e-3)
assert np.allclose(f.plv_features(full_car, keep), correct, atol=1e-5)
log(f"PASS leak check: reusing 19-ch CAR differs from correct per-montage CAR (max abs diff {np.abs(correct - wrong).max():.3f}); plv_features is idempotent to prior referencing")

# 5. Feature groups: each electrode touches (k-1)*4 features; groups cover every feature at least twice overall.
g = f.electrode_feature_groups(CH19[:5])
assert all(len(v) == 4 * 4 for v in g.values())
assert sum(len(v) for v in g.values()) == 2 * len(f.feature_names(CH19[:5]))
log("PASS groups: each of 5 electrodes touches 16 features; every feature counted twice")

# 6. Real cleaned subject (if cache exists): finite, in [0,1], right shape at every montage size.
cache = Path(__file__).with_name("cache")
npys = sorted(cache.glob("sub-*.npy"))
if npys:
    data = np.load(npys[0]).astype(np.float64)
    za = f.analytic_bands(data)
    for k in (19, 9, 5, 3):
        X = f.plv_features(za, list(range(k)))
        assert X.shape == (data.shape[0], len(f.feature_names(CH19[:k])))
        assert np.isfinite(X).all() and X.min() >= 0 and X.max() <= 1.0 + 1e-6
        log(f"PASS real subject {npys[0].stem}: k={k} shape={X.shape} PLV range [{X.min():.2f}, {X.max():.2f}] mean {X.mean():.2f}")
Path(__file__).with_name("logs").mkdir(exist_ok=True)
Path(__file__).with_name("logs").joinpath("part_b_check.txt").write_text("\n".join(lines) + "\n")
print("ALL CHECKS PASSED")
