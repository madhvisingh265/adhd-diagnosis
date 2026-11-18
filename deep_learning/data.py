"""
Load the EEG recordings and cut them into model-ready windows.

Input  : the IEEE DataPort "EEG data for ADHD / Control children" recordings (121 children,
         19 channels, 128 Hz), as the .mat files inside archive/*.zip in this repo.
         Unzip them into data/ so you have data/ADHD_part1, data/ADHD_part2, data/Control_part1, data/Control_part2.
Output : X  (windows, 19 channels, 256 samples)   - 2-second windows, 50% overlap
         y  (1 = ADHD, 0 = Control)
         groups (child ID) - used so the SAME child never appears in both train and test

Steps:
  1. band-pass filter 1-45 Hz per child (removes slow drift and line noise)
  2. z-score each channel per child (removes amplitude differences between recordings)
  3. slice into overlapping windows
"""
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
FS = 128
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]


def iter_recordings(data_dir=ROOT / "data"):
    """Yield (child_id, label, signal[19, T]) for every .mat file. Label 1 = ADHD, 0 = Control."""
    for folder in sorted(Path(data_dir).glob("*_part*")):
        label = int(folder.name.startswith("ADHD"))
        for f in sorted(folder.glob("*.mat")):
            m = sio.loadmat(f)
            key = [k for k in m if not k.startswith("__")][0]
            yield f.stem, label, m[key].T.astype(np.float32)


def load_windows(data_dir=ROOT / "data", win_sec=2.0, overlap=0.5, cache=True):
    cache_file = ROOT / "data" / f"windows_{win_sec}s_{overlap}.npz"
    if cache and cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        return z["X"], z["y"], z["groups"]

    sos = butter(4, [1, 45], btype="band", fs=FS, output="sos")
    win, step = int(win_sec * FS), int(win_sec * FS * (1 - overlap))
    X, y, groups = [], [], []
    for sid, label, sig in iter_recordings(data_dir):              # sig: (19, T)
        if sig.shape[1] < win:
            continue
        sig = sosfiltfilt(sos, sig, axis=1)
        sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-6)
        for start in range(0, sig.shape[1] - win + 1, step):
            X.append(sig[:, start:start + win]); y.append(label); groups.append(sid)
    X = np.stack(X).astype(np.float32); y = np.array(y); groups = np.array(groups)
    if cache:
        np.savez_compressed(cache_file, X=X, y=y, groups=groups)
    return X, y, groups


if __name__ == "__main__":
    X, y, g = load_windows()
    print(f"windows: {X.shape}, ADHD windows: {y.mean():.1%}, children: {len(np.unique(g))}")
