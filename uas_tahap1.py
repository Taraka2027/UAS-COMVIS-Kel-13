# UAS Comvis Kelompok 13 - Tahap 1: enhancement (+filter), pilih terbaik, Canny/Laplacian, ekspor data untuk deep learning
import argparse, csv, glob, os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../UTS_Kelompok/data", help="folder berisi horse/ dan mask/")
ap.add_argument("--out", default="hasil")
ap.add_argument("--filter", default="median", choices=["median", "gaussian"])
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--size", type=int, default=128, help="ukuran citra untuk deep learning")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def contrast_stretch(g, lo=2, hi=98):
    a, b = np.percentile(g, (lo, hi))
    return np.clip((g.astype(np.float32) - a) * 255 / max(b - a, 1), 0, 255).astype(np.uint8)


ENH = {"Histogram Equalization": cv2.equalizeHist, "Contrast Stretching": contrast_stretch,
       "CLAHE": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply}
filt = (lambda g: cv2.medianBlur(g, args.k)) if args.filter == "median" else (lambda g: cv2.GaussianBlur(g, (args.k, args.k), 0))
FN = args.filter.capitalize()
# 6 kandidat: 3 enhancement saja + 3 enhancement dengan filter yang sama
CAND = {}
for n, f in ENH.items():
    CAND[n] = f
    CAND["%s + %s" % (n, FN)] = (lambda f: lambda g: filt(f(g)))(f)


def mse_psnr(ref, img):
    mse = float(np.mean((ref.astype(np.float64) - img.astype(np.float64)) ** 2))
    return mse, (99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse))


def canny(g):
    m = np.median(g)
    return cv2.Canny(g, 0.66 * m, 1.33 * m)


def laplacian(g):
    g = cv2.GaussianBlur(g, (0, 0), 4)
    l = np.abs(cv2.Laplacian(g, cv2.CV_64F, ksize=3))
    return ((l > np.percentile(l, 93)) * 255).astype(np.uint8)


CLOSE_K = {"Canny": 7, "Laplacian": 15}


def to_mask(edge, k):
    e = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnt, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = e.shape
    ok = []
    for c in cnt:
        a = cv2.contourArea(c)
        M = cv2.moments(c)
        cx, cy = M["m10"] / max(M["m00"], 1), M["m01"] / max(M["m00"], 1)
        if 0.02 * H * W <= a <= 0.7 * H * W and 0.2 * W < cx < 0.8 * W and 0.2 * H < cy < 0.8 * H:
            ok.append(c)
    m = np.zeros_like(e)
    if ok:
        cv2.drawContours(m, [max(ok, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return m


def iou_dice(pred, gt):
    p, t = pred > 0, gt > 0
    inter = (p & t).sum()
    return inter / max((p | t).sum(), 1), 2 * inter / max(p.sum() + t.sum(), 1)


SEG = {"Canny": canny, "Laplacian": laplacian}
files = sorted(glob.glob(os.path.join(args.data, "horse", "*.png")))
assert files, "citra tidak ditemukan"
name = lambda p: os.path.splitext(os.path.basename(p))[0]
gray = lambda p: cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
gts = [cv2.imread(os.path.join(args.data, "mask", os.path.basename(p)), cv2.IMREAD_GRAYSCALE) for p in files]

# ---- 1) evaluasi 6 kandidat (MSE & PSNR terhadap grayscale asli) ----
rows = []
for p in files:
    g = gray(p)
    for n, f in CAND.items():
        mse, psnr = mse_psnr(g, f(g))
        rows.append([name(p), n, round(mse, 4), round(psnr, 4)])
with open(os.path.join(args.out, "evaluasi_6_kandidat.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["citra", "metode", "MSE", "PSNR_dB"]); w.writerows(rows)
avg = {n: (np.mean([r[2] for r in rows if r[1] == n]), np.mean([r[3] for r in rows if r[1] == n])) for n in CAND}
best = max(avg, key=lambda n: avg[n][1])
filtered = [n for n in CAND if n.endswith(" + " + FN)]
best_filt = max(filtered, key=lambda n: avg[n][1])
inputs = [best] if best in filtered else [best, best_filt]

lines = ["Filter: %s (kernel %d) | jumlah citra: %d" % (args.filter, args.k, len(files)), "",
         "%-34s %12s %12s" % ("Kandidat", "Rata2 MSE", "Rata2 PSNR")]
lines += ["%-34s %12.2f %9.2f dB" % (n, *avg[n]) for n in CAND]
lines += ["", "Terbaik dari 6 (PSNR tertinggi): %s" % best,
          "Berasal dari kelompok filter: %s" % ("YA -> 1 input, 4 output" if best in filtered else "TIDAK -> 2 input (%s dan %s), 8 output" % (best, best_filt)),
          "Input yang dipakai: " + "; ".join(inputs)]

# ---- 2) split data (sama untuk Canny/Laplacian dan deep learning) ----
rs = np.random.RandomState(42)
perm = rs.permutation(len(files))
n_te = int(round(0.2 * len(files)))
test_idx = np.sort(perm[:n_te])
is_test = np.zeros(len(files), bool); is_test[test_idx] = True

# ---- 3) Canny & Laplacian pada tiap input ----
seg_rows, S = [], args.size
npz = {"test": is_test, "names": np.array([name(p) for p in files])}
Y = np.stack([cv2.resize((gt > 0).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) for gt in gts])
npz["Y"] = Y
for k, inp in enumerate(inputs):
    Xs = []
    for p, gt, te in zip(files, gts, is_test):
        g = CAND[inp](gray(p))
        Xs.append(cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA))
        for n, f in SEG.items():
            m = to_mask(f(g), CLOSE_K[n])
            iou, dice = iou_dice(m, gt)
            seg_rows.append([inp, n, name(p), "uji" if te else "latih", round(iou, 4), round(dice, 4)])
    npz["X%d" % k] = np.stack(Xs)
    npz["Xname%d" % k] = np.array(inp)
with open(os.path.join(args.out, "segmentasi_klasik.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["input", "metode", "citra", "subset", "IoU", "Dice"]); w.writerows(seg_rows)
np.savez_compressed(os.path.join(args.out, "data_dl.npz"), **npz)

lines += ["", "%-34s %-10s %-8s %10s %10s" % ("Input", "Metode", "Subset", "Rata2 IoU", "Rata2 Dice")]
for inp in inputs:
    for n in SEG:
        for sub in ("semua", "uji"):
            s = [r for r in seg_rows if r[0] == inp and r[1] == n and (sub == "semua" or r[3] == "uji")]
            lines.append("%-34s %-10s %-8s %10.4f %10.4f" % (inp, n, sub, np.mean([r[4] for r in s]), np.mean([r[5] for r in s])))
txt = "\n".join(lines)
print(txt)
open(os.path.join(args.out, "ringkasan_tahap1.txt"), "w").write(txt + "\n")
