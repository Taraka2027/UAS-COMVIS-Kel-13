# UAS Comvis Kelompok 13 - Tahap 2: U-Net dan DeepLabV3 (dilatih dari nol, tanpa bobot pretrained)
# Alur sesuai diagram dosen: masukan terbaik -> Metode 1 (Canny) / Metode 2 (Laplacian) -> U-Net / DeepLabV3.
# Untuk setiap masukan enhancement (hasil uas_tahap1.py), model dilatih dengan tiga jenis masukan:
#   - "Citra"     : citra hasil enhancement langsung (pembanding, tanpa Canny/Laplacian)
#   - "Canny"     : peta tepi keluaran Canny (Metode 1)
#   - "Laplacian" : peta tepi keluaran Laplacian (Metode 2)
# Target selalu mask ground truth. Setiap kombinasi dilatih dengan U-Net dan DeepLabV3.
# Peta tepi dihitung di sini dari citra asli (fungsi yang sama dengan uas_tahap1.py), lalu diperkecil ke 128 x 128.
import argparse, csv, glob, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="weizmann_horse_db", help="folder berisi horse/ dan mask/")
ap.add_argument("--npz", default="hasil/data_dl.npz", help="keluaran uas_tahap1.py")
ap.add_argument("--out", default="hasil")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--k", type=int, default=5, help="kernel median (harus sama dengan uas_tahap1.py)")
ap.add_argument("--size", type=int, default=128)
args = ap.parse_args()

d = np.load(args.npz)
Y, test, names = d["Y"].astype(np.float32)[..., None], d["test"], [str(n) for n in d["names"]]
inputs = [str(d["Xname%d" % k]) for k in range(10) if "Xname%d" % k in d.files]
SRC = ["Citra", "Canny", "Laplacian"]


# ---- enhancement dan metode klasik (sama dengan uas_tahap1.py) ----
def contrast_stretch(g, lo=2, hi=98):
    a, b = np.percentile(g, (lo, hi))
    return np.clip((g.astype(np.float32) - a) * 255 / max(b - a, 1), 0, 255).astype(np.uint8)


ENH = {"Histogram Equalization": cv2.equalizeHist, "Contrast Stretching": contrast_stretch,
       "CLAHE": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply}
filt = lambda g: cv2.medianBlur(g, args.k)
CAND = {}
for n, f in ENH.items():
    CAND[n] = f
    CAND["%s + Median" % n] = (lambda f: lambda g: filt(f(g)))(f)


def canny(g):
    m = np.median(g)
    return cv2.Canny(g, 0.66 * m, 1.33 * m)


def laplacian(g):
    g = cv2.GaussianBlur(g, (0, 0), 4)
    l = np.abs(cv2.Laplacian(g, cv2.CV_64F, ksize=3))
    return ((l > np.percentile(l, 93)) * 255).astype(np.uint8)


METODE = {"Canny": canny, "Laplacian": laplacian}

files = sorted(glob.glob(os.path.join(args.data, "horse", "*.png")))
assert files, "citra tidak ditemukan"
assert [os.path.splitext(os.path.basename(p))[0] for p in files] == names, "daftar citra tidak sama dengan data_dl.npz"
for inp in inputs:
    assert inp in CAND, "masukan tidak dikenal: %s" % inp

S = args.size
DATA = {}  # (masukan enhancement, sumber) -> larik N x S x S
for inp in inputs:
    E = {s: [] for s in SRC}
    for p in files:
        g = CAND[inp](cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY))
        E["Citra"].append(cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA))
        for m, f in METODE.items():  # keluaran Metode 1 / Metode 2 pada resolusi asli, lalu diperkecil
            E[m].append(cv2.resize(f(g), (S, S), interpolation=cv2.INTER_AREA))
    for s in SRC:
        DATA[(inp, s)] = np.stack(E[s])
    print("%-30s peta tepi: Canny %.2f%%, Laplacian %.2f%% piksel tepi" % (
        inp, 100 * (DATA[(inp, "Canny")] > 0).mean(), 100 * (DATA[(inp, "Laplacian")] > 0).mean()), flush=True)


def conv(x, f, k=3, rate=1):
    x = L.Conv2D(f, k, padding="same", dilation_rate=rate, use_bias=False)(x)
    return L.ReLU()(L.BatchNormalization()(x))


def unet(S=128, b=16):
    i = keras.Input((S, S, 1)); x = i; sk = []
    for f in (b, 2 * b, 4 * b, 8 * b):
        x = conv(conv(x, f), f); sk.append(x); x = L.MaxPooling2D()(x)
    x = conv(conv(x, 16 * b), 16 * b)
    for f, s in zip((8 * b, 4 * b, 2 * b, b), reversed(sk)):
        x = L.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = conv(conv(L.Concatenate()([x, s]), f), f)
    return keras.Model(i, L.Conv2D(1, 1, activation="sigmoid")(x), name="UNet")


def res_block(x, f, stride=1, rate=1):
    s = x
    y = conv(x, f, 3, rate) if stride == 1 else L.Conv2D(f, 3, strides=stride, padding="same", use_bias=False)(x)
    if stride != 1:
        y = L.ReLU()(L.BatchNormalization()(y))
    y = L.BatchNormalization()(L.Conv2D(f, 3, padding="same", dilation_rate=rate, use_bias=False)(y))
    if stride != 1 or s.shape[-1] != f:
        s = L.BatchNormalization()(L.Conv2D(f, 1, strides=stride, use_bias=False)(s))
    return L.ReLU()(L.Add()([y, s]))


def deeplabv3(S=128, b=16):
    # encoder ResNet kecil (output stride 8, blok terakhir atrous) + ASPP + upsampling bilinear
    i = keras.Input((S, S, 1))
    x = conv(i, b)
    x = res_block(x, 2 * b, 2)                  # /2
    x = res_block(x, 4 * b, 2)                  # /4
    x = res_block(x, 8 * b, 2)                  # /8
    x = res_block(x, 16 * b, 1, rate=2)         # atrous, tetap /8
    n = S // 8
    br = [conv(x, 8 * b, 1)] + [conv(x, 8 * b, 3, r) for r in (6, 12, 18)]
    ip = L.GlobalAveragePooling2D(keepdims=True)(x)
    ip = L.UpSampling2D(n, interpolation="bilinear")(conv(ip, 8 * b, 1))
    x = conv(L.Concatenate()(br + [ip]), 8 * b, 1)
    x = L.Dropout(0.1)(x)
    x = L.UpSampling2D(8, interpolation="bilinear")(L.Conv2D(1, 1)(x))
    return keras.Model(i, L.Activation("sigmoid")(x), name="DeepLabV3")


def dice_loss(y, p):
    inter = tf.reduce_sum(y * p, (1, 2, 3))
    return 1 - tf.reduce_mean((2 * inter + 1) / (tf.reduce_sum(y, (1, 2, 3)) + tf.reduce_sum(p, (1, 2, 3)) + 1))


def loss(y, p):
    return tf.reduce_mean(keras.losses.binary_crossentropy(y, p)) + dice_loss(y, p)


def iou_dice(p, t):
    p, t = p > 0.5, t > 0.5
    inter = (p & t).sum((1, 2, 3)); uni = (p | t).sum((1, 2, 3))
    return inter / np.maximum(uni, 1), 2 * inter / np.maximum(p.sum((1, 2, 3)) + t.sum((1, 2, 3)), 1)


# Augmentasi diterapkan pada peta tepi dan mask sekaligus (digabung jadi 2 kanal), sama seperti uas_dl.py.
AUG = keras.Sequential([L.RandomFlip("horizontal"), L.RandomTranslation(0.1, 0.1), L.RandomZoom(0.15), L.RandomRotation(0.05)])


def augment(x, y):
    z = AUG(tf.concat([x, y], -1), training=True)
    return z[..., :1], tf.cast(z[..., 1:] > 0.5, tf.float32)


class BestAfter(keras.callbacks.Callback):  # simpan bobot val_loss terbaik setelah epoch awal (BN belum stabil di awal)
    def __init__(self, start=10):
        super().__init__(); self.start, self.best, self.w = start, 1e9, None

    def on_epoch_end(self, e, logs=None):
        if e >= self.start and logs["val_loss"] < self.best:
            self.best, self.w = logs["val_loss"], self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.w is not None:
            self.model.set_weights(self.w)


rows, preds = [], {}
tr_idx = np.where(~test)[0]
rs = np.random.RandomState(0); rs.shuffle(tr_idx)
nv = int(0.15 * len(tr_idx)); va_idx, fit_idx = tr_idx[:nv], tr_idx[nv:]
te_idx = np.where(test)[0]

for inp in inputs:
    for src in SRC:
        X = (DATA[(inp, src)].astype(np.float32) / 255.0)[..., None]   # masukan model
        for mk in (unet, deeplabv3):
            keras.utils.set_random_seed(42)
            m = mk()
            m.compile(keras.optimizers.Adam(1e-3), loss=loss)
            ds = tf.data.Dataset.from_tensor_slices((X[fit_idx], Y[fit_idx])).shuffle(256, seed=1).batch(args.batch).map(augment)
            cb = [BestAfter(10), keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)]
            h = m.fit(ds, validation_data=(X[va_idx], Y[va_idx]), epochs=args.epochs, callbacks=cb, verbose=0)
            p = m.predict(X[te_idx], verbose=0)
            iou, dice = iou_dice(p, Y[te_idx])
            rows.append([inp, src, m.name, len(h.history["loss"]), round(float(iou.mean()), 4), round(float(dice.mean()), 4),
                         int((iou >= 0.5).sum()), len(te_idx), m.count_params()])
            preds[(inp, src, m.name)] = (p, iou)
            print(rows[-1], flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "dl_metrik.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["input", "masukan_model", "model", "epoch", "IoU_uji", "Dice_uji", "IoU>=0.5", "n_uji", "parameter"])
    w.writerows(rows)

# gambar: 3 citra uji per (masukan enhancement, sumber) -> masukan model | GT | U-Net | DeepLabV3
sel = np.linspace(0, len(te_idx) - 1, 3).astype(int)
for k, inp in enumerate(inputs):
    for src in SRC:
        X = DATA[(inp, src)]
        fig, ax = plt.subplots(3, 4, figsize=(11, 8.4))
        for r, j in enumerate(sel):
            i = te_idx[j]
            panels = [(X[i], "%s\nMasukan model: %s (%s)" % (names[i], src, inp)), (Y[i, ..., 0], "Ground truth")]
            for mn in ("UNet", "DeepLabV3"):
                p, iou = preds[(inp, src, mn)]
                panels.append((p[j, ..., 0] > 0.5, "%s (IoU %.2f)" % (mn, iou[j])))
            for c, (im, t) in enumerate(panels):
                ax[r, c].imshow(im, cmap="gray"); ax[r, c].set_title(t, fontsize=7); ax[r, c].axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_dl_%d_%s.png" % (k + 1, src.lower())), dpi=120); plt.close(fig)
