import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

SAVE_DIR = "saved"

# ------------------------------------------------------------------ config
K = 3                 # k-mer size
MAX_LEN = 48          # padded token length
EMB_DIM = 16          # embedding size  -> (1 x 48 x 16)
SEQ_LEN = 41          # benchmark sequences are 41 bp
EPOCHS = 12
BATCH_SIZE = 64

_KMERS = ["".join(p) for p in itertools.product("ACGT", repeat=K)]
VOCAB = {kmer: i + 1 for i, kmer in enumerate(_KMERS)}   # 0 is reserved for padding
VOCAB_SIZE = len(VOCAB) + 1


# ------------------------------------------------------------ preprocessing
def clean_sequence(seq: str) -> str:
    return "".join(str(seq).upper().split())


def is_valid_sequence(seq: str) -> bool:
    seq = clean_sequence(seq)
    return len(seq) >= K and set(seq) <= set("ACGT")


def encode_sequence(seq: str) -> np.ndarray:
    """41 bp sequence -> overlapping 3-mer ids, zero padded to MAX_LEN."""
    seq = clean_sequence(seq)
    ids = [VOCAB[seq[i:i + K]] for i in range(len(seq) - K + 1)]
    ids = ids[:MAX_LEN]
    return np.array(ids + [0] * (MAX_LEN - len(ids)), dtype="int32")


def encode_many(seqs) -> np.ndarray:
    return np.stack([encode_sequence(s) for s in seqs])


# ------------------------------------------------- features (interpretation)
def extract_features(seq: str) -> dict:
    s = clean_sequence(seq)
    n = max(len(s), 1)
    a, t, c, g = (s.count(x) for x in "ATCG")
    cpg = s.count("CG")
    return {
        "gc_content": (g + c) / n,
        "cpg_density": cpg / n,
        "at_skew": (a - t) / (a + t) if (a + t) else 0.0,
        "gc_skew": (g - c) / (g + c) if (g + c) else 0.0,
    }


def interpret(features: dict, label: str) -> str:
    if label == "Methylated":
        if features["gc_content"] > 0.55 and features["cpg_density"] > 0.1:
            return ("High CpG density and GC content - typical of promoter CpG "
                    "islands, which can be hypermethylated in cancer.")
        return "Sequence pattern learned by the model is associated with a methylated site."
    return "No strong methylation-associated pattern found in this sequence."


# -------------------------------------------------------------------- data
def _find_col(df, names):
    for c in df.columns:
        if c.strip().lower() in names:
            return c
    raise ValueError(f"None of the columns {names} found in {list(df.columns)}")


def load_dataset(path: str):
    """
    Load a labelled dataset. Supported formats:
      * CSV / TSV with a sequence column (sequence/seq) and a label column
        (label/class/y) containing 1 (modified) / 0 (non-modified)
      * FASTA whose header ends with the label, e.g.  >seq1|1   or  >seq2|0
    Returns (list_of_sequences, np.array_of_labels).
    """
    path = path.strip()
    if "," in path:   # "positive_file, negative_file"  (e.g. iDNA-MS H.sapiens_pos.fasta, H.sapiens_neg.fasta)
        pos_path, neg_path = [p.strip().strip('"').strip("'") for p in path.split(",", 1)]
        return load_pos_neg(pos_path, neg_path)
    path = path.strip('"').strip("'")
    low = os.path.basename(path).lower()
    if low.endswith((".fasta", ".fa")) and ("_pos" in low or "_neg" in low):
        # iDNA-MS style: H.sapiens_pos.fasta / H.sapiens_neg.fasta (and ..._posTest / ..._negTest).
        # Give either file and the partner file in the same folder is loaded automatically.
        d, f = os.path.split(path)
        i = f.lower().index("_pos") if "_pos" in f.lower() else f.lower().index("_neg")
        return load_pos_neg(os.path.join(d, f[:i] + "_pos" + f[i + 4:]), os.path.join(d, f[:i] + "_neg" + f[i + 4:]))
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(path, sep=None, engine="python")
        seqs = df[_find_col(df, {"sequence", "seq"})].astype(str).tolist()
        labels = df[_find_col(df, {"label", "class", "y"})].astype(int).values
    else:  # FASTA
        seqs, labels, cur = [], [], None
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    cur = int(line.split("|")[-1].strip())
                elif line and cur is not None:
                    seqs.append(line)
                    labels.append(cur)
        labels = np.array(labels)
    keep = [i for i, s in enumerate(seqs) if is_valid_sequence(s)]
    return [clean_sequence(seqs[i]) for i in keep], np.asarray(labels)[keep]


def load_pos_neg(pos_path: str, neg_path: str):
    """For datasets distributed as separate positive / negative files (one sequence per line or FASTA)."""
    def read(p):
        with open(p) as fh:
            return [clean_sequence(l) for l in fh if l.strip() and not l.startswith(">")]
    pos, neg = read(pos_path), read(neg_path)
    seqs = [s for s in pos + neg if is_valid_sequence(s)]
    labels = np.array([1] * len(pos) + [0] * len(neg))
    keep = [i for i, s in enumerate(pos + neg) if is_valid_sequence(s)]
    return seqs, labels[keep]


def make_demo_data(n=2000, seed=42):
    """Synthetic data ONLY for testing that the code runs (not real biology)."""
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for _ in range(n):
        if rng.random() < 0.5:   # 'methylated' - CpG / GC rich
            p = [0.2, 0.2, 0.3, 0.3]
            y = 1
        else:                    # 'non-methylated' - AT rich
            p = [0.35, 0.35, 0.15, 0.15]
            y = 0
        seqs.append("".join(rng.choice(list("ATCG"), size=SEQ_LEN, p=p)))
        labels.append(y)
    return seqs, np.array(labels)


# ------------------------------------------------------------------- model
def build_model(use_mlp=True, use_tcn=True, use_bilstm=True, use_attention=True):
    """
    Hybrid MLP + TCN + BiLSTM + Attention (with residual connections).
    The use_* flags allow the ablation study mentioned in the PPT.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    class Attention(layers.Layer):
        """Additive attention over time steps -> weighted sum vector."""
        def build(self, input_shape):
            self.w = layers.Dense(64, activation="tanh")
            self.v = layers.Dense(1)
            super().build(input_shape)

        def call(self, x):
            score = self.v(self.w(x))               # (batch, time, 1)
            weights = tf.nn.softmax(score, axis=1)
            return tf.reduce_sum(weights * x, axis=1)

    def tcn_block(x, filters, dilation, dropout=0.2):
        shortcut = x
        y = layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation, activation="relu")(x)
        y = layers.SpatialDropout1D(dropout)(y)
        y = layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation, activation="relu")(y)
        y = layers.SpatialDropout1D(dropout)(y)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1)(shortcut)
        return layers.Activation("relu")(layers.Add()([shortcut, y]))

    inp = layers.Input(shape=(MAX_LEN,), dtype="int32")
    x = layers.Embedding(VOCAB_SIZE, EMB_DIM)(inp)          # (48 x 16)

    if use_mlp:                                             # MLP block + residual
        h = layers.Dense(64, activation="relu")(x)
        h = layers.Dense(EMB_DIM, activation="relu")(h)
        x = layers.LayerNormalization()(layers.Add()([x, h]))

    if use_tcn:                                             # TCN block (dilated)
        for d in (1, 2, 4):
            x = tcn_block(x, 64, d)

    if use_bilstm:                                          # BiLSTM block
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    if use_attention and (use_bilstm or use_tcn or use_mlp):
        x = Attention()(x)
    else:
        x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ----------------------------------------------------------------- metrics
def evaluate(y_true, y_prob):
    from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
    y_pred = (np.asarray(y_prob).ravel() >= 0.5).astype(int)
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def predict_sequence(model, seq: str):
    """Returns (label, confidence, raw_score)."""
    x = encode_sequence(seq)[None, :]
    p = float(model.predict(x, verbose=0)[0][0])
    label = "Methylated" if p >= 0.5 else "Non-Methylated"
    confidence = p if p >= 0.5 else 1 - p
    return label, confidence, p


# ------------------------------------------------------- training & plots
def plot_history(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(hist["accuracy"], label="train"); ax[0].plot(hist["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(hist["loss"], label="train"); ax[1].plot(hist["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-Meth", "Meth"], yticklabels=["Non-Meth", "Meth"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y, p, auc, path):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}"); ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC Curve"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run(seqs_tr, y_tr, seqs_te, y_te, name="model", epochs=EPOCHS, **flags):
    os.makedirs(SAVE_DIR, exist_ok=True)
    Xtr, Xte = encode_many(seqs_tr), encode_many(seqs_te)
    Xa, Xv, ya, yv = train_test_split(Xtr, y_tr, test_size=0.1, stratify=y_tr, random_state=42)
    model = build_model(**flags)
    h = model.fit(Xa, ya, validation_data=(Xv, yv), epochs=epochs, batch_size=BATCH_SIZE, verbose=2)
    prob = model.predict(Xte, verbose=0).ravel()
    res = evaluate(y_te, prob)
    return model, h.history, prob, res




# ------------------------------------------------------------ ablation study
def run_ablation(seqs_tr, y_tr, seqs_te, y_te, epochs=EPOCHS):
    variants = {
        "Full model": {},
        "w/o MLP": {"use_mlp": False}, "w/o TCN": {"use_tcn": False},
        "w/o BiLSTM": {"use_bilstm": False}, "w/o Attention": {"use_attention": False},
    }
    rows = []
    for vname, fl in variants.items():
        _, _, _, r = run(seqs_tr, y_tr, seqs_te, y_te, epochs=epochs, **fl)
        rows.append({"Model": vname, "ACC": round(r["ACC"], 4), "MCC": round(r["MCC"], 4), "AUC": round(r["AUC"], 4)})
    return rows


# ====================================================================== WEB APP
st.set_page_config(page_title="Epigenetic Pattern Prediction", page_icon="🧬", layout="wide")
SAVE_DIR = "saved"

st.title("🧬 Epigenetic Pattern Prediction Using Deep Learning")
st.caption("Aim: Fast and cost-effective alternative to experimental methods for DNA methylation prediction")

# ---------------- sidebar
mtype = st.sidebar.selectbox("Methylation type", ["4mC", "5hmC", "6mA"])
species = st.sidebar.selectbox("Species", ["A. thaliana", "C. elegans", "D. melanogaster", "H. sapiens", "Other"])
name = mtype  # one model file per methylation type: saved/4mC.weights.h5, ...
st.sidebar.header("📊 Model Info")
for img, cap in [(f"{name}_confusion.png", "Confusion Matrix"), (f"{name}_history.png", "Training History"), (f"{name}_roc.png", "ROC Curve")]:
    p = os.path.join(SAVE_DIR, img)
    if os.path.exists(p):
        st.sidebar.image(p, caption=cap)


@st.cache_resource
def load_model(path, mtime):
    model = build_model()      # rebuild the architecture, then load the trained weights
    model.load_weights(path)
    return model


model_path = os.path.join(SAVE_DIR, f"{name}.weights.h5")
tab1, tab2 = st.tabs(["🔬 Prediction", "📈 Training & Results"])

with tab1:
    st.subheader("Enter Genomic Sequence (A,T,C,G)")
    seq = st.text_area("DNA Sequence:", "CGCGCGCGCGCGATATATATACGCGCGCGCGCGCGCGCG", height=120)
    if st.button("Predict Methylation", type="primary"):
        s = clean_sequence(seq)
        if not is_valid_sequence(s):
            st.error("Invalid sequence: use only A, T, C, G (minimum 3 bases).")
        elif not os.path.exists(model_path):
            st.warning(f"No trained model found for {mtype}. Train one in the 'Training & Results' tab first.")
        else:
            if len(s) != SEQ_LEN:
                st.info(f"Model was trained on {SEQ_LEN} bp sequences; your input has {len(s)} bp.")
            model = load_model(model_path, os.path.getmtime(model_path))
            label, conf, raw = predict_sequence(model, s)
            feats = extract_features(s)
            c1, c2 = st.columns(2)
            with c1:
                (st.success if label == "Methylated" else st.warning)(f"{label} ({conf*100:.2f}%)")
                st.progress(min(max(conf, 0.0), 1.0))
                st.write(f"Raw Score: {raw:.4f}  |  Type: {mtype}  |  Species: {species}")
            with c2:
                st.write("**Extracted Features:**")
                st.json({k: round(v, 4) for k, v in feats.items()})
            st.info("**Biological Interpretation:** " + interpret(feats, label))
            st.caption("Supports research in cancer and personalized medicine")

with tab2:
    st.subheader(f"Train model for {mtype}")
    data_path = st.text_input("Training dataset path (CSV/FASTA) - leave empty for synthetic demo data", "")
    test_path = st.text_input("Independent test dataset path (optional)", "")
    if st.button("Train Model (12 epochs)"):
        import numpy as np
        with st.spinner("Training... this can take a few minutes"):
            if data_path:
                seqs, y = load_dataset(data_path)
            else:
                st.warning("Using synthetic demo data (not real methylation data).")
                seqs, y = make_demo_data()
            if test_path:
                seqs_te, y_te = load_dataset(test_path); seqs_tr, y_tr = seqs, y
            else:
                seqs_tr, seqs_te, y_tr, y_te = train_test_split(seqs, y, test_size=0.2, stratify=y, random_state=42)
            y_tr, y_te = np.asarray(y_tr), np.asarray(y_te)
            model, hist, prob, res = run(seqs_tr, y_tr, seqs_te, y_te, name, 12)
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_weights(model_path)
            plot_history(hist, os.path.join(SAVE_DIR, f"{name}_history.png"))
            plot_confusion(res["confusion_matrix"], os.path.join(SAVE_DIR, f"{name}_confusion.png"))
            plot_roc(y_te, prob, res["AUC"], os.path.join(SAVE_DIR, f"{name}_roc.png"))
            json.dump({k: float(v) for k, v in res.items() if k != "confusion_matrix"},
                      open(os.path.join(SAVE_DIR, f"{name}_metrics.json"), "w"))
        st.success("Training complete. Reload the page to refresh the sidebar plots.")
    mp = os.path.join(SAVE_DIR, f"{name}_metrics.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        a, b, c = st.columns(3)
        a.metric("ACC", f"{m['ACC']*100:.2f}%"); b.metric("MCC", f"{m['MCC']:.4f}"); c.metric("AUC", f"{m['AUC']*100:.2f}%")

    st.divider()
    if st.button("Run Ablation Study (trains 5 models - slow)"):
        with st.spinner("Running ablation study..."):
            if data_path:
                seqs, y = load_dataset(data_path)
            else:
                st.warning("Using synthetic demo data (not real methylation data).")
                seqs, y = make_demo_data()
            seqs_tr, seqs_te, y_tr, y_te = train_test_split(seqs, y, test_size=0.2, stratify=y, random_state=42)
            rows = run_ablation(seqs_tr, np.asarray(y_tr), seqs_te, np.asarray(y_te), epochs=12)
        st.table(rows)