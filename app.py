"""
app.py  –  Parkinson's Disease Voice Detection  (Streamlit)
Run:  streamlit run app.py
"""

import os
import pickle
import tempfile
import time

import numpy as np
import pandas as pd
import streamlit as st

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Voice Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"]          { background: #161b27; }
    h1, h2, h3, h4               { color: #e8eaf6; }
    p, li, label                  { color: #b0bec5; }

    .metric-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #162032 100%);
        border: 1px solid #2d3f55;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-card .lbl { font-size: 0.8rem; color: #78909c; margin: 0; }

    .result-positive {
        background: linear-gradient(135deg, #4a1a1a 0%, #7f0000 100%);
        border: 2px solid #ef5350;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #1a2e1a 0%, #1b5e20 100%);
        border: 2px solid #66bb6a;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-title { font-size: 1.7rem; font-weight: 800; margin-bottom: 0.4rem; }
    .result-sub   { font-size: 1rem; color: #cfd8dc; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── feature engineering (mirrors notebook) ───────────────────────────────────
def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    jitter_cols  = [c for c in df.columns if "Jitter"  in c or "jitter"  in c.lower()]
    shimmer_cols = [c for c in df.columns if "Shimmer" in c or "shimmer" in c.lower()]

    if jitter_cols and shimmer_cols:
        df_new["jitter_shimmer_product"]   = df[jitter_cols].mean(axis=1) * df[shimmer_cols].mean(axis=1)
        df_new["avg_jitter"]               = df[jitter_cols].mean(axis=1)
        df_new["avg_shimmer"]              = df[shimmer_cols].mean(axis=1)
        df_new["voice_perturbation_index"] = (df[jitter_cols].mean(axis=1) + df[shimmer_cols].mean(axis=1)) / 2

    if "HNR" in df.columns and "NHR" in df.columns:
        df_new["hnr_nhr_ratio"] = df["HNR"] / (df["NHR"] + 1e-10)

    if "MDVP:Fhi(Hz)" in df.columns and "MDVP:Flo(Hz)" in df.columns:
        df_new["frequency_range"] = df["MDVP:Fhi(Hz)"] - df["MDVP:Flo(Hz)"]

    if all(c in df.columns for c in ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"]):
        df_new["frequency_cv"] = (df["MDVP:Fhi(Hz)"] - df["MDVP:Flo(Hz)"]) / (df["MDVP:Fo(Hz)"] + 1e-10)

    for col in ["spread1", "spread2", "DFA", "PPE"]:
        if col not in df_new.columns:
            df_new[col] = 0.0

    df_new["spread_interaction"] = df_new["spread1"] * df_new["spread2"]
    df_new["dfa_ppe_product"]    = df_new["DFA"]     * df_new["PPE"]
    return df_new


# ── model loading ─────────────────────────────────────────────────────────────
ARTIFACT_DIR_DEFAULT = os.environ.get("ARTIFACT_DIR", "parkinsons_outputs")

@st.cache_resource(show_spinner="Loading model…")
def load_model(artifact_dir: str):
    try:
        model         = pickle.load(open(f"{artifact_dir}/best_model.pkl",    "rb"))
        scaler        = pickle.load(open(f"{artifact_dir}/scaler.pkl",        "rb"))
        feature_names = pickle.load(open(f"{artifact_dir}/feature_names.pkl", "rb"))
        return model, scaler, feature_names, None
    except FileNotFoundError as e:
        return None, None, None, str(e)


# ── prediction ────────────────────────────────────────────────────────────────
def predict(raw_features: dict, model, scaler, feature_names):
    df_raw = pd.DataFrame([raw_features])
    df_eng = create_engineered_features(df_raw)
    for feat in feature_names:
        if feat not in df_eng.columns:
            df_eng[feat] = 0.0
    X_scaled = scaler.transform(df_eng[feature_names].values)
    prob  = model.predict_proba(X_scaled)[0]
    label = model.predict(X_scaled)[0]
    return int(label), float(prob[1])


# ── result renderer ───────────────────────────────────────────────────────────
def show_results(label: int, prob_parkinsons: float, raw_feats: dict, elapsed: float):
    st.markdown("---")
    st.markdown("##  Analysis Results")
    prob_pct = prob_parkinsons * 100

    if prob_pct >= 70:
        st.markdown(
            f'<div class="result-positive">'
            f'<p class="result-title" style="color:#ef9a9a;">⚠️ Possible Parkinson\'s Indicators Detected</p>'
            f'<p class="result-sub">Confidence: <strong>{prob_pct:.1f}%</strong></p>'
            f'<p class="result-sub">Please consult a neurologist for clinical evaluation.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif prob_pct >= 50:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#2a2200 0%,#4a3800 100%);'
            f'border:2px solid #ffa726;border-radius:14px;padding:1.8rem;text-align:center;">'
            f'<p class="result-title" style="color:#ffcc80;">🔍 Inconclusive — Low Confidence Result</p>'
            f'<p class="result-sub">Parkinson\'s probability: <strong>{prob_pct:.1f}%</strong></p>'
            f'<p class="result-sub">The model is not confident enough to draw a conclusion. '
            f'Try a higher-quality recording or use the Manual Entry tab with Praat-computed values.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-negative">'
            f'<p class="result-title" style="color:#a5d6a7;">✅ No Significant Parkinson\'s Indicators</p>'
            f'<p class="result-sub">Confidence (healthy): <strong>{(1-prob_parkinsons)*100:.1f}%</strong></p>'
            f'<p class="result-sub">Voice biomarkers appear within normal range.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, f"{prob_pct:.1f}%",                                  "#ef5350", "Parkinson's Probability"),
        (c2, f"{raw_feats.get('MDVP:Fo(Hz)', 0):.1f} Hz",         "#42a5f5", "Fundamental Frequency"),
        (c3, f"{raw_feats.get('HNR', 0):.2f} dB",                 "#ab47bc", "Harmonics-to-Noise"),
        (c4, f"{elapsed*1000:.0f} ms",                             "#26a69a", "Analysis Time"),
    ]
    for col, val, color, lbl in cards:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="val" style="color:{color};">{val}</p>'
                f'<p class="lbl">{lbl}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### 🔬 Key Biomarkers")
    b1, b2 = st.columns(2)

    with b1:
        st.markdown("**Frequency Measures**")
        st.dataframe(pd.DataFrame({
            "Feature": ["Fo (avg)", "Fhi (max)", "Flo (min)", "Freq Range"],
            "Value":   [
                f'{raw_feats.get("MDVP:Fo(Hz)", 0):.2f} Hz',
                f'{raw_feats.get("MDVP:Fhi(Hz)", 0):.2f} Hz',
                f'{raw_feats.get("MDVP:Flo(Hz)", 0):.2f} Hz',
                f'{raw_feats.get("MDVP:Fhi(Hz)", 0) - raw_feats.get("MDVP:Flo(Hz)", 0):.2f} Hz',
            ],
        }), hide_index=True, width='stretch')

        st.markdown("**Jitter (pitch instability)**")
        st.dataframe(pd.DataFrame({
            "Feature": ["Local %", "Absolute", "RAP", "PPQ", "DDP"],
            "Value":   [
                f'{raw_feats.get("MDVP:Jitter(%)", 0):.5f}',
                f'{raw_feats.get("MDVP:Jitter(Abs)", 0):.7f}',
                f'{raw_feats.get("MDVP:RAP", 0):.5f}',
                f'{raw_feats.get("MDVP:PPQ", 0):.5f}',
                f'{raw_feats.get("Jitter:DDP", 0):.5f}',
            ],
        }), hide_index=True, width='stretch')

    with b2:
        st.markdown("**Shimmer (amplitude instability)**")
        st.dataframe(pd.DataFrame({
            "Feature": ["Local", "dB", "APQ3", "APQ5", "APQ11", "DDA"],
            "Value":   [
                f'{raw_feats.get("MDVP:Shimmer", 0):.5f}',
                f'{raw_feats.get("MDVP:Shimmer(dB)", 0):.4f}',
                f'{raw_feats.get("Shimmer:APQ3", 0):.5f}',
                f'{raw_feats.get("Shimmer:APQ5", 0):.5f}',
                f'{raw_feats.get("MDVP:APQ", 0):.5f}',
                f'{raw_feats.get("Shimmer:DDA", 0):.5f}',
            ],
        }), hide_index=True, width='stretch')

        st.markdown("**Nonlinear / Noise Measures**")
        st.dataframe(pd.DataFrame({
            "Feature": ["NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"],
            "Value":   [
                f'{raw_feats.get("NHR", 0):.5f}',
                f'{raw_feats.get("HNR", 0):.4f}',
                f'{raw_feats.get("RPDE", 0):.4f}',
                f'{raw_feats.get("DFA", 0):.4f}',
                f'{raw_feats.get("spread1", 0):.4f}',
                f'{raw_feats.get("spread2", 0):.4f}',
                f'{raw_feats.get("D2", 0):.4f}',
                f'{raw_feats.get("PPE", 0):.4f}',
            ],
        }), hide_index=True, width='stretch')

    st.markdown("### Prediction Confidence")
    bar_color = "#ef5350" if prob_pct > 50 else "#66bb6a"
    st.markdown(
        f'<div style="background:#1e2a3a;border-radius:10px;padding:4px;margin-bottom:6px;">'
        f'<div style="background:{bar_color};width:{prob_pct:.1f}%;height:28px;border-radius:8px;'
        f'display:flex;align-items:center;padding-left:12px;color:white;font-weight:700;">'
        f'{prob_pct:.1f}% Parkinson\'s</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("###  Download Report")
    report = {**raw_feats,
              "Prediction": "Parkinson's" if label == 1 else "Healthy",
              "Probability_Parkinsons_%": round(prob_pct, 2)}
    csv_bytes = (
        pd.DataFrame([report]).T
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Value"})
        .to_csv(index=False)
        .encode()
    )
    st.download_button(
        "⬇️ Download Feature Report (CSV)",
        data=csv_bytes,
        file_name="parkinsons_voice_report.csv",
        mime="text/csv",
        width='stretch',
    )

    st.warning(
        " **Medical Disclaimer:** This tool is for **research and screening purposes only** "
        "and does not constitute a clinical diagnosis. Always consult a qualified neurologist."
    )


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  About")
    st.info(
        "Analyses vocal biomarkers — jitter, shimmer, HNR and "
        "nonlinear measures — to screen for signs of Parkinson's Disease.\n\n"
        "Does **not** replace clinical diagnosis."
    )
    st.markdown("###  Recording Tips")
    st.markdown(
        "- Sustain the vowel **'ahhh'** for 5–10 s  \n"
        "- Sit in a quiet room  \n"
        "- Hold the mic ~10 cm from your mouth  \n"
        "- Export as **WAV (16-bit, mono)**"
    )
    st.markdown("---")
    st.markdown("###  Artifact Directory")
    artifact_dir = st.text_input("Path to model outputs folder", value=ARTIFACT_DIR_DEFAULT)


# ── main ──────────────────────────────────────────────────────────────────────
st.title("🎙️ Parkinson's Disease Voice Detection")
st.caption(
    "Upload a WAV recording of a sustained vowel — the pipeline extracts "
    "vocal biomarkers and feeds them to your trained classifier."
)

model, scaler, feature_names, load_err = load_model(artifact_dir)

if load_err:
    st.error(
        f"**Could not load model artifacts from `{artifact_dir}/`.**\n\n"
        f"`{load_err}`\n\n"
        "Run the training notebook first, then point this app at the output folder "
        "using the sidebar text box."
    )
    st.stop()

st.success(" Model loaded successfully")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_upload, tab_manual = st.tabs(["🎤 Voice Upload", "🔬 Manual Feature Entry"])

# ─── Tab 1: WAV upload ────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("### Upload a WAV Recording")
    uploaded = st.file_uploader(
        "Choose a WAV file",
        type=["wav"],
        help="Record a sustained 'ahhh' vowel for 5–10 seconds, then upload.",
    )

    if uploaded:
        st.audio(uploaded, format="audio/wav")
        if st.button("🔍 Analyse Recording", type="primary", width='stretch'):
            with st.spinner("Extracting vocal biomarkers…"):
                try:
                    from extract_features import extract_voice_features

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    t0 = time.time()
                    raw_feats = extract_voice_features(tmp_path)
                    elapsed   = time.time() - t0
                    os.unlink(tmp_path)

                    label, prob_parkinsons = predict(raw_feats, model, scaler, feature_names)
                    show_results(label, prob_parkinsons, raw_feats, elapsed)

                except ImportError as ie:
                    st.error(
                        f"Missing dependency: **{ie}**\n\n"
                        "```bash\npip install librosa praat-parselmouth\n```"
                    )
                except Exception as e:
                    st.error(f"Feature extraction failed:\n\n`{e}`")
    else:
        st.info(" Upload a WAV file above to begin analysis.")


# ─── Tab 2: Manual entry ──────────────────────────────────────────────────────
with tab_manual:
    st.markdown("### Enter Voice Features Manually")
    st.caption("Use this if you have pre-computed features from Praat or another acoustic analysis tool.")

    defaults = {
        "MDVP:Fo(Hz)": 154.23,  "MDVP:Fhi(Hz)": 197.10, "MDVP:Flo(Hz)": 116.30,
        "MDVP:Jitter(%)": 0.006, "MDVP:Jitter(Abs)": 0.00004,
        "MDVP:RAP": 0.003,       "MDVP:PPQ": 0.003,       "Jitter:DDP": 0.009,
        "MDVP:Shimmer": 0.030,   "MDVP:Shimmer(dB)": 0.28,
        "Shimmer:APQ3": 0.016,   "Shimmer:APQ5": 0.020,
        "MDVP:APQ": 0.024,       "Shimmer:DDA": 0.047,
        "NHR": 0.014,            "HNR": 21.90,
        "RPDE": 0.50,            "DFA": 0.72,
        "spread1": -5.68,        "spread2": 0.23,
        "D2": 2.38,              "PPE": 0.21,
    }

    cols = st.columns(3)
    user_vals = {}
    for i, (feat, default) in enumerate(defaults.items()):
        with cols[i % 3]:
            user_vals[feat] = st.number_input(feat, value=float(default), format="%.6f", key=f"m_{feat}")

    if st.button(" Run Prediction", type="primary", width='stretch'):
        with st.spinner("Running…"):
            t0 = time.time()
            label, prob_parkinsons = predict(user_vals, model, scaler, feature_names)
            elapsed = time.time() - t0
        show_results(label, prob_parkinsons, user_vals, elapsed)
