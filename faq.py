"""FAQ page of the Streamlit app (registered in app.py with st.navigation, linked at the bottom of the sidebar).

A short tour for first-time users and judges: the paper (PDF and LaTeX source) to download, how the pipeline
works, the main features, the explanatory figures from docs/figures and a few questions about the demo.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
PAPER_DIR = ROOT / "docs" / "paper"
FIGURES_DIR = ROOT / "docs" / "figures"
FIGURES = {  # tab label -> (file in docs/figures, caption)
    "From forecast to diagnosis": (
        "01_transformation.png",
        "One real series from forecast to diagnosis: scaled errors and gated windows, feature fingerprint, "
        "place on the map, Gemma's report.",
    ),
    "Latent space": (
        "02_latent_space.png",
        "t-SNE map of all windows: the injected failure types next to the clusters the pipeline found.",
    ),
    "vs Gemma alone": (
        "03_pipeline_vs_gemma_alone.png",
        "Accuracy, macro-F1, per-class accuracy and Gemma calls: the pipeline against the same Gemma model "
        "looking at a plain full-series chart.",
    ),
    "Cluster panels": (
        "05_cluster_gallery.png",
        "The chart panels Gemma sees for the four largest clusters, with Gemma's name and the cluster size.",
    ),
}
QUESTIONS = {
    "Which runs can I explore?": (
        "Pick a run in the sidebar. **synth_features** has 400 synthetic series with injected, labelled "
        "failures, so it is the run with the **Evaluation** tab (pipeline vs Gemma alone). **m5_features** is "
        "real Walmart sales data (M5): 1,000 items forecast by two models. It has no ground truth, so there is "
        "nothing to score, but every cluster is diagnosed the same way."
    ),
    "Where should I start?": (
        "**Atlas** lists the failure patterns, largest first. Press **Open** on a card, then read the "
        "**Cluster** tab: Gemma's diagnosis, example windows, the measured features and the member series. "
        "**Series** shows one forecast with its abnormal windows, **Latent space** shows all windows on one "
        "map (click a dot), **Evaluation** shows the proof, and **Ask the Doctor** answers your own questions."
    ),
    "Does the app call Gemma live?": (
        "Only in **Ask the Doctor**. Everything else reads a run that the command-line pipeline computed once "
        "and saved as a folder, so browsing costs nothing. Ask the Doctor sends your question and the run's "
        "measured numbers to Gemma (`gemma-4-31b-it`); an answer takes about 15 to 70 seconds, and the same "
        "question on the same scope comes back instantly from the cache."
    ),
    "How do I read the error numbers?": (
        "Errors are actual minus forecast, so positive means the forecast was too low. Each series is scaled "
        "by its typical seasonal-naive error (MASE scaling), so 1.0 is about one normal week-over-week change "
        "and numbers are comparable across series."
    ),
    "What are the unclustered windows?": (
        "Abnormal windows that fit none of the patterns. They get no Gemma call; review them by hand."
    ),
    "What does the expert review in the Cluster tab do?": (
        "Confirm, reject or rename Gemma's diagnosis. Verdicts are saved in the run folder and, on the next "
        "run (`--expert-from`), shown to Gemma as reference examples. On this hosted demo, saved reviews are "
        "shared by all visitors and reset when the app restarts."
    ),
}


@st.cache_data
def paper_source_zip() -> bytes:
    """main.tex and the TikZ figure files, as a zip that compiles with tectonic or on Overleaf."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PAPER_DIR.rglob("*.tex")):
            zf.write(path, path.relative_to(PAPER_DIR).as_posix())
    return buf.getvalue()


st.title("FAQ: how TimeSeriousDoctor works")
st.caption("A two-minute tour of the method, the main features and the technical paper.")

st.subheader("The paper")
st.markdown(
    "A short technical report (two columns, 8 pages) with the method, every feature formula, the "
    "evaluation protocol and the results."
)
left, right, _ = st.columns([1, 1, 2])
pdf = PAPER_DIR / "main.pdf"
if pdf.exists():
    left.download_button(
        "Download the paper (PDF)", pdf.read_bytes(), file_name="TimeSeriousDoctor.pdf", mime="application/pdf",
        icon=":material/download:", type="primary", on_click="ignore", width="stretch",
    )
right.download_button(
    "LaTeX source (.zip)", paper_source_zip(), file_name="TimeSeriousDoctor_latex.zip", mime="application/zip",
    icon=":material/code:", on_click="ignore", width="stretch",
)

st.subheader("How it works")
st.markdown(
    "**Code measures, Gemma explains.** Every number in the app is computed by code; Gemma 4 only reads "
    "the numbers and charts it is given and explains them.\n"
    "1. **Residuals.** For every series, error = actual − forecast, scaled so 1.0 is a typical "
    "seasonal-naive error.\n"
    "2. **Windows and gate.** The errors are cut into 28-step windows; only abnormal windows are kept.\n"
    "3. **Failure features.** Twelve measured numbers per window: lag, bias, missed peaks, flatness, "
    "error trend, ...\n"
    "4. **Clusters.** Similar windows are grouped into recurring failure patterns.\n"
    "5. **Gemma diagnosis.** One Gemma call per cluster: chart panel + feature table + the expert's "
    "taxonomy in, a validated diagnosis out (name, evidence, causes, fixes, metric to monitor)."
)
overview = FIGURES_DIR / "04_llm_plus_expert.png"
if overview.exists():
    st.image(str(overview), width="stretch")

st.subheader("Main features")
st.markdown(
    "- **Failure atlas:** thousands of forecasts reduced to a ranked list of explained failure patterns.\n"
    "- **Expert knowledge built in:** `taxonomy.yaml` (definitions, signatures, causes, fixes) goes into "
    "every prompt; expert Confirm / Reject / Rename verdicts become examples for the next run.\n"
    "- **Explorable evidence:** example windows, feature tables, model mix, member series and a clickable "
    "latent-space map behind every diagnosis.\n"
    "- **Ask the Doctor:** ask about the whole run, one cluster or one series; Gemma answers only from that "
    "run's measured evidence.\n"
    "- **Cheap and repeatable:** one Gemma call per cluster instead of one per series, schema-validated "
    "JSON, every response cached on disk.\n"
    "- **Proof built in:** the Evaluation tab scores the pipeline against Gemma alone on the same series."
)

st.subheader("Figures")
for tab, (name, caption) in zip(st.tabs(list(FIGURES)), FIGURES.values()):
    with tab:
        path = FIGURES_DIR / name
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
        else:
            st.info(f"Figure not found: docs/figures/{name}")

st.subheader("Questions")
for question, answer in QUESTIONS.items():
    with st.expander(question):
        st.markdown(answer)
