# docs

Technical description of TimeSeriousDoctor, written as a short research article (7 pages, two columns) with TikZ / PGFPlots figures.

```
docs/
  README.md              this file
  paper/
    main.tex             the article (sections: introduction, method, evaluation protocol, implementation, limitations)
    main.pdf             compiled output, committed so it can be read without LaTeX
    figures/
      fig1_architecture.tex        pipeline flow, "code measures / Gemma explains" split
      fig2_windowing_gate.tex      residual windows and anomaly gate on an illustrative series
      fig3_failure_signatures.tex  the four failure signatures (2x2 small multiples)
      fig4_gemma_diagnosis.tex     the Gemma 4 diagnosis step (inputs, prompt, call, validation)
      fig5_eval_design.tex         evaluation design: pipeline vs Gemma alone
```

Every figure is one self-contained TikZ picture that `main.tex` pulls in with `\input`. The curves in figures 2 and 3 are closed-form functions defined inside the figure files; there are no data files, so the paper builds anywhere.

## Build

Use [Tectonic](https://tectonic-typesetting.github.io), a self-contained LaTeX engine that downloads the packages it needs on first use (needs network once):

```
tectonic docs/paper/main.tex
```

This writes `docs/paper/main.pdf`. Tectonic was used to build the committed PDF (XeTeX engine). Packages used: `mathpazo`, `helvet`, `amsmath`, `booktabs`, `microtype`, `enumitem`, `caption`, `tikz`, `pgfplots` (+ `groupplots`), `hyperref`.

Alternatively, upload the `docs/paper/` folder (`main.tex` and `figures/`) to [Overleaf](https://www.overleaf.com) and compile `main.tex`.

## What is in the paper, and what is not

The paper describes the method as specified in `SPEC.md` (sections 1, 2, 6, 7.3 to 7.11 and 8): residual scaling, windows and gate, the ten failure features, the embedders, clustering, the Gemma 4 diagnosis, the expert knowledge loop, the evaluation protocol and the implementation. It also reports the Gemma smoke-test facts recorded in `NOTES.md` (P0.2).

It contains **no experimental results**. Section 3.4 has two template tables (`tab:compare`, `tab:cluster`) whose cells read `tbd`. When the final runs exist, fill them from `eval.json` (written by `scripts/run_eval.py`):

| table cell | `eval.json` key |
|---|---|
| Pipeline / Gemma alone: Acc., Macro-F1 | `comparison.pipeline.accuracy`, `.macro_f1` / `comparison.gemma_alone.accuracy`, `.macro_f1` |
| Gemma calls, sample | `comparison.pipeline.gemma_calls` / `comparison.gemma_alone.gemma_calls` |
| Gemma calls, full data | `comparison.gemma_calls_full_dataset` (the table shows the symbols `k+1` and `N`) |
| Gate F1 | `gate.f1` |
| ARI, NMI, Purity, Noise | `clustering.ari`, `.nmi`, `.purity`, `.noise_share` |
| Naming | `naming_accuracy` |

Two statements in the paper describe optional parts of the system and should be re-checked against the final code before submission: the MOMENT embedder (`moment` / `both`) and the expert feedback loop (`expert_feedback.json`, few-shot examples).
