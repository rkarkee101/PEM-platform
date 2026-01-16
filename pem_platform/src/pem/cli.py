from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from pem.core.config import ProjectConfig
from pem.core.data import load_dataset, validate_dataset
from pem.doe.analyze import analyze_doe
from pem.doe.design import fractional_factorial_2level, full_factorial_2level, plackett_burman, latin_hypercube
from pem.model.metrics import cross_validate
from pem.model.surrogate import ModelPattern, TrainedModelBundle, train_bundle
from pem.optimize.ax_bo import recommend_next_ax
from pem.optimize.recommend import recommend_next
from pem.rag.index import build_index, RagIndex
from pem.rag.chat import run_retrieval_chat
from pem.viz.doe_plots import save_pareto_effects_plot
from pem.viz.model_plots import save_parity_plot, save_residuals_plot
from pem.viz.doe_plots import save_pareto_effects_plot
from pem.viz.model_plots import save_parity_plot, save_residuals_plot


app = typer.Typer(add_completion=False, help="PEM Platform CLI")
console = Console()


def _load_cfg(config: str) -> ProjectConfig:
    return ProjectConfig.from_yaml(config)


def _print_dataframe(df: pd.DataFrame, title: str, max_rows: int = 20) -> None:
    tbl = Table(title=title, show_lines=False)
    for c in df.columns:
        tbl.add_column(str(c))
    for _, row in df.head(max_rows).iterrows():
        tbl.add_row(*[str(row[c]) for c in df.columns])
    console.print(tbl)
    if len(df) > max_rows:
        console.print(f"(showing first {max_rows} of {len(df)} rows)")


@app.command("validate-data")
def validate_data(
    config: str = typer.Option(..., "--config", help="Path to project YAML config"),
    data: str = typer.Argument(..., help="Dataset CSV or Parquet"),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Fail on unknown levels / non-numeric"),
):
    cfg = _load_cfg(config)
    df = load_dataset(data)
    validate_dataset(df, cfg, strict=strict)
    console.print("Data validated successfully.")


design_app = typer.Typer(help="DoE design generation")
app.add_typer(design_app, name="design")


@design_app.command("full-factorial")
def design_full_factorial(
    config: str = typer.Option(..., "--config"),
    output: str = typer.Option("design_full.csv", "--output"),
):
    cfg = _load_cfg(config)
    res = full_factorial_2level(cfg)
    res.natural.to_csv(output, index=False)
    console.print(f"Wrote {len(res.natural)} runs to {output}")


@design_app.command("fractional-factorial")
def design_fractional_factorial(
    config: str = typer.Option(..., "--config"),
    generator: str = typer.Option(..., "--generator", help='pyDOE2 generator, e.g. "a b c ab"'),
    output: str = typer.Option("design_frac.csv", "--output"),
):
    cfg = _load_cfg(config)
    res = fractional_factorial_2level(cfg, generator=generator)
    res.natural.to_csv(output, index=False)
    console.print(f"Wrote {len(res.natural)} runs to {output}")


@design_app.command("plackett-burman")
def design_pb(
    config: str = typer.Option(..., "--config"),
    output: str = typer.Option("design_pb.csv", "--output"),
):
    cfg = _load_cfg(config)
    res = plackett_burman(cfg)
    res.natural.to_csv(output, index=False)
    console.print(f"Wrote {len(res.natural)} runs to {output}")


@design_app.command("lhs")
def design_lhs(
    config: str = typer.Option(..., "--config"),
    n_samples: int = typer.Option(20, "--n-samples"),
    output: str = typer.Option("design_lhs.csv", "--output"),
    seed: int = typer.Option(0, "--seed"),
):
    cfg = _load_cfg(config)
    df = latin_hypercube(cfg, n_samples=n_samples, seed=seed)
    df.to_csv(output, index=False)
    console.print(f"Wrote {len(df)} runs to {output}")


doe_app = typer.Typer(help="DoE analysis")
app.add_typer(doe_app, name="doe")


@doe_app.command("analyze")
def doe_analyze(
    config: str = typer.Option(..., "--config"),
    data: str = typer.Option(..., "--data"),
    response: str = typer.Option(..., "--response"),
    max_interaction_order: int = typer.Option(2, "--max-interaction-order"),
    out_prefix: Optional[str] = typer.Option(None, "--out-prefix", help="If set, saves CSVs with this prefix"),
):
    cfg = _load_cfg(config)
    df = load_dataset(data)
    validate_dataset(df, cfg, strict=False)

    res = analyze_doe(df, cfg, response=response, max_interaction_order=max_interaction_order)

    console.print(f"Formula: {res.formula}")
    _print_dataframe(res.coef_table, title="Coefficients / Effects")
    _print_dataframe(res.anova_table, title="ANOVA")

    if len(res.alias_warnings) > 0:
        _print_dataframe(res.alias_warnings, title="Alias warnings (corr-based)")
    else:
        console.print("No strong alias indicators found (corr threshold).")

    if out_prefix:
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
        res.coef_table.to_csv(f"{out_prefix}_coef.csv", index=False)
        res.anova_table.to_csv(f"{out_prefix}_anova.csv", index=False)
        res.alias_warnings.to_csv(f"{out_prefix}_alias.csv", index=False)
        console.print(f"Saved CSVs with prefix: {out_prefix}_*.csv")


model_app = typer.Typer(help="Surrogate modeling")
app.add_typer(model_app, name="model")


@model_app.command("train")
def model_train(
    config: str = typer.Option(..., "--config"),
    data: str = typer.Option(..., "--data"),
    responses: List[str] = typer.Option(..., "--responses", help="One or more response column names"),
    pattern: ModelPattern = typer.Option(ModelPattern.pure_gp, "--pattern"),
    top_k: int = typer.Option(5, "--top-k", help="Used for doe_screen_then_gp"),
    kernel_kind: str = typer.Option("matern", "--kernel-kind", help="matern or rbf"),
    nu: float = typer.Option(2.5, "--nu", help="Matern nu (ignored for rbf)"),
    n_restarts_optimizer: int = typer.Option(
        2, "--n-restarts", help="GPR hyperparameter optimizer restarts"
    ),
    ridge_alpha: float = typer.Option(
        1.0, "--ridge-alpha", help="Ridge alpha for trend_residual_gp"
    ),
    out: str = typer.Option("model.joblib", "--out"),
):
    cfg = _load_cfg(config)
    df = load_dataset(data)
    validate_dataset(df, cfg, strict=False)

    bundle = train_bundle(
        df,
        cfg,
        responses=responses,
        pattern=pattern,
        top_k=top_k,
        kernel_kind=kernel_kind,
        nu=nu,
        n_restarts_optimizer=n_restarts_optimizer,
        ridge_alpha=ridge_alpha,
    )
    bundle.save(out)
    console.print(f"Trained {pattern} for responses={responses}. Saved to {out}")


@model_app.command("predict")
def model_predict(
    config: str = typer.Option(..., "--config"),
    model: str = typer.Option(..., "--model"),
    data: str = typer.Option(..., "--data", help="CSV/Parquet containing factor columns"),
    output: Optional[str] = typer.Option(None, "--output", help="If set, write predictions CSV"),
):
    cfg = _load_cfg(config)
    bundle = TrainedModelBundle.load(model)

    df = load_dataset(data)

    Xcols = cfg.factor_names(include_categorical=True)
    missing = [c for c in Xcols if c not in df.columns]
    if missing:
        raise typer.BadParameter(f"Prediction data is missing factor columns: {missing}")

    X = df[Xcols].copy()
    pred = bundle.predict(X)
    out_df = pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)

    _print_dataframe(out_df, title="Predictions", max_rows=25)

    if output:
        out_df.to_csv(output, index=False)
        console.print(f"Wrote predictions to {output}")


@model_app.command("cv")
def model_cv(
    config: str = typer.Option(..., "--config"),
    data: str = typer.Option(..., "--data"),
    responses: List[str] = typer.Option(..., "--responses"),
    pattern: ModelPattern = typer.Option(ModelPattern.pure_gp, "--pattern"),
    n_splits: int = typer.Option(5, "--n-splits"),
    top_k: int = typer.Option(5, "--top-k", help="Used for doe_screen_then_gp"),
    kernel_kind: str = typer.Option("matern", "--kernel-kind", help="matern or rbf"),
    nu: float = typer.Option(2.5, "--nu", help="Matern nu (ignored for rbf)"),
    n_restarts_optimizer: int = typer.Option(2, "--n-restarts"),
    ridge_alpha: float = typer.Option(1.0, "--ridge-alpha"),
):
    cfg = _load_cfg(config)
    df = load_dataset(data)
    validate_dataset(df, cfg, strict=False)

    summaries = cross_validate(
        df,
        cfg,
        responses=responses,
        pattern=pattern,
        n_splits=n_splits,
        top_k=top_k,
        kernel_kind=kernel_kind,
        nu=nu,
        n_restarts_optimizer=n_restarts_optimizer,
        ridge_alpha=ridge_alpha,
    )
    table = Table(title="Cross-validation metrics")
    table.add_column("response")
    table.add_column("rmse")
    table.add_column("mae")
    table.add_column("r2")
    for s in summaries:
        table.add_row(s.response, f"{s.rmse:.4g}", f"{s.mae:.4g}", f"{s.r2:.4g}")
    console.print(table)


@app.command("recommend")
def recommend(
    config: str = typer.Option(..., "--config"),
    model: str = typer.Option(..., "--model"),
    responses: Optional[List[str]] = typer.Option(None, "--responses", help="Responses to optimize"),
    n_candidates: int = typer.Option(2000, "--n-candidates"),
    n_recommendations: int = typer.Option(8, "--n-recommendations"),
    acquisition: str = typer.Option("ucb", "--acquisition", help="ucb or ei"),
    kappa: float = typer.Option(2.0, "--kappa"),
    seed: int = typer.Option(0, "--seed"),
    output: Optional[str] = typer.Option(None, "--output", help="Write recommendations CSV"),
):
    cfg = _load_cfg(config)
    bundle = TrainedModelBundle.load(model)

    res = recommend_next(
        bundle,
        cfg,
        responses=responses,
        n_candidates=n_candidates,
        n_recommendations=n_recommendations,
        acquisition=acquisition,
        kappa=kappa,
        seed=seed,
    )

    _print_dataframe(res.ranked, title="Top recommendations", max_rows=n_recommendations)

    if output:
        res.ranked.to_csv(output, index=False)
        console.print(f"Wrote recommendations to {output}")


rag_app = typer.Typer(help="RAG-style retrieval")
app.add_typer(rag_app, name="rag")


@rag_app.command("index")
def rag_index(
    docs: str = typer.Option(..., "--docs", help="Directory containing .txt/.md/.rst"),
    index_dir: str = typer.Option(..., "--index-dir"),
    backend: str = typer.Option("tfidf", "--backend", help="tfidf or sentence_transformers"),
):
    idx = build_index(docs, backend=backend)
    idx.save(index_dir)
    console.print(f"Indexed docs under {docs}. Saved index to {index_dir}")


@rag_app.command("query")
def rag_query(
    index_dir: str = typer.Option(..., "--index-dir"),
    question: str = typer.Option(..., "--question"),
    top_k: int = typer.Option(5, "--top-k"),
):
    idx = RagIndex.load(index_dir)
    hits = idx.query(question, top_k=top_k)
    if not hits:
        console.print("No results")
        raise typer.Exit(code=1)

    for score, chunk in hits:
        console.print(f"\n[{score:.3f}] {chunk.source} (chunk {chunk.chunk_id})\n")
        console.print(chunk.text)


@rag_app.command("chat")
def rag_chat(
    index_dir: str = typer.Option(..., "--index-dir"),
    top_k: int = typer.Option(5, "--top-k"),
):
    run_retrieval_chat(index_dir, top_k=top_k)


plot_app = typer.Typer(help="Plotting helpers (saves .png files)")
app.add_typer(plot_app, name="plot")


@plot_app.command("doe-effects")
def plot_doe_effects(
    config: str = typer.Option(..., "--config"),
    data: str = typer.Option(..., "--data"),
    response: str = typer.Option(..., "--response"),
    max_interaction_order: int = typer.Option(2, "--max-interaction-order"),
    out_dir: str = typer.Option("plots", "--out-dir"),
    top_n: int = typer.Option(20, "--top-n"),
    p_value_threshold: float = typer.Option(0.05, "--p-value-threshold"),
):
    """Generate DoE effects plots.

    Currently outputs:
    - Pareto chart of absolute effects (from DoE regression)
    """

    cfg = _load_cfg(config)
    df = load_dataset(data)
    validate_dataset(df, cfg, strict=False)

    res = analyze_doe(df, cfg, response=response, max_interaction_order=max_interaction_order)
    out_path = Path(out_dir) / f"doe_{response}_pareto.png"
    save_pareto_effects_plot(
        res.coef_table,
        response=response,
        out_path=out_path,
        top_n=top_n,
        p_value_threshold=p_value_threshold,
    )
    console.print(f"Wrote: {out_path}")


@plot_app.command("model-diagnostics")
def plot_model_diagnostics(
    config: str = typer.Option(..., "--config"),
    model: str = typer.Option(..., "--model"),
    data: str = typer.Option(..., "--data"),
    responses: Optional[List[str]] = typer.Option(None, "--responses"),
    out_dir: str = typer.Option("plots", "--out-dir"),
    with_uncertainty: bool = typer.Option(True, "--with-uncertainty/--no-uncertainty"),
):
    """Generate parity + residual plots for a trained surrogate."""

    cfg = _load_cfg(config)
    bundle = TrainedModelBundle.load(model)
    df = load_dataset(data)

    Xcols = cfg.factor_names(include_categorical=True)
    missing = [c for c in Xcols if c not in df.columns]
    if missing:
        raise typer.BadParameter(f"Diagnostic data is missing factor columns: {missing}")

    plot_responses = list(responses) if responses is not None else list(bundle.responses)

    X = df[Xcols].copy()
    pred = bundle.predict(X)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    for r in plot_responses:
        if r not in df.columns:
            raise typer.BadParameter(f"Diagnostic data is missing response column: {r}")

        y_true = pd.to_numeric(df[r], errors="coerce").to_numpy()
        y_pred = pred[f"{r}__mean"].to_numpy()
        y_std = pred[f"{r}__std"].to_numpy() if with_uncertainty else None

        parity_path = out_dir_p / f"parity_{r}.png"
        resid_path = out_dir_p / f"residual_{r}.png"

        save_parity_plot(
            y_true=y_true,
            y_pred=y_pred,
            y_std=y_std,
            out_path=parity_path,
            title=f"Parity plot: {r}",
        )
        save_residuals_plot(
            y_true=y_true,
            y_pred=y_pred,
            out_path=resid_path,
            title=f"Residual plot: {r}",
        )
        console.print(f"Wrote: {parity_path}")
        console.print(f"Wrote: {resid_path}")


ax_app = typer.Typer(help="Bayesian optimization (Ax / BoTorch)")
app.add_typer(ax_app, name="ax")


@ax_app.command("recommend")
def ax_recommend(
    config: str = typer.Option(..., "--config"),
    data: str = typer.Option(..., "--data"),
    responses: Optional[List[str]] = typer.Option(None, "--responses"),
    n_recommendations: int = typer.Option(8, "--n-recommendations"),
    seed: int = typer.Option(0, "--seed"),
    output: Optional[str] = typer.Option(None, "--output"),
):
    """Recommend next trials using Ax Bayesian optimization.

    This command fits an Ax model to historical runs and proposes new parameter settings.
    Multi-response is handled via scalarization into a single objective called `utility`.

    Requires optional dependency: ax-platform.
    Install with: pip install -e ".[ax]"
    """

    cfg = _load_cfg(config)
    df = load_dataset(data)
    validate_dataset(df, cfg, strict=False)

    try:
        res = recommend_next_ax(
            cfg,
            df,
            responses=responses,
            n_recommendations=n_recommendations,
            seed=seed,
        )
    except ImportError as e:
        console.print(str(e))
        raise typer.Exit(code=1) from e

    console.print(f"Attached {res.n_attached} historical rows to Ax experiment.")
    _print_dataframe(res.suggestions, title="Ax recommendations", max_rows=n_recommendations)

    if output:
        res.suggestions.to_csv(output, index=False)
        console.print(f"Wrote recommendations to {output}")
