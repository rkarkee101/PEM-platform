from pathlib import Path

import pandas as pd

from pem.core.config import ProjectConfig
from pem.core.data import load_dataset, validate_dataset
from pem.doe.design import full_factorial_2level
from pem.doe.analyze import analyze_doe
from pem.model.surrogate import ModelPattern, train_bundle
from pem.optimize.recommend import recommend_next
from pem.rag.index import build_index


ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_smoke():
    cfg = ProjectConfig.from_yaml(ROOT / "examples" / "configs" / "tool_demo.yaml")
    df = load_dataset(ROOT / "examples" / "datasets" / "demo_single_response.csv")
    validate_dataset(df, cfg, strict=False)

    # DoE design
    design = full_factorial_2level(cfg)
    assert len(design.natural) == 16
    assert set(design.natural.columns).issuperset({"temperature_C", "pressure_mTorr", "rf_power_W", "gas_ratio"})

    # DoE analysis
    doe_res = analyze_doe(df, cfg, response="thickness_nm", max_interaction_order=2)
    assert "term" in doe_res.coef_table.columns
    assert "effect" in doe_res.coef_table.columns

    # Train model
    bundle = train_bundle(df, cfg, responses=["thickness_nm"], pattern=ModelPattern.trend_residual_gp)
    pred = bundle.predict(df[cfg.factor_names(include_categorical=True)])
    assert "thickness_nm__mean" in pred.columns
    assert "thickness_nm__std" in pred.columns

    # Recommend
    rec = recommend_next(bundle, cfg, n_candidates=300, n_recommendations=5)
    assert len(rec.ranked) == 5


def test_rag_index_build_and_query():
    idx = build_index(ROOT / "examples" / "knowledge", backend="tfidf")
    hits = idx.query("RF power", top_k=3)
    assert len(hits) >= 1
