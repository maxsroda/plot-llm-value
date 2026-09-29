import copy

import pytest

from plot_llm_value.data import SelectionError, compare, normalize, select, split_effort


@pytest.mark.parametrize(
    "name,family,effort",
    [
        ("GPT-5 (xhigh)", "GPT-5", "xhigh"),
        ("Claude Fable 5.1 (max with fallback)", "Claude Fable 5.1", "max with fallback"),
        ("Claude Opus 4.7 (Non-reasoning, high)", "Claude Opus 4.7", "non-reasoning, high"),
        ("Claude Fable 5 (with fallback)", "Claude Fable 5", "with fallback"),
        ("Model (March 2026)", "Model (March 2026)", None),
        ("Model (preview) (high)", "Model (preview)", "high"),
        ("Model (Thinking)", "Model", "thinking"),
        ("Model (Non-reasoning)", "Model", "non-reasoning"),
        ("Model (16k)", "Model", "16k"),
        ("Unspecified", "Unspecified", None),
    ],
)
def test_effort_suffix_preserves_release_qualifiers(name, family, effort):
    assert split_effort(name) == (family, effort)


def test_normalizes_measured_task_cost_without_token_price_substitution(raw_models):
    rows = normalize(raw_models)
    assert rows[0]["cost_per_task_usd"] == 2
    assert rows[5]["cost_per_task_usd"] is None
    assert rows[0]["indices"] == {"intelligence": 45, "coding": 40, "agentic": 46}
    assert rows[0]["family"] == "Alpha"
    assert rows[0]["effort"] == "high"


def test_three_selection_modes_and_union(raw_models):
    rows = normalize(raw_models)
    assert len(select(rows, providers=["openai", "ANTHROPIC"])) == 7
    assert [r["name"] for r in select(rows, models=["alpha"])] == [
        "Alpha (high)",
        "Alpha (low)",
        "Alpha (medium)",
    ]
    assert [r["id"] for r in select(rows, variants=["Alpha (low)", "model-3"])] == ["id-1", "id-3"]
    assert len(select(rows, variants=["id-1", "Alpha (low)"])) == 1


@pytest.mark.parametrize(
    "selection",
    [
        dict(models=["Alph"]),
        dict(providers=["missing"]),
        dict(variants=["Alpha"]),
        dict(models=["Alpha", "missing"]),
        dict(models=["Alpha"], providers=["openai"]),
    ],
)
def test_selection_fails_loudly_on_partial_unknown_or_mixed_modes(raw_models, selection):
    with pytest.raises(SelectionError):
        select(normalize(raw_models), **selection)


def test_same_family_name_from_different_creators_is_ambiguous(raw_models):
    clone = copy.deepcopy(raw_models[0])
    clone["id"] = "other-id"
    clone["model_creator"] = {"id": "other", "name": "Other"}
    with pytest.raises(SelectionError, match="ambiguous"):
        select(normalize(raw_models + [clone]), models=["Alpha"])


def test_nulls_zero_and_metric_choice_are_explicit(raw_models):
    result = compare(normalize(raw_models), "coding")
    assert result["rows"][0]["score"] == 40
    assert result["rows"][5]["exclusion_reasons"] == ["missing cost"]
    assert result["rows"][6]["exclusion_reasons"] == ["missing score"]
    assert result["rows"][7]["cost_per_task_usd"] == 0
    assert result["rows"][7]["comparable"] is True
    assert compare(normalize(raw_models), "agent")["metric"]["key"] == "agentic"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, "2", True])
def test_invalid_costs_are_not_comparable(raw_models, bad):
    raw_models[0]["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = (
        bad
    )
    result = compare(normalize(raw_models), "intelligence")
    assert result["rows"][0]["comparable"] is False
    assert result["rows"][0]["cost_per_task_usd"] is None


def test_pareto_frontier_keeps_ties_and_excludes_dominated_missing(raw_models):
    raw_models[0]["evaluations"]["artificial_analysis_intelligence_index"] = 42
    raw_models[4]["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = 1
    raw_models[4]["evaluations"]["artificial_analysis_intelligence_index"] = 42
    result = compare(normalize(raw_models), "intelligence")
    assert [r["pareto_optimal"] for r in result["rows"]] == [
        False,
        True,
        True,
        True,
        True,
        None,
        None,
        True,
    ]


def test_null_nested_objects_are_unavailable_not_crashes(raw_models):
    raw_models[0]["evaluations"] = None
    raw_models[0]["artificial_analysis_intelligence_index_cost"] = None
    assert compare(normalize(raw_models), "intelligence")["rows"][0]["exclusion_reasons"] == [
        "missing score",
        "missing cost",
    ]


def test_nonreasoning_sub_efforts_sort_low_to_high():
    from plot_llm_value.data import effort_order

    settings = ["high", "non-reasoning, high", "low", "non-reasoning, low", "non-reasoning, medium"]
    assert sorted(settings, key=effort_order) == [
        "non-reasoning, low",
        "non-reasoning, medium",
        "non-reasoning, high",
        "low",
        "high",
    ]


@pytest.mark.parametrize(
    "name,family,effort",
    [
        ("Claude Opus 5 (Adaptive Reasoning, High Effort)", "Claude Opus 5", "high"),
        (
            "Claude Sonnet 4.6 (Non-reasoning, Low Effort)",
            "Claude Sonnet 4.6",
            "non-reasoning, low",
        ),
        (
            "Claude Fable 5.1 (Adaptive Reasoning, Xhigh Effort, Default Fallback)",
            "Claude Fable 5.1",
            "xhigh with fallback",
        ),
        (
            "Claude Fable 5 (Adaptive Reasoning, Max Effort, Opus 4.8 Fallback)",
            "Claude Fable 5",
            "max with opus 4.8 fallback",
        ),
    ],
)
def test_live_aa_verbose_effort_names(name, family, effort):
    assert split_effort(name) == (family, effort)


def test_named_fallback_preserves_effort_order():
    from plot_llm_value.data import effort_order

    assert effort_order("low with opus 4.8 fallback") < effort_order("high")


@pytest.mark.parametrize(
    "names,labels",
    [
        (["GLM-5.2 (max)", "GLM-5.3 (max)"], ["5.2 (max)", "5.3 (max)"]),
        (["Alpha (low)", "Beta (high)"], ["Alpha (low)", "Beta (high)"]),
        (["Model X", "Model Y"], ["X", "Y"]),
    ],
)
def test_series_point_label_cuts_shared_prefix_at_a_word_boundary(names, labels):
    from plot_llm_value.data import _point_label

    assert [_point_label(name, names) for name in names] == labels
