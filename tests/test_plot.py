import colorsys
import copy

import matplotlib

matplotlib.use("Agg")
import pytest
from matplotlib import pyplot as plt
from PIL import Image

from plot_llm_value.data import (
    SelectionError,
    _dominates,
    compare,
    normalize,
    pareto_frontier,
)
from plot_llm_value.plot import (
    CAPABILITY_LINE_LIMIT,
    PARETO_COLOR,
    build_figure,
    family_colors,
    save_plot,
)


def comparison(raw):
    result = compare(normalize(raw))
    result["source"] = {
        "name": "Artificial Analysis",
        "url": "https://artificialanalysis.ai/",
        "retrieved_at": "2026-09-09T12:00:00+00:00",
        "intelligence_index_version": 4.3,
    }
    return result


def test_log_plot_connects_efforts_in_order_and_labels_each_family(raw_models):
    fig, info = build_figure(comparison(raw_models))
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    alpha = [line for line in ax.lines if line.get_gid() == "series:openai:Alpha"]
    assert len(alpha) == 2
    assert list(alpha[0].get_xdata()) == [0.5, 1]
    assert list(alpha[1].get_xdata()) == [1, 2]
    assert "Alpha" in [t.get_text() for t in ax.texts]
    assert "Beta" in [t.get_text() for t in ax.texts]
    assert {"low", "medium", "high"} <= {t.get_text() for t in ax.texts}
    assert info["plotted_count"] == 5
    # Only a missing score removes a row entirely; a missing cost still shows capability.
    assert [row["name"] for row in info["excluded"]] == ["Gamma (max)"]
    assert {row["name"]: row["reason"] for row in info["capability_only"]} == {
        "Gamma (Non-reasoning, high)": "cost unavailable",
        "Free Model": "zero cost on log x",
    }
    beta = [line for line in ax.lines if line.get_gid() == "series:anthropic:Beta"]
    assert beta[0].get_linestyle() == "--"
    assert ax.get_legend() is not None
    plt.close(fig)


def test_pareto_line_is_drawn_by_default_over_the_plotted_rows(raw_models):
    fig, info = build_figure(comparison(raw_models))
    ax = fig.axes[0]
    (line,) = [line for line in ax.lines if line.get_gid() == "pareto"]
    # The zero-cost row cannot appear on a log axis, so it also cannot dominate
    # the cheapest plotted point from offstage.
    assert list(zip(line.get_xdata(), line.get_ydata())) == [
        (0.5, 35),
        (0.8, 38),
        (1.0, 42),
        (2.0, 45),
        (3.0, 48),
    ]
    assert [row["name"] for row in info["pareto_frontier"]] == [
        "Alpha (low)",
        "Beta (low with fallback)",
        "Alpha (medium)",
        "Alpha (high)",
        "Beta (max with fallback)",
    ]
    # Dashed and grey, so it never reads as one more colored release series.
    assert line.get_linestyle() != "-" and line.get_color() == PARETO_COLOR
    assert "Pareto frontier" in [text.get_text() for text in ax.get_legend().get_texts()]
    plt.close(fig)


def test_dominated_models_stay_off_the_pareto_line(raw_models):
    raw_models[2]["evaluations"]["artificial_analysis_intelligence_index"] = 30
    fig, info = build_figure(comparison(raw_models))
    (line,) = [line for line in fig.axes[0].lines if line.get_gid() == "pareto"]
    assert 1.0 not in list(line.get_xdata())
    assert "Alpha (medium)" not in [row["name"] for row in info["pareto_frontier"]]
    plt.close(fig)


def test_linear_axis_puts_the_zero_cost_row_on_the_pareto_line(raw_models):
    fig, info = build_figure(comparison(raw_models), linear_x=True)
    (line,) = [line for line in fig.axes[0].lines if line.get_gid() == "pareto"]
    assert list(line.get_xdata())[0] == 0
    assert info["pareto_frontier"][0]["name"] == "Free Model"
    plt.close(fig)


def test_no_pareto_omits_the_line_but_still_reports_the_frontier(raw_models):
    fig, info = build_figure(comparison(raw_models), pareto=False)
    ax = fig.axes[0]
    assert not [line for line in ax.lines if line.get_gid() == "pareto"]
    assert "Pareto frontier" not in [text.get_text() for text in ax.get_legend().get_texts()]
    assert len(info["pareto_frontier"]) == 5
    plt.close(fig)


def test_linear_keeps_zero_and_single_row_renders(raw_models):
    fig, info = build_figure(comparison(raw_models[-1:]), linear_x=True)
    assert fig.axes[0].get_xscale() == "linear"
    assert info["plotted_count"] == 1
    assert fig.axes[0].get_xlim()[0] <= 0
    plt.close(fig)


def test_missing_cost_becomes_a_horizontal_capability_line_off_the_frontier(raw_models):
    fig, info = build_figure(comparison(raw_models))
    ax = fig.axes[0]
    flat = {line.get_gid(): line for line in ax.lines if str(line.get_gid()).startswith("capab")}
    assert set(flat) == {"capability:id-5", "capability:id-7"}
    # Gamma non-reasoning scores 20 with no cost; Free Model scores 10 at zero cost.
    assert [line.get_ydata()[0] for line in flat.values()] == [20, 10]
    assert all(line.get_linestyle() == "-" and line.get_linewidth() < 2 for line in flat.values())
    labels = [text.get_text() for text in ax.texts]
    # The line already means "no cost"; only a measured zero cost needs saying.
    assert "Gamma (Non-reasoning, high)" in labels
    assert "Free Model · zero cost on log x" in labels
    assert not {"id-5", "id-7"} & {row["id"] for row in info["pareto_frontier"]}
    plt.close(fig)


def test_unpriced_effort_inside_a_gap_is_an_interpolated_estimate_with_an_error_bar(raw_models):
    gap_filler = copy.deepcopy(raw_models[4])
    gap_filler.update(id="id-gap", name="Beta (medium with fallback)")
    gap_filler["evaluations"]["artificial_analysis_intelligence_index"] = 44
    gap_filler["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = None
    fig, info = build_figure(comparison(raw_models + [gap_filler]))
    ax = fig.axes[0]
    (estimate,) = [c for c in ax.collections if c.get_gid() == "estimated:anthropic:Beta"]
    # Rank 3 of the low(2) -> max(6) gap: a quarter of the way in log cost.
    (point,) = estimate.get_offsets().tolist()
    assert point == pytest.approx([0.8 * (3.0 / 0.8) ** 0.25, 44])
    # The error bar spans the two measured neighbours the estimate sits between.
    (bars,) = [c for c in ax.collections if c.get_gid() == "range:anthropic:Beta"]
    assert bars.get_segments()[0][:, 0].tolist() == pytest.approx([0.8, 3.0])
    assert "medium with fallback" in [text.get_text() for text in ax.texts]
    assert info["inferred_cost"] == [
        {
            "id": "id-gap",
            "name": "Beta (medium with fallback)",
            "score": 44,
            "inferred_cost_usd": pytest.approx(0.8 * (3.0 / 0.8) ** 0.25),
            "cost_range_usd": pytest.approx([0.8, 3.0]),
            "interpolated_between": ["Beta (low with fallback)", "Beta (max with fallback)"],
        }
    ]
    assert "id-gap" not in [row["id"] for row in info["capability_only"]]
    assert "id-gap" not in [row["id"] for row in info["pareto_frontier"]]
    plt.close(fig)


def test_no_capability_lines_hides_the_lines_but_keeps_the_rows_in_the_result(raw_models):
    fig, info = build_figure(comparison(raw_models), capability_lines=False)
    ax = fig.axes[0]
    assert not [line for line in ax.lines if str(line.get_gid()).startswith("capability:")]
    labels = [text.get_text() for text in ax.texts]
    assert "Gamma (Non-reasoning, high)" not in labels
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    # Gamma has nothing drawn, so it gets no legend entry; the explainer goes too.
    assert "Gamma" not in legend and "Capability only (no measured cost)" not in legend
    assert info["capability_lines_drawn"] is False
    assert [row["name"] for row in info["capability_only"]] == [
        "Gamma (Non-reasoning, high)",
        "Free Model",
    ]
    plt.close(fig)


def test_too_many_capability_lines_are_omitted_to_keep_the_plot_readable(raw_models):
    rows = raw_models[:5]
    for i in range(CAPABILITY_LINE_LIMIT + 1):
        row = copy.deepcopy(raw_models[0])
        row.update(id=f"unpriced-{i}", name=f"Unpriced {i}")
        row["evaluations"]["artificial_analysis_intelligence_index"] = 20 + i
        row["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = None
        rows.append(row)
    fig, info = build_figure(comparison(rows))
    ax = fig.axes[0]
    assert not [line for line in ax.lines if str(line.get_gid()).startswith("capability:")]
    assert info["capability_lines_drawn"] is False
    assert len(info["capability_only"]) == CAPABILITY_LINE_LIMIT + 1
    assert f"more than {CAPABILITY_LINE_LIMIT}" in fig.texts[-1].get_text()
    # One fewer, and they are drawn again.
    fig2, info2 = build_figure(comparison(rows[:-1]))
    assert info2["capability_lines_drawn"] is True
    assert (
        len([line for line in fig2.axes[0].lines if str(line.get_gid()).startswith("capability:")])
        == CAPABILITY_LINE_LIMIT
    )
    plt.close(fig)
    plt.close(fig2)


def test_no_coordinates_fails_instead_of_blank_image(raw_models):
    with pytest.raises(SelectionError):
        build_figure(comparison(raw_models[5:7]))


def test_same_release_from_different_providers_never_connects(raw_models):
    clone = copy.deepcopy(raw_models[0])
    clone["id"] = "clone"
    clone["model_creator"] = {"id": "other", "name": "Other"}
    fig, _ = build_figure(comparison(raw_models[:3] + [clone]))
    series = [line for line in fig.axes[0].lines if line.get_gid().startswith("series:")]
    assert series and all(len(line.get_xdata()) == 2 for line in series)
    groups = [
        c for c in fig.axes[0].collections if c.get_gid() and c.get_gid().startswith("points:")
    ]
    assert len(groups) == 2
    assert not (
        groups[0].get_paths()[0].vertices.shape == groups[1].get_paths()[0].vertices.shape
        and (groups[0].get_paths()[0].vertices == groups[1].get_paths()[0].vertices).all()
    )
    plt.close(fig)


@pytest.mark.parametrize("linear", [False, True])
def test_dense_labels_do_not_overlap_labels_points_or_series(raw_models, linear):
    rows = []
    for i in range(24):
        row = copy.deepcopy(raw_models[0])
        row.update(id=f"dense-{i}", name=f"Release {i // 3} ({['low', 'medium', 'high'][i % 3]})")
        row["evaluations"]["artificial_analysis_intelligence_index"] = 40 + (i % 4) * 0.05
        row["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = (
            1 + (i % 5) * 0.005
        )
        rows.append(row)
    fig, _ = build_figure(comparison(rows), linear_x=linear)
    ax = fig.axes[0]
    fig.canvas.draw()
    boxes = [t.get_window_extent(fig.canvas.get_renderer()) for t in ax.texts]
    assert len(boxes) == 32
    for i, box in enumerate(boxes):
        assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
        assert not any(box.overlaps(other) for other in boxes[i + 1 :])
        for collection in ax.collections:
            for point in ax.transData.transform(collection.get_offsets()):
                assert not box.padded(6).contains(*point)
        for line in ax.lines:
            if line.get_gid() and line.get_gid().startswith(("series:", "pareto")):
                assert (
                    not line.get_path()
                    .transformed(line.get_transform())
                    .intersects_bbox(box.padded(1))
                )
    plt.close(fig)


def test_every_drawn_coordinate_traces_back_to_a_row(raw_models):
    """Nothing is invented, nothing is silently dropped, nothing lands off-canvas."""
    gap_filler = copy.deepcopy(raw_models[4])
    gap_filler.update(id="id-gap", name="Beta (medium with fallback)")
    gap_filler["evaluations"]["artificial_analysis_intelligence_index"] = 44
    gap_filler["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = None
    result = comparison(raw_models + [gap_filler])
    rows = {row["id"]: row for row in result["rows"]}
    fig, info = build_figure(result)
    ax = fig.axes[0]

    # 1. Every row is accounted for exactly once, in exactly one role.
    roles = (
        [row["id"] for row in info["excluded"]]
        + [row["id"] for row in info["capability_only"]]
        + [row["id"] for row in info["inferred_cost"]]
    )
    assert len(roles) == len(set(roles))
    assert info["plotted_count"] + len(roles) == len(result["rows"])

    # 2. Measured points sit exactly at their row's (cost, score); no others exist.
    drawn = set()
    for collection in ax.collections:
        gid = collection.get_gid() or ""
        if gid.startswith("points:"):
            drawn |= {tuple(point) for point in collection.get_offsets().tolist()}
    assert drawn == {
        (row["cost_per_task_usd"], row["score"])
        for row in result["rows"]
        if row["comparable"] and row["cost_per_task_usd"] > 0
    }

    # 3. Capability lines sit at their row's measured score.
    for line in ax.lines:
        gid = line.get_gid() or ""
        if gid.startswith("capability:"):
            assert line.get_ydata()[0] == rows[gid.removeprefix("capability:")]["score"]

    # 4. A hollow point keeps its measured score and only guesses cost, inside the gap.
    by_name = {row["name"]: row for row in result["rows"]}
    for guess in info["inferred_cost"]:
        assert guess["score"] == rows[guess["id"]]["score"]
        start, end = (by_name[name] for name in guess["interpolated_between"])
        assert start["cost_per_task_usd"] < guess["inferred_cost_usd"] < end["cost_per_task_usd"]

    # 5. The frontier is exactly the undominated set of what is drawn, in cost order.
    frontier = info["pareto_frontier"]
    plotted = [row for row in result["rows"] if row["comparable"] and row["cost_per_task_usd"] > 0]
    assert [f["id"] for f in frontier] == [
        row["id"] for row in pareto_frontier(plotted) if row["cost_per_task_usd"] > 0
    ]
    costs = [f["cost_per_task_usd"] for f in frontier]
    scores = [f["score"] for f in frontier]
    assert costs == sorted(costs) and scores == sorted(scores)
    for member in frontier:
        assert not any(_dominates(other, rows[member["id"]]) for other in plotted)
    off_frontier = [row for row in plotted if row["id"] not in {f["id"] for f in frontier}]
    for row in off_frontier:
        assert any(_dominates(rows[f["id"]], row) for f in frontier)

    # 6. Every text is a real family, effort, or capability label - never a stray string.
    allowed = (
        {row["family"] for row in result["rows"]}
        | {row["effort"] for row in result["rows"] if row["effort"]}
        | {f"{row['effort']}?" for row in result["rows"] if row["effort"]}
        | {row["name"] for row in result["rows"]}
        | {f"{row['name']} · zero cost on log x" for row in result["rows"]}
    )
    assert {text.get_text() for text in ax.texts} <= allowed

    # 7. Nothing is drawn outside the visible axes.
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for point in drawn | {(g["inferred_cost_usd"], g["score"]) for g in info["inferred_cost"]}:
        assert x0 <= point[0] <= x1 and y0 <= point[1] <= y1
    for line in ax.lines:
        if (line.get_gid() or "").startswith("capability:"):
            assert y0 <= line.get_ydata()[0] <= y1
    plt.close(fig)


def test_saved_png_and_svg_are_real_artifacts(raw_models, tmp_path):
    for extension in ("png", "svg", "pdf"):
        path = tmp_path / f"plot.{extension}"
        result = save_plot(comparison(raw_models), path)
        assert result["image_path"] == str(path)
        assert path.stat().st_size > 1000
    with Image.open(tmp_path / "plot.png") as image:
        assert image.width / image.height < 1.55
    assert "<svg" in (tmp_path / "plot.svg").read_text()
    assert (tmp_path / "plot.pdf").read_bytes().startswith(b"%PDF")


def test_log_axis_has_readable_ticks_within_one_decade(raw_models):
    fig, _ = build_figure(comparison(raw_models))
    ax = fig.axes[0]
    low, high = ax.get_xlim()
    visible = [tick for tick in ax.get_xticks() if low <= tick <= high]
    assert len(visible) >= 3
    plt.close(fig)


def test_large_catalog_legend_clears_footer_and_axis_labels(raw_models):
    rows = []
    for i in range(40):
        row = copy.deepcopy(raw_models[0])
        row.update(id=f"legend-{i}", name=f"Model Release {i}")
        row["evaluations"]["artificial_analysis_intelligence_index"] = i
        row["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = i + 1
        rows.append(row)
    fig, _ = build_figure(comparison(rows))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = fig.axes[0].get_legend().get_window_extent(renderer)
    assert not legend.overlaps(fig.texts[-1].get_window_extent(renderer))
    assert not legend.overlaps(fig.axes[0].xaxis.label.get_window_extent(renderer))
    assert fig.bbox.contains(legend.x0, legend.y0) and fig.bbox.contains(legend.x1, legend.y1)
    plt.close(fig)


def test_leader_lines_do_not_cross_text(raw_models):
    rows = []
    for i in range(18):
        row = copy.deepcopy(raw_models[0])
        row.update(id=f"leaders-{i}", name=f"Long Named Release {i} (high)")
        rows.append(row)
    fig, _ = build_figure(comparison(rows))
    fig.canvas.draw()
    ax = fig.axes[0]
    boxes = [text.get_window_extent(fig.canvas.get_renderer()) for text in ax.texts]
    assert ax.patches
    for patch in ax.patches:
        path = patch.get_path().transformed(patch.get_transform())
        assert not any(path.intersects_bbox(box) for box in boxes)
    plt.close(fig)


def test_each_provider_has_one_hue_and_the_best_model_is_the_darkest():
    comparison = {
        "rows": [
            {"provider_id": "a", "provider": "Anthropic"},
            {"provider_id": "g", "provider": "Google"},
        ]
    }
    keys = [("a", "Opus"), ("a", "Haiku"), ("g", "Argon")]
    best_score = {("a", "Opus"): 58.0, ("a", "Haiku"): 36.0, ("g", "Argon"): 50.0}
    colors = family_colors(comparison, keys, best_score)
    hls = {key: colorsys.rgb_to_hls(*rgb) for key, rgb in colors.items()}
    assert hls[("a", "Opus")][0] == pytest.approx(hls[("a", "Haiku")][0])
    assert hls[("a", "Opus")][1] < hls[("a", "Haiku")][1]
    assert hls[("g", "Argon")][0] != pytest.approx(hls[("a", "Opus")][0])


def test_lowest_unpriced_effort_is_estimated_from_other_families_low_to_next_ratio(raw_models):
    medium = copy.deepcopy(raw_models[2])
    medium.update(id="id-delta-medium", name="Delta (medium)")
    medium["model_creator"] = {"id": "delta", "name": "Delta"}
    medium["evaluations"]["artificial_analysis_intelligence_index"] = 40
    medium["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = 2.0
    low = copy.deepcopy(raw_models[1])
    low.update(id="id-delta-low", name="Delta (low)")
    low["model_creator"] = {"id": "delta", "name": "Delta"}
    low["evaluations"]["artificial_analysis_intelligence_index"] = 30
    low["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = None
    # Two other families give low/medium ratios of 0.5 (Alpha) and 0.4 (Epsilon).
    eps_medium = copy.deepcopy(raw_models[2])
    eps_medium.update(id="id-eps-medium", name="Epsilon (medium)")
    eps_medium["model_creator"] = {"id": "epsilon", "name": "Epsilon"}
    eps_low = copy.deepcopy(raw_models[1])
    eps_low.update(id="id-eps-low", name="Epsilon (low)")
    eps_low["model_creator"] = {"id": "epsilon", "name": "Epsilon"}
    eps_low["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = 0.4
    eps_medium["artificial_analysis_intelligence_index_cost"]["cost_per_task"]["total_cost"] = 1.0
    fig, info = build_figure(comparison(raw_models + [medium, low, eps_medium, eps_low]))
    ax = fig.axes[0]
    # Median of 0.5 and 0.4 is 0.45, so the estimate is Delta's medium cost (2.0) times 0.45;
    # the error bar runs from 0.4 to 0.5 times 2.0.
    (estimate,) = [c for c in ax.collections if c.get_gid() == "estimated:delta:Delta"]
    assert estimate.get_offsets().tolist() == [pytest.approx([0.9, 30])]
    (bars,) = [c for c in ax.collections if c.get_gid() == "range:delta:Delta"]
    assert bars.get_segments()[0][:, 0].tolist() == pytest.approx([0.8, 1.0])
    assert [line for line in ax.lines if line.get_gid() == "capability:id-delta-low"] == []
    (connector,) = [line for line in ax.lines if line.get_gid() == "estimate:delta:Delta"]
    assert connector.get_linestyle() == "-"
    assert info["inferred_cost"] == [
        {
            "id": "id-delta-low",
            "name": "Delta (low)",
            "score": 30,
            "inferred_cost_usd": pytest.approx(0.9),
            "cost_range_usd": pytest.approx([0.8, 1.0]),
            "estimated_from": "Delta (medium)",
            "cost_ratio_low_to_next": pytest.approx(0.45),
            "cost_ratio_range": pytest.approx([0.4, 0.5]),
        }
    ]
