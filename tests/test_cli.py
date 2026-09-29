import json
from pathlib import Path

import httpx
import pytest

from plot_llm_value.cli import main


@pytest.fixture
def live_api(monkeypatch, raw_models):
    client_type = httpx.Client
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "tier": "free",
                "intelligence_index_version": 4.3,
                "pagination": {"page": 1, "page_size": 200, "total_pages": 1, "has_more": False},
                "data": raw_models,
            },
        )

    monkeypatch.setattr(
        "plot_llm_value.api.httpx.Client",
        lambda **kw: client_type(transport=httpx.MockTransport(respond), **kw),
    )
    monkeypatch.setenv("ARTIFICIAL_ANALYSIS_API_KEY", "fixture-key")
    return requests


@pytest.mark.parametrize("args", [["--help"], ["-h"], ["help"], ["plot", "--help"], ["data", "-h"]])
def test_help_verbatim_without_credentials_or_network(args, capsys, monkeypatch):
    monkeypatch.delenv("ARTIFICIAL_ANALYSIS_API_KEY", raising=False)
    monkeypatch.delenv("AA_API_KEY", raising=False)
    assert main(args) == 0
    assert capsys.readouterr().out == Path("src/plot_llm_value/help.txt").read_text()


def test_data_selects_exact_variants_and_preserves_metadata(live_api, capsys):
    assert (
        main(["data", "--variant", "Alpha (low)", "Beta (max with fallback)", "--metric", "agent"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["metric"]["key"] == "agentic"
    assert len(result["rows"]) == 2
    assert result["rows"][0]["score"] == 36
    assert result["source"]["intelligence_index_version"] == 4.3
    assert len(live_api) == 1


def test_default_plot_returns_ephemeral_image_path(live_api, capsys):
    assert main(["--provider", "openai"]) == 0
    result = json.loads(capsys.readouterr().out)
    path = Path(result["image_path"])
    assert path.is_absolute() and path.read_bytes().startswith(b"\x89PNG")
    assert result["plotted_count"] == 3
    path.unlink()
    path.parent.rmdir()


def test_plot_reports_the_frontier_and_no_pareto_is_accepted(live_api, capsys, tmp_path):
    path = tmp_path / "comparison.png"
    assert main(["--output", str(path)]) == 0
    assert [row["name"] for row in json.loads(capsys.readouterr().out)["pareto_frontier"]][0] == (
        "Alpha (low)"
    )
    assert main(["--no-pareto", "--output", str(path)]) == 0
    assert len(json.loads(capsys.readouterr().out)["pareto_frontier"]) == 5
    assert main(["--no-capability-lines", "--output", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["capability_lines_drawn"] is False and len(result["capability_only"]) == 2


def test_reasoning_only_drops_marked_variants_and_keeps_unqualified_models(live_api, capsys):
    assert main(["models"]) == 0
    everything = json.loads(capsys.readouterr().out)["rows"]
    assert main(["models", "--reasoning-only"]) == 0
    kept = json.loads(capsys.readouterr().out)
    names = [row["name"] for row in kept["rows"]]
    assert "Gamma (Non-reasoning, high)" in [row["name"] for row in everything]
    assert "Gamma (Non-reasoning, high)" not in names
    # No effort qualifier means a single-configuration model, which is not filtered.
    assert "Free Model" in names
    assert kept["filters"] == {"reasoning_only": True}


def test_reasoning_only_that_empties_the_selection_is_an_error(live_api, capsys):
    assert main(["data", "--variant", "Gamma (Non-reasoning, high)", "--reasoning-only"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "--reasoning-only" in captured.err


def test_repeated_flags_union_models_and_explicit_output(live_api, capsys, tmp_path):
    path = tmp_path / "comparison.svg"
    assert main(["plot", "--model", "Alpha", "--model", "Beta", "--output", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["plotted_count"] == 5
    assert "<svg" in path.read_text()


def test_catalog_retains_missing_scores_and_prints_selectors(live_api, capsys):
    assert main(["models", "--provider", "anthropic"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["rows"]) == 4
    assert {"family", "effort", "name", "id", "slug", "provider"} <= result["rows"][0].keys()


@pytest.mark.parametrize(
    "args",
    [
        ["--provider", "missing"],
        ["--model", "Alpha", "--variant", "id-0"],
        ["data", "--linear-x"],
        ["data", "--no-pareto"],
        ["models", "--no-capability-lines"],
        ["data", "--output", "x.png"],
        ["--timeout", "nan"],
        ["--timeout", "-1"],
        ["--output", "x.txt"],
    ],
)
def test_usage_errors_have_no_success_output(live_api, capsys, args):
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err


def test_missing_key_is_actionable_and_env_file_is_explicit(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("ARTIFICIAL_ANALYSIS_API_KEY", raising=False)
    monkeypatch.delenv("AA_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("AA_API_KEY=not-loaded-automatically\n")
    assert main(["data"]) == 1
    assert "ARTIFICIAL_ANALYSIS_API_KEY" in capsys.readouterr().err


def test_explicit_env_file_and_environment_precedence(live_api, monkeypatch, tmp_path, capsys):
    env = tmp_path / ".env"
    env.write_text('AA_API_KEY="file-key"\n')
    assert main(["data", "--env-file", str(env)]) == 0
    assert live_api[-1].headers["x-api-key"] == "fixture-key"
    monkeypatch.delenv("ARTIFICIAL_ANALYSIS_API_KEY")
    assert main(["data", "--env-file", str(env)]) == 0
    assert live_api[-1].headers["x-api-key"] == "file-key"
    capsys.readouterr()


def test_each_command_fetches_fresh(live_api, capsys):
    for command in ["models", "data", "data"]:
        assert main([command]) == 0
        json.loads(capsys.readouterr().out)
    assert len(live_api) == 3


def test_series_joins_releases_in_given_order_and_adds_to_selection(live_api, capsys):
    argv = ["data", "--variant", "Alpha (low)", "--series", "Joined", "Beta (max with fallback)"]
    assert main(argv + ["Gamma (max)"]) == 0
    rows = {row["name"]: row for row in json.loads(capsys.readouterr().out)["rows"]}
    assert set(rows) == {"Alpha (low)", "Beta (max with fallback)", "Gamma (max)"}
    beta, gamma = rows["Beta (max with fallback)"], rows["Gamma (max)"]
    assert (beta["family"], beta["effort"], beta["series_rank"]) == (
        "Joined",
        "Beta (max with fallback)",
        0,
    )
    assert (gamma["family"], gamma["series_rank"]) == ("Joined", 1)
    assert "series_rank" not in rows["Alpha (low)"]


def test_series_alone_selects_only_its_variants(live_api, capsys):
    assert main(["models", "--series", "A", "Alpha (low)", "Alpha (high)"]) == 0
    rows = json.loads(capsys.readouterr().out)["rows"]
    assert [(r["effort"], r["series_rank"]) for r in rows] == [("(low)", 0), ("(high)", 1)]


@pytest.mark.parametrize(
    "series",
    [
        ["Solo", "Alpha (low)"],
        ["Mixed", "Alpha (low)", "Beta (low with fallback)"],
        ["X", "Alpha (low)", "Nope"],
    ],
)
def test_invalid_series_fails_with_usage_error(series, live_api, capsys):
    assert main(["data", "--series", *series]) == 2
    assert capsys.readouterr().out == ""


def test_series_variant_in_two_series_fails(live_api, capsys):
    argv = ["data", "--series", "A", "Alpha (low)", "Alpha (high)"]
    assert main(argv + ["--series", "B", "Alpha (low)", "Alpha (medium)"]) == 2


def test_x_e2e_uses_measured_response_time_and_its_own_frontier(live_api, capsys):
    assert main(["data", "--provider", "openai", "--x", "e2e"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["x_axis"]["key"] == "e2e"
    assert result["x_axis"]["field"] == "performance.median_end_to_end_response_time_seconds"
    assert {row["x"] for row in result["rows"]} == {5}
    assert all(row["x"] == row["e2e_seconds"] for row in result["rows"])


def test_x_e2e_plot_labels_the_axis_in_seconds(live_api, capsys):
    assert main(["--provider", "openai", "--x", "e2e"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["pareto_frontier"][0]["x"] == 5


def test_x_is_rejected_for_models(live_api):
    assert main(["models", "--x", "e2e"]) == 2
