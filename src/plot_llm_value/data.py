"""Normalize official rows without inventing scores, efforts, or costs."""

import math
import re

from .api import APIError

METRICS = {
    "intelligence": "Intelligence Index",
    "coding": "Coding Index",
    "agentic": "Agentic Index",
}
# Each x-axis is a lower-is-better measurement AA reports per row.
X_AXES = {
    "cost": {
        "field": "artificial_analysis_intelligence_index_cost.cost_per_task.total_cost",
        "row_key": "cost_per_task_usd",
        "label": "Cost per Intelligence Index task",
        "title": "cost per task",
        "unit": "USD",
        "noun": "cost",
        "better": "cheaper",
        "workload": "Artificial Analysis Intelligence Index task",
    },
    "e2e": {
        "field": "performance.median_end_to_end_response_time_seconds",
        "row_key": "e2e_seconds",
        "label": "Median end-to-end response time",
        "title": "end-to-end response time",
        "unit": "s",
        "noun": "time",
        "better": "faster",
        "workload": "AA speed benchmark prompt, not Intelligence Index tasks",
    },
    "first-token": {
        "field": "performance.median_time_to_first_token_seconds",
        "row_key": "first_token_seconds",
        "label": "Median time to first token",
        "title": "time to first token",
        "unit": "s",
        "noun": "time",
        "better": "faster",
        "workload": "AA speed benchmark prompt, not Intelligence Index tasks",
    },
    "first-answer-token": {
        "field": "performance.median_time_to_first_answer_token_seconds",
        "row_key": "first_answer_token_seconds",
        "label": "Median time to first answer token",
        "title": "time to first answer token",
        "unit": "s",
        "noun": "time",
        "better": "faster",
        "workload": "AA speed benchmark prompt, not Intelligence Index tasks",
    },
}
EFFORTS = {
    "none": 0,
    "non-reasoning": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "xhigh": 5,
    "max": 6,
    "ultra": 7,
    "thinking": 4,
    "reasoning": 4,
}


class SelectionError(Exception):
    """Invalid selectors or no comparable data."""


def split_effort(name):
    match = re.search(r"\s+\(([^()]*)\)$", name)
    if match:
        suffix = match[1].strip().casefold()
        verbose = re.fullmatch(
            r"(adaptive reasoning|non-reasoning),\s*(minimal|low|medium|high|xhigh|max|ultra) effort"
            r"(?:,\s*(.+ fallback))?",
            suffix,
        )
        if verbose:
            effort = verbose[2]
            if verbose[1] == "non-reasoning":
                effort = "non-reasoning, " + effort
            if verbose[3]:
                fallback = "fallback" if verbose[3] == "default fallback" else verbose[3]
                effort += " with " + fallback
            return name[: match.start()], effort
        short = re.fullmatch(r"(minimal|low|medium|high|xhigh|max|ultra),\s*default fallback", suffix)
        if short:
            return name[: match.start()], short[1] + " with fallback"
        if (
            re.fullmatch(
                r"(?:non-reasoning(?:,\s*(?:low|medium|high))?|none|minimal|low|"
                r"medium|high|xhigh|max|ultra|thinking|reasoning)(?: with fallback)?",
                suffix,
            )
            or suffix == "with fallback"
            or re.fullmatch(r"\d+(?:\.\d+)?k?(?: tokens)?", suffix)
        ):
            return name[: match.start()], suffix
    return name, None


def effort_order(effort):
    if effort is None:
        return (-1, "")
    base = effort.partition(" with ")[0]
    if base.startswith("non-reasoning"):
        qualifier = base.partition(",")[2].strip()
        return ({"low": 0.1, "medium": 0.2, "high": 0.3}.get(qualifier, 0), base)
    if base in EFFORTS:
        return (EFFORTS[base], base)
    if re.fullmatch(r"\d+(?:\.\d+)?k?(?: tokens)?", base):
        number = base.removesuffix(" tokens")
        return (float(number.rstrip("k")) * (1000 if number.endswith("k") else 1), base)
    return (99, base)


def _dominates(other, row, x="cost_per_task_usd"):
    """True when `other` is no worse on x and no less capable, and strictly better once."""
    return (
        other[x] <= row[x]
        and other["score"] >= row["score"]
        and (other[x] < row[x] or other["score"] > row["score"])
    )


def pareto_frontier(rows, x="cost_per_task_usd"):
    """Undominated rows, ordered by x. Exact ties stay on the frontier together.

    Rows must carry a numeric score and a numeric value under `x`, where lower is
    better. Membership depends on the set given: the frontier of the comparable
    rows is not the frontier of the plotted ones.
    """
    return sorted(
        (row for row in rows if not any(_dominates(other, row, x) for other in rows)),
        key=lambda row: (row[x], row["score"], row["id"]),
    )


def is_non_reasoning(row):
    """True when AA marks the row's reasoning as turned off.

    Only the explicit marker counts. A row with no effort qualifier at all is a
    single-configuration model, not a non-reasoning variant, so it is kept.
    """
    effort = (row["effort"] or "").casefold()
    return effort.startswith("non-reasoning") or effort == "none"


def _number(value, *, nonnegative=False):
    if type(value) in (int, float) and math.isfinite(value):
        if not nonnegative or value >= 0:
            return value
    return None


def _object(value):
    return value if isinstance(value, dict) else {}


def normalize(raw):
    result = []
    for row in raw:
        try:
            creator = row["model_creator"]
            for value in (row["id"], row["name"], row["slug"], creator["id"], creator["name"]):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError
            family, effort = split_effort(row["name"])
            evaluations = _object(row.get("evaluations"))
            performance = _object(row.get("performance"))
            cost = _object(
                _object(row.get("artificial_analysis_intelligence_index_cost")).get("cost_per_task")
            ).get("total_cost")
            result.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "slug": row["slug"],
                    "family": family,
                    "effort": effort,
                    "provider": creator["name"],
                    "provider_id": creator["id"],
                    "provider_slug": creator.get("slug")
                    or re.sub(r"[^a-z0-9]+", "-", creator["name"].casefold()).strip("-"),
                    "indices": {
                        key: _number(evaluations.get(f"artificial_analysis_{key}_index"))
                        for key in METRICS
                    },
                    "cost_per_task_usd": _number(cost, nonnegative=True),
                    **{
                        axis["row_key"]: _number(
                            performance.get(axis["field"].partition(".")[2]), nonnegative=True
                        )
                        for name, axis in X_AXES.items()
                        if name != "cost"
                    },
                    "url": "https://artificialanalysis.ai/models/" + row["slug"],
                }
            )
        except (KeyError, TypeError, ValueError):
            raise APIError("Invalid model identity in AA response.") from None
    return result


def select(rows, *, providers=None, models=None, variants=None):
    modes = [(providers, "provider"), (models, "model"), (variants, "variant")]
    active = [(values, mode) for values, mode in modes if values]
    if len(active) > 1:
        raise SelectionError("Choose one selection mode: --provider, --model, or --variant.")
    if not active:
        return rows
    values, mode = active[0]
    selected = set()
    for value in values:
        query = value.strip().casefold()
        fields = {
            "provider": ("provider", "provider_id", "provider_slug"),
            "model": ("family",),
            "variant": ("name", "slug", "id"),
        }[mode]
        matches = [row for row in rows if any(row[f].casefold() == query for f in fields)]
        if not matches:
            raise SelectionError(f"Unknown {mode}: {value!r}. Run models to discover exact names.")
        identities = {
            (r["provider_id"], r["family"].casefold())
            if mode == "model"
            else r["provider_id"]
            if mode == "provider"
            else r["id"]
            for r in matches
        }
        if len(identities) != 1:
            raise SelectionError(f"{mode} {value!r} is ambiguous; use exact variant IDs instead.")
        selected.update(row["id"] for row in matches)
    return [row for row in rows if row["id"] in selected]


def _point_label(name, names):
    """The variant name without the prefix every name in its series shares.

    The prefix is cut back to its last space, hyphen or underscore so a label
    never starts mid-word: 'GLM-5.2 (max)' and 'GLM-5.3 (max)' become
    '5.2 (max)' and '5.3 (max)', not '2 (max)' and '3 (max)'.
    """
    prefix = names[0]
    for other in names[1:]:
        while not other.startswith(prefix):
            prefix = prefix[:-1]
    cut = max(prefix.rfind(c) for c in " -_") + 1
    return name[cut:].strip() or name


def join_series(rows, catalog, series):
    """Draw several variants, even of different releases, as one connected series.

    Each entry of `series` is [LABEL, VARIANT, VARIANT, ...]. Variants resolve
    like --variant against the whole catalog and are added to the selection.
    Each joined row takes LABEL as its family, a point label as its effort, and
    its position in the list as series_rank, which orders and connects it.
    """
    if not series:
        return rows
    rows = list(rows)
    claimed = {}
    for label, *variants in series:
        if not label.strip():
            raise SelectionError("--series needs a non-empty label before its variants.")
        if len(variants) < 2:
            raise SelectionError(f"--series {label!r} needs at least two variants.")
        members = [select(catalog, variants=[variant])[0] for variant in variants]
        if len({row["provider_id"] for row in members}) != 1:
            raise SelectionError(f"--series {label!r} mixes model creators; join one creator's.")
        names = [row["name"] for row in members]
        for rank, member in enumerate(members):
            if member["id"] in claimed:
                raise SelectionError(
                    f"{member['name']!r} is in both --series {claimed[member['id']]!r} "
                    f"and {label!r}."
                )
            claimed[member["id"]] = label
            joined = {
                **member,
                "family": label,
                "effort": _point_label(member["name"], names),
                "series_rank": rank,
            }
            rows = [row for row in rows if row["id"] != member["id"]] + [joined]
    return rows


def compare(rows, metric="intelligence", x_axis="cost"):
    metric = "agentic" if metric == "agent" else metric
    if metric not in METRICS:
        raise SelectionError(f"Unknown metric {metric!r}; choose intelligence, coding, or agentic.")
    if x_axis not in X_AXES:
        raise SelectionError(f"Unknown x-axis {x_axis!r}; choose {', '.join(X_AXES)}.")
    axis = X_AXES[x_axis]
    result = []
    for row in rows:
        score, x = row["indices"][metric], row[axis["row_key"]]
        reasons = ([] if score is not None else ["missing score"]) + (
            [] if x is not None else [f"missing {axis['noun']}"]
        )
        result.append(
            {
                **row,
                "score": score,
                "x": x,
                "comparable": not reasons,
                "exclusion_reasons": reasons,
                "pareto_optimal": None,
            }
        )
    comparable = [r for r in result if r["comparable"]]
    optimal = {row["id"] for row in pareto_frontier(comparable, "x")}
    for row in comparable:
        row["pareto_optimal"] = row["id"] in optimal
    return {
        "schema_version": 1,
        "metric": {"key": metric, "label": METRICS[metric], "higher_is_better": True},
        "x_axis": {"key": x_axis, **axis, "lower_is_better": True},
        "cost": {
            "field": "artificial_analysis_intelligence_index_cost.cost_per_task.total_cost",
            "unit": "USD",
            "workload": "Artificial Analysis Intelligence Index task",
            "lower_is_better": True,
        },
        "rows": result,
        "selected_count": len(result),
        "comparable_count": len(comparable),
        "excluded_count": len(result) - len(comparable),
    }
