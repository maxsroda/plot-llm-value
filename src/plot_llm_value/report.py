"""One spec, one fetch: the plots and the which-model table that reads them."""

import json
from pathlib import Path

from .data import X_AXES, SelectionError, compare, is_non_reasoning, join_series, select

FAST_SECONDS, SLOW_SECONDS = 10, 100
SPEC_KEYS = {
    "name",
    "models",
    "series",
    "reasoning_only",
    "plots",
    "dots",
    "rows",
    "skip",
    "caveats",
}


def load_spec(path):
    try:
        spec = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise SelectionError(f"Cannot read spec {path}: {error}") from None
    if not isinstance(spec, dict):
        raise SelectionError("Spec must be a JSON object.")
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise SelectionError(f"Unknown spec keys: {', '.join(sorted(unknown))}.")
    for key in ("name", "models", "rows"):
        if not spec.get(key):
            raise SelectionError(f"Spec needs a non-empty {key!r}.")
    for axis in spec.get("plots", ["cost"]):
        if axis not in X_AXES:
            raise SelectionError(f"Unknown plot axis {axis!r}; choose {', '.join(X_AXES)}.")
    for row in spec["rows"]:
        if not row.get("use") or not row.get("variants"):
            raise SelectionError("Every table row needs 'use' and at least one variant.")
    for item in spec.get("skip", []):
        if not item.get("why") or not item.get("variants"):
            raise SelectionError("Every skip entry needs 'why' and at least one variant.")
    return spec


def _variant(catalog, name):
    return select(catalog, variants=[name])[0]


def _family(row):
    return row["family"].removeprefix("Claude ")


def _effort(row):
    return (row["effort"] or "").partition(" with ")[0]


def _short(row):
    """'Claude Sonnet 5.5' + 'medium with fallback' -> 'Sonnet 5.5 medium'."""
    return f"{_family(row)} {_effort(row)}".strip()


def _int(row):
    return row["indices"]["intelligence"]


def _seconds(value):
    return "n/a" if value is None else f"{value:.0f} s" if value >= 100 else f"{value:.1f} s"


def _number(value, digits):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _usd(value):
    """Cents for normal costs; two significant figures below 0.1 USD, so none reads 0.00."""
    if value is None:
        return "n/a"
    return f"{value:.2f}" if value >= 0.1 or value == 0 else f"{value:.2g}"


def _cell(rows, value, fmt, bold_first):
    parts = [fmt(value(row)) for row in rows]
    if bold_first:
        parts[0] = f"**{parts[0]}**"
    return " · ".join(parts)


def _table_row(spec, catalog, row):
    members = [_variant(catalog, name) for name in row["variants"]]
    key = bool(row.get("key"))
    dots = spec.get("dots", {})
    names = []
    for i, member in enumerate(members):
        name = _short(member)
        name = f"**{name}**" if key and i == 0 else name
        dot = dots.get(member["family"])
        names.append(f"{name} {dot}" if dot else name)
    time = _cell(members, lambda r: r["e2e_seconds"], _seconds, key)
    first = members[0]["e2e_seconds"]
    if first is not None and first < FAST_SECONDS:
        time = "⚡ " + time
    elif first is not None and first >= SLOW_SECONDS:
        time = "🐢 " + time
    cells = [
        ("★ " if key else "") + row["use"],
        " · ".join(names),
        _cell(members, _int, lambda v: _number(v, 1), key),
        _cell(members, lambda r: r["cost_per_task_usd"], _usd, key),
        time,
    ]
    for cell in cells:
        if "$" in cell or "<" in cell or "|" in cell:
            raise SelectionError(f"Table cell {cell!r} contains $, < or |, which renderers break.")
    return _int(members[0]) or 0, "| " + " | ".join(cells) + " |"


def _skip_line(catalog, item):
    members = [_variant(catalog, name) for name in item["variants"]]
    if len({m["family"] for m in members}) == 1:
        label = f"{_family(members[0])} " + " / ".join(_effort(m) for m in members)
    else:
        label = " / ".join(_short(m) for m in members)
    ints = " / ".join(_number(_int(m), 1) for m in members)
    costs = " / ".join(_usd(m["cost_per_task_usd"]) for m in members)
    times = " / ".join(_seconds(m["e2e_seconds"]) for m in members)
    why = item["why"].replace("$", "USD ")
    return f"- {label}: Int {ints}, {costs} USD, {times} — {why}"


def build_table(spec, catalog, source, plot_files):
    ordered = sorted((_table_row(spec, catalog, row) for row in spec["rows"]), key=lambda p: p[0])
    used = [_variant(catalog, n) for r in spec["rows"] for n in r["variants"]]
    lines = [
        "# Which model, which effort",
        "",
        f"Values: Artificial Analysis, retrieved {source['retrieved_at'][:16].replace('T', ' ')}"
        f" UTC (Intelligence Index v{source['intelligence_index_version']}).",
        "Plots: " + " · ".join(f"[{label}]({file})" for label, file in plot_files) + ".",
    ]
    if any("with fallback" in (m["effort"] or "") for m in used):
        lines.append(
            'Every Claude setting below is AA\'s "Adaptive Reasoning, … Effort, Default Fallback"'
            " row."
        )
    if any(r.get("key") for r in spec["rows"]):
        lines += ["", "★ = the key options. Rows are ordered by Int, the AA Intelligence Index."]
    lines += [
        "",
        "| Use case | Model and effort | Int | Cost/task (USD) | End-to-end time |",
        "| --- | --- | --- | --- | --- |",
        *(line for _, line in ordered),
    ]
    if spec.get("skip"):
        lines += ["", "Skip:", "", *(_skip_line(catalog, item) for item in spec["skip"])]
    caveats = [
        "One snapshot; differences of 1–2 index points may be within benchmark noise (AA publishes"
        " no error bars).",
        "Times are AA's speed-benchmark medians, not Intelligence Index tasks; cost is USD per"
        " Intelligence Index task at API prices, not subscription usage.",
        *spec.get("caveats", []),
    ]
    lines += ["", "Caveats:", "", *(f"- {c.replace('$', 'USD ')}" for c in caveats)]
    return "\n".join(lines) + "\n"


def run_report(spec, catalog, source, out_dir):
    from .plot import save_plot

    out_dir = Path(out_dir).expanduser().resolve()
    if not out_dir.is_dir():
        raise OSError(f"--out-dir {out_dir} does not exist.")
    rows = join_series(select(catalog, models=spec["models"]), catalog, spec.get("series"))
    if spec.get("reasoning_only", True):
        rows = [row for row in rows if not is_non_reasoning(row)]
    written, plot_files = [], []
    for axis in spec.get("plots", ["cost"]):
        file = f"{spec['name']}.png" if axis == "cost" else f"{spec['name']}-{axis}.png"
        comparison = compare(rows, "intelligence", axis)
        comparison["source"] = source
        save_plot(comparison, out_dir / file, reasoning_only=spec.get("reasoning_only", True))
        plot_files.append((X_AXES[axis]["title"], file))
        written.append(str(out_dir / file))
    table = out_dir / f"{spec['name']}.md"
    table.write_text(build_table(spec, catalog, source, plot_files))
    written.append(str(table))
    return {"schema_version": 1, "source": source, "written": written}
