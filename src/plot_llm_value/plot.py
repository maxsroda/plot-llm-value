"""Headless Matplotlib rendering with label placement in display coordinates."""

import math
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
from matplotlib.transforms import Bbox

from .data import SelectionError, effort_order, pareto_frontier


def _rank(row):
    """Position along a series: the --series order when joined, else the effort order."""
    if row.get("series_rank") is not None:
        return row["series_rank"]
    return effort_order(row["effort"])[0]


BACKGROUND = "#FBFAF7"
COLORS = ["#6B4F9B", "#D97745", "#2C7A7B", "#4C78A8", "#59A14F", "#B279A2", "#8C6D31", "#B64857"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]
PARETO_COLOR = "#4A4A4A"
PARETO_DASHES = (7, 3.5)
# Past this many scored-but-unpriced rows, full-width capability lines stop
# aiding the eye and start hiding the priced points; the rows stay in the JSON.
CAPABILITY_LINE_LIMIT = 12


def _place_labels(ax, requests, point_positions, segments):
    """Greedy nearest free candidate, measured after log/linear transforms.

    Text is placed only where its padded rectangle clears points, previous text
    and series segments. Search all group points for model labels; effort labels
    have one anchor. Dense inputs grow the canvas rather than hiding labels.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied, leaders = [], []
    bounds = ax.bbox.padded(-5)
    points = np.asarray(point_positions)
    line_paths = [MplPath(np.asarray(s)) for s in segments]
    for label, anchors, color, model_label in requests:
        text = ax.text(
            0,
            0,
            label,
            fontsize=9 if model_label else 7.5,
            weight="bold" if model_label else "normal",
            color=color,
            ha="center",
            va="center",
            zorder=5,
        )
        extent = text.get_window_extent(renderer)
        width, height = extent.width + 6, extent.height + 5
        anchor_pixels = ax.transData.transform(anchors)
        # A fine display-space lattice gives a deterministic fallback even when
        # many labels share exactly the same anchor.
        xs = np.arange(bounds.x0 + width / 2, bounds.x1 - width / 2, 10)
        ys = np.arange(bounds.y0 + height / 2, bounds.y1 - height / 2, 10)
        xx, yy = np.meshgrid(xs, ys)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
        if not len(candidates):
            return False
        distances = ((candidates[:, None, :] - anchor_pixels[None, :, :]) ** 2).sum(axis=2)
        closest = distances.argmin(axis=1)
        order = np.argsort(distances.min(axis=1), kind="stable")
        found = None
        for index in order:
            x, y = candidates[index]
            box = Bbox.from_bounds(x - width / 2, y - height / 2, width, height)
            if any(box.overlaps(other) for other in occupied):
                continue
            if len(points) and np.any(
                (points[:, 0] >= box.x0 - 9)
                & (points[:, 0] <= box.x1 + 9)
                & (points[:, 1] >= box.y0 - 9)
                & (points[:, 1] <= box.y1 + 9)
            ):
                continue
            if any(path.intersects_bbox(box.padded(2)) for path in line_paths):
                continue
            found = (x, y, box, anchor_pixels[closest[index]])
            break
        if found is None:
            return False
        x, y, box, anchor = found
        text.set_position(ax.transData.inverted().transform((x, y)))
        occupied.append(box)
        # Leaders terminate at the label edge; they do not cover the text.
        target = np.array([np.clip(anchor[0], box.x0, box.x1), np.clip(anchor[1], box.y0, box.y1)])
        if np.linalg.norm(target - anchor) > 15:
            leaders.append((target, anchor, color))
    # Clip leaders around every label rectangle. A dense cloud can completely
    # surround an anchor with text; forcing a clear straight ray would reject
    # otherwise readable layouts. Small gaps keep leaders from obscuring text.
    for target, anchor, color in leaders:
        delta = anchor - target
        hidden = []
        for box in occupied:
            box = box.padded(2)
            enter, leave = 0.0, 1.0
            for dim, (low, high) in enumerate(((box.x0, box.x1), (box.y0, box.y1))):
                if abs(delta[dim]) < 1e-10:
                    if not low <= target[dim] <= high:
                        enter, leave = 1.0, 0.0
                        break
                else:
                    t1, t2 = (low - target[dim]) / delta[dim], (high - target[dim]) / delta[dim]
                    enter, leave = max(enter, min(t1, t2)), min(leave, max(t1, t2))
            if enter < leave:
                hidden.append((enter, leave))
        cursor = 0.0
        for start, end in sorted(hidden) + [(1.0, 1.0)]:
            if start > cursor and np.linalg.norm(delta * (start - cursor)) > 4:
                ax.add_patch(
                    FancyArrowPatch(
                        ax.transData.inverted().transform(target + cursor * delta),
                        ax.transData.inverted().transform(target + start * delta),
                        arrowstyle="-",
                        color=color,
                        lw=0.6,
                        alpha=0.5,
                        zorder=1,
                        shrinkA=0,
                        shrinkB=0,
                    )
                )
            cursor = max(cursor, end)
    return True


def _inferred_cost(row, gaps, *, linear_x):
    """Cost implied for an unpriced effort by the dashed segment spanning its rank.

    Only the cost is guessed: the score is AA's own measurement. The effort is
    placed at its rank's share of the gap, in the axis's own spacing. Returns
    None when no gap of measured neighbors spans the effort's rank.
    """
    rank = _rank(row)
    for start, end in gaps:
        low, high = _rank(start), _rank(end)
        if low < rank < high:
            fraction = (rank - low) / (high - low)
            a, b = start["x"], end["x"]
            if linear_x:
                return a + fraction * (b - a), (start, end)
            return math.exp(math.log(a) + fraction * math.log(b / a)), (start, end)
    return None


def build_figure(
    comparison, *, linear_x=False, pareto=True, reasoning_only=False, capability_lines=True
):
    axis = comparison["x_axis"]
    noun = axis["noun"]
    unavailable = f"{noun} unavailable"
    # inferred_cost_usd predates the other axes; keep its name for existing callers.
    inferred_key = "inferred_cost_usd" if axis["key"] == "cost" else "inferred_" + axis["row_key"]
    groups, capability_only, excluded = defaultdict(list), defaultdict(list), []
    for row in comparison["rows"]:
        key = (row["provider_id"], row["family"])
        if row["score"] is None:
            excluded.append(
                {"id": row["id"], "name": row["name"], "reasons": list(row["exclusion_reasons"])}
            )
        elif row["x"] is None:
            capability_only[key].append((row, unavailable))
        elif row["x"] == 0 and not linear_x:
            capability_only[key].append((row, f"zero {noun} on log x"))
        else:
            groups[key].append(row)
    if not groups:
        raise SelectionError(
            f"No measured {noun}s to plot; a {noun} axis needs at least one measured {noun}."
        )
    count = sum(map(len, groups.values()))
    flat = [entry for entries in capability_only.values() for entry in entries]
    draw_capability_lines = capability_lines and len(flat) <= CAPABILITY_LINE_LIMIT
    label_count = count + len(groups) + (2 * len(flat) if draw_capability_lines else 0)
    scale = max(1, math.sqrt(label_count / 45))
    providers = sorted({r["provider"] for rows in groups.values() for r in rows})
    marker_map = {p: MARKERS[i % len(MARKERS)] for i, p in enumerate(providers)}
    keys = sorted(set(groups) | set(capability_only))
    # The frontier follows the measured points only. A row without a cost states
    # nothing about cost, so it can neither dominate nor be dominated.
    frontier = pareto_frontier([r for rows in groups.values() for r in rows], "x")
    # Always reported in the result; --no-pareto only suppresses the drawn line.
    frontier_xy = [(r["x"], r["score"]) for r in frontier] if pareto else []
    for expansion in (1, 1.3, 1.7):
        width, height = 10 * scale * expansion, 7.2 * scale * expansion
        fig, ax = plt.subplots(figsize=(width, height), dpi=180)
        fig.patch.set_facecolor(BACKGROUND)
        ax.set_facecolor(BACKGROUND)
        legend_rows = math.ceil(len(keys) / 4) + (
            math.ceil(len(providers) / 5) if len(providers) > 1 else 0
        )
        bottom = max(0.22, (0.85 + legend_rows * 0.19) / height)
        fig.subplots_adjust(left=0.095, right=0.96, top=0.85, bottom=min(bottom, 0.48))
        ax.set_xscale("linear" if linear_x else "log")
        requests, handles, coordinates, data_segments = [], [], [], []
        efforts, flat_labels, flat_rows, inferred_rows = [], [], [], []
        for i, (provider_id, family) in enumerate(keys):
            rows = sorted(
                groups.get((provider_id, family), []),
                key=lambda r: (_rank(r), effort_order(r["effort"]), r["id"]),
            )
            color = (
                COLORS[i % len(COLORS)]
                if len(keys) <= len(COLORS)
                else plt.cm.turbo((i + 0.5) / len(keys))
            )
            gaps = [
                (start, end) for start, end in zip(rows, rows[1:]) if _rank(end) - _rank(start) > 1
            ]
            unpriced = sorted(
                capability_only.get((provider_id, family), []),
                key=lambda entry: (
                    _rank(entry[0]),
                    effort_order(entry[0]["effort"]),
                    entry[0]["id"],
                ),
            )
            inferred = []
            for row, reason in unpriced:
                # An unpriced effort that falls inside its own family's gap is a
                # guess the dashed segment already makes; a measured zero cost is
                # not a guess, so it keeps its capability line.
                guess = (
                    _inferred_cost(row, gaps, linear_x=linear_x) if reason == unavailable else None
                )
                if guess is None:
                    if draw_capability_lines:
                        line = ax.axhline(row["score"], color=color, lw=0.7, alpha=0.55, zorder=1.2)
                        line.set_gid(f"capability:{row['id']}")
                        # An unavailable cost is what the line itself means; a measured
                        # zero cost is a different fact and still worth spelling out.
                        label = (
                            row["name"] if reason == unavailable else f"{row['name']} · {reason}"
                        )
                        flat_labels.append((label, row["score"], color))
                    flat_rows.append(
                        {
                            "id": row["id"],
                            "name": row["name"],
                            "score": row["score"],
                            "reason": reason,
                        }
                    )
                else:
                    cost, (start, end) = guess
                    inferred.append((row, cost))
                    inferred_rows.append(
                        {
                            "id": row["id"],
                            "name": row["name"],
                            "score": row["score"],
                            inferred_key: cost,
                            "interpolated_between": [start["name"], end["name"]],
                        }
                    )
            if not rows:
                # A family with only capability lines has no legend entry when
                # those lines are not drawn: nothing on the canvas would match it.
                if draw_capability_lines:
                    handles.append(Line2D([], [], color=color, lw=0.7, label=family))
                continue
            marker = marker_map[rows[0]["provider"]]
            xy = [(r["x"], r["score"]) for r in rows]
            coordinates.extend(xy)
            for start, end in zip(rows, rows[1:]):
                a, b = (
                    (start["x"], start["score"]),
                    (end["x"], end["score"]),
                )
                gap = _rank(end) - _rank(start) > 1
                (line,) = ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    color=color,
                    lw=2,
                    alpha=0.75,
                    linestyle="--" if gap else "-",
                    zorder=2,
                )
                line.set_gid(f"series:{provider_id}:{family}")
                data_segments.append((a, b))
            points = ax.scatter(
                *zip(*xy),
                s=65,
                marker=marker,
                color=color,
                edgecolor=BACKGROUND,
                linewidth=1.2,
                zorder=3,
            )
            points.set_gid(f"points:{provider_id}:{family}")
            requests.append((family, xy, color, True))
            for row, point in zip(rows, xy):
                if row["effort"]:
                    efforts.append((row["effort"], [point], color, False))
            if inferred:
                guessed_xy = [(cost, row["score"]) for row, cost in inferred]
                hollow = ax.scatter(
                    *zip(*guessed_xy),
                    s=60,
                    marker=marker,
                    facecolor=BACKGROUND,
                    edgecolor=color,
                    linewidth=1.6,
                    zorder=3,
                )
                hollow.set_gid(f"inferred:{provider_id}:{family}")
                coordinates.extend(guessed_xy)
                for (row, _), point in zip(inferred, guessed_xy):
                    efforts.append((f"{row['effort']}?", [point], color, False))
            handles.append(Line2D([], [], color=color, marker=marker, lw=2, label=family))
        if len(frontier_xy) > 1:
            (line,) = ax.plot(
                *zip(*frontier_xy),
                color=PARETO_COLOR,
                lw=1.6,
                alpha=0.6,
                zorder=1.5,
                # A longer dash than the effort-gap segments, which are colored.
                dashes=PARETO_DASHES,
            )
            line.set_gid("pareto")
            data_segments.extend(zip(frontier_xy, frontier_xy[1:]))
        ax.margins(x=0.18, y=0.2)
        if linear_x:
            low, high = ax.get_xlim()
            ax.set_xlim(max(0, low), high)
        # Anchors span the settled x-range: a capability-only label may sit
        # anywhere along its line, wherever the layout has room.
        low, high = ax.get_xlim()
        spread = (
            np.linspace(low, high, 9)
            if linear_x
            else np.logspace(math.log10(low), math.log10(high), 9)
        )
        for label, score, color in flat_labels:
            requests.append((label, [(x, score) for x in spread], color, False))
        ax.grid(True, which="major", color="#DEDCD6", lw=0.8)
        ax.grid(True, which="minor", axis="x", color="#ECEAE4", lw=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if not linear_x:
            low, high = ax.get_xlim()
            decades = math.log10(high / low)
            if decades < 0.3:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            else:
                ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 3, 5) if decades < 2 else (1,)))
        tick = "${:g}" if axis["unit"] == "USD" else "{:g} " + axis["unit"]
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: tick.format(value)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=8)
        metric = comparison["metric"]["label"]
        ax.set_ylabel("Artificial Analysis " + metric, fontsize=9.5, labelpad=9)
        ax.set_xlabel(
            f"{axis['label']} ({axis['unit']}" + (", log scale)" if not linear_x else ")"),
            fontsize=9.5,
            labelpad=9,
        )
        fig.text(
            0.095, 0.953, metric + " vs. " + axis["title"], weight="bold", fontsize=16, ha="left"
        )
        fig.text(
            0.095,
            0.914,
            "Reasoning settings connected within each release · Higher and further left is better",
            color="#555555",
            fontsize=8.5,
        )
        if len(providers) > 1:
            handles.extend(
                Line2D([], [], color="#555555", marker=marker_map[p], lw=0, label=p)
                for p in providers
            )
        if flat_labels:
            handles.append(
                Line2D(
                    [], [], color="#555555", lw=0.7, label=f"Capability only (no measured {noun})"
                )
            )
        if inferred_rows:
            handles.append(
                Line2D(
                    [],
                    [],
                    color="#555555",
                    markerfacecolor=BACKGROUND,
                    markeredgecolor="#555555",
                    marker="o",
                    lw=0,
                    label=f"Hollow: {noun} inferred from gap",
                )
            )
        if len(frontier_xy) > 1:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=PARETO_COLOR,
                    alpha=0.6,
                    lw=1.6,
                    # Shorter dashes than the drawn line, which the short
                    # legend swatch would otherwise render as one solid stroke.
                    dashes=(4, 2),
                    label="Pareto frontier",
                )
            )
        source = comparison.get("source", {})
        notes = [
            "Source: Artificial Analysis · https://artificialanalysis.ai/",
            f"Retrieved {source.get('retrieved_at', 'unknown')} · "
            f"Intelligence Index v{source.get('intelligence_index_version', '?')}",
            f"{count} / {len(comparison['rows'])} rows plotted · Dashed color: effort gap · "
            + (
                "Costs use Intelligence Index tasks"
                if axis["key"] == "cost"
                else "Times are AA speed-benchmark medians, not Intelligence Index tasks"
            ),
        ]
        if reasoning_only:
            notes.append(
                "Non-reasoning variants excluded (--reasoning-only), so the frontier "
                "describes reasoning settings only"
            )
        if flat_rows and draw_capability_lines:
            notes.append(
                f"{len(flat_rows)} scored row(s) have no plottable {noun}: thin capability line "
                "at the score, kept off the frontier"
            )
        elif flat_rows:
            why = (
                "--no-capability-lines"
                if not capability_lines
                else f"more than {CAPABILITY_LINE_LIMIT} would clutter the plot"
            )
            notes.append(
                f"{len(flat_rows)} scored row(s) have no plottable {noun}: not drawn ({why}); "
                "listed in capability_only"
            )
        if inferred_rows:
            notes.append(
                f"{len(inferred_rows)} unmeasured effort(s) sit hollow on their dashed gap: score "
                f"measured, {noun} interpolated by effort rank, kept off the frontier"
            )
        if len(frontier_xy) > 1:
            notes.append(
                f"Dashed grey Pareto frontier: no plotted model is both {axis['better']} and at "
                "least as capable"
            )
        fig.text(0.095, 0.025, "\n".join(notes), fontsize=6.5, color="#666666", linespacing=1.4)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        footer_top = fig.texts[-1].get_window_extent(renderer).y1
        max_label_width = max(
            renderer.get_text_width_height_descent(
                handle.get_label(), FontProperties(size=7.2), False
            )[0]
            for handle in handles
        )
        columns = max(1, min(4, int(fig.bbox.width * 0.865 / (max_label_width + 90))))
        legend = ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.095, (footer_top + 12) / fig.bbox.height),
            bbox_transform=fig.transFigure,
            ncol=columns,
            frameon=False,
            fontsize=7.2,
            columnspacing=1.1,
            borderaxespad=0,
        )
        fig.canvas.draw()
        legend_top = legend.get_window_extent(renderer).y1
        bottom = (legend_top + 110) / fig.bbox.height
        if bottom > 0.65:
            plt.close(fig)
            continue
        fig.subplots_adjust(bottom=bottom)
        fig.canvas.draw()
        pixels = ax.transData.transform(coordinates)
        segments = [ax.transData.transform(s) for s in data_segments]
        if _place_labels(ax, requests + efforts, pixels, segments):
            return fig, {
                "plotted_count": count,
                "excluded_count": len(excluded),
                "excluded": excluded,
                "capability_only": flat_rows,
                "capability_lines_drawn": draw_capability_lines,
                "capability_line_limit": CAPABILITY_LINE_LIMIT,
                "inferred_cost": inferred_rows,
                "x_scale": "linear" if linear_x else "log",
                "pareto_frontier": [
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "cost_per_task_usd": row["cost_per_task_usd"],
                        "x": row["x"],
                        "score": row["score"],
                    }
                    for row in frontier
                ],
            }
        plt.close(fig)
    raise SelectionError(
        "Labels do not fit clearly; select fewer models for a readable comparison."
    )


def save_plot(
    comparison,
    path=None,
    *,
    linear_x=False,
    pareto=True,
    reasoning_only=False,
    capability_lines=True,
):
    if path is not None:
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() not in {".png", ".svg", ".pdf"}:
            raise SelectionError("--output must end in .png, .svg, or .pdf.")
        if not path.parent.is_dir():
            raise OSError("Output parent directory does not exist.")
    fig, info = build_figure(
        comparison,
        linear_x=linear_x,
        pareto=pareto,
        reasoning_only=reasoning_only,
        capability_lines=capability_lines,
    )
    try:
        if path is None:
            path = Path(tempfile.mkdtemp(prefix="plot-llm-value-")) / "comparison.png"
        fig.savefig(path, facecolor=BACKGROUND)
    finally:
        plt.close(fig)
    return {
        "image_path": str(path),
        **info,
        "source": comparison.get("source", {}),
        "display_instruction": "Display image_path inline; respond with only the plot unless analysis was requested.",
    }
