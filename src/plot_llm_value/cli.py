"""Agent-facing command boundary. Stdout is always a complete result."""

import argparse
import json
import math
import os
import sys
from importlib.resources import files
from pathlib import Path

from dotenv import dotenv_values

from .api import APIError, fetch_models
from .data import X_AXES, SelectionError, compare, is_non_reasoning, join_series, normalize, select


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise SelectionError(message)


def _arguments(argv):
    command = argv[0] if argv and argv[0] in {"plot", "data", "models"} else "plot"
    if argv and argv[0] == command:
        argv = argv[1:]
    parser = Parser(add_help=False)
    selectors = parser.add_mutually_exclusive_group()
    for option in ("provider", "model", "variant"):
        selectors.add_argument("--" + option, nargs="+", action="extend")
    parser.add_argument(
        "--metric", choices=["intelligence", "coding", "agentic", "agent"], default="intelligence"
    )
    parser.add_argument("--series", nargs="+", action="append")
    parser.add_argument("--x", choices=list(X_AXES), default="cost")
    parser.add_argument("--reasoning-only", action="store_true")
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument("--no-pareto", action="store_true")
    parser.add_argument("--no-capability-lines", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args(argv)
    plot_only = (args.linear_x, args.no_pareto, args.no_capability_lines, args.output)
    if command == "models" and args.x != "cost":
        raise SelectionError("--x applies to plot and data, not models.")
    if command != "plot" and any(plot_only):
        raise SelectionError(
            "--linear-x, --no-pareto, --no-capability-lines and --output are plot-only options."
        )
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SelectionError("--timeout must be positive and finite.")
    if args.output and args.output.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise SelectionError("--output must end in .png, .svg, or .pdf.")
    return command, args


def _key(env_file):
    values = {}
    if env_file is not None:
        if not env_file.is_file():
            raise APIError("--env-file does not exist or is not a file.")
        values = dotenv_values(env_file, interpolate=False)
    return (
        os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
        or os.environ.get("AA_API_KEY")
        or values.get("ARTIFICIAL_ANALYSIS_API_KEY")
        or values.get("AA_API_KEY")
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv or argv == ["help"]:
        print(files("plot_llm_value").joinpath("help.txt").read_text(), end="")
        return 0
    try:
        command, args = _arguments(argv)
        snapshot = fetch_models(_key(args.env_file), timeout=args.timeout)
        catalog = normalize(snapshot["data"])
        rows = select(catalog, providers=args.provider, models=args.model, variants=args.variant)
        if args.series and not (args.provider or args.model or args.variant):
            rows = []  # --series alone selects only its own variants, not the whole catalog.
        rows = join_series(rows, catalog, args.series)
        if args.reasoning_only:
            rows = [row for row in rows if not is_non_reasoning(row)]
            if not rows:
                raise SelectionError(
                    "--reasoning-only removed every selected row; drop it or widen the selection."
                )
        if command == "models":
            result = {"schema_version": 1, "source": snapshot["source"], "rows": rows}
        else:
            result = compare(rows, args.metric, args.x)
            result["source"] = snapshot["source"]
            if command == "plot":
                from .plot import save_plot

                result = save_plot(
                    result,
                    args.output,
                    linear_x=args.linear_x,
                    pareto=not args.no_pareto,
                    reasoning_only=args.reasoning_only,
                    capability_lines=not args.no_capability_lines,
                )
        result["filters"] = {"reasoning_only": args.reasoning_only}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except SelectionError as error:
        print(f"plot-llm-value: {error}", file=sys.stderr)
        return 2
    except (APIError, OSError) as error:
        print(f"plot-llm-value: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("plot-llm-value: interrupted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
