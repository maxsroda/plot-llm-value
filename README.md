# plot-llm-value

Compare language models using fresh official Artificial Analysis data: capability
on the y-axis and **measured USD per Intelligence Index task** on the x-axis.
Python + Matplotlib; designed for agents that execute a command and read its output.

```sh
uv sync
uv run plot-llm-value --help
```

Or install the command globally in an isolated environment:

```sh
uv tool install .
plot-llm-value --help
```

Set `ARTIFICIAL_ANALYSIS_API_KEY` (or `AA_API_KEY`) using a key from
[Artificial Analysis](https://artificialanalysis.ai/data-api). Alternatively,
store it in an ignored `.env` and pass `--env-file .env` explicitly.

```sh
# Default: a fresh plot, logarithmic cost axis, Intelligence Index.
uv run plot-llm-value --provider anthropic openai

# Discover exact family names, variants, slugs and stable IDs.
uv run plot-llm-value models --provider openai

# All available efforts of a release; substitute a family from models output.
uv run plot-llm-value --model 'gpt-oss-20B' --metric coding

# Exact settings only, without generating a plot.
uv run plot-llm-value data --variant 'gpt-oss-20B (high)'

# Agentic Index, linear cost axis, persistent export.
uv run plot-llm-value --provider anthropic --metric agentic --linear-x --output compare.svg

# Reasoning settings only, dropping the variants AA marks Non-reasoning.
uv run plot-llm-value --provider anthropic openai --reasoning-only

# Median end-to-end response time instead of cost on the x-axis.
uv run plot-llm-value --provider anthropic --x e2e

# Plots plus a which-model Markdown table from one spec file (see --help, report).
uv run plot-llm-value report spec.json --out-dir ~/Pictures

# Two releases drawn as one connected series, labelled '5.2 (max)' and '5.3 (max)'.
uv run plot-llm-value --model 'Claude Opus 5.5' --series z.ai 'GLM-5.2 (max)' 'GLM-5.3 (max)'
```

`--reasoning-only` drops variants AA marks `Non-reasoning` (or `none`) from every
command, and every result echoes the filters it applied. It is off by default
because the frontier is always relative to the rows kept, and hiding a measured
row should be a visible choice. A row with **no** effort qualifier is a
single-configuration model rather than a non-reasoning variant, so it is kept —
several such models sit on the frontier. In AA's current data the flag mostly
removes clutter: of 97 non-reasoning variants in the catalog, 2 have a measured
cost per task and none are on the full-catalog frontier, so the rest can only ever
appear here as capability lines.

`--help` is the complete operating guide: selectors, API-key precedence, error
codes, output schemas and the image-display workflow. No separate agent skill is
needed. Providers mean model creators (OpenAI, Anthropic, etc.), not inference hosts.

Plots include model names beside their groups, effort labels, model colors,
provider marker shapes, and legends. Labels are placed in rendered coordinates
against points, series and other labels; leader lines are clipped around text.
The canvas grows for larger selections. Dashed connections mark gaps in known
effort ranks.

A dashed grey **Pareto frontier** is drawn by default, as on the Artificial Analysis
charts: it connects the plotted points that no other plotted point beats on both
axes at once, so its height at any cost is the best score available at or below
that cost, and every point below it is dominated. It is a hull of real data
points, not a fit — a segment's slope is an artifact of which two models happen
to be adjacent, not an exchange rate between index points and dollars. Membership
depends on the selection shown, and the drawn line uses only the plotted rows, so
it can differ from `data`'s `pareto_optimal` when a row is excluded from the plot.
`--no-pareto` omits the line; the returned JSON still lists `pareto_frontier`.

The default output is a JSON result containing the absolute path to a temporary
PNG. This is more useful to shell-based agents than binary stdout: they can open
that path with their image tool and display **only the plot** in chat, as the help
instructs. Display depends on the host agent; the CLI cannot control chat rendering.
Temporary files persist until OS cleanup. Use `--output` for PNG, SVG or PDF you
want to keep.

A model AA scores but does not price — cost unavailable, or zero cost on a
logarithmic axis — is drawn as a thin horizontal line at its capability, labelled
with the reason and listed in `capability_only`. Only a missing score drops a row
from the plot. A selection where nothing has a measured cost fails, because a cost
axis with no costs on it would be fiction.

Full-width lines scale badly: a whole-provider selection can carry sixty unpriced
rows, and that many lines bury the priced points. Past twelve, none are drawn, the
footer says so, and `capability_lines_drawn` in the result is `false` while
`capability_only` still lists every row. `--no-capability-lines` omits them at any
count.

An unpriced reasoning setting whose score AA measures is drawn as an estimate: a
normal marker, joined to its family's line, with an **error bar** giving the range
the cost could take. The error bar is the only mark of an estimate; the label carries
no `?`. The score stays AA's measurement; only the cost (or time, with `--x`) is
guessed. Two cases:

- **Inside a dashed gap of its own family**: the cost is interpolated by effort rank,
  in the axis's own spacing. The error bar spans the two measured neighbours.
- **Directly below its family's lowest measured effort** (for example Gemini 3.8
  Flash low, when medium is measured): the cost is that effort's cost times the
  median of the lowest-to-next cost ratios of the other families in the plot. The
  error bar runs from the lowest to the highest of those ratios. An estimate needs
  at least one other family; with one, the bar has no width and is not drawn.

`inferred_cost` in the JSON names the effort each estimate came from, the ratio used
and its range. API prices alone cannot give the estimate, because AA reports prices
and total cost but not the tokens each task used. The method, its checks and how to
render the report are in the `llm-value` skill's `SKILL.md`.

Treat estimates as estimates, not data: effort ranks need not be evenly spaced in
cost, and the ratio is one assumption shared across families.

Neither estimates nor capability lines join the Pareto frontier: a row with no
measured cost makes no claim about cost, so it can neither dominate nor be dominated.

`data` produces structured JSON with all three indices, selected score, measured
cost, explicit exclusions, source/version/retrieval metadata and Pareto membership.
Missing values are null, never zero. Zero cost remains valid in JSON and linear
plots, but cannot be plotted on a logarithmic axis. Even for Coding or Agentic
Index, the cost workload remains **Intelligence Index tasks**, not a measured
coding-specific or agent-specific workload.

Every data-producing command requests all pages from the official
[AA Free API](https://artificialanalysis.ai/data-api/docs), with bounded retries
for network/server errors. There is no cached-data fallback. API quotas apply per
request/page. Attribution is included in plots and JSON.

Development:

```sh
uv sync
uv run pytest
uv run ruff check src tests
uv build
```

Tests use synthetic API-shaped fixtures and verify pagination, selectors,
missing/zero values, HTTP failures, export files, label geometry and CLI output.
Live authenticated validation requires your AA key.
