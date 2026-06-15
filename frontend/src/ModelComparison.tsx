import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { fetchModelComparison, type ModelComparisonRow } from "./api";
import { movesLabel, percent } from "./format";

type MetricDirection = "higher" | "lower" | "none";

type MatrixMetric = {
  label: string;
  hint?: string;
  direction: MetricDirection;
  get: (row: ModelComparisonRow) => number | null;
  format: (row: ModelComparisonRow) => string;
  title?: (row: ModelComparisonRow) => string | undefined;
};

const MATRIX_METRICS: MatrixMetric[] = [
  {
    label: "Games played",
    direction: "none",
    get: (r) => r.games_played,
    format: (r) => `${r.games_played}`,
    title: (r) => `${r.run_count} run(s)`,
  },
  {
    label: "Win rate",
    hint: "wins / games",
    direction: "higher",
    get: (r) => (r.games_played ? r.win_rate : null),
    format: (r) => (r.games_played ? percent(r.win_rate) : "—"),
    title: (r) => `95% CI ${percent(r.win_rate_ci_low)}–${percent(r.win_rate_ci_high)}`,
  },
  {
    label: "Record W-D-L-U",
    direction: "none",
    get: () => null,
    format: (r) => `${r.wins}-${r.draws}-${r.losses}-${r.unfinished}`,
  },
  {
    label: "Illegal-move rate",
    hint: "lower is better",
    direction: "lower",
    get: (r) => (r.attempt_count ? r.illegal_rate : null),
    format: (r) => (r.attempt_count ? percent(r.illegal_rate) : "—"),
    title: (r) => `95% CI ${percent(r.illegal_rate_ci_low)}–${percent(r.illegal_rate_ci_high)}`,
  },
  {
    label: "Malformed-parse rate",
    hint: "lower is better",
    direction: "lower",
    get: (r) => (r.attempt_count ? r.malformed_rate : null),
    format: (r) => (r.attempt_count ? percent(r.malformed_rate) : "—"),
    title: (r) =>
      `95% CI ${percent(r.malformed_rate_ci_low)}–${percent(r.malformed_rate_ci_high)}`,
  },
  {
    label: "Accuracy",
    hint: "evaluated moves",
    direction: "higher",
    get: (r) => (r.evaluated_move_count ? r.accuracy_rate : null),
    format: (r) =>
      r.evaluated_move_count ? `${percent(r.accuracy_rate)} (n=${r.evaluated_move_count})` : "—",
  },
  {
    label: "Avg CPL",
    hint: "centipawn loss, lower is better",
    direction: "lower",
    get: (r) => r.avg_cpl,
    format: (r) => (r.avg_cpl === null ? "—" : r.avg_cpl.toFixed(1)),
  },
  {
    label: "Blunders / game",
    hint: "lower is better",
    direction: "lower",
    get: (r) => (r.games_played ? r.blunders / r.games_played : null),
    format: (r) => (r.games_played ? (r.blunders / r.games_played).toFixed(2) : "—"),
  },
  {
    label: "Avg game length",
    hint: "moves",
    direction: "none",
    get: (r) => r.avg_game_plies,
    format: (r) => movesLabel(r.avg_game_plies),
  },
  {
    label: "Avg retries / move",
    hint: "lower is better",
    direction: "lower",
    get: (r) => r.avg_retries,
    format: (r) => r.avg_retries.toFixed(2),
  },
  {
    label: "Avg latency",
    hint: "per move, lower is better",
    direction: "lower",
    get: (r) => r.avg_latency_ms,
    format: (r) =>
      r.avg_latency_ms >= 1000
        ? `${(r.avg_latency_ms / 1000).toFixed(1)}s`
        : `${Math.round(r.avg_latency_ms)}ms`,
  },
  {
    label: "Forfeit rate",
    hint: "forfeit_invalid / games, lower is better",
    direction: "lower",
    get: (r) => (r.games_played ? r.forfeit_invalid_count / r.games_played : null),
    format: (r) =>
      r.games_played ? percent(r.forfeit_invalid_count / r.games_played) : "—",
  },
  {
    label: "Total tokens",
    direction: "none",
    get: (r) => r.total_tokens,
    format: (r) => r.total_tokens.toLocaleString(),
  },
];

function bestColumns(values: (number | null)[], direction: MetricDirection): Set<number> {
  const best = new Set<number>();
  if (direction === "none") {
    return best;
  }
  const known = values.filter((value): value is number => value !== null);
  if (known.length < 2) {
    return best; // nothing to compare against
  }
  const target = direction === "higher" ? Math.max(...known) : Math.min(...known);
  values.forEach((value, index) => {
    if (value !== null && value === target) {
      best.add(index);
    }
  });
  return best;
}

function samplerParam(row: ModelComparisonRow, key: string) {
  const value = row.sampler_params?.[key];
  if (typeof value === "number") {
    return Number.isInteger(value)
      ? value.toString()
      : value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (typeof value === "string" && value.length > 0) {
    return value;
  }
  return null;
}

// Compact config descriptors — only used to tell otherwise same-named models apart.
function configFields(row: ModelComparisonRow): string[] {
  const fields: string[] = [];
  if (row.quantization) {
    fields.push(row.quantization);
  }
  if (row.context_window) {
    fields.push(`ctx ${row.context_window}`);
  }
  const temperature = samplerParam(row, "temperature");
  if (temperature !== null) {
    fields.push(`temp ${temperature}`);
  }
  const topP = samplerParam(row, "top_p");
  if (topP !== null) {
    fields.push(`top_p ${topP}`);
  }
  const numPredict = samplerParam(row, "num_predict");
  if (numPredict !== null) {
    fields.push(`pred ${numPredict}`);
  }
  return fields;
}

function formatDay(iso: string | null) {
  if (!iso) {
    return null;
  }
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

// "When was this played" — a single day, or a range when aggregated across runs.
// Falls back to the snapshot registration date if run dates are unavailable.
function playedLabel(row: ModelComparisonRow) {
  const from = formatDay(row.played_from);
  const to = formatDay(row.played_to ?? row.snapshot_created_at);
  if (from && to && from !== to) {
    return `${from} – ${to}`;
  }
  return to ?? from;
}

// Full detail for the hover tooltip — everything that doesn't fit inline lives here.
function modelComparisonTitle(row: ModelComparisonRow) {
  const lines = [row.label];
  const played = playedLabel(row);
  if (played) {
    lines.push(`played ${played}`);
  }
  const config = configFields(row);
  if (config.length) {
    lines.push(config.join(" · "));
  }
  if (row.runtime_version) {
    lines.push(`runtime ${row.runtime_version}`);
  }
  if (row.model_snapshot_id !== null) {
    lines.push(`snapshot #${row.model_snapshot_id}`);
  }
  if (row.run_ids?.length) {
    lines.push(`runs: ${row.run_ids.map((id) => `#${id}`).join(", ")}`);
  }
  lines.push(`${row.legality_mode} · ${row.color}`);
  return lines.filter(Boolean).join("\n");
}

// Build the short secondary line under each model name. Readability first: show
// *when* it was played plus only the config bits needed to disambiguate models that
// share a display name. Computed over the full row set so the line is stable
// regardless of which columns are currently toggled on. Keyed by model_key.
function buildModelMetaMap(allRows: ModelComparisonRow[]): Map<string, string> {
  const byLabel = new Map<string, ModelComparisonRow[]>();
  for (const row of allRows) {
    const group = byLabel.get(row.label) ?? [];
    group.push(row);
    byLabel.set(row.label, group);
  }
  const meta = new Map<string, string>();
  for (const row of allRows) {
    const siblings = byLabel.get(row.label) ?? [row];
    const parts: string[] = [];
    const played = playedLabel(row);
    if (played) {
      parts.push(played);
    }
    if (row.run_count > 1) {
      parts.push(`${row.run_count} runs`);
    }
    if (siblings.length > 1) {
      // Surface only config fields not shared by every same-named sibling.
      const counts = new Map<string, number>();
      for (const sibling of siblings) {
        for (const field of configFields(sibling)) {
          counts.set(field, (counts.get(field) ?? 0) + 1);
        }
      }
      for (const field of configFields(row)) {
        if ((counts.get(field) ?? 0) < siblings.length) {
          parts.push(field);
        }
      }
    }
    meta.set(row.model_key, parts.join(" · "));
  }
  // Final guard: if same-named rows still render identically, append an id so every
  // column stays distinguishable.
  const fingerprints = new Map<string, ModelComparisonRow[]>();
  for (const row of allRows) {
    const fingerprint = `${row.label}|${meta.get(row.model_key) ?? ""}`;
    const group = fingerprints.get(fingerprint) ?? [];
    group.push(row);
    fingerprints.set(fingerprint, group);
  }
  for (const group of fingerprints.values()) {
    if (group.length < 2) {
      continue;
    }
    for (const row of group) {
      const id =
        row.model_snapshot_id !== null
          ? `snap #${row.model_snapshot_id}`
          : row.run_ids[0] !== undefined
            ? `run #${row.run_ids[0]}`
            : null;
      if (!id) {
        continue;
      }
      const existing = meta.get(row.model_key) ?? "";
      meta.set(row.model_key, existing ? `${existing} · ${id}` : id);
    }
  }
  return meta;
}

export function ModelComparison() {
  const [legality, setLegality] = useState<string>("constrained");
  const [color, setColor] = useState<string>("all");
  // Track which models are hidden (by stable model_key) so newly-seen models
  // default to visible. Empty set => show every model.
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const comparisonQuery = useQuery({
    queryKey: ["model-comparison", legality, color],
    queryFn: () => fetchModelComparison({ legalityMode: legality, color }),
  });
  const allRows = useMemo(() => comparisonQuery.data ?? [], [comparisonQuery.data]);
  const metaByKey = useMemo(() => buildModelMetaMap(allRows), [allRows]);
  const rows = allRows.filter((row) => !hidden.has(row.model_key));
  const hasLowSample = rows.some((row) => row.low_sample);
  const gridTemplate = {
    gridTemplateColumns: `minmax(180px, 220px) repeat(${rows.length}, minmax(120px, 1fr))`,
  };
  const toggleModel = (key: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });

  return (
    <section className="model-comparison">
      <header className="model-comparison-header">
        <div>
          <p className="eyebrow">Benchmark Aggregate</p>
          <h1>Model Comparison</h1>
          <p className="muted">
            Per-model results aggregated across all runs. Best value in each row is
            highlighted. Open and constrained legality are separate benchmarks.
          </p>
        </div>
        <div className="filter-row">
          <select
            aria-label="Legality mode"
            value={legality}
            onChange={(event) => setLegality(event.target.value)}
          >
            <option value="constrained">Constrained</option>
            <option value="open">Open</option>
          </select>
          <select
            aria-label="Color"
            value={color}
            onChange={(event) => setColor(event.target.value)}
          >
            <option value="all">Both colors</option>
            <option value="white">White</option>
            <option value="black">Black</option>
          </select>
          {allRows.length > 0 ? (
            <details className="model-picker">
              <summary>
                Models ({allRows.length - hidden.size}/{allRows.length})
              </summary>
              <div className="model-picker-menu">
                <div className="model-picker-actions">
                  <button type="button" onClick={() => setHidden(new Set())}>
                    All
                  </button>
                  <button
                    type="button"
                    onClick={() => setHidden(new Set(allRows.map((row) => row.model_key)))}
                  >
                    None
                  </button>
                </div>
                {allRows.map((row) => (
                  <label
                    key={row.model_key}
                    className="model-picker-item"
                    title={modelComparisonTitle(row)}
                  >
                    <input
                      type="checkbox"
                      checked={!hidden.has(row.model_key)}
                      onChange={() => toggleModel(row.model_key)}
                    />
                    <span className="model-picker-label">
                      <span className="model-picker-name">{row.label}</span>
                      {metaByKey.get(row.model_key) ? (
                        <span className="model-picker-meta">{metaByKey.get(row.model_key)}</span>
                      ) : null}
                    </span>
                  </label>
                ))}
              </div>
            </details>
          ) : null}
        </div>
      </header>

      {comparisonQuery.isLoading ? (
        <p className="muted">Loading…</p>
      ) : allRows.length === 0 ? (
        <p className="muted">
          No materialized summaries yet for this mode. Play games, then run{" "}
          <code>arena rebuild-summaries</code> (Stockfish matches rebuild automatically).
        </p>
      ) : rows.length === 0 ? (
        <p className="muted">All models are hidden — pick at least one in the Models menu.</p>
      ) : (
        <div className="matrix-table">
          <div className="matrix-head" style={gridTemplate}>
            <span className="matrix-metric">Metric</span>
            {rows.map((row) => (
              <span key={row.model_key} className="matrix-model" title={modelComparisonTitle(row)}>
                <span className="matrix-model-name">
                  {row.label}
                  {row.low_sample ? <sup>*</sup> : null}
                </span>
                {metaByKey.get(row.model_key) ? (
                  <small>{metaByKey.get(row.model_key)}</small>
                ) : null}
              </span>
            ))}
          </div>
          {MATRIX_METRICS.map((metric) => {
            const values = rows.map((row) => metric.get(row));
            const best = bestColumns(values, metric.direction);
            return (
              <div key={metric.label} className="matrix-row" style={gridTemplate}>
                <span className="matrix-metric">
                  {metric.label}
                  {metric.hint ? <small>{metric.hint}</small> : null}
                </span>
                {rows.map((row, index) => (
                  <span
                    key={row.model_key}
                    className={best.has(index) ? "matrix-cell best" : "matrix-cell"}
                    title={metric.title?.(row)}
                  >
                    {metric.format(row)}
                  </span>
                ))}
              </div>
            );
          })}
        </div>
      )}

      {hasLowSample ? (
        <p className="matrix-footnote">
          <sup>*</sup> fewer than 10 games — rates have wide confidence intervals; hover a
          cell for the 95% CI.
        </p>
      ) : null}
    </section>
  );
}
