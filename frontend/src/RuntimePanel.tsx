import { useMemo } from "react";

import type { GameDetail, GameJob, Move, RuntimeTelemetry } from "./api";

export function RuntimePanel({
  telemetry,
  game,
  currentMove,
  activeJob,
}: {
  telemetry: RuntimeTelemetry | undefined;
  game: GameDetail | undefined;
  currentMove: Move | undefined;
  activeJob: GameJob | undefined;
}) {
  const liveGameIsActive =
    activeJob !== undefined && activeJob.game_id === game?.id && activeJob.status === "running";
  const nextColor = (game?.moves.length ?? 0) % 2 === 0 ? "white" : "black";
  const outcome = liveGameIsActive ? null : gameOutcome(game);

  return (
    <section className="panel runtime-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Live Telemetry</p>
          <h3>Runtime Monitor</h3>
        </div>
        <span className="runtime-sample">{telemetry ? relativeSampleTime(telemetry.sampled_at) : "waiting"}</span>
      </div>

      <div className="runtime-group-heading">
        <p className="eyebrow">Per-Model Metrics</p>
        <span>Tracked separately for white and black.</span>
      </div>

      <div className="model-runtime-grid">
        <PlayerRuntimeCard
          color="white"
          name={game?.white_player ?? "White"}
          game={game}
          telemetry={telemetry}
          currentMove={currentMove}
          status={modelStatus("white", liveGameIsActive, nextColor, activeJob)}
          outcome={cardOutcome("white", outcome)}
        />
        <PlayerRuntimeCard
          color="black"
          name={game?.black_player ?? "Black"}
          game={game}
          telemetry={telemetry}
          currentMove={currentMove}
          status={modelStatus("black", liveGameIsActive, nextColor, activeJob)}
          outcome={cardOutcome("black", outcome)}
        />
      </div>

      <div className="runtime-group-heading system">
        <p className="eyebrow">Shared System Runtime</p>
        <span>GPU load and Ollama residency are shared by all local models.</span>
      </div>

      <div className="runtime-sections">
        <div>
          <h4>GPU / VRAM Total</h4>
          {telemetry?.gpus.length ? (
            <div className="resource-list">
              {telemetry.gpus.map((gpu) => (
                <ResourceMeter
                  key={gpu.name}
                  label={gpu.name}
                  value={gpu.memory_used_mb}
                  max={gpu.memory_total_mb}
                  suffix="MB"
                  detail={`${gpu.utilization_percent ?? "—"}% GPU`}
                />
              ))}
            </div>
          ) : (
            <p className="muted compact">No NVIDIA telemetry available.</p>
          )}
        </div>

        <div>
          <h4>Loaded Models</h4>
          {telemetry?.ollama_models.length ? (
            <div className="runtime-model-list">
              {telemetry.ollama_models.map((model) => (
                <div key={model.name} className="runtime-model-row">
                  <strong>{model.name}</strong>
                  <span>
                    {residencyLabel(model)}
                    {model.processor ? ` · ${model.processor}` : ""}
                    {model.context_window ? ` · ctx ${model.context_window}` : ""}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted compact">No model is resident right now.</p>
          )}
        </div>
      </div>
    </section>
  );
}

function PlayerRuntimeCard({
  color,
  name,
  game,
  telemetry,
  currentMove,
  status,
  outcome,
}: {
  color: "white" | "black";
  name: string;
  game: GameDetail | undefined;
  telemetry: RuntimeTelemetry | undefined;
  currentMove: Move | undefined;
  status: string;
  outcome: "winner" | "loser" | "draw" | null;
}) {
  const stats = useMemo(() => modelRuntimeStats(game, color), [game, color]);
  const residentModel = telemetry?.ollama_models.find((model) => model.name === name);
  const moveUsage =
    currentMove?.color === color
      ? currentMove.attempts.find((attempt) => attempt.legal_ok)?.token_usage ??
        currentMove.attempts[0]?.token_usage ??
        null
      : stats.lastUsage;

  return (
    <div className={`model-runtime-card ${color}${outcome ? ` outcome-${outcome}` : ""}`}>
      {outcome === "winner" && (
        <div className="outcome-ribbon winner">
          <span aria-hidden>👑</span> Winner
        </div>
      )}
      {outcome === "draw" && (
        <div className="outcome-ribbon draw">
          <span aria-hidden>½</span> Draw
        </div>
      )}
      <div className="model-runtime-head">
        <span className="piece-badge" aria-hidden>
          {color === "white" ? "♔" : "♚"}
        </span>
        <div>
          <strong>{name}</strong>
          <span className="color-label">{color}</span>
        </div>
        <span className={`status-pill ${status}`}>{status}</span>
      </div>

      <div className="model-runtime-metrics">
        <Metric label="Move Tokens" value={moveUsage?.total_tokens ?? "—"} />
        <Metric label="Context Left" value={contextLabel(moveUsage)} />
        <Metric label="Game Tokens" value={stats.totalTokens || "—"} />
        <Metric
          label="Avg Latency"
          value={stats.averageLatencyMs === null ? "—" : `${stats.averageLatencyMs.toFixed(1)} ms`}
        />
      </div>

      <dl className="model-runtime-facts">
        <dt>Retries</dt>
        <dd>{stats.retries}</dd>
        <dt>Invalid attempts</dt>
        <dd>{stats.invalidAttempts}</dd>
        <dt>Residency</dt>
        <dd>{residentModel ? residencyStatusLabel(residentModel.offload_status) : "not resident"}</dd>
        <dt>VRAM</dt>
        <dd>{residentModel ? bytesLabel(residentModel.size_vram_bytes) : "—"}</dd>
        <dt>CPU</dt>
        <dd>{residentModel ? bytesLabel(residentModel.size_cpu_bytes) : "—"}</dd>
        <dt>Processor</dt>
        <dd>{residentModel?.processor ?? "—"}</dd>
      </dl>
    </div>
  );
}

function ResourceMeter({
  label,
  value,
  max,
  suffix,
  detail,
}: {
  label: string;
  value: number;
  max: number;
  suffix: string;
  detail: string;
}) {
  const percentage = max > 0 ? Math.min(100, Math.round((value / max) * 100)) : 0;
  return (
    <div className="resource-meter">
      <div>
        <strong>{label}</strong>
        <span>
          {value} / {max} {suffix} · {detail}
        </span>
      </div>
      <div className="meter-track" aria-hidden="true">
        <span style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}

export function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function modelRuntimeStats(game: GameDetail | undefined, color: "white" | "black") {
  const moves = game?.moves.filter((move) => move.color === color) ?? [];
  const attempts = moves.flatMap((move) => move.attempts);
  const lastAttemptWithUsage = [...attempts]
    .reverse()
    .find((attempt) => attempt.token_usage !== null);
  const latencyTotalMs = moves.reduce((total, move) => total + move.latency_total_ms, 0);

  return attempts.reduce(
    (stats, attempt) => {
      const usage = attempt.token_usage;
      return {
        totalTokens: stats.totalTokens + (usage?.total_tokens ?? 0),
        retries: stats.retries,
        invalidAttempts: stats.invalidAttempts + (attempt.legal_ok ? 0 : 1),
        lastUsage: lastAttemptWithUsage?.token_usage ?? null,
        averageLatencyMs: moves.length > 0 ? latencyTotalMs / moves.length : null,
      };
    },
    {
      totalTokens: 0,
      retries: moves.reduce((total, move) => total + move.retries_used, 0),
      invalidAttempts: 0,
      lastUsage: lastAttemptWithUsage?.token_usage ?? null,
      averageLatencyMs: moves.length > 0 ? latencyTotalMs / moves.length : null,
    },
  );
}

function gameOutcome(game: GameDetail | undefined): "white" | "black" | "draw" | null {
  switch (game?.result) {
    case "1-0":
      return "white";
    case "0-1":
      return "black";
    case "1/2-1/2":
      return "draw";
    default:
      return null;
  }
}

function cardOutcome(
  color: "white" | "black",
  outcome: "white" | "black" | "draw" | null,
): "winner" | "loser" | "draw" | null {
  if (outcome === null) {
    return null;
  }
  if (outcome === "draw") {
    return "draw";
  }
  return outcome === color ? "winner" : "loser";
}

function modelStatus(
  color: "white" | "black",
  liveGameIsActive: boolean,
  nextColor: "white" | "black",
  activeJob: GameJob | undefined,
) {
  if (liveGameIsActive) {
    return color === nextColor ? "generating" : "waiting";
  }
  if (activeJob?.status === "failed") {
    return "failed";
  }
  if (activeJob?.status === "completed") {
    return "finished";
  }
  return "idle";
}

function contextLabel(
  usage:
    | {
        estimated_context_remaining: number;
        estimated_context_window: number;
      }
    | null
    | undefined,
) {
  if (!usage) {
    return "—";
  }
  return `${usage.estimated_context_remaining} / ${usage.estimated_context_window}`;
}

function bytesLabel(value: number | null | undefined) {
  if (!value) {
    return "size unknown";
  }
  const gib = value / 1024 ** 3;
  if (gib >= 1) {
    return `${gib.toFixed(1)} GB`;
  }
  return `${(value / 1024 ** 2).toFixed(0)} MB`;
}

function residencyLabel(model: {
  size_vram_bytes: number | null;
  size_cpu_bytes: number | null;
  vram_percent: number | null;
  offload_status: string;
}) {
  const status = residencyStatusLabel(model.offload_status);
  const percent = model.vram_percent === null ? "" : ` · ${model.vram_percent.toFixed(0)}% VRAM`;
  const cpu =
    model.size_cpu_bytes !== null && model.size_cpu_bytes > 0
      ? ` · CPU ${bytesLabel(model.size_cpu_bytes)}`
      : "";
  return `${status} · VRAM ${bytesLabel(model.size_vram_bytes)}${cpu}${percent}`;
}

function residencyStatusLabel(status: string) {
  if (status === "gpu") {
    return "GPU resident";
  }
  if (status === "mixed") {
    return "Mixed offload";
  }
  if (status === "cpu") {
    return "CPU only";
  }
  return "resident";
}

function relativeSampleTime(sampledAt: string) {
  const elapsedMs = Date.now() - new Date(sampledAt).getTime();
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
    return "live";
  }
  const seconds = Math.round(elapsedMs / 1000);
  return seconds <= 1 ? "live" : `${seconds}s ago`;
}
