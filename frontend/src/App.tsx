import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RefreshCcw,
  SkipBack,
  SkipForward,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  fetchGame,
  fetchGameJobs,
  fetchGames,
  fetchLeaderboard,
  fetchModelOptions,
  fetchRunComparison,
  fetchRunEvents,
  fetchRuntimeTelemetry,
  startGame,
  type GameDetail,
  type GameJob,
  type LeaderboardRow,
  type ModelOption,
  type Move,
  type OllamaPreset,
  type RuntimeTelemetry,
  type RunComparisonRow,
  type StartGamePayload,
} from "./api";
import { moveSquares, parseFenBoard, startFen } from "./chess";

export default function App() {
  const gamesQuery = useQuery({ queryKey: ["games"], queryFn: fetchGames });
  const modelOptionsQuery = useQuery({ queryKey: ["models"], queryFn: fetchModelOptions });
  const gameJobsQuery = useQuery({
    queryKey: ["game-jobs"],
    queryFn: fetchGameJobs,
    refetchInterval: 2_000,
  });
  const runtimeQuery = useQuery({
    queryKey: ["runtime-telemetry"],
    queryFn: fetchRuntimeTelemetry,
    refetchInterval: 2_000,
  });
  const [leaderboardRunId, setLeaderboardRunId] = useState<number | "all">("all");
  const [leaderboardColor, setLeaderboardColor] = useState<string>("all");
  const [leaderboardLegality, setLeaderboardLegality] = useState<string>("all");
  const leaderboardQuery = useQuery({
    queryKey: ["leaderboard", leaderboardRunId, leaderboardColor, leaderboardLegality],
    queryFn: () =>
      fetchLeaderboard({
        runId: leaderboardRunId === "all" ? undefined : leaderboardRunId,
        color: leaderboardColor === "all" ? undefined : leaderboardColor,
        legalityMode: leaderboardLegality === "all" ? undefined : leaderboardLegality,
      }),
  });
  const comparisonQuery = useQuery({
    queryKey: ["run-comparison"],
    queryFn: fetchRunComparison,
  });
  const [sidebarTab, setSidebarTab] = useState<"start" | "history">("start");
  const [workspaceTab, setWorkspaceTab] = useState<"debug" | "analysis" | "benchmark">("debug");
  const [liveEvents, setLiveEvents] = useState(0);
  const [selectedGameId, setSelectedGameId] = useState<number | null>(null);
  const [activeLiveJobId, setActiveLiveJobId] = useState<string | null>(null);
  const effectiveGameId = selectedGameId ?? gamesQuery.data?.[0]?.id ?? null;

  const gameQuery = useQuery({
    queryKey: ["game", effectiveGameId],
    queryFn: () => fetchGame(effectiveGameId!),
    enabled: effectiveGameId !== null,
  });
  const [plyIndex, setPlyIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackDelayMs, setPlaybackDelayMs] = useState(900);
  const game = gameQuery.data;
  const refetchGames = gamesQuery.refetch;
  const refetchGame = gameQuery.refetch;
  const refetchLeaderboard = leaderboardQuery.refetch;
  const refetchComparison = comparisonQuery.refetch;
  const refetchJobs = gameJobsQuery.refetch;

  const selectGameWhenJobStarts = (jobId: string) => {
    const poll = async (attempt: number) => {
      const jobs = await fetchGameJobs();
      void refetchJobs();
      const job = jobs.find((row) => row.id === jobId);
      if (job?.game_id) {
        setSelectedGameId(job.game_id);
        setPlyIndex(0);
        void refetchGames();
        void refetchGame();
        return;
      }
      if (!job || job.status === "failed" || attempt >= 90) {
        return;
      }
      window.setTimeout(() => {
        void poll(attempt + 1);
      }, 1_000);
    };
    void poll(0);
  };

  const startGameMutation = useMutation({
    mutationFn: startGame,
    onSuccess: (response) => {
      setActiveLiveJobId(response.job_id);
      void refetchJobs();
      void refetchGames();
      selectGameWhenJobStarts(response.job_id);
    },
  });

  useEffect(() => {
    const source = new EventSource("/api/stream/games?interval_seconds=2");
    source.addEventListener("games", () => {
      setLiveEvents((value) => value + 1);
      void refetchGames();
      if (effectiveGameId !== null) {
        void refetchGame();
      }
      void refetchLeaderboard();
      void refetchComparison();
      void refetchJobs();
    });
    source.onerror = () => source.close();
    return () => source.close();
  }, [
    effectiveGameId,
    refetchComparison,
    refetchGame,
    refetchGames,
    refetchJobs,
    refetchLeaderboard,
  ]);

  const maxPly = game?.moves.length ?? 0;
  const activeLiveJob = gameJobsQuery.data?.find((job) => job.id === activeLiveJobId);
  const followLiveGame =
    activeLiveJob?.game_id === effectiveGameId && activeLiveJob.status !== "failed";
  const replayPlyIndex = followLiveGame ? maxPly : Math.min(plyIndex, maxPly);
  const currentMove = replayPlyIndex > 0 ? game?.moves[replayPlyIndex - 1] : undefined;
  const fen = currentMove?.fen_after ?? game?.moves[0]?.fen_before ?? startFen;
  const whitePlayer = game?.white_player ?? "White";
  const blackPlayer = game?.black_player ?? "Black";

  useEffect(() => {
    if (!isPlaying || replayPlyIndex >= maxPly) {
      return;
    }
    const timer = window.setTimeout(() => {
      const nextPly = Math.min(maxPly, replayPlyIndex + 1);
      setPlyIndex(nextPly);
      if (nextPly >= maxPly) {
        setIsPlaying(false);
      }
    }, playbackDelayMs);
    return () => window.clearTimeout(timer);
  }, [isPlaying, maxPly, playbackDelayMs, replayPlyIndex]);

  const togglePlayback = () => {
    setActiveLiveJobId(null);
    if (maxPly === 0) {
      return;
    }
    if (!isPlaying && replayPlyIndex >= maxPly) {
      setPlyIndex(0);
    }
    setIsPlaying((value) => !value);
  };

  return (
    <main className="arena-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div>
            <p className="eyebrow">Software Arena</p>
            <h1>Replay</h1>
          </div>
          <button
            className="icon-button"
            type="button"
            onClick={() => {
              void gamesQuery.refetch();
              if (effectiveGameId !== null) {
                void gameQuery.refetch();
              }
            }}
            aria-label="Refresh games"
            title="Refresh games"
          >
            <RefreshCcw size={18} />
          </button>
        </div>

        <div className="sidebar-tabs" role="tablist" aria-label="Sidebar sections">
          <button
            className={sidebarTab === "start" ? "active" : ""}
            type="button"
            role="tab"
            aria-selected={sidebarTab === "start"}
            onClick={() => setSidebarTab("start")}
          >
            Start
          </button>
          <button
            className={sidebarTab === "history" ? "active" : ""}
            type="button"
            role="tab"
            aria-selected={sidebarTab === "history"}
            onClick={() => setSidebarTab("history")}
          >
            History
            <span>{gamesQuery.data?.length ?? 0}</span>
          </button>
        </div>

        {sidebarTab === "start" ? (
          <StartGamePanel
            modelOptions={modelOptionsQuery.data ?? []}
            jobs={gameJobsQuery.data ?? []}
            loadingModels={modelOptionsQuery.isLoading}
            submitting={startGameMutation.isPending}
            error={startGameMutation.error}
            onSubmit={(payload) => startGameMutation.mutate(payload)}
          />
        ) : (
          <GameList
            games={gamesQuery.data ?? []}
            selectedGameId={effectiveGameId}
            onSelect={(id) => {
              setSelectedGameId(id);
              setActiveLiveJobId(null);
              setIsPlaying(false);
              setPlyIndex(0);
            }}
            loading={gamesQuery.isLoading}
          />
        )}
      </aside>

      <section className="workspace">
        <header className="game-header">
          <div>
            <p className="eyebrow">Game {game?.id ?? "-"}</p>
            <h2>{game ? `${whitePlayer} vs ${blackPlayer}` : "No game selected"}</h2>
            {game && (
              <p className="game-result">
                {game.result} · {game.termination_reason ?? "running"}
              </p>
            )}
          </div>
          <PlyControls
            plyIndex={replayPlyIndex}
            maxPly={maxPly}
            isPlaying={isPlaying}
            playbackDelayMs={playbackDelayMs}
            onStart={() => {
              setActiveLiveJobId(null);
              setIsPlaying(false);
              setPlyIndex(0);
            }}
            onPrev={() => {
              setActiveLiveJobId(null);
              setPlyIndex((value) => Math.max(0, value - 1));
            }}
            onTogglePlay={togglePlayback}
            onNext={() => {
              setActiveLiveJobId(null);
              setPlyIndex((value) => Math.min(maxPly, value + 1));
            }}
            onEnd={() => {
              setActiveLiveJobId(null);
              setIsPlaying(false);
              setPlyIndex(maxPly);
            }}
            onChangePly={(value) => {
              setActiveLiveJobId(null);
              setIsPlaying(false);
              setPlyIndex(value);
            }}
            onSpeedChange={setPlaybackDelayMs}
          />
        </header>
        <div className="live-strip">
          <span className="live-dot" />
          <span>Live stream connected</span>
          <strong>{liveEvents}</strong>
        </div>

        <div className="replay-grid">
          <ChessBoard
            fen={fen}
            activeMove={currentMove}
            plyIndex={replayPlyIndex}
            whitePlayer={whitePlayer}
            blackPlayer={blackPlayer}
          />
          <div className="side-stack">
            <RuntimePanel
              telemetry={runtimeQuery.data}
              game={game}
              currentMove={currentMove}
              activeJob={activeLiveJob}
            />
            <MoveListPanel
              game={game}
              plyIndex={replayPlyIndex}
              onSelectPly={(value) => {
                setActiveLiveJobId(null);
                setIsPlaying(false);
                setPlyIndex(value);
              }}
            />
          </div>
        </div>

        <div className="workspace-tabs" role="tablist" aria-label="Game detail sections">
          <button
            className={workspaceTab === "debug" ? "active" : ""}
            type="button"
            role="tab"
            aria-selected={workspaceTab === "debug"}
            onClick={() => setWorkspaceTab("debug")}
          >
            Move Debug
          </button>
          <button
            className={workspaceTab === "analysis" ? "active" : ""}
            type="button"
            role="tab"
            aria-selected={workspaceTab === "analysis"}
            onClick={() => setWorkspaceTab("analysis")}
          >
            Analysis
          </button>
          <button
            className={workspaceTab === "benchmark" ? "active" : ""}
            type="button"
            role="tab"
            aria-selected={workspaceTab === "benchmark"}
            onClick={() => setWorkspaceTab("benchmark")}
          >
            Benchmark
          </button>
        </div>

        {workspaceTab === "debug" && <MoveDebugPanel currentMove={currentMove} />}
        {workspaceTab === "analysis" && (
          <AnalysisPanel game={game} currentMove={currentMove} />
        )}
        {workspaceTab === "benchmark" && (
          <BenchmarkPanel
            game={game}
            leaderboardRows={leaderboardQuery.data ?? []}
            comparisonRows={comparisonQuery.data ?? []}
            leaderboardRunId={leaderboardRunId}
            leaderboardColor={leaderboardColor}
            leaderboardLegality={leaderboardLegality}
            runOptions={uniqueRunIds(gamesQuery.data ?? [])}
            onRunIdChange={setLeaderboardRunId}
            onColorChange={setLeaderboardColor}
            onLegalityModeChange={setLeaderboardLegality}
          />
        )}
      </section>
    </main>
  );
}

function RuntimePanel({
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

  return (
    <section className="panel runtime-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Live Telemetry</p>
          <h3>Model Runtime</h3>
        </div>
        <span className="runtime-sample">{telemetry ? relativeSampleTime(telemetry.sampled_at) : "waiting"}</span>
      </div>

      <div className="model-runtime-grid">
        <PlayerRuntimeCard
          color="white"
          name={game?.white_player ?? "White"}
          game={game}
          telemetry={telemetry}
          currentMove={currentMove}
          status={modelStatus("white", liveGameIsActive, nextColor, activeJob)}
        />
        <PlayerRuntimeCard
          color="black"
          name={game?.black_player ?? "Black"}
          game={game}
          telemetry={telemetry}
          currentMove={currentMove}
          status={modelStatus("black", liveGameIsActive, nextColor, activeJob)}
        />
      </div>

      <div className="runtime-sections">
        <div>
          <h4>Global GPU / VRAM</h4>
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
          <h4>Runtime Residency</h4>
          {telemetry?.ollama_models.length ? (
            <div className="runtime-model-list">
              {telemetry.ollama_models.map((model) => (
                <div key={model.name} className="runtime-model-row">
                  <strong>{model.name}</strong>
                  <span>
                    {bytesLabel(model.size_vram_bytes ?? model.size_bytes)}
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
}: {
  color: "white" | "black";
  name: string;
  game: GameDetail | undefined;
  telemetry: RuntimeTelemetry | undefined;
  currentMove: Move | undefined;
  status: string;
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
    <div className={`model-runtime-card ${color}`}>
      <div className="model-runtime-head">
        <span className="color-swatch" />
        <div>
          <strong>{name}</strong>
          <span>{color}</span>
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
        <dd>{residentModel ? "resident" : "not resident"}</dd>
        <dt>VRAM</dt>
        <dd>{residentModel ? bytesLabel(residentModel.size_vram_bytes ?? residentModel.size_bytes) : "—"}</dd>
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

function RunComparison({ rows }: { rows: RunComparisonRow[] }) {
  return (
    <section className="comparison-panel">
      <div className="section-heading">
        <p className="eyebrow">Run Comparison</p>
        <h3>Runs</h3>
      </div>
      {rows.length === 0 ? (
        <p className="muted">No run summaries available.</p>
      ) : (
        <div className="comparison-table">
          <div className="comparison-head">
            <span>Run</span>
            <span>Games</span>
            <span>W-D-L-U</span>
            <span>Avg CPL</span>
            <span>Illegal</span>
            <span>Retries</span>
            <span>Latency</span>
            <span>Tokens</span>
          </div>
          {rows.map((row) => (
            <div key={row.run_id} className="comparison-row">
              <span>{row.run_id}</span>
              <span>{row.games_played}</span>
              <span>
                {row.wins}-{row.draws}-{row.losses}-{row.unfinished}
              </span>
              <span>{row.avg_cpl === null ? "—" : row.avg_cpl.toFixed(1)}</span>
              <span>{(row.illegal_rate * 100).toFixed(1)}%</span>
              <span>{row.avg_retries.toFixed(2)}</span>
              <span>{row.avg_latency_ms.toFixed(1)} ms</span>
              <span>{row.total_tokens}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function EvalGraph({ game }: { game: GameDetail | undefined }) {
  const data = useMemo(() => {
    if (!game) {
      return [];
    }
    return game.moves
      .map((move) => {
        const evaluation = move.engine_evaluations[0];
        return {
          ply: move.ply,
          after: evaluation?.eval_after_cp ?? null,
          cpl: evaluation?.centipawn_loss ?? null,
          label: move.accepted_san,
        };
      })
      .filter((row) => row.after !== null || row.cpl !== null);
  }, [game]);

  return (
    <section className="eval-graph">
      <div className="section-heading">
        <p className="eyebrow">Engine Trace</p>
        <h3>Evaluation Graph</h3>
      </div>
      {!game ? (
        <p className="muted">No game selected.</p>
      ) : data.length === 0 ? (
        <p className="muted">No engine evaluations are stored for this game yet.</p>
      ) : (
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={data} margin={{ top: 10, right: 18, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="#e3ded4" />
              <XAxis dataKey="ply" />
              <YAxis width={54} />
              <Tooltip
                formatter={(value, name) => [value ?? "—", name === "after" ? "Eval cp" : "CPL"]}
                labelFormatter={(label) => `Ply ${label}`}
              />
              <Line
                type="monotone"
                dataKey="after"
                stroke="#1e6b58"
                strokeWidth={2}
                dot={{ r: 3 }}
                connectNulls={false}
              />
              <Line
                type="monotone"
                dataKey="cpl"
                stroke="#b45f06"
                strokeWidth={2}
                dot={{ r: 3 }}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

function OperationalEventsPanel({ game }: { game: GameDetail | undefined }) {
  const eventsQuery = useQuery({
    queryKey: ["run-events", game?.run_id],
    queryFn: () => fetchRunEvents(game!.run_id!),
    enabled: game?.run_id !== null && game?.run_id !== undefined,
  });

  if (!game?.run_id) {
    return null;
  }
  const events = eventsQuery.data ?? [];
  return (
    <section className="events-panel">
      <div className="section-heading">
        <p className="eyebrow">Operational Telemetry</p>
        <h3>Run Events</h3>
      </div>
      {events.length === 0 ? (
        <p className="muted">No operational events recorded for this run.</p>
      ) : (
        <div className="event-list">
          {events.map((event) => (
            <div key={event.id} className={`event-row ${event.severity}`}>
              <strong>{event.event_kind}</strong>
              <span>{event.message}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function uniqueRunIds(rows: { run_id: number | null }[]): number[] {
  return [...new Set(rows.map((row) => row.run_id).filter((id) => id !== null))].sort(
    (a, b) => b - a,
  );
}

function GameList({
  games,
  selectedGameId,
  onSelect,
  loading,
}: {
  games: { id: number; result: string; termination_reason: string | null; ended_at: string | null }[];
  selectedGameId: number | null;
  onSelect: (id: number) => void;
  loading: boolean;
}) {
  if (loading) {
    return <div className="empty-state">Loading games</div>;
  }
  if (games.length === 0) {
    return <div className="empty-state">No games persisted yet</div>;
  }
  return (
    <div className="game-list">
      {games.map((game) => (
        <button
          key={game.id}
          className={game.id === selectedGameId ? "game-row selected" : "game-row"}
          type="button"
          onClick={() => onSelect(game.id)}
        >
          <span>Game {game.id}</span>
          <span>{game.result}</span>
          <small>{game.termination_reason ?? "running"}</small>
        </button>
      ))}
    </div>
  );
}

function StartGamePanel({
  modelOptions,
  jobs,
  loadingModels,
  submitting,
  error,
  onSubmit,
}: {
  modelOptions: ModelOption[];
  jobs: GameJob[];
  loadingModels: boolean;
  submitting: boolean;
  error: Error | null;
  onSubmit: (payload: StartGamePayload) => void;
}) {
  const ollamaOptions = modelOptions.filter((option) => option.provider === "ollama");
  const defaultWhite = ollamaOptions[0]?.id ?? modelOptions[0]?.id ?? "random";
  const defaultBlack = ollamaOptions[1]?.id ?? ollamaOptions[0]?.id ?? modelOptions[1]?.id ?? defaultWhite;
  const [white, setWhite] = useState<string | null>(null);
  const [black, setBlack] = useState<string | null>(null);
  const [legalityMode, setLegalityMode] = useState<"open" | "constrained">("constrained");
  const [ollamaPreset, setOllamaPreset] = useState<OllamaPreset>("strict");
  const [maxPlies, setMaxPlies] = useState("");
  const selectedWhite = white ?? defaultWhite;
  const selectedBlack = black ?? defaultBlack;

  const recentJobs = jobs.slice(0, 4);

  return (
    <section className="start-game">
      <div className="section-heading">
        <p className="eyebrow">Match Control</p>
        <h3>Start Game</h3>
      </div>
      <form
        className="start-game-form"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit({
            white: selectedWhite,
            black: selectedBlack,
            legality_mode: legalityMode,
            ollama_preset: ollamaPreset,
            max_plies: maxPlies ? Number(maxPlies) : null,
          });
        }}
      >
        <label>
          <span>White</span>
          <ModelInput
            value={selectedWhite}
            options={modelOptions}
            disabled={submitting}
            onChange={setWhite}
          />
        </label>
        <label>
          <span>Ollama preset</span>
          <select
            value={ollamaPreset}
            disabled={submitting}
            onChange={(event) => setOllamaPreset(event.target.value as OllamaPreset)}
          >
            <option value="strict">Strict deterministic</option>
            <option value="low_creativity">Low creativity</option>
            <option value="thinking_if_supported">Thinking if supported</option>
          </select>
        </label>
        <label>
          <span>Black</span>
          <ModelInput
            value={selectedBlack}
            options={modelOptions}
            disabled={submitting}
            onChange={setBlack}
          />
        </label>
        <label>
          <span>Legality</span>
          <select
            value={legalityMode}
            disabled={submitting}
            onChange={(event) => setLegalityMode(event.target.value as "open" | "constrained")}
          >
            <option value="constrained">Legal move list</option>
            <option value="open">Open</option>
          </select>
        </label>
        <label>
          <span>Max plies</span>
          <input
            type="number"
            min="1"
            value={maxPlies}
            placeholder="No limit"
            disabled={submitting}
            onChange={(event) => setMaxPlies(event.target.value)}
          />
        </label>
        <button className="primary-button" type="submit" disabled={submitting || !selectedWhite || !selectedBlack}>
          {submitting ? "Starting" : "Start game"}
        </button>
      </form>
      {loadingModels && <p className="muted compact">Loading local Ollama models.</p>}
      {error && <p className="error-text">{error.message}</p>}
      {recentJobs.length > 0 && (
        <div className="job-list">
          {recentJobs.map((job) => (
            <div key={job.id} className={`job-row ${job.status}`}>
              <span>
                {job.white} vs {job.black}
              </span>
              <strong>{job.ollama_preset} · {jobLabel(job)}</strong>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function ModelInput({
  value,
  options,
  disabled,
  onChange,
}: {
  value: string;
  options: ModelOption[];
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
      {options.map((option) => (
        <option key={`${option.provider}-${option.id}`} value={option.id}>
          {option.label} · {option.provider}
        </option>
      ))}
    </select>
  );
}

function jobLabel(job: GameJob) {
  if (job.status === "running") {
    return "running";
  }
  if (job.status === "failed") {
    return job.error ?? "failed";
  }
  return job.game_id ? `Game ${job.game_id}` : "completed";
}

function PlyControls({
  plyIndex,
  maxPly,
  isPlaying,
  playbackDelayMs,
  onStart,
  onPrev,
  onTogglePlay,
  onNext,
  onEnd,
  onChangePly,
  onSpeedChange,
}: {
  plyIndex: number;
  maxPly: number;
  isPlaying: boolean;
  playbackDelayMs: number;
  onStart: () => void;
  onPrev: () => void;
  onTogglePlay: () => void;
  onNext: () => void;
  onEnd: () => void;
  onChangePly: (value: number) => void;
  onSpeedChange: (value: number) => void;
}) {
  return (
    <div className="replay-controls" aria-label="Replay controls">
      <div className="ply-controls">
        <button
          className="icon-button"
          type="button"
          onClick={onStart}
          disabled={plyIndex === 0}
          aria-label="First position"
          title="First position"
        >
          <SkipBack size={18} />
        </button>
        <button
          className="icon-button"
          type="button"
          onClick={onPrev}
          disabled={plyIndex === 0}
          aria-label="Previous ply"
          title="Previous ply"
        >
          <ChevronLeft size={20} />
        </button>
        <button
          className="icon-button play-button"
          type="button"
          onClick={onTogglePlay}
          disabled={maxPly === 0}
          aria-label={isPlaying ? "Pause replay" : "Play replay"}
          title={isPlaying ? "Pause replay" : "Play replay"}
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <button
          className="icon-button"
          type="button"
          onClick={onNext}
          disabled={plyIndex === maxPly}
          aria-label="Next ply"
          title="Next ply"
        >
          <ChevronRight size={20} />
        </button>
        <button
          className="icon-button"
          type="button"
          onClick={onEnd}
          disabled={plyIndex === maxPly}
          aria-label="Final position"
          title="Final position"
        >
          <SkipForward size={18} />
        </button>
        <span>
          {plyIndex} / {maxPly}
        </span>
      </div>
      <div className="timeline-controls">
        <input
          type="range"
          min="0"
          max={maxPly}
          value={plyIndex}
          onChange={(event) => onChangePly(Number(event.target.value))}
          aria-label="Replay timeline"
        />
        <select
          value={playbackDelayMs}
          onChange={(event) => onSpeedChange(Number(event.target.value))}
          aria-label="Playback speed"
        >
          <option value={1400}>0.5x</option>
          <option value={900}>1x</option>
          <option value={450}>2x</option>
        </select>
      </div>
    </div>
  );
}

function ChessBoard({
  fen,
  activeMove,
  plyIndex,
  whitePlayer,
  blackPlayer,
}: {
  fen: string;
  activeMove: Move | undefined;
  plyIndex: number;
  whitePlayer: string;
  blackPlayer: string;
}) {
  const squares = useMemo(() => parseFenBoard(fen), [fen]);
  const highlighted = useMemo(() => moveSquares(activeMove?.accepted_uci), [activeMove]);

  return (
    <div className="board-wrap">
      <PlayerStrip color="black" name={blackPlayer} active={activeMove?.color === "black"} />
      <div className="board" aria-label="Chess board">
        {squares.map((square, index) => {
          const file = index % 8;
          const rank = Math.floor(index / 8);
          const dark = (file + rank) % 2 === 1;
          return (
            <div
              key={square.square}
              className={[
                "square",
                dark ? "dark" : "light",
                highlighted.has(square.square) ? "highlight" : "",
              ].join(" ")}
            >
              <span className={`piece ${square.piece?.color ?? ""}`}>{square.piece?.symbol}</span>
              <span className="coord">{square.square}</span>
            </div>
          );
        })}
      </div>
      <PlayerStrip
        color="white"
        name={whitePlayer}
        active={plyIndex === 0 || activeMove?.color === "white"}
      />
    </div>
  );
}

function PlayerStrip({
  color,
  name,
  active,
}: {
  color: "white" | "black";
  name: string;
  active: boolean;
}) {
  return (
    <div className={active ? `player-strip ${color} active` : `player-strip ${color}`}>
      <span className="color-swatch" />
      <strong>{name}</strong>
      <span>{color}</span>
    </div>
  );
}

function MoveListPanel({
  game,
  plyIndex,
  onSelectPly,
}: {
  game: GameDetail | undefined;
  plyIndex: number;
  onSelectPly: (ply: number) => void;
}) {
  if (!game) {
    return <section className="panel">Select a game to inspect its moves.</section>;
  }

  return (
    <section className="panel">
      <div className="section-heading">
        <p className="eyebrow">Move List</p>
        <h3>{plyIndex === 0 ? "Starting Position" : `Ply ${plyIndex}`}</h3>
      </div>
      <MoveTable moves={game.moves} activePly={plyIndex} onSelectPly={onSelectPly} />
    </section>
  );
}

function MoveDebugPanel({ currentMove }: { currentMove: Move | undefined }) {
  if (!currentMove) {
    return (
      <section className="panel detail-panel">
        <div className="section-heading">
          <p className="eyebrow">Move Debug</p>
          <h3>No move selected</h3>
        </div>
        <p className="muted">Move attempts appear after selecting a ply in the replay timeline.</p>
      </section>
    );
  }

  const acceptedAttempt = currentMove.attempts.find((attempt) => attempt.legal_ok);
  const tokenUsage = acceptedAttempt?.token_usage ?? currentMove.attempts[0]?.token_usage;

  return (
    <section className="panel detail-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Move Debug</p>
          <h3>
            Ply {currentMove.ply} · {currentMove.color} · {currentMove.accepted_san}
          </h3>
        </div>
      </div>

      <div className="metric-grid">
        <Metric label="Retries" value={currentMove.retries_used} />
        <Metric label="Attempts" value={currentMove.attempts.length} />
        <Metric label="Latency" value={`${currentMove.latency_total_ms.toFixed(1)} ms`} />
        <Metric label="Tokens" value={tokenUsage?.total_tokens ?? "—"} />
      </div>

      <div className="attempt-list">
        {currentMove.attempts.map((attempt) => (
          <div key={attempt.id} className="attempt-row">
            <span>#{attempt.attempt_number}</span>
            <span>{attempt.parsed_move ?? "-"}</span>
            <span>{attempt.legal_ok ? "legal" : attempt.error_type ?? "invalid"}</span>
            <span>{attempt.latency_ms.toFixed(1)} ms</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function AnalysisPanel({
  game,
  currentMove,
}: {
  game: GameDetail | undefined;
  currentMove: Move | undefined;
}) {
  if (!game) {
    return (
      <section className="panel detail-panel">
        <div className="section-heading">
          <p className="eyebrow">Analysis</p>
          <h3>No game selected</h3>
        </div>
      </section>
    );
  }

  const evaluation = currentMove?.engine_evaluations[0];
  const hasEvaluation = game.moves.some((move) => move.engine_evaluations.length > 0);

  return (
    <div className="detail-stack">
      <section className="panel detail-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Stockfish</p>
            <h3>{currentMove ? `Ply ${currentMove.ply} · ${currentMove.accepted_san}` : "No move selected"}</h3>
          </div>
          {currentMove && (
            <span className={`badge ${evaluation?.classification ?? "pending"}`}>
              {evaluation?.classification ?? "no eval"}
            </span>
          )}
        </div>
        {currentMove ? (
          <dl>
            <dt>Best move</dt>
            <dd>{evaluation?.best_move_uci ?? "—"}</dd>
            <dt>Before</dt>
            <dd>{scoreLabel(evaluation?.eval_before_cp, evaluation?.mate_before)}</dd>
            <dt>After</dt>
            <dd>{scoreLabel(evaluation?.eval_after_cp, evaluation?.mate_after)}</dd>
            <dt>Nodes</dt>
            <dd>{evaluation?.nodes ?? "—"}</dd>
          </dl>
        ) : (
          <p className="muted">Select a move to inspect engine evaluation for that ply.</p>
        )}
        {!hasEvaluation && (
          <p className="muted compact">No engine evaluations are stored for this game yet.</p>
        )}
      </section>

      {currentMove && currentMove.annotations.length > 0 && (
        <section className="panel detail-panel">
          <h4>Commentary</h4>
          {currentMove.annotations.map((annotation) => (
            <p key={`${annotation.persona}-${annotation.created_at}`}>
              <strong>{annotation.persona}</strong>: {annotation.commentary}
            </p>
          ))}
        </section>
      )}

      <EvalGraph game={game} />
    </div>
  );
}

function BenchmarkPanel({
  game,
  leaderboardRows,
  comparisonRows,
  leaderboardRunId,
  leaderboardColor,
  leaderboardLegality,
  runOptions,
  onRunIdChange,
  onColorChange,
  onLegalityModeChange,
}: {
  game: GameDetail | undefined;
  leaderboardRows: LeaderboardRow[];
  comparisonRows: RunComparisonRow[];
  leaderboardRunId: number | "all";
  leaderboardColor: string;
  leaderboardLegality: string;
  runOptions: number[];
  onRunIdChange: (value: number | "all") => void;
  onColorChange: (value: string) => void;
  onLegalityModeChange: (value: string) => void;
}) {
  return (
    <div className="detail-stack">
      <Leaderboard
        rows={leaderboardRows}
        runId={leaderboardRunId}
        color={leaderboardColor}
        legalityMode={leaderboardLegality}
        runOptions={runOptions}
        onRunIdChange={onRunIdChange}
        onColorChange={onColorChange}
        onLegalityModeChange={onLegalityModeChange}
      />
      <OperationalEventsPanel game={game} />
      <RunComparison rows={comparisonRows} />
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MoveTable({
  moves,
  activePly,
  onSelectPly,
}: {
  moves: Move[];
  activePly: number;
  onSelectPly: (ply: number) => void;
}) {
  const pairs = useMemo(() => {
    const grouped: { number: number; white?: Move; black?: Move }[] = [];
    moves.forEach((move) => {
      const number = Math.ceil(move.ply / 2);
      const row = grouped[number - 1] ?? { number };
      if (move.color === "white") {
        row.white = move;
      } else {
        row.black = move;
      }
      grouped[number - 1] = row;
    });
    return grouped;
  }, [moves]);

  return (
    <div className="move-table">
      <div className="move-table-head">
        <span>#</span>
        <span>White</span>
        <span>Black</span>
      </div>
      {pairs.map((pair) => (
        <div key={pair.number} className="move-pair">
          <span className="move-number">{pair.number}</span>
          <MoveCell move={pair.white} activePly={activePly} onSelectPly={onSelectPly} />
          <MoveCell move={pair.black} activePly={activePly} onSelectPly={onSelectPly} />
        </div>
      ))}
    </div>
  );
}

function MoveCell({
  move,
  activePly,
  onSelectPly,
}: {
  move: Move | undefined;
  activePly: number;
  onSelectPly: (ply: number) => void;
}) {
  if (!move) {
    return <span className="move-cell empty">—</span>;
  }
  return (
    <button
      className={move.ply === activePly ? "move-cell active" : "move-cell"}
      type="button"
      onClick={() => onSelectPly(move.ply)}
    >
      <span>{move.accepted_san}</span>
      <small>
        {move.retries_used} retry · {move.engine_evaluations[0]?.centipawn_loss ?? "—"} CPL
      </small>
    </button>
  );
}

function scoreLabel(cp: number | null | undefined, mate: number | null | undefined) {
  if (mate !== null && mate !== undefined) {
    return `M${mate}`;
  }
  if (cp !== null && cp !== undefined) {
    return `${cp} cp`;
  }
  return "—";
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

function relativeSampleTime(sampledAt: string) {
  const elapsedMs = Date.now() - new Date(sampledAt).getTime();
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
    return "live";
  }
  const seconds = Math.round(elapsedMs / 1000);
  return seconds <= 1 ? "live" : `${seconds}s ago`;
}

function Leaderboard({
  rows,
  runId,
  color,
  legalityMode,
  runOptions,
  onRunIdChange,
  onColorChange,
  onLegalityModeChange,
}: {
  rows: LeaderboardRow[];
  runId: number | "all";
  color: string;
  legalityMode: string;
  runOptions: number[];
  onRunIdChange: (value: number | "all") => void;
  onColorChange: (value: string) => void;
  onLegalityModeChange: (value: string) => void;
}) {
  return (
    <section className="leaderboard">
      <div className="section-heading leaderboard-heading">
        <div>
          <p className="eyebrow">Materialized Summary</p>
          <h3>Leaderboard</h3>
        </div>
        <div className="filter-row">
          <select
            aria-label="Filter leaderboard by run"
            value={runId}
            onChange={(event) =>
              onRunIdChange(event.target.value === "all" ? "all" : Number(event.target.value))
            }
          >
            <option value="all">All runs</option>
            {runOptions.map((id) => (
              <option key={id} value={id}>
                Run {id}
              </option>
            ))}
          </select>
          <select
            aria-label="Filter leaderboard by color"
            value={color}
            onChange={(event) => onColorChange(event.target.value)}
          >
            <option value="all">All colors</option>
            <option value="white">White</option>
            <option value="black">Black</option>
          </select>
          <select
            aria-label="Filter leaderboard by legality mode"
            value={legalityMode}
            onChange={(event) => onLegalityModeChange(event.target.value)}
          >
            <option value="all">All modes</option>
            <option value="open">Open</option>
            <option value="constrained">Constrained</option>
          </select>
        </div>
      </div>
      {rows.length === 0 ? (
        <p className="muted">No summary rows yet. Run `arena rebuild-summaries` after a tournament.</p>
      ) : (
        <div className="leaderboard-table">
          <div className="leaderboard-head">
            <span>Run</span>
            <span>Participant</span>
            <span>Mode</span>
            <span>Color</span>
            <span>W-D-L-U</span>
            <span>Avg CPL</span>
            <span>Illegal</span>
            <span>Retries</span>
            <span>Tokens</span>
          </div>
          {rows.map((row) => (
            <div key={row.id} className="leaderboard-row">
              <span>{row.run_id}</span>
              <span>{row.participant}</span>
              <span>
                {row.mode}/{row.legality_mode}
              </span>
              <span>{row.color}</span>
              <span>
                {row.wins}-{row.draws}-{row.losses}-{row.unfinished}
              </span>
              <span>{row.avg_cpl === null ? "—" : row.avg_cpl.toFixed(1)}</span>
              <span>{(row.illegal_rate * 100).toFixed(1)}%</span>
              <span>{row.avg_retries.toFixed(2)}</span>
              <span>{row.total_tokens}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
