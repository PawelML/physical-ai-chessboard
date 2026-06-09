import { useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RefreshCcw,
  SkipBack,
  SkipForward,
  Square,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  cancelGameJob,
  cancelHumanGame,
  fetchGame,
  fetchGameJobs,
  fetchGames,
  fetchHumanGames,
  fetchLeaderboard,
  fetchModelOptions,
  fetchRunComparison,
  fetchRunEvents,
  fetchRuntimeTelemetry,
  fetchGameDefaults,
  playHumanMove,
  saveGameDefaults,
  startGame,
  startHumanGame,
  startStockfishMatch,
  type GameDefaults,
  type GameDetail,
  type GameJob,
  type GuidanceMode,
  type HumanGameState,
  type LeaderboardRow,
  type ModelOption,
  type Move,
  type RuntimeTelemetry,
  type RunComparisonRow,
  type StartGamePayload,
  type StartHumanGamePayload,
  type StartStockfishMatchPayload,
  type StockfishLevel,
} from "./api";
import { applyUciMoveToFen, moveSquares, parseFenBoard, startFen, type Piece } from "./chess";

export default function App() {
  const queryClient = useQueryClient();
  const gamesQuery = useQuery({ queryKey: ["games"], queryFn: fetchGames });
  const modelOptionsQuery = useQuery({ queryKey: ["models"], queryFn: fetchModelOptions });
  const gameJobsQuery = useQuery({
    queryKey: ["game-jobs"],
    queryFn: fetchGameJobs,
    refetchInterval: 2_000,
  });
  const humanGamesQuery = useQuery({
    queryKey: ["human-games"],
    queryFn: fetchHumanGames,
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
  const [activeHumanGameId, setActiveHumanGameId] = useState<string | null>(null);
  const [optimisticHumanMove, setOptimisticHumanMove] = useState<OptimisticHumanMove | null>(null);
  const activeLiveJob = gameJobsQuery.data?.find((job) => job.id === activeLiveJobId);
  const activeHumanGame = humanGamesQuery.data?.find((row) => row.id === activeHumanGameId);
  const activeJobGameId =
    activeLiveJob?.status !== "failed" && activeLiveJob?.status !== "cancelled"
      ? activeLiveJob?.game_id
      : null;
  const activeHumanGameDbId =
    activeHumanGame?.status !== "failed" && activeHumanGame?.status !== "cancelled"
      ? activeHumanGame?.game_id
      : null;
  const effectiveGameId =
    activeHumanGameDbId ?? activeJobGameId ?? selectedGameId ?? gamesQuery.data?.[0]?.id ?? null;

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
  const refetchHumanGames = humanGamesQuery.refetch;

  const selectGameWhenJobStarts = (jobId: string) => {
    const poll = async (attempt: number) => {
      const jobs = await fetchGameJobs();
      void refetchJobs();
      const job = jobs.find((row) => row.id === jobId);
      if (job?.run_id) {
        setLeaderboardRunId(job.run_id);
      }
      if (job?.game_id) {
        setSelectedGameId(job.game_id);
        setPlyIndex(0);
        void refetchGames();
        void refetchGame();
        void refetchLeaderboard();
        void refetchComparison();
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

  const startStockfishMatchMutation = useMutation({
    mutationFn: startStockfishMatch,
    onSuccess: (response) => {
      setActiveLiveJobId(response.job_id);
      setWorkspaceTab("benchmark");
      void refetchJobs();
      void refetchGames();
      selectGameWhenJobStarts(response.job_id);
    },
  });

  const startHumanGameMutation = useMutation({
    mutationFn: startHumanGame,
    onSuccess: (state) => {
      setActiveLiveJobId(null);
      setActiveHumanGameId(state.id);
      setOptimisticHumanMove(null);
      setSelectedGameId(state.game_id);
      setPlyIndex(0);
      upsertHumanGame(queryClient, state);
      void refetchHumanGames();
      void refetchGames();
      void refetchGame();
    },
  });

  const playHumanMoveMutation = useMutation({
    mutationFn: ({ humanGameId, move }: { humanGameId: string; move: string }) =>
      playHumanMove(humanGameId, move),
    onSuccess: (state) => {
      setOptimisticHumanMove(null);
      setActiveHumanGameId(state.id);
      setSelectedGameId(state.game_id);
      upsertHumanGame(queryClient, state);
      void refetchHumanGames();
      void refetchGames();
      void refetchGame();
    },
    onError: () => {
      setOptimisticHumanMove(null);
    },
  });

  const cancelHumanGameMutation = useMutation({
    mutationFn: cancelHumanGame,
    onSuccess: (state) => {
      setOptimisticHumanMove(null);
      if (state.id === activeHumanGameId) {
        setActiveHumanGameId(null);
      }
      upsertHumanGame(queryClient, state);
      void refetchHumanGames();
      void refetchGames();
      void refetchGame();
    },
  });

  const cancelGameMutation = useMutation({
    mutationFn: cancelGameJob,
    onSuccess: (response) => {
      if (response.job_id === activeLiveJobId) {
        setActiveLiveJobId(null);
      }
      setIsPlaying(false);
      void refetchJobs();
      void refetchGames();
      if (effectiveGameId !== null) {
        void refetchGame();
      }
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
      void refetchHumanGames();
    });
    source.onerror = () => source.close();
    return () => source.close();
  }, [
    effectiveGameId,
    refetchComparison,
    refetchGame,
    refetchGames,
    refetchJobs,
    refetchHumanGames,
    refetchLeaderboard,
  ]);

  const maxPly = game?.moves.length ?? 0;
  const followLiveGame =
    (activeLiveJob?.game_id === effectiveGameId && activeLiveJob.status !== "failed") ||
    (activeHumanGame?.game_id === effectiveGameId && activeHumanGame.status === "running");
  const replayPlyIndex = followLiveGame ? maxPly : Math.min(plyIndex, maxPly);
  const currentMove = replayPlyIndex > 0 ? game?.moves[replayPlyIndex - 1] : undefined;
  const replayFen = currentMove?.fen_after ?? game?.moves[0]?.fen_before ?? startFen;
  const liveHumanFen = activeHumanGame?.fen ?? replayFen;
  const activeOptimisticMove =
    optimisticHumanMove && optimisticHumanMove.humanGameId === activeHumanGame?.id
      ? optimisticHumanMove
      : null;
  const fen =
    followLiveGame && activeHumanGame
      ? (activeOptimisticMove?.fen ?? liveHumanFen)
      : replayFen;
  const whitePlayer = game?.white_player ?? "White";
  const blackPlayer = game?.black_player ?? "Black";

  const submitHumanMove = (move: string) => {
    if (!activeHumanGame) {
      return;
    }
    setOptimisticHumanMove({
      humanGameId: activeHumanGame.id,
      move,
      fen: applyUciMoveToFen(liveHumanFen, move),
    });
    playHumanMoveMutation.mutate({ humanGameId: activeHumanGame.id, move });
  };

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
            submitting={
              startGameMutation.isPending ||
              startStockfishMatchMutation.isPending ||
              startHumanGameMutation.isPending
            }
            error={
              startGameMutation.error ??
              startStockfishMatchMutation.error ??
              startHumanGameMutation.error
            }
            onSubmit={(payload) => startGameMutation.mutate(payload)}
            onSubmitStockfishMatch={(payload) => startStockfishMatchMutation.mutate(payload)}
            onSubmitHumanGame={(payload) => startHumanGameMutation.mutate(payload)}
            cancellingJobId={cancelGameMutation.isPending ? cancelGameMutation.variables : null}
            onCancel={(jobId) => cancelGameMutation.mutate(jobId)}
          />
        ) : (
          <GameList
            games={gamesQuery.data ?? []}
            selectedGameId={effectiveGameId}
            onSelect={(id) => {
              setSelectedGameId(id);
              setActiveLiveJobId(null);
              setActiveHumanGameId(null);
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
              setActiveHumanGameId(null);
              setIsPlaying(false);
              setPlyIndex(0);
            }}
            onPrev={() => {
              setActiveLiveJobId(null);
              setActiveHumanGameId(null);
              setPlyIndex((value) => Math.max(0, value - 1));
            }}
            onTogglePlay={togglePlayback}
            onNext={() => {
              setActiveLiveJobId(null);
              setActiveHumanGameId(null);
              setPlyIndex((value) => Math.min(maxPly, value + 1));
            }}
            onEnd={() => {
              setActiveLiveJobId(null);
              setActiveHumanGameId(null);
              setIsPlaying(false);
              setPlyIndex(maxPly);
            }}
            onChangePly={(value) => {
              setActiveLiveJobId(null);
              setActiveHumanGameId(null);
              setIsPlaying(false);
              setPlyIndex(value);
            }}
            onSpeedChange={setPlaybackDelayMs}
          />
        </header>
        <div className="live-strip">
          <span className="live-dot" />
          <span>Live stream connected</span>
          {activeLiveJob?.kind === "stockfish_match" && activeLiveJob.status === "running" && (
            <span className="live-match-status">{stockfishLiveLabel(activeLiveJob)}</span>
          )}
          <strong>{liveEvents}</strong>
        </div>
        {activeHumanGame && (
          <HumanMovePanel
            state={activeHumanGame}
            submitting={playHumanMoveMutation.isPending}
            waitingForOpponent={Boolean(activeOptimisticMove)}
            cancelling={cancelHumanGameMutation.isPending}
            error={playHumanMoveMutation.error}
            onCancel={() => cancelHumanGameMutation.mutate(activeHumanGame.id)}
          />
        )}

        <div className="replay-grid">
          <ChessBoard
            fen={fen}
            activeMove={currentMove}
            plyIndex={replayPlyIndex}
            whitePlayer={whitePlayer}
            blackPlayer={blackPlayer}
            humanGame={activeHumanGame}
            submittingHumanMove={playHumanMoveMutation.isPending}
            waitingForOpponent={Boolean(activeOptimisticMove)}
            onSubmitHumanMove={submitHumanMove}
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
            <span>Moves</span>
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
              <span>{movesLabel(row.avg_game_plies)}</span>
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
  onSubmitStockfishMatch,
  onSubmitHumanGame,
  cancellingJobId,
  onCancel,
}: {
  modelOptions: ModelOption[];
  jobs: GameJob[];
  loadingModels: boolean;
  submitting: boolean;
  error: Error | null;
  onSubmit: (payload: StartGamePayload) => void;
  onSubmitStockfishMatch: (payload: StartStockfishMatchPayload) => void;
  onSubmitHumanGame: (payload: StartHumanGamePayload) => void;
  cancellingJobId: string | null | undefined;
  onCancel: (jobId: string) => void;
}) {
  const ollamaOptions = modelOptions.filter((option) => option.provider === "ollama");
  const matchModelOptions = modelOptions.filter(
    (option) => option.provider !== "engine" && option.id !== "random",
  );
  const stockfishOption = modelOptions.find((option) => option.id === "stockfish");
  const defaultWhite = ollamaOptions[0]?.id ?? modelOptions[0]?.id ?? "random";
  const defaultBlack = ollamaOptions[1]?.id ?? ollamaOptions[0]?.id ?? modelOptions[1]?.id ?? defaultWhite;
  const defaultMatchModel = matchModelOptions[0]?.id ?? defaultWhite;
  const [startMode, setStartMode] = useState<"game" | "stockfish" | "human">("game");
  const [white, setWhite] = useState<string | null>(null);
  const [black, setBlack] = useState<string | null>(null);
  const [matchModel, setMatchModel] = useState<string | null>(null);
  const [humanOpponent, setHumanOpponent] = useState<string | null>(null);
  const [humanColor, setHumanColor] = useState<"white" | "black">("white");
  const [stockfishLevel, setStockfishLevel] = useState<StockfishLevel>("beginner");
  const [matchGameCount, setMatchGameCount] = useState("4");
  const [legalityMode, setLegalityMode] = useState<"open" | "constrained">("constrained");
  // null = untouched -> fall back to the server-saved default for display.
  const [temperature, setTemperature] = useState<string | null>(null);
  const [topP, setTopP] = useState<string | null>(null);
  const [numCtx, setNumCtx] = useState<string | null>(null);
  const [numPredict, setNumPredict] = useState<string | null>(null);
  const [savedNotice, setSavedNotice] = useState(false);
  const [ollamaThinking, setOllamaThinking] = useState(false);
  const [ollamaCpuOffload, setOllamaCpuOffload] = useState(false);
  const [guidanceMode, setGuidanceMode] = useState<GuidanceMode>("legal_list");
  const [maxPlies, setMaxPlies] = useState("");
  const selectedWhite = white ?? defaultWhite;
  const selectedBlack = black ?? defaultBlack;
  const selectedMatchModel = matchModel ?? defaultMatchModel;
  const selectedHumanOpponent =
    humanOpponent ?? stockfishOption?.id ?? ollamaOptions[0]?.id ?? modelOptions[0]?.id ?? "random";

  const defaultsQuery = useQuery({ queryKey: ["game-defaults"], queryFn: fetchGameDefaults });
  const savedDefaults = defaultsQuery.data;
  const numText = (value: number | null | undefined): string =>
    value === null || value === undefined ? "" : String(value);
  // Edited value wins; otherwise show the server-saved default.
  const temperatureValue = temperature ?? (savedDefaults ? String(savedDefaults.temperature) : "0");
  const topPValue = topP ?? numText(savedDefaults?.top_p);
  const numCtxValue = numCtx ?? numText(savedDefaults?.num_ctx);
  const numPredictValue = numPredict ?? numText(savedDefaults?.num_predict);

  const numOrNull = (value: string): number | null => {
    const trimmed = value.trim();
    if (trimmed === "") {
      return null;
    }
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
  };
  const samplingPayload = (): GameDefaults => ({
    temperature: numOrNull(temperatureValue) ?? 0,
    top_p: numOrNull(topPValue),
    num_ctx: numOrNull(numCtxValue),
    num_predict: numOrNull(numPredictValue),
  });

  const saveDefaultsMutation = useMutation({
    mutationFn: (payload: GameDefaults) => saveGameDefaults(payload),
    onSuccess: () => {
      setSavedNotice(true);
      window.setTimeout(() => setSavedNotice(false), 2000);
    },
  });

  const recentJobs = jobs.slice(0, 4);

  return (
    <section className="start-game">
      <div className="section-heading">
        <p className="eyebrow">Match Control</p>
        <h3>Start Game</h3>
      </div>
      <form
        className="start-game-form"
        noValidate
        onSubmit={(event) => {
          event.preventDefault();
          const sharedPayload = {
            legality_mode: legalityMode,
            ...samplingPayload(),
            ollama_thinking: ollamaThinking,
            ollama_cpu_offload: ollamaCpuOffload,
            guidance_mode: guidanceMode,
            max_plies: maxPlies ? Number(maxPlies) : null,
          };
          if (startMode === "stockfish") {
            onSubmitStockfishMatch({
              model: selectedMatchModel,
              stockfish_level: stockfishLevel,
              game_count: Math.max(1, Number(matchGameCount) || 1),
              ...sharedPayload,
            });
            return;
          }
          if (startMode === "human") {
            onSubmitHumanGame({
              human_color: humanColor,
              opponent: selectedHumanOpponent,
              stockfish_level: selectedHumanOpponent === "stockfish" ? stockfishLevel : null,
              ...sharedPayload,
            });
            return;
          }
          onSubmit({
            white: selectedWhite,
            black: selectedBlack,
            ...sharedPayload,
          });
        }}
      >
        <div className="mode-toggle" role="tablist" aria-label="Start mode">
          <button
            className={startMode === "game" ? "active" : ""}
            type="button"
            role="tab"
            aria-selected={startMode === "game"}
            disabled={submitting}
            onClick={() => setStartMode("game")}
          >
            Single game
          </button>
          <button
            className={startMode === "stockfish" ? "active" : ""}
            type="button"
            role="tab"
            aria-selected={startMode === "stockfish"}
            disabled={submitting}
            onClick={() => setStartMode("stockfish")}
          >
            Stockfish eval
          </button>
          <button
            className={startMode === "human" ? "active" : ""}
            type="button"
            role="tab"
            aria-selected={startMode === "human"}
            disabled={submitting}
            onClick={() => setStartMode("human")}
          >
            Human game
          </button>
        </div>

        {startMode === "game" ? (
          <fieldset className="start-game-group">
            <legend>Player Models</legend>
            <label>
              <span>White model</span>
              <ModelInput
                value={selectedWhite}
                options={modelOptions}
                disabled={submitting}
                onChange={setWhite}
              />
            </label>
            <label>
              <span>Black model</span>
              <ModelInput
                value={selectedBlack}
                options={modelOptions}
                disabled={submitting}
                onChange={setBlack}
              />
            </label>
          </fieldset>
        ) : startMode === "stockfish" ? (
          <fieldset className="start-game-group">
            <legend>Stockfish Evaluation</legend>
            <label>
              <span>Model under test</span>
              <ModelInput
                value={selectedMatchModel}
                options={matchModelOptions.length ? matchModelOptions : modelOptions}
                disabled={submitting}
                onChange={setMatchModel}
              />
            </label>
            <label>
              <span>Stockfish level</span>
              <select
                value={stockfishLevel}
                disabled={submitting}
                onChange={(event) => setStockfishLevel(event.target.value as StockfishLevel)}
              >
                <option value="beginner">Beginner · 1320 Elo</option>
                <option value="club">Club · 1600 Elo</option>
              </select>
            </label>
            <label>
              <span>Games in match</span>
              <input
                type="number"
                min="1"
                max="200"
                value={matchGameCount}
                disabled={submitting}
                onChange={(event) => setMatchGameCount(event.target.value)}
              />
            </label>
          </fieldset>
        ) : (
          <fieldset className="start-game-group">
            <legend>Human Player</legend>
            <label>
              <span>Your color</span>
              <select
                value={humanColor}
                disabled={submitting}
                onChange={(event) => setHumanColor(event.target.value as "white" | "black")}
              >
                <option value="white">White</option>
                <option value="black">Black</option>
              </select>
            </label>
            <label>
              <span>Opponent</span>
              <ModelInput
                value={selectedHumanOpponent}
                options={modelOptions}
                disabled={submitting}
                onChange={setHumanOpponent}
              />
            </label>
            {stockfishOption && selectedHumanOpponent === "stockfish" && (
              <label>
                <span>Stockfish level</span>
                <select
                  value={stockfishLevel}
                  disabled={submitting}
                  onChange={(event) => setStockfishLevel(event.target.value as StockfishLevel)}
                >
                  <option value="beginner">Beginner · 1320 Elo</option>
                  <option value="club">Club · 1600 Elo</option>
                </select>
              </label>
            )}
          </fieldset>
        )}

        <fieldset className="start-game-group shared">
          <legend>Shared Game Settings</legend>
          <label>
            <span>Prompt help for both</span>
            <select
              value={guidanceMode}
              disabled={submitting}
              onChange={(event) => setGuidanceMode(event.target.value as GuidanceMode)}
            >
              <option value="legal_list">Show legal moves</option>
              <option value="strategic_memory">Strategic memory</option>
            </select>
          </label>
          <details className="sampling-panel">
            <summary>Sampling &amp; runtime parameters</summary>
            <p className="muted compact">
              Applied to both Ollama models. Leave a field blank to use the model&apos;s
              own default. Thinking / mixed-offload below may still raise num_ctx and
              num_predict when enabled.
            </p>
            <label>
              <span>temperature</span>
              <input
                type="number"
                min="0"
                max="2"
                step="0.1"
                value={temperatureValue}
                disabled={submitting}
                onChange={(event) => setTemperature(event.target.value)}
              />
              <small className="muted">
                Randomness of move choice. 0 = deterministic (same game every run);
                0.3–0.7 = more varied, livelier play; higher = riskier.
              </small>
            </label>
            <label>
              <span>top_p</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={topPValue}
                placeholder="model default"
                disabled={submitting}
                onChange={(event) => setTopP(event.target.value)}
              />
              <small className="muted">
                Nucleus sampling: keep only the most probable tokens summing to this
                mass. Only matters when temperature &gt; 0. Typical: 0.9.
              </small>
            </label>
            <label>
              <span>num_ctx</span>
              <input
                type="number"
                min="1"
                step="1"
                value={numCtxValue}
                placeholder="model default"
                disabled={submitting}
                onChange={(event) => setNumCtx(event.target.value)}
              />
              <small className="muted">
                Context window in tokens. Bigger = more board history fits, but slower
                and more VRAM. Long games may need 16k–32k.
              </small>
            </label>
            <label>
              <span>num_predict</span>
              <input
                type="number"
                min="1"
                step="1"
                value={numPredictValue}
                placeholder="model default"
                disabled={submitting}
                onChange={(event) => setNumPredict(event.target.value)}
              />
              <small className="muted">
                Max tokens generated per move. The move JSON is tiny, but thinking
                models need room (≥512). Too low can truncate the answer.
              </small>
            </label>
            <button
              type="button"
              className="secondary-button"
              disabled={submitting || saveDefaultsMutation.isPending}
              onClick={() => saveDefaultsMutation.mutate(samplingPayload())}
            >
              {saveDefaultsMutation.isPending ? "Saving" : "Save as default"}
            </button>
            {savedNotice && <span className="muted compact">Saved as default.</span>}
            {saveDefaultsMutation.error && (
              <span className="error-text">{saveDefaultsMutation.error.message}</span>
            )}
          </details>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={ollamaThinking}
              disabled={submitting}
              onChange={(event) => setOllamaThinking(event.target.checked)}
            />
            <span>Thinking for supported models</span>
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={ollamaCpuOffload}
              disabled={submitting}
              onChange={(event) => setOllamaCpuOffload(event.target.checked)}
            />
            <span>Mixed GPU/CPU offload for large models</span>
          </label>
          <label>
            <span>Move validation for both</span>
            <select
              value={legalityMode}
              disabled={submitting}
              onChange={(event) => setLegalityMode(event.target.value as "open" | "constrained")}
            >
              <option value="constrained">Reject illegal moves</option>
              <option value="open">Record raw model moves</option>
            </select>
          </label>
          <label>
            <span>Max plies for game</span>
            <input
              type="number"
              min="1"
              value={maxPlies}
              placeholder="No limit"
              disabled={submitting}
              onChange={(event) => setMaxPlies(event.target.value)}
            />
          </label>
        </fieldset>
        <button
          className="primary-button"
          type="submit"
          disabled={
            submitting ||
            (startMode === "game"
              ? !selectedWhite || !selectedBlack
              : startMode === "stockfish"
                ? !selectedMatchModel
                : loadingModels || !selectedHumanOpponent)
          }
        >
          {submitting
            ? "Starting"
            : startMode === "stockfish"
              ? "Start evaluation"
              : startMode === "human"
                ? "Start human game"
                : "Start game"}
        </button>
      </form>
      {loadingModels && <p className="muted compact">Loading local Ollama models.</p>}
      {error && <p className="error-text">{error.message}</p>}
      {recentJobs.length > 0 && (
        <div className="job-list">
          {recentJobs.map((job) => (
            <div key={job.id} className={`job-row ${job.status}`}>
              <div>
                <span>
                  {job.white} vs {job.black}
                </span>
                <strong>
                  {job.kind === "stockfish_match" ? stockfishJobOptions(job) : job.guidance_mode} ·{" "}
                  {ollamaJobOptions(job)} · {jobLabel(job)}
                </strong>
              </div>
              {job.status === "running" && (
                <button
                  type="button"
                  className="danger-button"
                  disabled={cancellingJobId === job.id}
                  onClick={() => onCancel(job.id)}
                >
                  <Square size={14} />
                  <span>{cancellingJobId === job.id ? "Ending" : "End"}</span>
                </button>
              )}
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
  const hasCurrentValue = options.some((option) => option.id === value);
  return (
    <select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
      {!hasCurrentValue && value && (
        <option value={value}>
          {value} · {value === "random" ? "built-in" : "configured"}
        </option>
      )}
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
    if (job.kind === "stockfish_match" && job.games_requested) {
      return stockfishLiveLabel(job);
    }
    return "running";
  }
  if (job.status === "failed") {
    return job.error ?? "failed";
  }
  if (job.kind === "stockfish_match") {
    return job.run_id ? `Run ${job.run_id}` : "completed";
  }
  return job.game_id ? `Game ${job.game_id}` : "completed";
}

function stockfishLiveLabel(job: GameJob) {
  const requested = job.games_requested ?? 1;
  const current = Math.min(job.games_completed + 1, requested);
  return `Game ${current}/${requested}`;
}

function stockfishJobOptions(job: GameJob) {
  const level = job.stockfish_level ?? "stockfish";
  const games = job.games_requested ? `${job.games_requested} games` : "match";
  return `${level} · ${games}`;
}

function ollamaJobOptions(job: GameJob) {
  const options: string[] = [`temp ${job.temperature}`];
  if (job.ollama_thinking) {
    options.push("thinking");
  }
  if (job.ollama_cpu_offload) {
    options.push("mixed offload");
  }
  return options.join(" + ");
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

function HumanMovePanel({
  state,
  submitting,
  waitingForOpponent,
  cancelling,
  error,
  onCancel,
}: {
  state: HumanGameState;
  submitting: boolean;
  waitingForOpponent: boolean;
  cancelling: boolean;
  error: Error | null;
  onCancel: () => void;
}) {
  const isHumanTurn = state.status === "running" && state.turn === state.human_color;

  return (
    <section className="human-move-panel">
      <div>
        <p className="eyebrow">Human Game</p>
        <h3>
          You play {state.human_color} vs {state.opponent}
        </h3>
        <p className="muted compact">
          {state.status === "running"
            ? waitingForOpponent
              ? "Opponent thinking"
              : isHumanTurn
                ? submitting
                  ? "Playing"
                  : "Your move"
                : `${state.turn ?? "Opponent"} to move`
            : `${state.result ?? "*"} · ${state.termination_reason ?? state.status}`}
        </p>
      </div>
      <div className="human-move-actions">
        <button
          className="secondary-button"
          type="button"
          disabled={cancelling || state.status !== "running"}
          onClick={onCancel}
        >
          {cancelling ? "Cancelling" : "Cancel"}
        </button>
      </div>
      {(state.error || error) && <p className="error-text">{state.error ?? error?.message}</p>}
    </section>
  );
}

function ChessBoard({
  fen,
  activeMove,
  plyIndex,
  whitePlayer,
  blackPlayer,
  humanGame,
  submittingHumanMove = false,
  waitingForOpponent = false,
  onSubmitHumanMove,
}: {
  fen: string;
  activeMove: Move | undefined;
  plyIndex: number;
  whitePlayer: string;
  blackPlayer: string;
  humanGame?: HumanGameState | null;
  submittingHumanMove?: boolean;
  waitingForOpponent?: boolean;
  onSubmitHumanMove?: (move: string) => void;
}) {
  const squares = useMemo(() => parseFenBoard(fen), [fen]);
  const orientation = humanGame?.human_color ?? "white";
  const displayedSquares = useMemo(
    () => (orientation === "white" ? squares : [...squares].reverse()),
    [orientation, squares],
  );
  const highlighted = useMemo(() => moveSquares(activeMove?.accepted_uci), [activeMove]);
  const [selectedSquare, setSelectedSquare] = useState<string | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [dragVisible, setDragVisible] = useState(false);
  const [promotion, setPromotion] = useState<PromotionPrompt | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const dragMovedRef = useRef(false);
  const suppressClickRef = useRef(false);
  const pointerSelectedRef = useRef<string | null>(null);
  const legalMoves = useMemo(() => humanGame?.legal_moves ?? [], [humanGame?.legal_moves]);
  const humanColor = humanGame?.human_color ?? "white";
  const isHumanTurn = humanGame?.status === "running" && humanGame.turn === humanGame.human_color;
  const canInteract =
    Boolean(humanGame && onSubmitHumanMove) && isHumanTurn && !submittingHumanMove;
  const legalMovesBySource = useMemo(() => {
    const grouped = new Map<string, string[]>();
    for (const legalMove of legalMoves) {
      const source = legalMove.slice(0, 2);
      grouped.set(source, [...(grouped.get(source) ?? []), legalMove]);
    }
    return grouped;
  }, [legalMoves]);
  const legalSources = useMemo(() => new Set(legalMovesBySource.keys()), [legalMovesBySource]);
  const effectiveSelectedSquare =
    selectedSquare && legalSources.has(selectedSquare) ? selectedSquare : null;
  const selectedMoves = useMemo(
    () => (effectiveSelectedSquare ? (legalMovesBySource.get(effectiveSelectedSquare) ?? []) : []),
    [effectiveSelectedSquare, legalMovesBySource],
  );
  const legalTargets = useMemo(
    () => new Set(selectedMoves.map((legalMove) => legalMove.slice(2, 4))),
    [selectedMoves],
  );
  const activeStripColor =
    waitingForOpponent && humanGame
      ? humanGame.human_color === "white"
        ? "black"
        : "white"
      : humanGame?.status === "running"
      ? humanGame.turn
      : plyIndex === 0
        ? "white"
        : activeMove?.color;

  const submitMove = useCallback(
    (move: string) => {
      if (!onSubmitHumanMove) {
        return;
      }
      setSelectedSquare(null);
      setPromotion(null);
      setDrag(null);
      setDragVisible(false);
      dragRef.current = null;
      onSubmitHumanMove(move);
    },
    [onSubmitHumanMove],
  );

  const attemptMove = useCallback(
    (from: string, to: string) => {
      if (!canInteract) {
        return;
      }
      const candidates = legalMoves.filter((legalMove) => legalMove.startsWith(`${from}${to}`));
      if (candidates.length === 0) {
        if (legalSources.has(to)) {
          setSelectedSquare(to);
        }
        return;
      }
      if (candidates.length === 1) {
        submitMove(candidates[0]);
        return;
      }
      setPromotion({ from, to, moves: candidates, color: humanColor });
    },
    [canInteract, humanColor, legalMoves, legalSources, submitMove],
  );

  const draggingFrom = drag?.from;
  useEffect(() => {
    if (!draggingFrom) {
      return;
    }

    const handlePointerMove = (event: PointerEvent) => {
      const current = dragRef.current;
      if (!current) {
        return;
      }
      if (Math.hypot(event.clientX - current.startX, event.clientY - current.startY) > 5) {
        dragMovedRef.current = true;
        setDragVisible(true);
      }
      setDrag((value) => {
        if (!value) {
          return value;
        }
        const next = { ...value, x: event.clientX, y: event.clientY };
        dragRef.current = next;
        return next;
      });
    };

    const handlePointerUp = (event: PointerEvent) => {
      const current = dragRef.current;
      if (current && dragMovedRef.current) {
        const targetSquare = squareFromPoint(event.clientX, event.clientY);
        if (targetSquare) {
          attemptMove(current.from, targetSquare);
        }
        suppressClickRef.current = true;
        window.setTimeout(() => {
          suppressClickRef.current = false;
        }, 0);
      }
      dragRef.current = null;
      setDrag(null);
      setDragVisible(false);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [attemptMove, draggingFrom]);

  const handleSquareClick = (square: string, piece: Piece | null) => {
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      pointerSelectedRef.current = null;
      return;
    }
    if (!canInteract || promotion) {
      return;
    }
    if (pointerSelectedRef.current === square) {
      pointerSelectedRef.current = null;
      return;
    }
    const isLegalSource = Boolean(piece && piece.color === humanColor && legalSources.has(square));
    if (!effectiveSelectedSquare) {
      if (isLegalSource) {
        setSelectedSquare(square);
      }
      return;
    }
    if (effectiveSelectedSquare === square) {
      setSelectedSquare(null);
      return;
    }
    if (legalTargets.has(square)) {
      attemptMove(effectiveSelectedSquare, square);
      return;
    }
    setSelectedSquare(isLegalSource ? square : null);
  };

  return (
    <div className="board-wrap">
      <PlayerStrip
        color={orientation === "white" ? "black" : "white"}
        name={orientation === "white" ? blackPlayer : whitePlayer}
        active={activeStripColor === (orientation === "white" ? "black" : "white")}
      />
      <div className={canInteract ? "board interactive" : "board"} aria-label="Chess board">
        {displayedSquares.map((square, index) => {
          const file = index % 8;
          const rank = Math.floor(index / 8);
          const dark = (file + rank) % 2 === 1;
          const isLegalSource = Boolean(
            canInteract &&
              square.piece &&
              square.piece.color === humanColor &&
              legalSources.has(square.square),
          );
          const isLegalTarget = legalTargets.has(square.square);
          const isDraggingSource = drag?.from === square.square;
          return (
            <button
              key={square.square}
              type="button"
              data-square={square.square}
              className={[
                "square",
                dark ? "dark" : "light",
                highlighted.has(square.square) ? "highlight" : "",
                effectiveSelectedSquare === square.square ? "selected" : "",
                isLegalSource ? "legal-source" : "",
                isLegalTarget ? "legal-target" : "",
                isLegalTarget && square.piece ? "legal-capture" : "",
                isDraggingSource ? "dragging-source" : "",
              ].join(" ")}
              onClick={() => handleSquareClick(square.square, square.piece)}
              onPointerDown={(event) => {
                if (!isLegalSource || !square.piece || event.button !== 0) {
                  return;
                }
                event.preventDefault();
                const next: DragState = {
                  from: square.square,
                  piece: square.piece,
                  x: event.clientX,
                  y: event.clientY,
                  startX: event.clientX,
                  startY: event.clientY,
                };
                dragMovedRef.current = false;
                setDragVisible(false);
                dragRef.current = next;
                setDrag(next);
                setPromotion(null);
                if (effectiveSelectedSquare !== square.square) {
                  pointerSelectedRef.current = square.square;
                  setSelectedSquare(square.square);
                } else {
                  pointerSelectedRef.current = null;
                }
              }}
              aria-label={`${square.square}${square.piece ? ` ${square.piece.color}` : ""}`}
            >
              <span className={`piece ${square.piece?.color ?? ""}`} aria-hidden="true">
                {square.piece?.symbol}
              </span>
              <span className="coord">{square.square}</span>
            </button>
          );
        })}
        {promotion && (
          <div className="promotion-menu" role="dialog" aria-label="Choose promotion piece">
            {promotion.moves.map((legalMove) => (
              <button
                key={legalMove}
                type="button"
                onClick={() => submitMove(legalMove)}
                aria-label={`Promote to ${promotionName(legalMove)}`}
                title={`Promote to ${promotionName(legalMove)}`}
              >
                {promotionSymbol(legalMove, promotion.color)}
              </button>
            ))}
          </div>
        )}
      </div>
      {drag && dragVisible && (
        <div
          className={`piece drag-piece ${drag.piece.color}`}
          style={{ left: drag.x, top: drag.y }}
          aria-hidden="true"
          data-from={drag.from}
        >
          {drag.piece.symbol}
        </div>
      )}
      <PlayerStrip
        color={orientation === "white" ? "white" : "black"}
        name={orientation === "white" ? whitePlayer : blackPlayer}
        active={activeStripColor === (orientation === "white" ? "white" : "black")}
      />
    </div>
  );
}

type DragState = {
  from: string;
  piece: Piece;
  x: number;
  y: number;
  startX: number;
  startY: number;
};

type PromotionPrompt = {
  from: string;
  to: string;
  moves: string[];
  color: "white" | "black";
};

function squareFromPoint(x: number, y: number): string | null {
  const element = document.elementFromPoint(x, y);
  const square = element?.closest<HTMLElement>("[data-square]");
  return square?.dataset.square ?? null;
}

function promotionSymbol(move: string, color: "white" | "black") {
  const piece = move.at(-1) ?? "q";
  const symbols: Record<"white" | "black", Record<string, string>> = {
    white: { q: "♕", r: "♖", b: "♗", n: "♘" },
    black: { q: "♛", r: "♜", b: "♝", n: "♞" },
  };
  return symbols[color][piece] ?? symbols[color].q;
}

function promotionName(move: string) {
  const names: Record<string, string> = {
    q: "queen",
    r: "rook",
    b: "bishop",
    n: "knight",
  };
  return names[move.at(-1) ?? "q"] ?? "queen";
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

type OptimisticHumanMove = {
  humanGameId: string;
  move: string;
  fen: string;
};

function upsertHumanGame(queryClient: QueryClient, state: HumanGameState) {
  queryClient.setQueryData<HumanGameState[]>(["human-games"], (current) => {
    const rows = current ?? [];
    const index = rows.findIndex((row) => row.id === state.id);
    if (index === -1) {
      return [state, ...rows];
    }
    return rows.map((row) => (row.id === state.id ? state : row));
  });
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
            <span>Score</span>
            <span>Moves</span>
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
              <span>{scorePercent(row)}</span>
              <span>{movesLabel(row.avg_game_plies)}</span>
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

function scorePercent(row: { games_played: number; wins: number; draws: number }) {
  if (row.games_played === 0) {
    return "—";
  }
  return `${(((row.wins + row.draws * 0.5) / row.games_played) * 100).toFixed(1)}%`;
}

function movesLabel(avgGamePlies: number) {
  return (avgGamePlies / 2).toFixed(1);
}
