import { useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RefreshCcw,
  SkipBack,
  SkipForward,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
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
  playHumanMove,
  startGame,
  startHumanGame,
  startStockfishMatch,
  type GameDetail,
  type HumanGameState,
  type LeaderboardRow,
  type Move,
  type RunComparisonRow,
} from "./api";
import { applyUciMoveToFen, startFen } from "./chess";
import { ChessBoard } from "./ChessBoard";
import { movesLabel, rateWithCi, sampleRate, scorePercent } from "./format";
import { ModelComparison } from "./ModelComparison";
import { Metric, RuntimePanel } from "./RuntimePanel";
import { StartGamePanel, stockfishLiveLabel } from "./StartGamePanel";

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
  const [appView, setAppView] = useState<"arena" | "models">("arena");
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

  const resetToReplay = useCallback((ply = 0) => {
    setActiveLiveJobId(null);
    setActiveHumanGameId(null);
    setIsPlaying(false);
    setPlyIndex(ply);
  }, []);

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
    setActiveHumanGameId(null);
    if (maxPly === 0) {
      return;
    }
    if (!isPlaying && replayPlyIndex >= maxPly) {
      setPlyIndex(0);
    }
    setIsPlaying((value) => !value);
  };

  const viewToggle = (
    <nav className="view-toggle" role="tablist" aria-label="Application view">
      <span className="view-toggle-brand">Chess Arena</span>
      <button
        className={appView === "arena" ? "active" : ""}
        type="button"
        role="tab"
        aria-selected={appView === "arena"}
        onClick={() => setAppView("arena")}
      >
        Arena
      </button>
      <button
        className={appView === "models" ? "active" : ""}
        type="button"
        role="tab"
        aria-selected={appView === "models"}
        onClick={() => setAppView("models")}
      >
        Models
      </button>
    </nav>
  );

  if (appView === "models") {
    return (
      <div className="app-root">
        {viewToggle}
        <ModelComparison />
      </div>
    );
  }

  return (
    <div className="app-root">
      {viewToggle}
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
              resetToReplay();
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
            onStart={() => resetToReplay()}
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
            onEnd={() => resetToReplay(maxPly)}
            onChangePly={resetToReplay}
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
                resetToReplay(value);
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
            <span>Win CI</span>
            <span>Moves</span>
            <span>Accuracy</span>
            <span>Avg CPL</span>
            <span>Illegal CI</span>
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
              <span>{rateWithCi(row.win_rate, row.win_rate_ci_low, row.win_rate_ci_high)}</span>
              <span>{movesLabel(row.avg_game_plies)}</span>
              <span>{sampleRate(row.accuracy_rate, row.evaluated_move_count)}</span>
              <span>{row.avg_cpl === null ? "—" : row.avg_cpl.toFixed(1)}</span>
              <span>
                {rateWithCi(row.illegal_rate, row.illegal_rate_ci_low, row.illegal_rate_ci_high)}
              </span>
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
            <span>Win CI</span>
            <span>Moves</span>
            <span>Accuracy</span>
            <span>Avg CPL</span>
            <span>Illegal CI</span>
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
              <span>
                {scorePercent(row)}
                {row.low_sample ? " n<10" : ""}
              </span>
              <span>{rateWithCi(row.win_rate, row.win_rate_ci_low, row.win_rate_ci_high)}</span>
              <span>{movesLabel(row.avg_game_plies)}</span>
              <span>{sampleRate(row.accuracy_rate, row.evaluated_move_count)}</span>
              <span>{row.avg_cpl === null ? "—" : row.avg_cpl.toFixed(1)}</span>
              <span>
                {rateWithCi(row.illegal_rate, row.illegal_rate_ci_low, row.illegal_rate_ci_high)}
              </span>
              <span>{row.avg_retries.toFixed(2)}</span>
              <span>{row.total_tokens}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
