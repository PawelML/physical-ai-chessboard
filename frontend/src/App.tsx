import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, RefreshCcw } from "lucide-react";
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
  fetchGames,
  fetchLeaderboard,
  fetchRunComparison,
  fetchRunEvents,
  type GameDetail,
  type LeaderboardRow,
  type Move,
  type RunComparisonRow,
} from "./api";
import { moveSquares, parseFenBoard, startFen } from "./chess";

export default function App() {
  const gamesQuery = useQuery({ queryKey: ["games"], queryFn: fetchGames });
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
  const [liveEvents, setLiveEvents] = useState(0);
  const [selectedGameId, setSelectedGameId] = useState<number | null>(null);
  const effectiveGameId = selectedGameId ?? gamesQuery.data?.[0]?.id ?? null;

  const gameQuery = useQuery({
    queryKey: ["game", effectiveGameId],
    queryFn: () => fetchGame(effectiveGameId!),
    enabled: effectiveGameId !== null,
  });
  const [plyIndex, setPlyIndex] = useState(0);
  const game = gameQuery.data;
  const refetchGames = gamesQuery.refetch;
  const refetchGame = gameQuery.refetch;
  const refetchLeaderboard = leaderboardQuery.refetch;
  const refetchComparison = comparisonQuery.refetch;

  useEffect(() => {
    const source = new EventSource("/api/stream/games?interval_seconds=2");
    source.addEventListener("games", () => {
      setLiveEvents((value) => value + 1);
      void refetchGames();
      void refetchGame();
      void refetchLeaderboard();
      void refetchComparison();
    });
    source.onerror = () => source.close();
    return () => source.close();
  }, [refetchComparison, refetchGame, refetchGames, refetchLeaderboard]);

  const maxPly = game?.moves.length ?? 0;
  const currentMove = plyIndex > 0 ? game?.moves[plyIndex - 1] : undefined;
  const fen = currentMove?.fen_after ?? game?.moves[0]?.fen_before ?? startFen;

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
              void gameQuery.refetch();
            }}
            aria-label="Refresh games"
            title="Refresh games"
          >
            <RefreshCcw size={18} />
          </button>
        </div>

        <GameList
          games={gamesQuery.data ?? []}
          selectedGameId={effectiveGameId}
          onSelect={(id) => {
            setSelectedGameId(id);
            setPlyIndex(0);
          }}
          loading={gamesQuery.isLoading}
        />
      </aside>

      <section className="workspace">
        <header className="game-header">
          <div>
            <p className="eyebrow">Game {game?.id ?? "-"}</p>
            <h2>{game ? `${game.result} · ${game.termination_reason ?? "running"}` : "No game selected"}</h2>
          </div>
          <PlyControls
            plyIndex={plyIndex}
            maxPly={maxPly}
            onPrev={() => setPlyIndex((value) => Math.max(0, value - 1))}
            onNext={() => setPlyIndex((value) => Math.min(maxPly, value + 1))}
          />
        </header>
        <div className="live-strip">
          <span className="live-dot" />
          <span>Live stream connected</span>
          <strong>{liveEvents}</strong>
        </div>

        <div className="replay-grid">
          <ChessBoard fen={fen} activeMove={currentMove} />
          <MovePanel game={game} plyIndex={plyIndex} currentMove={currentMove} />
        </div>
        <EvalGraph game={game} />
        <Leaderboard
          rows={leaderboardQuery.data ?? []}
          runId={leaderboardRunId}
          color={leaderboardColor}
          legalityMode={leaderboardLegality}
          runOptions={uniqueRunIds(gamesQuery.data ?? [])}
          onRunIdChange={setLeaderboardRunId}
          onColorChange={setLeaderboardColor}
          onLegalityModeChange={setLeaderboardLegality}
        />
        <OperationalEventsPanel game={game} />
        <RunComparison rows={comparisonQuery.data ?? []} />
      </section>
    </main>
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
    return game.moves.map((move) => {
      const evaluation = move.engine_evaluations[0];
      return {
        ply: move.ply,
        after: evaluation?.eval_after_cp ?? null,
        cpl: evaluation?.centipawn_loss ?? null,
        label: move.accepted_san,
      };
    });
  }, [game]);

  return (
    <section className="eval-graph">
      <div className="section-heading">
        <p className="eyebrow">Engine Trace</p>
        <h3>Evaluation Graph</h3>
      </div>
      {data.length === 0 ? (
        <p className="muted">No moves to graph yet.</p>
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
  onPrev,
  onNext,
}: {
  plyIndex: number;
  maxPly: number;
  onPrev: () => void;
  onNext: () => void;
}) {
  return (
    <div className="ply-controls" aria-label="Replay controls">
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
      <span>
        {plyIndex} / {maxPly}
      </span>
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
    </div>
  );
}

function ChessBoard({ fen, activeMove }: { fen: string; activeMove: Move | undefined }) {
  const squares = useMemo(() => parseFenBoard(fen), [fen]);
  const highlighted = useMemo(() => moveSquares(activeMove?.accepted_uci), [activeMove]);

  return (
    <div className="board-wrap">
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
    </div>
  );
}

function MovePanel({
  game,
  plyIndex,
  currentMove,
}: {
  game: GameDetail | undefined;
  plyIndex: number;
  currentMove: Move | undefined;
}) {
  if (!game) {
    return <section className="panel">Select a persisted game to inspect its move trace.</section>;
  }
  if (plyIndex === 0 || !currentMove) {
    return (
      <section className="panel">
        <h3>Initial Position</h3>
        <p className="muted">Step forward to inspect accepted moves, retries, latency, tokens, and eval rows.</p>
        <MoveTable moves={game.moves} activePly={plyIndex} />
      </section>
    );
  }

  const evaluation = currentMove.engine_evaluations[0];
  const acceptedAttempt = currentMove.attempts.find((attempt) => attempt.legal_ok);
  const tokenUsage = acceptedAttempt?.token_usage ?? currentMove.attempts[0]?.token_usage;

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Ply {currentMove.ply}</p>
          <h3>
            {currentMove.color} · {currentMove.accepted_san}
          </h3>
        </div>
        <span className={`badge ${evaluation?.classification ?? "pending"}`}>
          {evaluation?.classification ?? "no eval"}
        </span>
      </div>

      <div className="metric-grid">
        <Metric label="CPL" value={evaluation?.centipawn_loss ?? "—"} />
        <Metric label="Retries" value={currentMove.retries_used} />
        <Metric label="Latency" value={`${currentMove.latency_total_ms.toFixed(1)} ms`} />
        <Metric label="Tokens" value={tokenUsage?.total_tokens ?? "—"} />
      </div>

      <div className="detail-grid">
        <div>
          <h4>Evaluation</h4>
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
        </div>
        <div>
          <h4>Attempts</h4>
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
        </div>
      </div>

      {currentMove.annotations.length > 0 && (
        <div className="commentary-list">
          <h4>Commentary</h4>
          {currentMove.annotations.map((annotation) => (
            <p key={`${annotation.persona}-${annotation.created_at}`}>
              <strong>{annotation.persona}</strong>: {annotation.commentary}
            </p>
          ))}
        </div>
      )}

      <MoveTable moves={game.moves} activePly={plyIndex} />
    </section>
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

function MoveTable({ moves, activePly }: { moves: Move[]; activePly: number }) {
  return (
    <div className="move-table">
      {moves.map((move) => (
        <div key={move.id} className={move.ply === activePly ? "move-line active" : "move-line"}>
          <span>{move.ply}</span>
          <span>{move.accepted_san}</span>
          <span>{move.retries_used} retry</span>
          <span>{move.engine_evaluations[0]?.centipawn_loss ?? "—"} CPL</span>
        </div>
      ))}
    </div>
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
