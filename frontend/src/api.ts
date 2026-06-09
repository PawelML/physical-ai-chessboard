export type GameListItem = {
  id: number;
  run_id: number | null;
  result: string;
  termination_reason: string | null;
  final_fen: string | null;
  started_at: string;
  ended_at: string | null;
};

export type TokenUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  estimated_context_window: number;
  estimated_context_remaining: number;
  truncation_applied: boolean;
  cost_usd: number | null;
};

export type Attempt = {
  id: number;
  ply: number;
  attempt_number: number;
  parsed_move: string | null;
  parse_ok: boolean;
  legal_ok: boolean;
  error_type: string | null;
  latency_ms: number;
  token_usage: TokenUsage | null;
};

export type EngineEvaluation = {
  engine_name: string;
  engine_version: string;
  nodes: number;
  depth_reached: number | null;
  eval_before_cp: number | null;
  eval_after_cp: number | null;
  mate_before: number | null;
  mate_after: number | null;
  best_move_uci: string | null;
  centipawn_loss: number | null;
  classification: string;
};

export type MoveAnnotation = {
  persona: string;
  commentary: string;
  created_at: string;
};

export type Move = {
  id: number;
  ply: number;
  color: "white" | "black";
  fen_before: string;
  fen_after: string;
  accepted_uci: string;
  accepted_san: string;
  legal_move_count: number;
  move_source: string;
  retries_used: number;
  latency_total_ms: number;
  attempts: Attempt[];
  engine_evaluations: EngineEvaluation[];
  annotations: MoveAnnotation[];
};

export type GameDetail = {
  id: number;
  run_id: number | null;
  white_participant_id: number | null;
  black_participant_id: number | null;
  white_player: string | null;
  black_player: string | null;
  result: string;
  termination_reason: string | null;
  final_fen: string | null;
  pgn: string | null;
  moves: Move[];
};

export type LeaderboardRow = {
  id: number;
  run_id: number;
  run_participant_id: number;
  participant: string;
  model_snapshot_id: number | null;
  color: string;
  mode: string;
  legality_mode: string;
  opening_suite_id: number | null;
  games_played: number;
  wins: number;
  draws: number;
  losses: number;
  unfinished: number;
  avg_cpl: number | null;
  blunders: number;
  mistakes: number;
  inaccuracies: number;
  illegal_rate: number;
  malformed_rate: number;
  avg_retries: number;
  forfeit_invalid_count: number;
  avg_latency_ms: number;
  total_tokens: number;
};

export type LeaderboardFilters = {
  runId?: number;
  color?: string;
  mode?: string;
  legalityMode?: string;
};

export type OperationalEvent = {
  id: number;
  run_id: number | null;
  event_kind: string;
  severity: string;
  message: string;
  payload: Record<string, unknown> | null;
  created_at: string;
};

export type RunComparisonRow = {
  run_id: number;
  games_played: number;
  wins: number;
  draws: number;
  losses: number;
  unfinished: number;
  avg_cpl: number | null;
  illegal_rate: number;
  malformed_rate: number;
  avg_retries: number;
  avg_latency_ms: number;
  total_tokens: number;
};

export type ModelOption = {
  id: string;
  label: string;
  provider: string;
};

export type GameJobStatus = "running" | "completed" | "failed";

export type GameJob = {
  id: string;
  status: GameJobStatus;
  white: string;
  black: string;
  legality_mode: string;
  ollama_preset: OllamaPreset;
  guidance_mode: GuidanceMode;
  max_plies: number | null;
  game_id: number | null;
  result: string | null;
  termination_reason: string | null;
  error: string | null;
  created_at: string;
  completed_at: string | null;
};

export type OllamaPreset = "strict" | "low_creativity" | "thinking_if_supported";
export type GuidanceMode = "legal_list" | "strategic_memory";

export type StartGamePayload = {
  white: string;
  black: string;
  legality_mode: "open" | "constrained";
  ollama_preset: OllamaPreset;
  guidance_mode: GuidanceMode;
  max_plies: number | null;
};

export type GpuTelemetry = {
  name: string;
  memory_used_mb: number;
  memory_total_mb: number;
  utilization_percent: number | null;
};

export type OllamaRuntimeModel = {
  name: string;
  size_bytes: number | null;
  size_vram_bytes: number | null;
  processor: string | null;
  context_window: number | null;
  expires_at: string | null;
};

export type RuntimeTelemetry = {
  sampled_at: string;
  gpus: GpuTelemetry[];
  ollama_models: OllamaRuntimeModel[];
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`/api${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${message}`);
  }
  return response.json() as Promise<T>;
}

export function fetchGames(): Promise<GameListItem[]> {
  return getJson<GameListItem[]>("/games");
}

export function fetchGame(gameId: number): Promise<GameDetail> {
  return getJson<GameDetail>(`/games/${gameId}`);
}

export function fetchLeaderboard(filters: LeaderboardFilters = {}): Promise<LeaderboardRow[]> {
  const params = new URLSearchParams();
  if (filters.runId !== undefined) {
    params.set("run_id", String(filters.runId));
  }
  if (filters.color) {
    params.set("color", filters.color);
  }
  if (filters.mode) {
    params.set("mode", filters.mode);
  }
  if (filters.legalityMode) {
    params.set("legality_mode", filters.legalityMode);
  }
  const suffix = params.size > 0 ? `?${params.toString()}` : "";
  return getJson<LeaderboardRow[]>(`/leaderboard${suffix}`);
}

export function fetchRunEvents(runId: number): Promise<OperationalEvent[]> {
  return getJson<OperationalEvent[]>(`/runs/${runId}/events`);
}

export function fetchRunComparison(): Promise<RunComparisonRow[]> {
  return getJson<RunComparisonRow[]>("/runs/compare");
}

export function fetchModelOptions(): Promise<ModelOption[]> {
  return getJson<ModelOption[]>("/models");
}

export function fetchRuntimeTelemetry(): Promise<RuntimeTelemetry> {
  return getJson<RuntimeTelemetry>("/runtime/telemetry");
}

export function fetchGameJobs(): Promise<GameJob[]> {
  return getJson<GameJob[]>("/games/jobs");
}

export function startGame(payload: StartGamePayload): Promise<{ job_id: string; status: string }> {
  return postJson<{ job_id: string; status: string }>("/games/start", payload);
}
