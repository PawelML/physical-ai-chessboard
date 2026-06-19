import { useMutation, useQuery } from "@tanstack/react-query";
import { Square } from "lucide-react";
import { useState } from "react";

import {
  fetchGameDefaults,
  saveGameDefaults,
  type GameDefaults,
  type GameJob,
  type GuidanceMode,
  type InferenceMode,
  type ModelOption,
  type StartGamePayload,
  type StartHumanGamePayload,
  type StartStockfishMatchPayload,
  type StockfishLevel,
} from "./api";
import { stockfishLiveLabel } from "./jobLabels";

export function StartGamePanel({
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
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>("single_shot");
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
            inference_mode: inferenceMode,
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
              <StockfishLevelSelect
                value={stockfishLevel}
                disabled={submitting}
                onChange={setStockfishLevel}
              />
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
                <StockfishLevelSelect
                  value={stockfishLevel}
                  disabled={submitting}
                  onChange={setStockfishLevel}
                />
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
          <label>
            <span>Inference mode</span>
            <select
              value={inferenceMode}
              disabled={submitting}
              onChange={(event) => setInferenceMode(event.target.value as InferenceMode)}
            >
              <option value="single_shot">Single-shot</option>
              <option value="native_think">Native thinking</option>
              <option value="revise">Two-pass revise</option>
              <option value="candidate_critic">Candidate + critic</option>
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
                  {inferenceJobOption(job)} · {ollamaJobOptions(job)} · {jobLabel(job)}
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

function StockfishLevelSelect({
  value,
  disabled,
  onChange,
}: {
  value: StockfishLevel;
  disabled: boolean;
  onChange: (value: StockfishLevel) => void;
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value as StockfishLevel)}
    >
      <option value="beginner">Beginner · 1320 Elo</option>
      <option value="club">Club · 1600 Elo</option>
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

function inferenceJobOption(job: GameJob) {
  if (job.inference_mode === "native_think") {
    return "native thinking";
  }
  if (job.inference_mode === "revise") {
    return "revise";
  }
  if (job.inference_mode === "candidate_critic") {
    return "candidate critic";
  }
  return "single-shot";
}
