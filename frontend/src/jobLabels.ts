import type { GameJob } from "./api";

export function stockfishLiveLabel(job: GameJob) {
  const requested = job.games_requested ?? 1;
  const current = Math.min(job.games_completed + 1, requested);
  return `Game ${current}/${requested}`;
}
