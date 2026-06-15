export function scorePercent(row: { games_played: number; wins: number; draws: number }) {
  if (row.games_played === 0) {
    return "—";
  }
  return `${(((row.wins + row.draws * 0.5) / row.games_played) * 100).toFixed(1)}%`;
}

export function rateWithCi(value: number, low: number, high: number) {
  return `${percent(value)} (${percent(low)}-${percent(high)})`;
}

export function sampleRate(value: number, sampleSize: number) {
  if (sampleSize === 0) {
    return "—";
  }
  return `${percent(value)} n=${sampleSize}`;
}

export function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function movesLabel(avgGamePlies: number) {
  return (avgGamePlies / 2).toFixed(1);
}
