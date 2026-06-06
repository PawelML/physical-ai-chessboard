export type Piece = {
  symbol: string;
  color: "white" | "black";
};

export type BoardSquare = {
  square: string;
  piece: Piece | null;
};

const pieceSymbols: Record<string, string> = {
  p: "♟",
  n: "♞",
  b: "♝",
  r: "♜",
  q: "♛",
  k: "♚",
  P: "♙",
  N: "♘",
  B: "♗",
  R: "♖",
  Q: "♕",
  K: "♔",
};

export const startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export function parseFenBoard(fen: string): BoardSquare[] {
  const board = fen.split(" ")[0];
  const ranks = board.split("/");
  const squares: BoardSquare[] = [];

  ranks.forEach((rank, rankIndex) => {
    let fileIndex = 0;
    for (const char of rank) {
      const emptyCount = Number.parseInt(char, 10);
      if (!Number.isNaN(emptyCount)) {
        for (let i = 0; i < emptyCount; i += 1) {
          squares.push({ square: squareName(fileIndex, rankIndex), piece: null });
          fileIndex += 1;
        }
        continue;
      }
      squares.push({
        square: squareName(fileIndex, rankIndex),
        piece: {
          symbol: pieceSymbols[char] ?? char,
          color: char === char.toUpperCase() ? "white" : "black",
        },
      });
      fileIndex += 1;
    }
  });

  return squares;
}

function squareName(fileIndex: number, rankIndex: number): string {
  const file = String.fromCharCode("a".charCodeAt(0) + fileIndex);
  const rank = 8 - rankIndex;
  return `${file}${rank}`;
}

export function moveSquares(uci: string | undefined): Set<string> {
  if (!uci || uci.length < 4) {
    return new Set();
  }
  return new Set([uci.slice(0, 2), uci.slice(2, 4)]);
}
