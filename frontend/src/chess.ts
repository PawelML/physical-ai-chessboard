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
  P: "♟",
  N: "♞",
  B: "♝",
  R: "♜",
  Q: "♛",
  K: "♚",
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

export function applyUciMoveToFen(fen: string, uci: string): string {
  const parts = fen.split(" ");
  const board = boardFromFen(parts[0]);
  const from = uci.slice(0, 2);
  const to = uci.slice(2, 4);
  const promotion = uci[4];
  const fromCoord = squareCoord(from);
  const toCoord = squareCoord(to);
  const piece = board[fromCoord.rank][fromCoord.file];
  const capturedPiece = board[toCoord.rank][toCoord.file];

  if (!piece) {
    return fen;
  }

  const isWhite = piece === piece.toUpperCase();
  const isPawn = piece.toLowerCase() === "p";
  const isKing = piece.toLowerCase() === "k";
  const previousEnPassant = parts[3] ?? "-";
  board[fromCoord.rank][fromCoord.file] = null;

  if (isPawn && to === previousEnPassant && !board[toCoord.rank][toCoord.file]) {
    board[toCoord.rank + (isWhite ? 1 : -1)][toCoord.file] = null;
  }

  if (isKing && Math.abs(toCoord.file - fromCoord.file) === 2) {
    const rookFromFile = toCoord.file > fromCoord.file ? 7 : 0;
    const rookToFile = toCoord.file > fromCoord.file ? 5 : 3;
    board[toCoord.rank][rookToFile] = board[toCoord.rank][rookFromFile];
    board[toCoord.rank][rookFromFile] = null;
  }

  board[toCoord.rank][toCoord.file] = promotion
    ? isWhite
      ? promotion.toUpperCase()
      : promotion
    : piece;

  const nextTurn = (parts[1] ?? "w") === "w" ? "b" : "w";
  const castling = updateCastlingRights(parts[2] ?? "-", piece, from, to);
  const enPassant =
    isPawn && Math.abs(toCoord.rank - fromCoord.rank) === 2
      ? squareName(fromCoord.file, (fromCoord.rank + toCoord.rank) / 2)
      : "-";
  const halfmove = isPawn || capturedPiece ? "0" : String(Number(parts[4] ?? 0) + 1);
  const fullmove =
    (parts[1] ?? "w") === "b" ? String(Number(parts[5] ?? 1) + 1) : (parts[5] ?? "1");

  return [boardToFen(board), nextTurn, castling, enPassant, halfmove, fullmove].join(" ");
}

function boardFromFen(boardFen: string): (string | null)[][] {
  return boardFen.split("/").map((rank) => {
    const row: (string | null)[] = [];
    for (const char of rank) {
      const emptyCount = Number.parseInt(char, 10);
      if (Number.isNaN(emptyCount)) {
        row.push(char);
      } else {
        for (let i = 0; i < emptyCount; i += 1) {
          row.push(null);
        }
      }
    }
    return row;
  });
}

function boardToFen(board: (string | null)[][]): string {
  return board
    .map((row) => {
      let empty = 0;
      let rank = "";
      for (const piece of row) {
        if (!piece) {
          empty += 1;
          continue;
        }
        if (empty > 0) {
          rank += String(empty);
          empty = 0;
        }
        rank += piece;
      }
      return rank + (empty > 0 ? String(empty) : "");
    })
    .join("/");
}

function squareCoord(square: string): { file: number; rank: number } {
  return {
    file: square.charCodeAt(0) - "a".charCodeAt(0),
    rank: 8 - Number(square[1]),
  };
}

function updateCastlingRights(rights: string, piece: string, from: string, to: string): string {
  let next = rights === "-" ? "" : rights;
  const remove = (value: string) => {
    next = next.replace(value, "");
  };

  if (piece === "K") {
    remove("K");
    remove("Q");
  }
  if (piece === "k") {
    remove("k");
    remove("q");
  }
  if (from === "h1" || to === "h1") {
    remove("K");
  }
  if (from === "a1" || to === "a1") {
    remove("Q");
  }
  if (from === "h8" || to === "h8") {
    remove("k");
  }
  if (from === "a8" || to === "a8") {
    remove("q");
  }

  return next || "-";
}
