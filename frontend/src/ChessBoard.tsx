import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { HumanGameState, Move } from "./api";
import { moveSquares, parseFenBoard, type Piece } from "./chess";

export function ChessBoard({
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
    white: { q: "♛", r: "♜", b: "♝", n: "♞" },
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
