import type { HeuristicType } from "../../types/types";
import { idOf, rcOf } from "../utils";

export const manhattan = (N: number, a: number, b: number) => {
  const A = rcOf(N, a),
    B = rcOf(N, b);
  return Math.abs(A.r - B.r) + Math.abs(A.c - B.c);
};
export const chebyshev = (N: number, a: number, b: number) => {
  const A = rcOf(N, a),
    B = rcOf(N, b);
  return Math.max(Math.abs(A.r - B.r), Math.abs(A.c - B.c));
};

const Euclidean = (N: number, a: number, b: number) => {
  const goalRC = rcOf(N, b);
  const p = rcOf(N, a);
  const dr = p.r - goalRC.r;
  const dc = p.c - goalRC.c;
  return Math.sqrt(dr * dr + dc * dc);
};

const wallPenalty = (N: number, id: number, blocks: Uint8Array) => {
  const { r, c } = rcOf(N, id);
  let count = 0;
  const dirs: [number, number][] = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ];

  for (const [dr, dc] of dirs) {
    const nr = r + dr;
    const nc = c + dc;
    if (nr < 0 || nr >= N || nc < 0 || nc >= N) {
      count++;
      continue;
    }
    if (blocks[idOf(N, nr, nc)] === 1) count++;
  }
  return count;
};

export function buildHeuristic(
  N: number,
  blocks: Uint8Array,
  goal: number,
  diag: boolean,
  type: HeuristicType
): (id: number) => number {
  const baseManhattan = (id: number) => manhattan(N, id, goal);
  const baseChebyshev = (id: number) => chebyshev(N, id, goal);
  const baseEuclidean = (id: number) => Euclidean(N, id, goal);

  switch (type) {
    case "WallAware": {
      const base = diag ? baseChebyshev : baseManhattan;
      const lambda = 1.2;

      return (id: number) => {
        const h = base(id);
        if (h === 0) return 0;
        const penalty = wallPenalty(N, id, blocks);
        return h + lambda * penalty;
      };
    }
    case "Chebyshev":
      return baseChebyshev;
    case "Euclidean":
      return baseEuclidean;
    default:
      return baseManhattan;
  }
}
