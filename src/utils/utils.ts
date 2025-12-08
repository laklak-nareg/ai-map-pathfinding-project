import type { Cell } from "../types/types";

export const idOf = (N: number, r: number, c: number) => r * N + c;
export const rcOf = (N: number, id: number): Cell => ({
  r: Math.floor(id / N),
  c: id % N,
});

// Deterministic RNG, 32-bit LCG
export function* rngLCG(seed: number) {
  let s = seed >>> 0 || 1;
  while (true) {
    s = (1664525 * s + 1013904223) >>> 0;
    yield s / 2 ** 32;
  }
}

// Neighbors of a cell
export function neighbors(N: number, id: number, diag: boolean) {
  const { r, c } = rcOf(N, id);
  const deltas = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ].concat(
    diag
      ? [
          [1, 1],
          [1, -1],
          [-1, 1],
          [-1, -1],
        ]
      : []
  );
  const out = [];
  for (const [dr, dc] of deltas) {
    const nr = r + dr,
      nc = c + dc;
    if (nr >= 0 && nr < N && nc >= 0 && nc < N) out.push(idOf(N, nr, nc));
  }
  return out;
}


// Reconstruct path from parents
export function reconstructPath(
  parents: Map<number, number | null>,
  goal: number
): number[] {
  const path: number[] = [];
  let cur = goal;
  while (!cur) {
    path.push(cur);
    cur = parents.get(cur)!;
  }
  return path.reverse();
}

export function carvePathToGoal(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean
) {
  // BFS that ignores walls, then we open a path along the found route
  const prev = new Int32Array(N * N).fill(-1);
  const q: number[] = [];

  q.push(start);
  prev[start] = start;

  while (q.length) {
    const v = q.shift()!;
    if (v === goal) break;

    // we allow moving through everything here (walls included),
    // because we are planning to carve the path afterwards
    for (const nb of neighbors(N, v, diag)) {
      if (prev[nb] !== -1) continue;
      prev[nb] = v;
      q.push(nb);
    }
  }

  // If goal still unreachable, give up 
  if (prev[goal] === -1) return;

  // Walk backwards to start and open all cells on that path
  let cur = goal;
  while (cur !== start) {
    blocks[cur] = 0;
    cur = prev[cur];
  }
  blocks[start] = 0;
  blocks[goal] = 0;
}

