import type { AlgoState } from "../interfaces/interfaces";
import { chebyshev, manhattan } from "../utils/heuristic/buildHeuristic";
import { MinHeap } from "../utils/MinHeap/MinHeap";
import { idOf, neighbors, reconstructPath, rngLCG } from "../utils/utils";

export function* algoGreedy(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean,
  heuristic: (id: number) => number
) {
  const heap = new MinHeap<number>();
  const seen = new Uint8Array(N * N);
  const parents = new Map<number, number | null>();
  parents.set(start, null);
  const open = new Set<number>();
  const closed = new Set<number>();
  const begin = performance.now();
  const meta = { nodesExpanded: 0, peakFrontier: 0 };

  heap.push(heuristic(start), start);
  open.add(start);

  while (heap.size()) {
    const n = heap.pop()!;
    open.delete(n);
    if (seen[n]) continue;
    seen[n] = 1;
    closed.add(n);
    meta.nodesExpanded++;

    const cur: AlgoState = {
      key: "Greedy",
      open: new Set(open),
      closed: new Set(closed),
      parents,
      found: false,
      finished: false,
      current: n,
      nodesExpanded: meta.nodesExpanded,
      peakFrontier: Math.max(meta.peakFrontier, open.size),
      lastRuntimeMs: performance.now() - begin,
    };
    yield cur;

    if (n === goal) {
      const path = reconstructPath(parents, goal);
      return {
        ...cur,
        found: true,
        finished: true,
        path,
        lastRuntimeMs: performance.now() - begin,
      };
    }

    for (const m of neighbors(N, n, diag)) {
      if (blocks[m] === 1) continue;
      if (!seen[m]) {
        if (!parents.has(m)) parents.set(m, n);
        heap.push(heuristic(m), m);
        open.add(m);
      }
    }
    meta.peakFrontier = Math.max(meta.peakFrontier, open.size);
  }

  return {
    key: "Greedy",
    open: new Set(),
    closed,
    parents,
    found: false,
    finished: true,
    nodesExpanded: meta.nodesExpanded,
    peakFrontier: meta.peakFrontier,
    lastRuntimeMs: performance.now() - begin,
  } as AlgoState;
}

export function pickDeceptiveStartGoal(
  N: number,
  blocks: Uint8Array,
  diag: boolean,
  seed: number
): { start: number; goal: number } {
  // collect all free cells
  const free: number[] = [];
  for (let i = 0; i < N * N; i++) {
    if (blocks[i] === 0) free.push(i);
  }
  // extreme fallback
  if (free.length < 2) {
    return { start: 0, goal: idOf(N, N - 1, N - 1) };
  }

  // deterministic RNG so it’s reproducible per seed
  const R = rngLCG(seed + 4242);
  const start = free[Math.floor((R.next().value as number) * free.length)];

  // BFS distances from start (only through free cells)
  const dist = new Int32Array(N * N).fill(-1);
  const q: number[] = [];
  dist[start] = 0;
  q.push(start);

  while (q.length) {
    const v = q.shift()!;
    for (const nb of neighbors(N, v, diag)) {
      if (blocks[nb] === 1) continue; // wall
      if (dist[nb] !== -1) continue;
      dist[nb] = dist[v] + 1;
      q.push(nb);
    }
  }

  const hFun = diag ? chebyshev : manhattan;

  let bestGoal = -1;
  let bestScore = -1;

  for (const cell of free) {
    if (cell === start) continue;
    if (dist[cell] === -1) continue; // unreachable (shouldn’t happen in a perfect maze)

    // how many free neighbours? dead end ≈ 1
    let freeNb = 0;
    for (const nb of neighbors(N, cell, diag)) {
      if (blocks[nb] === 0) freeNb++;
    }
    const isDeadEnd = freeNb === 1;

    const h = hFun(N, start, cell);
    const d = dist[cell];

    // we want cases where d >> h
    const gap = d - h;
    if (gap <= 0) continue;

    const score = gap + (isDeadEnd ? 5 : 0); // dead ends get a bonus

    if (score > bestScore) {
      bestScore = score;
      bestGoal = cell;
    }
  }

  // fallback: just pick the farthest reachable cell by BFS
  if (bestGoal === -1) {
    let far = start;
    let farDist = 0;
    for (let i = 0; i < N * N; i++) {
      if (dist[i] > farDist) {
        farDist = dist[i];
        far = i;
      }
    }
    bestGoal = far;
  }

  return { start, goal: bestGoal };
}
