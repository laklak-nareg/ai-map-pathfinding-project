import React, { useEffect, useMemo, useRef, useState } from "react";

// =====================
// Pathfinding Lab — Single-file React app
// - Side-by-side visualization of BFS, Dijkstra, Greedy, A*, BiA*
// - Configurable grid size, map type (Empty / Random / Maze), density, seed
// - Lockstep animation so each algorithm advances one expansion per tick
// - Canvas rendering for speed
// =====================

// ---------- Types ----------
type Cell = { r: number; c: number };

type MapType = "Empty" | "Random" | "Maze";

type AlgoKey = "BFS" | "Dijkstra" | "Greedy" | "A*" | "BiA*";

type HeuristicType = "Manhattan" | "Chebyshev" | "Euclidean" | "WallAware";

interface AlgoState {
  key: AlgoKey;
  open: Set<number>; // ids waiting (frontier) (for visualization only)
  closed: Set<number>; // visited/expanded
  parents: Map<number, number | null>;
  found: boolean;
  finished: boolean; // found or exhausted
  current?: number; // node popped this tick
  path?: number[]; // final path ids when found
  nodesExpanded: number;
  peakFrontier: number;
  lastRuntimeMs: number;
}

interface RunConfig {
  N: number;
  mapType: MapType;
  density: number; // for Random
  seed: number;
  diag: boolean; // allow 8-neighbors
}

// ---------- Utilities ----------
const idOf = (N: number, r: number, c: number) => r * N + c;
const rcOf = (N: number, id: number): Cell => ({
  r: Math.floor(id / N),
  c: id % N,
});

function* rngLCG(seed: number) {
  // Deterministic RNG, 32-bit LCG
  let s = (seed >>> 0) || 1;
  while (true) {
    s = (1664525 * s + 1013904223) >>> 0;
    yield s / 2 ** 32;
  }
}

// Priority Queue (binary heap)
class MinHeap<T> {
  private a: { k: number; v: T }[] = [];
  size() {
    return this.a.length;
  }
  push(k: number, v: T) {
    this.a.push({ k, v });
    this.bubbleUp(this.a.length - 1);
  }
  pop(): T | undefined {
    if (this.a.length === 0) return undefined;
    const top = this.a[0].v;
    const last = this.a.pop()!;
    if (this.a.length) {
      this.a[0] = last;
      this.bubbleDown(0);
    }
    return top;
  }
  peekKey(): number | undefined {
    return this.a[0]?.k;
  }
  private bubbleUp(i: number) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.a[p].k <= this.a[i].k) break;
      [this.a[p], this.a[i]] = [this.a[i], this.a[p]];
      i = p;
    }
  }
  private bubbleDown(i: number) {
    const n = this.a.length;
    while (true) {
      let l = i * 2 + 1,
        r = l + 1,
        m = i;
      if (l < n && this.a[l].k < this.a[m].k) m = l;
      if (r < n && this.a[r].k < this.a[m].k) m = r;
      if (m === i) break;
      [this.a[m], this.a[i]] = [this.a[i], this.a[m]];
      i = m;
    }
  }
}

// Manhattan / Chebyshev heuristics
const manhattan = (N: number, a: number, b: number) => {
  const A = rcOf(N, a),
    B = rcOf(N, b);
  return Math.abs(A.r - B.r) + Math.abs(A.c - B.c);
};
const chebyshev = (N: number, a: number, b: number) => {
  const A = rcOf(N, a),
    B = rcOf(N, b);
  return Math.max(Math.abs(A.r - B.r), Math.abs(A.c - B.c));
};

// Neighbors (4- or 8-connected)
function neighbors(N: number, id: number, diag: boolean) {
  const { r, c } = rcOf(N, id);
  const deltas4 = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ];
  const deltas8 = diag
    ? deltas4.concat([
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
      ])
    : deltas4;
  const out: number[] = [];
  for (const [dr, dc] of deltas8) {
    const nr = r + dr,
      nc = c + dc;
    if (nr >= 0 && nr < N && nc >= 0 && nc < N) out.push(idOf(N, nr, nc));
  }
  return out;
}

// Reconstruct path from parents
function reconstructPath(
  parents: Map<number, number | null>,
  goal: number
): number[] {
  const path: number[] = [];
  let cur: number | null | undefined = goal;
  while (cur != null) {
    path.push(cur);
    cur = parents.get(cur)!;
  }
  return path.reverse();
}

// ---------- Map Generation ----------
function generateEmpty(N: number) {
  const blocks = new Uint8Array(N * N); // 0 free, 1 wall
  return blocks;
}

function generateRandom(
  N: number,
  density: number,
  seed: number,
  start: number,
  goal: number
) {
  const blocks = new Uint8Array(N * N);
  const R = rngLCG(seed);
  for (let i = 0; i < N * N; i++) {
    const p = R.next().value as number;
    blocks[i] = p < density ? 1 : 0;
  }
  blocks[start] = 0;
  blocks[goal] = 0;
  return blocks;
}

// Maze via DFS backtracker on a coarser cell grid
function generateMaze(N: number, seed: number) {
  // Start with all walls
  const blocks = new Uint8Array(N * N).fill(1);
  const R = rngLCG(seed);

  // We'll carve passages on a grid of "cells" at odd coordinates
  const inBoundsCell = (r: number, c: number) =>
    r > 0 && r < N - 1 && c > 0 && c < N - 1 && r % 2 === 1 && c % 2 === 1;

  const visited = new Uint8Array(N * N);
  const stack: Cell[] = [];

  // Start carving from an odd cell (1,1) if possible
  let sr = 1,
    sc = 1;
  if (N <= 2) {
    sr = 0;
    sc = 0;
  }

  stack.push({ r: sr, c: sc });
  visited[idOf(N, sr, sc)] = 1;
  blocks[idOf(N, sr, sc)] = 0;

  const cellDirs: [number, number][] = [
    [2, 0],
    [-2, 0],
    [0, 2],
    [0, -2],
  ];

  while (stack.length) {
    const cur = stack[stack.length - 1];

    // shuffle directions for randomness
    for (let i = cellDirs.length - 1; i > 0; i--) {
      const j = Math.floor((R.next().value as number) * (i + 1));
      [cellDirs[i], cellDirs[j]] = [cellDirs[j], cellDirs[i]];
    }

    let moved = false;

    for (const [dr, dc] of cellDirs) {
      const nr = cur.r + dr;
      const nc = cur.c + dc;
      if (!inBoundsCell(nr, nc)) continue;

      const nid = idOf(N, nr, nc);
      if (visited[nid]) continue;

      visited[nid] = 1;
      blocks[nid] = 0;

      // carve the wall between current cell and next cell
      const wr = cur.r + dr / 2;
      const wc = cur.c + dc / 2;
      blocks[idOf(N, wr, wc)] = 0;

      stack.push({ r: nr, c: nc });
      moved = true;
      break;
    }

    if (!moved) {
      stack.pop();
    }
  }

  return blocks;
}

function carvePathToGoal(
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

    for (const nb of neighbors(N, v, diag)) {
      if (prev[nb] !== -1) continue;
      // we allow moving through everything here (walls included),
      // because we are planning to carve the path afterwards
      prev[nb] = v;
      q.push(nb);
    }
  }

  // If goal still unreachable, give up (should be rare)
  if (prev[goal] === -1) return;

  // Walk backwards from goal to start and open all cells on that path
  let cur = goal;
  while (cur !== start) {
    blocks[cur] = 0;
    cur = prev[cur];
  }
  blocks[start] = 0;
  blocks[goal] = 0;
}

// Connectivity check via quick BFS (to avoid unsolvable randoms);
function solvable(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean
) {
  const q: number[] = [start];
  const seen = new Uint8Array(N * N);
  seen[start] = 1;
  while (q.length) {
    const x = q.shift()!;
    if (x === goal) return true;
    for (const y of neighbors(N, x, diag))
      if (!seen[y] && blocks[y] === 0) {
        seen[y] = 1;
        q.push(y);
      }
  }
  return false;
}

// Maze-friendly heuristic builder
function buildHeuristic(
  N: number,
  blocks: Uint8Array,
  goal: number,
  diag: boolean,
  type: HeuristicType
): (id: number) => number {
  const goalRC = rcOf(N, goal);

  const baseManhattan = (id: number) => {
    const p = rcOf(N, id);
    return Math.abs(p.r - goalRC.r) + Math.abs(p.c - goalRC.c);
  };

  const baseChebyshev = (id: number) => {
    const p = rcOf(N, id);
    return Math.max(Math.abs(p.r - goalRC.r), Math.abs(p.c - goalRC.c));
  };

  const baseEuclidean = (id: number) => {
    const p = rcOf(N, id);
    const dr = p.r - goalRC.r;
    const dc = p.c - goalRC.c;
    return Math.sqrt(dr * dr + dc * dc);
  };

  const wallPenalty = (id: number) => {
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

  // Wall-aware heuristic: punish cells that are squeezed by walls (tunnels/dead ends)
  if (type === "WallAware") {
    const base = diag ? baseChebyshev : baseManhattan;
    const lambda = 1.2; // stronger penalty

    return (id: number) => {
      const h = base(id);
      if (h === 0) return 0;
      const penalty = wallPenalty(id);
      return h + lambda * penalty;
    };
  }

  if (type === "Chebyshev") return baseChebyshev;
  if (type === "Euclidean") return baseEuclidean;
  return baseManhattan;
}

// ---------- Algorithm Step Generators ----------

// Bidirectional A* (visual version)
function* algoBiAStar(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean,
  hForward: (id: number) => number,
  hBackward: (id: number) => number
): Generator<AlgoState, AlgoState, void> {
  const heapF = new MinHeap<number>(); // forward (start -> goal)
  const heapB = new MinHeap<number>(); // backward (goal -> start)

  const gF = new Float64Array(N * N).fill(Infinity);
  const gB = new Float64Array(N * N).fill(Infinity);
  gF[start] = 0;
  gB[goal] = 0;

  const seenF = new Uint8Array(N * N);
  const seenB = new Uint8Array(N * N);

  const parentsF = new Map<number, number | null>(); // from start
  const parentsB = new Map<number, number | null>(); // from goal
  parentsF.set(start, null);
  parentsB.set(goal, null);

  const openF = new Set<number>();
  const openB = new Set<number>();
  const closedF = new Set<number>();
  const closedB = new Set<number>();

  const begin = performance.now();
  const meta = { nodesExpanded: 0, peakFrontier: 0 };

  heapF.push(hForward(start), start);
  heapB.push(hBackward(goal), goal);
  openF.add(start);
  openB.add(goal);

  let meeting: number | null = null;

  const reconstructBiPath = (): number[] => {
    if (meeting === null) return [];
    // forward: start -> meeting
    const pathF = reconstructPath(parentsF, meeting);
    // backward: goal -> meeting
    const pathB = reconstructPath(parentsB, meeting); // [goal, ..., meeting]
    pathB.reverse(); // [meeting, ..., goal]
    // concat, drop duplicate meeting
    return pathF.concat(pathB.slice(1));
  };

  while (heapF.size() || heapB.size()) {
    // Expand FORWARD once
    if (heapF.size()) {
      const n = heapF.pop()!;
      openF.delete(n);
      if (!seenF[n]) {
        seenF[n] = 1;
        closedF.add(n);
        meta.nodesExpanded++;

        if (seenB[n]) {
          meeting = n;
          const path = reconstructBiPath();
          const openAll = new Set<number>([...openF, ...openB]);
          const closedAll = new Set<number>([...closedF, ...closedB]);
          return {
            key: "BiA*",
            open: openAll,
            closed: closedAll,
            parents: parentsF, // forward parents are enough to show start-side path
            found: true,
            finished: true,
            current: n,
            path,
            nodesExpanded: meta.nodesExpanded,
            peakFrontier: Math.max(meta.peakFrontier, openAll.size),
            lastRuntimeMs: performance.now() - begin,
          };
        }

        for (const m of neighbors(N, n, diag)) {
          if (blocks[m] === 1) continue;
          const w =
            diag &&
            rcOf(N, n).r !== rcOf(N, m).r &&
            rcOf(N, n).c !== rcOf(N, m).c
              ? Math.SQRT2
              : 1;
          const ng = gF[n] + w;
          if (ng < gF[m]) {
            gF[m] = ng;
            parentsF.set(m, n);
            const f = ng + hForward(m);
            heapF.push(f, m);
            openF.add(m);
          }
        }
      }
    }

    // Expand BACKWARD once
    if (heapB.size()) {
      const n = heapB.pop()!;
      openB.delete(n);
      if (!seenB[n]) {
        seenB[n] = 1;
        closedB.add(n);
        meta.nodesExpanded++;

        if (seenF[n]) {
          meeting = n;
          const path = reconstructBiPath();
          const openAll = new Set<number>([...openF, ...openB]);
          const closedAll = new Set<number>([...closedF, ...closedB]);
          return {
            key: "BiA*",
            open: openAll,
            closed: closedAll,
            parents: parentsF,
            found: true,
            finished: true,
            current: n,
            path,
            nodesExpanded: meta.nodesExpanded,
            peakFrontier: Math.max(meta.peakFrontier, openAll.size),
            lastRuntimeMs: performance.now() - begin,
          };
        }

        for (const m of neighbors(N, n, diag)) {
          if (blocks[m] === 1) continue;
          const w =
            diag &&
            rcOf(N, n).r !== rcOf(N, m).r &&
            rcOf(N, n).c !== rcOf(N, m).c
              ? Math.SQRT2
              : 1;
          const ng = gB[n] + w;
          if (ng < gB[m]) {
            gB[m] = ng;
            parentsB.set(m, n);
            const f = ng + hBackward(m);
            heapB.push(f, m);
            openB.add(m);
          }
        }
      }
    }

    // yield combined visualization state
    const openAll = new Set<number>([...openF, ...openB]);
    const closedAll = new Set<number>([...closedF, ...closedB]);
    meta.peakFrontier = Math.max(meta.peakFrontier, openAll.size);

    const cur: AlgoState = {
      key: "BiA*",
      open: openAll,
      closed: closedAll,
      parents: parentsF,
      found: false,
      finished: false,
      current: undefined as any,
      nodesExpanded: meta.nodesExpanded,
      peakFrontier: meta.peakFrontier,
      lastRuntimeMs: performance.now() - begin,
    };

    yield cur;
  }

  // No meeting → no path
  const openAll = new Set<number>([...openF, ...openB]);
  const closedAll = new Set<number>([...closedF, ...closedB]);
  return {
    key: "BiA*",
    open: openAll,
    closed: closedAll,
    parents: parentsF,
    found: false,
    finished: true,
    current: undefined,
    path: undefined,
    nodesExpanded: meta.nodesExpanded,
    peakFrontier: meta.peakFrontier,
    lastRuntimeMs: performance.now() - begin,
  } as AlgoState;
}

function* algoBFS(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean
) {
  const openQ: number[] = [start];
  const inQ = new Uint8Array(N * N);
  inQ[start] = 1;
  const closed = new Set<number>();
  const parents = new Map<number, number | null>();
  parents.set(start, null);
  const meta = { nodesExpanded: 0, peakFrontier: 1 };
  const begin = performance.now();

  while (openQ.length) {
    meta.peakFrontier = Math.max(meta.peakFrontier, openQ.length);
    const n = openQ.shift()!;
    inQ[n] = 0;
    closed.add(n);
    meta.nodesExpanded++;
    const cur: AlgoState = {
      key: "BFS",
      open: new Set(openQ),
      closed: new Set(closed),
      parents,
      found: false,
      finished: false,
      current: n,
      nodesExpanded: meta.nodesExpanded,
      peakFrontier: meta.peakFrontier,
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
      if (!parents.has(m)) {
        parents.set(m, n);
        openQ.push(m);
        inQ[m] = 1;
      }
    }
  }
  return {
    key: "BFS",
    open: new Set(),
    closed: closed,
    parents,
    found: false,
    finished: true,
    nodesExpanded: meta.nodesExpanded,
    peakFrontier: meta.peakFrontier,
    lastRuntimeMs: performance.now() - begin,
  } as AlgoState;
}


function* algoDijkstra(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean
) {
  const heap = new MinHeap<number>();
  const g = new Float64Array(N * N).fill(Infinity);
  g[start] = 0;
  const seen = new Uint8Array(N * N);
  const parents = new Map<number, number | null>();
  parents.set(start, null);
  const open = new Set<number>();
  const closed = new Set<number>();
  const meta = { nodesExpanded: 0, peakFrontier: 0 };
  const begin = performance.now();

  heap.push(0, start);
  open.add(start);
  while (heap.size()) {
    const n = heap.pop()!;
    open.delete(n);
    if (seen[n]) continue;
    seen[n] = 1;
    closed.add(n);
    meta.nodesExpanded++;

    const cur: AlgoState = {
      key: "Dijkstra",
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
      const w =
        diag &&
        rcOf(N, n).r !== rcOf(N, m).r &&
        rcOf(N, n).c !== rcOf(N, m).c
          ? Math.SQRT2
          : 1;
      const ng = g[n] + w;
      if (ng < g[m]) {
        g[m] = ng;
        parents.set(m, n);
        heap.push(ng, m);
        open.add(m);
      }
    }
    meta.peakFrontier = Math.max(meta.peakFrontier, open.size);
  }
  return {
    key: "Dijkstra",
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

function* algoGreedy(
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

function* algoAStar(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean,
  heuristic: (id: number) => number
) {
  const heap = new MinHeap<number>();
  const g = new Float64Array(N * N).fill(Infinity);
  g[start] = 0;
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
      key: "A*",
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
      const w =
        diag &&
        rcOf(N, n).r !== rcOf(N, m).r &&
        rcOf(N, n).c !== rcOf(N, m).c
          ? Math.SQRT2
          : 1;
      const ng = g[n] + w;
      if (ng < g[m]) {
        g[m] = ng;
        parents.set(m, n);
        const f = ng + heuristic(m);
        heap.push(f, m);
        open.add(m);
      }
    }
    meta.peakFrontier = Math.max(meta.peakFrontier, open.size);
  }

  return {
    key: "A*",
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

// ---------- Canvas Drawing ----------
function drawPanel(
  ctx: CanvasRenderingContext2D,
  N: number,
  sizePx: number,
  blocks: Uint8Array,
  state: AlgoState | null,
  start: number,
  goal: number
) {
  const cell = sizePx / N;
  ctx.clearRect(0, 0, sizePx, sizePx);
  // background cells
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const id = idOf(N, r, c);
      if (blocks[id]) {
        ctx.fillStyle = "#0f172a";
      } else {
        ctx.fillStyle = "#f8fafc";
      }
      ctx.fillRect(c * cell, r * cell, cell, cell);
    }
  }
  if (state) {
    // visited/closed
    ctx.fillStyle = "#fde68a"; // amber-200
    state.closed.forEach((id) => {
      const { r, c } = rcOf(N, id);
      ctx.fillRect(c * cell, r * cell, cell, cell);
    });
    // open/frontier
    ctx.fillStyle = "#bfdbfe"; // blue-200
    state.open.forEach((id) => {
      const { r, c } = rcOf(N, id);
      ctx.fillRect(c * cell, r * cell, cell, cell);
    });
    // current
    if (state.current != null) {
      const { r, c } = rcOf(N, state.current);
      ctx.fillStyle = "#ef4444";
      ctx.fillRect(c * cell, r * cell, cell, cell);
    }
    // final path
    if (state.path) {
      ctx.fillStyle = "#86efac"; // green-300
      for (const id of state.path) {
        const { r, c } = rcOf(N, id);
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
    }
  }
  // start/goal overlays
  const sRC = rcOf(N, start),
    gRC = rcOf(N, goal);
  ctx.fillStyle = "#22c55e";
  ctx.fillRect(sRC.c * cell, sRC.r * cell, cell, cell);
  ctx.fillStyle = "#8b5cf6";
  ctx.fillRect(gRC.c * cell, gRC.r * cell, cell, cell);
  // grid lines (light)
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= N; i++) {
    ctx.beginPath();
    ctx.moveTo(0, i * cell);
    ctx.lineTo(sizePx, i * cell);
    ctx.stroke();
  }
  for (let j = 0; j <= N; j++) {
    ctx.beginPath();
    ctx.moveTo(j * cell, 0);
    ctx.lineTo(j * cell, sizePx);
    ctx.stroke();
  }
}

// Pick a deceptive start/goal pair in a maze to mislead Greedy
function pickDeceptiveStartGoal(
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

// ---------- Main Component ----------
export default function PathfindingLab() {
  // UI State
  const [N, setN] = useState(16);
  const [mapType, setMapType] = useState<MapType>("Empty");
  const [density, setDensity] = useState(0.25);
  const [seed, setSeed] = useState(12345);
  const [diag, setDiag] = useState(false);
  const [speed, setSpeed] = useState(10); // steps per second
  const [running, setRunning] = useState(false);

  const [heuristicType, setHeuristicType] =
    useState<HeuristicType>("Manhattan");
  const [deceptiveMaze, setDeceptiveMaze] = useState(false);

  const sizePx = 360; // per panel

  // Start and goal are now stateful, so we can adapt them to the map
  const [start, setStart] = useState(0);
  const [goal, setGoal] = useState(0);

  // Map generation
  const { blocks, startId, goalId } = useMemo(() => {
    let b: Uint8Array;
    let s = start;
    let g = goal;

    if (mapType === "Empty") {
      // Simple: top-left to bottom-right
      b = generateEmpty(N);
      s = 0;
      g = idOf(N, N - 1, N - 1);
    } else if (mapType === "Random") {
      // Start fixed at 0, goal random but not equal to start
      s = 0;
      const R = rngLCG(seed + 999);
      let gr = Math.floor((R.next().value as number) * N);
      let gc = Math.floor((R.next().value as number) * N);
      let gid = idOf(N, gr, gc);
      if (gid === s) {
        gr = (gr + 1) % N;
        gid = idOf(N, gr, gc);
      }
      g = gid;

      // 1) Generate a random obstacle field
      b = generateRandom(N, density, seed, s, g);

      // 2) GUARANTEE at least one path from s → g
      carvePathToGoal(N, b, s, g, diag);
    } else {
      // Maze
      b = generateMaze(N, seed);

      if (deceptiveMaze) {
        // Special case: pick a start/goal that misleads Greedy
        const pair = pickDeceptiveStartGoal(N, b, diag, seed);
        s = pair.start;
        g = pair.goal;
      } else {
        // Normal maze: just pick two far-ish open cells
        const open: number[] = [];
        for (let i = 0; i < N * N; i++) {
          if (b[i] === 0) open.push(i);
        }

        if (open.length >= 2) {
          s = open[0];
          g = open[open.length - 1];
        } else {
          // Extreme fallback
          s = 0;
          g = idOf(N, N - 1, N - 1);
          b[s] = 0;
          b[g] = 0;
        }
      }
    }

    return { blocks: b!, startId: s, goalId: g };
  }, [N, mapType, density, seed, diag, deceptiveMaze]);

  // Heuristic for Greedy
  const greedyHeuristicFn = useMemo(() => {
    const typeForGreedy =
      deceptiveMaze && mapType === "Maze"
        ? "Manhattan" // force naive heuristic in trap mode
        : heuristicType; // otherwise follow user selection

    return buildHeuristic(N, blocks, goal, diag, typeForGreedy);
  }, [N, blocks, goal, diag, heuristicType, deceptiveMaze, mapType]);

  // Heuristic for A*
  const aStarHeuristicFn = useMemo(() => {
    const typeForAStar =
      deceptiveMaze && mapType === "Maze"
        ? "WallAware" // smarter heuristic only for A* in trap mode
        : heuristicType;

    return buildHeuristic(N, blocks, goal, diag, typeForAStar);
  }, [N, blocks, goal, diag, heuristicType, deceptiveMaze, mapType]);

  // Heuristic for Bidirectional A* — forward search (start -> goal)
  const biAStarForwardHeuristicFn = useMemo(() => {
    const typeForBiAStar =
      deceptiveMaze && mapType === "Maze" ? "WallAware" : heuristicType;

    // same as A*: estimate distance to goal
    return buildHeuristic(N, blocks, goal, diag, typeForBiAStar);
  }, [N, blocks, goal, diag, heuristicType, deceptiveMaze, mapType]);

  // Heuristic for Bidirectional A* — backward search (goal -> start)
  const biAStarBackwardHeuristicFn = useMemo(() => {
    const typeForBiAStar =
      deceptiveMaze && mapType === "Maze" ? "WallAware" : heuristicType;

    // symmetric: now we estimate distance to START instead
    return buildHeuristic(N, blocks, start, diag, typeForBiAStar);
  }, [N, blocks, start, diag, heuristicType, deceptiveMaze, mapType]);

  // Algorithm generators and state
  const gensRef =
    useRef<Record<AlgoKey, Generator<AlgoState, AlgoState, void> | null>>({
      BFS: null,
      Dijkstra: null,
      Greedy: null,
      "A*": null,
      "BiA*": null,
    });

  const [algoStates, setAlgoStates] = useState<
    Record<AlgoKey, AlgoState | null>
  >({
    BFS: null,
    Dijkstra: null,
    Greedy: null,
    "A*": null,
    "BiA*": null,
  });
  const [allFinished, setAllFinished] = useState(false);

  // Initialize generators when map or N changes
  useEffect(() => {
    gensRef.current.BFS = algoBFS(N, blocks, start, goal, diag);
    gensRef.current.Dijkstra = algoDijkstra(N, blocks, start, goal, diag);
    gensRef.current.Greedy = algoGreedy(
      N,
      blocks,
      start,
      goal,
      diag,
      greedyHeuristicFn
    );
    gensRef.current["A*"] = algoAStar(
      N,
      blocks,
      start,
      goal,
      diag,
      aStarHeuristicFn
    );
    gensRef.current["BiA*"] = algoBiAStar(
      N,
      blocks,
      start,
      goal,
      diag,
      biAStarForwardHeuristicFn,
      biAStarBackwardHeuristicFn
    );

    setAlgoStates({
      BFS: null,
      Dijkstra: null,
      Greedy: null,
      "A*": null,
      "BiA*": null,
    });
    setAllFinished(false);
    setRunning(false);
  }, [
    N,
    blocks,
    start,
    goal,
    diag,
    greedyHeuristicFn,
    aStarHeuristicFn,
    biAStarForwardHeuristicFn,
    biAStarBackwardHeuristicFn,
  ]);

  // Animation loop (lockstep)
  useEffect(() => {
    if (!running) return;
    let handle: number;
    let acc = 0;
    const stepInterval = 1000 / speed;
    let last = performance.now();

    const tick = () => {
      const now = performance.now();
      acc += now - last;
      last = now;

      while (acc >= stepInterval) {
        acc -= stepInterval;

        const updates: Partial<Record<AlgoKey, AlgoState>> = {};

        (Object.keys(gensRef.current) as AlgoKey[]).forEach((k) => {
          const g = gensRef.current[k];
          if (!g) return; // already finished

          const res = g.next();
          if (res.done) {
            updates[k] = res.value;
            gensRef.current[k] = null; // mark finished
          } else {
            updates[k] = res.value;
          }
        });

        setAlgoStates((s) => ({ ...s, ...(updates as any) }));

        // Stop only when ALL 5 are finished
        const allDone = Object.values(gensRef.current).every((g) => g === null);
        if (allDone) {
          setAllFinished(true);
          setRunning(false);
          break;
        }
      }

      handle = requestAnimationFrame(tick);
    };

    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [running, speed]);

  // Canvas refs and drawing
  const canvasRefs: Record<AlgoKey, React.RefObject<HTMLCanvasElement>> = {
    BFS: useRef(null),
    Dijkstra: useRef(null),
    Greedy: useRef(null),
    "A*": useRef(null),
    "BiA*": useRef(null),
  };
  useEffect(() => {
    (Object.keys(canvasRefs) as AlgoKey[]).forEach((key) => {
      const cvs = canvasRefs[key].current;
      if (!cvs) return;
      const ctx = (cvs as HTMLCanvasElement).getContext("2d");
      if (!ctx) return;
      drawPanel(ctx, N, sizePx, blocks, algoStates[key], start, goal);
    });
  }, [algoStates, N, blocks, start, goal]);

  const resetRun = () => {
    gensRef.current.BFS = algoBFS(N, blocks, start, goal, diag);
    gensRef.current.Dijkstra = algoDijkstra(N, blocks, start, goal, diag);
    gensRef.current.Greedy = algoGreedy(
      N,
      blocks,
      start,
      goal,
      diag,
      greedyHeuristicFn
    );
    gensRef.current["A*"] = algoAStar(
      N,
      blocks,
      start,
      goal,
      diag,
      aStarHeuristicFn
    );
    gensRef.current["BiA*"] = algoBiAStar(
      N,
      blocks,
      start,
      goal,
      diag,
      biAStarForwardHeuristicFn,
      biAStarBackwardHeuristicFn
    );

    setAlgoStates({
      BFS: null,
      Dijkstra: null,
      Greedy: null,
      "A*": null,
      "BiA*": null,
    });
    setAllFinished(false);
    setRunning(false);
  };

  const winningOrder = useMemo(() => {
    const items = Object.entries(algoStates)
      .map(([k, s]) => ({ key: k as AlgoKey, s }))
      .filter(({ s }) => s && s.finished) // only finished
      .map(({ key, s }) => ({
        key,
        t: s!.lastRuntimeMs || Number.POSITIVE_INFINITY,
        found: !!s!.found,
      }))
      .sort((a, b) => a.t - b.t);

    return items;
  }, [algoStates]);

  useEffect(() => {
    setStart(startId);
    setGoal(goalId);
  }, [startId, goalId]);

  return (
    <div className="min-h-screen">
      <div className="wrapper">
        <header className="mb-6">
          <h1 className="text-3xl font-bold tracking-tight">Pathfinding Lab</h1>
          <p className="text-slate-600">
            Side-by-side BFS · Dijkstra · Greedy · A* · BiA* on configurable
            grids
          </p>
        </header>

        {/* Controls */}
        <div className="controls">
          <div className="control-card">
            <label className="block text-sm mb-1">Grid Size (N×N)</label>
            <input
              type="number"
              value={N}
              min={5}
              max={150}
              onChange={(e) =>
                setN(
                  Math.max(
                    5,
                    Math.min(150, Number(e.target.value) || 16)
                  )
                )
              }
              className="w-full border rounded px-3 py-2"
            />
            <div className="text-xs text-slate-500 mt-1">
              Try 9, 16, 50, 100…
            </div>
          </div>
          <div className="control-card">
            <label className="block text-sm mb-1">Map Type</label>
            <select
              value={mapType}
              onChange={(e) => setMapType(e.target.value as MapType)}
              className="w-full border rounded px-3 py-2"
            >
              <option>Empty</option>
              <option>Random</option>
              <option>Maze</option>
            </select>
            {mapType === "Random" && (
              <div className="mt-3">
                <label className="block text-sm">
                  Obstacle Density: {(density * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min={0}
                  max={0.5}
                  step={0.01}
                  value={density}
                  onChange={(e) => setDensity(Number(e.target.value))}
                  className="w-full"
                />
              </div>
            )}
            <div className="mt-3 flex items-center gap-2">
              <input
                id="diag"
                type="checkbox"
                checked={diag}
                onChange={(e) => setDiag(e.target.checked)}
              />
              <label htmlFor="diag" className="text-sm">
                Allow diagonal moves
              </label>
            </div>
            {mapType === "Maze" && (
              <div className="mt-2 flex items-center gap-2">
                <input
                  id="deceptiveMaze"
                  type="checkbox"
                  checked={deceptiveMaze}
                  onChange={(e) => setDeceptiveMaze(e.target.checked)}
                />
                <label htmlFor="deceptiveMaze" className="text-sm">
                  Deceptive maze (Greedy trap)
                </label>
              </div>
            )}
            {/* Heuristic Selector */}
            <div className="mt-3">
              <label className="block text-sm mb-1">Heuristic</label>
              <select
                value={heuristicType}
                onChange={(e) =>
                  setHeuristicType(e.target.value as HeuristicType)
                }
                className="w-full border rounded px-3 py-2"
              >
                <option value="Manhattan">Manhattan (L1)</option>
                <option value="Euclidean">Euclidean (L2)</option>
                <option value="Chebyshev">Chebyshev (L∞)</option>
                <option value="WallAware">Wall-Aware (experimental)</option>
              </select>
            </div>
          </div>
          <div className="control-card">
            <label className="block text-sm mb-1">Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) =>
                setSeed(Number(e.target.value) || 0)
              }
              className="w-full border rounded px-3 py-2"
            />
            <div className="text-xs text-slate-500 mt-1">
              Reproducible maps
            </div>
            <label className="block text-sm mt-4">
              Speed: {speed} steps/s
            </label>
            <input
              type="range"
              min={1}
              max={60}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="bg-white rounded-2xl shadow p-4 flex flex-col gap-2">
            <button
              onClick={() => setRunning(true)}
              disabled={running}
              className="px-4 py-2 rounded-xl bg-emerald-600 text-white disabled:opacity-50"
            >
              Start
            </button>
            <button
              onClick={() => setRunning(false)}
              className="px-4 py-2 rounded-xl bg-amber-500 text-white"
            >
              Pause
            </button>
            <button
              onClick={resetRun}
              className="px-4 py-2 rounded-xl bg-slate-800 text-white"
            >
              Reset
            </button>
            <div className="text-xs text-slate-500">
              Start: ({Math.floor(start / N)},{start % N}) · Goal: (
              {Math.floor(goal / N)},{goal % N})
            </div>
          </div>
        </div>

        {/* Panels */}
        <div className="panels">
          {(Object.keys(canvasRefs) as AlgoKey[]).map((key) => {
            const s = algoStates[key];
            return (
              <div key={key} className="panel">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="font-semibold">{key}</h2>
                  <div className="text-xs text-slate-500">
                    {s?.found
                      ? "Found"
                      : s?.finished
                      ? "No path"
                      : running
                      ? "Running"
                      : "Idle"}
                  </div>
                </div>
                <canvas
                  ref={canvasRefs[key]}
                  width={sizePx}
                  height={sizePx}
                />
                <div className="stats">
                  <div className="text-slate-500">Expanded</div>
                  <div className="font-mono">
                    {s?.nodesExpanded ?? 0}
                  </div>
                  <div className="text-slate-500">Peak frontier</div>
                  <div className="font-mono">
                    {s?.peakFrontier ?? 0}
                  </div>
                  <div className="text-slate-500">Runtime</div>
                  <div className="font-mono">
                    {s
                      ? s.lastRuntimeMs.toFixed(1) + " ms"
                      : "0.0 ms"}
                  </div>
                  <div className="text-slate-500">Path length</div>
                  <div className="font-mono">
                    {s?.path ? s.path.length : "—"}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Podium / order */}
        <div className="finish-order">
          <h3 className="font-semibold mb-2">Finish Order</h3>
          {winningOrder.length === 0 ? (
            <div className="text-sm text-slate-500">
              No algorithm has finished yet.
            </div>
          ) : (
            <ol className="list-decimal list-inside space-y-1">
              {winningOrder.map((w, i) => (
                <li key={i} className="text-sm">
                  {w.key} —{" "}
                  <span className="font-mono">
                    {isFinite(w.t)
                      ? `${w.t.toFixed(1)} ms`
                      : "—"}
                  </span>
                  {!w.found && (
                    <span className="text-sm"> (no path)</span>
                  )}
                </li>
              ))}
            </ol>
          )}
        </div>

        <footer className="mt-8 text-xs text-slate-500">
          Tips: Use small N (9 or 16) to watch the expansions; then
          scale to 50–100 for performance comparisons. Colors — walls:
          slate-900, visited: amber-200, frontier: blue-200, path:
          green-300, current: red; start: green; goal: violet.
        </footer>
      </div>
    </div>
  );
}

// =====================
// OFFLINE EXPERIMENT HELPERS (no visualization)
// =====================

interface AlgoSummary {
  found: boolean;
  nodesExpanded: number;
  peakFrontier: number;
  lastRuntimeMs: number;
  pathLength: number;
}

// Consume a generator-based algorithm (BFS, Dijkstra, Greedy, A*, BiA*) to completion
function runGeneratorToEnd(
  gen: Generator<AlgoState, AlgoState, void>
): AlgoSummary {
  let last = gen.next();
  while (!last.done) {
    last = gen.next();
  }
  const s = last.value;
  return {
    found: s.found,
    nodesExpanded: s.nodesExpanded,
    peakFrontier: s.peakFrontier,
    lastRuntimeMs: s.lastRuntimeMs,
    pathLength: s.path ? s.path.length : 0,
  };
}

// Bidirectional A* for offline experiments (no visualization generator)
function runBiAStarSummary(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean,
  heuristic: (id: number) => number
): AlgoSummary {
  const begin = performance.now();

  const fwdG = new Float64Array(N * N).fill(Infinity);
  const bwdG = new Float64Array(N * N).fill(Infinity);

  const fwdHeap = new MinHeap<number>();
  const bwdHeap = new MinHeap<number>();

  const fwdOpen = new Set<number>();
  const bwdOpen = new Set<number>();
  const fwdClosed = new Set<number>();
  const bwdClosed = new Set<number>();

  const fwdParents = new Map<number, number | null>();
  const bwdParents = new Map<number, number | null>();

  const fwdSeen = new Uint8Array(N * N);
  const bwdSeen = new Uint8Array(N * N);

  fwdG[start] = 0;
  bwdG[goal] = 0;
  fwdParents.set(start, null);
  bwdParents.set(goal, null);

  fwdHeap.push(heuristic(start), start);
  bwdHeap.push(heuristic(goal), goal);
  fwdOpen.add(start);
  bwdOpen.add(goal);

  let nodesExpanded = 0;
  let peakFrontier = fwdOpen.size + bwdOpen.size;

  const totalFrontierSize = () => fwdOpen.size + bwdOpen.size;

  const reconstructBiPathLength = (meet: number): number => {
    const startToMeet = reconstructPath(fwdParents, meet); // [start,...,meet]
    const goalToMeet = reconstructPath(bwdParents, meet); // [goal,...,meet]
    const meetToGoal = goalToMeet.slice(0, goalToMeet.length - 1).reverse();
    const fullPath = startToMeet.concat(meetToGoal);
    return fullPath.length;
  };

  while (fwdHeap.size() || bwdHeap.size()) {
    peakFrontier = Math.max(peakFrontier, totalFrontierSize());

    const fwdTop = fwdHeap.peekKey();
    const bwdTop = bwdHeap.peekKey();

    const expandForward =
      !bwdHeap.size() ||
      (fwdHeap.size() && (fwdTop ?? Infinity) <= (bwdTop ?? Infinity));

    if (expandForward && fwdHeap.size()) {
      const n = fwdHeap.pop()!;
      fwdOpen.delete(n);
      if (fwdSeen[n]) continue;
      fwdSeen[n] = 1;
      fwdClosed.add(n);
      nodesExpanded++;

      if (bwdClosed.has(n)) {
        const pathLength = reconstructBiPathLength(n);
        const runtime = performance.now() - begin;
        return {
          found: true,
          nodesExpanded,
          peakFrontier,
          lastRuntimeMs: runtime,
          pathLength,
        };
      }

      for (const m of neighbors(N, n, diag)) {
        if (blocks[m] === 1) continue;
        const w =
          diag &&
          rcOf(N, n).r !== rcOf(N, m).r &&
          rcOf(N, n).c !== rcOf(N, m).c
            ? Math.SQRT2
            : 1;
        const ng = fwdG[n] + w;
        if (ng < fwdG[m]) {
          fwdG[m] = ng;
          fwdParents.set(m, n);
          const f = ng + heuristic(m);
          fwdHeap.push(f, m);
          fwdOpen.add(m);
        }
      }
    } else if (bwdHeap.size()) {
      const n = bwdHeap.pop()!;
      bwdOpen.delete(n);
      if (bwdSeen[n]) continue;
      bwdSeen[n] = 1;
      bwdClosed.add(n);
      nodesExpanded++;

      if (fwdClosed.has(n)) {
        const pathLength = reconstructBiPathLength(n);
        const runtime = performance.now() - begin;
        return {
          found: true,
          nodesExpanded,
          peakFrontier,
          lastRuntimeMs: runtime,
          pathLength,
        };
      }

      for (const m of neighbors(N, n, diag)) {
        if (blocks[m] === 1) continue;
        const w =
          diag &&
          rcOf(N, n).r !== rcOf(N, m).r &&
          rcOf(N, n).c !== rcOf(N, m).c
            ? Math.SQRT2
            : 1;
        const ng = bwdG[n] + w;
        if (ng < bwdG[m]) {
          bwdG[m] = ng;
          bwdParents.set(m, n);
          const f = ng + heuristic(m);
          bwdHeap.push(f, m);
          bwdOpen.add(m);
        }
      }
    }
  }

  const runtime = performance.now() - begin;
  return {
    found: false,
    nodesExpanded,
    peakFrontier,
    lastRuntimeMs: runtime,
    pathLength: 0,
  };
}

// =====================
// EXPERIMENT CONFIG & CSV EXPORT
// =====================

type ExperimentConfig = {
  label: string; // e.g. "maze_N50_wallAware_noDiag"
  N: number;
  mapType: MapType; // "Empty" | "Random" | "Maze"
  density?: number; // only for Random
  diag: boolean;
  heuristicType: HeuristicType;
  runs: number; // how many seeds for this config
  deceptiveMaze?: boolean; // optional: use trap-mode maze
};

/**
 * Run many experiments over multiple configs.
 * For each config and each run:
 *   - Generate a map (same logic as the UI)
 *   - Run BFS, Dijkstra, Greedy, A*, BiA* to completion (no drawing)
 *   - Append one CSV row per algorithm
 *
 * Returns a CSV string – you can copy it from console and paste into Excel.
 */
function runPathfindingExperiments(
  configs: ExperimentConfig[],
  baseSeed: number = 1234
): string {
  const rows: string[] = [];

  const header = [
    "configLabel",
    "run",
    "algo",
    "N",
    "mapType",
    "heuristic",
    "diag",
    "density",
    "seed",
    "pathLength",
    "found",
    "nodesExpanded",
    "peakFrontier",
    "runtimeMs",
  ];
  rows.push(header.join(","));

  const addRow = (
    cfg: ExperimentConfig,
    runIdx: number,
    algoName: string,
    s: AlgoSummary,
    seedUsed: number
  ) => {
    rows.push(
      [
        cfg.label,
        String(runIdx),
        algoName,
        String(cfg.N),
        cfg.mapType,
        cfg.heuristicType,
        cfg.diag ? "1" : "0",
        cfg.mapType === "Random" ? String(cfg.density ?? 0) : "",
        String(seedUsed),
        String(s.pathLength),
        s.found ? "yes" : "no",
        String(s.nodesExpanded),
        String(s.peakFrontier),
        s.lastRuntimeMs.toFixed(3),
      ].join(",")
    );
  };

  for (const cfg of configs) {
    for (let runIdx = 1; runIdx <= cfg.runs; runIdx++) {
      const seed =
        baseSeed + runIdx * 997 + cfg.label.length * 13;

      // --------- Map generation (mirror the UI logic, no visualization) ---------
      let b: Uint8Array;
      let s = 0;
      let g = 0;

      if (cfg.mapType === "Empty") {
        b = generateEmpty(cfg.N);
        s = 0;
        g = idOf(cfg.N, cfg.N - 1, cfg.N - 1);
      } else if (cfg.mapType === "Random") {
        s = 0;
        const R = rngLCG(seed + 999);
        let gr = Math.floor((R.next().value as number) * cfg.N);
        let gc = Math.floor((R.next().value as number) * cfg.N);
        let gid = idOf(cfg.N, gr, gc);
        if (gid === s) {
          gr = (gr + 1) % cfg.N;
          gid = idOf(cfg.N, gr, gc);
        }
        g = gid;

        b = generateRandom(cfg.N, cfg.density ?? 0.25, seed, s, g);
        carvePathToGoal(cfg.N, b, s, g, cfg.diag);
      } else {
        // Maze
        b = generateMaze(cfg.N, seed);

        if (cfg.deceptiveMaze) {
          const pair = pickDeceptiveStartGoal(
            cfg.N,
            b,
            cfg.diag,
            seed
          );
          s = pair.start;
          g = pair.goal;
        } else {
          const open: number[] = [];
          for (let i = 0; i < cfg.N * cfg.N; i++) {
            if (b[i] === 0) open.push(i);
          }

          if (open.length >= 2) {
            s = open[0];
            g = open[open.length - 1];
          } else {
            s = 0;
            g = idOf(cfg.N, cfg.N - 1, cfg.N - 1);
            b[s] = 0;
            b[g] = 0;
          }
        }
      }

      // Heuristic for Greedy / A* / BiA*
      const heuristicFn = buildHeuristic(
        cfg.N,
        b,
        g,
        cfg.diag,
        cfg.heuristicType
      );

      // --------- Run all 5 algorithms ---------
      const bfsSummary = runGeneratorToEnd(
        algoBFS(cfg.N, b, s, g, cfg.diag)
      );
      const dijkSummary = runGeneratorToEnd(
        algoDijkstra(cfg.N, b, s, g, cfg.diag)
      );
      const greedySummary = runGeneratorToEnd(
        algoGreedy(cfg.N, b, s, g, cfg.diag, heuristicFn)
      );
      const aStarSummary = runGeneratorToEnd(
        algoAStar(cfg.N, b, s, g, cfg.diag, heuristicFn)
      );
      const biAStarSummary = runBiAStarSummary(
        cfg.N,
        b,
        s,
        g,
        cfg.diag,
        heuristicFn
      );

      // --------- Log rows ---------
      addRow(cfg, runIdx, "BFS", bfsSummary, seed);
      addRow(cfg, runIdx, "Dijkstra", dijkSummary, seed);
      addRow(cfg, runIdx, "Greedy", greedySummary, seed);
      addRow(cfg, runIdx, "A*", aStarSummary, seed);
      addRow(cfg, runIdx, "BiA*", biAStarSummary, seed);
    }
  }

  return rows.join("\n");
}
