// experiments/experiment-runner.ts
//
// Offline experiments for your Pathfinding Lab.
// Runs BFS, Dijkstra, Greedy, A*, BiA* on many random/maze maps
// and writes a CSV file with timings, expansions, frontier size, path length, optimality, etc.
//
// Run with:
//   npx ts-node experiments/experiment-runner.ts
//
// CSV output: experiments/results.csv

import { writeFileSync } from "fs";
import { performance } from "perf_hooks";

// ---------- Types ----------
type Cell = { r: number; c: number };

type MapType = "Empty" | "Random" | "Maze";

type AlgoKey = "BFS" | "Dijkstra" | "Greedy" | "A*" | "BiA*";

type HeuristicType = "Manhattan" | "Chebyshev" | "Euclidean" | "WallAware";

interface AlgoResult {
  algo: AlgoKey;
  heuristic: HeuristicType | "None";
  runtimeMs: number;
  nodesExpanded: number;
  peakFrontier: number;
  pathLength: number | null;
  found: boolean;
  optimal: boolean | null; // null if no baseline
}

interface MapConfig {
  N: number;
  mapType: MapType;
  density: number; // for Random
  seed: number;
  diag: boolean;
  heuristicType: HeuristicType;
  deceptiveMaze: boolean;
}

// ---------- Experiment parameters (EDIT THESE AS YOU LIKE) ----------
const OUTPUT_CSV = "experiments/results_general.csv";

// how many seeds per configuration
const NUM_TRIALS = 100;

// grid sizes to test
// const NS = [1024, 4048]; // e.g. [16, 32, 64, 128] if you want bigger

const NS = [64,128];
// map types to test
// const MAP_TYPES: MapType[] = ["Random", "Maze"];

const MAP_TYPES: MapType[] = ["Maze","Random","Empty"];


// densities for Random maps
const DENSITIES = [0.20,0.35];

// heuristics to test
const HEURISTICS: HeuristicType[] = ["Manhattan", "WallAware","Euclidean","Chebyshev"];

// use diagonals or not (for simplicity: keep false if you want uniform edge costs)
const DIAG = false;


// whether to use "deceptive" mazes that trap Greedy
const USE_DECEPTIVE_MAZE = true;

// ---------- Utilities ----------
const idOf = (N: number, r: number, c: number) => r * N + c;
const rcOf = (N: number, id: number): Cell => ({ r: Math.floor(id / N), c: id % N });

function* rngLCG(seed: number) {
  let s = (seed >>> 0) || 1;
  while (true) {
    s = (1664525 * s + 1013904223) >>> 0;
    yield s / 2 ** 32;
  }
}

class MinHeap<T> {
  private a: { k: number; v: T }[] = [];
  size() { return this.a.length; }
  push(k: number, v: T) { this.a.push({ k, v }); this.bubbleUp(this.a.length - 1); }
  pop(): T | undefined {
    if (this.a.length === 0) return undefined;
    const top = this.a[0].v;
    const last = this.a.pop()!;
    if (this.a.length) { this.a[0] = last; this.bubbleDown(0); }
    return top;
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
      let l = i * 2 + 1, r = l + 1, m = i;
      if (l < n && this.a[l].k < this.a[m].k) m = l;
      if (r < n && this.a[r].k < this.a[m].k) m = r;
      if (m === i) break;
      [this.a[m], this.a[i]] = [this.a[i], this.a[m]];
      i = m;
    }
  }
}

// Heuristics (as in the app)
const manhattan = (N: number, a: number, b: number) => {
  const A = rcOf(N, a), B = rcOf(N, b);
  return Math.abs(A.r - B.r) + Math.abs(A.c - B.c);
};
const chebyshev = (N: number, a: number, b: number) => {
  const A = rcOf(N, a), B = rcOf(N, b);
  return Math.max(Math.abs(A.r - B.r), Math.abs(A.c - B.c));
};
const euclidean = (N: number, a: number, b: number) => {
  const A = rcOf(N, a), B = rcOf(N, b);
  const dr = A.r - B.r;
  const dc = A.c - B.c;
  return Math.sqrt(dr * dr + dc * dc);
};

// Neighbors
function neighbors(N: number, id: number, diag: boolean) {
  const { r, c } = rcOf(N, id);
  const deltas4 = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  const deltas8 = diag ? deltas4.concat([[1, 1], [1, -1], [-1, 1], [-1, -1]]) : deltas4;
  const out: number[] = [];
  for (const [dr, dc] of deltas8) {
    const nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < N && nc >= 0 && nc < N) out.push(idOf(N, nr, nc));
  }
  return out;
}

function reconstructPath(parents: Map<number, number | null>, goal: number): number[] {
  const path: number[] = [];
  let cur: number | null | undefined = goal;
  while (cur != null) { path.push(cur); cur = parents.get(cur)!; }
  return path.reverse();
}

// ---------- Map Generation ----------
function generateEmpty(N: number) {
  return new Uint8Array(N * N); // all 0 (free)
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
    const p = (R.next().value as number);
    blocks[i] = p < density ? 1 : 0;
  }
  blocks[start] = 0;
  blocks[goal] = 0;
  return blocks;
}

// Maze via randomized DFS backtracker (same idea as app)
function generateMaze(N: number, seed: number) {
  const blocks = new Uint8Array(N * N).fill(1);
  const R = rngLCG(seed);

  const inBoundsCell = (r: number, c: number) =>
    r > 0 && r < N - 1 && c > 0 && c < N - 1 && r % 2 === 1 && c % 2 === 1;

  const visited = new Uint8Array(N * N);
  const stack: Cell[] = [];

  let sr = 1, sc = 1;
  if (N <= 2) { sr = 0; sc = 0; }

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

    // shuffle directions
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

      const wr = cur.r + dr / 2;
      const wc = cur.c + dc / 2;
      blocks[idOf(N, wr, wc)] = 0;

      stack.push({ r: nr, c: nc });
      moved = true;
      break;
    }

    if (!moved) stack.pop();
  }

  return blocks;
}

function solvable(N: number, blocks: Uint8Array, start: number, goal: number, diag: boolean) {
  const q: number[] = [start];
  const seen = new Uint8Array(N * N); seen[start] = 1;
  while (q.length) {
    const x = q.shift()!;
    if (x === goal) return true;
    for (const y of neighbors(N, x, diag)) {
      if (!seen[y] && blocks[y] === 0) {
        seen[y] = 1;
        q.push(y);
      }
    }
  }
  return false;
}

// For deceptive maze runs: pick start/goal that mislead Greedy
function pickDeceptiveStartGoal(
  N: number,
  blocks: Uint8Array,
  diag: boolean,
  seed: number
): { start: number; goal: number } {
  const free: number[] = [];
  for (let i = 0; i < N * N; i++) {
    if (blocks[i] === 0) free.push(i);
  }
  if (free.length < 2) {
    return { start: 0, goal: idOf(N, N - 1, N - 1) };
  }

  const R = rngLCG(seed + 4242);
  const start = free[Math.floor((R.next().value as number) * free.length)];

  const dist = new Int32Array(N * N).fill(-1);
  const q: number[] = [];
  dist[start] = 0;
  q.push(start);

  while (q.length) {
    const v = q.shift()!;
    for (const nb of neighbors(N, v, diag)) {
      if (blocks[nb] === 1) continue;
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
    if (dist[cell] === -1) continue;

    let freeNb = 0;
    for (const nb of neighbors(N, cell, diag)) {
      if (blocks[nb] === 0) freeNb++;
    }
    const isDeadEnd = freeNb === 1;

    const h = hFun(N, start, cell);
    const d = dist[cell];

    const gap = d - h;
    if (gap <= 0) continue;

    const score = gap + (isDeadEnd ? 5 : 0);

    if (score > bestScore) {
      bestScore = score;
      bestGoal = cell;
    }
  }

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

// ---------- Heuristic builder ----------
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

  if (type === "WallAware") {
    const base = diag ? baseChebyshev : baseManhattan;
    const lambda = 1.2;
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

// ---------- Algorithms (non-visual versions) ----------

function runBFS(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean
): AlgoResult {
  const openQ: number[] = [start];
  const inQ = new Uint8Array(N * N); inQ[start] = 1;
  const closed = new Set<number>();
  const parents = new Map<number, number | null>(); parents.set(start, null);
  const meta = { nodesExpanded: 0, peakFrontier: 1 };
  const begin = performance.now();

  while (openQ.length) {
    meta.peakFrontier = Math.max(meta.peakFrontier, openQ.length);
    const n = openQ.shift()!;
    inQ[n] = 0;
    closed.add(n);
    meta.nodesExpanded++;

    if (n === goal) {
      const path = reconstructPath(parents, goal);
      return {
        algo: "BFS",
        heuristic: "None",
        runtimeMs: performance.now() - begin,
        nodesExpanded: meta.nodesExpanded,
        peakFrontier: meta.peakFrontier,
        pathLength: path.length,
        found: true,
        optimal: null, // set later
      };
    }

    for (const m of neighbors(N, n, diag)) {
      if (blocks[m] === 1) continue;
      if (!parents.has(m)) {
        parents.set(m, n);
        openQ.push(m); inQ[m] = 1;
      }
    }
  }

  return {
    algo: "BFS",
    heuristic: "None",
    runtimeMs: performance.now() - begin,
    nodesExpanded: meta.nodesExpanded,
    peakFrontier: meta.peakFrontier,
    pathLength: null,
    found: false,
    optimal: null,
  };
}

function runDijkstra(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean
): AlgoResult {
  const heap = new MinHeap<number>();
  const g = new Float64Array(N * N).fill(Infinity); g[start] = 0;
  const seen = new Uint8Array(N * N);
  const parents = new Map<number, number | null>(); parents.set(start, null);
  const open = new Set<number>();
  const closed = new Set<number>();
  const meta = { nodesExpanded: 0, peakFrontier: 0 };
  const begin = performance.now();

  heap.push(0, start); open.add(start);
  while (heap.size()) {
    const n = heap.pop()!; open.delete(n);
    if (seen[n]) continue; seen[n] = 1; closed.add(n);
    meta.nodesExpanded++;
    meta.peakFrontier = Math.max(meta.peakFrontier, open.size);

    if (n === goal) {
      const path = reconstructPath(parents, goal);
      return {
        algo: "Dijkstra",
        heuristic: "None",
        runtimeMs: performance.now() - begin,
        nodesExpanded: meta.nodesExpanded,
        peakFrontier: meta.peakFrontier,
        pathLength: path.length,
        found: true,
        optimal: null,
      };
    }

    for (const m of neighbors(N, n, diag)) {
      if (blocks[m] === 1) continue;
      const w = 1; // keep unweighted for clean comparison
      const ng = g[n] + w;
      if (ng < g[m]) {
        g[m] = ng;
        parents.set(m, n);
        heap.push(ng, m);
        open.add(m);
      }
    }
  }

  return {
    algo: "Dijkstra",
    heuristic: "None",
    runtimeMs: performance.now() - begin,
    nodesExpanded: meta.nodesExpanded,
    peakFrontier: meta.peakFrontier,
    pathLength: null,
    found: false,
    optimal: null,
  };
}

function runGreedy(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean,
  heuristic: (id: number) => number,
  heuristicType: HeuristicType
): AlgoResult {
  const heap = new MinHeap<number>();
  const seen = new Uint8Array(N * N);
  const parents = new Map<number, number | null>(); parents.set(start, null);
  const open = new Set<number>();
  const closed = new Set<number>();
  const begin = performance.now();
  const meta = { nodesExpanded: 0, peakFrontier: 0 };

  heap.push(heuristic(start), start); open.add(start);

  while (heap.size()) {
    const n = heap.pop()!;
    open.delete(n);
    if (seen[n]) continue;
    seen[n] = 1;
    closed.add(n);
    meta.nodesExpanded++;
    meta.peakFrontier = Math.max(meta.peakFrontier, open.size);

    if (n === goal) {
      const path = reconstructPath(parents, goal);
      return {
        algo: "Greedy",
        heuristic: heuristicType,
        runtimeMs: performance.now() - begin,
        nodesExpanded: meta.nodesExpanded,
        peakFrontier: meta.peakFrontier,
        pathLength: path.length,
        found: true,
        optimal: null,
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
  }

  return {
    algo: "Greedy",
    heuristic: heuristicType,
    runtimeMs: performance.now() - begin,
    nodesExpanded: meta.nodesExpanded,
    peakFrontier: meta.peakFrontier,
    pathLength: null,
    found: false,
    optimal: null,
  };
}

function runAStar(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean,
  heuristic: (id: number) => number,
  heuristicType: HeuristicType
): AlgoResult {
  const heap = new MinHeap<number>();
  const g = new Float64Array(N * N).fill(Infinity); g[start] = 0;
  const seen = new Uint8Array(N * N);
  const parents = new Map<number, number | null>(); parents.set(start, null);
  const open = new Set<number>();
  const closed = new Set<number>();
  const begin = performance.now();
  const meta = { nodesExpanded: 0, peakFrontier: 0 };

  heap.push(heuristic(start), start); open.add(start);

  while (heap.size()) {
    const n = heap.pop()!;
    open.delete(n);
    if (seen[n]) continue;
    seen[n] = 1;
    closed.add(n);
    meta.nodesExpanded++;
    meta.peakFrontier = Math.max(meta.peakFrontier, open.size);

    if (n === goal) {
      const path = reconstructPath(parents, goal);
      return {
        algo: "A*",
        heuristic: heuristicType,
        runtimeMs: performance.now() - begin,
        nodesExpanded: meta.nodesExpanded,
        peakFrontier: meta.peakFrontier,
        pathLength: path.length,
        found: true,
        optimal: null,
      };
    }

    for (const m of neighbors(N, n, diag)) {
      if (blocks[m] === 1) continue;
      const w = 1;
      const ng = g[n] + w;
      if (ng < g[m]) {
        g[m] = ng;
        parents.set(m, n);
        const f = ng + heuristic(m);
        heap.push(f, m);
        open.add(m);
      }
    }
  }

  return {
    algo: "A*",
    heuristic: heuristicType,
    runtimeMs: performance.now() - begin,
    nodesExpanded: meta.nodesExpanded,
    peakFrontier: meta.peakFrontier,
    pathLength: null,
    found: false,
    optimal: null,
  };
}

function runBiAStar(
  N: number,
  blocks: Uint8Array,
  start: number,
  goal: number,
  diag: boolean,
  hForward: (id: number) => number,
  hBackward: (id: number) => number,
  heuristicType: HeuristicType
): AlgoResult {
  const heapF = new MinHeap<number>();
  const heapB = new MinHeap<number>();

  const gF = new Float64Array(N * N).fill(Infinity);
  const gB = new Float64Array(N * N).fill(Infinity);
  gF[start] = 0;
  gB[goal] = 0;

  const seenF = new Uint8Array(N * N);
  const seenB = new Uint8Array(N * N);

  const parentsF = new Map<number, number | null>();
  const parentsB = new Map<number, number | null>();
  parentsF.set(start, null);
  parentsB.set(goal, null);

  const openF = new Set<number>();
  const openB = new Set<number>();
  const closedF = new Set<number>();
  const closedB = new Set<number>();

  const begin = performance.now();
  const meta = { nodesExpanded: 0, peakFrontier: 0 };

  heapF.push(hForward(start), start); openF.add(start);
  heapB.push(hBackward(goal), goal); openB.add(goal);

  let meeting: number | null = null;

  const reconstructBiPath = (): number[] => {
    if (meeting === null) return [];
    const pathF = reconstructPath(parentsF, meeting);
    const pathB = reconstructPath(parentsB, meeting);
    pathB.reverse();
    return pathF.concat(pathB.slice(1));
  };

  while (heapF.size() || heapB.size()) {
    // forward
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
          return {
            algo: "BiA*",
            heuristic: heuristicType,
            runtimeMs: performance.now() - begin,
            nodesExpanded: meta.nodesExpanded,
            peakFrontier: meta.peakFrontier,
            pathLength: path.length || null,
            found: path.length > 0,
            optimal: null,
          };
        }

        for (const m of neighbors(N, n, diag)) {
          if (blocks[m] === 1) continue;
          const w = 1;
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

    // backward
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
          return {
            algo: "BiA*",
            heuristic: heuristicType,
            runtimeMs: performance.now() - begin,
            nodesExpanded: meta.nodesExpanded,
            peakFrontier: meta.peakFrontier,
            pathLength: path.length || null,
            found: path.length > 0,
            optimal: null,
          };
        }

        for (const m of neighbors(N, n, diag)) {
          if (blocks[m] === 1) continue;
          const w = 1;
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

    const totalOpen = openF.size + openB.size;
    meta.peakFrontier = Math.max(meta.peakFrontier, totalOpen);
  }

  return {
    algo: "BiA*",
    heuristic: heuristicType,
    runtimeMs: performance.now() - begin,
    nodesExpanded: meta.nodesExpanded,
    peakFrontier: meta.peakFrontier,
    pathLength: null,
    found: false,
    optimal: null,
  };
}

// ---------- Build one map config & run all algorithms ----------
function buildMap(config: MapConfig) {
  const { N, mapType, density, seed, diag, heuristicType, deceptiveMaze } = config;
  let blocks: Uint8Array;
  let start = 0;
  let goal = 0;

  if (mapType === "Empty") {
    blocks = generateEmpty(N);
    start = 0;
    goal = idOf(N, N - 1, N - 1);
  } else if (mapType === "Random") {
    start = 0;
    goal = idOf(N, N - 1, N - 1);
    let attempt = 0;
    do {
      blocks = generateRandom(N, density, seed + attempt, start, goal);
      attempt++;
    } while (attempt < 30 && !solvable(N, blocks!, start, goal, diag));
  } else {
    blocks = generateMaze(N, seed);
    if (deceptiveMaze) {
      const pair = pickDeceptiveStartGoal(N, blocks, diag, seed);
      start = pair.start;
      goal = pair.goal;
    } else {
      const open: number[] = [];
      for (let i = 0; i < N * N; i++) {
        if (blocks[i] === 0) open.push(i);
      }
      if (open.length >= 2) {
        start = open[0];
        goal = open[open.length - 1];
      } else {
        start = 0;
        goal = idOf(N, N - 1, N - 1);
        blocks[start] = 0;
        blocks[goal] = 0;
      }
    }
  }

  return { blocks: blocks!, start, goal };
}

function runAllAlgorithms(config: MapConfig): { mapInfo: any; results: AlgoResult[] } {
  const { N, mapType, density, seed, diag, heuristicType, deceptiveMaze } = config;
  const { blocks, start, goal } = buildMap(config);

  // Baseline: Dijkstra (true shortest path for unit-weight grid)
  const dijkstraRes = runDijkstra(N, blocks, start, goal, diag);
  const baseline = dijkstraRes.found ? dijkstraRes.pathLength! : null;

  // BFS
  const bfsRes = runBFS(N, blocks, start, goal, diag);

  // Greedy heuristic choice (can mimic your app logic)
  const greedyHeurType =
    deceptiveMaze && mapType === "Maze" ? "Manhattan" : heuristicType;
  const greedyH = buildHeuristic(N, blocks, goal, diag, greedyHeurType);
  const greedyRes = runGreedy(N, blocks, start, goal, diag, greedyH, greedyHeurType);

  // A*
  const aStarHeurType =
    deceptiveMaze && mapType === "Maze" ? "WallAware" : heuristicType;
  const aStarH = buildHeuristic(N, blocks, goal, diag, aStarHeurType);
  const aStarRes = runAStar(N, blocks, start, goal, diag, aStarH, aStarHeurType);

  // BiA*
  const biForwardH = buildHeuristic(N, blocks, goal, diag, aStarHeurType);
  const biBackwardH = buildHeuristic(N, blocks, start, diag, aStarHeurType);
  const biRes = runBiAStar(
    N,
    blocks,
    start,
    goal,
    diag,
    biForwardH,
    biBackwardH,
    aStarHeurType
  );

  const all = [bfsRes, dijkstraRes, greedyRes, aStarRes, biRes];

  // Mark optimality vs Dijkstra baseline
  if (baseline !== null) {
    for (const r of all) {
      if (!r.found || r.pathLength == null) {
        r.optimal = false;
      } else {
        r.optimal = r.pathLength === baseline;
      }
    }
  } else {
    for (const r of all) r.optimal = null;
  }

  return {
    mapInfo: { N, mapType, density, seed, diag, heuristicType, deceptiveMaze },
    results: all,
  };
}

// ---------- Main experiment loop ----------
function main() {
  const rows: string[] = [];
  rows.push([
    "trial",
    "N",
    "mapType",
    "density",
    "seed",
    "diag",
    "heuristicUI",
    "deceptiveMaze",
    "algo",
    "algoHeuristic",
    "runtimeMs",
    "nodesExpanded",
    "peakFrontier",
    "pathLength",
    "found",
    "optimal",
  ].join(","));

  let trialIndex = 0;

  for (const N of NS) {
    for (const mapType of MAP_TYPES) {
      for (const density of (mapType === "Random" ? DENSITIES : [0])) {
        for (const heuristicType of HEURISTICS) {
          for (let t = 0; t < NUM_TRIALS; t++) {
            const seed = 1000 * trialIndex + t;
            const config: MapConfig = {
              N,
              mapType,
              density,
              seed,
              diag: DIAG,
              heuristicType,
              deceptiveMaze: USE_DECEPTIVE_MAZE && mapType === "Maze",
            };

            const { mapInfo, results } = runAllAlgorithms(config);

            for (const r of results) {
              rows.push([
                trialIndex.toString(),
                mapInfo.N.toString(),
                mapInfo.mapType,
                mapInfo.density.toString(),
                mapInfo.seed.toString(),
                mapInfo.diag ? "1" : "0",
                mapInfo.heuristicType,
                mapInfo.deceptiveMaze ? "1" : "0",
                r.algo,
                r.heuristic,
                r.runtimeMs.toFixed(4),
                r.nodesExpanded.toString(),
                r.peakFrontier.toString(),
                r.pathLength == null ? "" : r.pathLength.toString(),
                r.found ? "1" : "0",
                r.optimal == null ? "" : (r.optimal ? "1" : "0"),
              ].join(","));
            }

            trialIndex++;
            console.log(
              `Done trial ${trialIndex} :: N=${N}, map=${mapType}, density=${density}, heuristicUI=${heuristicType}, seed=${seed}`
            );
          }
        }
      }
    }
  }

  writeFileSync(OUTPUT_CSV, rows.join("\n"), "utf8");
  console.log(`\n✅ Wrote ${rows.length - 1} rows to ${OUTPUT_CSV}`);
}

main();
