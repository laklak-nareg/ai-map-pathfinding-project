import React, { useEffect, useMemo, useRef, useState } from "react";

// =====================
// Pathfinding Lab — Single-file React app
// - Side-by-side visualization of BFS, Dijkstra, Greedy, and A*
// - Configurable grid size, map type (Empty / Random / Maze), density, seed
// - Lockstep animation so each algorithm advances one expansion per tick
// - Canvas rendering for speed
// =====================

// ---------- Types ----------
type Cell = { r: number; c: number };

type MapType = "Empty" | "Random" | "Maze";

type AlgoKey = "BFS" | "Dijkstra" | "Greedy" | "A*";

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
const rcOf = (N: number, id: number): Cell => ({ r: Math.floor(id / N), c: id % N });

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
  size() { return this.a.length; }
  push(k: number, v: T) { this.a.push({ k, v }); this.bubbleUp(this.a.length - 1); }
  pop(): T | undefined {
    if (this.a.length === 0) return undefined;
    const top = this.a[0].v;
    const last = this.a.pop()!;
    if (this.a.length) { this.a[0] = last; this.bubbleDown(0); }
    return top;
  }
  peekKey(): number | undefined { return this.a[0]?.k; }
  private bubbleUp(i: number) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.a[p].k <= this.a[i].k) break;
      [this.a[p], this.a[i]] = [this.a[i], this.a[p]]; i = p;
    }
  }
  private bubbleDown(i: number) {
    const n = this.a.length;
    while (true) {
      let l = i * 2 + 1, r = l + 1, m = i;
      if (l < n && this.a[l].k < this.a[m].k) m = l;
      if (r < n && this.a[r].k < this.a[m].k) m = r;
      if (m === i) break; [this.a[m], this.a[i]] = [this.a[i], this.a[m]]; i = m;
    }
  }
}

// Manhattan / Chebyshev heuristics
const manhattan = (N: number, a: number, b: number) => {
  const A = rcOf(N, a), B = rcOf(N, b);
  return Math.abs(A.r - B.r) + Math.abs(A.c - B.c);
};
const chebyshev = (N: number, a: number, b: number) => {
  const A = rcOf(N, a), B = rcOf(N, b);
  return Math.max(Math.abs(A.r - B.r), Math.abs(A.c - B.c));
};

// Neighbors (4- or 8-connected)
function neighbors(N: number, id: number, diag: boolean) {
  const { r, c } = rcOf(N, id);
  const deltas4 = [ [1,0],[ -1,0],[0,1],[0,-1] ];
  const deltas8 = diag ? deltas4.concat([ [1,1],[1,-1],[-1,1],[-1,-1] ]) : deltas4;
  const out: number[] = [];
  for (const [dr, dc] of deltas8) {
    const nr = r + dr, nc = c + dc;
    if (nr>=0 && nr<N && nc>=0 && nc<N) out.push(idOf(N,nr,nc));
  }
  return out;
}

// Reconstruct path from parents
function reconstructPath(parents: Map<number, number | null>, goal: number): number[] {
  const path: number[] = [];
  let cur: number | null | undefined = goal;
  while (cur != null) { path.push(cur); cur = parents.get(cur)!; }
  return path.reverse();
}

// ---------- Map Generation ----------
function generateEmpty(N: number) {
  const blocks = new Uint8Array(N * N); // 0 free, 1 wall
  return blocks;
}

function generateRandom(N: number, density: number, seed: number, start: number, goal: number) {
  const blocks = new Uint8Array(N * N);
  const R = rngLCG(seed);
  for (let i = 0; i < N * N; i++) {
    const p = (R.next().value as number);
    blocks[i] = p < density ? 1 : 0;
  }
  blocks[start] = 0; blocks[goal] = 0;
  return blocks;
}

// Maze via DFS backtracker on cell grid (treat cells, carve passages)
function generateMaze(N: number, seed: number, start: number, goal: number) {
  // We create a perfect maze by starting with walls and carving passages between cells.
  // Implementation: use a grid of cells and carve in 4-direction. For simplicity, we operate directly on cell blocks.
  const blocks = new Uint8Array(N * N).fill(1); // start all walls
  const R = rngLCG(seed);
  const inBounds = (r: number, c: number) => r>=0 && r<N && c>=0 && c<N;
  const visited = new Uint8Array(N * N);
  // choose odd-only positions as carveable cells for thicker walls if N large; here carve every cell via random DFS.
  const stack: Cell[] = [];
  const sr = 0, sc = 0; // start carving from 0,0
  stack.push({ r: sr, c: sc });
  visited[idOf(N,sr,sc)] = 1; blocks[idOf(N,sr,sc)] = 0;
  const order = [ [1,0],[-1,0],[0,1],[0,-1] ];
  while (stack.length) {
    const cur = stack[stack.length-1];
    // shuffle directions
    for (let i=order.length-1;i>0;i--) {
      const j = Math.floor(((R.next().value as number))*(i+1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    let moved = false;
    for (const [dr,dc] of order) {
      const nr = cur.r + dr, nc = cur.c + dc;
      if (!inBounds(nr,nc)) continue;
      const id = idOf(N,nr,nc);
      if (!visited[id]) {
        visited[id] = 1; blocks[id] = 0; stack.push({ r: nr, c: nc }); moved = true; break;
      }
    }
    if (!moved) stack.pop();
  }
  blocks[start] = 0; blocks[goal] = 0;
  return blocks;
}

// Connectivity check via quick BFS (to avoid unsolvable randoms);
function solvable(N: number, blocks: Uint8Array, start: number, goal: number, diag: boolean) {
  const q: number[] = [start];
  const seen = new Uint8Array(N*N); seen[start] = 1;
  while (q.length) {
    const x = q.shift()!;
    if (x === goal) return true;
    for (const y of neighbors(N,x,diag)) if (!seen[y] && blocks[y]===0) { seen[y]=1; q.push(y); }
  }
  return false;
}

// ---------- Algorithm Step Generators ----------
// Each returns a generator that yields one expansion per step; on completion it returns final AlgoState fields.

function* algoBFS(N: number, blocks: Uint8Array, start: number, goal: number, diag: boolean) {
  const openQ: number[] = [start];
  const inQ = new Uint8Array(N*N); inQ[start] = 1;
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
      return { ...cur, found: true, finished: true, path, lastRuntimeMs: performance.now() - begin };
    }
    for (const m of neighbors(N, n, diag)) {
      if (blocks[m]===1) continue;
      if (!parents.has(m)) {
        parents.set(m, n);
        openQ.push(m); inQ[m] = 1;
      }
    }
  }
  return {
    key: "BFS", open: new Set(), closed: closed, parents, found: false, finished: true,
    nodesExpanded: meta.nodesExpanded, peakFrontier: meta.peakFrontier, lastRuntimeMs:  performance.now() - begin
  } as AlgoState;
}

function* algoDijkstra(N: number, blocks: Uint8Array, start: number, goal: number, diag: boolean) {
  const heap = new MinHeap<number>();
  const g = new Float64Array(N*N).fill(Infinity); g[start]=0;
  const seen = new Uint8Array(N*N);
  const parents = new Map<number, number | null>(); parents.set(start,null);
  const open = new Set<number>();
  const closed = new Set<number>();
  const meta = { nodesExpanded: 0, peakFrontier: 0 };
  const begin = performance.now();

  heap.push(0, start); open.add(start);
  while (heap.size()) {
    const n = heap.pop()!; open.delete(n);
    if (seen[n]) continue; seen[n]=1; closed.add(n);
    meta.nodesExpanded++;

    const cur: AlgoState = {
      key: "Dijkstra", open: new Set(open), closed: new Set(closed), parents,
      found: false, finished: false, current: n,
      nodesExpanded: meta.nodesExpanded, peakFrontier: Math.max(meta.peakFrontier, open.size), lastRuntimeMs: performance.now()-begin
    };
    yield cur;

    if (n===goal) {
      const path = reconstructPath(parents, goal);
      return { ...cur, found: true, finished: true, path, lastRuntimeMs: performance.now()-begin };
    }
    for (const m of neighbors(N,n,diag)) {
      if (blocks[m]===1) continue; const w = (diag && (rcOf(N,n).r!==rcOf(N,m).r && rcOf(N,n).c!==rcOf(N,m).c)) ? Math.SQRT2 : 1;
      const ng = g[n] + w; if (ng < g[m]) { g[m]=ng; parents.set(m,n); heap.push(ng, m); open.add(m); }
    }
    meta.peakFrontier = Math.max(meta.peakFrontier, open.size);
  }
  return { key: "Dijkstra", open:new Set(), closed, parents, found:false, finished:true, nodesExpanded:meta.nodesExpanded, peakFrontier:meta.peakFrontier, lastRuntimeMs: performance.now()-begin } as AlgoState;
}

function* algoGreedy(N: number, blocks: Uint8Array, start: number, goal: number, diag: boolean) {
  const heap = new MinHeap<number>();
  const hFun = diag ? chebyshev : manhattan;
  const seen = new Uint8Array(N*N);
  const parents = new Map<number, number | null>(); parents.set(start,null);
  const open = new Set<number>();
  const closed = new Set<number>();
  const begin = performance.now();
  const meta = { nodesExpanded: 0, peakFrontier: 0 };

  heap.push(hFun(N,start,goal), start); open.add(start);
  while (heap.size()) {
    const n = heap.pop()!; open.delete(n);
    if (seen[n]) continue; seen[n]=1; closed.add(n);
    meta.nodesExpanded++;

    const cur: AlgoState = { key: "Greedy", open: new Set(open), closed: new Set(closed), parents, found:false, finished:false, current:n, nodesExpanded:meta.nodesExpanded, peakFrontier:Math.max(meta.peakFrontier, open.size), lastRuntimeMs: performance.now()-begin };
    yield cur;

    if (n===goal) {
      const path = reconstructPath(parents, goal);
      return { ...cur, found: true, finished: true, path, lastRuntimeMs: performance.now()-begin };
    }
    for (const m of neighbors(N,n,diag)) {
      if (blocks[m]===1) continue;
      if (!seen[m]) { parents.set(m,n); heap.push(hFun(N,m,goal), m); open.add(m); }
    }
    meta.peakFrontier = Math.max(meta.peakFrontier, open.size);
  }
  return { key: "Greedy", open:new Set(), closed, parents, found:false, finished:true, nodesExpanded:meta.nodesExpanded, peakFrontier:meta.peakFrontier, lastRuntimeMs: performance.now()-begin } as AlgoState;
}

function* algoAStar(N: number, blocks: Uint8Array, start: number, goal: number, diag: boolean) {
  const heap = new MinHeap<number>();
  const hFun = diag ? chebyshev : manhattan;
  const g = new Float64Array(N*N).fill(Infinity); g[start]=0;
  const seen = new Uint8Array(N*N);
  const parents = new Map<number, number | null>(); parents.set(start,null);
  const open = new Set<number>();
  const closed = new Set<number>();
  const begin = performance.now();
  const meta = { nodesExpanded: 0, peakFrontier: 0 };

  heap.push(hFun(N,start,goal), start); open.add(start);
  while (heap.size()) {
    const n = heap.pop()!; open.delete(n);
    if (seen[n]) continue; seen[n]=1; closed.add(n);
    meta.nodesExpanded++;

    const cur: AlgoState = { key: "A*", open: new Set(open), closed: new Set(closed), parents, found:false, finished:false, current:n, nodesExpanded:meta.nodesExpanded, peakFrontier:Math.max(meta.peakFrontier, open.size), lastRuntimeMs: performance.now()-begin };
    yield cur;

    if (n===goal) {
      const path = reconstructPath(parents, goal);
      return { ...cur, found: true, finished: true, path, lastRuntimeMs: performance.now()-begin };
    }
    for (const m of neighbors(N,n,diag)) {
      if (blocks[m]===1) continue; const w = (diag && (rcOf(N,n).r!==rcOf(N,m).r && rcOf(N,n).c!==rcOf(N,m).c)) ? Math.SQRT2 : 1;
      const ng = g[n] + w;
      if (ng < g[m]) { g[m]=ng; parents.set(m,n); const f = ng + hFun(N,m,goal); heap.push(f, m); open.add(m); }
    }
    meta.peakFrontier = Math.max(meta.peakFrontier, open.size);
  }
  return { key: "A*", open:new Set(), closed, parents, found:false, finished:true, nodesExpanded:meta.nodesExpanded, peakFrontier:meta.peakFrontier, lastRuntimeMs: performance.now()-begin } as AlgoState;
}

// ---------- Canvas Drawing ----------
function drawPanel(ctx: CanvasRenderingContext2D, N: number, sizePx: number, blocks: Uint8Array, state: AlgoState | null, start: number, goal: number) {
  const cell = sizePx / N;
  ctx.clearRect(0,0,sizePx,sizePx);
  // background cells
  for (let r=0;r<N;r++) {
    for (let c=0;c<N;c++) {
      const id = idOf(N,r,c);
      if (blocks[id]) { ctx.fillStyle = "#0f172a"; } else { ctx.fillStyle = "#f8fafc"; }
      ctx.fillRect(c*cell, r*cell, cell, cell);
    }
  }
  if (state) {
    // visited/closed
    ctx.fillStyle = "#fde68a"; // amber-200
    state.closed.forEach(id => {
      const { r, c } = rcOf(N,id); ctx.fillRect(c*cell, r*cell, cell, cell);
    });
    // open/frontier
    ctx.fillStyle = "#bfdbfe"; // blue-200
    state.open.forEach(id => {
      const { r, c } = rcOf(N,id); ctx.fillRect(c*cell, r*cell, cell, cell);
    });
    // current
    if (state.current != null) {
      const { r, c } = rcOf(N,state.current); ctx.fillStyle = "#ef4444"; ctx.fillRect(c*cell, r*cell, cell, cell);
    }
    // final path
    if (state.path) {
      ctx.fillStyle = "#86efac"; // green-300
      for (const id of state.path) { const { r, c } = rcOf(N,id); ctx.fillRect(c*cell, r*cell, cell, cell); }
    }
  }
  // start/goal overlays
  const sRC = rcOf(N,start), gRC = rcOf(N,goal);
  ctx.fillStyle = "#22c55e"; ctx.fillRect(sRC.c*cell, sRC.r*cell, cell, cell);
  ctx.fillStyle = "#8b5cf6"; ctx.fillRect(gRC.c*cell, gRC.r*cell, cell, cell);
  // grid lines (light)
  ctx.strokeStyle = "#e2e8f0"; ctx.lineWidth = 0.5;
  for (let i=0;i<=N;i++) { ctx.beginPath(); ctx.moveTo(0, i*cell); ctx.lineTo(sizePx, i*cell); ctx.stroke(); }
  for (let j=0;j<=N;j++) { ctx.beginPath(); ctx.moveTo(j*cell, 0); ctx.lineTo(j*cell, sizePx); ctx.stroke(); }
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

  const sizePx = 360; // per panel
  const start = 0;
  const goal = useMemo(() => {
  const R = rngLCG(seed + 999);        // deterministic per seed
  let gr = Math.floor((R.next().value as number) * N);
  let gc = Math.floor((R.next().value as number) * N);
  let gid = idOf(N, gr, gc);
  if (gid === start) {
    gr = (gr + 1) % N;
    gid = idOf(N, gr, gc);
  }
  return gid;
}, [N, seed]);

  // Map generation
  const blocks = useMemo(()=>{
    let b: Uint8Array;
    if (mapType === "Empty") b = generateEmpty(N);
    else if (mapType === "Random") {
      // regenerate until solvable (limit attempts)
      let attempt = 0; do { b = generateRandom(N,density,seed+attempt,start,goal); attempt++; } while (attempt<30 && !solvable(N,b!,start,goal,diag));
    } else {
      b = generateMaze(N, seed, start, goal);
      if (!solvable(N,b,start,goal,diag)) {
        // open a sparse corridor by forcing empty on a diagonal line
        for (let i=0;i<N;i++) b[idOf(N,i,i)] = 0;
      }
    }
    return b!;
  }, [N,mapType,density,seed,start,goal,diag]);

  // Algorithm generators and state
  const gensRef = useRef<Record<AlgoKey, Generator<AlgoState, AlgoState, void> | null>>({ BFS:null, Dijkstra:null, Greedy:null, "A*":null });
  const [algoStates, setAlgoStates] = useState<Record<AlgoKey, AlgoState | null>>({ BFS:null, Dijkstra:null, Greedy:null, "A*":null });
  const [allFinished, setAllFinished] = useState(false);

  // Initialize generators when map or N changes
  useEffect(()=>{
    gensRef.current.BFS = algoBFS(N, blocks, start, goal, diag);
    gensRef.current.Dijkstra = algoDijkstra(N, blocks, start, goal, diag);
    gensRef.current.Greedy = algoGreedy(N, blocks, start, goal, diag);
    gensRef.current["A*"] = algoAStar(N, blocks, start, goal, diag);
    setAlgoStates({ BFS:null, Dijkstra:null, Greedy:null, "A*":null });
    setAllFinished(false);
    setRunning(false);
  }, [N, blocks, start, goal, diag]);

  // Animation loop (lockstep)
  useEffect(() => {
    if (!running) return;
    let handle: number;
    let acc = 0;
    const stepInterval = 1000 / speed;
    let last = performance.now();
  
    const tick = () => {
      const now = performance.now();
      acc += (now - last);
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
  
        // ✅ Stop only when ALL 4 are finished
        const allDone = (Object.values(gensRef.current).every((g) => g === null));
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
    BFS: useRef(null), Dijkstra: useRef(null), Greedy: useRef(null), "A*": useRef(null)
  };
  useEffect(()=>{
    (Object.keys(canvasRefs) as AlgoKey[]).forEach(key => {
      const cvs = canvasRefs[key].current; if (!cvs) return;
      const ctx = (cvs as HTMLCanvasElement).getContext("2d"); if (!ctx) return;
      drawPanel(ctx, N, sizePx, blocks, algoStates[key], start, goal);
    });
  }, [algoStates, N, blocks, start, goal]);

  const resetRun = () => {
    gensRef.current.BFS = algoBFS(N, blocks, start, goal, diag);
    gensRef.current.Dijkstra = algoDijkstra(N, blocks, start, goal, diag);
    gensRef.current.Greedy = algoGreedy(N, blocks, start, goal, diag);
    gensRef.current["A*"] = algoAStar(N, blocks, start, goal, diag);
    setAlgoStates({ BFS:null, Dijkstra:null, Greedy:null, "A*":null });
    setAllFinished(false);
    setRunning(false);
  };

  const winningOrder = useMemo(() => {
    const items = Object.entries(algoStates)
      .map(([k, s]) => ({ key: k as AlgoKey, s }))
      .filter(({ s }) => s && s.finished)                            // only finished
      .map(({ key, s }) => ({
        key,
        t: s!.lastRuntimeMs || Number.POSITIVE_INFINITY,
        found: !!s!.found,
      }))
      .sort((a, b) => a.t - b.t);
  
    return items;
  }, [algoStates]);

  return (
    <div className="min-h-screen">
      <div className="wrapper">
        <header className="mb-6">
          <h1 className="text-3xl font-bold tracking-tight">Pathfinding Lab</h1>
          <p className="text-slate-600">Side‑by‑side BFS · Dijkstra · Greedy · A* on configurable grids</p>
        </header>

        {/* Controls */}
        <div className="controls">
        <div className="control-card">
            <label className="block text-sm mb-1">Grid Size (N×N)</label>
            <input type="number" value={N} min={5} max={150} onChange={e=>setN(Math.max(5, Math.min(150, Number(e.target.value)||16)))} className="w-full border rounded px-3 py-2" />
            <div className="text-xs text-slate-500 mt-1">Try 9, 16, 50, 100…</div>
          </div>
          <div className="control-card">
            <label className="block text-sm mb-1">Map Type</label>
            <select value={mapType} onChange={e=>setMapType(e.target.value as MapType)} className="w-full border rounded px-3 py-2">
              <option>Empty</option>
              <option>Random</option>
              <option>Maze</option>
            </select>
            {mapType === "Random" && (
              <div className="mt-3">
                <label className="block text-sm">Obstacle Density: {(density*100).toFixed(0)}%</label>
                <input type="range" min={0} max={0.5} step={0.01} value={density} onChange={e=>setDensity(Number(e.target.value))} className="w-full" />
              </div>
            )}
            <div className="mt-3 flex items-center gap-2">
              <input id="diag" type="checkbox" checked={diag} onChange={e=>setDiag(e.target.checked)} />
              <label htmlFor="diag" className="text-sm">Allow diagonal moves</label>
            </div>
          </div>
          <div className="control-card">
            <label className="block text-sm mb-1">Seed</label>
            <input type="number" value={seed} onChange={e=>setSeed(Number(e.target.value)||0)} className="w-full border rounded px-3 py-2" />
            <div className="text-xs text-slate-500 mt-1">Reproducible maps</div>
            <label className="block text-sm mt-4">Speed: {speed} steps/s</label>
            <input type="range" min={1} max={60} value={speed} onChange={e=>setSpeed(Number(e.target.value))} className="w-full" />
          </div>
          <div className="bg-white rounded-2xl shadow p-4 flex flex-col gap-2">
            <button onClick={()=>setRunning(true)} disabled={running} className="px-4 py-2 rounded-xl bg-emerald-600 text-white disabled:opacity-50">Start</button>
            <button onClick={()=>setRunning(false)} className="px-4 py-2 rounded-xl bg-amber-500 text-white">Pause</button>
            <button onClick={resetRun} className="px-4 py-2 rounded-xl bg-slate-800 text-white">Reset</button>
            <div className="text-xs text-slate-500">Start: (0,0) · Goal: ({N-1},{N-1})</div>
          </div>
        </div>

        {/* Panels */}
        <div className="panels">
          {(Object.keys(canvasRefs) as AlgoKey[]).map(key=>{
            const s = algoStates[key];
            return (
              <div key={key} className="panel">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="font-semibold">{key}</h2>
                  <div className="text-xs text-slate-500">{s?.found ? "Found" : s?.finished ? "No path" : running ? "Running" : "Idle"}</div>
                </div>
                <canvas ref={canvasRefs[key]} width={sizePx} height={sizePx} />
                <div className="stats">
                  <div className="text-slate-500">Expanded</div><div className="font-mono">{s?.nodesExpanded ?? 0}</div>
                  <div className="text-slate-500">Peak frontier</div><div className="font-mono">{s?.peakFrontier ?? 0}</div>
                  <div className="text-slate-500">Runtime</div><div className="font-mono">{s ? s.lastRuntimeMs.toFixed(1)+" ms" : "0.0 ms"}</div>
                  <div className="text-slate-500">Path length</div><div className="font-mono">{s?.path ? s.path.length : "—"}</div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Podium / order */}
        <div className="finish-order">
          <h3 className="font-semibold mb-2">Finish Order</h3>
          {winningOrder.length===0 ? (
            <div className="text-sm text-slate-500">No algorithm has finished yet.</div>
          ) : (
            <ol className="list-decimal list-inside space-y-1">
              {winningOrder.map((w, i) => (
                <li key={i} className="text-sm">
                  {w.key} — <span className="font-mono">{isFinite(w.t) ? `${w.t.toFixed(1)} ms` : "—"}</span>
                  {!w.found && <span className="text-sm"> (no path)</span>}
                </li>
                ))}
            </ol>
          )}
        </div>

        <footer className="mt-8 text-xs text-slate-500">
          Tips: Use small N (9 or 16) to watch the expansions; then scale to 50–100 for performance comparisons. Colors — walls: slate‑900, visited: amber‑200, frontier: blue‑200, path: green‑300, current: red; start: green; goal: violet.
        </footer>
      </div>
    </div>
  );
}
