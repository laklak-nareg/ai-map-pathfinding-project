import React, { useEffect, useMemo, useRef, useState } from "react";
import type { AlgoState } from "./interfaces/interfaces";
import type { AlgoKey, HeuristicType, MapType } from "./types/types";
import {
  generateEmpty,
  generateMaze,
  generateRandom,
} from "./utils/mapGen/mapGen";
import { carvePathToGoal, idOf, rngLCG } from "./utils/utils";
import { algoGreedy, pickDeceptiveStartGoal } from "./algorithms/Greedy";
import { buildHeuristic } from "./utils/heuristic/buildHeuristic";
import { algoAStar } from "./algorithms/aStar";
import { algoBFS } from "./algorithms/BFS";
import { algoDijkstra } from "./algorithms/Dijkstra";
import { algoBiAStar } from "./algorithms/biAstar";
import { drawPanel } from "./utils/drawpanel/drawpanel";

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

  const { blocks, startId, goalId } = useMemo(() => {
    let b: Uint8Array;
    let s = start;
    let g = goal;

    switch (mapType) {
      case "Empty":
        b = generateEmpty(N);
        s = 0;
        g = idOf(N, N - 1, N - 1);
        break;
      case "Random": {
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

        //Generate a random obstacle field
        b = generateRandom(N, density, seed, s, g);
        //GUARANTEE at least one path from s → g
        carvePathToGoal(N, b, s, g, diag);
        break;
      }
      default:
        // Maze
        b = generateMaze(N, seed);

        if (deceptiveMaze) {
          // Special case: pick a start/goal that misleads Greedy
          const pair = pickDeceptiveStartGoal(N, b, diag, seed);
          s = pair.start;
          g = pair.goal;
        } else {
          // Normal maze: just pick two far open cells
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
        break;
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
  const gensRef = useRef<
    Record<AlgoKey, Generator<AlgoState, AlgoState, void> | null>
  >({
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

        // ✅ Stop only when ALL 4 are finished
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
            Side‑by‑side BFS · Dijkstra · Greedy · A* on configurable grids
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
                setN(Math.max(5, Math.min(150, Number(e.target.value) || 16)))
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
              onChange={(e) => setSeed(Number(e.target.value) || 0)}
              className="w-full border rounded px-3 py-2"
            />
            <div className="text-xs text-slate-500 mt-1">Reproducible maps</div>
            <label className="block text-sm mt-4">Speed: {speed} steps/s</label>
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
              Start: (0,0) · Goal: ({N - 1},{N - 1})
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
                <canvas ref={canvasRefs[key]} width={sizePx} height={sizePx} />
                <div className="stats">
                  <div className="text-slate-500">Expanded</div>
                  <div className="font-mono">{s?.nodesExpanded ?? 0}</div>
                  <div className="text-slate-500">Peak frontier</div>
                  <div className="font-mono">{s?.peakFrontier ?? 0}</div>
                  <div className="text-slate-500">Runtime</div>
                  <div className="font-mono">
                    {s ? s.lastRuntimeMs.toFixed(1) + " ms" : "0.0 ms"}
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
                    {isFinite(w.t) ? `${w.t.toFixed(1)} ms` : "—"}
                  </span>
                  {!w.found && <span className="text-sm"> (no path)</span>}
                </li>
              ))}
            </ol>
          )}
        </div>

        <footer className="mt-8 text-xs text-slate-500">
          Tips: Use small N (9 or 16) to watch the expansions; then scale to
          50–100 for performance comparisons. Colors — walls: slate‑900,
          visited: amber‑200, frontier: blue‑200, path: green‑300, current: red;
          start: green; goal: violet.
        </footer>
      </div>
    </div>
  );
}
