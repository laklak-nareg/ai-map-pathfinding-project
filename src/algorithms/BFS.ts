import type { AlgoState } from "../interfaces/interfaces";
import { neighbors, reconstructPath } from "../utils/utils";

export function* algoBFS(
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
