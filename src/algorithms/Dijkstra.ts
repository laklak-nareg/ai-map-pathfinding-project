import type { AlgoState } from "../interfaces/interfaces";
import { MinHeap } from "../utils/MinHeap/MinHeap";
import { neighbors, rcOf, reconstructPath } from "../utils/utils";

export function* algoDijkstra(
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
        diag && rcOf(N, n).r !== rcOf(N, m).r && rcOf(N, n).c !== rcOf(N, m).c
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
