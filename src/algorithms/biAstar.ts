import type { AlgoState } from "../interfaces/interfaces";
import { MinHeap } from "../utils/MinHeap/MinHeap";
import { neighbors, rcOf, reconstructPath } from "../utils/utils";

export function* algoBiAStar(
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
            parents: parentsF, // forward parents are enough to explain from start side
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
      current: null as any, // we don't highlight a single node here
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
