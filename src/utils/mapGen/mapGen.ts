import type { Cell } from "../../types/types";
import { idOf, rngLCG } from "../utils";

// ---------- Map Generation ----------
export function generateEmpty(N: number) {
  const blocks = new Uint8Array(N * N); // 0 free, 1 wall
  return blocks;
}

//random Maze
export function generateRandom(
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
// Maze via DFS backtracker
export function generateMaze(N: number, seed: number) {
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
