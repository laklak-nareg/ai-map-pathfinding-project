import type { AlgoKey, MapType } from "../types/types";

export interface AlgoState {
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

export interface RunConfig {
  N: number;
  mapType: MapType;
  density: number; // for Random
  seed: number;
  diag: boolean; // allow 8-neighbors
}
