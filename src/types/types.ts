export type Cell = { r: number; c: number };

export type MapType = "Empty" | "Random" | "Maze";

export type AlgoKey = "BFS" | "Dijkstra" | "Greedy" | "A*" | "BiA*";

export type HeuristicType =
  | "Manhattan"
  | "Chebyshev"
  | "Euclidean"
  | "WallAware";
