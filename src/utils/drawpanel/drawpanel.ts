import type { AlgoState } from "../../interfaces/interfaces";
import { map_color_constants } from "../constants";
import { idOf, rcOf } from "../utils";
const {
  emptyColor,
  wallColor,
  visitedColor,
  frontierColor,
  currentColor,
  finalPathColor,
  startGoalColor,
  endGoalColor,
} = map_color_constants;
// ---------- Canvas Drawing ----------

export const drawPanel = (
  ctx: CanvasRenderingContext2D,
  N: number,
  sizePx: number,
  blocks: Uint8Array,
  state: AlgoState | null,
  start: number,
  goal: number
) => {
  const cell = sizePx / N;
  ctx.clearRect(0, 0, sizePx, sizePx);
  // background cells
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const id = idOf(N, r, c);
      ctx.fillStyle = blocks[id] ? wallColor : emptyColor;
      ctx.fillRect(c * cell, r * cell, cell, cell);
    }
  }
  if (state) {
    // visited/closed
    ctx.fillStyle = visitedColor;
    state.closed.forEach((id) => {
      const { r, c } = rcOf(N, id);
      ctx.fillRect(c * cell, r * cell, cell, cell);
    });
    // open/frontier
    ctx.fillStyle = frontierColor;
    state.open.forEach((id) => {
      const { r, c } = rcOf(N, id);
      ctx.fillRect(c * cell, r * cell, cell, cell);
    });
    // current
    if (state.current != null) {
      const { r, c } = rcOf(N, state.current);
      ctx.fillStyle = currentColor;
      ctx.fillRect(c * cell, r * cell, cell, cell);
    }
    // final path
    if (state.path) {
      ctx.fillStyle = finalPathColor;
      for (const id of state.path) {
        const { r, c } = rcOf(N, id);
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
    }
  }
  // start/goal overlays
  const sRC = rcOf(N, start),
    gRC = rcOf(N, goal);
  ctx.fillStyle = startGoalColor;
  ctx.fillRect(sRC.c * cell, sRC.r * cell, cell, cell);
  ctx.fillStyle = endGoalColor;
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
};
