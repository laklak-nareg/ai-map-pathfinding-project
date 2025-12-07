# 🧭 AI Pathfinding Lab

An interactive web application that visualizes and compares the performance of classical AI pathfinding algorithms — **BFS**, **Dijkstra**, **Greedy Best-First Search**, **A\***, and **Bidirectional A\*** — on configurable maps and procedurally generated mazes.  
Built with **React + TypeScript**, this project allows users to explore how different search strategies behave, measure their performance, and visualize optimal vs. non-optimal paths in real time.

---

## 🚀 Features

- 🔄 **Real-time visualization** of 5 algorithms running side by side  
- 🧱 **Configurable map sizes** — 9×9, 16×16, 50×50, and beyond  
- 🌫️ **Map generation options:** Empty grid, Random obstacles, or Maze  
- ⚙️ **Adjustable parameters:**
  - Obstacle density (for Random maps)
  - Maze generation via randomized DFS backtracking
  - **Greedy Trap mode** (for Maze maps): selects a deceptive start–goal pair that misleads Greedy
  - Search speed (steps per second)
  - Heuristic type (Manhattan / Euclidean / Chebyshev / Wall-Aware)
  - Diagonal movement toggle
- 🎯 Randomized or maze-derived start and goal positions  
- 🧩 **Performance metrics:**
  - Nodes expanded
  - Peak frontier size
  - Path length
  - Runtime
  - Optimality indication

---

## ⚗️ Algorithms Implemented

| Algorithm               | Strategy                     | Finds Optimal Path?                  | Uses Heuristic? |
|-------------------------|------------------------------|--------------------------------------|-----------------|
| **BFS**                | Uniform search               | ✅ Yes (on unweighted grids)         | ❌ No           |
| **Dijkstra**           | Uniform-cost search          | ✅ Yes                               | ❌ No           |
| **Greedy Best-First**  | Heuristic-only search        | ❌ Not guaranteed                    | ✅ Yes          |
| **A\***                | Cost + heuristic (g + h)     | ✅ Yes (with admissible heuristic)   | ✅ Yes          |
| **Bidirectional A\***  | Two A* searches meeting mid  | ✅ Yes                               | ✅ Yes          |

---

## 🧮 Heuristics

### Manhattan Distance (L₁ metric)

$$
h(n) = |x_1 - x_2| + |y_1 - y_2|
$$

- Natural for 4-directional movement.
- Admissible and consistent on grids with unit edge costs.

---

### Euclidean Distance (L₂ metric)

$$
h(n) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$

- Geometric straight-line distance.
- Admissible when movement cost is at least the Euclidean step cost.

---

### Chebyshev Distance (L∞ metric)

$$
h(n) = \max\left(|x_1 - x_2|,\ |y_1 - y_2|\right)
$$

- Suited for 8-directional movement.
- Matches the number of king moves on a chessboard.
- Admissible for grids where diagonal and straight moves have compatible costs.

---

### Wall-Aware Heuristic (custom)

$$
h(n) = h_{\text{base}}(n) + \lambda \cdot \text{wallPenalty}(n)
$$

Where:
- \( h_{\text{base}}(n) \) is either Manhattan or Chebyshev (depending on movement).
- `wallPenalty(n)` counts how many of the 4 cardinal neighbours of \( n \) are walls or outside the grid.
- \( \lambda > 0 \) is a tunable weight.

This heuristic:

- Penalizes cells close to walls and narrow tunnels.
- Helps A\* and BiA\* avoid dead-end corridors.
- Is **not strictly admissible** (by design), but very useful in “Greedy Trap” mazes.

---

## 🎭 Greedy Trap Mode (Maze Only)

For maze maps, there is an optional **“Deceptive maze (Greedy trap)”** mode:

- The maze is generated using randomized depth-first backtracking.
- A start–goal pair is chosen such that:
  - The heuristic distance (e.g. Manhattan) between them is small,
  - But the true shortest path is much longer and often passes through dead-end–like corridors.
- **Greedy Best-First Search** tends to follow the misleading heuristic gradient into a dead end and wastes many expansions.
- **A\*** and **Bidirectional A\*** use the combination of cost and heuristic to recover and still find (near-)optimal paths faster.

This mode makes it easy to **visually demonstrate** why heuristic design matters and why greedy search can perform poorly in maze-like environments.

---

## 🧑‍💻 Tech Stack

- **React + TypeScript**
- **Vite** (fast dev server + bundler)
- **HTML Canvas** (for rendering grids and search progress)
- **CSS / Tailwind** (for styling and layout)

---

## ⚙️ Installation & Running Locally

```bash
# Clone the repository
git clone https://github.com/laklak-nareg/ai-map-pathfinding-project.git
cd ai-map-pathfinding-project

# Install dependencies
npm install

# Start the development server
npm run dev

# Then open:
# http://localhost:5173/
