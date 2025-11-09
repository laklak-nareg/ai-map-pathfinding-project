# ai-map-pathfinding-project
project for an AI course — visualization and comparative analysis of BFS, Dijkstra, Greedy, and A* algorithms on dynamic grid maps.


# 🧭 AI Pathfinding Lab

An **interactive web application** that visualizes and compares the performance of classical **AI pathfinding algorithms** — **BFS**, **Dijkstra**, **Greedy Best-First Search**, and **A\*** — on configurable maps and procedurally generated mazes.  
Built with **React + TypeScript**, this project allows users to explore how different search strategies behave, measure their performance, and visualize optimal vs. non-optimal paths in real time.

---

## 🚀 Features

- 🔄 **Real-time visualization** of 4 algorithms running side by side  
- 🧱 **Configurable map sizes** — 9×9, 16×16, 50×50, and beyond  
- 🌫️ **Map generation options:** Empty grid, Random obstacles, or Maze  
- ⚙️ **Adjustable parameters:**
  - Obstacle density (for Random maps)
  - Maze tightness (for Maze maps)
  - Search speed (steps per second)
  - Heuristic type (Manhattan / Chebyshev)
  - Diagonal movement toggle  
- 🎯 **Randomized start and goal points**
- 🧩 **Performance metrics:**
  - Nodes expanded
  - Frontier size
  - Path length and cost
  - Execution time
  - Optimal path detection  

---

## 📸 Demo Preview

![Alt text](path/to/image.png)



---

## ⚗️ Algorithms Implemented

| Algorithm | Strategy | Finds Optimal Path? | Uses Heuristic? |
|------------|-----------|---------------------|------------------|
| **BFS** | Uniform search | ✅ Yes (uniform costs only) | ❌ No |
| **Dijkstra** | Uniform-cost search | ✅ Yes | ❌ No |
| **Greedy Best-First Search** | Heuristic-only | ❌ Not always | ✅ Yes |
| **A\*** | Cost + heuristic | ✅ Yes (with admissible h) | ✅ Yes |

---

## 🧮 Heuristics

- **Manhattan Distance:**  
  \( h(n) = |x_1 - x_2| + |y_1 - y_2| \) — best for 4-direction movement  

- **Chebyshev Distance:**  
  \( h(n) = \max(|x_1 - x_2|, |y_1 - y_2|) \) — good for 8-direction movement  

Users can experiment with both to see how heuristics influence speed and path quality.

---

## 🧑‍💻 Tech Stack

- **React + TypeScript**
- **Vite** (for fast development and hot reloading)
- **HTML Canvas** (for visualization)
- **CSS** (for styling)

---

## ⚙️ Installation & Running Locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/ai-pathfinding-lab.git
cd ai-pathfinding-lab

# Install dependencies
npm install

# Start the development server
npm run dev

# Open your browser and visit:
# http://localhost:5173/





