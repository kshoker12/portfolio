# ♟️ 3D Chess AI Engine — Serverless ML Inference, Alpha–Beta Search, and 3D Web UI

**Author:** Karandeep “Karn” Shoker (UBC CS & Statistics) · **Rapid summary:** Full‑stack chess system featuring a **React Three Fiber** 3D frontend and a **serverless AWS** backend that serves moves via **alpha–beta search** with **neural evaluation** (two ML models) and a **Stockfish** “hard” mode. Built end‑to‑end to highlight graphics, algorithms, MLOps, and cloud architecture. As a ~**1600 ELO** player, building this humbled me (I still can’t beat Stockfish!) and deepened my positional understanding.

> **What recruiters should notice**
> - Depth across **graphics**, **algorithms**, **machine learning**, and **cloud**.
> - Clear **system design** and **operational excellence** (cost, latency, scalability).
> - Evidence of **experimentation**, **metrics**, and **trade‑off reasoning**.

---

## 1) System Overview

```
Client (React + R3F/Three.js + Redux) 
   └──> HTTPS (fetch) 
         └──> Amazon API Gateway (REST)
                └──> AWS Lambda (container, ARM64) running FastAPI
                        ├── PyTorch MLP (easy/fast)
                        ├── scikit-learn MLPRegressor (default/medium)
                        └── Stockfish 16 (hard/accurate)
                              ↑
                      ECR container image (models + engine logic)
```

- **Frontend:** React 18 + **React Three Fiber (R3F)** over Three.js for a smooth 3D chessboard; chess rules and legal move generation from **chess.js**; client state via **Redux Toolkit**.
- **Backend:** **FastAPI** app packaged as an **AWS Lambda** container (**ARM64/Graviton**) behind **Amazon API Gateway**. Returns the best move for a provided FEN with a chosen difficulty, using **alpha–beta** + **quiescence** search and a **hybrid evaluation** (neural + classical heuristics), or **Stockfish** at “hard.”
- **Data/Models:** Two learned evaluation models trained using positions from games by **GM‑level players** (e.g., Magnus Carlsen, Bobby Fischer, Hikaru Nakamura) labeled with **Stockfish centipawn** evaluations.

---

## 2) Frontend (3D + UX) — React Three Fiber, Three.js, Redux

### Rendering & Interaction
- **R3F** scene with board, pieces, camera and lighting; responsive layout; high‑FPS piece animations; orbit/drag interactions.
- **Chess.js** for rule validation and legal moves; **hover highlights** and **check indicators**.
- **Move list, clocks, resign/draw**, and **VS Bot / Pass‑and‑Play** modes.

### State & Performance
- **Redux Toolkit** slices (board/game/UI); memoized selectors to limit re‑renders to **changed squares**.
- **Lazy chunking** of heavy components; request **debounce**; **AbortController** to cancel stale API calls.
- **Mobile** friendly (pointer events) and keyboard accessibility for desktop.

---

## 3) Engine: Algorithms, Data Structures & High‑Level Logic

### 3.1 Board Representation & Features
- The ML evaluator consumes a **776‑dimensional feature vector** per position:
  - **12×64 piece planes** (one-hot: piece × square) → 768
  - **Side to move**, **castling rights (4)**, **half‑move clock**, **full‑move number**, **material tallies** → remaining features
- Classical evaluation adds **material balance**, **piece‑square tables**, **mobility**, **king safety** penalties, and simple **pawn structure** heuristics.

### 3.2 Search Core
- **Alpha–Beta with iterative deepening** and **aspiration windows**.
- **Quiescence** on capture/tactical leafs to mitigate horizon effects.
- **Move ordering:** principal variation moves, **killer moves**, **history heuristic**, **MVV‑LVA**, and **SEE** (static exchange evaluation).
- **Pruning/Reductions:** **null‑move pruning**, **late‑move reductions (LMR)**, futility pruning in quiet nodes.
- **Transposition Table:** **Zobrist hashing** (64‑bit) with **(score, depth, bound)** entries; replacement scheme favors deeper nodes.
- **Blunder filters:** lightweight guards against hanging high‑value pieces in shallow lines.
  
**Pseudocode (abridged)**
```text
function SEARCH(node, depth, alpha, beta):
  if depth == 0: return QUIESCENCE(node, alpha, beta)

  if TT.has(node) and TT.depth(node) >= depth:
      entry = TT[node]
      if entry.bound == EXACT: return entry.score
      if entry.bound == LOWER: alpha = max(alpha, entry.score)
      if entry.bound == UPPER: beta  = min(beta,  entry.score)
      if alpha >= beta: return entry.score

  if NULL_MOVE_OK(node) and depth >= R:
      score = -SEARCH(NULL_MOVE(node), depth-1-R, -beta, -beta+1)
      if score >= beta: return beta   // fail-hard

  moves = ORDER(GENERATE(node))  // PV, killers, history, MVV-LVA, SEE
  best = -INF; pv = None
  for (i, m) in enumerate(moves):
      newDepth = depth - 1
      if i > LMR_THRESHOLD and !m.capt: newDepth -= 1   // LMR
      score = -SEARCH(APPLY(node, m), newDepth, -beta, -alpha)
      if score > best:
          best = score; pv = m
          alpha = max(alpha, score)
          if alpha >= beta: break   // cutoff

  TT.store(node, best, depth, bound=BOUND(alpha,beta,best), pv)
  return best
```

### 3.3 Hybrid Evaluation
- **Neural eval (70–85%)** blended with **classical eval (15–30%)** depending on difficulty and game phase.
- Classical stabilizes edge cases (e.g., material trades, simple mates) while the neural net captures **positional** signals (space, tension, outposts, pawn structure) hard to encode by hand.

---

## 4) Machine Learning: Data, Training, and the “776‑Feature” Problem

### 4.1 Dataset & Labeling
- **Source:** PGN games from elite players (Magnus, Fischer, Hikaru, etc.).
- **Extraction:** Sample mid‑game and late‑opening positions; compute labels with **Stockfish** at calibrated depth (≈ 15) returning **centipawn** evaluations.
- **De‑duplication** of near‑identical positions; **balance** across phases (opening/middlegame/endgame) to avoid skew.

### 4.2 Models
- **Model A (PyTorch MLP)** — fast inference (“easy”): BN + Dropout, GELU activations; trained with AdamW + cosine LR.
- **Model B (scikit‑learn `MLPRegressor`)** — default (“medium”): `TransformedTargetRegressor(StandardScaler)` around MLP to stabilize targets.
- **“Hard” mode** uses **Stockfish 16** directly (no learning), prioritizing strength over latency.

### 4.3 Training Process & Tuning
- **Train/val/test split** on **games**, not just positions (leak prevention).
- **k‑fold cross‑validation** for hyper‑parameters: depth, width, dropout, L2, batch size, LR schedule.
- **Early stopping** on validation **MAE**; checkpoint the top‑k models and **ensemble average** candidates.
- **When underperforming**, fixes included:
  - **More data** (especially tactical/middlegame density).
  - **Wider/deeper MLP** (additional hidden layers) with **regularization**.
  - Adjusting **target scaling**; clipping outliers beyond ±2000 cp.
  - Rethinking feature interactions instead of aggressive **PCA** (kept raw planes to preserve spatial semantics).

### 4.4 Curse of Dimensionality (776 features)
- 776 features create a **high‑dimensional space** where distances become less informative and generalization suffers.
- Mitigations applied:
  - **L2 weight decay**, **dropout**, and **batch norm** for smoother optimization.
  - **Target engineering:** compress centipawns near 0 (small margins matter) and clip extremes.
  - **Data volume:** aggressively augment with diverse positions to increase coverage of sparse regions.
- **Observed metrics:** **MAE ≈ 300 cp**, **R² ≈ 0.30–0.42** depending on mix and depth of labels (indicative for a non‑GPU budget project and good enough for play with search).

---

## 5) AWS Architecture & Ops (Serverless)

### 5.1 Services and Roles
- **AWS Amplify**: CI/CD for the static SPA; **CloudFront CDN**; environment variables (e.g., `VITE_API_BASE`). Custom domain + HTTPS.
- **Amazon API Gateway (REST)**: `POST /v1/api/move` (Lambda proxy), **rate limiting**, **CORS** (restrict origin to Amplify domain).
- **AWS Lambda (container, ARM64/Graviton)**: Runs **FastAPI** with models and engine logic packaged into a single **Docker** image.
- **Amazon ECR**: Container registry for versioned Lambda images.
- **Amazon CloudWatch**: Logs, metrics (p50/p90 latency), alarms on 5XX, throttle, or cold‑start spikes.
- **(Optional)** **S3** for temporary artifacts; **Provisioned Concurrency** for strict p95 latency SLOs.

### 5.2 Container & Lambda Configuration
- **Base image**: slim Python on ARM64; **manylinux2014** wheels for sklearn/torch to avoid build‑time compilation.
- **Image size** target: **< 600 MB** by pruning dev packages, caches, and docs.
- **Memory**: 1024 MB; **timeout**: 15 s (tight, but sufficient for depth‑limited search).
- **Concurrency**: default on‑demand; enable **provisioned concurrency** when showcasing to a live audience.
- **Model caching**: load models **once** at module import; keep global singletons to survive between invocations.
- **Threading**: disable multithreading for BLAS in Lambda (e.g., `MKL_NUM_THREADS=1`) to avoid noisy neighbors.
- **Cold‑start** mitigations**:** small dependency graph; lazy import of Stockfish; trimmed fonts/locales; **pydantic** compiled mode.

### 5.3 API Contract
- **Endpoint:** `POST /v1/api/move`
- **Request:**
  ```json
  { "fen": "<FEN>", "difficulty": "easy|medium|hard", "max_depth": 4 }
  ```
- **Response:**
  ```json
  { "best_move": "e2e4", "score_cp": 25, "nodes": 54512, "depth": 4, "pv": ["e2e4", "c7c5", "g1f3"] }
  ```
- **CORS:** `Access-Control-Allow-Origin: https://<your-amplify-domain>`

### 5.4 Cost, Scale, and Reliability
- **Pay‑per‑request** with automatic horizontal scaling; ideal for **1–1000** demos/day.
- **ARM64** reduces compute cost vs x86 (~15–20% improvement).
- **Retry & timeouts** configured at API Gateway; Lambda DLQ/alerts via CloudWatch.
- **Zero servers to patch**; deploys are **atomic** via ECR image tags and Amplify CI.

---

## 6) Results & Benchmarks

- **Evaluator quality (MLP):** ~**300 cp MAE**; **R² ~ 0.30–0.42** on held‑out sets labeled by Stockfish at controlled depth.
- **Latency (typical)** by difficulty on Lambda@1GB:
  - `easy` (PyTorch MLP): **70–150 ms**
  - `medium` (sklearn MLP): **120–250 ms**
  - `hard` (Stockfish, shallow depth): **300–800+ ms**
- **UX**: consistent 60 FPS rendering on modern laptops; legal‑move highlights feel instant; analysis is visibly stronger at “hard.”

> These numbers vary by FEN complexity and depth; the design prioritizes **determinism** and **predictable latency** for demos.

---

## 7) Local Dev & Quickstart

### Frontend
```bash
pnpm install
pnpm dev   # http://localhost:5173
# .env: VITE_API_BASE="https://<api-id>.execute-api.<region>.amazonaws.com/prod"
```

### Backend (local)
```bash
pip install -r requirements.txt
uvicorn app:app --reload  # http://localhost:8000

curl -X POST http://localhost:8000/v1/api/move   -H "Content-Type: application/json"   -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","difficulty":"medium","max_depth":4}'
```

### Deploy (high-level)
1. **Build** Docker image (ARM64) and **push** to **ECR**.
2. Create **Lambda** from image; set **memory/timeout/env**; point **API Gateway** to Lambda proxy.
3. Hook **Amplify** to the frontend repo; set **VITE_API_BASE**; deploy with CI/CD.
4. **Verify** CORS, test latency, add **CloudWatch** alarms.

---

## 8) What I Personally Delivered

- **Architecture**: designed and implemented the full system end‑to‑end.
- **Engine**: alpha‑beta + quiescence, ordering/pruning, TT with Zobrist, blunder filters.
- **ML**: data pipeline from PGNs → features → labels; trained **two evaluators**; cross‑validated hyper‑params; handled the **curse of dimensionality**.
- **3D Frontend**: R3F/Three.js scene, input handling, UI/UX polish, performance tuning.
- **DevOps**: containerized FastAPI; **Lambda@ARM64**; API Gateway; Amplify CI/CD; CloudWatch observability; cost/perf optimizations.

---

## 9) Roadmap
- Online multiplayer (WebSockets) + ELO; spectate & analysis boards.
- Post‑game **eval graph**, **blunder/spike** detection, natural‑language analysis.
- Opening book + endgame tablebases; deeper NN w/ distillation.
- Optional **Provisioned Concurrency** profile for guaranteed p95 latency in live demos.
- Stronger accessibility & mobile tactile feedback.

---

## 10) Skills Demonstrated
**Frontend:** React, TypeScript, React Three Fiber, Three.js, Redux Toolkit, Vite  
**Backend:** FastAPI, REST, Python, chess.js integration  
**Algorithms:** Alpha‑Beta, Quiescence, TT (Zobrist), Null‑Move, LMR, Killer/History, SEE, Aspiration Windows  
**ML:** scikit‑learn, PyTorch, feature engineering, cross‑validation, regularization, early stopping  
**Cloud:** AWS Amplify, API Gateway, Lambda (ARM64 containers), ECR, CloudWatch, CORS, CI/CD, cost/perf tuning

---

### Personal Note
As a ~**1600 ELO** player, this project sharpened my understanding of **positional chess**. Encoding ideas like **outposts, space, king safety, and pawn structure** into features/targets taught me *why* certain moves work, while getting crushed by “hard” mode reminded me how far there is to go 🤝.

