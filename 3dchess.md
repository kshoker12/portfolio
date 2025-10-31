# ♟️ 3D Chess AI Engine 
**3D Chess Engine • AI-Powered Gameplay • Serverless Architecture**

As a ~**1600 ELO** chess player, I developed a fully interactive **3D Chess AI Engine** built to combine my passions for **Chess**, **Machine Learning**, **Full-Stack Development**, and **Cloud Architectures**, into a responsive 3D frontend for immersive gameplay with a backend AI engine that integrates machine learning models trained on grandmaster games. My goal was to build a rich, intelligent, and scalable 3D chess experience while showcasing technical depth across *Data Structures*, *ML Engineering*, *3D Rendering*, and *Serverless Cloud Architecture*. 

Developing this chess engine helped me realize the complexity of good decision-making in chess and humbled me because despite my ELO rating, I struggle to beat the engine on **HARD Mode**, highlighting how subtle positional advantages can dominate games.

- **Live App**: [`3D Chess AI Engine`](https://main.d12st6uisht1xd.amplifyapp.com)
- **Frontend Repo**: https://github.com/kshoker12/3D-Chess-App
- **Backend Repo**: https://github.com/kshoker12/Chess-Engine

![Demo – Feature Showcase](/video/chess.mp4)

---

## System Overview

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

### Components
- **Frontend (React Three Fiber + Redux + AWS Amplify)**: Interactive 3D board, smooth piece movement animations, move highlights, timers, and camera controls.  
- **Backend (FastAPI on AWS Lambda behind API Gateway)**: Runs evaluation and returns the best move for a given board state (FEN) using classical search algorithms (**alpha-beta** + **Quiescence** search) and a hybrid evaluation (**Neural Network + classical heuristics**), or **Stockfish** for "hard".
- **Machine Learning Models**: Two learned evaluation models trained using 1M positions from games by **GM‑level players** (e.g., Magnus Carlsen, Bobby Fischer, Hikaru Nakamura) labeled with **Stockfish centipawn** evaluations.
- **Deployment**: Fully serverless architecture via **AWS Amplify, AWS Lambda (ARM64), API Gateway, and ECR containers.**
---

## Frontend — 3D Experience
The frontend delivers an immersive 3D chess experience, showcasing expertise in graphics programming, React state management, and performance optimization.
- **3D Board Rendering**: Built with *React Three Fiber (RTF)* on top of *Three.js*, enabling declarative 3D scene management in *React*. *RTF* handles the component-driven scene graph, while *Three.js* powers *WebGL* rendering, lighting, and materials.
- **Interactivity & Logic**: Integrated *chess.js* for legal move validation and board generation (FEN), ensuring the 3D board stays synchronized with game state in real time. Supports move highlighting, check/mate detection, and move history tracking.
- **3D Models**: Uses `GLTF` chess piece models with realistic textures and lighting for a polished, physical feel.
- **Animation**: Smooth piece transitions and camera motion powered by *React Spring*, maintaining 60 FPS animations across moves and camera orbits.
- **State Management**: *Redux Toolkit* manages board state (arrays of piece positions), timers, and asynchronous API calls to the backend. Custom selectors and memoization prevent unnecessary re-renders within the 60 FPS loop.
- **Performance Optimizations**: Employs *React* hooks (e.g. `useMemo`, `useCallback`) to minimize reconciliation cost and sustain stable frame rates during continuous interaction.

This combination of 3D rendering, real-time synchronization, and efficient *React* state architecture produces a visually engaging and technically robust frontend for chess gameplay.

---

## Backend - Engine Design 
The backend engine is a hybrid system blending classical search algorithms with ML-based evaluation, showcasing sophisticated data structures and logic for tactical precision.
### High-Level Logic
 The engine parses FEN (`Forsyth-Edwards Notation`) strings which captures the chess board into a 776-dimensional feature vector (12 piece types × 64 squares + metadata like castling). It uses iterative deepening with alpha-beta pruning to explore the game tree, evaluating leaf nodes via a blended score: 80% ML prediction + 20% heuristics (material, center control). Move ordering prioritizes captures using MVV-LVA heuristics to optimize search efficiency.
### Key Data Structures
- **Transposition Table**: A hash map (Python dict with Zobrist hashing as keys) to cache position evaluations, reducing recomputations by 40% and preventing horizon effects.
- **Priority Queue (heapq)**: For move sorting, implementing killer and history heuristics to evaluate promising moves first.
- **Numpy Arrays**: For efficient feature vectors and batch predictions during ML evaluation.
- **Stack for Quiescence Search**: Recursion stack to extend search for capture sequences, using material delta thresholds to prune non-quiet positions.
### Complexity & Optimizations
The alpha-beta search handles exponential tree growth (up to 35^depth branches) with pruning, reducing effective branching factor from 35 to ~3. Blunder prevention uses static exchange evaluation (SEE) with a tree-based lookahead (1-2 ply) to filter moves losing >50 centipawns. This design balances depth (6-8) with speed, achieving ~2100 ELO (`Hard mode`) while running in 2-5s on Lambda (Once model is loaded).
### Core Pseudocode 

**alpha-beta search**: Alpha-beta search is an optimized minimax algorithm used in game trees (e.g., chess) to find the best move by efficiently pruning branches that can't improve the outcome.

```text
function SEARCH(position, depth, α, β):
    # Base case: depth reached — evaluate the position
    if depth == 0:
        return EVALUATE(position)

    # Generate all legal moves for current side
    moves = GENERATE_MOVES(position)

    best_score = -∞

    # Loop through each move
    for move in ORDER(moves):
        # Apply the move to reach the next position
        next_pos = APPLY_MOVE(position, move)

        # Recursively search from the opponent's perspective
        score = -SEARCH(next_pos, depth - 1, -β, -α)

        # If this move improves the best score, remember it
        if score > best_score:
            best_score = score
            best_move = move

        # Alpha–beta pruning: cut off branches that can't improve the outcome
        α = max(α, score)
        if α >= β:
            break  # Prune the rest — no need to explore

    return best_score
```
**Quiescence Search**: To prevent horizon effects in tactical positions, the engine employs quiescence search as an extension to the main search, focusing on capture sequences until a "quiet" position is reached.
```
function QUIESCENCE_SEARCH(position, α, β):
    # Stand pat: evaluate current position without moves
    stand_pat = EVALUATE(position)
    
    if stand_pat >= β:
        return β  # Beta cutoff
    
    if α < stand_pat:
        α = stand_pat  # Update alpha

    # Generate capture moves only
    moves = GENERATE_CAPTURE_MOVES(position)
    
    for move in ORDER(moves):
        # Apply capture move
        next_pos = APPLY_MOVE(position, move)
        
        # Recursively search the response
        score = -QUIESCENCE_SEARCH(next_pos, -β, -α)
        
        if score >= β:
            return β  # Beta cutoff
        
        if score > α:
            α = score  # Update alpha

    return α
```
**Static Exchange Evaluation (SEE)**: Used in blunder prevention and move ordering, SEE evaluates if a capture sequence is profitable by simulating exchanges on a single square.
```
function SEE(position, target_square):
    # Get initial attacker value
    attacker_value = GET_PIECE_VALUE(GET_LEAST_VALUABLE_ATTACKER(position, target_square))
    
    # Simulate capture
    captured_value = GET_PIECE_VALUE(GET_PIECE_ON_SQUARE(position, target_square))
    
    score = captured_value - attacker_value
    
    # Continue exchanging until no attackers left
    while HAS_ATTACKERS(position, target_square):
        # Swap sides
        position = SWAP_SIDES(position)
        
        # Get next attacker
        next_attacker = GET_LEAST_VALUABLE_ATTACKER(position, target_square)
        
        # Subtract attacker value
        score = -score - GET_PIECE_VALUE(next_attacker)

    return max(0, score)  # Positive if profitable for initial side
```
**Move Ordering**: Efficient move ordering reduces the branching factor by trying promising moves first.
```
function ORDER_MOVES(position, moves):
    # Score each move
    scored_moves = []
    
    for move in moves:
        score = 0
        
        # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
        if IS_CAPTURE(move):
            score += 10 * GET_PIECE_VALUE(GET_CAPTURED_PIECE(move)) - GET_PIECE_VALUE(GET_ATTACKER_PIECE(move))
        
        # Killer moves: Add bonus for recent beta-cutoff moves
        if IS_KILLER_MOVE(move):
            score += 9000
        
        # History heuristic: Bonus based on historical success
        score += HISTORY_TABLE[GET_FROM_SQUARE(move)][GET_TO_SQUARE(move)]
        
        scored_moves.append((score, move))
    
    # Sort descending by score
    scored_moves.sort(key=lambda x: -x[0])
    
    return [move for score, move in scored_moves]
```
---

## Machine Learning Models 
Two ML models were trained on 1M+ positions from grandmaster games (e.g., Carlsen, Fischer, Nakamura, Anand, Kasparov), emphasizing rigorous engineering to overcome challenges like data quality and hyperparameter tuning.
- **Data Preparation**: PGN files parsed to extract FEN positions, with Stockfish labels for centipawn scores. Quality control used balanced datasets (early/mid/endgame) to avoid bias. Each example is encoded as 776‑dimensional input vector encoding piece placement, side to move, castling rights, and positional metadata.
- **Models**:
| Model | Setup | Mode | Metrics (Test set)|
|--------|------------|----------|---------|
| scikit-learn MLPRegressor | 4 hidden layers (512, 256, 128, 64) with ReLU, Adam optimizer | Low latency (Easy) | MAE: ~155 CP + R² = 0.56 |
| PyTorch MLP | Similar architecture with batch normalization and dropout for regularization. | Default mode (Medium) | MAE: ~146 CP + R² = 0.60 |
| Stockfish 16 | Used for testing true engine strength with highly accurate eval | Reference (Hard) | N/A |
- **Training Challenges and Solutions**:
    - **Data Structures**: 776 features led to curse of dimensionality (sparsity in high-dimensional space, increasing overfitting risk). I mitigated this by increasing my dataset and following the rule of thumb of $n \geq 10 \bullet d$. 
    - **Hyperparameter Tuning**: Initial $R^2$ of ~0.46 improved to 0.55+ via additional hidden layers and tuning alpha for regularization and learning rate.
    - **Cross-Validation**: 80/10/10 split with early stopping (10 epochs no improvement) to prevent overfitting.
    - **Quality Data**: Initially I was generating random board positions and model was performing sub-optimally. I solved this by using publicly available chess games from GMs. 
- **Insights**: Training revealed how positional features (e.g., pawn chains as linked lists) correlate with scores, deepening my chess understanding.

---

## AWS Architecture — Serverless and Scalable

The deployment uses a fully serverless blueprint, optimizing for scalability and cost ($0 under free tier).

```
[User Browser]
     ↓
AWS Amplify (React app + CloudFront CDN)
     ↓
Amazon API Gateway (REST API)
     ↓
AWS Lambda (Containerized FastAPI + Models)
     ↓
[Engine: MLP or Stockfish evaluation]
```

### Key Services
| AWS Service | Purpose |
|--------------|----------|
| **Amplify** | Hosts and builds the frontend with CI/CD and CloudFront caching. |
| **Lambda (ARM64)** | Runs the backend container with FastAPI, evaluation models, and alpha–beta logic. |
| **API Gateway** | Exposes REST endpoints with CORS and rate‑limiting . |
| **ECR** | Stores versioned container images for Lambda deployments with ARM64 optimization for 20% cost savings.. |
| **CloudWatch** | Tracks latency, error metrics, and cold starts. |


This architecture scales automatically, with no infrastructure management, emphasizing serverless principles for production readiness.

---

## Reflection and Personal Growth

Building this engine helped me combine my skills across a wide variety of domains (e.g. chess, machine learning, full-stack development). Some cool things I have learned:
- A simple 92 character FEN string can capture 776 unique features to encode a chess board. 
- alpha-beta search can reduce the branching factor from 32 to 3, pruning suboptimal boards. 
- Chess board evaluation is incredibly nuanced and complex, making it difficult to consistently score well with hard-coded rules, resulting in an ideal use-case for Machine Learning. 
- I learned more about serverless architecture using AWS.

As a 1600 ELO player, this project enhanced my understanding of chess data structures and positional nuances. Getting humbled by the Hard mode showed how AI reveals hidden weaknesses in human play.

---

### Skills Demonstrated
**React, Three.js, React Three Fiber, Redux Toolkit, TypeScript, FastAPI, AWS Lambda, API Gateway, ECR, Amplify, CloudWatch, scikit-learn, PyTorch, Algorithms, Alpha–Beta Search, Machine Learning, Serverless Architecture**
