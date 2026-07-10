# ⚽ World Cup 2026 Predictor
**Round-by-round calibrated ensemble forecasting for knockout football**

**Karandeep Shoker**  
[LinkedIn](https://www.linkedin.com/in/kshoker12/)

### Quick Access
- **[Live Forecast](https://kshoker12.github.io/World-Cup-Predictor/)** — Updated knockout forecast + round history  
- **[GitHub Repo](https://github.com/kshoker12/World-Cup-Predictor)** — training pipeline, Kaggle notebooks, simulation engine

## Abstract

This project forecasts FIFA World Cup knockout outcomes using a **calibrated ensemble** of three models.

- a **tabular model** (LightGBM) trained on pre-match indicators such as Elo strength, recent goals-for/goals-against, and tournament context,
- a **sequence model** (LSTM) trained on each team’s recent match history to capture time-dependent form that rolling averages can miss, and
- a **hierarchical Bayesian model** (Dixon–Coles style) that estimates each team’s latent attacking and defensive strength, with shrinkage toward the population mean so estimates stay stable when a national team has played relatively few international matches.

Each model outputs expected goals \\((\lambda&#95;h, \lambda&#95;a)\\), the Poisson rate parameters for home and away scoring. From these rates the system assigns a probability to every final score (e.g., 1–0, 2–1, 0–0). A Monte Carlo simulator then plays the knockout bracket forward many times. It draws a score for each match, breaks ties with extra time and penalties, and updates team form after each result so later rounds reflect earlier outcomes.

This project aims to predict the **2026 FIFA World Cup** winner and refresh that forecast **round by round**. At the start of each knockout round, newly completed results are added to the data and \\(80{,}000\\) simulations are run from the current bracket, producing updated probabilities for advancing and winning the tournament.

## 1 Introduction

### 1.1 Motivation

Tournament forecasts are harder than single-match predictions because uncertainty **compounds across rounds**. A team can be a slight favorite in every knockout match and still have a modest chance to win the cup once you chain four uncertain results together. For example, even a \\(55\\%\\) win chance repeated across four knockout matches yields only

$$
0.55^4 \approx 0.0915 \quad (\text{about } 9\\%)
$$

to win the tournament. This project therefore separates the problem into two steps:

1. **Match modeling:** predict how likely each scoreline is for a single match.
2. **Tournament simulation:** play out the full bracket many times to estimate advancement and tournament win probabilities.

### 1.2 Statistical Modelling Approach

**Scoreline:** A scoreline is the ordered pair \\((G&#95;h, G&#95;a)\\) of goals scored by the home team and away team respectively.

**Core assumption:** Each match is modeled by expected goals \\((\lambda&#95;h, \lambda&#95;a)\\). The simplest classical model assumes independent Poisson goals:

$$
G&#95;h \sim \text{Poisson}(\lambda&#95;h), \qquad
G&#95;a \sim \text{Poisson}(\lambda&#95;a)
$$

which implies a probability for any scoreline \\((i,j)\\):

$$
P(G&#95;h=i\,G&#95;a=j) = \text{Pois}(i;\lambda&#95;h) \cdot \text{Pois}(j;\lambda&#95;a)
$$

**Dixon–Coles correction:** The independent Poisson model above treats home and away goals as unrelated, it builds the probability of a scoreline by multiplying two separate Poisson terms. That assumption breaks down on low-scoring matches. When both teams play cautiously, goals remain low for *both* sides at once, so scorelines like 0–0 and 1–1 occur more often than the product of two independent rates would predict. On higher-scoring results (e.g., 2–3 or 3–4), both teams are clearly attacking and the independence approximation is much closer to reality.

Dixon and Coles ([Dixon & Coles, 1997](#ref-2)) fix this by scaling only the four lowest-score outcomes 0–0, 1–0, 0–1, and 1–1 with a multiplicative factor \\(\tau(i,j;\rho)\\) controlled by one parameter \\(\rho\\). The corrected scoreline probability \\(P&#95;{\text{DC}}(i,j)\\) is

$$
P&#95;{\text{DC}}(i,j) \propto \tau(i,j;\rho) \cdot \text{Pois}(i;\lambda&#95;h) \cdot \text{Pois}(j;\lambda&#95;a)
$$

\\(\rho\\) is a single parameter that controls how much 0–0, 1–0, 0–1, and 1–1 are adjusted away from independent Poisson; every other scoreline is unchanged. This Dixon–Coles correction aligns \\(P&#95;{\text{DC}}\\) with real international football, so regulation draws are sampled at realistic rates and the knockout simulator can resolve ties through extra time and shootouts before picking a winner.

**Knockout tie resolution:** A knockout match cannot end in a draw. The simulator follows the same order of resolution as a real knockout match:

1. **Regulation (always):** sample a scoreline \\((G&#95;h, G&#95;a)\\) from \\(P&#95;{\text{DC}}\\).
2. **Extra time (only if regulation ends tied):** sample an *additional* 30-minute score from \\(P&#95;{\text{DC}}\\) again, but with goal rates \\(\\lambda&#95;h/3\\) and \\(\\lambda&#95;a/3\\) (30 minutes vs. 90), and add it to the regulation total.
3. **Penalties (only if still tied after extra time):** declare a winner by a Bernoulli draw. This version does not fit a separate penalty model. Elo difference instead provides a simple strength-based tiebreak. The better-rated team is slightly more likely to win, but the probability stays near 50–50 because real shootouts are highly random. Let \\(\\Delta\\) be home Elo minus away Elo. The home-team shootout win probability is
$$
P(\text{home wins shootout}) = \frac{1}{1 + 10^{-\Delta / 400}}
$$
The stronger side is only mildly favored, and shootouts remain high-variance.

### 1.3 Dataset and Features

The primary training signal is international match history from the international results dataset ([martj42, 2025](#ref-1)), augmented with WC 2026 results as the tournament progresses. Club match data from Big-5 European leagues ([Understat](#ref-3), [eatpizzanot, 2025](#ref-4)) is used only to pretrain the LSTM sequence encoder ([Section 2.2](#22-sequence-model-lstm)).

At any match date, features are computed using only information available **strictly before kickoff**, by replaying match history in chronological order. This prevents data leakage, so future results cannot influence past predictions, and the same rule is enforced during simulation.

Core tabular features are summarized below. Precise definitions appear in [Appendix A](#appendix-a-feature-glossary):

- **Elo strength gap:** Elo is a running strength rating that rises after wins and falls after losses. Home Elo minus away Elo summarizes long-run team quality in one number, with positive values meaning the home side is rated stronger.
- **Recent scoring form:** Goals scored and conceded in each team’s last \\(k\\) matches (default \\(k=5\\)), compared as home-minus-away differences. This captures short-run attacking and defensive momentum beyond overall strength.
- **Decayed results form:** A weighted record of recent wins, draws, and losses, where more recent matches count more. It signals whether a team is on an upward or downward run of results.
- **Head-to-head:** A decayed goal-difference summary from previous matches between the same two teams, capturing matchup-specific history that global ratings may miss.
- **Context:** Whether the venue is neutral (removing home advantage) and a scalar for tournament stakes (e.g., friendly vs. World Cup), so the model can treat high-pressure matches differently.

### 1.4 Paper Overview

The remainder of this paper is organized as follows.

- **Section 2 (Architecture):** discusses the three match models (tabular LightGBM, LSTM sequence model, and hierarchical Bayesian Dixon–Coles), explains how calibration combines them into one ensemble that outputs \\((\lambda&#95;h, \lambda&#95;a)\\) per match, summarizes the end-to-end inference graph, and reports held-out test metrics comparing each component to the ensemble.

- **Section 3 (Tournament simulation):** describes pre-round setup, one bracket simulation (feature lookup, score sampling, tie breaks, state updates), converting \\(N\\) simulation counts into probabilities, and the live round refresh workflow.

- **Section 4 (Round-by-round WC 2026 results):** reports live tournament output, including pre-round forecasts and post-round scoring of those picks, and the current advancement and title probabilities from 80,000 simulations.

- **Section 5 (Conclusion):** summarizes what the ensemble and simulation pipeline achieve, and notes the main limitations and directions for future work.

- **Section 6 (References):** lists data sources and methodological papers. Feature definitions are collected in [Appendix A](#appendix-a-feature-glossary). AI use is disclosed in [Appendix B](#appendix-b-ai-disclosure).

---

## 2 Architecture

Every match model in this system answers the same question: given pre-match information, what are the expected home and away goal rates? Each model \\(m \\in \\{\\text{GBM}, \\text{NN}, \\text{Bayes}\\}\\) maps tabular features \\(x\\) and team histories \\(s&#95;h, s&#95;a\\) (for the sequence model) to a pair of rates

$$
(\lambda&#95;h, \lambda&#95;a) = f&#95;m(x, s&#95;h, s&#95;a).
$$

The three models are trained independently on international match history (chronological train/validation/test splits, with no future leakage). A calibration step then rescales their outputs and combines them into one ensemble rate per match. The hyperparameters in this section are from the production Kaggle training run ([run_kaggle_pipeline.py](https://github.com/kshoker12/World-Cup-Predictor/blob/main/scripts/run_kaggle_pipeline.py) with `--profile kaggle`, see [kaggle.yaml](https://github.com/kshoker12/World-Cup-Predictor/blob/main/config/profiles/kaggle.yaml)), which produced the deployed WC 2026 forecasts. Training scripts live in [`scripts/`](https://github.com/kshoker12/World-Cup-Predictor/tree/main/scripts) and model code in [`src/worldcup_predictor/models/`](https://github.com/kshoker12/World-Cup-Predictor/tree/main/src/worldcup_predictor/models).

### 2.1 Tabular model (LightGBM Poisson)

The tabular model treats each side’s goal count as a Poisson random variable and uses gradient-boosted decision trees to predict the rate:

$$
\begin{aligned}
G&#95;h &\sim \text{Poisson}(\lambda&#95;h), \qquad \log \lambda&#95;h = f&#95;h(x), \\\\
G&#95;a &\sim \text{Poisson}(\lambda&#95;a), \qquad \log \lambda&#95;a = f&#95;a(x).
\end{aligned}
$$

The log link guarantees \\(\lambda \gt 0\\). Each \\(f\\) is a sum of \\(T\\) small regression trees trained sequentially. At iteration \\(t\\), a new tree fits the remaining Poisson error from iterations \\(1, \ldots, t-1\\). Splits are chosen on tabular features such as Elo gap, recent goals for/against, and tournament context ([Section 1.3](#13-dataset-and-features)).

Two separate boosters are fit, one for \\(G&#95;h\\) and one for \\(G&#95;a\\), each minimizing Poisson deviance on the training split and monitored on validation with early stopping.

**Training:** LightGBM Poisson regression with early stopping on the validation split. Production settings:

- learning rate \\(0.05\\)
- `num_leaves` \\(127\\)
- up to \\(1200\\) boosting rounds
- early stopping after \\(60\\) rounds without improvement

**Role in the ensemble:** LightGBM is the strongest reader of structured pre-match indicators. It anchors the ensemble on Elo, rolling form, and matchup context.

Training script: [train_gbm.py](https://github.com/kshoker12/World-Cup-Predictor/blob/main/scripts/train_gbm.py)  
Model implementation: [gbm.py](https://github.com/kshoker12/World-Cup-Predictor/blob/main/src/worldcup_predictor/models/gbm.py)

### 2.2 Sequence model (LSTM)

International teams play far fewer matches than club teams, so a single rolling average can miss how form is evolving. The sequence model reads each team’s recent match history as a time series and compresses it into a fixed vector before predicting goal rates.

Let \\(S&#95;h\\) and \\(S&#95;a\\) denote the last \\(L=10\\) matches for the home and away teams. Each match in the sequence is encoded as a \\(10\\)-dimensional vector:

- goals scored
- goals conceded
- opponent Elo (divided by \\(1500\\))
- log days since the previous match
- home/neutral indicator
- tournament importance
- normalized expected goals (club pretraining only)
- normalized expected goals against (club pretraining only)
- normalized shots (club pretraining only)
- feature mask (marks whether club stats are present)

An LSTM encoder maps each history to a hidden state:

$$
h&#95;h = \text{LSTM}(S&#95;h), \qquad h&#95;a = \text{LSTM}(S&#95;a).
$$

The two states are concatenated and passed through a two-layer MLP with a softplus output so rates stay positive:

$$
(\lambda&#95;h, \lambda&#95;a) = \text{softplus}\bigl(\text{MLP}([h&#95;h; h&#95;a])\bigr).
$$

**Architecture:** Two LSTM layers (input dimension \\(10\\), hidden dimension \\(256\\), dropout \\(0.1\\)) share weights across home and away histories. The final hidden states are concatenated and passed through a two-layer MLP head (\\(512 \to 256 \to 2\\)). The network has \\(\approx 0.93\\text{M}\\) trainable parameters.

**Training:** The encoder is first pretrained on club football from Big-5 European leagues ([Understat](#ref-3), [eatpizzanot, 2025](#ref-4)), where xG and shot features are available for most timesteps. It is then finetuned on international matches (club-only fields are zeroed and masked). Optimization uses Adam with early stopping on validation Poisson deviance:

- weight decay \\(10^{-4}\\)
- pretrain: \\(50\\) epochs, learning rate \\(10^{-3}\\), batch size \\(512\\)
- finetune: \\(60\\) epochs, learning rate \\(10^{-4}\\), batch size \\(256\\)
- early-stopping patience \\(10\\) epochs

**Dual use:** The LSTM contributes in two ways:

- it predicts \\((\lambda&#95;h, \lambda&#95;a)\\) directly as an ensemble member
- its encoder outputs are fed into LightGBM as additional inputs ([Section 2.1](#21-tabular-model-lightgbm-poisson)), giving \\(256\\) home-minus-away embedding-difference features beyond the \\(8\\) tabular columns

**Role in the ensemble:** The sequence model captures trajectory and opponent quality in recent games, especially patterns that fixed windows like “last 5 goals” smooth over.

Model implementation: [model.py](https://github.com/kshoker12/World-Cup-Predictor/blob/main/src/worldcup_predictor/models/nn/model.py)  
Sequence features: [sequences.py](https://github.com/kshoker12/World-Cup-Predictor/blob/main/src/worldcup_predictor/models/sequences.py)

### 2.3 Bayesian hierarchical team model (Dixon–Coles style)

The Bayesian component is a classical Dixon–Coles Poisson model with team-level random effects. For a match between home team \\(h\\) and away team \\(a\\),

$$
\begin{aligned}
G&#95;h &\sim \text{Poisson}(\lambda&#95;h), \qquad \log \lambda&#95;h = \alpha + a&#95;h - d&#95;a + \beta^\top x \\\\
G&#95;a &\sim \text{Poisson}(\lambda&#95;a), \qquad \log \lambda&#95;a = \alpha + a&#95;a - d&#95;h + \beta^\top x
\end{aligned}
$$

where \\(\alpha\\) is a global intercept, \\(a&#95;t\\) is team \\(t\\)'s attacking strength, \\(d&#95;t\\) is its defensive strength, and \\(\beta\\) weights the same tabular features \\(x\\) used elsewhere. Attack boosts a team's own scoring rate. Defense suppresses the opponent's. Team effects are centered so their sample mean is zero, which keeps them identifiable.

Sparse national-team schedules make raw per-team estimates noisy. The model therefore places shrinkage priors on latent strengths,

$$
a&#95;t \sim \mathcal{N}(0, \sigma&#95;a^2), \qquad d&#95;t \sim \mathcal{N}(0, \sigma&#95;d^2),
$$

with \\(\sigma&#95;a, \sigma&#95;d\\) learned from data. Teams with few matches are pulled toward the population average; teams with long histories can separate. Low-score correlation is handled by the same Dixon–Coles \\(\rho\\) parameter from Section 1.2, with \\(\rho \sim \text{Uniform}(-0.2, 0.1)\\).

**Training:** The model is fit with PyMC using NUTS sampling on international matches from 2000 onward on the **train split only** (validation is reserved for calibration). Production sampler settings:

- \\(4\\) chains
- \\(1000\\) tuning steps
- \\(1000\\) posterior draws
- target acceptance \\(0.9\\)

The fitted posterior includes team attack/defense means, \\(\beta\\), and \\(\rho\\) (production \\(\hat{\rho} \approx -0.05\\) across \\(\approx 9{,}500\\) matches and \\(268\\) teams).

**Inference:** At prediction time, posterior means \\(\hat{a}&#95;t\\), \\(\hat{d}&#95;t\\), and \\(\hat{\beta}\\) are plugged into the log-rate equations above to produce \\((\lambda&#95;h, \lambda&#95;a)\\). The posterior mean of \\(\rho\\) is also stored and reused when converting ensemble rates to score distributions during simulation.

**Role in the ensemble:** The Bayesian model supplies a statistically disciplined baseline with explicit team strengths and built-in shrinkage for data-sparse sides.

Training script: [train_bayesian.py](https://github.com/kshoker12/World-Cup-Predictor/blob/main/scripts/train_bayesian.py)  
Model implementation: [bayesian/](https://github.com/kshoker12/World-Cup-Predictor/tree/main/src/worldcup_predictor/models/bayesian)

### 2.4 Calibration and ensemble

The three models each output \\((\lambda&#95;h, \lambda&#95;a)\\) on slightly different scales. Calibration aligns them on held-out validation matches in two steps.

1. **Per-model rate scaling:** Each model \\(m\\) receives separate positive scalers for home and away goals (optimized in \\([0.5, 2.0]\\)):

$$
\tilde{\lambda}&#95;{m,h} = s&#95;{m,h}\lambda&#95;{m,h}, \qquad
\tilde{\lambda}&#95;{m,a} = s&#95;{m,a}\lambda&#95;{m,a}.
$$

2. **Ensemble averaging:** Scaled rates are combined with weights \\(w&#95;m \ge 0\\), \\(\sum&#95;{m} w&#95;m = 1\\):

$$
\lambda&#95;h = \sum&#95;{m} w&#95;m \cdot \tilde{\lambda}&#95;{m,h}, \qquad
\lambda&#95;a = \sum&#95;{m} w&#95;m \cdot \tilde{\lambda}&#95;{m,a}.
$$

Weights are chosen to minimize validation Poisson deviance. Each active model is constrained to carry at least \\(10\\%\\) of the total weight so the optimizer cannot collapse the ensemble onto a single predictor.

**Production weights after calibration:**

| Component | Weight \\(w\\) | What it contributes |
|-----------|--------------:|---------------------|
| LightGBM | 0.43 | Structured pre-match indicators (Elo, form, context) plus LSTM embedding differences |
| LSTM | 0.33 | Time-dependent form from the last \\(10\\) matches per team |
| Bayesian | 0.24 | Hierarchical team attack/defense with shrinkage toward the population mean |

Together, LightGBM handles tabular structure, the LSTM adds sequential form, and the Bayesian term regularizes team-specific effects. The ensemble rate \\((\lambda&#95;h, \lambda&#95;a)\\) is what the knockout simulator uses to build score probabilities ([Section 3](#3-tournament-simulation-knockout-monte-carlo)).

Calibration code: [calibration/](https://github.com/kshoker12/World-Cup-Predictor/tree/main/src/worldcup_predictor/calibration)  
Pipeline script: [run_kaggle_pipeline.py](https://github.com/kshoker12/World-Cup-Predictor/blob/main/scripts/run_kaggle_pipeline.py) (fits scaling, weights, and writes `calibration.json`)

### 2.5 Architectural Overview

<figure id="fig-ensemble-architecture" class="arch-diagram">
  <img src="worldcup_content/world_cup_ensemble.svg" width="1292" height="982" alt="World Cup ensemble architecture: feature pipeline into parallel LSTM, LightGBM, and Bayesian models, then calibration (0.43/0.33/0.24) and Dixon-Coles prediction layer" />
  <figcaption>
    Ensemble match-prediction architecture. The feature pipeline replays history for match \(h\) vs \(a\) into Elo, form, head-to-head, and match sequences. Three models then predict goal rates in parallel: the LSTM turns the last 10 matches into rates plus a 256-dim embedding; LightGBM fits Poisson rates on 8 tabular features plus those embeddings; the Bayesian model combines home attack vs away defense (and the reverse) with tabular features under shrinkage. The calibration layer scales each rate and takes a weighted sum (0.43 / 0.33 / 0.24) to produce ensemble \((\lambda_h, \lambda_a)\). The prediction layer builds the Dixon–Coles grid \(P_{\mathrm{DC}}\) and samples a scoreline \(G_h, G_a\) (or W/D/L / knockout tiebreak).
  </figcaption>
</figure>

### 2.6 Held-out test evaluation

All models are trained only on the training split. We report match-level quality on the held-out **test** split (\\(8{,}192\\) international matches) using the production artifacts. Metrics are computed by [evaluate_test_metrics.py](https://github.com/kshoker12/World-Cup-Predictor/blob/main/scripts/evaluate_test_metrics.py).

**Goal-rate metrics** measure how well predicted expected goals \\((\lambda&#95;h, \lambda&#95;a)\\) match realized scores:

- **Poisson deviance** (primary training loss): average negative log-likelihood under independent Poisson goals; lower is better.
- **Goal MAE**: mean absolute error between \\(\lambda\\) and the actual goal count; lower is better.

**Outcome metrics** convert \\((\lambda&#95;h, \lambda&#95;a)\\) to win/draw/loss probabilities via the Dixon–Coles score grid (\\(\rho = -0.05\\)) and score the full distribution:

- **WDL log loss**: multiclass log loss for the actual result; lower is better.
- **Outcome accuracy**: fraction of matches where the highest-probability outcome is correct; higher is better.

A **naive mean** baseline predicts every match at the training-set average score (\\(\approx 1.5\\)–\\(1.2\\) home/away) and anchors how much signal the learned models extract.

| Model | Poisson deviance | Goal MAE | WDL log loss | Outcome accuracy |
|-------|----------------:|---------:|-------------:|-----------------:|
| LightGBM | 2.393 | 0.946 | 0.875 | 60.0% |
| LSTM | 2.426 | 0.950 | 0.901 | 58.1% |
| Bayesian | 2.653 | 1.009 | 0.936 | 58.1% |
| **Ensemble** | **2.352** | **0.934** | **0.871** | **60.4%** |
| Naive mean | 3.171 | 1.126 | 1.056 | 47.8% |

LightGBM is the strongest single model on goal-rate error, which matches its role as the tabular backbone. The LSTM adds complementary sequential signal (slightly worse in isolation but weighted at \\(0.33\\) in the ensemble). The Bayesian term is the weakest point predictor on this split, yet its \\(0.24\\) weight still helps by regularizing extreme team rates.

The calibrated ensemble improves on the best single model across every reported metric: Poisson deviance by \\(1.7\\%\\), WDL log loss by \\(0.5\\%\\), and outcome accuracy from \\(60.0\\%\\) to \\(60.4\\%\\). The gain is modest in absolute terms (international football is noisy), but consistent: combining heterogeneous predictors beats any one model alone, which is why the knockout simulator uses the ensemble rate from [Section 2.4](#24-calibration-and-ensemble).

---

## 3 Tournament simulation (knockout Monte Carlo)

Knockout forecasting has two stages. First, rebuild each team's pre-match features from all results available before the round. Second, run \\(N\\) independent bracket simulations, each playing out every remaining matches by sampling scorelines from the calibrated ensemble ([Section 2.4](#24-calibration-and-ensemble)). Reported probabilities are event frequencies: the fraction of simulations in which an outcome occurs.

Production runs use \\(N = 80{,}000\\) ([kaggle.yaml](https://github.com/kshoker12/World-Cup-Predictor/blob/main/config/profiles/kaggle.yaml)). Simulation code: [simulation/](https://github.com/kshoker12/World-Cup-Predictor/tree/main/src/worldcup_predictor/simulation).

### 3.1 Pre-round setup

Before drawing any scores, the forecast script does four things once:

1. **Load match history:** read international results and append new WC 2026 scores from [wc2026_results.csv](https://github.com/kshoker12/World-Cup-Predictor/blob/main/docs/data/wc2026_results.csv).
2. **Snapshot team state:** replay all matches with kickoff strictly before the round date through the training feature pipeline (Elo, rolling form, head-to-head, LSTM histories), producing pre-match features for every team entering the round.
3. **Load models:** read the calibrated ensemble from disk ([Section 2.4](#24-calibration-and-ensemble)).
4. **Fix the bracket:** set the knockout pairings for the current round (e.g. [world_cup_2026_knockout.yaml](https://github.com/kshoker12/World-Cup-Predictor/blob/main/config/tournaments/world_cup_2026_knockout.yaml)).

Steps 1–2 capture *what we know about each team so far*; steps 3–4 capture *how strong they are and who plays whom*.

### 3.2 One bracket simulation

Each simulation is one full play-through of the remaining knockout tree, starting from the same pre-round snapshot and using its own random seed. Simulations do not share state.

**Per match:** (home \\(h\\) vs away \\(a\\))

1. **Features:** build tabular inputs and the last \\(L=10\\) international matches per team from the current pipeline state.
2. **Rates:** predict \\((\lambda&#95;h, \lambda&#95;a)\\) from the calibrated ensemble.
3. **Score grid:** form \\(P&#95;{\text{DC}}(i,j)\\) from those rates and \\(\rho = -0.05\\) ([Section 1.2](#12-statistical-modelling-approach)).
4. **Regulation:** sample a scoreline \\((G&#95;h, G&#95;a)\\) from the grid.
5. **Ties:** if regulation is drawn, sample extra time from \\(P&#95;{\text{DC}}\\) at rates \\(\lambda&#95;h/3\\) and \\(\lambda&#95;a/3\\); if still tied, pick a penalty winner from the Elo-based rule in [Section 1.2](#12-statistical-modelling-approach).
6. **Update:** record the winner, advance team state (Elo, form, sequences) as if the result were real, and continue to the next match.

Winners advance round by round until one team is champion. The next simulation resets to the pre-round snapshot.

### 3.3 From simulation counts to probabilities

For any event \\(E\\) (a match winner, a team reaching a round, or a title), let \\(C(E)\\) be the number of simulations in which \\(E\\) occurs. The estimated probability is

$$
\hat{P}(E) = \frac{C(E)}{N}.
$$

**Match win probability:** for match \\((h, a)\\), \\(\hat{P}(h \text{ wins}) = C(h \text{ beats } a) / N\\). Away win probability is \\(1 - \hat{P}(h \text{ wins})\\) because every knockout match has exactly one winner.

**Advancement probability:** \\(\hat{P}(T \text{ reaches round } R) = C(T \text{ alive at } R) / N\\).

**Title probability:** \\(\hat{P}(T \text{ champion}) = C(T \text{ wins tournament}) / N\\).

With \\(N = 80{,}000\\), sampling error on a \\(50\\%\\) probability is about \\(0.2\\) percentage points, so reported percentages are stable to roughly one decimal place.

### 3.4 Live round refresh

At the start of each knockout round during the tournament:

1. **Ingest results:** append completed matches to [wc2026_results.csv](https://github.com/kshoker12/World-Cup-Predictor/blob/main/docs/data/wc2026_results.csv).
2. **Update bracket:** set the next round's fixtures in the tournament config.
3. **Run setup** ([Section 3.1](#31-pre-round-setup)).
4. **Simulate:** run \\(N = 80{,}000\\) bracket simulations ([Sections 3.2–3.3](#32-one-bracket-simulation)).
5. **Publish:** write forecast JSON (match, advancement, and title probabilities) and refresh the [GitHub Pages](https://kshoker12.github.io/World-Cup-Predictor/) view.

Forecast script: [run_wc2026_forecast.py](https://github.com/kshoker12/World-Cup-Predictor/blob/main/scripts/run_wc2026_forecast.py).

---

## 4 Round-by-round WC 2026 results

Live forecasts are published [here](https://kshoker12.github.io/World-Cup-Predictor/). This project began at the **start of the Round of 16**. Earlier group-stage matches were not forecast by this system.

All win probabilities below come from the Monte Carlo simulator in [Section 3](#3-tournament-simulation-knockout-monte-carlo). For each round, \\(80{,}000\\) independent bracket simulations are run from the current knockout tree. A match win probability is \\(\hat{P}(h \text{ wins}) = C(h \text{ beats } a) / 80{,}000\\) ([Section 3.3](#33-from-simulation-counts-to-probabilities)). The **predicted winner** is the team with the higher win probability.

After each round is played, pre-round picks are scored against the actual results and archived in [round_history.json](https://github.com/kshoker12/World-Cup-Predictor/blob/main/docs/data/round_history.json).

### 4.1 Current tournament outlook

The latest forecast ([wc2026_forecast_qf.json](https://github.com/kshoker12/World-Cup-Predictor/blob/main/docs/data/wc2026_forecast_qf.json), \\(80{,}000\\) simulations from the quarter-final bracket) gives the current title and advancement odds for the eight remaining teams:

| Team | \\(P(\\text{reach SF})\\) | \\(P(\\text{reach final})\\) | \\(P(\\text{champion})\\) |
|------|------------------------:|---------------------------:|------------------------:|
| France | 69.58% | 43.19% | 27.39% |
| Argentina | 79.85% | 45.90% | 22.47% |
| Spain | 66.55% | 33.65% | 21.30% |
| England | 63.29% | 32.38% | 12.59% |
| Morocco | 30.42% | 13.61% | 6.26% |
| Norway | 36.71% | 15.29% | 4.30% |
| Belgium | 33.45% | 9.56% | 4.11% |
| Switzerland | 20.15% | 6.43% | 1.58% |

France leads on title probability (27.39%), but Argentina has the highest chance of reaching the final (45.90%) because its quarter-final matchup is the most favorable on paper.

### 4.2 Round of 16

Pre-round forecast archived as [forecast_pre_r16.json](https://github.com/kshoker12/World-Cup-Predictor/blob/main/docs/data/history/forecast_pre_r16.json) (\\(80{,}000\\) simulations). Per-match predictions and outcomes:

| Match | \\(P(\\text{home wins})\\) | \\(P(\\text{away wins})\\) | Predicted winner | Actual winner | Score | Correct |
|-------|------------------------:|-------------------------:|------------------|---------------|-------|:-------:|
| Paraguay vs France | 21.01% | 78.99% | France | France | 0–1 | ✓ |
| Canada vs Morocco | 33.83% | 66.17% | Morocco | Morocco | 0–3 | ✓ |
| Brazil vs Norway | 76.03% | 23.97% | Brazil | Norway | 1–2 | ✗ |
| Mexico vs England | 45.43% | 54.57% | England | England | 2–3 | ✓ |
| Spain vs Portugal | 69.00% | 31.00% | Spain | Spain | 1–0 | ✓ |
| United States vs Belgium | 45.12% | 54.88% | Belgium | Belgium | 1–4 | ✓ |
| Argentina vs Egypt | 83.03% | 16.97% | Argentina | Argentina | 3–2 | ✓ |
| Switzerland vs Colombia | 48.33% | 51.67% | Colombia | Switzerland | 1–0 | ✗ |

**Accuracy: 6/8 correct** (75.0%). The two misses were Brazil–Norway (model favored Brazil) and Switzerland–Colombia (model slightly favored Colombia).

### 4.3 Quarter-finals

Pre-round forecast from the QF bracket ([wc2026_forecast_qf.json](https://github.com/kshoker12/World-Cup-Predictor/blob/main/docs/data/wc2026_forecast_qf.json), \\(80{,}000\\) simulations). Quarter-final fixtures and predicted winners:

| Match | \\(P(\\text{home wins})\\) | \\(P(\\text{away wins})\\) | Predicted winner | Actual winner | Score | Correct |
|-------|------------------------:|-------------------------:|------------------|---------------|-------|:-------:|
| Argentina vs Switzerland | 79.85% | 20.15% | Argentina | TBA | TBA | TBA |
| Belgium vs Spain | 33.45% | 66.55% | Spain | TBA | TBA | TBA |
| Morocco vs France | 30.42% | 69.58% | France | TBA | TBA | TBA |
| Norway vs England | 36.71% | 63.29% | England | TBA | TBA | TBA |

**Accuracy: TBA** (scored after the quarter-finals are played).

### 4.4 Semi-finals

| Match | \\(P(\\text{home wins})\\) | \\(P(\\text{away wins})\\) | Predicted winner | Actual winner | Score | Correct |
|-------|------------------------:|-------------------------:|------------------|---------------|-------|:-------:|
| TBA | TBA | TBA | TBA | TBA | TBA | TBA |
| TBA | TBA | TBA | TBA | TBA | TBA | TBA |

**Accuracy: TBA.** Forecast and scoring will be published when the semi-final bracket is set.

### 4.5 Final

| Match | \\(P(\\text{home wins})\\) | \\(P(\\text{away wins})\\) | Predicted winner | Actual winner | Score | Correct |
|-------|------------------------:|-------------------------:|------------------|---------------|-------|:-------:|
| TBA | TBA | TBA | TBA | TBA | TBA | TBA |

**Accuracy: TBA.** Forecast and scoring will be published when the final matchup is known.

---

## 5 Conclusion

This project forecasts knockout World Cup matches with a calibrated ensemble of three models (LightGBM, LSTM, hierarchical Bayesian Dixon–Coles) instead of a single end-to-end predictor. LightGBM reads tabular pre-match indicators, the LSTM encodes recent match sequences, and the Bayesian term regularizes team attack and defense. Calibration merges their outputs into one \\((\lambda&#95;h, \lambda&#95;a)\\) pair per match ([Section 2](#2-architecture)).

On held-out international matches, the ensemble beats every component on goal-rate and outcome metrics ([Section 2.6](#26-held-out-test-evaluation)). Dixon–Coles score sampling and \\(80{,}000\\) independent bracket simulations turn those rates into match, advancement, and title probabilities ([Section 3](#3-tournament-simulation-knockout-monte-carlo)). The Round of 16 forecast scored 6/8; the live quarter-final outlook assigns plausible odds to the eight remaining teams ([Section 4](#4-round-by-round-wc-2026-results)).

**Limitations:**

- **Penalty shootouts:** ties after extra time are broken by an Elo-based Bernoulli rule ([Section 1.2](#12-statistical-modelling-approach)), not a model trained on historical shootout data.
- **Live evaluation:** only one knockout round has been scored so far; the same pre-round forecast and post-round workflow should run through the quarter-finals, semi-finals, and final.
- **Uncertainty reporting:** published probabilities do not yet include Monte Carlo standard errors, so small round-to-round changes are hard to judge statistically.

**Future work:** Fit a dedicated penalty model and complete multi-round scoring ([Appendix A](#appendix-a-feature-glossary)).

---

## 6 References

<p id="ref-1" class="reference-entry"><strong>1.</strong> <strong><a href="https://github.com/martj42/international_results">International football results from 1872 to 2025</a></strong><br>
martj42 (2025).<br>
<a href="https://github.com/martj42/international_results">GitHub repository</a><br>
Primary international match history used for training, validation, and feature replay.</p>

<p id="ref-2" class="reference-entry"><strong>2.</strong> <strong><a href="https://doi.org/10.1111/1467-9884.00091">Modelling association football scores and inefficiencies in the football betting market</a></strong><br>
Dixon, M. J., &amp; Coles, S. G. (1997).<br>
<em>Journal of the Royal Statistical Society: Series C</em>, 46(2), 265–280.<br>
<a href="https://doi.org/10.1111/1467-9884.00091">DOI: 10.1111/1467-9884.00091</a><br>
Introduces the Dixon–Coles low-score correlation correction used when sampling regulation and extra-time scorelines.</p>

<p id="ref-3" class="reference-entry"><strong>3.</strong> <strong><a href="https://understat.com">Understat football match data</a></strong><br>
Understat.<br>
<a href="https://understat.com">understat.com</a><br>
Club xG and shot features for Big-5 European leagues used to pretrain the LSTM sequence encoder.</p>

<p id="ref-4" class="reference-entry"><strong>4.</strong> <strong><a href="https://github.com/eatpizzanot/soccer-dataset">soccer-dataset (Big-5 proxy)</a></strong><br>
eatpizzanot (2025).<br>
<a href="https://github.com/eatpizzanot/soccer-dataset">GitHub repository</a><br>
Supplementary club match fixtures and stats merged with Understat for LSTM pretraining.</p>

---

## Appendix A Feature glossary

<a id="appendix-a-feature-glossary"></a>

| Name | Definition (at match kickoff) |
|------|------------------------------|
| Elo | A running rating updated after each match. Higher values mean stronger teams. |
| `elo_diff` | Home Elo − Away Elo (in Elo points). |
| `gf_last_5_diff` | (Home goals scored in last 5 matches) − (Away goals scored in last 5). |
| `ga_last_5_diff` | (Home goals conceded in last 5 matches) − (Away goals conceded in last 5). |
| `form_diff` | Exponentially-decayed results signal (home − away), where recent results weigh more. |
| `h2h_gd_weighted` | Exponentially-decayed head-to-head goal difference (home − away). |
| `is_neutral` | Whether the match is played at a neutral venue. |
| `tournament_importance` | A scalar representing the stakes of the match (e.g., friendly vs. World Cup). |

---

## Appendix B AI disclosure

<a id="appendix-b-ai-disclosure"></a>

AI tools (primarily Cursor) were used in two places after the system design was already fixed.

1. **Implementation:** A locked [technical specification](https://github.com/kshoker12/World-Cup-Predictor/blob/main/.cursor/skills/world-cup-predictor-spec/reference.md) was written first (models, features, splits, calibration, and simulation rules). AI then assisted with coding against that blueprint: scaffolding modules, wiring the training and forecast pipelines, and iterating on tests. Design choices and final code were reviewed and owned by the author.
2. **Write-up:** AI helped format this paper (math rendering, section structure, and prose cleanup). Technical claims, results, and the narrative were checked and edited by the author.

The specification that guided implementation lives in the project repository under [`.cursor/skills/world-cup-predictor-spec/`](https://github.com/kshoker12/World-Cup-Predictor/tree/main/.cursor/skills/world-cup-predictor-spec).
