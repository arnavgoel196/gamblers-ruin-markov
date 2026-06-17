# Gambler's Ruin — Markov Chain Analysis

A progressive study of the **Gambler's Ruin problem** using **Discrete Time Markov Chains (DTMCs)**, implemented from scratch in Python.

The project models a gambler who starts with `k` dollars and plays a game where they win $1 with probability `p` and lose $1 with probability `q = 1 - p` each round. Four increasingly complex variants are analyzed, covering closed-form derivations, linear system solving, 2D state space expansion, and stationary distribution computation.

---

## Models

### Question 1 — Timid Strategy (Bet $1 every round)

The classic Gambler's Ruin. The gambler bets exactly $1 each round. The game ends when wealth hits 0 (ruin) or N (goal).

**Computes:**
- Win probability W(k) — derived analytically from the recurrence `W(k) = p·W(k+1) + q·W(k-1)`
- Expected game duration D(k) — derived from `D(k) = 1 + p·D(k+1) + q·D(k-1)`

**Closed-form results:**

| Case | Win Probability | Expected Duration |
|------|----------------|-------------------|
| Fair (p = 0.5) | k / N | k(N - k) |
| Unfair (p ≠ 0.5) | (1 - (q/p)^k) / (1 - (q/p)^N) | [k - N·W(k)] / (q - p) |

---

### Question 2 — Bold Strategy (Bet as much as possible)

The gambler bets aggressively to reach N in as few rounds as possible:
- If `k < N/2`: bet all `k` (go to 2k or 0)
- If `k ≥ N/2`: bet `N - k` (go to N or 2k - N)

Since jumps are non-uniform, no closed-form exists. Instead, a transition matrix is built and the system `A · W = b` is solved using `numpy.linalg.solve`.

**Key insight:** Bold strategy is provably optimal when the game is unfair (p < 0.5) — minimizing rounds played minimizes exposure to an unfavorable game.

---

### Question 3 — Timid Strategy with Quit Rules

The gambler now also quits if:
- Wealth drops to a floor `t` (protect capital)
- Win streak reaches `W` consecutive wins (quit while ahead)

**State space expansion:** Since streak length matters, the state is now `(k, s)` — wealth and current streak. This 2D state space is flattened to 1D using `idx(k, s) = (k - t) * (W + 1) + s` and solved as a linear system.

**Terminal conditions:**
- `k = t` → floor hit
- `k = N` → goal reached  
- `s = W` → streak reached

---

### Question 4 — General Markov Chain (State-dependent transitions)

The most general model. Each state `k` has its own transition probabilities:
- `p[k]` — move up (wealth +1)
- `q[k]` — move down (wealth -1)
- `r[k]` — stay at k (round is a draw)

**Computes three things:**

1. **Stationary Distribution π** — fraction of time spent at each state in the long run. Solved via `(P - I)^T · π = 0` with `Σπ[k] = 1`, using `numpy.linalg.lstsq`.

2. **Expected Time from state a to state b** — built as a linear system from the per-state recurrence and solved with `lstsq`.

3. **Expected Wealth** — `Σ k · π[k]` under the stationary distribution.

---

## Installation

```bash
git clone https://github.com/your-username/gamblers-ruin-markov.git
cd gamblers-ruin-markov
pip install numpy
```

---

## Usage

```bash
python gamblers_ruin.py
```

This runs all four models with example inputs and prints results.

You can also import individual functions:

```python
from gamblers_ruin import q1_win_probability, q1_expected_duration

# Start with $3, goal is $10, win prob per round = 0.4
print(q1_win_probability(k=3, N=10, p=0.4))    # 0.0419
print(q1_expected_duration(k=3, N=10, p=0.4))  # 12.9044
```

---

## Concepts Covered

- Discrete Time Markov Chains (DTMCs)
- Absorbing and transient states
- Recurrence relations and closed-form derivations
- Transition matrix construction and linear system solving (`numpy.linalg.solve`)
- 2D state space expansion and flattening
- Stationary distributions (`numpy.linalg.lstsq`)
- Bold vs timid strategy comparison (optimal betting theory)

---

## Requirements

- Python 3.x
- NumPy

---
  
## Live Demo
[Interactive Calculator](https://arnavgoel196.github.io/gamblers-ruin-markov/)
