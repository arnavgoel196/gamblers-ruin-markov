"""
Gambler's Ruin — Markov Chain Analysis
=======================================
Four progressively complex models of the Gambler's Ruin problem,
analyzed using Discrete Time Markov Chains.
"""

import numpy as np


# =============================================================================
# QUESTION 1 — Timid Strategy (Bet $1 every round)
# =============================================================================

def q1_win_probability(k, N, p):
    """
    Probability of reaching N before 0, starting from k.
    Betting $1 every round.
    """
    q = 1 - p
    if p == 0.5:
        return k / N
    else:
        r = q / p
        return (1 - r**k) / (1 - r**N)


def q1_expected_duration(k, N, p):
    """
    Expected number of rounds until game ends (win or ruin),
    starting from k. Betting $1 every round.
    """
    q = 1 - p
    if p == 0.5:
        return k * (N - k)
    else:
        W = q1_win_probability(k, N, p)
        return (k - N * W) / (q - p)


# =============================================================================
# QUESTION 2 — Bold Strategy (Bet as much as possible each round)
# =============================================================================

def q2_win_probability(N, p):
    """
    Probability of reaching N before 0 for all starting states,
    using the bold strategy.
    Returns a vector W where W[k] = win probability from state k.
    """
    q = 1 - p
    A = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)

    # Boundary conditions
    A[0][0] = 1;  b[0] = 0
    A[N][N] = 1;  b[N] = 1

    for k in range(1, N):
        A[k][k] = 1
        if k < N / 2:
            # Bet all k: go to 2k or 0
            A[k][2 * k] = -p
            b[k] = 0
        else:
            # Bet N-k: go to N or 2k-N
            A[k][2 * k - N] = -q
            b[k] = p

    W = np.linalg.solve(A, b)
    return W


def q2_expected_duration(N, p):
    """
    Expected number of rounds until game ends for all starting states,
    using the bold strategy.
    Returns a vector D where D[k] = expected duration from state k.
    """
    q = 1 - p
    A = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)

    # Boundary conditions
    A[0][0] = 1;  b[0] = 0
    A[N][N] = 1;  b[N] = 0

    for k in range(1, N):
        A[k][k] = 1
        if k < N / 2:
            # Bet all k: go to 2k or 0
            A[k][2 * k] = -p
            b[k] = 1
        else:
            # Bet N-k: go to N or 2k-N
            A[k][2 * k - N] = -q
            b[k] = 1

    D = np.linalg.solve(A, b)
    return D


# =============================================================================
# QUESTION 3 — Timid Strategy with Quit Rules
# Quit if wealth hits floor t, OR win streak reaches W rounds
# =============================================================================

def q3_expected_duration(k_start, N, t, W, p):
    """
    Expected number of rounds until game ends, starting from k_start
    with streak 0. Game ends when:
      - Wealth hits N (goal)
      - Wealth hits t (floor)
      - Win streak reaches W consecutive wins

    State: (k, s) = (current wealth, current win streak)
    """
    q = 1 - p

    def idx(k, s):
        return (k - t) * (W + 1) + s

    size = (N - t + 1) * (W + 1)
    A = np.zeros((size, size))
    b_vec = np.zeros(size)

    for k in range(t, N + 1):
        for s in range(W + 1):
            i = idx(k, s)

            # Terminal states
            if k == t or k == N or s == W:
                A[i][i] = 1
                b_vec[i] = 0
                continue

            # D(k,s) = 1 + p*D(k+1, s+1) + q*D(k-1, 0)
            A[i][i] = 1
            A[i][idx(k + 1, s + 1)] = -p
            A[i][idx(k - 1, 0)] = -q
            b_vec[i] = 1

    D = np.linalg.solve(A, b_vec)
    return D[idx(k_start, 0)]


# =============================================================================
# QUESTION 4 — General Markov Chain (State-dependent transitions + stay prob)
# =============================================================================

def q4_build_transition_matrix(N, p, q, r):
    """
    Build the (N+1) x (N+1) transition matrix P.
    p, q, r are lists of length N+1.
    Boundary states use only their defined transitions.
    """
    P = np.zeros((N + 1, N + 1))
    for k in range(N + 1):
        P[k][k] = r[k]
        if k + 1 <= N:
            P[k][k + 1] = p[k]
        if k - 1 >= 0:
            P[k][k - 1] = q[k]
    return P


def q4_stationary_distribution(N, p, q, r):
    """
    Compute the stationary distribution pi where pi = pi * P.
    Solved by: (P - I)^T * pi = 0, with sum(pi) = 1.
    Uses lstsq for numerical robustness.
    """
    P = q4_build_transition_matrix(N, p, q, r)

    A = (P - np.eye(N + 1)).T
    A[-1] = 1
    b = np.zeros(N + 1)
    b[-1] = 1

    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return pi


def q4_expected_time(N, p, q, r, a, b):
    """
    Expected number of rounds to reach state b starting from state a.
    Equation per state k (k != b):
      -p[k]*E[k+1] + (1 - r[k])*E[k] - q[k]*E[k-1] = 1
    """
    size = N + 1
    A = np.zeros((size, size))
    b_vec = np.zeros(size)

    A[b][b] = 1
    b_vec[b] = 0

    for k in range(0, N + 1):
        if k == b:
            continue

        A[k][k] = 1 - r[k]
        b_vec[k] = 1

        if k + 1 <= N:
            A[k][k + 1] = -p[k]
        if k - 1 >= 0:
            A[k][k - 1] = -q[k]

    E, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    return E[a]


def q4_expected_wealth(N, pi):
    """
    Expected wealth under the stationary distribution.
    E[wealth] = sum(k * pi[k])
    """
    return sum(k * pi[k] for k in range(N + 1))


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("QUESTION 1 — Timid Strategy")
    print("=" * 60)
    k, N, p = 3, 10, 0.4
    print(f"  Start: k={k}, Goal: N={N}, p={p}")
    print(f"  Win probability:   {q1_win_probability(k, N, p):.4f}")
    print(f"  Expected duration: {q1_expected_duration(k, N, p):.4f} rounds")

    k, N, p = 3, 10, 0.5
    print(f"\n  Start: k={k}, Goal: N={N}, p={p} (fair game)")
    print(f"  Win probability:   {q1_win_probability(k, N, p):.4f}")
    print(f"  Expected duration: {q1_expected_duration(k, N, p):.4f} rounds")

    print()
    print("=" * 60)
    print("QUESTION 2 — Bold Strategy")
    print("=" * 60)
    N, p = 10, 0.4
    W = q2_win_probability(N, p)
    D = q2_expected_duration(N, p)
    print(f"  Goal: N={N}, p={p}")
    for k in range(1, N):
        print(f"  k={k:2d} -> Win prob: {W[k]:.4f}, Expected duration: {D[k]:.4f}")

    print()
    print("=" * 60)
    print("QUESTION 3 — Timid Strategy with Quit Rules")
    print("=" * 60)
    k_start, N, t, W_streak, p = 5, 10, 2, 3, 0.5
    dur = q3_expected_duration(k_start, N, t, W_streak, p)
    print(f"  Start: k={k_start}, Goal: N={N}, Floor: t={t}, Streak: W={W_streak}, p={p}")
    print(f"  Expected duration: {dur:.4f} rounds")

    print()
    print("=" * 60)
    print("QUESTION 4 — General Markov Chain")
    print("=" * 60)
    N = 5
    # State-dependent probabilities summing to 1 at each state
    p4 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.0]
    q4 = [0.0, 0.3, 0.3, 0.3, 0.3, 0.5]
    r4 = [0.5, 0.2, 0.2, 0.2, 0.2, 0.5]

    pi = q4_stationary_distribution(N, p4, q4, r4)
    print(f"  N={N}, state-dependent transitions")
    print(f"  Stationary distribution: {np.round(pi, 4)}")
    print(f"  Expected wealth:         {q4_expected_wealth(N, pi):.4f}")

    a, b_state = 1, 4
    et = q4_expected_time(N, p4, q4, r4, a, b_state)
    print(f"  Expected time from state {a} to state {b_state}: {et:.4f} rounds")
