import numpy as np
from typing import Optional

DEFAULT_WEIGHTS = {
    "sinistri": 0.28,
    "guida": 0.22,
    "geo": 0.14,
    "patrimonio": 0.12,
    "digitale": 0.13,
    "network": 0.11,
}

LAMBDA = 0.15
GAMMA = 0.12


def sigmoid_normalize(x: float) -> float:
    return float(1 / (1 + np.exp(-x / 20)) * 100)


def compute_base_score(factors: dict, weights: dict) -> float:
    score = 0.0
    for key, weight in weights.items():
        score += weight * factors.get(key, 0.0)
    return score


def compute_inertia(current_raw: float, previous_prs: Optional[float]) -> float:
    if previous_prs is None:
        return 0.0
    delta = current_raw - previous_prs
    return LAMBDA * delta


def compute_network_contagion(connections: list) -> float:
    if not connections:
        return 0.0
    total_weight = 0.0
    weighted_sum = 0.0
    for conn in connections:
        if conn.get("consent_given") and conn.get("prs_score") is not None:
            w = conn.get("weight", 1.0)
            weighted_sum += w * conn["prs_score"]
            total_weight += w
    if total_weight == 0:
        return 0.0
    network_avg = weighted_sum / total_weight
    return GAMMA * network_avg


def compute_prs(factors: dict, weights: dict = None, previous_prs: Optional[float] = None, connections: list = None) -> dict:
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if connections is None:
        connections = []
    base = compute_base_score(factors, weights)
    inertia = compute_inertia(base, previous_prs)
    network = compute_network_contagion(connections)
    raw_total = base + inertia + network
    final_score = sigmoid_normalize(raw_total)
    return {
        "score": round(final_score, 2),
        "base_score": round(base, 2),
        "inertia_contribution": round(inertia, 2),
        "network_contribution": round(network, 2),
        "raw_total": round(raw_total, 2),
    }
