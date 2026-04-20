import numpy as np
from scipy.special import expit
from typing import Optional

# Pesi di default (aggiornati per utente con Bayesian learning)
DEFAULT_WEIGHTS = {
    "sinistri": 0.28,
    "guida": 0.22,
    "geo": 0.14,
    "patrimonio": 0.12,
    "digitale": 0.13,
    "network": 0.11,
}

# Costanti brevettabili
LAMBDA = 0.15   # peso inerzia temporale
GAMMA = 0.12    # peso contagio rete sociale


def sigmoid_normalize(x: float) -> float:
    """Normalizza il valore grezzo in [0, 100] via sigmoid."""
    return float(expit(x / 20) * 100)


def compute_base_score(factors: dict, weights: dict) -> float:
    """
    NOVITÀ 1 — Score pesato dei fattori di rischio.
    Σ wᵢ · fᵢ
    """
    score = 0.0
    for key, weight in weights.items():
        score += weight * factors.get(key, 0.0)
    return score


def compute_inertia(current_raw: float, previous_prs: Optional[float]) -> float:
    """
    NOVITÀ BREVETTABILE 1 — Inerzia temporale.
    Se il rischio sta crescendo, accelera. Se scende, frena.
    λ · ΔPRS(t-1)
    """
    if previous_prs is None:
        return 0.0
    delta = current_raw - previous_prs
    return LAMBDA * delta


def compute_network_contagion(connections: list[dict]) -> float:
    """
    NOVITÀ BREVETTABILE 2 — Contagio da rete sociale.
    Media pesata dei PRS dei contatti con consenso.
    γ · C(network)
    """
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


def compute_prs(
    factors: dict,
    weights: dict = None,
    previous_prs: Optional[float] = None,
    connections: list[dict] = None,
) -> dict:
    """
    Formula principale brevettabile:
    PRS(t) = σ( Σ wᵢ·fᵢ(t) + λ·ΔPRS(t-1) + γ·C(network) )
    """
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