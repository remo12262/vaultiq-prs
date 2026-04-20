import numpy as np

def bayesian_update(
    current_weights: dict,
    current_variance: dict,
    actual_event: float,
    predicted_score: float,
    factors: dict,
    learning_rate: float = 0.1,
) -> tuple[dict, dict]:
    """
    NOVITÀ BREVETTABILE 4 — Bayesian online learning.
    Aggiorna i pesi dell'utente ogni volta che si verifica
    un evento reale (sinistro, rinnovo, disdetta).

    Se il modello ha sbagliato la previsione, corregge
    i pesi proporzionalmente all'errore e alla varianza.
    """
    new_weights = {}
    new_variance = {}

    error = actual_event - (predicted_score / 100)

    for key in current_weights:
        factor_value = factors.get(key, 0.0) / 100
        variance = current_variance.get(key, 1.0)

        # Kalman gain — quanto fidarsi dell'aggiornamento
        kalman_gain = variance / (variance + 1.0)

        # Aggiornamento peso
        new_w = current_weights[key] + kalman_gain * error * factor_value
        new_w = float(np.clip(new_w, 0.01, 0.99))

        # Aggiornamento varianza
        new_v = (1 - kalman_gain) * variance
        new_v = float(np.clip(new_v, 0.01, 10.0))

        new_weights[key] = round(new_w, 4)
        new_variance[key] = round(new_v, 4)

    # Rinormalizza i pesi a somma 1
    total = sum(new_weights.values())
    new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}

    return new_weights, new_variance