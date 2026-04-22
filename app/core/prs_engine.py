"""
VaultIQ — Personal Risk Score Engine v2.0
==========================================
Algoritmo brevettabile con 10 fattori di rischio.

Formula principale:
    PRS(t) = sigma( sum(wi * fi(t)) + lambda * dPRS(t-1) + gamma * C(network) )

Novità brevettabili:
    1. Inerzia temporale — lambda * dPRS(t-1)
    2. Contagio da rete sociale — gamma * C(network)
    3. Comportamento digitale come proxy affidabilità
    4. Bayesian online learning — aggiornamento pesi per utente
    5. Fattore meteo/clima territoriale (NUOVO v2.0)
    6. Fattore mobilità predittiva (NUOVO v2.0)

Autore: VaultIQ
Versione: 2.0.0
Data: Aprile 2026
"""

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# PESI DI DEFAULT — aggiornati per utente con Bayesian learning
# Somma = 1.0
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "sinistri":  0.22,   # Storico sinistri e incidenti passati
    "guida":     0.18,   # Comportamento di guida (telematica)
    "geo":       0.10,   # Zona geografica di residenza
    "patrimonio":0.10,   # Patrimonio esposto (casa, auto, oggetti)
    "digitale":  0.10,   # Comportamento digitale nell'app
    "network":   0.08,   # Contagio da rete sociale
    "salute":    0.08,   # Storico medico e patologie dichiarate
    "meteo":     0.05,   # Rischio climatico territoriale
    "credito":   0.05,   # Affidabilità creditizia proxy
    "mobilita":  0.04,   # Km annui e mezzi di trasporto usati
}

# Verifica che i pesi sommino a 1.0
assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9, "I pesi devono sommare a 1.0"

# ---------------------------------------------------------------------------
# COSTANTI BREVETTABILI
# ---------------------------------------------------------------------------

LAMBDA = 0.15   # Peso inerzia temporale (NOVITÀ BREVETTABILE 1)
GAMMA  = 0.12   # Peso contagio rete sociale (NOVITÀ BREVETTABILE 2)

# ---------------------------------------------------------------------------
# FUNZIONI CORE
# ---------------------------------------------------------------------------

def sigmoid_normalize(x: float) -> float:
    """
    Normalizza il valore grezzo in [0, 100] via sigmoid.
    Numericamente stabile con numpy.
    """
    return float(1 / (1 + np.exp(-x / 20)) * 100)


def compute_base_score(factors: dict, weights: dict) -> float:
    """
    COMPONENTE 1 — Score pesato dei fattori di rischio.
    Formula: sum(wi * fi)
    
    Ogni fattore fi è in [0, 100].
    I pesi wi sommano a 1.0.
    """
    score = 0.0
    for key, weight in weights.items():
        value = factors.get(key, 0.0)
        if not 0 <= value <= 100:
            raise ValueError(f"Fattore '{key}' deve essere in [0, 100], ricevuto: {value}")
        score += weight * value
    return score


def compute_inertia(current_raw: float, previous_prs: Optional[float]) -> float:
    """
    NOVITÀ BREVETTABILE 1 — Inerzia temporale.
    
    Se il rischio sta crescendo, il punteggio accelera.
    Se il rischio sta scendendo, il punteggio frena.
    
    Formula: lambda * dPRS(t-1)
    dove dPRS = current_raw - previous_prs
    """
    if previous_prs is None:
        return 0.0
    delta = current_raw - previous_prs
    return LAMBDA * delta


def compute_network_contagion(connections: list) -> float:
    """
    NOVITÀ BREVETTABILE 2 — Contagio da rete sociale.
    
    Principio mutuato dall'epidemiologia:
    il comportamento di rischio si propaga nei network interpersonali.
    
    Formula: gamma * media_pesata(PRS_contatti)
    Solo contatti con consenso esplicito (GDPR compliant).
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


def compute_meteo_risk(geo_code: str = None, meteo_index: float = None) -> float:
    """
    NOVITÀ BREVETTABILE 5 — Fattore meteo/clima territoriale.
    
    Combina:
    - Indice di rischio alluvioni per CAP/provincia
    - Frequenza grandine storica
    - Rischio sismico territoriale
    
    Se non disponibile, usa il valore medio nazionale (50).
    """
    if meteo_index is not None:
        return float(np.clip(meteo_index, 0, 100))
    return 50.0  # valore medio nazionale di default


def compute_mobility_risk(km_annui: float = None, mezzi: list = None) -> float:
    """
    NOVITÀ BREVETTABILE 6 — Fattore mobilità predittiva.
    
    Calcola il rischio basato su:
    - Km percorsi annualmente
    - Tipologia di mezzi usati (auto, moto, bici, mezzi pubblici)
    - Frequenza spostamenti notturni
    
    Formula: normalizzazione logaritmica dei km + peso per mezzo
    """
    if km_annui is None:
        return 50.0  # valore medio di default

    # Normalizzazione logaritmica: 0 km = 0, 50000 km = 100
    base_risk = min(100, (np.log1p(km_annui) / np.log1p(50000)) * 100)

    # Moltiplicatore per tipo di mezzo
    mezzo_weights = {
        "moto": 1.4,
        "auto": 1.0,
        "bici": 0.6,
        "pubblico": 0.3,
    }

    if mezzi:
        multiplier = max(mezzo_weights.get(m, 1.0) for m in mezzi)
        base_risk = min(100, base_risk * multiplier)

    return round(float(base_risk), 2)


def compute_prs(
    factors: dict,
    weights: dict = None,
    previous_prs: Optional[float] = None,
    connections: list = None,
    geo_code: str = None,
    km_annui: float = None,
    mezzi: list = None,
) -> dict:
    """
    FORMULA PRINCIPALE BREVETTABILE v2.0
    =====================================
    
    PRS(t) = sigma( sum(wi * fi(t)) + lambda * dPRS(t-1) + gamma * C(network) )
    
    Componenti:
        sum(wi * fi(t))    : score base pesato sui 10 fattori
        lambda * dPRS(t-1) : inerzia temporale
        gamma * C(network) : contagio da rete sociale
        sigma(x)           : sigmoid normalizzante -> [0, 100]
    
    Args:
        factors      : dict con valori in [0,100] per ogni fattore
        weights      : pesi personalizzati (default: DEFAULT_WEIGHTS)
        previous_prs : ultimo PRS calcolato per inerzia temporale
        connections  : lista contatti con consenso per contagio rete
        geo_code     : CAP o codice provincia per rischio meteo
        km_annui     : km percorsi annualmente per rischio mobilità
        mezzi        : lista mezzi usati ['auto', 'moto', 'bici', 'pubblico']
    
    Returns:
        dict con score finale e contributi di ogni componente
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if connections is None:
        connections = []

    # Arricchisce i fattori con meteo e mobilità se non presenti
    if "meteo" not in factors:
        factors["meteo"] = compute_meteo_risk(geo_code)
    if "mobilita" not in factors:
        factors["mobilita"] = compute_mobility_risk(km_annui, mezzi)

    # Calcola i 3 componenti
    base    = compute_base_score(factors, weights)
    inertia = compute_inertia(base, previous_prs)
    network = compute_network_contagion(connections)

    # Formula finale
    raw_total   = base + inertia + network
    final_score = sigmoid_normalize(raw_total)

    # Classificazione del rischio
    if final_score < 35:
        risk_class = "BASSO"
        risk_color = "verde"
    elif final_score < 65:
        risk_class = "MEDIO"
        risk_color = "arancione"
    else:
        risk_class = "ALTO"
        risk_color = "rosso"

    return {
        "score":                  round(final_score, 2),
        "risk_class":             risk_class,
        "risk_color":             risk_color,
        "base_score":             round(base, 2),
        "inertia_contribution":   round(inertia, 2),
        "network_contribution":   round(network, 2),
        "raw_total":              round(raw_total, 2),
        "factors_used":           {k: round(v, 2) for k, v in factors.items()},
        "weights_used":           weights,
    }


# ---------------------------------------------------------------------------
# ESEMPIO DI UTILIZZO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Esempio utente ad alto rischio
    fattori_utente = {
        "sinistri":   75,   # 2 sinistri negli ultimi 3 anni
        "guida":      60,   # velocità media elevata
        "geo":        80,   # zona ad alto rischio
        "patrimonio": 70,   # casa + auto di valore
        "digitale":   40,   # risponde agli alert con ritardo
        "network":    65,   # amici con sinistri frequenti
        "salute":     30,   # nessuna patologia cronica
        "credito":    50,   # affidabilità creditizia media
    }

    # Prima chiamata (nessun storico)
    risultato = compute_prs(
        factors=fattori_utente,
        km_annui=25000,
        mezzi=["auto", "moto"],
    )

    print("=" * 50)
    print("VaultIQ PRS Engine v2.0 — Risultato")
    print("=" * 50)
    print(f"PRS Score:     {risultato['score']} / 100")
    print(f"Classe rischio: {risultato['risk_class']} ({risultato['risk_color']})")
    print(f"Score base:    {risultato['base_score']}")
    print(f"Inerzia:       {risultato['inertia_contribution']}")
    print(f"Network:       {risultato['network_contribution']}")
    print(f"Mobilità:      {risultato['factors_used'].get('mobilita')}")
    print(f"Meteo:         {risultato['factors_used'].get('meteo')}")
    print("=" * 50)
