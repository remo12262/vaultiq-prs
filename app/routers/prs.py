from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.core.prs_engine import compute_prs, DEFAULT_WEIGHTS
from app.core.bayesian import bayesian_update
from app.database import supabase

router = APIRouter()


class PRSRequest(BaseModel):
    user_id: str
    factors: dict
    include_network: bool = True


class EventFeedback(BaseModel):
    user_id: str
    actual_event: float  # 0.0 = nessun sinistro, 1.0 = sinistro avvenuto
    predicted_score: float
    factors: dict


@router.post("/compute")
async def compute_user_prs(request: PRSRequest):
    """
    Calcola il PRS per un utente.
    Recupera i pesi personalizzati, lo storico e la rete sociale.
    """
    try:
        # Recupera utente e pesi Bayesiani personalizzati
        user_resp = supabase.table("users")\
            .select("bayes_weights, bayes_variance")\
            .eq("id", request.user_id)\
            .single()\
            .execute()

        if not user_resp.data:
            raise HTTPException(status_code=404, detail="Utente non trovato")

        weights = user_resp.data.get("bayes_weights", DEFAULT_WEIGHTS)

        # Recupera ultimo PRS per inerzia temporale
        history_resp = supabase.table("prs_history")\
            .select("score")\
            .eq("user_id", request.user_id)\
            .order("computed_at", desc=True)\
            .limit(1)\
            .execute()

        previous_prs = None
        if history_resp.data:
            previous_prs = history_resp.data[0]["score"]

        # Recupera connessioni sociali con consenso
        connections = []
        if request.include_network:
            net_resp = supabase.table("social_connections")\
                .select("connected_user_id, weight, consent_given")\
                .eq("user_id", request.user_id)\
                .eq("consent_given", True)\
                .execute()

            for conn in (net_resp.data or []):
                # Recupera ultimo PRS del contatto
                conn_hist = supabase.table("prs_history")\
                    .select("score")\
                    .eq("user_id", conn["connected_user_id"])\
                    .order("computed_at", desc=True)\
                    .limit(1)\
                    .execute()

                if conn_hist.data:
                    connections.append({
                        "consent_given": True,
                        "weight": conn["weight"],
                        "prs_score": conn_hist.data[0]["score"],
                    })

        # Calcola PRS
        result = compute_prs(
            factors=request.factors,
            weights=weights,
            previous_prs=previous_prs,
            connections=connections,
        )

        # Salva in storico
        supabase.table("prs_history").insert({
            "user_id": request.user_id,
            "score": result["score"],
            "raw_factors": request.factors,
            "network_contribution": result["network_contribution"],
            "inertia_contribution": result["inertia_contribution"],
        }).execute()

        return {
            "user_id": request.user_id,
            "prs": result,
            "weights_used": weights,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def update_weights(feedback: EventFeedback):
    """
    Aggiorna i pesi Bayesiani dell'utente dopo un evento reale.
    Chiamato quando avviene un sinistro, rinnovo o disdetta.
    """
    try:
        user_resp = supabase.table("users")\
            .select("bayes_weights, bayes_variance")\