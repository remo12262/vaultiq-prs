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
    actual_event: float
    predicted_score: float
    factors: dict


@router.post("/compute")
async def compute_user_prs(request: PRSRequest):
    try:
        user_resp = supabase.table("users").select("bayes_weights, bayes_variance").eq("id", request.user_id).single().execute()
        if not user_resp.data:
            raise HTTPException(status_code=404, detail="Utente non trovato")
        weights = user_resp.data.get("bayes_weights", DEFAULT_WEIGHTS)
        history_resp = supabase.table("prs_history").select("score").eq("user_id", request.user_id).order("computed_at", desc=True).limit(1).execute()
        previous_prs = None
        if history_resp.data:
            previous_prs = history_resp.data[0]["score"]
        connections = []
        if request.include_network:
            net_resp = supabase.table("social_connections").select("connected_user_id, weight, consent_given").eq("user_id", request.user_id).eq("consent_given", True).execute()
            for conn in (net_resp.data or []):
                conn_hist = supabase.table("prs_history").select("score").eq("user_id", conn["connected_user_id"]).order("computed_at", desc=True).limit(1).execute()
                if conn_hist.data:
                    connections.append({"consent_given": True, "weight": conn["weight"], "prs_score": conn_hist.data[0]["score"]})
        result = compute_prs(factors=request.factors, weights=weights, previous_prs=previous_prs, connections=connections)
        supabase.table("prs_history").insert({"user_id": request.user_id, "score": result["score"], "raw_factors": request.factors, "network_contribution": result["network_contribution"], "inertia_contribution": result["inertia_contribution"]}).execute()
        return {"user_id": request.user_id, "prs": result, "weights_used": weights}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def update_weights(feedback: EventFeedback):
    try:
        user_resp = supabase.table("users").select("bayes_weights, bayes_variance").eq("id", feedback.user_id).single().execute()
        if not user_resp.data:
            raise HTTPException(status_code=404, detail="Utente non trovato")
        current_weights = user_resp.data["bayes_weights"]
        current_variance = user_resp.data["bayes_variance"]
        new_weights, new_variance = bayesian_update(current_weights=current_weights, current_variance=current_variance, actual_event=feedback.actual_event, predicted_score=feedback.predicted_score, factors=feedback.factors)
        supabase.table("users").update({"bayes_weights": new_weights, "bayes_variance": new_variance}).eq("id", feedback.user_id).execute()
        return {"user_id": feedback.user_id, "updated_weights": new_weights, "message": "Pesi aggiornati con successo"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{user_id}")
async def get_prs_history(user_id: str, limit: int = 10):
    try:
        resp = supabase.table("prs_history").select("*").eq("user_id", user_id).order("computed_at", desc=True).limit(limit).execute()
        return {"user_id": user_id, "history": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
