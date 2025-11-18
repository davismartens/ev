from pydantic import BaseModel
from typing import Literal, List

class CreditRiskDecision(BaseModel):
    business_name: str
    risk_level: Literal["low", "medium", "high"]
    recommendation: Literal["approve", "review", "decline"]
    key_factors: List[str]
    explanation: str
