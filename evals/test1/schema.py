from pydantic import BaseModel
from typing import Literal

class TransactionDecision(BaseModel):
    company_name: str
    year: int
    requested_amount: float
    industry: str
    revenue: float
    profit_margin: float
    decision: Literal["accept", "review", "flag"]
    explanation: str
