from pydantic import BaseModel, Field
from typing import Optional, List

class Iteration(BaseModel):
  goal: bool
  score: int


class EvalOut(BaseModel):
  name: str
  iteration: List[Iteration]
  max_iterations: Optional[int]
