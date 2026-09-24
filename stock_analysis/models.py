from pydantic import BaseModel, Field
from typing import Optional

class PatternResult(BaseModel):
    confirmed: bool
    confidence: float = Field(default = 0, ge=0, le=100)
    confidence_breakdown: Optional[dict[str, float]] = None
    confidence_metadata: Optional[dict[str, str | None]] = None
    structure_detected: bool | None = None
    pivots: Optional[list[dict]] = None
    direction: str = "neutral"
