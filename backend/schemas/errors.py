from pydantic import BaseModel
from typing import List, Any, Optional, Dict

class APIErrorDetail(BaseModel):
    loc: Optional[List[str]] = None
    msg: str
    type: str

class ErrorResponse(BaseModel):
    error:str
    message:str
    details: Optional[List[APIErrorDetail]] = None
    failed_models = Optional[List[str]] = None