from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional

class FeatureInput(BaseModel):
    # Numerical Features (10)
    Administrative: int = Field(default=0, ge=0, le=30, description="Number of administrative pages visited")
    Administrative_Duration: float = Field(default=0.0, ge=0.0, le=5000.0, description="Time spent on administrative pages (sec)")
    Informational: int = Field(default=0, ge=0, le=30, description="Number of informational pages visited")
    Informational_Duration: float = Field(default=0.0, ge=0.0, le=5000.0, description="Time spent on informational pages (sec)")
    ProductRelated: int = Field(default=1, ge=0, le=1000, description="Number of product-related pages visited")
    ProductRelated_Duration: float = Field(default=10.0, ge=0.0, le=70000.0, description="Time spent on product pages (sec)")
    BounceRates: float = Field(default=0.0, ge=0.0, le=1.0, description="Percentage of visitors leaving immediately")
    ExitRates: float = Field(default=0.02, ge=0.0, le=1.0, description="Percentage of pageviews that were last in session")
    PageValues: float = Field(default=0.0, ge=0.0, le=400.0, description="Google Analytics page value index")
    SpecialDay: float = Field(default=0.0, ge=0.0, le=1.0, description="Closeness of site visit to a special day (0.0 to 1.0)")
    
    # Categorical Features (7)
    Month: str = Field(default="May", description="Month of session (Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec)")
    OperatingSystems: int = Field(default=1, ge=1, le=8, description="Operating System ID")
    Browser: int = Field(default=2, ge=1, le=13, description="Browser ID")
    Region: int = Field(default=1, ge=1, le=9, description="Geographic Region ID")
    TrafficType: int = Field(default=2, ge=1, le=20, description="Traffic Source Type ID")
    VisitorType: str = Field(default="Returning_Visitor", description="Visitor type (Returning_Visitor, New_Visitor, Other)")
    Weekend: int = Field(default=0, ge=0, le=1, description="Session on weekend (0=No, 1=Yes)")

    @field_validator('Month')
    @classmethod
    def validate_month(cls, v:str) -> str:
        valid_months = {"Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
        if v not in valid_months:
            raise  ValueError(f"Invalid Month '{v}'. Must be one of: {sorted(list(valid_months))}")
        return v

    @field_validator("VisitorType")
    @classmethod
    def validate_visitor_type(cls, v:str) -> str:
        valid_visitors = {"Returning_Visitor", "New_Visitor", "Other"}
        if v not in valid_visitors:
            raise ValueError(f"Invalid VisitorType '{v}'. Must be one of: {sorted(list(valid_visitors))}")
        return v

class PredictionRequest(BaseModel):
    features: FeatureInput

class SingleModelPrediction(BaseModel):
    model_name: str
    model_key: str
    prediction: int
    prediction_label: str
    confidence: float
    inference_time_ms: float

class ConsensusSummary(BaseModel):
    majority_prediction: int
    majority_label: str
    agreement_ratio: float
    models_agreeing: int
    models_total: int

class PredictionResponse(BaseModel):
    predictions: List[SingleModelPrediction]
    consensus: ConsensusSummary
    input_echo: Dict[str, Any]