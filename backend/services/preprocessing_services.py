import pandas as pd
import numpy as np
from typing import Tuple
from backend.services.model_service import model_container
from backend.schemas.predict import FeatureInput
from backend.schemas.predict import FeatureInput
from backend.routers.error_handlers import MLValidationException

NUM_COLS = [
    'Administrative', 'Administrative_Duration', 'Informational', 
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]
INT_CAT_COLS = ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend']
OHE_CAT_COLS = ['Month', 'VisitorType']

def validate_domain_boundaries(features: FeatureInput) -> None:
    """Enforces PRD.md §12 domain limits on input feature values."""
    if features.BounceRates < 0.0 or features.BounceRates > 1.0:
        raise MLValidationException(
            message=f"BounceRates must be between 0.0 and 1.0, got {features.BounceRates}",
            field="BounceRates"
        )

    if features.ExitRates < 0.0 or features.ExitRates > 1.0:
        raise MLValidationException(
            message=f"ExitRates must be between 0.0 and 1.0, got {features.ExitRates}",
            field="ExitRates"
        )

    if features.SpecialDay < 0.0 or features.SpecialDay > 1.0:
        raise MLValidationException(
            message=f"SpecialDay must be between 0.0 and 1.0, got {features.SpecialDay}",
            field="SpecialDay"
        )

def transform_features_for_inference(feature_input: FeatureInput) -> pd.DataFrame:
    """
    Transforms raw Pydantic FeatureInput into the processed DataFrame feature matrix
    expected by serialized scikit-learn models. Reuses saved scaler and encoder.
    """
    if not model_container.is_loaded:
        raise RuntimeError("Model Container is not initialized")

    validate_domain_boundaries(feature_input)

    scaler = model_container.scaler
    encoder = model_container.encoder

    raw_dict = feature_input.model_dump()
    raw_df = pd.DataFrame([raw_dict])

    encoded_cats = encoder.transform(raw_df[OHE_CAT_COLS])
    encoded_feature_names = encoder.get_feature_names_out(OHE_CAT_COLS)
    df_encoded = pd.DataFrame(encoded_cats, columns=encoded_feature_names, index=raw_df.index)

    feature_cols = NUM_COLS + INT_CAT_COLS
    x_raw = pd.concat([raw_df[feature_cols], df_encoded], axis=1)

    x_processed = x_raw.copy()
    x_processed[NUM_COLS] = scaler.transform(x_raw[NUM_COLS])

    return x_processed

