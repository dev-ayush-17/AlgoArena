import json, logging
from fastapi import APIRouter, HTTPException, status
from backend.config import settings

router = APIRouter(tags=["Schema"])
logger = logging.getLogger("backend.routers.schema")

_schema_cache: dict = None

def get_fallback_feature_config() -> dict:
    """Returns fallback schema if feature_config.json is missing or invalid."""
    return {
        "numerical_features": [
            {"name": "Administrative", "dtype": "int64", "min": 0, "max": 27, "default": 0, "description": "Number of administrative pages visited"},
            {"name": "Administrative_Duration", "dtype": "float64", "min": 0.0, "max": 3398.0, "default": 0.0, "description": "Total time spent on administrative pages (seconds)"},
            {"name": "Informational", "dtype": "int64", "min": 0, "max": 24, "default": 0, "description": "Number of informational pages visited"},
            {"name": "Informational_Duration", "dtype": "float64", "min": 0.0, "max": 2549.0, "default": 0.0, "description": "Total time spent on informational pages (seconds)"},
            {"name": "ProductRelated", "dtype": "int64", "min": 0, "max": 705, "default": 1, "description": "Number of product-related pages visited"},
            {"name": "ProductRelated_Duration", "dtype": "float64", "min": 0.0, "max": 63973.5, "default": 10.0, "description": "Total time spent on product-related pages (seconds)"},
            {"name": "BounceRates", "dtype": "float64", "min": 0.0, "max": 0.2, "default": 0.0, "description": "Percentage of visitors entering site and leaving immediately"},
            {"name": "ExitRates", "dtype": "float64", "min": 0.0, "max": 0.2, "default": 0.02, "description": "Percentage of pageviews that were last in session"},
            {"name": "PageValues", "dtype": "float64", "min": 0.0, "max": 361.0, "default": 0.0, "description": "Average Google Analytics value of visited pages"},
            {"name": "SpecialDay", "dtype": "float64", "min": 0.0, "max": 1.0, "default": 0.0, "description": "Closeness of site visit time to a special day (0.0 to 1.0)"}
        ],
        "categorical_features": [
            {"name": "Month", "dtype": "string", "categories": ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], "default": "May", "description": "Month of the session"},
            {"name": "OperatingSystems", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8], "default": 1, "description": "Operating system ID"},
            {"name": "Browser", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "default": 2, "description": "Browser ID"},
            {"name": "Region", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9], "default": 1, "description": "Geographic region ID"},
            {"name": "TrafficType", "dtype": "int64", "categories": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "default": 2, "description": "Traffic source type ID"},
            {"name": "VisitorType", "dtype": "string", "categories": ["Returning_Visitor", "New_Visitor", "Other"], "default": "Returning_Visitor", "description": "Visitor status"},
            {"name": "Weekend", "dtype": "int64", "categories": [0, 1], "default": 0, "description": "Session occurred on weekend (0=No, 1=Yes)"}
        ],
        "target": {
            "name": "Revenue",
            "classes": [0, 1],
            "class_labels": ["No Purchase", "Purchase Completed"]
        }
    }

@router.get("/form-schema", status_code=status.HTTP_200_OK)
def get_from_schema():
    """
    Returns feature definitions, valid ranges, categories, and defaults
    for dynamic frontend form rendering.
    """
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    config_path = settings.ARTIFACTS_DIR / settings.FEATURE_CONFIG_FILENAME

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                content = json.load(f)
                if isinstance(content, dict) and "numerical_features" in content:
                    _schema_cache = content
                    return _schema_cache

        except Exception as e:
            logger.warning(f"Could not parse {config_path}: {e}")

    _schema_cache = get_fallback_feature_config()
    return _schema_cache