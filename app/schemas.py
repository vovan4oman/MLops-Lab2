from pydantic import BaseModel, Field


class WineFeatures(BaseModel):
    alcohol: float = Field(..., ge=0, le=20, description="Alcohol content (%)")
    malic_acid: float = Field(..., ge=0, le=10, description="Malic acid (g/L)")
    ash: float = Field(..., ge=0, le=5, description="Ash (g/L)")
    alcalinity_of_ash: float = Field(..., ge=0, le=40, description="Alcalinity of ash")
    magnesium: float = Field(..., ge=0, le=200, description="Magnesium (mg/L)")
    total_phenols: float = Field(..., ge=0, le=10, description="Total phenols")
    flavanoids: float = Field(..., ge=0, le=10, description="Flavanoids")
    nonflavanoid_phenols: float = Field(..., ge=0, le=5, description="Nonflavanoid phenols")
    proanthocyanins: float = Field(..., ge=0, le=10, description="Proanthocyanins")
    color_intensity: float = Field(..., ge=0, le=20, description="Color intensity")
    hue: float = Field(..., ge=0, le=5, description="Hue")
    od280_od315: float = Field(..., ge=0, le=5, description="OD280/OD315 of diluted wines")
    proline: float = Field(..., ge=0, le=2000, description="Proline (mg/L)")


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probability: float
