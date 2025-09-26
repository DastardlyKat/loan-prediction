from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import logging

# ---- CONFIG ----
MODEL_PATH = "loan_model.pkl"   # <-- change to "models/loan_model.pkl" if your model is inside models/
ALLOW_ORIGINS = ["*"]           # dev: allow all. In production, set to your frontend origin(s)

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- FastAPI app ----
app = FastAPI(title="Loan Default Prediction API")

# Enable CORS (prevents "Failed to fetch" in browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model once (correct) ----
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load model from %s: %s", MODEL_PATH, e)
    raise RuntimeError(f"Could not load model: {e}")


# ---- Pydantic input model ----
class LoanApplication(BaseModel):
    Gender: str = Field(..., description="Male or Female")
    Married: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="0, 1, 2, 3+")
    Education: str = Field(..., description="Graduate or Not Graduate")
    Self_Employed: str = Field(..., description="Yes or No")
    ApplicantIncome: float = Field(..., description="Applicant's income (in thousands)")
    CoapplicantIncome: float = Field(..., description="Co-applicant's income (in thousands)")
    LoanAmount: float = Field(..., description="Loan Amount (in thousands)")
    Loan_Amount_Term: float = Field(..., description="Term of loan (e.g., 360)")
    Credit_History: int = Field(..., description="0 or 1")
    Property_Area: str = Field(..., description="Urban, Semiurban, Rural")

    # validators - allow case-insensitive input
    @validator("Gender")
    def validate_gender(cls, v):
        if str(v).strip().lower() not in {"male", "female"}:
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v

    @validator("Married")
    def validate_married(cls, v):
        if str(v).strip().lower() not in {"yes", "no"}:
            raise ValueError("Married must be 'Yes' or 'No'")
        return v

    @validator("Dependents")
    def validate_dependents(cls, v):
        if str(v).strip() not in {"0", "1", "2", "3+"}:
            raise ValueError("Dependents must be 0, 1, 2, or 3+")
        return v

    @validator("Education")
    def validate_education(cls, v):
        if str(v).strip().lower().replace('-', ' ') not in {"graduate", "not graduate", "not_graduate"}:
            raise ValueError("Education must be 'Graduate' or 'Not Graduate'")
        return v

    @validator("Self_Employed")
    def validate_self_employed(cls, v):
        if str(v).strip().lower() not in {"yes", "no"}:
            raise ValueError("Self_Employed must be 'Yes' or 'No'")
        return v

    @validator("Property_Area")
    def validate_property_area(cls, v):
        if str(v).strip().lower().replace('-', '').replace('_', '') not in {"urban", "semiurban", "rural"}:
            raise ValueError("Property_Area must be Urban, Semiurban, or Rural")
        return v

    @validator("Credit_History")
    def validate_credit_history(cls, v):
        if int(v) not in {0, 1}:
            raise ValueError("Credit_History must be 0 or 1")
        return v


# ---- Preprocessing (robust) ----
def preprocess_input(app_model: LoanApplication) -> pd.DataFrame:
    data = app_model.dict()

    try:
        # normalize and map categorical fields (case-insensitive)
        gender_map = {"male": 1, "female": 0}
        married_map = {"yes": 1, "no": 0}
        depend_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
        education_map = {"graduate": 1, "not graduate": 0, "not_graduate": 0}
        self_map = {"yes": 1, "no": 0}
        property_map = {"urban": 1, "semiurban": 2, "rural": 3, "semi-urban": 2, "semi_urban": 2}

        # lowercase inputs
        g = str(data.get("Gender", "")).strip().lower()
        m = str(data.get("Married", "")).strip().lower()
        dep = str(data.get("Dependents", "")).strip()
        edu = str(data.get("Education", "")).strip().lower().replace('-', ' ')
        se = str(data.get("Self_Employed", "")).strip().lower()
        pa_raw = str(data.get("Property_Area", "")).strip().lower()

        if g not in gender_map:
            raise ValueError(f"Invalid Gender: {data.get('Gender')}")
        data["Gender"] = gender_map[g]

        if m not in married_map:
            raise ValueError(f"Invalid Married: {data.get('Married')}")
        data["Married"] = married_map[m]

        if dep in depend_map:
            data["Dependents"] = depend_map[dep]
        else:
            try:
                data["Dependents"] = int(dep)
            except Exception:
                raise ValueError(f"Invalid Dependents: {data.get('Dependents')}")

        if edu not in education_map:
            raise ValueError(f"Invalid Education: {data.get('Education')}")
        data["Education"] = education_map[edu]

        if se not in self_map:
            raise ValueError(f"Invalid Self_Employed: {data.get('Self_Employed')}")
        data["Self_Employed"] = self_map[se]

        # Normalize property area variants
        pa_key = pa_raw.replace('_', '').replace('-', '')
        if pa_key not in property_map:
            raise ValueError(f"Invalid Property_Area: {data.get('Property_Area')}")
        data["Property_Area"] = property_map[pa_key]

        # numeric fields: cast and set defaults if needed
        data["ApplicantIncome"] = float(data.get("ApplicantIncome", 0))
        data["CoapplicantIncome"] = float(data.get("CoapplicantIncome", 0))
        data["LoanAmount"] = float(data.get("LoanAmount", 0))
        data["Loan_Amount_Term"] = float(data.get("Loan_Amount_Term", 360))
        data["Credit_History"] = int(data.get("Credit_History", 1))

    except ValueError as ve:
        # return a 400 Bad Request with helpful message
        raise HTTPException(status_code=400, detail=str(ve))

    return pd.DataFrame([data])


# ---- Prediction endpoint ----
@app.post("/predict")
async def predict_loan_status(application: LoanApplication):
    input_df = preprocess_input(application)

    try:
        pred_arr = model.predict(input_df)
        pred_val = pred_arr[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df).tolist()
    except Exception as e:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    try:
        pred_int = int(pred_val)
    except Exception:
        pred_int = int(float(pred_val))

    label = "Approved" if pred_int == 1 else "Rejected"

    return {
        "input": application.dict(),
        "prediction": str(pred_int),
        "label": label,
        "probability": proba
    }
