import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = joblib.load("model.joblib")

class Inquiry(BaseModel):
    message: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/inquiry")
async def handle_inquiry(inquiry: Inquiry):
    category = model.predict([inquiry.message])[0]

    # Dummy mock KB check (update later)
    info_found = "error" in inquiry.message.lower()
    if category == "Technical Support":
        if info_found:
            response = f"Thanks for reaching out! Regarding your issue, here's some information: Try restarting the app. Does this resolve your issue?"
        else:
            response = f"Thanks for your query. I couldn't find an immediate answer, but I've routed your request to our support team."
    elif category == "Product Feature Request":
        feature = inquiry.message
        with open("feature_requests.txt", "a") as f:
            f.write(feature + "\n")
        response = f"Thank you for your suggestion! We've logged your feature request for our product team to review."
    elif category == "Sales Lead":
        if "company" in inquiry.message.lower() or "enterprise" in inquiry.message.lower():
            response = "Thanks for your interest! Our sales team will be in touch soon. Could you please provide your company name?"
        else:
            response = "Thanks for your interest! Our sales team will be in touch soon. In the meantime, could you tell us more about your needs?"
    else:
        response = "Thank you for your message. We will get back to you soon."

    return {"category": category, "response": response}
