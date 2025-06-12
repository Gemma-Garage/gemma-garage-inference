from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import inference

app = FastAPI(title="LLM Garage API - Inference Engine")

origins = [
    "http://localhost:3000",  # your React app's origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#inference endpoint
app.include_router(inference.router, prefix="/inference", tags=["Inference"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
