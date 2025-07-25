from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import inference

app = FastAPI(title="LLM Garage API - Inference Engine")

origins = [
    "http://localhost:3000",
    "http://localhost",
    "http://127.0.0.1:3000",
    "https://gemma-garage.web.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Only allow specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#inference endpoint
app.include_router(inference.router, tags=["Inference"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
