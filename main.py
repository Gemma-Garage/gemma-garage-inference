from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import inference

app = FastAPI(title="LLM Garage API - Inference Engine")

origins = [
    "http://localhost:3000",
    "http://localhost",
    "http://127.0.0.1:3000",
    "https://gemma-garage.web.app",
    "https://gemma-garage.firebaseapp.com",
    "*"  # Temporarily allow all origins for debugging
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Only allow specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "inference"}

#inference endpoint
app.include_router(inference.router, tags=["Inference"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
