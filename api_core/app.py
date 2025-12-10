from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .model import lifespan
from .endpoints import router as shared_router

def create_app(title="Sign Language Prediction API", description="API for Sign Language Prediction"):
    app = FastAPI(title=title, description=description, lifespan=lifespan)
    
    app.add_middleware(
        CORSMiddleware, allow_origins=['*'],
        allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    
    app.include_router(shared_router)
    
    return app
