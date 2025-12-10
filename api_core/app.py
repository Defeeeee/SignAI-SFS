from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .model import lifespan
import pkgutil
import importlib
import logging
from . import routers

logger = logging.getLogger(__name__)

def create_app(title="Sign Language Prediction API", description="API for Sign Language Prediction"):
    app = FastAPI(title=title, description=description, lifespan=lifespan)
    
    app.add_middleware(
        CORSMiddleware, allow_origins=['*'],
        allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    
    # Dynamically load routers from the routers package
    package_path = routers.__path__
    prefix = routers.__name__ + "."

    for _, name, _ in pkgutil.iter_modules(package_path, prefix):
        try:
            module = importlib.import_module(name)
            if hasattr(module, "router"):
                app.include_router(module.router)
                logger.info(f"Included router from module: {name}")
        except Exception as e:
            logger.error(f"Failed to load router from module {name}: {e}")
    
    return app
