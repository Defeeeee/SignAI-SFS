from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .model import lifespan
import pkgutil
import importlib
import logging
from . import routers
from typing import List, Optional

logger = logging.getLogger(__name__)

def create_app(title="Sign Language Prediction API", description="API for Sign Language Prediction", include_routers: Optional[List[str]] = None):
    app = FastAPI(title=title, description=description, lifespan=lifespan)
    
    app.add_middleware(
        CORSMiddleware, allow_origins=['*'],
        allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "detail": str(exc)},
        )

    # Dynamically load routers from versioned packages
    from .routers import v1, v2, v3
    
    versions = {
        "v1": v1,
        "v2": v2,
        "v3": v3
    }

    for version_name, version_package in versions.items():
        package_path = version_package.__path__
        prefix = version_package.__name__ + "."

        for _, name, _ in pkgutil.iter_modules(package_path, prefix):
            try:
                module_name = name.split('.')[-1]
                
                # If include_routers is specified, only load those routers
                # Note: This filtering is simple and applies to all versions. 
                # Ideally, we might want version-specific filtering, but this works for now.
                if include_routers is not None and module_name not in include_routers:
                    continue

                module = importlib.import_module(name)
                if hasattr(module, "router"):
                    # Mount at /{version} (e.g., /v1, /v2)
                    app.include_router(module.router, prefix=f"/{version_name}")
                    
                    # Special case for v1: Mount at root / for backward compatibility
                    if version_name == "v1":
                        app.include_router(module.router, include_in_schema=False)

                    logger.info(f"Included router from module: {name} (Version: {version_name})")
            except Exception as e:
                logger.error(f"Failed to load router from module {name}: {e}")
    
    return app
