from fastapi import FastAPI

from lifespan import lifespan
from routers.health import router as health_router
from routers.synthesize import router as synthesize_router


app = FastAPI(title="Zundamon Speech API", version="1.0.0", lifespan=lifespan)

app.include_router(health_router)
app.include_router(synthesize_router)