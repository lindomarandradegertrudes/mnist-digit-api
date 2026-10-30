from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.services import carregar_modelo
from app.routers import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modelo = carregar_modelo()
    yield    

app = FastAPI(title="MNIST Digit API", lifespan=lifespan)
app.include_router(router)
