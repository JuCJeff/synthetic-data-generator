from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from api.routes.pipeline import router

app = FastAPI(title="DIY Pipeline API", version="0.1.0")

app.include_router(router, prefix="/api")
