import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.db.session import init_db
from app.api import papers, search, chat, compare

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on application start
    init_db()
    yield

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Advanced RAG System for Scientific Papers",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(papers.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(compare.router)

# Mount Frontend Web Interface
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend_app")

@app.get("/")
def root():
    return RedirectResponse(url="/app")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT
    }
