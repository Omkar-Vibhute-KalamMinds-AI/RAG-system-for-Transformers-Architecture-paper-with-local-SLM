import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import *
from pathlib import Path  

import pyfiles_path  # noqa: F401
from logger import logger
from ModelLoader import load_model
from app import AdvancedRAGPipeline, init_retriever
from config_loader import load_config
#-----------------------------

config = load_config()
MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", config["llm"]["local_path"])
LLM_TEMPERATURE = config["llm"].get("temperature", 0.4)
LLM_MAX_NEW_TOKENS = config["llm"].get("max_new_tokens", 1024)
LLM_DO_SAMPLE = config["llm"].get("do_sample", True)
RETRIEVER_TOP_K = config["retriever"]["top_k"]
RETRIEVER_MIN_SCORE = config["retriever"]["min_score"]
PIPELINE_USE_HISTORY = config["AdvancedRAGPipeline"].get("use_history", True)
PIPELINE_HISTORY_WINDOW = config["AdvancedRAGPipeline"].get("history_window", 1)
PIPELINE_STREAM = config["AdvancedRAGPipeline"].get("stream", True)
PIPELINE_SUMMARIZE = config["AdvancedRAGPipeline"].get("summarize", False)

processor = None
tokenizer = None
model = None
pipeline = None

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=LLM_MAX_NEW_TOKENS, ge=1, le=4096)
    temperature: float = Field(default=LLM_TEMPERATURE, ge=0.0, le=2.0)
    do_sample: bool = LLM_DO_SAMPLE

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_new_tokens: int = Field(default=LLM_MAX_NEW_TOKENS, ge=1, le=4096)
    temperature: float = Field(default=LLM_TEMPERATURE, ge=0.0, le=2.0)
    do_sample: bool = LLM_DO_SAMPLE


class GenerateResponse(BaseModel):
    text: str
    elapsed_seconds: float
    device: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=RETRIEVER_TOP_K, ge=1, le=10)
    min_score: float = Field(default=RETRIEVER_MIN_SCORE, ge=0.0, le=1.0)
    summarize: bool = PIPELINE_SUMMARIZE
    use_history: bool = PIPELINE_USE_HISTORY
    stream: bool = PIPELINE_STREAM
    max_new_tokens: int = Field(default=LLM_MAX_NEW_TOKENS, ge=1, le=4096)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    summary: Optional[str] = None
    history: Optional[list] = None
    elapsed_seconds: float

#-----------------------------
def _ensure_loaded():
    if pipeline is None or processor is None or tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, tokenizer, model, pipeline
    logger.info(f"Loading {config['llm']['model_name']} from {MODEL_PATH}")
    processor, tokenizer, model = load_model(MODEL_PATH)
    from query_variationar import attach_llm
    attach_llm(processor, tokenizer, model)
    pipeline = AdvancedRAGPipeline(
        init_retriever(),
        model,
        processor,
        use_history=PIPELINE_USE_HISTORY,
        history_window=PIPELINE_HISTORY_WINDOW,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Model ready on {device}")
    yield

app = FastAPI(
    title="Gemma 4 E2B API",
    description="Inference API for the local gemma-4-E2B-it model",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SWAGGER_THEME_CSS = """
<style>
  body { background: #f3f3f3 !important; }
  .swagger-ui { background: #f3f3f3; color: #111; }
  .swagger-ui .topbar { background: #111; border-bottom: 1px solid #111; }
  .swagger-ui .info .title, .swagger-ui .opblock-tag { color: #111 !important; }
  .swagger-ui .scheme-container { background: #fff; box-shadow: none; border: 1px solid #d8d8d8; }
  .swagger-ui .opblock { background: #fff; border-color: #111 !important; }
  .swagger-ui .opblock .opblock-summary-method { background: #111 !important; color: #fff !important; }
  .swagger-ui .btn.execute { background: #111 !important; color: #fff !important; border-color: #111 !important; }
  .swagger-ui input, .swagger-ui textarea, .swagger-ui select {
    background: #fff !important; color: #111 !important; border-color: #d8d8d8 !important;
  }
</style>
"""

STATIC_DIR = Path(__file__).resolve().parent / "static"

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/docs", include_in_schema=False)
def custom_docs():
    html = get_swagger_ui_html( 
        openapi_url="/openapi.json",
        title="Gemma 4 E2B API",
    ).body.decode("utf-8")
    return HTMLResponse(html.replace("</head>", SWAGGER_THEME_CSS + "</head>"))

#-----------------------------
@app.get("/api/config")
def public_config():
    return {
        "llm": {
            "model_name": config["llm"]["model_name"],
            "temperature": LLM_TEMPERATURE,
            "max_new_tokens": LLM_MAX_NEW_TOKENS,
            "do_sample": LLM_DO_SAMPLE,
        },
        "retriever": {
            "top_k": RETRIEVER_TOP_K,
            "min_score": RETRIEVER_MIN_SCORE,
        },
        "pipeline": {
            "use_history": PIPELINE_USE_HISTORY,
            "history_window": PIPELINE_HISTORY_WINDOW,
            "stream": PIPELINE_STREAM,
            "summarize": PIPELINE_SUMMARIZE,
        },
    }

#----------------------------
@app.get("/health")
def health():
    device = str(model.device) if model is not None else "unloaded"
    return {
        "status": "ok" if pipeline is not None else "loading",
        "model_path": MODEL_PATH,
        "model_name": config["llm"]["model_name"],
        "embed_model": config["embedding model"]["embedmodel_name"],
        "device": device,
        "pipeline": "ready" if pipeline is not None else "unloaded",
    }
#-----------------------------

@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    _ensure_loaded()
    start = time.time()
    try:
        text = pipeline.generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample,
        )
        return GenerateResponse(
            text=text,
            elapsed_seconds=round(time.time() - start, 3),
            device=str(model.device),
        )
    except Exception as e:
        logger.error(f"/generate failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
#-----------------------------

@app.post("/chat", response_model=GenerateResponse)
def chat(request: ChatRequest): 
    _ensure_loaded()
    start = time.time()
    try:
        messages = [m.model_dump() for m in request.messages]
        text = pipeline.generate(
            messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample,
        )
        return GenerateResponse(
            text=text,
            elapsed_seconds=round(time.time() - start, 3),
            device=str(model.device),
        )
    except Exception as e:
        logger.error(f"/chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
#-----------------------------

@app.post("/generate/stream")
def generate_stream(request: GenerateRequest):
    _ensure_loaded()

    def event_stream():
        try:
            for chunk in pipeline.generate_streaming(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.do_sample,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"/generate/stream failed: {e}", exc_info=True)
            yield f"\n[error] {e}"

    return StreamingResponse(event_stream(), media_type="text/plain")
#-----------------------------

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    _ensure_loaded() 
    start = time.time()
    try:
        pipeline.use_history = request.use_history
        pipeline.history_window = PIPELINE_HISTORY_WINDOW
        result = pipeline.query(
            request.question,
            top_k=request.top_k,
            min_score=request.min_score,
            stream=False,
            summarize=request.summarize,
        )
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],  
            sources=result["sources"], 
            summary=result["summary"],
            history=result["history"],
            elapsed_seconds=round(time.time() - start, 3),
        )
    except Exception as e:
        logger.error(f"/query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
#-----------------------------

@app.post("/query/stream")
def query_stream(request: QueryRequest):
    _ensure_loaded()
    pipeline.use_history = request.use_history
    pipeline.history_window = PIPELINE_HISTORY_WINDOW

    def event_stream():
        try:
            for chunk in pipeline.query_streaming(
                request.question,
                top_k=request.top_k,
                min_score=request.min_score,
                summarize=request.summarize,
                max_new_tokens=request.max_new_tokens,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"/query/stream failed: {e}", exc_info=True)
            yield f"\n[error] {e}"

    return StreamingResponse(event_stream(), media_type="text/plain")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

#-----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("modelapi_app:app", host="0.0.0.0", port=8000, reload=False)

#pip install fastapi "uvicorn[standard]" pydantic
#cd D:\LLMOps\pyfiles
#python modelapi_app.py    
#uvicorn modelapi_app:app --host 0.0.0.0 --port 8000 

# python -m pytest tests/test_llmops_project.py tests/test_integration.py -v --tb=short 
