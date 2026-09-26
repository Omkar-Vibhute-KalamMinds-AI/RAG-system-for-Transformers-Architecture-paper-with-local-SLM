import ast
import re
import pyfiles_path  # noqa: F401
from config_loader import load_config
from EmbedModelLoader import EmbeddingManager
from ModelLoader import load_model
from logger import logger
import numpy as np 


config = load_config()
MIN_QUERY_SIM = config["retriever"]["min_score_query_variation"]
_embedding_manager = None

processor = None
tokenizer = None
model = None

def attach_embedding_manager(manager: EmbeddingManager):
    """Reuse the embedder from app.py so it is not loaded twice."""
    global _embedding_manager
    _embedding_manager = manager

def _get_embedding_manager():
    global _embedding_manager
    if _embedding_manager is None:
        path = config["embedding model"]["local_path"]
    #    logger.info(f"Loading embedding model for query variations from {path}")
        _embedding_manager = EmbeddingManager(embedmodel_path=path)
    return _embedding_manager


def attach_llm(loaded_processor, loaded_tokenizer, loaded_model):
    """Reuse a model already loaded elsewhere so Gemma is not loaded twice."""
    global processor, tokenizer, model
    processor = loaded_processor
    tokenizer = loaded_tokenizer
    model = loaded_model

def _ensure_llm():
    global processor, tokenizer, model

    if model is None:
        local_path = config["llm"]["local_path"]
    #    logger.info(f"Loading LLM for query variations from {local_path}")
        processor, tokenizer, model = load_model(local_path)
    return processor, tokenizer, model

def _generate_text(prompt_text, max_new_tokens=256, temperature=0.8):
    proc, tok, llm = _ensure_llm()
    messages = [{"role": "user", "content": prompt_text}]
    inputs = proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(llm.device)
    outputs = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

#--------------------------------
def query_veriation_generator(query):
    prompt = f"""Generate two different in-context variational queries
of the original query. You may only reorder keywords or add reasonable
related sub-keywords. Keep full sentences.

Original query: {query}

Respond with a Python list only, e.g. ['query1', 'query2'] whose data type is python list.
"""
    text = _generate_text(prompt, max_new_tokens=256, temperature=0.8)
    if not text:
        return []

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = ast.literal_eval(match.group(0))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass

    return [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
#--------------------------------

def query_variations(original_query, min_sim=MIN_QUERY_SIM):
    try:
        variations = query_veriation_generator(original_query)
        variations = [v for v in (variations or []) if v and v != original_query]
        if not variations:
            return []

        vectors = _get_embedding_manager().generate_embeddings([original_query] + variations)
        original_vec = vectors[0]
        orig_norm = np.linalg.norm(original_vec)

        accepted = []  
        for variation, vec in zip(variations, vectors[1:]):
            score = float(np.dot(original_vec, vec) / (orig_norm * np.linalg.norm(vec)))
            if score >= min_sim:
                accepted.append(variation)
        return accepted
    except Exception as e:
        logger.error(f"Error in filter_query_variations: {e}")
        return []
