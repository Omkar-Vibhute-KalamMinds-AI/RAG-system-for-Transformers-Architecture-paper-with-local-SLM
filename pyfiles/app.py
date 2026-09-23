import pyfiles_path  # noqa: F401
from logger import logger
from config_loader import load_config
from query_variationar import attach_embedding_manager, query_variations

config = load_config()

# Production LanceDB / embedder are created on first use so tests can import
# AdvancedRAGPipeline without a local transformer_table.
db = None
transformers_table = None
embedding_manager = None
rag_retrieve = None


def init_retriever():
    """Connect LanceDB and load the embedder (no-op if already initialized)."""
    global db, transformers_table, embedding_manager, rag_retrieve
    if rag_retrieve is not None:
        return rag_retrieve

    import lancedb
    from EmbedModelLoader import EmbeddingManager
    from RetrievalSystem import RAGRetriever

    db = lancedb.connect(config["vectordb"]["path"])
    transformers_table = db.open_table(config["vectordb"]["table"])
    embed_model_name = config["embedding model"]["local_path"]
    embedding_manager = EmbeddingManager(embed_model_name)
    attach_embedding_manager(embedding_manager)
    rag_retrieve = RAGRetriever(transformers_table, embedding_manager)
    return rag_retrieve

#----------Main systme------------------------------ 
import time
import torch
from threading import Thread
from transformers import TextIteratorStreamer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdvancedRAGPipeline:
    def __init__(
        self,
        retriever,
        model,
        processor,
        use_history: bool = True,
        history_window: int = 3,
        use_query_variations: bool = True,
    ):
        """
        use_history: system-level setting — if False, conversation history is never
                     tracked or included in prompts, regardless of what happens during queries.
        history_window: number of previous turns to include in the prompt when use_history is True.
        use_query_variations: if False, retrieve only with the original question (skip LLM paraphrases).
        """
        self.retriever = retriever
        self.model = model
        self.processor = processor
        pipeline_cfg = config["AdvancedRAGPipeline"]
        self.use_history = pipeline_cfg.get("use_history", use_history)
        self.history_window = pipeline_cfg.get("history_window", history_window)
        self.use_query_variations = pipeline_cfg.get("use_query_variations", use_query_variations)
        self.history = []
        self.default_top_k = config["retriever"]["top_k"]
        self.default_min_score = config["retriever"]["min_score"]
        self.default_max_new_tokens = config["llm"].get("max_new_tokens", 1024)
        self.default_temperature = config["llm"].get("temperature", 0.4)
        self.default_do_sample = config["llm"].get("do_sample", True)
        self.default_stream = pipeline_cfg.get("stream", True)
        self.default_summarize = pipeline_cfg.get("summarize", False)

    def _build_inputs(self, prompt_text):
        """Builds model inputs using Gemma's chat template — required for multimodal processors."""
        if isinstance(prompt_text, str):
            messages = [{"role": "user", "content": prompt_text}]
        else:
            messages = prompt_text
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        return inputs
   
#----------------------------------------------------------  
    def generate(self, prompt_text, max_new_tokens=None, temperature=None, do_sample=None):
        """Generates a response all at once (no streaming) — used for summaries etc."""
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        if temperature is None:
            temperature = self.default_temperature
        if do_sample is None:
            do_sample = self.default_do_sample
        inputs = self._build_inputs(prompt_text)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
#-------------------------------------------------- 

    def generate_streaming(self, prompt_text, max_new_tokens=None, temperature=None, do_sample=None, echo: bool = False):
        """Generates a response token-by-token. Yields text chunks for the API."""
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        if temperature is None:
            temperature = self.default_temperature
        if do_sample is None:
            do_sample = self.default_do_sample
        inputs = self._build_inputs(prompt_text)
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if echo:
                print(new_text, end="", flush=True)
            yield new_text

        thread.join()
        if echo:
            print() 
#--------------------------------------------------   
   
#--------------------------------------------------  
    def _collect_hits(self, search_queries, top_k):
        results = []
        seen_ids = set()
        for q in search_queries:
            hits = self.retriever.retrieve(q, top_k=top_k)
            for doc in hits:
                if doc["id"] in seen_ids:
                    continue
                seen_ids.add(doc["id"])
                results.append(doc)
        return results

    def _prepare_query(self, question: str, top_k: int = None, use_query_variations: bool = None) -> dict:
        """Retrieve context and build the RAG prompt (shared by JSON and streaming)."""
        if top_k is None:
            top_k = self.default_top_k
        if use_query_variations is None:
            use_query_variations = self.use_query_variations

        variations = []
        if use_query_variations:
            variations = query_variations(question) or []
        search_queries = [question] + [v for v in variations if v != question]
        results = self._collect_hits(search_queries, top_k)
        sources = []
        context = ""
        if results:
            context = "\n\n".join([doc["content"] for doc in results])
            sources = [{
                "source": doc["source_file"],
                "rank": doc.get("rank"),
                "distances": doc.get("distance"),
                "chunk_index": doc["chunk_index"],
                "doc_id": doc["id"],
                "preview": doc["content"][:120] + "...",
            } for doc in results]

        history_text = ""
        if self.use_history:
            for turn in self.history[-self.history_window:]:
                history_text += f"Previous Question: {turn['question']}\nPrevious Answer: {turn['answer']}\n\n"

        if use_query_variations and variations:
            variation_text = "; ".join(variations)
            prompt = f""" System_role: You are a enterprise ai assistant to a company, you reffer the user as Sir.

        Use the following context and conversation history and variations of the original query, which are semantically similar in meaning,
        to answer the question concisely.
        Variations of the original query: {variation_text}
        Conversation history: {history_text}
        Context: {context}
        Question: {question}
        """
        else:
            prompt = f""" System_role: You are a enterprise ai assistant to a company, you reffer the user as Sir.

        Use the following context and conversation history to answer the question concisely.
        Conversation history: {history_text}
        Context: {context}
        Question: {question}
        """
        return {"results": results, "sources": sources, "prompt": prompt}

    @staticmethod
    def _citations_suffix(sources) -> str:
        if not sources:
            return ""
        citations = [f"[{i+1}] {src['source']}" for i, src in enumerate(sources)]
        return "\n\nCitations:\n" + "\n".join(citations)

    def _record_history(self, question, answer, sources, summary):
        if self.use_history:
            self.history.append({
                "question": question,
                "answer": answer,
                "sources": sources,
                "summary": summary,
            })

    def query_streaming(
        self,
        question: str,
        top_k: int = None,
        min_score: float = None,
        summarize: bool = None,
        max_new_tokens: int = None,
        use_query_variations: bool = None,
    ):
        """Yield RAG answer tokens as they are generated, then citations."""
        prepared = self._prepare_query(
            question, top_k=top_k, use_query_variations=use_query_variations
        )
        sources = prepared["sources"]
        if not prepared["results"]:
            yield "No relevant context found"
            self._record_history(question, "No relevant context found", [], None)
            return

        answer_parts = []
        for chunk in self.generate_streaming(prepared["prompt"], max_new_tokens=max_new_tokens):
            answer_parts.append(chunk)
            yield chunk

        citations = self._citations_suffix(sources)
        if citations:
            yield citations

        answer = "".join(answer_parts)
        summary = None
        if summarize is None:
            summarize = self.default_summarize
        if summarize and answer:
            summary = self.generate(f"Summarize the following answer in 2 sentences: \n{answer}")
        self._record_history(question, answer, sources, summary)

    def query(
        self,
        question: str,
        top_k: int = None,
        min_score: float = None,
        stream: bool = None,
        summarize: bool = None,
        use_query_variations: bool = None,
    ) -> dict:
        if stream is None:
            stream = self.default_stream
        if summarize is None:
            summarize = self.default_summarize

        prepared = self._prepare_query(
            question, top_k=top_k, use_query_variations=use_query_variations
        )
        sources = prepared["sources"]
        results = prepared["results"]
        prompt = prepared["prompt"]

        if not results:
            answer = "No relevant context found"
        elif stream:
            answer = "".join(self.generate_streaming(prompt, echo=True))
        else:
            answer = self.generate(prompt)

        answer_with_citations = answer + self._citations_suffix(sources)

        summary = None
        if summarize and results and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences: \n{answer}"
            summary = self.generate(summary_prompt)

        self._record_history(question, answer, sources, summary)

        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history if self.use_history else None,
        }
       #-------------------------------------------------- 
            
#if __name__ == "__main__":
#print("RAG retriever is ready. Start the API with: python modelapi_app.py") 
