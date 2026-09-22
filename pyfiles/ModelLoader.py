from transformers import AutoProcessor, AutoModelForImageTextToText
import sys

import _bootstrap  # noqa: F401
from exception import ModelLoaderException
import torch 
from logger import logger 

def load_model(local_path):
    """Load processor, tokenizer, and model from a local path."""
    try:
        processor = AutoProcessor.from_pretrained(local_path)
        tokenizer = processor.tokenizer

        model = AutoModelForImageTextToText.from_pretrained(
            local_path,
            low_cpu_mem_usage=True
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    #    print('Model loaded successfully')
        return processor, tokenizer, model

    except Exception as e:
        raise ModelLoaderException(e, sys) from e


import time
import logging

#-----------------------------------------------------

def generate(prompt_text, tokenizer, model, max_new_tokens=1024):

    logger.debug(f"Prompt text ({len(prompt_text)} chars): {prompt_text[:200]}...")
    start = time.time()

    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        logger.debug(f"Tokenized input — input_ids shape: {inputs['input_ids'].shape}")

        outputs = model.generate(
            **inputs,  
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        elapsed = time.time() - start
        return response

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Generation failed after {elapsed:.2f}s: {e}", exc_info=True)
        raise

