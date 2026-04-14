#!/usr/bin/env python3
"""run_eval.py

LLM evaluation runner with LangSmith integration and Semantic Data Auditor.

Setup:
    Set these environment variables:
    - LANGCHAIN_API_KEY: LangSmith API key
    - OPENAI_API_KEY: OpenAI API key
    - ANTHROPIC_API_KEY: Anthropic API key
    - GOOGLE_API_KEY: Google Generative AI key (optional)
    - SARVAM_API_KEY: Sarvam API key (optional)
    - OPENROUTER_API_KEY: OpenRouter API key (optional)

Usage:
    export LANGCHAIN_API_KEY=your_key
    export OPENAI_API_KEY=your_key
    export ANTHROPIC_API_KEY=your_key
    python run_eval.py
"""

from __future__ import annotations

import os
import time
import json
from typing import Dict, Any, Callable, Optional
from pathlib import Path

import anthropic
import google.generativeai as genai
from openai import OpenAI
from langsmith import Client


# Load environment variables
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize LangSmith client if key provided
client: Optional[Client] = None
if LANGCHAIN_API_KEY:
    client = Client(
        api_url="https://api.smith.langchain.com",
        api_key=LANGCHAIN_API_KEY
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file and return list of objects."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# MODEL CALL FUNCTIONS

def call_gpt(prompt: str) -> Dict[str, Any]:
    """Call OpenAI GPT model."""
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY environment variable not set"}
    
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
    try:
        start_time = time.time()
        res = client_openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        duration = time.time() - start_time
        return {
            "output": res.choices[0].message.content,
            "time_seconds": round(duration, 2),
            "tokens": {
                "prompt": res.usage.prompt_tokens,
                "completion": res.usage.completion_tokens,
                "total": res.usage.total_tokens
            }
        }
    except Exception as e:
        return {"error": str(e), "time_seconds": 0}


def call_claude_sonnet(prompt: str) -> Dict[str, Any]:
    """Call Anthropic Claude Sonnet model."""
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY environment variable not set"}
    
    client_ant = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        start_time = time.time()
        res = client_ant.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        duration = time.time() - start_time
        return {
            "output": res.content[0].text,
            "time_seconds": round(duration, 2),
            "tokens": {
                "prompt": res.usage.input_tokens,
                "completion": res.usage.output_tokens,
                "total": res.usage.input_tokens + res.usage.output_tokens
            }
        }
    except Exception as e:
        return {"error": str(e), "time_seconds": 0}


def call_claude_haiku(prompt: str) -> Dict[str, Any]:
    """Call Anthropic Claude Haiku model."""
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY environment variable not set"}
    
    client_ant = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        start_time = time.time()
        res = client_ant.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        duration = time.time() - start_time
        return {
            "output": res.content[0].text,
            "time_seconds": round(duration, 2),
            "tokens": {
                "prompt": res.usage.input_tokens,
                "completion": res.usage.output_tokens,
                "total": res.usage.input_tokens + res.usage.output_tokens
            }
        }
    except Exception as e:
        return {"error": str(e), "time_seconds": 0}


def call_gemini(prompt: str) -> Dict[str, Any]:
    """Call Google Gemini model."""
    if not GOOGLE_API_KEY:
        return {"error": "GOOGLE_API_KEY environment variable not set"}
    
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    try:
        start_time = time.time()
        res = model.generate_content(prompt)
        duration = time.time() - start_time
        return {
            "output": res.text,
            "time_seconds": round(duration, 2),
            "tokens": {
                "prompt": res.usage_metadata.prompt_token_count,
                "completion": res.usage_metadata.candidates_token_count,
                "total": res.usage_metadata.total_token_count
            }
        }
    except Exception as e:
        return {"error": str(e), "time_seconds": 0}


def call_qwen_72b(prompt: str) -> Dict[str, Any]:
    """Call Qwen model via OpenRouter."""
    if not OPENROUTER_API_KEY:
        return {"error": "OPENROUTER_API_KEY environment variable not set"}
    
    client_qwen = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    try:
        start_time = time.time()
        res = client_qwen.chat.completions.create(
            model="qwen/qwen-2.5-72b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000
        )
        duration = time.time() - start_time
        return {
            "output": res.choices[0].message.content,
            "time_seconds": round(duration, 2),
            "tokens": {
                "prompt": res.usage.prompt_tokens,
                "completion": res.usage.completion_tokens,
                "total": res.usage.total_tokens
            }
        }
    except Exception as e:
        return {"error": str(e), "time_seconds": 0}


def call_sarvam_105b(prompt: str) -> Dict[str, Any]:
    """Call Sarvam model."""
    if not SARVAM_API_KEY:
        return {"error": "SARVAM_API_KEY environment variable not set"}
    
    client_sarvam = OpenAI(
        base_url="https://api.sarvam.ai/v1",
        api_key=SARVAM_API_KEY
    )
    try:
        start_time = time.time()
        res = client_sarvam.chat.completions.create(
            model="sarvam-105b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10000
        )
        duration = time.time() - start_time
        return {
            "output": res.choices[0].message.content,
            "time_seconds": round(duration, 2),
            "tokens": {
                "prompt": res.usage.prompt_tokens,
                "completion": res.usage.completion_tokens,
                "total": res.usage.total_tokens
            }
        }
    except Exception as e:
        return {"error": str(e), "time_seconds": 0}


# SEMANTIC AUDITOR EVALUATOR

def semantic_audit_evaluator(run, example) -> Dict[str, Any]:
    """Semantic data auditor for evaluating model outputs."""
    if not OPENAI_API_KEY:
        return {"key": "correctness", "score": 0, "comment": "OPENAI_API_KEY not set"}
    
    eval_client = OpenAI(api_key=OPENAI_API_KEY)
    
    audit_prompt = f"""
    You are a Semantic Data Auditor. Determine if the Model Output correctly 
    captured the DATA from the Reference, even if formatting varies.

    Scoring Philosophy:
    - "GST_Number" vs "gstin" = MATCH
    - "Delhi" vs "DELHI" = MATCH
    - "" vs null = MATCH
    - ONLY penalize if actual DATA is wrong, missing, or hallucinated

    Reference: {json.dumps(example.outputs)}
    Model Output: {run.outputs.get("output")}

    Return JSON:
    {{
      "score": <0-1>,
      "mismatched_fields": [],
      "missing_fields": [],
      "hallucinated_fields": [],
      "explanation": "Brief summary"
    }}
    """
    try:
        response = eval_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": audit_prompt}],
            response_format={"type": "json_object"}
        )
        audit_data = json.loads(response.choices[0].message.content)
        
        return {
            "key": "correctness",
            "score": audit_data.get("score", 0),
            "comment": json.dumps(audit_data)
        }
    except Exception as e:
        return {"key": "correctness", "score": 0, "comment": f"Error: {str(e)}"}


def run_model_test(inputs: Dict[str, str], model_func: Callable) -> Dict[str, Any]:
    """Run model and return structured results."""
    res = model_func(inputs.get("prompt", ""))
    return {
        "output": res.get("output") or res.get("error"),
        "metadata": {
            "latency": res.get("time_seconds"),
            "total_tokens": res.get("tokens", {}).get("total", 0),
            "prompt_tokens": res.get("tokens", {}).get("prompt", 0),
            "completion_tokens": res.get("tokens", {}).get("completion", 0)
        }
    }


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Compute exact match percentage."""
    if not predictions or not references:
        return 0.0
    matched = sum(
        1 for p, r in zip(predictions, references)
        if p.strip() == r.strip()
    )
    return float(matched) / len(predictions) * 100


def run_eval(predictions_path: Path, references_path: Path) -> None:
    """Run evaluation on predictions vs references."""
    predictions_data = load_jsonl(predictions_path)
    references_data = load_jsonl(references_path)

    if len(predictions_data) != len(references_data):
        raise ValueError(
            "Prediction and reference files must contain the same number of lines"
        )

    predictions = [
        item.get("prediction", "") if isinstance(item, dict) else str(item)
        for item in predictions_data
    ]
    references = [
        item.get("reference", "") if isinstance(item, dict) else str(item)
        for item in references_data
    ]

    em = compute_exact_match(predictions, references)
    print(f"Exact Match: {em:.2f}%")


if __name__ == "__main__":
    # Example usage
    models_to_test = [
        ("gpt-4", call_gpt),
        ("claude-sonnet", call_claude_sonnet),
        ("claude-haiku", call_claude_haiku),
        ("gemini", call_gemini),
    ]

    if client:
        print("🚀 Starting LangSmith evaluations...")
        
        for name, func in models_to_test:
            print(f"  Testing {name}...")
            
            def target(inputs):
                return run_model_test(inputs, func)
            
            # Uncomment to run with LangSmith:
            # client.evaluate(
            #     target,
            #     data="dataset_id_here",
            #     evaluators=[semantic_audit_evaluator],
            #     experiment_prefix=f"eval-{name}",
            #     max_concurrency=1
            # )
    else:
        print("⚠️  LANGCHAIN_API_KEY not set. Skipping LangSmith evaluations.")
        print("Set environment variables to use this script fully.")
