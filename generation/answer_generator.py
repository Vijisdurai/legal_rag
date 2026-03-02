"""
generation/answer_generator.py
-------------------------------
RAG answer generation using Ollama's cloud LLMs.
No API keys, no token billing — completely free and reproducible.

Supported models (Ollama cloud):
  glm-5:cloud, deepseek-v3.2:cloud, gpt-oss:120b-cloud
"""

import os
import sys
import requests
import json

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, BASE_DIR)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Only GPT-OSS 120B cloud model is used
AVAILABLE_MODELS = [
    "gpt-oss:120b-cloud",      # GPT-OSS 120B — frontier quality, structured output
]

DEFAULT_MODEL = "gpt-oss:120b-cloud"

_SYSTEM_PROMPT = (
    "You are a precise Indian legal assistant. Your task is to answer legal "
    "queries using ONLY the IPC sections provided. Always cite section numbers "
    "explicitly (e.g., 'Under Section 302 IPC...'). If the sections do not "
    "contain enough information, say so clearly. Do not hallucinate or add "
    "information not present in the provided sections."
)


def is_ollama_running(url: str = OLLAMA_URL) -> bool:
    """Check if Ollama server is reachable."""
    try:
        r = requests.get(f"{url}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def list_local_models(url: str = OLLAMA_URL) -> list[str]:
    """Return installed Ollama model names."""
    try:
        r = requests.get(f"{url}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def format_sections(retrieved: list[dict]) -> str:
    """Format retrieved clauses for prompt injection."""
    parts = []
    for i, r in enumerate(retrieved[:4], 1):
        sec = r.get("section_number", "?")
        title = r.get("title", r.get("snippet", "")[:80])
        text = r.get("text", "")[:600]
        parts.append(f"[{i}] Section {sec} IPC — {title}\n{text}")
    return "\n\n".join(parts)


def generate_answer(query: str,
                    retrieved_clauses: list[dict],
                    model: str = DEFAULT_MODEL,
                    ollama_url: str = OLLAMA_URL,
                    stream: bool = False) -> dict:
    """
    Generate a grounded legal answer using Ollama.

    Args:
        query:             User's legal query.
        retrieved_clauses: Top retrieved IPC clause dicts.
        model:             Ollama model name.
        ollama_url:        Ollama server base URL.
        stream:            If True, stream response tokens (for UI).

    Returns:
        {
            "answer":    str,   # Generated answer text
            "model":     str,   # Model used
            "sections":  list,  # Sections cited (section_number list)
            "grounded":  bool,  # True if any section number appears in answer
            "error":     str | None
        }
    """
    sections_text = format_sections(retrieved_clauses)
    sections_used = [r["section_number"] for r in retrieved_clauses[:4]]

    prompt = (
        f"Legal Query: {query}\n\n"
        f"Relevant IPC Sections:\n{sections_text}\n\n"
        f"Answer (cite section numbers, stay within provided sections):"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        "stream": stream,
        "options": {
            "temperature": 0.1,      # Low temperature — factual, precise
            "top_p": 0.9,
            "num_predict": 512,
        }
    }

    try:
        r = requests.post(
            f"{ollama_url}/api/chat",
            json=payload,
            timeout=60
        )
        r.raise_for_status()
        answer = r.json().get("message", {}).get("content", "").strip()

        # Hallucination guard: check if any section number is cited
        grounded = any(f"Section {s}" in answer or f"§{s}" in answer or
                       f"section {s}" in answer.lower()
                       for s in sections_used)

        return {
            "answer": answer,
            "model": model,
            "sections": sections_used,
            "grounded": grounded,
            "error": None,
        }

    except requests.exceptions.ConnectionError:
        return {
            "answer": "",
            "model": model,
            "sections": sections_used,
            "grounded": False,
            "error": (
                "Ollama is not running. Start it with: `ollama serve`\n"
                "Then pull a model: `ollama pull qwen3-coder`"
            ),
        }
    except Exception as e:
        return {
            "answer": "",
            "model": model,
            "sections": sections_used,
            "grounded": False,
            "error": str(e),
        }


if __name__ == '__main__':
    print(f"Ollama running: {is_ollama_running()}")
    print(f"Local models  : {list_local_models()}")

    # Quick test
    mock_clauses = [{
        "section_number": "302",
        "title": "Punishment for murder",
        "text": "302. Punishment for murder.\nWhoever commits murder shall be punished with death, or "
                "imprisonment for life, and shall also be liable to fine.",
    }]
    result = generate_answer("What is the punishment for murder?", mock_clauses)
    print(f"\nModel: {result['model']}")
    print(f"Grounded: {result['grounded']}")
    if result['error']:
        print(f"Error: {result['error']}")
    else:
        print(f"Answer: {result['answer']}")
