"""
retrieval/query_rewriter.py
---------------------------
Improvement 7: LLM-based Legal Query Rewriting.

Converts informal natural-language queries into precise legal terminology
before retrieval, bridging the vocabulary gap between everyday language
and formal statute text.

Example:
  Input : "My neighbor built something on my land without asking"
  Output: "criminal trespass encroachment property possession without permission IPC punishment"

This is a lightweight HyDE-inspired (Hypothetical Document Embeddings) approach
where instead of generating a fake document, we generate a legal keyword expansion.
"""

import requests
import json
import re

OLLAMA_URL = "http://localhost:11434"

# Best model for query rewriting: gemma3:4b-cloud
# - Not a thinking/reasoning model → keywords go directly to message.content
# - Cloud-accelerated, fast, instruction-tuned for precise output
# - Avoid thinking models (gpt-oss:120b-cloud, deepseek-r1) — they output to
#   message.thinking only and get truncated before producing keywords
REWRITE_MODEL = "gemma3:4b-cloud"

SYSTEM_PROMPT = """You are a legal terminology expert for Indian criminal law.

Your task: Convert informal queries into precise legal terminology and concepts.

CRITICAL RULES:
1. Output ONLY legal terms and concepts - NO section numbers (no "section 420", no "IPC 441")
2. Focus on: offence names, legal elements, criminal acts, legal consequences
3. Use formal legal language from Indian Penal Code terminology
4. Keep output to 8-12 keywords maximum
5. Output format: comma-separated keywords only

What to include:
- Legal offence names (theft, cheating, criminal trespass, defamation)
- Legal elements (dishonestly, wrongful gain, wrongful loss, without consent)
- Criminal acts (causing hurt, wrongful restraint, criminal intimidation)
- Legal consequences (punishment, imprisonment, fine)
- Property/rights terms (movable property, immovable property, possession)

What to EXCLUDE:
- Section numbers (never mention "section 420" or "IPC 441")
- Explanations or reasoning
- Questions or uncertainty markers

Examples:
Query: "My neighbor built a wall on my land without permission"
Output: criminal trespass, encroachment, immovable property, unlawful entry, possession, punishment

Query: "Someone cheated me and took my money"
Output: cheating, fraud, dishonestly inducing, wrongful gain, movable property, delivery, punishment

Query: "My husband beats me"
Output: cruelty, voluntarily causing hurt, domestic violence, physical harm, punishment

Query: "Someone spread false rumors about me"
Output: defamation, false imputation, reputation harm, publishing defamatory matter, punishment

CRITICAL: Your response must be ONLY the comma-separated keyword line. No explanation, no preamble."""


def rewrite_query(query: str, model: str = None) -> dict:
    """
    Rewrite an informal query into legal search terminology using Ollama.

    Args:
        query: The original user query (informal language).
        model: Ollama model name. If None, uses the first available model.

    Returns:
        dict with keys:
          - original   : original query
          - rewritten  : legal keyword expansion
          - model      : model used
          - error      : None or error message
    """
    result = {
        "original":  query,
        "rewritten": query,   # fallback: use original if rewriting fails
        "model":     model or "unknown",
        "error":     None,
    }

    try:
        if not model:
            model = REWRITE_MODEL
            result["model"] = model

        # Build prompt using chat format since large cloud models require it
        payload = {
            "model":  model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,     # low temperature for deterministic legal terms
                "num_predict": 300,     # enough for keyword output; safety net for any model
                "top_p": 0.9,
            },
        }

        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        response_data = resp.json()

        # Guard: if model hit token cap mid-output, don't parse a truncated result
        if response_data.get("done_reason") == "length":
            result["error"] = "Model response truncated (token cap hit — increase num_predict)"
            return result

        message = response_data.get("message", {})

        # Non-thinking models put output in content; thinking models use thinking
        raw = message.get("content", "").strip()
        
        # If content is empty, try to extract keywords from thinking field
        if not raw and "thinking" in message:
            thinking = message.get("thinking", "").strip()
            
            # Try to find the cleanest keyword list
            # Pattern 1: Look for "craft: keywords" or "output: keywords"
            craft_match = re.search(r'(?:craft|output):\s*([^.]+?)(?:\.|Count|Let|But|Also|$)', thinking, re.IGNORECASE)
            if craft_match:
                raw = craft_match.group(1).strip()
            else:
                # Pattern 2: Find sequences with multiple commas and legal terms
                # Split by sentences and find the one with most commas (likely the keyword list)
                sentences = re.split(r'[.!?]\s+', thinking)
                # Look for sentences with legal terms but NOT section numbers
                keyword_candidates = [
                    s for s in sentences 
                    if s.count(',') >= 3 
                    and any(term in s.lower() for term in ['criminal', 'offence', 'punishment', 'wrongful', 'dishonest', 'causing', 'property'])
                    and not re.search(r'\b(?:section|IPC|BNS)\s+\d+', s, re.IGNORECASE)
                ]
                if keyword_candidates:
                    raw = max(keyword_candidates, key=lambda s: s.count(','))

        # Clean output: remove any explanatory sentences, keep keywords
        first_line = raw.split("\n")[0].strip()
        
        # Remove common prefixes and reasoning phrases
        prefix_patterns = [
            r'^(?:We need to (?:output|produce) keywords?:|So keywords?:|This is about|Something like:|Could include)\s*',
            r'^[^:]+:\s+(?=\w)',  # Remove "Query description: " patterns
        ]
        for pattern in prefix_patterns:
            first_line = re.sub(pattern, '', first_line, flags=re.IGNORECASE)
        
        # Remove section numbers and IPC/BNS references (to prevent hallucination)
        section_patterns = [
            r'\b(?:IPC|BNS)\s+(?:section\s+)?\d+[A-Z]?\b',  # "IPC 420", "BNS section 441"
            r'\bsection\s+\d+[A-Z]?\b',  # "section 420"
            r'\b\d+[A-Z]?\s+(?:IPC|BNS)\b',  # "420 IPC"
        ]
        for pattern in section_patterns:
            first_line = re.sub(pattern, '', first_line, flags=re.IGNORECASE)
        
        # Remove common reasoning phrases and trailing incomplete words
        cleanup_patterns = [
            r'\s*\(.*?\)\s*',  # Remove parenthetical explanations
            r'\s+(?:maybe|also|but|however|actually|under|could be|is not|possibly|could include).*$',  # Remove reasoning tail
            r'\s*\?+\s*',  # Remove question marks
            r'\s+(?:Count|Let|But|Also|And|Or|Maybe|Actually|However|This|Could)\.?$',  # Remove trailing words
            r'^["\']|["\']$',  # Remove leading/trailing quotes
        ]
        for pattern in cleanup_patterns:
            first_line = re.sub(pattern, '', first_line, flags=re.IGNORECASE)
        
        # Clean up multiple commas and spaces
        first_line = re.sub(r',\s*,+', ',', first_line)  # Remove duplicate commas
        first_line = re.sub(r'\s+', ' ', first_line)  # Normalize spaces
        first_line = first_line.strip().rstrip(',').strip()

        # Remove common prefixes the model might add
        for prefix in ["Keywords:", "Output:", "Legal keywords:", "Search terms:"]:
            if first_line.lower().startswith(prefix.lower()):
                first_line = first_line[len(prefix):].strip()

        # If empty or too long, fall back to original
        if first_line and len(first_line) < 200:
            result["rewritten"] = first_line
        else:
            result["error"] = "Rewritten query was empty or malformed"

    except requests.exceptions.ConnectionError:
        result["error"] = "Ollama not running"
    except requests.exceptions.Timeout:
        result["error"] = "Ollama timeout (>30s)"
    except Exception as e:
        result["error"] = str(e)

    return result


def hybrid_rewrite(query: str, model: str = None) -> str:
    """
    Returns a fused query: original + rewritten keywords.
    This ensures BM25 and TF-IDF see BOTH the user's words AND legal terms.

    Example:
      "My neighbor built on my land" + "criminal trespass encroachment property"
      → "My neighbor built on my land criminal trespass encroachment property"
    """
    result = rewrite_query(query, model)
    if result["error"] or result["rewritten"] == result["original"]:
        return query
    # Fuse: append rewritten keywords to original
    return f"{query} {result['rewritten']}"


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "My neighbor built something on my land without asking",
        "Someone stole my phone at the market",
        "My boss hasn't paid my salary for 3 months",
        "A police officer slapped me during questioning",
        "Someone posted fake photos of me online to ruin my reputation",
    ]

    print("=" * 65)
    print("  LEGAL QUERY REWRITER TEST")
    print("=" * 65)

    for q in test_queries:
        res = rewrite_query(q)
        print(f"\nOriginal : {q}")
        if res["error"]:
            print(f"Error    : {res['error']}")
        else:
            print(f"Rewritten: {res['rewritten']}")
            print(f"Fused    : {hybrid_rewrite(q)}")
        print("-" * 65)
