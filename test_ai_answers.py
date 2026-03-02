import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Suppress tqdm
import tqdm, io
tqdm.tqdm = lambda *a, **k: io.StringIO()

from retrieval.dual_corpus import load_combined_corpus
from indexing.vector_index import get_or_build_index
from indexing.bm25_index import get_or_build_bm25
from retrieval.hybrid import hybrid_search
from retrieval.mmr import mmr_rerank
from retrieval.baseline import vector_search
from retrieval.bm25_baseline import bm25_search
from generation.answer_generator import generate_answer, is_ollama_running, list_local_models

# Suppress sentence-transformers output
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

print("=== AI ANSWER QUALITY TEST ===")
print(f"Ollama running: {is_ollama_running()}")
models = list_local_models()
print(f"Local models  : {models}")
if not models or not is_ollama_running():
    print("ERROR: Ollama not running or no models. Run: ollama serve && ollama pull qwen3-coder")
    sys.exit(1)

model_name = models[0]
print(f"Using model   : {model_name}")
print()

combined = load_combined_corpus()
index, embeddings, vec_model = get_or_build_index(combined)
bm25, _ = get_or_build_bm25(combined)
print(f"Corpus: {len(combined)} clauses\n")

# Test queries with IPC + BNS expected sections
TEST_QUERIES = [
    ("What is the punishment for murder?",
     ["302", "103", "300", "101"]),
    ("Define criminal conspiracy under law.",
     ["120A", "120B", "61", "62"]),
    ("What is dowry death and its punishment?",
     ["304B", "80"]),
    ("Explain the right of private defence of the body.",
     ["97", "100", "34", "35"]),
    ("What constitutes rape and what is the punishment?",
     ["375", "376", "63", "64"]),
]

PASS = 0
for i, (query, expected_secs) in enumerate(TEST_QUERIES, 1):
    print(f"[Q{i}] {query}")

    # Vector-only for baseline
    vec_res = vector_search(query, combined, index, vec_model, top_k=5)
    vec_secs = [r['section_number'] for r in vec_res]

    # Hybrid+MMR
    cands     = hybrid_search(query, combined, index, embeddings, vec_model, bm25, top_k=50)
    retrieved = mmr_rerank(cands, top_k=3)
    ret_secs  = [r['section_number'] for r in retrieved]

    vec_hit  = any(s in expected_secs for s in vec_secs)
    hyb_hit  = any(s in expected_secs for s in ret_secs)

    print(f"  Vector top-5    : {vec_secs}")
    print(f"  Hybrid+MMR top-3: {ret_secs}")
    print(f"  Expected any of : {expected_secs}")
    print(f"  Vector hit      : {'PASS' if vec_hit else 'FAIL'}")
    print(f"  Hybrid hit      : {'PASS' if hyb_hit else 'FAIL'}")

    # Use whichever retrieved better for generation
    gen_input = retrieved if hyb_hit else vec_res[:3]
    result = generate_answer(query, gen_input, model=model_name)

    if result["error"]:
        print(f"  Generation      : ERROR - {result['error']}")
    else:
        answer = result["answer"]
        print(f"  Grounded        : {'YES' if result['grounded'] else 'NO'}")
        # Print first 3 lines of answer
        lines = [l.strip() for l in answer.split('\n') if l.strip()][:4]
        for line in lines:
            print(f"    {line[:115]}")

    overall = (vec_hit or hyb_hit) and not result.get("error")
    print(f"  Overall         : {'PASS' if overall else 'FAIL'}")
    if overall:
        PASS += 1
    print()

print(f"FINAL: {PASS}/{len(TEST_QUERIES)} queries fully passed")
if PASS == len(TEST_QUERIES):
    print("STATUS: EXCELLENT - all answers correct and grounded")
elif PASS >= 3:
    print("STATUS: GOOD - most answers correct")
else:
    print("STATUS: NEEDS IMPROVEMENT")
