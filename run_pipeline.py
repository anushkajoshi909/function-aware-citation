#!/usr/bin/env python3
import os, sys, json, subprocess, shutil, re, time
from pathlib import Path

# --- Paths ---
ROOT   = Path("/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project").resolve()
RAG    = ROOT / "RetrievalAugmentedGeneration"

CLASSIFIER = RAG / "classifying_question.py"
NB_IPYNB   = RAG / "Retreival_query_based.ipynb"
NB_PY      = RAG / "Retreival_query_based.py"      # produced by nbconvert
FUNCTION   = RAG / "function_based_answer.py"
FUNCTION_PATCHED = RAG / "function_based_final_patched.py"

CLASSIFIED = RAG / "classified_outputs.jsonl"
TOPK_RAG   = RAG / "outputs" / "topk_candidates_query.jsonl"
TOPK_ROOT  = ROOT / "outputs" / "topk_candidates_query.jsonl"   # optional compatibility link

# --- Helpers ---
def run(cmd, cwd=None):
    print(f"-> {' '.join(map(str, cmd))}  (cwd={cwd or os.getcwd()})")
    subprocess.run(cmd, cwd=cwd, check=True)

def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except FileNotFoundError:
            pass
    try:
        dst.symlink_to(src)
        print(f"🔗 Symlinked: {dst} -> {src}")
    except Exception as e:
        print(f"Symlink failed ({e}); copying instead.")
        shutil.copyfile(src, dst)
        print(f"📄 Copied: {src} -> {dst}")

def harden_exported_script(path: Path):
    """Make the exported notebook script safe to run headless."""
    txt = path.read_text()
    if "display(" in txt and "from IPython.display import display" not in txt:
        header = (
            "try:\n"
            "    from IPython.display import display\n"
            "except Exception:\n"
            "    def display(*args, **kwargs):\n"
            "        pass\n\n"
        )
        txt = header + txt
    if "get_ipython()" in txt:
        txt = re.sub(r"^.*get_ipython\(\).*?$", "pass  # stripped IPython magic", txt, flags=re.MULTILINE)
    path.write_text(txt)
    print(f"🩹 Hardened exported script: {path.name}")

def patch_function_final(src: Path, dst: Path):
    """Write a patched copy of function_based_final.py with a robust llm_chat override."""
    code = src.read_text()

    # Inject just BEFORE the __main__ block so we override any earlier llm_chat
    m = re.search(r'\nif __name__ == [\'"]__main__[\'"]\s*:\s*', code)
    inject_idx = m.start() if m else len(code)

    override = r'''
# ==== Injected by run_pipeline.py: Robust llm_chat override (self-contained) ====
import os as _rp_os
import time as _rp_time

try:
    from openai import OpenAI as _RP_OpenAI
except Exception:
    _RP_OpenAI = None  # will raise if needed

def _rp_get_client():
    # reuse existing client if present
    if "client" in globals() and globals().get("client") is not None:
        return globals()["client"]
    if _RP_OpenAI is None:
        raise RuntimeError("OpenAI client not available for injected llm_chat.")
    # try to build a client from env/file
    api_key = (_rp_os.getenv("SCADS_API_KEY")
               or (_rp_os.path.exists(_rp_os.path.expanduser("~/.scadsai-api-key")) and
                   open(_rp_os.path.expanduser("~/.scadsai-api-key")).read().strip())
               or _rp_os.getenv("OPENAI_API_KEY"))
    base_url = _rp_os.getenv("SCADS_BASE_URL", "https://llm.scads.ai/v1")
    c = _RP_OpenAI(base_url=base_url, api_key=api_key) if base_url else _RP_OpenAI(api_key=api_key)
    globals()["client"] = c
    return c

def _rp_model_fallback():
    return globals().get("DEFAULT_MODEL") or _rp_os.getenv("MODEL_NAME") or "openai/gpt-oss-120b"

def _extract_choice_text(resp):
    try:
        if not resp or not getattr(resp, "choices", None):
            return None
        ch = resp.choices[0]
        msg = getattr(ch, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if content:
                return content if isinstance(content, str) else str(content)
        txt = getattr(ch, "text", None)
        if txt:
            return txt if isinstance(txt, str) else str(txt)
        cnt = getattr(ch, "content", None)
        if cnt:
            return cnt if isinstance(cnt, str) else str(cnt)
    except Exception:
        return None
    return None

def llm_chat(prompt, model=None, max_tokens=512, temperature=0.2, retries=2, sleep_s=0.7):
    """
    Safe chat wrapper:
    - creates client if missing
    - tolerates None/empty content
    - supports both chat and text-style responses
    - light retry to dodge transient empties under load
    """
    _rp_debug = bool(globals().get("DEBUG", False))
    if model is None:
        model = _rp_model_fallback()
    last_err = None
    for attempt in range(retries + 1):
        try:
            _c = _rp_get_client()
            resp = _c.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = _extract_choice_text(resp)
            if text and text.strip():
                return text.strip()
            if _rp_debug:
                print(f"[llm_chat] Empty content (attempt {attempt+1}/{retries+1}); resp={getattr(resp, 'id', 'no-id')}", flush=True)
        except Exception as e:
            last_err = e
            if _rp_debug:
                print(f"[llm_chat] API error (attempt {attempt+1}/{retries+1}): {e}", flush=True)
        _rp_time.sleep(sleep_s)
        temperature = 0.1  # nudge on retry
    if _rp_debug:
        print(f"[llm_chat] Giving up after {retries+1} attempts. Last error: {last_err}", flush=True)
    return ""  # never crash
# ==== End injected override ====
'''.lstrip("\n")

    patched = code[:inject_idx] + override + code[inject_idx:]
    dst.write_text(patched)
    print(f"🧩 Patched final-stage script written to: {dst.name}")

# --- Pipeline ---
def main():
    # Ensure outputs dirs exist
    (RAG / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    # 1) Classify
    run([sys.executable, str(CLASSIFIER)], cwd=str(RAG))
    if not CLASSIFIED.exists():
        raise FileNotFoundError(f"Expected {CLASSIFIED}")
    print("✅ Saved to classified_outputs.jsonl")
    print("🔧 Ready for retrieval module (handled in a separate file).")

    # 2) Export notebook to .py
    run(["jupyter", "nbconvert", "--to", "script", str(NB_IPYNB)], cwd=str(RAG))
    nb_script = NB_PY if NB_PY.exists() else (RAG / (NB_IPYNB.stem + ".py"))
    if not nb_script.exists():
        raise FileNotFoundError(f"Expected exported script at {NB_PY} or {RAG / (NB_IPYNB.stem + '.py')}")

    # 2b) Harden and run retrieval script
    harden_exported_script(nb_script)
    run([sys.executable, str(nb_script)], cwd=str(RAG))

    # 3) Check retrieval outputs (now in RAG/outputs)
    if not TOPK_RAG.exists():
        raise FileNotFoundError(f"Expected retrieval output at {TOPK_RAG}")
    print(f"✅ Top-k candidates: {TOPK_RAG}")

    # Optional: create compatibility link at ROOT/outputs
    safe_symlink(TOPK_RAG, TOPK_ROOT)

    # 4) Patch final stage and run with a small cool-down + retry
    patch_function_final(FUNCTION, FUNCTION_PATCHED)

    time.sleep(2)
    try:
        run([sys.executable, str(FUNCTION_PATCHED), "--debug", "--max-check", "20"], cwd=str(RAG))
    except subprocess.CalledProcessError:
        print("⚠️ Final stage failed once; pausing and retrying…")
        time.sleep(4)
        run([sys.executable, str(FUNCTION_PATCHED), "--debug", "--max-check", "20"], cwd=str(RAG))

    print("\n🎉 Pipeline complete.")
    print(f"  • Classified file : {CLASSIFIED}")
    print(f"  • TopK (canonical): {TOPK_RAG}")
    print(f"  • Final outputs   : {RAG / 'outputs'}")

if __name__ == "__main__":
    main()
