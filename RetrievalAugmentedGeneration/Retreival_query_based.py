try:
    from IPython.display import display
except Exception:
    def display(*args, **kwargs):
        pass

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time


# In[2]:


# ---- EDIT THESE PATHS IF NEEDED ----
jsonl_files = [
    "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/Corpus/processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0101_001.jsonl",
    "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/Corpus/processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0102_001.jsonl",
    "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/Corpus/processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0103_001.jsonl",
    "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/Corpus/processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0104_001.jsonl",
]

# Directory where we'll save the index + metadata
out_dir = Path("e5_index_subset_1")
out_dir.mkdir(exist_ok=True, parents=True)

# Model to use: 'intfloat/e5-base-v2' is a good speed/quality trade-off
E5_MODEL_NAME = "intfloat/e5-small-v2"

# Limit (optional): set to None to index everything
MAX_PAPERS = None  # e.g., 10000


# In[3]:


# 🔒 Force CPU + tame threads + use local cache to avoid any network / CUDA shenanigans
import os, torch, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # force no-GPU path
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # quieter + safer in notebooks
os.environ["HF_HOME"] = "./hf_cache"             # local cache (no network)

from sentence_transformers import SentenceTransformer

E5_MODEL_NAME = "intfloat/e5-small-v2"  # swap to e5-base-v2 later

t0 = time.time()
model = SentenceTransformer(E5_MODEL_NAME, device="cpu", cache_folder="./hf_cache")
print("Loaded model in", round(time.time()-t0, 2), "s")

# warmup to avoid first-call lag
_ = model.encode(["query: warmup"], normalize_embeddings=True)
print("Warmup ok")


# In[4]:


# build_meta_with_authors.py (cell)
import json, ast, re, os
from datetime import datetime
from tqdm import tqdm
import pandas as pd

MAX_PAPERS = None  # or an int to truncate during dev

_yr_re = re.compile(r"(19|20)\d{2}")

def best_year_from_obj(obj):
    for key in ("year","published","date","update_date","created"):
        if key in obj and obj[key]:
            s = str(obj[key])
            try:
                y = int(s[:4])
                if 1900 <= y <= datetime.now().year + 1:
                    return y
            except Exception:
                m = _yr_re.search(s)
                if m: return int(m.group(0))
    md = obj.get("metadata") or {}
    pid = obj.get("paper_id") or md.get("id") or obj.get("id") or md.get("arxiv_id") or obj.get("arxiv_id")
    if isinstance(pid, str) and "/" in pid:
        try:
            yy = int(pid.split("/")[1][:2])
            return 2000 + yy if yy < 50 else 1900 + yy
        except Exception:
            pass
    return None

def extract_title_abstract(obj):
    title = None
    md = obj.get("metadata") or {}
    title = md.get("title") or obj.get("title")
    abstract = None
    if isinstance(md.get("abstract"), str):
        abstract = md["abstract"]
    if not abstract and isinstance(obj.get("abstract"), dict):
        abstract = obj["abstract"].get("text")
    if not abstract:
        abstract = obj.get("abstract")
    return title, abstract

def norm_raw_authors(raw):
    if raw is None: return []
    if isinstance(raw, (list, tuple)): return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        try: data = json.loads(s)
        except Exception:
            try: data = ast.literal_eval(s)
            except Exception: data = None
        if isinstance(data, list): return norm_raw_authors(data)
    sep = ";" if ";" in s else ","
    return [t.strip() for t in s.split(sep) if t.strip()]

def authors_from_parsed(ap):
    out=[]
    if isinstance(ap, list):
        for it in ap:
            if isinstance(it, dict):
                nm=(" ".join([it.get("first",""), it.get("last","")])).strip()
            elif isinstance(it, (list,tuple)):
                last=str(it[0]).strip() if len(it)>0 else ""
                first=str(it[1]).strip() if len(it)>1 else ""
                nm=(" ".join([first,last])).strip()
            else:
                nm=str(it).strip()
            if nm: out.append(nm)
    return out

def authors_from_obj(obj):
    md = obj.get("metadata") or {}
    if "authors_parsed" in obj:
        a = authors_from_parsed(obj["authors_parsed"])
        if a: return a
    if "authors_parsed" in md:
        a = authors_from_parsed(md["authors_parsed"])
        if a: return a
    if "authors" in obj:
        a = norm_raw_authors(obj["authors"])
        if a: return a
    if "authors" in md:
        a = norm_raw_authors(md["authors"])
        if a: return a
    return []

def get_pid(obj):
    md = obj.get("metadata") or {}
    return obj.get("paper_id") or md.get("id") or obj.get("id") or md.get("arxiv_id") or obj.get("arxiv_id")

rows = []
for path in jsonl_files:
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {os.path.basename(path)}"):
            line = line.strip()
            if not line: 
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = get_pid(obj)
            title, abstract = extract_title_abstract(obj)
            if not pid or not title or not abstract:
                continue
            authors = authors_from_obj(obj)
            year = best_year_from_obj(obj)
            rows.append({
                "paper_id": pid,
                "title": title.strip(),
                "abstract": str(abstract).strip(),
                "authors": authors,
                "year": year
            })

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["paper_id"]).reset_index(drop=True)
if MAX_PAPERS: df = df.head(MAX_PAPERS)
print(f"Loaded {len(df)} unique papers")
print(df.head(3)[["paper_id","title","authors","year"]])




# In[5]:


# # Reload index & metadata
index = faiss.read_index(str(out_dir / "index.faiss"))
meta  = pd.read_parquet(out_dir / "meta.parquet")

# ✅ Reuse the already-loaded model from earlier
q_model = model                      # <-- do NOT call SentenceTransformer() again
_ = q_model.encode(["query: warmup"], normalize_embeddings=True)  # quick warmup

def encode_query(q: str):
    return q_model.encode([f"query: {q}"], normalize_embeddings=True).astype("float32")


# In[6]:


import re
import ast
import json
import numpy as np
import pandas as pd
from datetime import datetime

# -----------------------
# helpers (format/display)
# -----------------------

def _trim(text, max_chars=450):
    if not text:
        return ""
    s = str(text).strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"


def _format_authors(a, k=3):
    if a is None:
        return []
    if isinstance(a, (list, tuple)):
        names = [str(x).strip() for x in a if str(x).strip()]
    else:
        sep = ";" if ";" in str(a) else ","
        names = [t.strip() for t in str(a).split(sep) if t.strip()]
    if not names:
        return []
    return names[:k] + (["et al."] if len(names) > k else [])


_yr_re = re.compile(r"(19|20)\d{2}")

def _best_year(row):
    """Robustly extract a plausible year (int) from heterogeneous row fields without ambiguous truth checks."""
    def _first_scalar(x):
        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            return x[0] if len(x) > 0 else None
        return x

    for key in ("year", "published", "date", "update_date", "created"):
        if key in row:
            val = _first_scalar(row.get(key))
            if val is None:
                continue
            s = str(val)
            # try first 4 chars
            try:
                y = int(s[:4])
                if 1900 <= y <= datetime.now().year + 1:
                    return y
            except Exception:
                pass
            # regex fallback anywhere in the string
            m = _yr_re.search(s)
            if m:
                y = int(m.group(0))
                if 1900 <= y <= datetime.now().year + 1:
                    return y

    # last resort: parse from paper_id-like strings
    pid = row.get("paper_id") or row.get("arxiv_id") or row.get("id")
    if isinstance(pid, str) and "/" in pid:
        try:
            yy = int(pid.split("/")[1][:2])
            return 2000 + yy if yy < 50 else 1900 + yy
        except Exception:
            pass
    return None


def _extract_abstract(abstract_field):
    if abstract_field is None:
        return ""
    if isinstance(abstract_field, dict):
        return str(abstract_field.get("text") or abstract_field.get("abstract") or "")
    return str(abstract_field)

# -----------------------
# stopwords + token utils
# -----------------------

DEFAULT_STOPWORDS = {
    "a","an","and","the","of","to","in","on","for","with","by","as","at","or","but","if","than","then",
    "from","into","over","under","between","within","without","about","via","per","through","across",
    "is","are","was","were","be","been","being","have","has","had","do","does","did","can","could",
    "may","might","will","would","shall","should","must","not","no","nor","also","both","either","neither",
    "this","that","these","those","it","its","their","our","your","his","her","them","they","we","you","i",
    "such","thus","there","here","where","when","which","who","whom","whose","what","why","how",
    "using","use","used","based","approach","approaches","method","methods","result","results","show",
    "shows","shown","paper","study","work","new"
}

def minmax_norm(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)

_token_re = re.compile(r"\b\w+\b", re.UNICODE)

def tokenize(text):
    if text is None:
        return []
    return [t.lower() for t in _token_re.findall(str(text))]

def content_terms(tokens, stopwords, min_len=3):
    return [t for t in tokens if not t.isdigit() and len(t) >= min_len and t not in stopwords]

# -----------------------
# robust author extraction
# -----------------------

def _authors_from_row(row):

    def _norm_raw_authors(raw):
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            return [str(x).strip() for x in raw if str(x).strip()]
        s = str(raw).strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                data = json.loads(s)
            except Exception:
                try:
                    data = ast.literal_eval(s)
                except Exception:
                    data = None
            if isinstance(data, list):
                return _norm_raw_authors(data)
        sep = ";" if ";" in s else ","
        return [t.strip() for t in s.split(sep) if t.strip()]

    def _norm_authors_parsed(ap):
        out = []
        if isinstance(ap, (list, tuple)):
            for item in ap:
                if isinstance(item, (list, tuple)):
                    last = str(item[0]).strip() if len(item) > 0 else ""
                    first = str(item[1]).strip() if len(item) > 1 else ""
                    name = " ".join([first, last]).strip()
                    if name:
                        out.append(name)
                elif isinstance(item, dict):
                    first = str(item.get("first", "")).strip()
                    last = str(item.get("last", "")).strip()
                    name = " ".join([first, last]).strip()
                    if name:
                        out.append(name)
                else:
                    s = str(item).strip()
                    if s:
                        out.append(s)
        elif isinstance(ap, str) and ap.strip():
            try:
                data = json.loads(ap)
            except Exception:
                try:
                    data = ast.literal_eval(ap)
                except Exception:
                    data = None
            if isinstance(data, list):
                return _norm_authors_parsed(data)
        return out

    # avoid pd.notna() on non-scalars — just check for presence/non-empty
    if "authors" in row:
        names = _norm_raw_authors(row.get("authors"))
        if names:
            return names

    if "authors_parsed" in row:
        names = _norm_authors_parsed(row.get("authors_parsed"))
        if names:
            return names

    # nested metadata dict (JSONL)
    if "metadata" in row and isinstance(row.get("metadata"), dict):
        md = row["metadata"]
        if "authors" in md:
            names = _norm_raw_authors(md.get("authors"))
            if names:
                return names
        if "authors_parsed" in md:
            names = _norm_authors_parsed(md.get("authors_parsed"))
            if names:
                return names

    return []

# -----------------------
# meta building from JSONL
# -----------------------

def _rows_from_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            md = obj.get("metadata", {}) or {}
            authors_field = md.get("authors") or obj.get("authors")
            yield {
                "paper_id": obj.get("paper_id") or md.get("id"),
                "title": md.get("title", "") or obj.get("title", ""),
                "abstract": obj.get("abstract", md.get("abstract", "")),
                "authors": authors_field,
                "metadata": md,
            }

def build_meta_from_jsonl(paths):
    if isinstance(paths, str):
        paths = [paths]
    all_rows = []
    for p in paths:
        for row in _rows_from_jsonl(p):
            all_rows.append(row)
    df = pd.DataFrame(all_rows)
    if "paper_id" in df.columns:
        df = df.drop_duplicates(subset=["paper_id"], keep="first")
    if "title" in df.columns:
        df = df.drop_duplicates(subset=["title"], keep="first")
    df = df.reset_index(drop=True)
    return df

# -----------------------
# main search function
# -----------------------

# COSINE-ONLY retrieval (query-only)
def search_post_filter(
    query,
    topN=20,
    topK_return=10,
    normalize_scores=False,
    stopwords=None,
    min_term_len=3,
    abstract_chars=450,
    authors_shown=3
):
    """
    Requires globals: index (faiss index), meta (DataFrame aligned to index), encode_query (-> np.float32 [1,d])
    """
    stopwords = DEFAULT_STOPWORDS if stopwords is None else set(stopwords)

    # 1) semantic retrieve
    qv = encode_query(query)  # shape (1, d)
    scores, idxs = index.search(qv, int(topN))  # shapes (1, topN)
    scores, idxs = scores[0].astype(np.float32), idxs[0]

    # 2) order purely by cosine
    order = np.argsort(-scores)
    display_scores = minmax_norm(scores) if normalize_scores else scores

    # 3) lexical explainers
    q_terms_all  = tokenize(query)
    q_terms_used = content_terms(q_terms_all, stopwords, min_len=min_term_len)

    out = []
    limit = min(int(topK_return), len(order))
    for rank_pos in range(limit):
        r = order[rank_pos]
        if r < 0 or r >= len(idxs):
            continue  # safety
        row = meta.iloc[idxs[r]]

        title = row.get("title", "")
        abstract_txt = _extract_abstract(row.get("abstract", ""))
        paper_id = row.get("paper_id") or row.get("arxiv_id") or row.get("id")

        title_tokens = content_terms(tokenize(title), stopwords, min_len=min_term_len)
        abs_tokens   = content_terms(tokenize(abstract_txt), stopwords, min_len=min_term_len)

        title_matches = sorted(set(q_terms_used) & set(title_tokens))
        abs_matches   = sorted(set(q_terms_used) & set(abs_tokens))

        authors_list = _authors_from_row(row)
        authors_fmt  = _format_authors(authors_list, k=authors_shown)

        out.append({
            "score": float(display_scores[r]),
            "cosine": float(scores[r]),
            "title": title,
            "abstract": _trim(abstract_txt, abstract_chars),  # preview
            "abstract_full": abstract_txt,                    # full
            "arxiv_id": paper_id,
            "year": _best_year(row),
            "authors": authors_fmt,
            "title_matches": title_matches,
            "abs_matches": abs_matches,
            "query_terms_used": q_terms_used,
            # removed 'function_requested' (no desired_function in signature anymore)
        })
    return out

# -----------------------
# convenience: results -> DataFrame
# -----------------------

def to_df(res):
    cols = ["cosine","title","year","arxiv_id","authors","title_matches","abs_matches"]
    return pd.DataFrame([{k: r.get(k) for k in cols} for r in res])


# In[7]:


import pandas as pd

def _safe_join(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return ", ".join(str(t) for t in x)
    return str(x)

def display_results_table(results):
    df = pd.DataFrame([
        {
            "Title": r.get("title", ""),
            "Year": r.get("year", ""),
            "Authors": _safe_join(r.get("authors", [])),
            "Abstract": r.get("abstract", ""),  # already trimmed upstream
            "Title Matches": _safe_join(r.get("title_matches", [])),
            "Abstract Matches": _safe_join(r.get("abs_matches", [])),
            "Score (norm)": r.get("score", r.get("cosine", None)),
            "Cosine (raw)": r.get("cosine", None),
        } for r in (results or [])
    ])
    return df


# In[8]:


def process_current_classification(
    classified_path="classified_outputs.jsonl",
    out_dir="outputs",
    topN=50,
    topK_return=10,
    normalize_scores=True,
    debug=False
):
    """
    Read the latest query from `classified_outputs.jsonl`, run query-only retrieval
    via `search(...)`, and write top-k candidates to JSONL and CSV.

    - No sentence-level loop
    - No answer/explanations
    - Prefers full abstract if present in results (abstract_full), else falls back to abstract
    - Robust against numpy arrays to avoid: "The truth value of an array with more than one element is ambiguous"
    """
    import os, json
    from datetime import datetime
    import pandas as pd

    def read_last_jsonl(path: str):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            raise ValueError(f"No lines found in {path}")
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError as e:
            raise ValueError(f"Last line is not valid JSON: {e}\nLine: {lines[-1][:200]}...")

    def as_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    def join_if_list(x, sep=", "):
        if isinstance(x, list):
            return sep.join(str(t) for t in x)
        return "" if x is None else str(x)

    def safe_year(y):
        try:
            # Avoid numpy types/arrays ambiguity
            if isinstance(y, (list, tuple)):
                y = y[0] if y else None
            s = str(y).strip()
            if not s:
                return None
            return int(s[:4])
        except Exception:
            return None

    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "topk_candidates_query.jsonl")
    out_csv   = os.path.join(out_dir, "topk_candidates_query.csv")

    obj = read_last_jsonl(classified_path)

    query = (obj.get("query") or "").strip()
    if not query:
        raise ValueError("No 'query' found in the latest classified_outputs.jsonl entry.")

    # provenance of classifier labels (not used for ranking)
    cls = obj.get("citation_function_classification") or {}
    cls_funcs = cls.get("citation_functions") or []

    if debug:
        print(f"[debug] query: {query}")
        if cls_funcs:
            print(f"[debug] classifier labels: {cls_funcs}")

    retrieved_at = datetime.utcnow().isoformat()

    # ---- Query-only retrieval (use search; fall back to search_post_filter if needed) ----
    retrieval_fn = globals().get("search")
    if retrieval_fn is None:
        retrieval_fn = globals().get("search_post_filter")
    if retrieval_fn is None:
        raise RuntimeError("Neither 'search' nor 'search_post_filter' is defined in the current scope.")

    results = []
    try:
        # Ensure we always get a Python list (not a numpy array)
        res = retrieval_fn(
            query=query,
            topN=int(topN),
            topK_return=int(topK_return),
            normalize_scores=bool(normalize_scores)
        )
        if isinstance(res, list):
            results = res
        elif res is None:
            results = []
        else:
            # Defensive: convert iterables to list
            try:
                results = list(res)
            except Exception:
                results = []
        if debug:
            print(f"[debug] got {len(results)} candidates")
    except Exception as e:
        if debug:
            print(f"[debug] retrieval error: {e}")
        results = []

    # ---- Flatten results for export ----
    rows = []
    for rank, r in enumerate(results):
        # prefer full abstract if present in retrieval output
        abs_full = r.get("abstract_full")
        if abs_full is None or not str(abs_full).strip():
            abs_full = r.get("abstract", "")  # fallback

        rows.append({
            "rank": int(rank),
            "paper_id": r.get("arxiv_id") or r.get("paper_id") or "",
            "title": r.get("title", ""),
            "year": safe_year(r.get("year")),
            "authors": join_if_list(r.get("authors")),
            "abstract": str(abs_full or ""),
            "score": (float(r.get("score")) if r.get("score") is not None else None),   # normalized if provided
            "cosine": (float(r.get("cosine")) if r.get("cosine") is not None else None),
            "title_matches": join_if_list(r.get("title_matches")),
            "abs_matches": join_if_list(r.get("abs_matches")),
            "query_terms_used": join_if_list(r.get("query_terms_used")),
            "classifier_functions": join_if_list(as_list(cls_funcs)),
            "retrieval_error": None,
            "retrieved_at": retrieved_at
        })

    # If no results, emit a stub row so downstream doesn’t break
    if len(rows) == 0:
        rows.append({
            "rank": None,
            "paper_id": "",
            "title": "",
            "year": None,
            "authors": "",
            "abstract": "",
            "score": None,
            "cosine": None,
            "title_matches": "",
            "abs_matches": "",
            "query_terms_used": "",
            "classifier_functions": join_if_list(as_list(cls_funcs)),
            "retrieval_error": "no_results",
            "retrieved_at": retrieved_at
        })

    # ---- Write outputs (overwrite each run) ----
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    if debug:
        print(f"rows written: {len(rows)}")
        print(f"- {out_jsonl}\n- {out_csv}")

    return df


# In[9]:


# 1) Run llm_test.py to regenerate the file (it overwrites classified_outputs.jsonl)

# 2) Process the latest query (query-only retrieval)
df_all = process_current_classification(  # <- use the query-only function
    classified_path="classified_outputs.jsonl",
    out_dir="outputs",
    topN=50,
    topK_return=20,
    normalize_scores=True,
    debug=True
)

# 3) Inspect results (no sentence_idx anymore)
if df_all.empty:
    print("No rows saved — check your classification or retrieval.")
else:
    cols = ["rank","score","cosine","title","year","authors","paper_id","title_matches","abs_matches","query_terms_used","retrieved_at","classifier_functions"]
    cols = [c for c in cols if c in df_all.columns]  # keep only existing
    display(df_all[df_all["rank"].notna()].sort_values("rank")[cols].head(10))

