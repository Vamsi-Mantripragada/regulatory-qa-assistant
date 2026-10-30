# ===== Regulatory Reporting Q&A Assistant (RAG): deployment version =====
# Same logic as the Colab version, without Colab-only lines (!pip, !wget, userdata).
# Runs on Hugging Face Spaces or any computer:  python app.py
# Keys come from environment variables (Space "Secrets"), never from code:
#   GROQ_API_KEY (free), GEMINI_API_KEY (free), OPENAI_API_KEY (paid). Missing ones are skipped.
# Public documents only.

import os
import re
import difflib
from collections import Counter

import requests
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from google import genai
from google.genai import types
from openai import OpenAI

PROVIDERS = ["groq", "gemini", "openai"]        # order to try

# ---------- 1. Keys and clients (skip providers with no key) ----------
def get_key(name):
    key = os.environ.get(name)
    if not key:
        print(f"ℹ️ {name} not available, so this provider is skipped")
    return key

MODELS = {"groq": None, "gemini": "gemini-3.8-flash", "openai": "gpt-4o-mini"}
clients = {}
if get_key("GROQ_API_KEY"):
    clients["groq"] = OpenAI(api_key=get_key("GROQ_API_KEY"),
                             base_url="https://api.groq.com/openai/v1", timeout=60)
if get_key("GEMINI_API_KEY"):
    clients["gemini"] = genai.Client(api_key=get_key("GEMINI_API_KEY"),
                                     http_options=types.HttpOptions(timeout=60000))
if get_key("OPENAI_API_KEY"):
    clients["openai"] = OpenAI(api_key=get_key("OPENAI_API_KEY"), timeout=60)

# ---------- 2. Pick a working Groq model automatically ----------
def pick_groq_model():
    skip = ("whisper", "tts", "guard", "embed", "playai", "distil")
    prefer = ("llama", "gpt-oss", "qwen", "kimi", "gemma", "mixtral")
    ids = [m.id for m in clients["groq"].models.list()]
    candidates = [i for i in ids if not any(s in i.lower() for s in skip)]
    candidates.sort(key=lambda i: next((n for n, p in enumerate(prefer) if p in i.lower()), len(prefer)))
    for model_id in candidates:
        try:
            clients["groq"].chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": "Say OK"}], max_tokens=5)
            print("✅ Using Groq model:", model_id)
            return model_id
        except Exception as e:
            print("⏭️ skipping", model_id, "→", str(e)[:60])
    return None

if "groq" in clients:
    MODELS["groq"] = pick_groq_model()
    if MODELS["groq"] is None:
        del clients["groq"]

# ---------- 3. Download and read the PDF (once) ----------
PDF_URL = ("https://www.bankofengland.co.uk/-/media/boe/files/"
           "prudential-regulation/crd-iv/mlar-notes-jan-2022.pdf")
PDF_PATH = "boe_guide.pdf"
if not os.path.exists(PDF_PATH):
    r = requests.get(PDF_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    with open(PDF_PATH, "wb") as f:
        f.write(r.content)
reader = PdfReader(PDF_PATH)
pages = [{"page": i, "text": p.extract_text() or ""} for i, p in enumerate(reader.pages, start=1)]

# ---------- 4. Clean headers, footers and page numbers ----------
counts = Counter()
for p in pages:
    counts.update(set(l.strip() for l in p["text"].split("\n") if l.strip()))
repeated = {l for l, n in counts.items() if n >= len(pages) * 0.3}

def clean(text):
    lines = [l for l in text.split("\n")
             if l.strip() not in repeated and not re.fullmatch(r"\d+", l.strip())]
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", "\n".join(lines))).strip()

pages = [{"page": p["page"], "text": clean(p["text"])} for p in pages]

# ---------- 5. Chunk (1,000 characters, 200 overlap, within each page) ----------
chunks = []
for p in pages:
    start = 0
    while start < len(p["text"]):
        chunks.append({"page": p["page"], "text": p["text"][start:start + 1000]})
        start += 800

# ---------- 6. Embed (free local model, 384 numbers per chunk) ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_vectors = embedder.encode([c["text"] for c in chunks])

# ---------- 7. Hybrid search: meaning + exact keywords ----------
KEYWORD_BONUS = 0.15   # pages containing a key word from the question move up
STOP = {"should", "report", "reported", "reporting", "about", "which", "there",
        "their", "document", "would", "could", "where", "what", "these"}

def search(question, top_k=8):    # 8, not 5: the keyword bonus pushed the LTV page (37) out of the top 5
    scores = util.cos_sim(embedder.encode(question), chunk_vectors)[0]
    words = [w.rstrip("s") for w in re.findall(r"[a-z\-]+", question.lower())
             if len(w) > 4 and w not in STOP]          # "possessions" -> "possession"
    for i, c in enumerate(chunks):
        if any(w in c["text"].lower() for w in words):
            scores[i] += KEYWORD_BONUS
    top = scores.argsort(descending=True)[:top_k]
    return [{"page": chunks[int(i)]["page"], "text": chunks[int(i)]["text"],
             "score": float(scores[int(i)])} for i in top]

# ---------- 8. Grounding rules + fixed answer layout ----------
instructions = """You are a regulatory reporting assistant.
Answer ONLY using the passages provided. Do not use outside knowledge.
If the answer is not in the passages, reply exactly: "I couldn't find this in the provided document."
Otherwise reply in this layout (same every time, so readers know where to look):
**Answer:** one or two plain sentences.
**Where:** the form, section and row if stated, else "Not stated".
**Rule (quote):** the exact sentence from the passage, in quotes.
**Conditions:** any exclusions or conditions, else "None stated".
**Source:** the page numbers, like (Page 12)."""

# ---------- 9. Ask the LLM, trying each provider in order ----------
def ask_llm(contents):
    for provider in PROVIDERS:
        if provider not in clients:
            continue
        try:
            if provider == "gemini":
                r = clients["gemini"].models.generate_content(
                        model=MODELS["gemini"], contents=contents,
                        config=types.GenerateContentConfig(system_instruction=instructions,
                                                           temperature=0))
                return r.text, provider
            else:   # groq and openai share the OpenAI format
                r = clients[provider].chat.completions.create(
                        model=MODELS[provider], temperature=0, messages=[
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": contents}])
                return r.choices[0].message.content, provider
        except Exception as e:
            print(f"⚠️ {provider} failed ({str(e)[:70]}...) → trying next")
    return None, None

# ---------- 10. Spelling fix (only to words that exist in the PDF) ----------
MIN_SCORE = 0.30   # pass mark: pages scoring below this are not sent to the LLM
vocab = set(re.findall(r"[a-z\-]+", " ".join(c["text"] for c in chunks).lower()))

def fix_spelling(question):
    # "morgage arears" -> "mortgage arrears"
    fixed = []
    for w in question.lower().split():
        core = re.sub(r"[^a-z\-]", "", w)
        if len(core) > 3 and core not in vocab:
            match = difflib.get_close_matches(core, vocab, n=1, cutoff=0.8)
            if match:
                w = w.replace(core, match[0])
        fixed.append(w)
    return " ".join(fixed)

# ---------- 11. RAG answer for the screen (threshold + sources) ----------
def ui_answer(question):
    if not question.strip():
        return "Please type a question."
    corrected = fix_spelling(question)
    note = f"🔤 Searched for: {corrected}\n\n" if corrected != question.lower().strip() else ""
    question = corrected
    results = search(question)
    good = [r for r in results if r["score"] >= MIN_SCORE]
    if not good:
        return note + (f"🤔 Not confident: the best match is only {results[0]['score']:.2f} "
                       f"(below {MIN_SCORE}).\nI couldn't find this in the provided document.")
    context = "".join(f"\n[Page {r['page']}]\n{r['text']}\n" for r in good)
    text, provider = ask_llm(f"PASSAGES:\n{context}\n\nQUESTION: {question}")
    if not text:
        return "❌ No AI provider is available right now. Please try again later."
    sources = "\n".join(f"- Page {r['page']}  (similarity {r['score']:.2f})" for r in good)
    return f"{note}{text}\n\n**Sources**\n{sources}\n\n*Answered by: {provider}*"

# ---------- 12. Web screen ----------
demo = gr.Interface(
    fn=ui_answer,
    inputs=gr.Textbox(label="Your question", lines=2,
                      placeholder="e.g. How should buy-to-let mortgages be reported?"),
    outputs=gr.Markdown(label="Answer (with page sources)"),
    title="Regulatory Reporting Q&A Assistant",
    description="Answers only from the Bank of England MLAR guidance (public document), with page citations.",
    examples=["How should buy-to-let mortgages be reported?",
              "How is loan-to-value calculated?",
              "What is the best recipe for chocolate cake?"],
)

print(f"Ready ✅  providers: {list(clients)} | {len(pages)} pages | {len(chunks)} chunks")

if __name__ == "__main__":     # "python app.py" starts the screen; "import app" (tests) does not
    demo.launch()
