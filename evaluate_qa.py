# ===== Evaluation (test set) =====
# Run:  python evaluate_qa.py   (loads app.py without starting the screen)
# "in" = the document should answer it; "out" = it should say "not found".

from app import search, ask_llm, MIN_SCORE

tests = [
    ("How should buy-to-let mortgages be reported?",          "in"),
    ("How are mortgage arrears reported?",                    "in"),
    ("What is a lifetime mortgage?",                          "in"),
    ("How is loan-to-value calculated?",                      "in"),
    ("What are gross advances?",                              "in"),
    ("How should possessions be reported?",                   "in"),
    ("What does the document say about let-to-buy?",          "out"),
    ("What is the Bank of England's current interest rate?",  "out"),
    ("What is the best recipe for chocolate cake?",           "out"),
    ("How do I open a cryptocurrency wallet?",                "out"),
]

NOT_FOUND = "couldn't find"

def run_one(question):
    results = search(question)
    good = [r for r in results if r["score"] >= MIN_SCORE]
    if not good:
        return "Not found (threshold)", [], results[0]["score"]
    context = "".join(f"\n[Page {r['page']}]\n{r['text']}\n" for r in good)
    text, provider = ask_llm(f"PASSAGES:\n{context}\n\nQUESTION: {question}")
    return (text or "No provider"), [r["page"] for r in good], results[0]["score"]

passed = 0
for n, (question, expected) in enumerate(tests, start=1):
    text, pages, best = run_one(question)
    said_not_found = NOT_FOUND in text.lower() or text.startswith("Not found")
    ok = (expected == "out") == said_not_found
    passed += ok
    print(f"{'✅' if ok else '❌'} Q{n} [{expected}] best score {best:.2f} | pages {pages}")
    print(f"   {question}")
    print(f"   → {text[:200].strip()}...\n")

print(f"Score: {passed} / {len(tests)}")
