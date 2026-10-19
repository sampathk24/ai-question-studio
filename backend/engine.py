"""Question evaluation, answer evaluation, upgrading, generation."""
import re, json, random, pathlib
import joblib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ml, llm

DATA = pathlib.Path(__file__).parent / "data"
LEVELS, DIFFS = ml.LEVELS, ml.DIFFS

_idx = joblib.load(DATA / "textbook_index.joblib")
_chunks = json.load(open(DATA / "textbook_chunks.json"))
_df = ml.load_df()
_pred = ml.get_predictor()
_qvec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit(_df.question)
_qX = _qvec.transform(_df.question)

_fw = _df.question.str.split().str[0].str.lower().value_counts()
ALLOWED_START = set(_fw[_fw >= 2].index)          # Explain, Define, Compare, Design ...
BANNED_START = {"what", "which", "how", "why", "when", "where", "who", "in", "during", "given", "if", "is", "are", "does", "do", "can", "consider", "suppose", "for", "using", "after", "while"}

VAGUE = {"thing", "things", "stuff", "something", "etc", "somehow", "good", "bad", "nice", "briefly", "anything"}
STOP = set("the a an of in on for to and or is are be with by as at from that this it its into using use makes make define explain describe state list compare discuss what how why given".split())

def _terms(text):
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", text.lower()) if w not in STOP}


# ----------------------------------------------------------------- retrieval
def _stem(w):
    return w[:6]


def textbook(query, k=3):
    """Hybrid retrieval: tf-idf cosine, re-ranked by query-term coverage; exercise/bibliography pages are down-weighted."""
    query = re.sub(r"^\s*[A-Za-z]+\s+(what|how|why|the)?\s*", "", query)
    query = re.sub(r"\b(makes?|makes|explain|describe|define|determine|given)\b", " ", query, flags=re.I)
    s = cosine_similarity(_idx["vec"].transform([query]), _idx["X"])[0]
    cand = s.argsort()[::-1][:40]
    idf = dict(zip(_idx["vec"].get_feature_names_out(), _idx["vec"].idf_))
    q_w = {w: idf.get(w, 4.0) for w in _terms(query) if w not in STOP}
    q_stems = {_stem(w): v for w, v in q_w.items()}
    scored = []
    for i in cand:
        low = _chunks[i]["text"].lower()
        stems = {_stem(w) for w in re.findall(r"[a-z]{4,}", low)}
        cov = sum(v for k, v in q_stems.items() if k in stems) / max(sum(q_stems.values()), 1e-9)
        exercise = len(re.findall(r"\b\d+\.\d+\s+(?:Compare|Prove|Show|Give|Write|Suppose|Consider|Define|Explain|Implement)\b", _chunks[i]["text"]))
        scored.append((float(s[i]) + 0.45 * cov - 0.04 * min(exercise, 3), i))
    scored.sort(reverse=True)
    top = [i for _, i in scored[:k]]
    return [{"page": _chunks[i]["page"], "score": round(sc, 3), "text": _chunks[i]["text"][:700]}
            for sc, i in scored[:k]], scored[0][0]


def similar_questions(q, level=None, k=4):
    s = cosine_similarity(_qvec.transform([q]), _qX)[0]
    if level:
        s = s * (_df.level.values == level)
    top = s.argsort()[::-1][:k]
    return [{"question": _df.question[i], "level": _df.level[i], "difficulty": _df.difficulty[i],
             "similarity": round(float(s[i]), 3)} for i in top if s[i] > 0]


# ----------------------------------------------------------------- predict
def predict(q):
    p = _pred.predict(q)
    i = LEVELS.index(p["level"])
    p["next_level"] = LEVELS[i + 1] if i < 5 else None
    p["prev_level"] = LEVELS[i - 1] if i > 0 else None
    return p


# ----------------------------------------------------------------- feature 1a: evaluate the QUESTION
def evaluate_question(q):
    q = q.strip()
    pred = predict(q)
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", q)
    n = len(words)
    refs, cover = textbook(q, 3)
    notes, scores = [], {}

    # clarity: sensible length, starts with a directive verb, ends properly, not vague
    c = 100
    first = words[0].lower() if words else ""
    verbs = set(_df.verb.str.lower()) | set(_df.question.str.split().str[0].str.lower())
    if first not in verbs:
        c -= 25; notes.append("Start with a clear action verb (e.g. Explain, Compare, Design).")
    if n < 7:
        c -= 25; notes.append("Too short — add the context or constraint the answer should address.")
    if n > 45:
        c -= 20; notes.append("Very long — consider splitting into two questions.")
    if [w for w in words if w.lower() in VAGUE]:
        c -= 20; notes.append("Replace vague words (" + ", ".join(sorted({w.lower() for w in words if w.lower() in VAGUE})) + ") with precise terms.")
    if not re.search(r"[.?]$", q):
        c -= 5; notes.append("End the question with a full stop or question mark.")
    scores["Clarity"] = max(c, 0)

    # focus: one task per question
    tasks = len(re.findall(r"\b(and|then|also)\s+(explain|define|compare|design|analy[sz]e|evaluate|calculate|describe|list|discuss|justify)\b", q.lower()))
    f = 100 - 30 * tasks
    if tasks:
        notes.append(f"Contains {tasks + 1} separate tasks — they will be marked unevenly; consider splitting.")
    scores["Focus"] = max(f, 0)

    # subject relevance: does the textbook actually cover this?
    scores["Syllabus relevance"] = int(min(100, cover / 0.25 * 100))
    if cover < 0.08:
        notes.append("Little match with the textbook — check the topic or terminology.")

    # specificity: domain terms (words appearing in the textbook vocabulary)
    vocab = _idx["vec"].vocabulary_
    content = [w.lower() for w in words if w.lower() not in STOP]
    dom = [w for w in content if w in vocab]
    scores["Specificity"] = int(min(100, 100 * len(dom) / max(len(content), 1) * 1.15 + (10 if n >= 10 else 0)))

    # cognitive demand: where it sits on Bloom's scale (informational, not good/bad)
    scores["Cognitive demand"] = int((LEVELS.index(pred["level"]) + 1) / 6 * 100)

    overall = round(0.30 * scores["Clarity"] + 0.15 * scores["Focus"] + 0.25 * scores["Syllabus relevance"]
                    + 0.30 * scores["Specificity"])
    grade = "Excellent" if overall >= 80 else "Good" if overall >= 65 else "Needs work" if overall >= 45 else "Weak"
    out = {"overall": overall, "grade": grade, "scores": scores, "suggestions": notes,
           "prediction": pred, "textbook_refs": refs, "source": "rules+ml"}

    ai = llm.ask_json(f"""You are an exam-setter reviewing a university AI-course question.
Question: {q}
Predicted Bloom level: {pred['level']}. Return JSON: {{"verdict": "<2 sentence critique>", "improved": "<a better-worded version at the same Bloom level>"}}""")
    if ai:
        out["ai_review"], out["source"] = ai, "rules+ml+llm"
    return out


# ----------------------------------------------------------------- feature 2: UPGRADE / DEGRADE after prediction
TEMPLATES = {
    "Remember": "Define {t}.",
    "Understand": "Explain {t} in your own words.",
    "Apply": "Apply {t} to a small worked example, showing each step.",
    "Analyze": "Analyze {t} by breaking it into its components and examining how they interact.",
    "Evaluate": "Evaluate {t}, and justify your judgement with criteria and trade-offs.",
    "Create": "Design a solution that builds on {t}, and specify its inputs, outputs and assumptions.",
}
_CUT = re.compile(r",|\s+(?:in terms of|based on|using|with respect to|to determine|to solve|and (?:justify|explain|describe|discuss|then)|"
                  r"when|during|so that|such that|which|that|by considering)\b", re.I)


def topic_phrase(q):
    """The subject of a question, e.g. 'Compare BFS and DFS in terms of memory.' -> 'BFS and DFS'."""
    t = re.sub(r"^\s*[A-Za-z]+\s+", "", q.strip()).rstrip(".?! ")
    t = re.sub(r"^(?:what|how|why|whether|which)\s+(?:is|are|does|do|makes?|can)?\s*", "", t, flags=re.I)
    t = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.I)
    m = _CUT.search(t)
    if m and m.start() >= 3:
        t = t[:m.start()]
    t = " ".join(t.split()[:12]).strip(" ,;:")
    return t or "the topic"


def rewrite(q, target=None, direction="up"):
    """Rewrite a question at a higher (direction='up') or lower ('down') Bloom level."""
    pred = predict(q)
    cur = pred["level"]
    i = LEVELS.index(cur)
    if target is None:
        target = pred["next_level"] if direction == "up" else pred["prev_level"]
    base = {"prediction": pred, "direction": direction, "examples": []}
    if target is None or target not in LEVELS:
        edge = "top of Bloom's taxonomy (Create)" if direction == "up" else "bottom of Bloom's taxonomy (Remember)"
        return {**base, "target_level": cur, "upgraded": None, "message": f"This question is already at the {edge}.",
                "examples": similar_questions(q, cur, 3)}
    ti = LEVELS.index(target)
    if (direction == "up" and ti <= i) or (direction == "down" and ti >= i):
        word = "higher" if direction == "up" else "lower"
        return {**base, "target_level": cur, "upgraded": None, "message": f"The target must be {word} than the current level ({cur}).",
                "examples": similar_questions(q, cur, 3)}
    rule = TEMPLATES[target].format(t=topic_phrase(q))
    out = {**base, "target_level": target, "upgraded": rule, "method": "template",
           "examples": similar_questions(q, target, 4),
           "verbs_for_target": sorted(_df[_df.level == target].verb.value_counts().head(8).index.tolist()),
           "verified_level": predict(rule)["level"]}
    word = "harder (higher-order thinking)" if direction == "up" else "easier (more basic recall/understanding)"
    ai = llm.ask_json(f"""Rewrite this exam question so it tests Bloom's '{target}' level (currently '{cur}'), i.e. make it {word}.
Keep the same topic. Write ONE clear, self-contained imperative question starting with an action verb suited to '{target}' (e.g. {', '.join(out['verbs_for_target'][:5])}).
Question: {q}
Return JSON: {{"upgraded": "...", "why": "<one sentence on what changed cognitively>"}}""")
    if ai and ai.get("upgraded"):
        out.update({"upgraded": ai["upgraded"], "why": ai.get("why", ""), "method": "llm", "template_fallback": rule,
                    "verified_level": predict(ai["upgraded"])["level"]})
    return out


def upgrade(q, target=None):
    return rewrite(q, target, "up")


def degrade(q, target=None):
    return rewrite(q, target, "down")


# ----------------------------------------------------------------- generate
def _dataset_questions(topic, level, difficulty, n, with_info=False, exclude=None):
    """Best dataset matches. If the exact level+difficulty filter gives fewer than n, relax difficulty, then level."""
    s = cosine_similarity(_qvec.transform([topic]), _qX)[0]
    if exclude:
        s = np.where(_df.question.isin(exclude).values, -1, s)          # do not repeat questions already shown
    lv, df_ = _df.level.values, _df.difficulty.values
    stages = [(np.ones(len(_df), bool) & ((lv == level) if level in LEVELS else True) & ((df_ == difficulty) if difficulty in DIFFS else True), "exact")]
    if level in LEVELS and difficulty in DIFFS:
        stages.append((lv == level, "difficulty relaxed"))
    if level in LEVELS:
        stages.append((np.ones(len(_df), bool), "level relaxed"))
    picked, used, exact = [], set(), 0
    for mask, tag in stages:
        sc = np.where(mask, s, -1)
        for i in sc.argsort()[::-1][:n * 3]:
            if sc[i] > 0 and i not in used and len(picked) < n:
                used.add(i)
                picked.append({"question": _df.question[i], "level": _df.level[i], "difficulty": _df.difficulty[i]})
                exact += tag == "exact"
        if len(picked) >= n:
            break
    return (picked, exact) if with_info else picked


def _valid_generated(item, seen, min_words=6):
    q = str(item.get("question", "")).strip()
    words = q.split()
    if not (min_words <= len(words) <= 40) or q.lower() in seen:
        return None
    fw = re.sub(r"[^a-z]", "", words[0].lower())
    if fw in BANNED_START or fw not in ALLOWED_START:
        return None
    if q.count("?") > 1 or not re.search(r"[.?]$", q):
        q = q.rstrip() + ("" if re.search(r"[.?]$", q) else ".")
    return q


def generate(topic, level=None, difficulty=None, n=5, exclude=None):
    level = level if level in LEVELS else None
    difficulty = difficulty if difficulty in DIFFS else None
    ds, exact = _dataset_questions(topic, level, difficulty, n, with_info=True, exclude=exclude)
    if not llm.available():
        note = ""
        if ds and exact < len(ds) and (level or difficulty):
            note = f"Only {exact} question(s) matched exactly ({level or 'any level'}, {difficulty or 'any difficulty'}); the rest are the closest matches. Add an API key to generate new ones."
        return {"questions": ds, "source": "dataset", "note": note}

    pool = _df[_df.level == level] if level else _df
    verbs = pool.question.str.split().str[0].str.lower().value_counts().head(14).index.tolist()
    shots = _dataset_questions(topic, level, difficulty, 6) or pool.sample(6, random_state=1)[["question", "level", "difficulty"]].to_dict("records")
    examples = "\n".join(f"- {x['question']}" for x in shots)
    good, out_items, seen, rejected = [], [], set(), 0

    for attempt in range(2):
        need = n - len(good)
        if need <= 0:
            break
        ai = llm.ask_json(f"""You write exam questions for a university Artificial Intelligence course (textbook: Russell & Norvig, AIMA 3e).
Write {need + 3} NEW questions on the topic: "{topic}".
Bloom level: {level or 'mixed levels'}. Difficulty: {difficulty or 'mixed'}.

STRICT STYLE RULES (copy the style of the examples):
1. Every question is ONE imperative sentence that STARTS with an action verb. Allowed first words: {', '.join(v.capitalize() for v in verbs)}.
2. NEVER start with What, Which, How, Why, When, In, During, Given, Consider, If, or any other question word / clause.
3. One task per question, no sub-parts, no multiple choice, no yes/no questions.
4. Under 30 words, specific to the topic, answerable from the textbook.

Style examples from the real question bank:
{examples}
{'Avoid repeating these already used: ' + '; '.join(list(good) + list(exclude or [])[:20]) if (good or exclude) else ''}
Return JSON: {{"questions": [{{"question": "...", "level": "...", "difficulty": "Easy|Medium|Hard"}}]}}""")
        if not ai:
            break
        cands = []
        for item in ai.get("questions", []):
            q = _valid_generated(item, seen)
            if not q:
                rejected += 1
                continue
            p = _pred.predict(q)
            cands.append((q, p, item))
        # prefer the requested level as judged by OUR trained model, then fill with the rest
        cands.sort(key=lambda c: (level is not None and c[1]["level"] != level))
        for q, p, item in cands:
            if len(good) >= n:
                break
            if level and p["level"] != level and len(cands) > need:
                rejected += 1
                continue
            seen.add(q.lower())
            d = difficulty or (item.get("difficulty") if item.get("difficulty") in DIFFS else p["difficulty"])
            good.append(q)
            out_items.append({"question": q, "level": p["level"], "difficulty": d})
    out = out_items[:n]
    if not out:
        return {"questions": ds, "source": "dataset", "note": "AI generation failed; showing dataset questions. " + (llm.last_error or "")}
    src = "llm"
    if len(out) < n:
        have = {o["question"].lower() for o in out}
        out += [d for d in ds if d["question"].lower() not in have][: n - len(out)]
        src = "llm+dataset"
    return {"questions": out, "source": src, "filtered_out": rejected}
