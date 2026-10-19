"""Dialogue manager: a conversation, not a command line.

* Understands a request in plain English and runs the right function.
* Asks for anything that is missing (one question at a time, with tap-to-answer options).
* Remembers what you were doing, so "more", "harder", "change level to Apply" just work.
* Guides you after an upload (question paper -> Bloom analysis; study material -> question format).

The server is stateless: the browser sends a small `context` dict with every message.
"""
import re

import engine, llm, docs, docgen, paper

LEVELS, DIFFS = engine.LEVELS, engine.DIFFS
LV = r"remember|understand|apply|analy[sz]e|evaluate|create"
LV_MAP = {"remember": "Remember", "understand": "Understand", "apply": "Apply", "analyze": "Analyze", "analyse": "Analyze", "evaluate": "Evaluate", "create": "Create"}
DIFF = r"easy|simple|basic|beginner|medium|moderate|intermediate|hard|difficult|tough|challenging"
DIFF_MAP = {"easy": "Easy", "simple": "Easy", "basic": "Easy", "beginner": "Easy", "medium": "Medium", "moderate": "Medium", "intermediate": "Medium",
            "hard": "Hard", "difficult": "Hard", "tough": "Hard", "challenging": "Hard"}
NUMS = dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10, twelve=12, fifteen=15, twenty=20)
CHATTY = {"i", "my", "please", "can", "could", "hello", "hi", "hey", "thanks", "thank", "ok", "okay", "yes", "no", "what's", "who", "make", "give", "show"}
TOPICS = ["Search algorithms", "Minimax and games", "Constraint satisfaction", "Bayesian networks", "Neural networks"]
FAIL = object()


# ----------------------------------------------------------------- small helpers
def chip(label, text, icon=None):
    return {"label": label, "text": text, "icon": icon}


def resp(reply, intent, blocks=None, chips=None, ctx=None):
    return {"reply": reply, "intent": intent, "blocks": blocks or [], "suggestions": chips or [], "context": ctx or {}}


def level_chips():
    return [chip(l, l) for l in LEVELS]


def find_level(text, before_questions=False):
    pats = [rf"\b({LV})[\s-]*level\b", rf"\blevel\s*(?:of|:|=|is)?\s*({LV})\b", rf"\bbloom(?:'?s)?\s+(?:level\s+)?({LV})\b",
            rf"\b(?:at|to|into|for)\s+(?:the\s+)?({LV})\b(?!\s+(?:this|that|the|a|an|my|it))"]
    if before_questions:
        pats.insert(0, rf"\b({LV})\s+(?:type\s+|based\s+)?questions?\b")
    for p in pats:
        m = re.search(p, text, re.I)
        if m:
            return LV_MAP[m.group(1).lower()]
    return None


def find_levels(text):
    """All Bloom levels named in a request, e.g. 'at Apply and Analyze level'. [] means 'not specified'."""
    low = text.lower()
    if re.search(r"higher[- ]order", low):
        return LEVELS[2:]
    if re.search(r"lower[- ]order", low):
        return LEVELS[:2]
    lst = rf"(?:{LV})(?:\s*(?:,|and|&|/|or)\s*(?:{LV}))*"
    m = (re.search(rf"\b({lst})[\s-]*(?:levels?|bloom)", low) or re.search(rf"\blevels?\s*(?:of|:|=|are|is)?\s*({lst})", low)
         or re.search(rf"\b(?:at|for)\s+(?:the\s+)?({lst})\b(?!\s+(?:this|that|the|a|an|my|it))", low))
    return list(dict.fromkeys(LV_MAP[x] for x in re.findall(LV, m.group(1)))) if m else []


def find_diff(text):
    for p in (rf"\b({DIFF})\b(?=\s+(?:level|difficulty|questions?|ones?))", rf"\bdifficulty\s*(?:level)?\s*(?:of|:|=|is)?\s*({DIFF})\b"):
        m = re.search(p, text, re.I)
        if m:
            return DIFF_MAP[m.group(1).lower()]
    return None


def find_n(text, default=None, cap=25):
    m = re.search(r"\b(\d{1,2})\b(?=\s+(?:[\w-]+\s+){0,4}?(?:questions?|qs|mcqs?)\b)", text, re.I) or \
        re.search(rf"\b({'|'.join(NUMS)})\b(?=\s+(?:[\w-]+\s+){{0,4}}?(?:questions?|mcqs?)\b)", text, re.I)
    if not m:
        return default
    v = m.group(1)
    return max(1, min(cap, int(v) if v.isdigit() else NUMS[v.lower()]))


def find_qtype(text):
    low = text.lower()
    if re.search(r"\b(mcqs?|multiple[- ]choice|objective)\b", low):
        return "MCQ"
    if re.search(r"\b(long|essay|descriptive|detailed)\b[\s-]*(answer|type|questions?)", low):
        return "Long answer"
    if re.search(r"\b(short|brief)\b[\s-]*(answer|type|questions?)", low):
        return "Short answer"
    if re.search(r"\bmixed\s+(types?|formats?)\b|\bmix of\b", low):
        return "Mixed"
    return None


def find_marks(text, default=None):
    m = re.search(r"(\d{1,2})\s*marks?\b|\bmarks?\s*(?:each|per question)?\s*[:=]\s*(\d{1,2})", text, re.I)
    return int(next(g for g in m.groups() if g)) if m else default


def find_topic(text):
    t = re.sub(rf"\b(?:at|of|for|in|to)?\s*(?:the\s+)?(?:bloom(?:'?s)?\s+)?(?:{LV})[\s-]*level\b", " ", text, flags=re.I)
    t = re.sub(rf"\blevel\s*(?:of|:|=)?\s*(?:{LV})\b", " ", t, flags=re.I)
    t = re.sub(rf"\b(?:{LV})\s+(?:type\s+)?(?=questions?)", " ", t, flags=re.I)
    t = re.sub(rf"\b(?:difficulty\s*(?:level)?\s*(?:of|:|=|is)?\s*)?(?:{DIFF})\b(?=\s+(?:level|difficulty|questions?|ones?))(?:\s+(?:level|difficulty))?", " ", t, flags=re.I)
    m = re.search(r"\b(?:on|about|regarding|covering|concerning|related to|topic(?:\s+of)?|from)\s*[:\-]?\s+(.+)$", t, re.I | re.S)
    if m:
        topic = m.group(1)
    else:
        topic = re.sub(r"\b(please|can you|could you|generate|create|make|give|write|prepare|draft|suggest|produce|i need|i want|need|want|some|few|set of|a|an|the|me|us|for|of|"
                       r"questions?|quiz|question paper|\d+|" + "|".join(NUMS) + r")\b", " ", t, flags=re.I)
    topic = re.sub(r"\s+", " ", topic).strip(" .,:;?!-\"'")
    for _ in range(4):
        topic = re.sub(r"\s+(?:please|thanks|thank you|with|at|of|for|in|and|having|that|are|is|to|by|on|about|using|the|a|an)$", "", topic, flags=re.I)
    return re.sub(r"^(?:the|a|an)\s+", "", topic, flags=re.I).strip(" .,:;?!-")


DOC_WORDS = r"(?:this|the|my|that|these|those|uploaded|attached)?\s*(?:document|doc|file|pdf|ppt|pptx|slides?|notes|material|presentation|chapter|lecture|text)s?"


def find_doc_topic(text):
    m = re.search(r"\b(?:on|about|covering|regarding|focus(?:ing)? on)\s+(.+)$", text, re.I)
    if not m:
        return None
    t = find_topic("questions on " + m.group(1))
    return None if (not t or re.fullmatch(DOC_WORDS, t, re.I) or len(t.split()) > 10) else t


def split_cmd_body(text):
    m = re.search(r'["“](.{15,}?)["”]', text, re.S) or re.search(r"(?<!\w)'([^']{15,})'(?!\w)", text, re.S)
    if m:
        return text.replace(m.group(0), " "), m.group(1).strip()
    m = re.match(r"^(.{0,120}?)[:\n]\s*(.+)$", text, re.S)
    if m and len(m.group(2).split()) >= 3:
        return m.group(1), m.group(2).strip()
    m = re.match(r"^\s*(?:please\s+)?(?:upgrade|degrade|downgrade|evaluate|review|check|rate|assess|critique|score|predict|classify)\s+"
                 r"(?:(?:the|this|that|my)\s+)?(?:question\s+)?(.+)$", text, re.I | re.S)
    if m:
        rest = m.group(1).strip()
        if len(rest.split()) >= 5 and not re.match(r"(it|this|that|to|into|at|one|again|further|more)\b", rest, re.I):
            return text[:text.rfind(rest)], rest
    return text, None


def looks_like_question(t):
    words = t.split()
    if len(words) < 5:
        return False
    fw = re.sub(r"[^a-z']", "", words[0].lower())
    if fw in CHATTY:
        return False
    return t.rstrip().endswith(("?", ".")) or fw in engine.ALLOWED_START


def shift_levels(levels, d):
    if not levels:
        return LEVELS[2:] if d > 0 else LEVELS[:3]
    return list(dict.fromkeys(LEVELS[max(0, min(5, LEVELS.index(l) + d))] for l in levels))


def shift_diff(diff, d):
    i = DIFFS.index(diff) if diff in DIFFS else 1
    return DIFFS[max(0, min(2, i + d))]


# ----------------------------------------------------------------- question analysis, upgrade, degrade
def after_question_chips(q, level):
    c = []
    if level != "Create":
        c.append(chip("Upgrade it", f"/upgrade {q}", "up"))
    if level != "Remember":
        c.append(chip("Degrade it", f"/degrade {q}", "down"))
    return c + [chip("Generate similar", f"/similar {q}", "spark")]


def do_analyze(q, ctx):
    e = engine.evaluate_question(q)
    p = e["prediction"]
    ctx = {**ctx, "question": q, "last": "question"}
    lvl = p["level"]
    ask = (" Would you like me to **upgrade** it to a higher level?" if lvl == "Remember" else
           " Would you like me to **degrade** it to an easier level?" if lvl == "Create" else
           " Would you like me to **upgrade** it (harder) or **degrade** it (easier)?")
    return resp(f"This looks like a **{lvl}** question ({p['difficulty']}, {round(p['level_confidence'] * 100)}% sure). "
                f"Quality score: **{e['overall']}/100 ({e['grade']})**.{ask}", "analyze", [{"type": "evaluation", "data": e}],
                after_question_chips(q, lvl), ctx)


def do_predict(q, ctx):
    p = engine.predict(q)
    ctx = {**ctx, "question": q, "last": "question"}
    return resp(f"This looks like a **{p['level']}** question ({p['difficulty']}, {round(p['level_confidence'] * 100)}% sure).", "predict",
                [{"type": "prediction", "data": p}], after_question_chips(q, p["level"]), ctx)


def do_rewrite(q, target, direction, ctx):
    u = engine.rewrite(q, target, direction)
    word = "Upgraded" if direction == "up" else "Degraded"
    if not u.get("upgraded"):
        return resp(u.get("message", "I couldn't change that question."), "rewrite", [{"type": "upgrade", "data": u}],
                    after_question_chips(q, u["prediction"]["level"]), {**ctx, "question": q, "last": "question"})
    new_q, tl = u["upgraded"], u["target_level"]
    chips = []
    if direction == "up":
        chips += [chip("Upgrade again", f"/upgrade {new_q}", "up")] if tl != "Create" else []
        chips += [chip("Degrade it back", f"/degrade {new_q}", "down")]
    else:
        chips += [chip("Degrade again", f"/degrade {new_q}", "down")] if tl != "Remember" else []
        chips += [chip("Upgrade it back", f"/upgrade {new_q}", "up")]
    chips.append(chip("Evaluate the new question", f"/evaluate {new_q}", "chart"))
    return resp(f"Predicted **{u['prediction']['level']}**. {word} to **{tl}**.", "rewrite", [{"type": "upgrade", "data": u}],
                chips, {**ctx, "question": new_q, "last": "question"})


# ----------------------------------------------------------------- generation
def run_gen(p, ctx):
    """Questions from the textbook question bank."""
    n, levels, diff = p["n"], p.get("levels") or [], p.get("difficulty")
    exclude = set(p.get("shown", []))
    if len(levels) > 1:
        plan = docgen.distribute(n, levels)
        parts = [engine.generate(p["topic"], l, diff, plan.count(l), exclude) for l in levels if plan.count(l)]
        r = {"questions": [q for x in parts for q in x["questions"]], "source": parts[0]["source"],
             "note": " ".join(x.get("note", "") for x in parts if x.get("note"))[:300]}
    else:
        r = engine.generate(p["topic"], levels[0] if levels else None, diff, n, exclude)
    qs = r["questions"]
    if not qs:
        return resp(f"I couldn't find questions on “{p['topic']}”. Try different wording, e.g. “search”, “planning” or “neural networks”.", "generate",
                    ctx={**ctx, "pending": None})
    shown = (list(exclude) + [q["question"] for q in qs])[-40:]
    last = {"kind": "gen", **{k: p.get(k) for k in ("topic", "levels", "difficulty", "n")}, "shown": shown}
    bits = " · ".join(x for x in [", ".join(levels), diff] if x) or "mixed levels"
    return resp(f"Here are **{len(qs)} questions** on **{p['topic']}** ({bits}). Use the buttons on a question to upgrade or degrade it.", "generate",
                [{"type": "questions", "data": r}],
                [chip("More questions", "more", "spark"), chip("Make them harder", "harder", "up"), chip("Make them easier", "easier", "down")],
                {**ctx, "pending": None, "last": "gen", "last_gen": last, "topic": p["topic"]})


def run_doc(p, ctx):
    """Questions written from the uploaded document, in the requested format."""
    doc = docs.get((ctx.get("doc") or {}).get("id", ""))
    if not doc:
        return resp("I no longer have that document in memory (the server may have restarted). Please attach it again.", "doc_missing", ctx={**ctx, "doc": None, "pending": None})
    spec = {k: p.get(k) for k in ("n", "levels", "difficulty", "qtype", "marks", "topic", "seed")}
    spec["n"], spec["qtype"], spec["seed"], spec["levels"] = spec["n"] or 10, spec["qtype"] or "Short answer", spec["seed"] or 0, spec["levels"] or None
    r = docgen.generate(doc, spec)
    if not r["questions"]:
        return resp("I couldn't write questions from this document. " + " ".join(r["notes"]), "docgen", ctx={**ctx, "pending": None})
    last = {"kind": "doc", **{k: r["spec"].get(k) for k in ("n", "levels", "difficulty", "qtype", "marks", "topic", "seed")}}
    return resp(f"Here are **{len(r['questions'])} questions** from **{doc['name']}** ({docgen.spec_text(r['spec'])}). Total marks: **{r['total_marks']}**. "
                "Download them as a Word paper or Excel sheet from the card, or upgrade/degrade any question.", "docgen",
                [{"type": "doc_questions", "data": r}],
                [chip("More questions", "more", "spark"), chip("Make them harder", "harder", "up"), chip("Make them easier", "easier", "down"), chip("Change format", "/customise", "tools")],
                {**ctx, "pending": None, "last": "doc", "last_gen": last})


# ----------------------------------------------------------------- slot filling (asking follow-up questions)
SLOT_ORDER = {"gen": ["topic", "levels", "difficulty", "n"], "doc": ["n", "levels", "qtype", "difficulty"]}


def slot_prompt(flow, slot):
    gen = flow == "gen"
    if slot == "topic":
        return "Which **topic** should the questions cover?", [chip(t, t) for t in TOPICS]
    if slot == "levels":
        extra = [] if gen else [chip("Higher-order", "Apply, Analyze, Evaluate, Create"), chip("Lower-order", "Remember, Understand")]
        return ("Which **Bloom level** would you like? Pick one, or type several like “Apply and Analyze”.",
                [chip("Any level", "Any")] + level_chips() + extra)
    if slot == "difficulty":
        return "What **difficulty**?", [chip("Any", "Any"), chip("Easy", "Easy"), chip("Medium", "Medium"), chip("Hard", "Hard")]
    if slot == "n":
        return "**How many** questions?", [chip(str(x), str(x)) for x in ((3, 5, 8, 10) if gen else (5, 10, 15, 20))]
    if slot == "qtype":
        return "What **type** of questions?", [chip(t, t) for t in ("Short answer", "Long answer", "MCQ", "Mixed")]
    return "", []


def parse_slot(slot, text):
    t, low = text.strip(), text.strip().lower()
    skip = re.fullmatch(r"(any|skip|none|no preference|default|mixed|balanced|all|anything|whatever|doesn'?t matter)(\s+(level|levels|difficulty|type))?", low)
    if slot == "topic":
        if re.match(r"^(upgrade|degrade|downgrade|evaluate|predict|/)", low) or len(t.split()) > 12 or looks_like_question(t):
            return FAIL
        return re.sub(r"^(?:on|about|the topic(?: of)?|topic[: ]+)\s*", "", t, flags=re.I).strip(" .") or FAIL
    if slot == "levels":
        if skip:
            return []
        lv = find_levels(t) or list(dict.fromkeys(LV_MAP[x] for x in re.findall(rf"\b({LV})\b", low)))
        return lv if lv else FAIL
    if slot == "difficulty":
        if skip:
            return None
        m = re.search(rf"\b({DIFF})\b", low)
        return DIFF_MAP[m.group(1)] if m else FAIL
    if slot == "n":
        m = re.search(r"\b(\d{1,2})\b", low) or re.search(rf"\b({'|'.join(NUMS)})\b", low)
        if not m:
            return FAIL
        v = m.group(1)
        return max(1, min(25, int(v) if v.isdigit() else NUMS[v]))
    if slot == "qtype":
        return "Mixed" if skip else (find_qtype(t) or find_qtype(t + " questions") or FAIL)
    return FAIL


def extract_spec(msg, flow):
    """Whatever the user already told us in one message."""
    got = {}
    n = find_n(msg, None, cap=25 if flow == "doc" else 10)
    if n:
        got["n"] = n
    if flow == "doc":
        lv = find_levels(msg)
    else:
        one = find_level(msg, True)
        lv = [one] if one else []
    if lv or re.search(r"\b(balanced|mixed levels?|any level)\b", msg, re.I):
        got["levels"] = lv
    d = find_diff(msg)
    if d or re.search(r"\bany difficulty\b", msg, re.I):
        got["difficulty"] = d
    if flow == "doc":
        for key, val in (("qtype", find_qtype(msg)), ("marks", find_marks(msg)), ("topic", find_doc_topic(msg))):
            if val:
                got[key] = val
    return got


def continue_flow(flow, slots, ctx, force=False, guided=False):
    """Ask for the next missing slot, or run when we know enough.
    guided=True means we are already in a question-and-answer exchange: keep asking until everything is known."""
    order = SLOT_ORDER[flow]
    missing = [s for s in order if s not in slots]
    extra = len([s for s in order if s in slots and s != "topic"])
    if flow == "gen" and "topic" not in slots:
        nxt = "topic"                                        # a topic is the one thing we cannot guess
    elif force or not missing:
        nxt = None
    elif guided:
        nxt = missing[0]
    elif (flow == "gen" and ("n" in slots or extra >= 2)) or (flow == "doc" and extra >= 2):
        nxt = None                                           # the user gave enough detail up front: use defaults for the rest
    else:
        nxt = missing[0]
    if nxt is None:
        p = {"n": slots.get("n", 5 if flow == "gen" else 10), "levels": slots.get("levels") or [], "difficulty": slots.get("difficulty"),
             "qtype": slots.get("qtype"), "marks": slots.get("marks"), "topic": slots.get("topic"), "seed": slots.get("seed", 0), "shown": slots.get("shown", [])}
        return run_gen(p, ctx) if flow == "gen" else run_doc(p, ctx)
    q, chips = slot_prompt(flow, nxt)
    chips = chips + ([chip("Generate now", "/defaults", "spark")] if nxt != "topic" else [])
    return resp(("Got it. " if slots else "") + q, "ask", chips=chips, ctx={**ctx, "pending": {"flow": flow, "slots": slots, "ask": nxt}})


def answer_pending(msg, ctx):
    pend = ctx["pending"]
    flow, slots, ask = pend["flow"], dict(pend["slots"]), pend.get("ask")
    if re.match(r"^\s*(cancel|stop|never ?mind|exit|forget it)\b", msg.lower()):
        return resp("No problem, cancelled. What would you like to do next?", "cancel", chips=main_chips(ctx), ctx={**ctx, "pending": None})
    if ask == "start":                                       # right after a document upload
        got = extract_spec(msg, "doc")
        if got:
            slots.update(got)
            return continue_flow("doc", slots, ctx)
        return None
    val = parse_slot(ask, msg)
    if val is FAIL:
        return None
    slots[ask] = val
    for k, v in extract_spec(msg, flow).items():            # the user may have answered several things at once
        slots.setdefault(k, v)
    return continue_flow(flow, slots, ctx, guided=True)


# ----------------------------------------------------------------- follow-ups on the last result
def followup(msg, ctx):
    lg, last = ctx.get("last_gen"), ctx.get("last")
    if not lg or last not in ("gen", "doc"):
        return None
    low = msg.lower().strip(" .!?")
    if len(low.split()) > 10:
        return None
    flow = lg["kind"]
    p = {k: lg.get(k) for k in ("topic", "levels", "difficulty", "n", "qtype", "marks", "seed", "shown")}
    p["levels"] = p.get("levels") or []
    m = re.match(r"^(?:give me |show me |generate |create |get )?(?:(\d{1,2})\s+)?(?:more|another|again|different|new|other|next)\b(?:\s+(?:ones|questions|set|batch))?$|"
                 r"^(?:regenerate|retry|redo|refresh)$", low)
    if m:
        if m.group(1):
            p["n"] = int(m.group(1))
    elif re.search(r"\b(harder|tougher|more difficult|more challenging|increase (?:the )?(?:difficulty|level))\b", low):
        p["levels"], p["difficulty"] = shift_levels(p["levels"], +1), shift_diff(p.get("difficulty"), +1)
    elif re.search(r"\b(easier|simpler|less difficult|less challenging|lower (?:the )?(?:difficulty|level))\b", low):
        p["levels"], p["difficulty"] = shift_levels(p["levels"], -1), shift_diff(p.get("difficulty"), -1)
    else:
        got = extract_spec(msg, flow)
        if not got or not re.match(r"^(?:change|switch|set|use|make|try|with|only|instead|now|and|also|same|but|just)\b", low):
            return None
        p.update(got)
    p["seed"] = (p.get("seed") or 0) + 1
    return run_gen(p, ctx) if flow == "gen" else run_doc(p, ctx)


# ----------------------------------------------------------------- uploads
def as_paper(doc, ctx, hint=True):
    ctx = {**ctx, "doc": {"id": doc["id"], "name": doc["name"]}, "pending": None, "last": "paper"}
    try:
        r = paper.analyze_doc(doc)
    except ValueError as e:
        return resp(str(e), "paper_error", ctx=ctx)
    n = r["summary"]["total"]
    scan = f" I read it with OCR ({doc['ocr']['engine']})." if doc.get("ocr") else ""
    chips = [chip("Use as study material instead", f"/mode material {doc['id']}", "file")] if hint else []
    return resp(f"I found **{n} question{'s' if n != 1 else ''}** in **{doc['name']}** and predicted the Bloom level of each.{scan} "
                "Use the arrows on any row to **upgrade** (harder) or **degrade** (easier) that question.", "paper", [{"type": "paper", "data": r}], chips, ctx)


def start_material(doc, ctx, hint=True):
    ctx = {**ctx, "doc": {"id": doc["id"], "name": doc["name"]}, "last": "doc"}
    ov = docgen.overview(doc)
    topics = ", ".join(ov["topics"][:5]) or "no clear headings found"
    where = {"PowerPoint": "slides", "Word document": "sections"}.get(doc["kind"], "pages")
    scan = f" (scanned with {doc['ocr']['engine']})" if doc.get("ocr") else ""
    notes = ("\n" + " ".join(doc["notes"])) if doc["notes"] else ""
    chips = [chip("10 mixed questions", "/defaults", "spark"), chip("Customise", "/customise", "tools")]
    if hint:
        chips.append(chip("Treat as a question paper", f"/mode paper {doc['id']}", "file"))
    return resp(f"I've read **{doc['name']}**{scan}: {ov['units']} {where}, about {ov['words']:,} words. Main topics: **{topics}**.{notes}\n\n"
                "What questions do you need? Tell me the format, for example **“10 short-answer questions at Apply and Analyze level, 5 marks each”**, "
                "or choose an option below.", "doc_ready", chips=chips, ctx={**ctx, "pending": {"flow": "doc", "slots": {}, "ask": "start"}})


def from_upload(doc, ctx):
    ctx = dict(ctx or {})
    return as_paper(doc, ctx) if docs.guess_kind(doc) == "paper" else start_material(doc, ctx)


# ----------------------------------------------------------------- help / routing
def main_chips(ctx):
    return [chip("Generate questions", "Generate questions on search algorithms", "spark"), chip("Predict a question's level", "What level is: Define an intelligent agent.", "chart")]


def help_msg(ctx):
    return resp("I'm your **question-paper assistant** for Bloom's taxonomy. Talk to me in plain English:\n"
                "• **Generate** questions — “Generate 5 hard questions on minimax”\n"
                "• **Predict and rewrite** — paste a question, then ask me to **upgrade** (harder) or **degrade** (easier) it\n"
                "• **Scan a question paper** — attach a PDF, Word file or photo; I'll read it (OCR for scans) and give every question's Bloom level\n"
                "• **Make questions from your notes** — attach slides (PPTX), a Word file or a PDF and tell me the format you need\n"
                "I'll ask follow-up questions whenever I need more detail.", "help", chips=main_chips(ctx), ctx=ctx)


def attach_help(kind, ctx):
    if kind == "paper":
        t = ("To analyse a question paper, **attach it** with the paperclip button (or drag the file onto this window). PDF, Word, PowerPoint and photos or scans "
             "(PNG, JPG) all work; scanned pages are read with OCR. I'll find each question, predict its Bloom level, and let you upgrade or degrade any of them.")
    else:
        t = ("To create questions from your material, **attach your slides, notes or PDF** (PPTX, DOCX, PDF, TXT or a photo of the pages) with the paperclip button. "
             "Then tell me the format you want: number of questions, Bloom level, type (short, long, MCQ), difficulty and marks.")
    return resp(t, "attach_help", ctx=ctx)


def llm_route(msg, ctx):
    if not llm.available():
        return None
    r = llm.ask_json(f"""Classify the user's message for a question-paper assistant and extract fields.
Intents: generate, upgrade, degrade, predict, chat.
Known last question: {ctx.get('question') or 'none'}
User message: {msg}
Return JSON: {{"intent": "...", "question": "<question text or null>", "topic": "<topic or null>", "level": "Remember|Understand|Apply|Analyze|Evaluate|Create|null",
"difficulty": "Easy|Medium|Hard|null", "n": <int or null>, "target_level": "<level or null>", "reply": "<one friendly sentence, only if intent is chat>"}}""")
    return r if isinstance(r, dict) and r.get("intent") else None


def dispatch_llm(r, ctx):
    it, q = r.get("intent"), (r.get("question") or ctx.get("question"))
    tl = r.get("target_level") if r.get("target_level") in LEVELS else None
    if it == "generate" and r.get("topic"):
        lv = [r["level"]] if r.get("level") in LEVELS else []
        return run_gen({"topic": r["topic"], "levels": lv, "difficulty": r.get("difficulty") if r.get("difficulty") in DIFFS else None, "n": int(r.get("n") or 5)}, ctx)
    if it in ("upgrade", "degrade") and q:
        return do_rewrite(q, tl, "up" if it == "upgrade" else "down", ctx)
    if it == "predict" and q:
        return do_predict(q, ctx)
    if it == "chat" and r.get("reply"):
        return resp(r["reply"], "chat", chips=main_chips(ctx), ctx=ctx)
    return None


UP_RE = (r"\bupgrad(?:e|ation)\b|\blevel[- ]?up\b|\bmake (?:it|this|that|the question) (?:harder|tougher|more \w+)\b|\bharder version\b|"
         r"\b(?:raise|increase)\b[^.?]{0,25}\b(?:level|difficulty)\b|\bnext (?:bloom'?s? )?level\b|\bhigher[- ]order\b|^\s*(?:harder|tougher)\b")
DOWN_RE = (r"\bdegrad(?:e|ation)\b|\bdowngrad(?:e|ing)\b|\blevel[- ]?down\b|\bmake (?:it|this|that|the question) (?:easier|simpler|less difficult|more basic)\b|\beasier version\b|"
           r"\b(?:lower|reduce|decrease|simplify)\b[^.?]{0,25}\b(?:level|difficulty)\b|\bprevious (?:bloom'?s? )?level\b|\bdumb(?:ed)? down\b|^\s*(?:easier|simpler)\b")


def slash(msg, ctx):
    m = re.match(r"^/(upgrade|degrade|evaluate|predict|similar|mode|defaults|customise|customize)\b\s*(.*)$", msg, re.S)
    if not m:
        return None
    cmd, arg = m.group(1), m.group(2).strip()
    if cmd == "upgrade" and arg:
        return do_rewrite(arg, None, "up", ctx)
    if cmd == "degrade" and arg:
        return do_rewrite(arg, None, "down", ctx)
    if cmd == "evaluate" and arg:
        return do_analyze(arg, ctx)
    if cmd == "predict" and arg:
        return do_predict(arg, ctx)
    if cmd == "similar" and arg:
        p = engine.predict(arg)
        return run_gen({"topic": " ".join(arg.split()[1:12]) or arg, "levels": [p["level"]], "difficulty": None, "n": 5}, ctx)
    if cmd == "mode":
        kind, _, did = arg.partition(" ")
        doc = docs.get(did.strip())
        if not doc:
            return resp("I no longer have that file in memory. Please attach it again.", "doc_missing", ctx={**ctx, "pending": None})
        return as_paper(doc, ctx, hint=False) if kind == "paper" else start_material(doc, ctx, hint=False)
    if cmd == "defaults":
        pend = ctx.get("pending")
        if pend:
            return continue_flow(pend["flow"], dict(pend["slots"]), ctx, force=True)
        if ctx.get("doc"):
            return continue_flow("doc", {}, ctx, force=True)
    if cmd in ("customise", "customize") and ctx.get("doc"):
        q, chips = slot_prompt("doc", "n")
        return resp("Let's set up the format. " + q, "ask", chips=chips + [chip("Generate now", "/defaults", "spark")],
                    ctx={**ctx, "pending": {"flow": "doc", "slots": {}, "ask": "n"}})
    return None


def handle(message, ctx=None):
    ctx = dict(ctx or {})
    ctx.pop("awaiting_answer", None)                        # older sessions: answer grading has been removed
    msg = (message or "").strip()
    if not msg:
        return help_msg(ctx)
    low = msg.lower()

    out = slash(msg, ctx)                                    # 1. buttons send slash commands
    if out:
        return out

    if ctx.get("pending"):                                   # 2. we asked something: is this the answer?
        if re.match(r"^\s*(generate now|use (the )?defaults?|just generate|go ahead|defaults?|that'?s (all|enough)|skip (all|the rest))\W*$", low):
            return slash("/defaults", ctx)
        out = answer_pending(msg, ctx)
        if out:
            return out
        starts_new = re.search(UP_RE + "|" + DOWN_RE, low) or re.match(r"^\s*(generate|create|make|give|predict|what level|evaluate)\b", low) or looks_like_question(msg)
        if not starts_new and ctx["pending"].get("ask") not in (None, "start"):
            q, chips = slot_prompt(ctx["pending"]["flow"], ctx["pending"]["ask"])
            return resp("Sorry, I didn't catch that. " + q, "ask", chips=chips + [chip("Generate now", "/defaults", "spark")], ctx=ctx)
        ctx["pending"] = None

    out = followup(msg, ctx)                                 # 3. "more", "harder", "change level to Apply"
    if out:
        return out

    if re.match(r"^\s*(thanks|thank you|thx|great|nice|cool|awesome|perfect)\b", low) and len(msg.split()) <= 6:      # 4. small talk
        return resp("You're welcome! Tell me what to do next, or tap an option.", "thanks", chips=main_chips(ctx), ctx=ctx)
    if re.match(r"^\s*(hi|hello|hey|help|what can you do|who are you|how does this work|how do you work|menu|start)\b", low) and len(msg.split()) <= 8:
        return help_msg(ctx)
    if re.match(r"^\s*(forget|clear|remove|stop using)\b[^.?]{0,20}\b(document|doc|file|slides|material)\b", low):
        return resp("Done. I've forgotten that document.", "cleardoc", chips=main_chips(ctx), ctx={**ctx, "doc": None, "pending": None})

    cmd, body = split_cmd_body(msg)

    mu, md = re.search(UP_RE, low), re.search(DOWN_RE, low)  # 5. upgrade / degrade / predict / evaluate a question
    for hit, direction in ((md, "down"), (mu, "up")):
        if hit and hit.start() < 45:
            q = body or ctx.get("question")
            if not q:
                w = "Upgrade" if direction == "up" else "Degrade"
                return resp(f"Which question should I {w.lower()}? Paste it after a colon, for example: **{w}: Explain the PEAS framework.**", "need_input", ctx=ctx)
            return do_rewrite(q, find_level(cmd), direction, ctx)
    if re.search(r"\b(predict|classify|identify)\b[^.?\n]{0,25}\b(level|bloom|difficulty)\b|\bwhat(?:'s| is)?\b[^.?\n]{0,15}\b(bloom|level|difficulty)\b|\bwhich (?:bloom'?s? )?level\b|\bbloom'?s? level of\b", low):
        q = body or ctx.get("question")
        return do_predict(q, ctx) if q else resp("Paste the question after a colon and I'll predict its Bloom level.", "need_input", ctx=ctx)
    if re.search(r"\b(evaluate|review|check|rate|assess|critique|score|analy[sz]e|improve|validate)\b[^.?\n]{0,30}\bquestion\b|\bquality of\b", low):
        q = body or ctx.get("question")
        return do_analyze(q, ctx) if q else resp("Paste the question after a colon and I'll analyse it.", "need_input", ctx=ctx)

    first = re.sub(r"[^a-z]", "", low.split()[0]) if low.split() else ""      # 6. asking in words about papers / documents
    wants_gen = re.match(r"^\s*(?:please\s+)?(?:can you\s+|could you\s+)?(?:generate|create|make|give|write|prepare|draft|suggest|produce|set|show|i (?:need|want)|need|want)\b", low)
    if wants_gen and not ctx.get("doc") and re.search(rf"\bfrom\s+{DOC_WORDS}\b|\b(?:my|this|the)\s+(?:slides|ppt|pptx|document|notes|pdf)\b", low):
        return attach_help("material", ctx)
    if (re.search(r"\b(upload|attach|uploaded|scan|ocr)\b|\b(question|exam|test)\s+paper\b", low) or (re.search(r"\bpdf\b", low) and first not in engine.ALLOWED_START)) \
            and len(msg.split()) <= 16 and not (wants_gen and re.search(r"\b(generate|create|prepare|draft|write|make|set)\b", low) and "paper" in low):
        return attach_help("paper", ctx)

    if wants_gen and re.search(r"\bquestions?\b|\bquiz\b|question paper|\bmcqs?\b", low):      # 7. generate
        if ctx.get("doc") and not re.search(r"\b(textbook|aima|question bank|dataset)\b", low):
            return continue_flow("doc", extract_spec(msg, "doc"), ctx)
        slots = extract_spec(msg, "gen")
        topic = find_topic(msg)
        if topic:
            slots["topic"] = topic
        return continue_flow("gen", slots, ctx)

    if looks_like_question(msg):                             # 8. a pasted question -> analyse it, then offer upgrade / degrade
        return do_analyze(msg, ctx)

    r = llm_route(msg, ctx)                                  # 9. let the AI try to understand it
    out = dispatch_llm(r, ctx) if r else None
    if out:
        return out
    return resp("I didn't quite get that. " + ("" if llm.available() else "(Without an API key I understand a fixed set of requests.) ") + "Here are some things I can do:",
                "unknown", chips=main_chips(ctx), ctx=ctx)
