import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, jsonify, request, render_template
import google.generativeai as palm

API_KEY = 'AIzaSyDKbnLMFZ3I1H2iXcBMsYpe31IIu4BM4aM'  # (kept as you requested)
model_id = "models/text-bison-001"

palm.configure(api_key=API_KEY)

GEN_CFG = dict(
    model=model_id,
    temperature=0.3,
    max_output_tokens=256,
)

# =========================
# Parsing / cleaning utils
# =========================
BULLET_LINE = re.compile(r"^\s*(?:[\*\-\u2022]|\d+[.)])\s*")  # *, -, •, 1. / 1)
MULTI_SPACE = re.compile(r"\s+")
CURRENCY_LINE = re.compile(r"\$[\d,]+(?:\s*[–-]\s*|\s*to\s*)\$?[\d,]+", re.IGNORECASE)

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", " ").replace("\n", " ").replace("\\", " ")
    s = re.sub(MULTI_SPACE, " ", s)
    return s.strip()

def _parse_star_list(text, max_items=None):
    """Robust bullet/number list -> list[str]. Falls back to comma-split."""
    if not text:
        return []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    items = []
    for line in lines:
        line = re.sub(BULLET_LINE, "", line).strip()
        if line:
            items.append(line)
    if not items and "," in text:
        items = [x.strip() for x in text.split(",") if x.strip()]
    if max_items:
        items = items[:max_items]
    return [clean_text(x) for x in items if x]

def _parse_college_degree_list(text, max_items=3):
    items = _parse_star_list(text, max_items=max_items)
    fixed = []
    for it in items:
        it = it.replace(" - ", " | ").replace(" — ", " | ").replace(" – ", " | ")
        fixed.append(it)
    return fixed

def _extract_salary_range(text):
    if not text:
        return ""
    m = CURRENCY_LINE.search(text.replace("\n", " "))
    if m:
        return clean_text(m.group(0))
    items = _parse_star_list(text, max_items=1)
    return items[0] if items else clean_text(text)

# =========================
# LLM wrapper (retries)
# =========================
def llm_generate(prompt: str, **overrides) -> str:
    cfg = GEN_CFG.copy()
    cfg.update(overrides or {})
    # defensive cap
    cfg["max_output_tokens"] = min(int(cfg.get("max_output_tokens", 256)), 512)

    backoff = 0.8
    for _ in range(3):
        try:
            resp = palm.generate_text(prompt=prompt, **cfg)
            result = getattr(resp, "result", None)
            if result:
                return str(result)
        except Exception:
            pass
        time.sleep(backoff)
        backoff *= 2
    return ""

# =========================
# Parallel map helper
# =========================
def _map_parallel(items, fn, max_workers=8):
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
        futures = {ex.submit(fn, idx, it): idx for idx, it in enumerate(items)}
        results = [None] * len(items)
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = ""
        return results

# =========================
# Per-role prompt workers
# =========================
def _jd_worker(_, role):
    prompt = (
        f"Explain what a {role} does in 2–3 plain sentences for a middle schooler. "
        "No headings, no lists, <= 280 characters."
    )
    return clean_text(llm_generate(prompt, max_output_tokens=160))

def _courses_worker(_, role):
    prompt = (
        f"List the top 5 college courses for becoming a {role} as a bulleted list using '*'. "
        "Only the course names. No titles, no extra text."
    )
    txt = llm_generate(prompt, max_output_tokens=160)
    return ", ".join(_parse_star_list(txt, max_items=5))

def _colleges_worker(_, role):
    prompt = (
        f"Give the top 3 schools for {role} and the relevant undergrad degree. "
        "Return as a bulleted list using '*' where each line is 'College | Degree'. "
        "No headings, no numbers."
    )
    txt = llm_generate(prompt, max_output_tokens=160)
    return ", ".join(_parse_college_degree_list(txt, max_items=3))

def _salary_worker(_, role):
    prompt = (
        f"Give a single US base salary range for an entry-level {role}. "
        "Return only the range like '$70,000–$110,000'. No words, no bullets."
    )
    txt = llm_generate(prompt, max_output_tokens=60)
    rng = _extract_salary_range(txt)
    return rng if rng else clean_text(txt)

def _resources_worker(_, role):
    prompt = (
        f"List the top 5 free learning resources (sites or courses) for becoming a {role}. "
        "Return as a bulleted list using '*', each item a concise name (optionally with URL). "
        "No titles, no descriptions beyond the name."
    )
    txt = llm_generate(prompt, max_output_tokens=200)
    return ", ".join(_parse_star_list(txt, max_items=5))

def findJD(jobroles):
    return _map_parallel(jobroles, _jd_worker)

def findcourses(jobroles):
    return _map_parallel(jobroles, _courses_worker)

def findcolleges(jobroles):
    return _map_parallel(jobroles, _colleges_worker)

def findsalaryrange(jobroles):
    return _map_parallel(jobroles, _salary_worker)

def findonlineresources(jobroles):
    return _map_parallel(jobroles, _resources_worker)

def _find_roles_list(interests):
    # Trim interests to keep prompts safe/cheap
    interests = clean_text(interests)[:600]
    prompt = (
        f"{interests}\n\n"
        "Return EXACTLY 5 job roles as a bulleted list using '*'. "
        "No headings, no extra text, no numbers. Example:\n"
        "* Software Engineer\n* Data Analyst\n* UX Designer\n* Robotics Engineer\n* Technical Writer"
    )
    result = llm_generate(prompt, max_output_tokens=128)
    roles = _parse_star_list(result, max_items=5)

    # dedupe preserve order
    seen = set()
    roles = [r for r in roles if not (r in seen or seen.add(r))]
    roles = roles[:5]
    # pad if needed (rare)
    while len(roles) < 5:
        roles.append("General Technologist")
    return roles

def findjobroles(interests):
    # Keep your behavior: return dict with arrays
    jobroles = _find_roles_list(interests)

    return {
        "jobroles": jobroles,
        "JD": findJD(jobroles),
        "courses": findcourses(jobroles),
        "colleges": findcolleges(jobroles),
        "salary": findsalaryrange(jobroles),
        "onlineRes": findonlineresources(jobroles),
    }

# =========================
# Flask app / routes
# =========================
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/findjobroles', methods=['POST'])
def get_job_roles():
    interests = (request.form.get('interests') or '').strip()
    if len(interests) < 3:
        return jsonify({"error": "Please enter a bit more detail (min 3 chars)."}), 400

    # keep your steering text
    interests = interests + ' Exclude any headers like top 5 job roles and explanations.'

    data = findjobroles(interests)

    # clean lists (keep your existing names/shape)
    data['jobroles'] = [clean_text(role) for role in data['jobroles']]
    data['JD']       = [clean_text(jd) for jd in data['JD']]
    data['courses']  = [clean_text(c) for c in data['courses']]
    data['colleges'] = [clean_text(c) for c in data['colleges']]
    data['salary']   = [clean_text(s) for s in data['salary']]
    data['onlineRes']= [clean_text(r) for r in data['onlineRes']]

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)