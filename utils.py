import re
import pdfplumber
from docx import Document
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SECTION_HINTS = [
    "summary", "profile", "skills", "technical skills", "experience",
    "work experience", "projects", "education", "certifications", "awards"
]

def extract_text_from_pdf(file_obj):
    text = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            # detect tables as a signal for ATS risk; we still keep text
            text.append(page.extract_text() or "")
    return "\n".join(text)

def extract_text_from_docx(file_obj):
    bytes_data = file_obj.read()
    doc = Document(BytesIO(bytes_data))
    return "\n".join([p.text for p in doc.paragraphs])

def safe_read_txt(file_obj):
    return file_obj.read().decode("utf-8", errors="ignore")

def clean_text(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9+#.\-()\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def compute_match_score(resume_clean, jd_clean):
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
    X = vect.fit_transform([jd_clean, resume_clean])
    sim = cosine_similarity(X[0:1], X[1:2]).ravel()[0]  # 0..1
    # collect top JD terms by TF-IDF weight to show
    jd_vec = X[0].toarray().ravel()
    terms = vect.get_feature_names_out()
    top_idx = jd_vec.argsort()[-15:][::-1]
    top_terms = [terms[i] for i in top_idx]
    # scale to 0..1; optional softening
    score = float(sim)
    return score, top_terms

def extract_keywords(jd_clean, top_k=40):
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X = vect.fit_transform([jd_clean])
    arr = X.toarray().ravel()
    terms = vect.get_feature_names_out()
    idx = arr.argsort()[-top_k:][::-1]
    # keep only alnum / common symbols
    kws = [terms[i] for i in idx if re.search(r"[a-z0-9]", terms[i])]
    # prefer multi-word phrases first
    kws_sorted = sorted(kws, key=lambda x: (-len(x.split()), x))
    return kws_sorted

def has_contact_info(text):
    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.search(r"(\+?\d[\d\-\s]{7,}\d)", text)
    return bool(email), bool(phone)

def has_sections(text):
    t = text.lower()
    hits = [s for s in SECTION_HINTS if s in t]
    return hits

def bullet_usage(text):
    # quick check for common bullet chars or multiple short lines
    bullets = re.findall(r"•|- |\* ", text)
    return len(bullets) >= 5

def has_dates(text):
    # crude year pattern
    return bool(re.search(r"(20\d{2}|19\d{2})", text))

def ats_checks(resume_text, ext):
    email_ok, phone_ok = has_contact_info(resume_text)
    sections = has_sections(resume_text)
    bullets_ok = bullet_usage(resume_text)
    dates_ok = has_dates(resume_text)
    tables_flag = "⚠️ Use tables sparingly; some ATS drops table content."
    file_ok = ext == "pdf" or ext == "docx"

    return {
        "File Type": {"ok": file_ok, "msg": "PDF/DOCX are ATS-safe. Avoid images/scans."},
        "Contact Info": {"ok": email_ok and phone_ok, "msg": "Include a professional email and reachable phone number."},
        "Clear Sections": {"ok": len(sections) >= 3, "msg": f"Detected sections: {', '.join(sections) or 'none'}."},
        "Bullet Structure": {"ok": bullets_ok, "msg": "Use concise bullet points with action verbs and impact."},
        "Dates Present": {"ok": dates_ok, "msg": "Include years for roles/projects (YYYY–YYYY)."},
        "Tables/Graphics": {"ok": True, "msg": tables_flag}
    }

def suggest_improvements(match_score, present, missing, ats_report):
    suggestions = []
    # Match score gates
    if match_score < 0.45:
        suggestions.append("Customize your summary and top bullets with JD keywords to raise the match score.")
    elif match_score < 0.65:
        suggestions.append("Strengthen skills and achievements using exact phrasing from the JD where truthful.")

    # Keyword gaps
    if missing:
        suggestions.append(f"Incorporate missing role-specific keywords where accurate: {', '.join(missing[:12])}.")

    # ATS fixes
    for k, v in ats_report.items():
        if not v["ok"]:
            suggestions.append(f"ATS: {v['msg']} (Improve: {k}).")

    # General best practices
    suggestions.extend([
        "Start bullets with strong action verbs (Built, Optimized, Automated) and quantify impact.",
        "Keep formatting simple: one column, standard fonts, no headers/footers.",
        "Save as text-based PDF (not scanned)."
    ])
    return suggestions
