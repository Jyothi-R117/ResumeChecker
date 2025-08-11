import streamlit as st
import re, io
from utils import (
    extract_text_from_pdf, extract_text_from_docx,
    clean_text, compute_match_score, extract_keywords,
    ats_checks, suggest_improvements, safe_read_txt
)

st.set_page_config(page_title="ResumeFit & ATS Check", page_icon="🧩", layout="centered")

st.title("🧩 ResumeFit & ATS Checker")
st.caption("Upload a resume and paste a job description. Get a match score, ATS health, and actionable suggestions.")

with st.sidebar:
    st.header("How it works")
    st.write("• TF-IDF + cosine similarity for overall fit\n"
             "• JD keyword coverage (% hit)\n"
             "• Simple ATS checks (sections, contact info, bullets, dates, tables)")
    st.write("Tip: PDFs parse best. Avoid images/scans.")

# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
with col2:
    jd_text = st.text_area("Paste Job Description", height=220, placeholder="Paste the JD here...")

if st.button("Analyze"):
    if not resume_file or not jd_text.strip():
        st.warning("Please upload a resume and paste a job description.")
        st.stop()

    # Read resume text
    ext = resume_file.name.lower().split(".")[-1]
    if ext == "pdf":
        resume_text = extract_text_from_pdf(resume_file)
    elif ext == "docx":
        resume_text = extract_text_from_docx(resume_file)
    else:
        resume_text = safe_read_txt(resume_file)

    if not resume_text.strip():
        st.error("Could not extract text from the resume (is it a scanned image?). Try a different file.")
        st.stop()

    # Clean
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    # Scores
    match_score, top_terms = compute_match_score(resume_clean, jd_clean)
    jd_keywords = extract_keywords(jd_clean, top_k=40)  # most salient JD terms
    present = [k for k in jd_keywords if k.lower() in resume_clean]
    missing = [k for k in jd_keywords if k.lower() not in resume_clean]

    # ATS checks
    ats_report = ats_checks(resume_text, ext)

    # Suggestions
    suggestions = suggest_improvements(
        match_score=match_score,
        present=present,
        missing=missing,
        ats_report=ats_report
    )

    # --- Output UI ---
    st.subheader("Results")
    st.metric("Match Score", f"{int(round(match_score*100))} / 100")
    st.progress(min(100, int(round(match_score*100))))

    colA, colB = st.columns(2)
    with colA:
        st.write("**JD Keywords Covered**")
        st.success(f"{len(present)} matched / {len(jd_keywords)} extracted")
        if present:
            st.write(", ".join(sorted(set(present))[:30]))
    with colB:
        st.write("**Missing / Weak Keywords**")
        if missing:
            st.warning(", ".join(sorted(set(missing))[:30]))
        else:
            st.success("You covered the top extracted keywords.")

    st.write("---")
    st.write("### ATS Health Check")
    for k, v in ats_report.items():
        icon = "✅" if v["ok"] else "⚠️"
        st.write(f"{icon} **{k}** — {v['msg']}")

    st.write("---")
    st.write("### Actionable Suggestions")
    for s in suggestions:
        st.write(f"- {s}")

    # Download checklist
    output = io.StringIO()
    output.write("ResumeFit & ATS Checklist\n\n")
    output.write(f"Match Score: {int(round(match_score*100))}/100\n\n")
    output.write("Present JD Keywords:\n")
    output.write(", ".join(present) + "\n\n")
    output.write("Missing/Weak JD Keywords:\n")
    output.write(", ".join(missing) + "\n\n")
    output.write("ATS Findings:\n")
    for k, v in ats_report.items():
        output.write(f"- {k}: {v['msg']}\n")
    output.write("\nSuggestions:\n")
    for s in suggestions:
        output.write(f"- {s}\n")

    st.download_button(
        "Download Suggestions",
        output.getvalue().encode("utf-8"),
        file_name="resumefit_ats_suggestions.txt",
        mime="text/plain"
    )

    with st.expander("Show extracted resume text (debug)"):
        st.text(resume_text[:8000])  # safety cap
