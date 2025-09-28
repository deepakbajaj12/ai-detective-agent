"""Lightweight Generative AI integration helpers.

This module is OPTIONAL: it tries to use an OpenAI API key (if provided) or a local
Transformers model for summarization. If neither is available, it falls back to a
simple heuristic text assembler so the endpoint still responds.

Environment Variables:
  OPENAI_API_KEY  -> if present, will attempt to call OpenAI Chat Completions
  GENAI_MODEL     -> override default local model name (for transformers fallback)

Add the following optional packages if you want richer output:
  pip install openai==1.* transformers>=4.40 accelerate==0.* safetensors faiss-cpu

The design keeps imports lazy so the core app runs without heavy deps.
"""
from __future__ import annotations
from typing import List, Dict, Any
import os
import math


def _truncate(txt: str, max_chars: int = 6000) -> str:
    if len(txt) <= max_chars:
        return txt
    return txt[: max_chars - 20] + "... <truncated>"


def _openai_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return "[OpenAI SDK not installed]"  # silent fallback
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an analytical forensic investigation assistant. Be concise, structured, and cite evidence succinctly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            max_tokens=700,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI call failed: {e}]"


def _transformers_available() -> bool:
    try:
        import transformers  # type: ignore
        return True
    except Exception:
        return False


_HF_PIPE = None


def _hf_summarize(text: str, max_chars: int = 3000) -> str:
    global _HF_PIPE
    if not _transformers_available():
        return "[Transformers not installed]"
    import torch  # type: ignore
    from transformers import pipeline  # type: ignore
    if _HF_PIPE is None:
        model_name = os.environ.get("GENAI_MODEL", "sshleifer/distilbart-cnn-12-6")
        try:
            _HF_PIPE = pipeline("summarization", model=model_name, device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            return f"[Failed to load transformers model {model_name}: {e}]"
    chunk = _truncate(text, max_chars)
    try:
        pieces = _HF_PIPE(chunk, max_length=260, min_length=60, do_sample=False)
        if pieces and isinstance(pieces, list):
            return pieces[0].get("summary_text", "[No summary text]")
    except Exception as e:
        return f"[Summarization failed: {e}]"
    return "[Empty summary]"


def _heuristic_summary(clues: List[str], suspects: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("Heuristic Analytical Summary (no AI backend active):")
    lines.append(f"Total clues: {len(clues)} | Total suspects: {len(suspects)}")
    if clues:
        lines.append("Representative clues (up to 5):")
        for c in clues[:5]:
            lines.append(f"  - {c[:140]}")
    if suspects:
        lines.append("Top suspects by composite score:")
        top_sorted = sorted(suspects, key=lambda s: s.get('composite_score', 0), reverse=True)[:3]
        for s in top_sorted:
            lines.append(f"  - {s['name']} | composite: {s.get('composite_score'):.2f} | risk: {s.get('risk_level')} | primary offense: {s.get('primary_offense','-')}")
    lines.append("(Enable OpenAI API key or install transformers for richer reasoning.)")
    return "\n".join(lines)


def generate_case_analysis(case_id: str, clues: List[str], suspects: List[Dict[str, Any]], style: str = "brief") -> Dict[str, Any]:
    """Produce a structured analytical narrative.

    Strategy (tiered):
      1. If OPENAI key -> structured prompt to GPT style model
      2. Else if transformers installed -> summarization of compiled context
      3. Else -> heuristic textual summary
    """
    style = style.lower().strip()
    style_tag = style if style in {"brief", "detailed"} else "brief"

    # Build context
    suspect_lines = []
    for s in sorted(suspects, key=lambda x: x.get('composite_score', 0), reverse=True):
        suspect_lines.append(
            f"- {s['name']} (comp={s.get('composite_score',0):.2f}, ml={s.get('score',0):.2f}, ev={s.get('evidence_score',0):.2f}, risk={s.get('risk_level')}, offense={s.get('primary_offense','-')}, sev={s.get('primary_offense_severity','-')})"
        )
    base_context = (
        f"CASE: {case_id}\nCLUES (count={len(clues)}):\n" +
        "\n".join(f"* {c}" for c in clues[:40]) +
        "\n\nSUSPECT METRICS:\n" + "\n".join(suspect_lines[:20])
    )

    prompt = f"""
You are an AI investigative analyst. Produce a {style_tag} structured report with sections:
1. Situation Overview
2. Key Clue Signals (bullet form)
3. Suspect Comparative Assessment (reference scores concisely)
4. Offense / Allegation Impact
5. Analytical Risks & Uncertainties
6. Recommended Next Investigative Actions

Rules:
- Be evidence grounded. Avoid fabricating clues.
- If data sparse, clearly state limitations.
- Keep each section succinct; avoid repetition.

DATA CONTEXT:\n{_truncate(base_context, 9000)}
""".strip()

    if _openai_available():
        content = _call_openai(prompt)
        backend = "openai"
    elif _transformers_available():
        # We create a pseudo summary then append a structured heuristic tail
        summary = _hf_summarize(base_context)
        tail = _heuristic_summary(clues, suspects)
        content = summary + "\n\n---\nHeuristic Context\n" + tail
        backend = "transformers"
    else:
        content = _heuristic_summary(clues, suspects)
        backend = "heuristic"

    return {
        "backend": backend,
        "style": style_tag,
        "report": content,
        "tokens_estimate": len(prompt.split()) + len(content.split())
    }


__all__ = ["generate_case_analysis"]
