# guardrails.py -- Week 5, Part 3 (Guardrails and Human in the Loop)
#
# Every block below is copied from the Week 5 companion guides.
# The "Source:" comment names the guide and listing. Nothing is adapted.
# This file has no imports from your own project, so it can be tested on its own.

import re
import unicodedata
from email.utils import parseaddr


# ---------------------------------------------------------------------------
# Layer 1: input screening
# Source: Guide 4 (Guardrails and Prompt Injection), Listing 1
# Guide's own warning: this catches the obvious attack and misses paraphrases.
# ---------------------------------------------------------------------------a

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"you are now (in )?(developer|admin|dan) mode",
    r"reveal (your )?(system prompt|instructions)",
]



def screen_text(text):
    hits = [p for p in INJECTION_PATTERNS if re.search(p, text, re.IGNORECASE)]
    return {"flagged": bool(hits), "matched": len(hits)}


# ---------------------------------------------------------------------------
# Layer 2: context hygiene
# Source: Guide 4, Listing 2 - strip hidden content and report what was found
# ---------------------------------------------------------------------------
ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\ufeff"
ZERO_WIDTH = dict.fromkeys(map(ord, ZERO_WIDTH_CHARS), None)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def sanitize_document(text):
    findings = []
    if HTML_COMMENT.search(text):
        findings.append("html_comment")
    if any(ch in text for ch in ZERO_WIDTH_CHARS):
        findings.append("zero_width_chars")
    cleaned = HTML_COMMENT.sub("", text).translate(ZERO_WIDTH)
    return unicodedata.normalize("NFKC", cleaned), findings


# Source: Guide 4, Listing 3 - mark retrieved text as untrusted data when building the prompt
# (only wrap_untrusted is here; the guide's SYSTEM_PROMPT belongs with the Knowledge agent)
def wrap_untrusted(chunks):
    parts = []
    for i, text in enumerate(chunks, 1):
        safe = text.replace("</untrusted_document>", "")  # prevent tag break-out
        parts.append(f'<untrusted_document id="{i}">\n{safe}\n</untrusted_document>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Layer 3: action guardrail - the email allow-list
# Source: Guide 4, Listing 4 (naive_check, the guide's buggy example, is left out)
# Replace "company.example" with the real domain(s) you want to allow.
# ---------------------------------------------------------------------------
COMPANY_DOMAINS = {"company.example","gmail.com",} # like .3du.jo


def domain_of(address):
    _, addr = parseaddr(address)
    return addr.rsplit("@", 1)[-1].lower() if "@" in addr else ""


def check_email_action(recipients, allowed_domains=COMPANY_DOMAINS, max_recipients=3):
    if not recipients:
        return {"allowed": False, "reason": "no recipients"}
    if len(recipients) > max_recipients:
        return {"allowed": False, "reason": f"more than {max_recipients} recipients"}
    for r in recipients:
        d = domain_of(r)
        if d not in allowed_domains:  # exact match on the WHOLE domain
            return {"allowed": False, "reason": f"domain '{d or r}' is not allow-listed"}
    return {"allowed": True, "reason": "ok"}


# ---------------------------------------------------------------------------
# Layer 4: output guardrails
# Source: Guide 4, Listing 5 - redaction, canary, and grounding checks
# ---------------------------------------------------------------------------
SECRET_PATTERNS = {
    "google_api_key": r"AIza[0-9A-Za-z_\-]{35}",
    "bearer_token": r"Bearer\s+[A-Za-z0-9._\-]{20,}",
    "sk_style_key": r"\bsk-[A-Za-z0-9]{20,}\b",
}


def redact_secrets(text):
    found = []
    for name, pat in SECRET_PATTERNS.items():
        if re.search(pat, text):
            found.append(name)
            text = re.sub(pat, "[REDACTED]", text)
    return text, found


CANARY = "PG-CANARY-7f3a91"  # placed inside the system prompt; must never appear


def leaks_system_prompt(output):
    return CANARY in output


def check_grounding(cited_ids, retrieved_ids, is_refusal=False):
    if is_refusal:
        return {"grounded": True, "reason": "refusal"}
    if not cited_ids:
        return {"grounded": False, "reason": "no sources cited"}
    unknown = set(cited_ids) - set(retrieved_ids)
    if unknown:
        return {"grounded": False, "reason": f"cites unretrieved sources {sorted(unknown)}"}
    return {"grounded": True, "reason": "ok"}


# ---------------------------------------------------------------------------
# Human-in-the-loop decision logic (a plain function, testable without LangGraph)
# Source: Guide 5 (Human in the Loop with LangGraph), Listing 3 - decide(): fail closed,
# and re-check edited recipients
# ---------------------------------------------------------------------------
ALLOWED = {"company.example","gmail.com",}

def recipients_ok(addresses):
    well_formed = all(isinstance(a, str) and a.count("@") == 1 for a in addresses)
    if not addresses or not well_formed:
        return False
    return all(a.split("@")[1].lower() in ALLOWED for a in addresses)


def decide(state, raw):
    '''Return (next_node, state_update). Anything unclear means: do NOT send.'''
    if not isinstance(raw, dict):
        return "cancelled", {"note": "invalid decision"}
    action = raw.get("action")
    if action == "approve":
        return "send", {}
    if action == "edit":
        edited = {k: raw[k] for k in ("to", "subject", "body") if k in raw}
        if not recipients_ok(edited.get("to", state["to"])):
            return "blocked", {"note": "edited recipients not allowed"}
        return "send", edited
    if action == "reject":
        return "cancelled", {"note": raw.get("reason", "rejected by reviewer")}
    return "cancelled", {"note": f"unknown action {action!r}"}