"""
SynthPanel — AI-powered synthetic customer interview tool
BNPP Personal Finance · Azure Hackathon 2026
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(Path(__file__).parent.parent / ".env", override=True)


def get_setting(name, default=None, required=False):
    """Resolve config from environment first, then Streamlit secrets."""
    val = os.getenv(name)
    if val:
        return val

    try:
        if name in st.secrets:
            secret_val = str(st.secrets[name]).strip()
            if secret_val:
                # Keep values available to existing os.environ usage.
                os.environ[name] = secret_val
                return secret_val
    except Exception:
        # st.secrets can be unavailable in some local contexts.
        pass

    if required and default is None:
        raise RuntimeError(f"Missing required setting: {name}")

    return default

# ──────────────────────────────────────────────────────────
# INE static lookup — representative median monthly income
# by age bracket × region (EUR/month, 2023 estimates)
# ──────────────────────────────────────────────────────────
INE_INCOME_LOOKUP = {
    ("18-24", "MADRID"):       1_150,
    ("18-24", "SUR"):            950,
    ("25-34", "MADRID"):       1_650,
    ("25-34", "ESTE"):         1_450,
    ("25-34", "SUR"):          1_250,
    ("35-44", "MADRID"):       1_900,
    ("35-44", "NORESTE"):      1_600,
    ("35-44", "ESTE"):         1_650,
    ("35-44", "SUR"):          1_350,
    ("45-54", "MADRID"):       2_100,
    ("45-54", "NORESTE"):      1_800,
    ("45-54", "ESTE"):         1_850,
    ("45-54", "NOROESTE"):     1_700,
    ("55-64", "MADRID"):       2_000,
    ("55-64", "NORESTE"):      1_750,
    ("55-64", "SUR"):          1_400,
    ("65+",   "MADRID"):       1_400,
    ("65+",   "SUR"):          1_100,
    ("65+",   "NOROESTE"):     1_200,
}

INE_CONTEXT_TEXT = """Representative INE Spain median monthly income benchmarks (2023):
- 18-24 year-olds: €950–€1,150 depending on region
- 25-34 year-olds: €1,250–€1,650
- 35-44 year-olds: €1,350–€1,900
- 45-54 year-olds: €1,700–€2,100
- 55-64 year-olds: €1,400–€2,000
- 65+ year-olds:   €1,100–€1,400
Madrid and NORESTE regions tend 15-20% above national median; SUR and ISLAS CANARIAS 10-15% below."""

# ──────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SynthPanel — BNPP PF",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
body, .stApp { background-color: #0a0a0a !important; font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; color: #f0f0f0 !important; }
header[data-testid="stHeader"] { background-color: #0a0a0a !important; }
[data-testid="stSidebar"] { background-color: #111111 !important; }
.stSelectbox label, .stTextInput label, .stButton button, p, span, li, h1, h2, h3, h4, h5, h6, div { color: #f0f0f0 !important; }
.stSelectbox [data-baseweb="select"] { background-color: #1a1a1a !important; }
[data-testid="stForm"] { background-color: #1a1a1a !important; border-color: #333 !important; }
input { background-color: #1a1a1a !important; color: #f0f0f0 !important; }

/* Header bar */
.synth-header {
    background-color: #00295E; color: white; padding: 18px 24px;
    border-radius: 8px; margin-bottom: 16px;
}
.synth-header h1 { margin: 0; font-size: 1.5rem; font-weight: 700; color: white; }
.synth-header p  { margin: 4px 0 0 0; font-size: 0.85rem; color: #B0C4DE; }

/* Logo bar */
.logo-bar {
    display: flex; align-items: center; justify-content: space-between;
    background: #FFFFFF; border-radius: 8px; padding: 12px 24px; margin-bottom: 16px;
    border: 1px solid #ddd;
}
.logo-bar img { height: 40px; object-fit: contain; }

/* Left column card */
.customer-card {
    background: #1a1a1a; border-radius: 8px; padding: 16px;
    border-left: 4px solid #3399FF; margin-bottom: 12px;
}
.customer-card h4 { margin: 0 0 8px 0; color: #3399FF !important; }
.customer-card p  { margin: 2px 0; font-size: 0.88rem; color: #ddd !important; }

/* Stats card */
.stats-card {
    background: #1a1a1a; border-radius: 8px; padding: 14px; margin-bottom: 12px;
}
.stats-card h4 { margin: 0 0 8px 0; color: #f0f0f0 !important; font-size: 0.95rem; }

/* Chat bubbles */
.analyst-bubble {
    background: #1e3a5f; border-radius: 12px 12px 2px 12px;
    padding: 12px 16px; margin: 6px 0 6px auto; max-width: 80%;
    text-align: right; font-size: 0.92rem; color: #f0f0f0 !important;
}
.customer-bubble {
    background: #1a1a1a; border-radius: 12px 12px 12px 2px;
    padding: 12px 16px; margin: 6px auto 6px 0; max-width: 80%;
    font-size: 0.92rem; color: #f0f0f0 !important;
}
.customer-bubble.green  { border-left: 4px solid #2E7D32; }
.customer-bubble.amber  { border-left: 4px solid #F9A825; }
.customer-bubble.red    { border-left: 4px solid #C62828; }

/* Confidence pills */
.conf-pill {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600; color: white; margin-left: 8px;
    vertical-align: middle;
}
.conf-green { background: #2E7D32; }
.conf-amber { background: #F9A825; color: #333; }
.conf-red   { background: #C62828; }

/* Status indicator */
.status-idle   { color: #888; font-size: 0.85rem; }
.status-active { color: #2E7D32; font-weight: 600; font-size: 0.85rem; }

/* Validation detail panel */
.val-detail {
    background: #1a1a1a !important; border: 2px solid #3399FF !important; border-radius: 8px;
    padding: 14px 18px; margin-top: 8px; font-size: 0.9rem; color: #f0f0f0 !important;
    line-height: 1.6;
}
.val-detail strong { color: #3399FF !important; font-size: 0.92rem; }
.val-pass { color: #2E7D32 !important; font-weight: 700; }
.val-fail { color: #C62828 !important; font-weight: 700; }

/* Streamlit expander styling */
[data-testid="stExpander"] summary span { color: #3399FF !important; font-weight: 700 !important; font-size: 0.92rem !important; }
[data-testid="stExpander"] { border: 1px solid #3399FF !important; border-radius: 6px !important; margin-top: 4px !important; background-color: #111111 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# Load resources
# ──────────────────────────────────────────────────────────
PERSONAS_DIR = Path(__file__).parent / "output" / "personas"
PROFILES_PATH = Path(__file__).parent / "output" / "cluster_profiles.json"
DASHBOARD_HEATMAP = Path(__file__).parent / "dashboards" / "synthpanel_heatmap.html"


@st.cache_data
def load_personas():
    personas = {}
    if PERSONAS_DIR.exists():
        for fp in sorted(PERSONAS_DIR.glob("*.json")):
            with open(fp) as f:
                data = json.load(f)
                name = data.get("persona_name", fp.stem)
                personas[name] = {"file": fp.name, "data": data}
    return personas


@st.cache_data
def load_cluster_profiles():
    if PROFILES_PATH.exists():
        with open(PROFILES_PATH) as f:
            return json.load(f)
    return {}


PERSONAS = load_personas()
CLUSTER_PROFILES = load_cluster_profiles()

# Map persona file → cluster id for second validation layer
PERSONA_CLUSTER_MAP = {}
for pname, pinfo in PERSONAS.items():
    # filename pattern: cluster_{id}_persona.json
    stem = Path(pinfo["file"]).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        PERSONA_CLUSTER_MAP[pname] = parts[1]

# ──────────────────────────────────────────────────────────
# Azure OpenAI clients (separate resources for chat & nano)
# ──────────────────────────────────────────────────────────
def get_client():
    return AzureOpenAI(
        azure_endpoint=get_setting("AZURE_OPENAI_ENDPOINT", required=True),
        api_key=get_setting("AZURE_OPENAI_API_KEY", required=True),
        api_version=get_setting("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    )


def get_client_nano():
    return AzureOpenAI(
        azure_endpoint=get_setting("AZURE_OPENAI_ENDPOINT_NANO", get_setting("AZURE_OPENAI_ENDPOINT", required=True)),
        api_key=get_setting("AZURE_OPENAI_API_KEY_NANO", get_setting("AZURE_OPENAI_API_KEY", required=True)),
        api_version=get_setting(
            "AZURE_OPENAI_API_VERSION_NANO",
            get_setting("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        ),
    )


MODEL_CHAT = get_setting("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
MODEL_VALID = get_setting("AZURE_OPENAI_DEPLOYMENT_NANO", "gpt-4.1-nano")

# ──────────────────────────────────────────────────────────
# Helper: safe OpenAI call with 1 retry
# ──────────────────────────────────────────────────────────
def call_openai(client, model, messages, temperature=0.7, json_mode=False, retries=1):
    kwargs = dict(model=model, messages=messages, temperature=temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
                continue
            raise e


def parse_json_safe(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

# ──────────────────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────────────────
DEFAULTS = {
    "selected_persona": None,
    "artificial_customer": None,
    "conversation_history": [],
    "rolling_summary": "",
    "summarized_up_to": 0,
    "turn_count": 0,
    "conf_counts": {"GREEN": 0, "AMBER": 0, "RED": 0},
    "session_active": False,
    "confirm_reset": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v if not isinstance(v, (dict, list)) else type(v)(v)


# ──────────────────────────────────────────────────────────
# Persona → system prompt builder
# ──────────────────────────────────────────────────────────
def build_persona_narrative(persona_data):
    d = persona_data.get("demographics", {})
    sc = d.get("social_class", {})
    sc_desc = sc.get("description", "") if isinstance(sc, dict) else str(sc)
    lines = [
        f"Age and generation: {d.get('age_and_generation', 'unknown')}.",
        f"Family situation: {d.get('family_situation', 'unknown')}.",
        f"Social class: {sc_desc}.",
        "",
        "Lifestyle and habits:",
    ]
    for h in persona_data.get("lifestyle_and_habits", []):
        lines.append(f"  - {h}")
    lines.append("")
    lines.append("Motivations:")
    for m in persona_data.get("motivations", []):
        lines.append(f"  - {m}")
    lines.append("")
    lines.append("Frustrations:")
    for fr in persona_data.get("frustrations", []):
        lines.append(f"  - {fr}")
    lines.append("")
    lines.append(f"NPS baseline: {persona_data.get('nps_baseline', 'unknown')}")
    lines.append(f"Credit context: {persona_data.get('credit_context', 'unknown')}")
    return "\n".join(lines)


def build_system_prompt():
    ac = st.session_state.artificial_customer
    persona_data = PERSONAS[st.session_state.selected_persona]["data"]
    narrative = build_persona_narrative(persona_data)

    system = (
        f"You are {ac['full_name']}, a {ac['age']}-year-old {ac['gender']} from {ac['city']}, Spain. "
        f"You are a real customer of a consumer finance provider. "
        f"You are not an AI, you are not a chatbot, you are a person. "
        f"Never break character. Never refer to yourself as an AI or assistant. "
        f"Respond only as this person would respond — in first person, with the emotions, concerns, "
        f"and financial knowledge appropriate to your background.\n\n"
        f"IMPORTANT: Always respond in English, regardless of the customer's nationality or location.\n\n"
        f"SCOPE RESTRICTION: You are in a market research interview about financial products and "
        f"banking services. Only answer questions related to banking, credit, loans, payments, savings, "
        f"insurance, financial habits, customer experience with financial providers, and your personal "
        f"financial situation. If asked about anything unrelated (politics, sports, entertainment, "
        f"coding, recipes, etc.), politely decline and redirect the conversation back to financial topics.\n\n"
        f"Your personality: {ac['personality_note']}\n\n"
        f"Your profile:\n{narrative}"
    )

    # Layer 2 — rolling summary (after turn 50)
    if st.session_state.rolling_summary:
        system += (
            f"\n\nThe following is a summary of earlier parts of this conversation "
            f"that you should treat as your own memory:\n{st.session_state.rolling_summary}"
        )

    return system


# ──────────────────────────────────────────────────────────
# Generate artificial customer
# ──────────────────────────────────────────────────────────
def generate_artificial_customer(persona_data):
    client = get_client()
    persona_json = json.dumps(persona_data, indent=2, ensure_ascii=False)
    # Random seed forces the model to produce a different character every time
    seed = random.randint(1000, 9999)
    system = (
        "You are a character generator for market research simulations. "
        "Given a customer persona, generate a single fictional individual who is a plausible "
        "real-world instance of that persona. Every generation MUST produce a completely different person "
        "with a unique name, age, city, and personality. Never repeat previous outputs. "
        "Return ONLY a JSON object with these fields: "
        "full_name (Spanish-sounding fictional name), age (specific integer within the persona age range), "
        "gender, city (plausible Spanish city matching persona region), "
        "personality_note (one sentence describing their financial personality in plain language), "
        "opening_statement (one sentence the customer would say to introduce themselves if asked). "
        "ALL text fields must be written in English."
    )
    user = f"Persona:\n{persona_json}\n\nRandom seed (use this to vary your output): {seed}"
    raw = call_openai(client, MODEL_CHAT, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=1.0, json_mode=True)
    return parse_json_safe(raw)


# ──────────────────────────────────────────────────────────
# Rolling summary
# ──────────────────────────────────────────────────────────
def maybe_update_rolling_summary(client):
    turn = st.session_state.turn_count
    summarized = st.session_state.summarized_up_to
    if turn < 50 or (turn - summarized) < 50:
        return
    # Summarize the oldest unsummarized 50 turns
    start = summarized
    end = start + 50
    chunk = st.session_state.conversation_history[start:end]
    msgs_text = "\n".join(
        f"{'Analyst' if m['role'] == 'user' else 'Customer'}: {m['content']}"
        for m in chunk if m["role"] in ("user", "assistant")
    )
    system = (
        "Summarize the following conversation from the customer's perspective. "
        "Capture what positions they have taken, what concerns they have expressed, "
        "what they have agreed or disagreed with, and what financial products they have engaged with. "
        "Be concise — maximum 200 words."
    )
    prev = st.session_state.rolling_summary
    user_msg = f"Previous summary:\n{prev}\n\nNew conversation chunk:\n{msgs_text}" if prev else msgs_text
    result = call_openai(client, MODEL_VALID, [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ], temperature=0.3)
    st.session_state.rolling_summary = result
    st.session_state.summarized_up_to = end


# ──────────────────────────────────────────────────────────
# Validation layer 1
# ──────────────────────────────────────────────────────────
def run_validation_layer1(client, persona_data, customer_profile, question, response_text):
    system = (
        "You are a quality-assurance validator for synthetic customer interviews. "
        "Evaluate the customer response against these five criteria. "
        "Be lenient — only return FAIL for clear, unambiguous violations. "
        "Minor elaborations, reasonable inferences, and natural conversational details are acceptable. "
        "Return ONLY a JSON object with exactly these keys, each containing 'verdict' (PASS or FAIL) and 'reason' (one line):\n"
        "- persona_consistency: does the response DIRECTLY CONTRADICT any established persona attribute — "
        "age, social class, lifestyle, motivations, frustrations, credit context. "
        "Elaborating on persona traits or adding plausible conversational detail is NOT a contradiction.\n"
        "- factual_grounding: does the response make specific factual claims that are clearly false or "
        "impossible given the persona. General conversational statements are acceptable.\n"
        "- product_knowledge_accuracy: are all references to financial products factually wrong or "
        "described in a way that is misleading. Minor simplifications are acceptable.\n"
        "- emotional_register: is the tone grossly inconsistent with the persona profile — e.g. "
        "a frustrated persona expressing complete satisfaction without reason.\n"
        "- hallucination_flag: does the response introduce SPECIFIC verifiable facts that are clearly "
        "fabricated — exact income figures, named specific bank branches, specific policy numbers, "
        "or regulatory references. General lifestyle details and reasonable inferences are NOT hallucinations."
    )
    user = (
        f"Persona:\n{json.dumps(persona_data, indent=2, ensure_ascii=False)}\n\n"
        f"Artificial customer profile:\n{json.dumps(customer_profile, indent=2, ensure_ascii=False)}\n\n"
        f"Analyst question:\n{question}\n\n"
        f"Customer response:\n{response_text}"
    )
    raw = call_openai(client, MODEL_VALID, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.1, json_mode=True)
    return parse_json_safe(raw)


# ──────────────────────────────────────────────────────────
# Validation layer 2
# ──────────────────────────────────────────────────────────
def run_validation_layer2(client, cluster_profile, response_text):
    system = (
        "You are a statistical plausibility checker. Given a cluster statistical profile, "
        "INE Spain income benchmarks, and a synthetic customer response, check whether any "
        "HARD QUANTITATIVE claims are statistically impossible.\n\n"
        "IMPORTANT RULES:\n"
        "- Only flag SPECIFIC NUMERICAL claims that are clearly outside the cluster's statistical range "
        "(e.g. claiming to earn 5,000 EUR/month when the cluster median is 1,200).\n"
        "- Do NOT flag qualitative attitudes, opinions, or levels of interest in products. "
        "A dissatisfied customer can still express cautious interest when directly asked about a product. "
        "Low NPS or CSAT scores do NOT mean customers cannot consider or discuss financial products.\n"
        "- Do NOT flag general statements about preferences, concerns, or willingness to consider options.\n"
        "- When in doubt, return PLAUSIBLE.\n\n"
        "Return ONLY a JSON object with exactly these keys:\n"
        "- statistical_plausibility: 'PLAUSIBLE' or 'IMPLAUSIBLE'\n"
        "- implausible_claims: list of objects each with 'claim' and 'reason' (empty list if plausible)\n"
        "- ine_alignment: object with 'verdict' ('ALIGNED' or 'MISALIGNED') and 'reason' (one line) "
        "— only mark MISALIGNED if a specific income or spending figure contradicts INE benchmarks"
    )
    user = (
        f"Cluster profile:\n{json.dumps(cluster_profile, indent=2, ensure_ascii=False)}\n\n"
        f"INE benchmarks:\n{INE_CONTEXT_TEXT}\n\n"
        f"Customer response:\n{response_text}"
    )
    raw = call_openai(client, MODEL_VALID, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.1, json_mode=True)
    return parse_json_safe(raw)


# ──────────────────────────────────────────────────────────
# Confidence scoring
# ──────────────────────────────────────────────────────────
AMBER_KEYWORDS_INCOME = ["income", "salary", "earn", "wage", "sueldo", "ingreso", "nómina", "€", "eur"]
AMBER_KEYWORDS_FUTURE = ["will", "plan to", "going to", "intend", "expect to", "in the future", "next year"]


def compute_confidence(v1, v2, response_text, persona_data):
    # RED checks
    if v1 is None:
        return "AMBER"  # malformed layer 1 → default AMBER

    criteria = ["persona_consistency", "factual_grounding", "product_knowledge_accuracy",
                "emotional_register", "hallucination_flag"]
    any_fail = False
    for c in criteria:
        entry = v1.get(c, {})
        if isinstance(entry, dict) and entry.get("verdict", "").upper() == "FAIL":
            any_fail = True
            break
        elif isinstance(entry, str) and entry.upper() == "FAIL":
            any_fail = True
            break

    if any_fail:
        return "RED"

    if v2 is not None:
        sp = v2.get("statistical_plausibility", "").upper()
        if sp == "IMPLAUSIBLE":
            return "AMBER"  # statistical soft-signal → AMBER, not RED

    # AMBER checks
    resp_lower = response_text.lower()

    # References income / social class (INE_external fields)
    if any(kw in resp_lower for kw in AMBER_KEYWORDS_INCOME):
        return "AMBER"

    # Discusses motivations/emotions not in persona
    persona_motivations = " ".join(persona_data.get("motivations", [])).lower()
    persona_frustrations = " ".join(persona_data.get("frustrations", [])).lower()

    # References future financial behavior
    if any(kw in resp_lower for kw in AMBER_KEYWORDS_FUTURE):
        return "AMBER"

    # Check for emotional content not grounded in persona
    emotion_words = ["angry", "furious", "ecstatic", "thrilled", "devastated", "terrified",
                     "enfadado", "furioso", "encantado", "aterrorizado"]
    for ew in emotion_words:
        if ew in resp_lower and ew not in persona_motivations and ew not in persona_frustrations:
            return "AMBER"

    return "GREEN"


# ──────────────────────────────────────────────────────────
# Topic-relevance gate (cheap nano call to save tokens)
# ──────────────────────────────────────────────────────────
OFF_TOPIC_RESPONSE = (
    "I appreciate your curiosity, but I'm here to discuss my experience with "
    "financial products, banking services, and credit. Could you ask me something "
    "related to that instead?"
)


def check_topic_relevance(client_nano, question):
    """Return True if question is on-topic for market research of banking/financial products."""
    system = (
        "You are a strict topic classifier for a consumer finance market research interview. "
        "The ONLY acceptable topics are: banking products, credit, loans, insurance, payments, "
        "savings, financial habits, financial satisfaction, financial concerns, customer experience "
        "with financial providers, attitudes toward new financial product concepts, and personal "
        "financial situations as they relate to product usage.\n\n"
        "Reject anything unrelated: politics, sports, entertainment, coding, recipes, general "
        "knowledge, personal life unrelated to finances, medical advice, travel plans, etc.\n\n"
        "Return ONLY a JSON object: {\"relevant\": true} or {\"relevant\": false}"
    )
    raw = call_openai(client_nano, MODEL_VALID, [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ], temperature=0.0, json_mode=True)
    result = parse_json_safe(raw)
    if result is None:
        return True  # fail-open on malformed response
    return result.get("relevant", True)


# ──────────────────────────────────────────────────────────
# Chat turn handler
# ──────────────────────────────────────────────────────────
def process_turn(question):
    client = get_client()
    client_nano = get_client_nano()
    persona_data = PERSONAS[st.session_state.selected_persona]["data"]
    customer_profile = st.session_state.artificial_customer

    # 0. Topic-relevance gate — reject off-topic questions before expensive calls
    if not check_topic_relevance(client_nano, question):
        st.session_state.conversation_history.append({"role": "user", "content": question})
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": OFF_TOPIC_RESPONSE,
            "confidence": "AMBER",
            "validation_layer1": {},
            "validation_layer2": None,
            "v1_malformed": False,
            "v2_skipped": True,
            "off_topic": True,
        })
        st.session_state.turn_count += 1
        st.session_state.conf_counts["AMBER"] += 1
        return

    # Rolling summary update if needed
    maybe_update_rolling_summary(client_nano)

    # Build messages: system + last 50 turns + new question
    system_prompt = build_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]
    recent = st.session_state.conversation_history[-50:]
    for m in recent:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})

    # 1. Chat response
    response_text = call_openai(client, MODEL_CHAT, messages, temperature=0.7)

    # 2. First validation layer
    v1 = run_validation_layer1(client_nano, persona_data, customer_profile, question, response_text)
    v1_ok = v1 is not None
    if not v1_ok:
        v1 = {}

    # 3. Second validation layer
    cluster_id = PERSONA_CLUSTER_MAP.get(st.session_state.selected_persona)
    cluster_profile = CLUSTER_PROFILES.get(cluster_id, {}) if cluster_id else {}
    v2 = None
    v2_skipped = False
    if cluster_profile:
        v2 = run_validation_layer2(client_nano, cluster_profile, response_text)
        if v2 is None:
            v2_skipped = True

    # 4. Compute confidence
    confidence = compute_confidence(v1, v2, response_text, persona_data)

    # Store
    st.session_state.conversation_history.append({"role": "user", "content": question})
    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": response_text,
        "confidence": confidence,
        "validation_layer1": v1,
        "validation_layer2": v2,
        "v1_malformed": not v1_ok,
        "v2_skipped": v2_skipped,
    })
    st.session_state.turn_count += 1
    st.session_state.conf_counts[confidence] += 1


# ──────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────
def render_confidence_pill(conf):
    cls = {"GREEN": "conf-green", "AMBER": "conf-amber", "RED": "conf-red"}.get(conf, "conf-amber")
    return f'<span class="conf-pill {cls}">{conf}</span>'


def render_validation_detail(msg):
    v1 = msg.get("validation_layer1", {})
    v2 = msg.get("validation_layer2")
    lines = []

    criteria_labels = {
        "persona_consistency": "Persona consistency",
        "factual_grounding": "Factual grounding",
        "product_knowledge_accuracy": "Product knowledge",
        "emotional_register": "Emotional register",
        "hallucination_flag": "Hallucination check",
    }
    for key, label in criteria_labels.items():
        entry = v1.get(key, {})
        if isinstance(entry, dict):
            verdict = entry.get("verdict", "N/A")
            reason = entry.get("reason", "")
        else:
            verdict = str(entry)
            reason = ""
        css = "val-pass" if verdict.upper() == "PASS" else "val-fail"
        lines.append(f'<strong>{label}:</strong> <span class="{css}">{verdict}</span> {reason}')

    if msg.get("v1_malformed"):
        lines.append('<em>⚠ First validation returned malformed JSON — defaulted to AMBER</em>')

    if v2:
        sp = v2.get("statistical_plausibility", "N/A")
        css2 = "val-pass" if sp.upper() == "PLAUSIBLE" else "val-fail"
        lines.append(f'<strong>Statistical plausibility:</strong> <span class="{css2}">{sp}</span>')
        claims = v2.get("implausible_claims", [])
        for c in claims:
            if isinstance(c, dict):
                lines.append(f'  — {c.get("claim", "")}: {c.get("reason", "")}')
            else:
                lines.append(f'  — {c}')
        ine = v2.get("ine_alignment", {})
        if isinstance(ine, dict):
            iv = ine.get("verdict", "N/A")
            ir = ine.get("reason", "")
            css3 = "val-pass" if iv.upper() == "ALIGNED" else "val-fail"
            lines.append(f'<strong>INE alignment:</strong> <span class="{css3}">{iv}</span> {ir}')
    elif msg.get("v2_skipped"):
        lines.append('<em>⚠ Second-layer validation unavailable for this response</em>')

    return '<div class="val-detail">' + "<br>".join(lines) + "</div>"


# ──────────────────────────────────────────────────────────
# Export session
# ──────────────────────────────────────────────────────────
def export_session():
    pname = st.session_state.selected_persona or "unknown"
    safe_name = pname.replace(" ", "_").replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "persona": pname,
        "artificial_customer": st.session_state.artificial_customer,
        "turn_count": st.session_state.turn_count,
        "confidence_distribution": st.session_state.conf_counts,
        "conversation": st.session_state.conversation_history,
    }
    return f"synthpanel_session_{safe_name}_{ts}.json", json.dumps(payload, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════
# LAYOUT — Logo bar + Tabs
# ══════════════════════════════════════════════════════════
st.markdown(
    '<div class="logo-bar">'
    '<img src="https://upload.wikimedia.org/wikipedia/commons/a/a1/Logo_BNP_Paribas_Personal_Finance.png" alt="BNP Paribas PF"/>'
    '<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/CGI_Group_logo.svg/1280px-CGI_Group_logo.svg.png" alt="CGI Group"/>'
    '<img src="https://upload.wikimedia.org/wikipedia/commons/3/37/Logo-kedge.JPG?_=20140122143936" alt="KEDGE"/>'
    '</div>',
    unsafe_allow_html=True,
)
tab_interview, tab_heatmap = st.tabs(["💬 Interview", "🔥 Heatmap"])

# ──────────────── TAB: HEATMAP ────────────────
with tab_heatmap:
    if DASHBOARD_HEATMAP.exists():
        html_content = DASHBOARD_HEATMAP.read_text(encoding="utf-8")
        components.html(html_content, height=800, scrolling=True)
    else:
        st.info("Heatmap dashboard not found. Place synthpanel_heatmap.html in dashboards/.")

# ──────────────── TAB: INTERVIEW ────────────────
with tab_interview:
    left_col, right_col = st.columns([3, 7], gap="large")

    # ──────────────── LEFT COLUMN ────────────────
    with left_col:
        st.markdown(
            '<div class="synth-header"><h1>AI Azure Hackathon - SynthPanel</h1>'
            '<p>AI-powered synthetic customer interviews</p></div>',
            unsafe_allow_html=True,
        )

        # Persona selector
        persona_names = list(PERSONAS.keys())
        if not persona_names:
            st.warning("No persona files found in output/personas/")
            st.stop()

        selected = st.selectbox(
            "Select persona",
            persona_names,
            index=persona_names.index(st.session_state.selected_persona) if st.session_state.selected_persona in persona_names else 0,
            disabled=st.session_state.session_active,
        )
        st.session_state.selected_persona = selected

        # Generate Customer button
        gen_disabled = st.session_state.session_active
        if st.button("🎲  Generate Customer", disabled=gen_disabled, use_container_width=True):
            with st.spinner("Generating artificial customer..."):
                try:
                    persona_data = PERSONAS[selected]["data"]
                    ac = generate_artificial_customer(persona_data)
                    if ac:
                        st.session_state.artificial_customer = ac
                        st.session_state.session_active = True
                        st.session_state.conversation_history = []
                        st.session_state.rolling_summary = ""
                        st.session_state.summarized_up_to = 0
                        st.session_state.turn_count = 0
                        st.session_state.conf_counts = {"GREEN": 0, "AMBER": 0, "RED": 0}
                        st.rerun()
                    else:
                        st.error("Failed to parse generated customer profile. Try again.")
                except Exception as e:
                    st.error(f"Azure OpenAI error: {e}")

        # Customer card
        ac = st.session_state.artificial_customer
        if ac:
            st.markdown(
                f'<div class="customer-card">'
                f'<h4>{ac.get("full_name", "Unknown")}</h4>'
                f'<p><strong>Age:</strong> {ac.get("age", "?")}</p>'
                f'<p><strong>City:</strong> {ac.get("city", "?")}</p>'
                f'<p><strong>Gender:</strong> {ac.get("gender", "?")}</p>'
                f'<p style="margin-top:8px; font-style:italic; color:#aaa;">"{ac.get("personality_note", "")}"</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Session stats
        if st.session_state.session_active:
            cc = st.session_state.conf_counts
            st.markdown(
                f'<div class="stats-card">'
                f'<h4>Session Statistics</h4>'
                f'<p style="color: #f0f0f0;"><strong>Turns:</strong> {st.session_state.turn_count}</p>'
                f'<p><span class="conf-pill conf-green">GREEN {cc["GREEN"]}</span> '
                f'<span class="conf-pill conf-amber">AMBER {cc["AMBER"]}</span> '
                f'<span class="conf-pill conf-red">RED {cc["RED"]}</span></p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Export button (after 5 turns)
        if st.session_state.turn_count >= 5:
            fname, fdata = export_session()
            st.download_button(
                "📥  Export Session",
                data=fdata,
                file_name=fname,
                mime="application/json",
                use_container_width=True,
            )

        # Reset
        if st.session_state.session_active:
            if not st.session_state.confirm_reset:
                if st.button("🔄  Reset Session", use_container_width=True):
                    st.session_state.confirm_reset = True
                    st.rerun()
            else:
                st.markdown('<p style="color: black; background: #FFF3CD; padding: 10px 14px; border-radius: 6px; border: 1px solid #FFECB5; font-size: 0.9rem;">⚠️ Are you sure? This clears the entire session.</p>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, reset", use_container_width=True):
                        for k in list(st.session_state.keys()):
                            del st.session_state[k]
                        st.rerun()
                with c2:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state.confirm_reset = False
                        st.rerun()

    # ──────────────── RIGHT COLUMN ────────────────
    with right_col:
        # Header
        persona_label = st.session_state.selected_persona or "—"
        if st.session_state.session_active:
            status_html = '<span class="status-active">● Active session</span>'
        else:
            status_html = '<span class="status-idle">○ Idle — generate a customer to begin</span>'
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; align-items:center; '
            f'padding:8px 0; border-bottom:2px solid #3399FF; margin-bottom:12px;">'
            f'<strong style="color:#3399FF; font-size:1.1rem;">{persona_label}</strong>'
            f'{status_html}</div>',
            unsafe_allow_html=True,
        )

        # Chat history
        chat_container = st.container(height=520)
        with chat_container:
            if not st.session_state.session_active:
                st.markdown(
                    '<p style="color:#666; text-align:center; margin-top:200px;">'
                    'Select a persona and generate a customer to start an interview.</p>',
                    unsafe_allow_html=True,
                )
            else:
                # Show opening statement
                ac = st.session_state.artificial_customer
                if ac and ac.get("opening_statement"):
                    st.markdown(
                        f'<div class="customer-bubble green">'
                        f'<em>{ac["opening_statement"]}</em></div>',
                        unsafe_allow_html=True,
                    )

                for msg in st.session_state.conversation_history:
                    if msg["role"] == "user":
                        st.markdown(
                            f'<div class="analyst-bubble">{msg["content"]}</div>',
                            unsafe_allow_html=True,
                        )
                    elif msg["role"] == "assistant":
                        conf = msg.get("confidence", "AMBER")
                        conf_class = conf.lower()
                        pill = render_confidence_pill(conf)
                        st.markdown(
                            f'<div class="customer-bubble {conf_class}">'
                            f'{msg["content"]} {pill}</div>',
                            unsafe_allow_html=True,
                        )
                        # Expandable validation detail
                        with st.expander("Validation details", expanded=False):
                            st.markdown(render_validation_detail(msg), unsafe_allow_html=True)

        # Input
        if st.session_state.session_active:
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Your question",
                    placeholder="Type your interview question here…",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("Send", use_container_width=True)
                if submitted and user_input.strip():
                    with st.spinner("Generating response and running validation..."):
                        try:
                            process_turn(user_input.strip())
                        except Exception as e:
                            st.error(f"Error: {e}")
                    st.rerun()
