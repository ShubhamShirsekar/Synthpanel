"""
SynthPanel Automated Test Script
Runs N interview turns per persona, validates confidence scoring,
and writes results to test_results.csv.
"""

import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
PERSONAS_DIR = Path(__file__).parent / "output" / "personas"
PROFILES_PATH = Path(__file__).parent / "output" / "cluster_profiles.json"
RESULTS_CSV = Path(__file__).parent / "test_results.csv"
TURNS_PER_PERSONA = 5  # number of interview questions per persona

# Test questions — designed to cross-reference both validation layers:
#   Q1: Financial situation probe → L1 persona_consistency + factual_grounding; L2 INE alignment
#   Q2: Provider relationship + emotional probe → L1 emotional_register + persona_consistency
#   Q3: Credit experience + product knowledge → L1 product_knowledge_accuracy + hallucination_flag
#   Q4: Specific product concept with numbers → L1 product_knowledge + L2 statistical plausibility
#   Q5: Concerns + future outlook → L1 persona_consistency + emotional_register; L2 income plausibility

TEST_QUESTIONS = [
    "Before we start talking about products, I'd like to understand your situation. "
    "Can you describe your current financial position — how you manage your monthly budget, "
    "what your main expenses are, and how much room you feel you have for any kind of credit repayment?",

    "How would you describe your relationship with your current financial provider? "
    "Do you feel they treat you fairly, understand your needs, and communicate transparently? "
    "What is the single biggest thing they could do to improve your experience?",

    "Walk me through your experience with credit so far — what products have you used, "
    "what went well, and what left you frustrated? If you haven't used credit products, "
    "what has held you back from doing so?",

    "We are testing a personal loan of up to 15,000 EUR with a fixed monthly instalment "
    "and no early repayment penalty. Given your financial situation, would you consider this, "
    "what monthly payment would feel manageable, and what would be your biggest concern before signing?",

    "Looking ahead, what are your main financial worries or goals for the next couple of years? "
    "Is there a product or service your bank doesn't offer today that would genuinely make "
    "your life easier, and what would make you trust a provider enough to take it up?",
]

INE_CONTEXT_TEXT = """Representative INE Spain median monthly income benchmarks (2023):
- 18-24 year-olds: €950–€1,150 depending on region
- 25-34 year-olds: €1,250–€1,650
- 35-44 year-olds: €1,350–€1,900
- 45-54 year-olds: €1,700–€2,100
- 55-64 year-olds: €1,400–€2,000
- 65+ year-olds:   €1,100–€1,400
Madrid and NORESTE regions tend 15-20% above national median; SUR and ISLAS CANARIAS 10-15% below."""

# ──────────────────────────────────────────────────────────
# Clients
# ──────────────────────────────────────────────────────────
def make_client():
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    )

def make_client_nano():
    return AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT_NANO", os.environ["AZURE_OPENAI_ENDPOINT"]),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY_NANO", os.environ["AZURE_OPENAI_API_KEY"]),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION_NANO",
                                   os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")),
    )

MODEL_CHAT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
MODEL_VALID = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NANO", "gpt-4.1-nano")

# ──────────────────────────────────────────────────────────
# Helpers
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
                time.sleep(2)
                continue
            raise e

def parse_json_safe(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


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
        return True
    return result.get("relevant", True)

# ──────────────────────────────────────────────────────────
# Load resources
# ──────────────────────────────────────────────────────────
def load_personas():
    personas = {}
    for fp in sorted(PERSONAS_DIR.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
            name = data.get("persona_name", fp.stem)
            # Extract cluster id from filename
            stem = fp.stem
            parts = stem.split("_")
            cluster_id = parts[1] if len(parts) >= 2 and parts[1].isdigit() else None
            personas[name] = {"file": fp.name, "data": data, "cluster_id": cluster_id}
    return personas

def load_cluster_profiles():
    if PROFILES_PATH.exists():
        with open(PROFILES_PATH) as f:
            return json.load(f)
    return {}

# ──────────────────────────────────────────────────────────
# Persona narrative
# ──────────────────────────────────────────────────────────
def build_persona_narrative(persona_data):
    d = persona_data.get("demographics", {})
    sc = d.get("social_class", {})
    sc_desc = sc.get("description", "") if isinstance(sc, dict) else str(sc)
    lines = [
        f"Age and generation: {d.get('age_and_generation', 'unknown')}.",
        f"Family situation: {d.get('family_situation', 'unknown')}.",
        f"Social class: {sc_desc}.",
        "", "Lifestyle and habits:",
    ]
    for h in persona_data.get("lifestyle_and_habits", []):
        lines.append(f"  - {h}")
    lines += ["", "Motivations:"]
    for m in persona_data.get("motivations", []):
        lines.append(f"  - {m}")
    lines += ["", "Frustrations:"]
    for fr in persona_data.get("frustrations", []):
        lines.append(f"  - {fr}")
    lines.append(f"\nNPS baseline: {persona_data.get('nps_baseline', 'unknown')}")
    lines.append(f"Credit context: {persona_data.get('credit_context', 'unknown')}")
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────
# Generate artificial customer
# ──────────────────────────────────────────────────────────
def generate_customer(client, persona_data):
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
    raw = call_openai(client, MODEL_CHAT, [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Persona:\n{json.dumps(persona_data, indent=2, ensure_ascii=False)}\n\nRandom seed (use this to vary your output): {seed}"},
    ], temperature=1.0, json_mode=True)
    return parse_json_safe(raw)

# ──────────────────────────────────────────────────────────
# Validation layer 1
# ──────────────────────────────────────────────────────────
def run_validation_layer1(client_nano, persona_data, customer_profile, question, response_text):
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
    raw = call_openai(client_nano, MODEL_VALID, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.1, json_mode=True)
    return parse_json_safe(raw)

# ──────────────────────────────────────────────────────────
# Validation layer 2
# ──────────────────────────────────────────────────────────
def run_validation_layer2(client_nano, cluster_profile, response_text):
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
    raw = call_openai(client_nano, MODEL_VALID, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.1, json_mode=True)
    return parse_json_safe(raw)

# ──────────────────────────────────────────────────────────
# Confidence scoring (mirrors synthpanel_app.py logic)
# ──────────────────────────────────────────────────────────
AMBER_KEYWORDS_INCOME = ["income", "salary", "earn", "wage", "sueldo", "ingreso", "nómina", "€", "eur"]
AMBER_KEYWORDS_FUTURE = ["will", "plan to", "going to", "intend", "expect to", "in the future", "next year"]

def compute_confidence(v1, v2, response_text, persona_data):
    if v1 is None:
        return "AMBER"

    criteria = ["persona_consistency", "factual_grounding", "product_knowledge_accuracy",
                "emotional_register", "hallucination_flag"]
    for c in criteria:
        entry = v1.get(c, {})
        if isinstance(entry, dict) and entry.get("verdict", "").upper() == "FAIL":
            return "RED"
        elif isinstance(entry, str) and entry.upper() == "FAIL":
            return "RED"

    if v2 is not None:
        if v2.get("statistical_plausibility", "").upper() == "IMPLAUSIBLE":
            return "AMBER"  # statistical soft-signal → AMBER, not RED

    resp_lower = response_text.lower()
    if any(kw in resp_lower for kw in AMBER_KEYWORDS_INCOME):
        return "AMBER"
    if any(kw in resp_lower for kw in AMBER_KEYWORDS_FUTURE):
        return "AMBER"

    emotion_words = ["angry", "furious", "ecstatic", "thrilled", "devastated", "terrified",
                     "enfadado", "furioso", "encantado", "aterrorizado"]
    persona_motivations = " ".join(persona_data.get("motivations", [])).lower()
    persona_frustrations = " ".join(persona_data.get("frustrations", [])).lower()
    for ew in emotion_words:
        if ew in resp_lower and ew not in persona_motivations and ew not in persona_frustrations:
            return "AMBER"

    return "GREEN"

# ──────────────────────────────────────────────────────────
# Extract individual verdicts for CSV columns
# ──────────────────────────────────────────────────────────
V1_CRITERIA = ["persona_consistency", "factual_grounding", "product_knowledge_accuracy",
               "emotional_register", "hallucination_flag"]

def extract_v1_verdicts(v1):
    results = {}
    for c in V1_CRITERIA:
        entry = v1.get(c, {}) if v1 else {}
        if isinstance(entry, dict):
            results[c + "_verdict"] = entry.get("verdict", "N/A")
            results[c + "_reason"] = entry.get("reason", "")
        else:
            results[c + "_verdict"] = str(entry)
            results[c + "_reason"] = ""
    return results

def extract_v2_verdicts(v2):
    if not v2:
        return {"stat_plausibility": "N/A", "ine_alignment": "N/A", "implausible_claims": ""}
    ine = v2.get("ine_alignment", {})
    claims = v2.get("implausible_claims", [])
    claims_str = "; ".join(
        f"{c.get('claim','')}: {c.get('reason','')}" if isinstance(c, dict) else str(c)
        for c in claims
    ) if claims else ""
    return {
        "stat_plausibility": v2.get("statistical_plausibility", "N/A"),
        "ine_alignment": ine.get("verdict", "N/A") if isinstance(ine, dict) else str(ine),
        "implausible_claims": claims_str,
    }

# ══════════════════════════════════════════════════════════
# Main test runner
# ══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("SynthPanel Automated Test")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    personas = load_personas()
    cluster_profiles = load_cluster_profiles()
    client = make_client()
    client_nano = make_client_nano()

    print(f"Loaded {len(personas)} personas, {len(cluster_profiles)} cluster profiles")
    print(f"Chat model: {MODEL_CHAT} | Validation model: {MODEL_VALID}")
    print(f"Turns per persona: {TURNS_PER_PERSONA}")
    print()

    # CSV setup
    fieldnames = [
        "timestamp", "persona_name", "cluster_id",
        "customer_name", "customer_age", "customer_city",
        "turn", "question", "response",
        "confidence",
        "persona_consistency_verdict", "persona_consistency_reason",
        "factual_grounding_verdict", "factual_grounding_reason",
        "product_knowledge_accuracy_verdict", "product_knowledge_accuracy_reason",
        "emotional_register_verdict", "emotional_register_reason",
        "hallucination_flag_verdict", "hallucination_flag_reason",
        "stat_plausibility", "ine_alignment", "implausible_claims",
        "v1_malformed", "v2_skipped",
        "response_time_sec",
    ]

    write_header = not RESULTS_CSV.exists()
    csv_file = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    total_turns = 0
    conf_totals = {"GREEN": 0, "AMBER": 0, "RED": 0}

    for persona_name, pinfo in personas.items():
        persona_data = pinfo["data"]
        cluster_id = pinfo["cluster_id"]

        print(f"\n{'─' * 50}")
        print(f"Persona: {persona_name} (cluster {cluster_id})")
        print(f"{'─' * 50}")

        # Generate artificial customer
        print("  Generating artificial customer...", end=" ", flush=True)
        try:
            ac = generate_customer(client, persona_data)
            if not ac:
                print("FAILED (malformed JSON)")
                continue
            print(f"→ {ac.get('full_name', '?')}, {ac.get('age', '?')}, {ac.get('city', '?')}")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Build system prompt
        narrative = build_persona_narrative(persona_data)
        system_prompt = (
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

        conversation = []
        cluster_profile = cluster_profiles.get(cluster_id, {}) if cluster_id else {}

        for turn_idx in range(TURNS_PER_PERSONA):
            question = TEST_QUESTIONS[turn_idx % len(TEST_QUESTIONS)]
            print(f"\n  Turn {turn_idx + 1}: {question[:60]}...")
            t0 = time.time()

            # 0. Topic-relevance gate
            if not check_topic_relevance(client_nano, question):
                print("    SKIPPED (off-topic)")
                conversation.append({"role": "user", "content": question})
                conversation.append({"role": "assistant", "content": OFF_TOPIC_RESPONSE})
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "persona_name": persona_name, "cluster_id": cluster_id,
                    "customer_name": ac.get("full_name", ""), "customer_age": ac.get("age", ""),
                    "customer_city": ac.get("city", ""),
                    "turn": turn_idx + 1, "question": question,
                    "response": OFF_TOPIC_RESPONSE, "confidence": "OFF_TOPIC",
                    **{c + "_verdict": "N/A" for c in V1_CRITERIA},
                    **{c + "_reason": "" for c in V1_CRITERIA},
                    "stat_plausibility": "N/A", "ine_alignment": "N/A", "implausible_claims": "",
                    "v1_malformed": False, "v2_skipped": True,
                    "response_time_sec": round(time.time() - t0, 2),
                }
                writer.writerow(row)
                csv_file.flush()
                total_turns += 1
                continue

            # 1. Chat response
            messages = [{"role": "system", "content": system_prompt}]
            for m in conversation[-50:]:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": question})

            try:
                response_text = call_openai(client, MODEL_CHAT, messages, temperature=0.7)
            except Exception as e:
                print(f"    CHAT ERROR: {e}")
                continue

            # 2. Validation layer 1
            v1_malformed = False
            try:
                v1 = run_validation_layer1(client_nano, persona_data, ac, question, response_text)
                if v1 is None:
                    v1_malformed = True
                    v1 = {}
            except Exception as e:
                print(f"    V1 ERROR: {e}")
                v1 = {}
                v1_malformed = True

            # 3. Validation layer 2
            v2 = None
            v2_skipped = False
            if cluster_profile:
                try:
                    v2 = run_validation_layer2(client_nano, cluster_profile, response_text)
                    if v2 is None:
                        v2_skipped = True
                except Exception as e:
                    print(f"    V2 ERROR: {e}")
                    v2_skipped = True

            # 4. Confidence
            confidence = compute_confidence(v1, v2, response_text, persona_data)
            elapsed = round(time.time() - t0, 2)

            conf_totals[confidence] += 1
            total_turns += 1

            # Store in conversation
            conversation.append({"role": "user", "content": question})
            conversation.append({"role": "assistant", "content": response_text})

            # Print result
            color = {"GREEN": "\033[92m", "AMBER": "\033[93m", "RED": "\033[91m"}.get(confidence, "")
            reset = "\033[0m"
            print(f"    Response: {response_text[:80]}...")
            print(f"    {color}Confidence: {confidence}{reset}  ({elapsed}s)")

            # Extract verdicts
            v1_data = extract_v1_verdicts(v1)
            v2_data = extract_v2_verdicts(v2)

            # Write CSV row
            row = {
                "timestamp": datetime.now().isoformat(),
                "persona_name": persona_name,
                "cluster_id": cluster_id,
                "customer_name": ac.get("full_name", ""),
                "customer_age": ac.get("age", ""),
                "customer_city": ac.get("city", ""),
                "turn": turn_idx + 1,
                "question": question,
                "response": response_text,
                "confidence": confidence,
                **v1_data,
                **v2_data,
                "v1_malformed": v1_malformed,
                "v2_skipped": v2_skipped,
                "response_time_sec": elapsed,
            }
            writer.writerow(row)
            csv_file.flush()

    csv_file.close()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total turns: {total_turns}")
    print(f"  GREEN: {conf_totals['GREEN']}  ({conf_totals['GREEN']/max(total_turns,1):.0%})")
    print(f"  AMBER: {conf_totals['AMBER']}  ({conf_totals['AMBER']/max(total_turns,1):.0%})")
    print(f"  RED:   {conf_totals['RED']}  ({conf_totals['RED']/max(total_turns,1):.0%})")
    print(f"\nResults saved to: {RESULTS_CSV.resolve()}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
