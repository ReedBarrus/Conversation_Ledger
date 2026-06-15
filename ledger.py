import argparse
import hashlib
import json
import math
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path
import re


DEFAULT_DB_PATH = Path("ledger.db")
LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
LMSTUDIO_URL = "http://localhost:1234/v1/embeddings"
DEFAULT_LLM_URL = "http://10.2.0.2:1234/v1/chat/completions"
DEFAULT_LLM_MODEL = "microsoft/phi-4"
REVERSAL_FLAG = "⚠ POSSIBLE REVERSAL"
POSITIVE_TYPES = {"belief", "conclusion", "inference", "supported", "decision"}
NEGATIVE_TYPES = {"falsified", "blocked"}
LLM_CLAIM_TYPES = {
    "belief",
    "decision",
    "conclusion",
    "inference",
    "plan",
    "non_collapse",
    "falsified",
    "supported",
    "blocked",
    "other",
}
SAMPLE_CONVERSATION_IDS = {"conv-alpha", "conv-beta", "conv-gamma"}
SAMPLE_CONVERSATION_TITLES = {"Alpha Stability", "Beta Support", "Gamma Reversal"}
LLM_EXTRACTION_INSTRUCTIONS = """You extract asserted claims from one message in a conversation.
A claim is a statement the speaker asserts as true, decided, concluded,
rejected, or planned - a position they are taking, not a question or aside.

RULES:
- Quote each claim VERBATIM from the message text. Do not paraphrase,
  summarize, rewrite, or correct. Use the speaker's exact words.
- If a claim spans a clause, quote the clause exactly as written.
- Extract only what is actually asserted in THIS message. Do not infer,
  do not add context, do not combine with outside knowledge.
- If the message asserts nothing (a question, a greeting, pure acknowledgment),
  return an empty list.
- Do not judge, score, or rate claims. Only extract them.
- Classify each as one of: belief, decision, conclusion, inference, plan,
  non_collapse, falsified, supported, blocked, other.

Return ONLY a JSON array, no prose:
[{"claim_text": "<verbatim quote>", "claim_type": "<type>"}]"""


SIMPLE_CLAUSE_SPECS = [
    ("belief", re.compile(r"\bI think\b\s+.+?(?=[.!?](?:\s|$)|$)", re.IGNORECASE)),
    ("decision", re.compile(r"\bwe decided\b\s+.+?(?=[.!?](?:\s|$)|$)", re.IGNORECASE)),
    ("conclusion", re.compile(r"\bthe conclusion is\b\s+.+?(?=[.!?](?:\s|$)|$)", re.IGNORECASE)),
    ("inference", re.compile(r"\bthis means\b\s+.+?(?=[.!?](?:\s|$)|$)", re.IGNORECASE)),
    ("plan", re.compile(r"\bthe next move is\b\s+.+?(?=[.!?](?:\s|$)|$)", re.IGNORECASE)),
]
CLAIM_LABEL_LINE_PATTERN = re.compile(
    r"(?mi)^(?P<label>supported|falsified|blocked|verdict|finding|conclusion)\s*:\s*(?P<body>.+)$"
)
LINE_INITIAL_STATUS_PATTERN = re.compile(
    r"(?mi)^(?P<label>supported|falsified|blocked)\b(?!\s*:)\s+(?P<body>.+)$"
)
NON_COLLAPSE_OPERATOR_PATTERN = re.compile(
    r"(?P<claim>[A-Z][\w /-]{1,40}?\s*(?:!=|≠)\s*[\w /-]{1,40}?)(?=[.,;!?]|$)"
)
SHORT_LEFT_OPERAND = r"[A-Z][\w/-]+(?: [\w/-]+){0,4}"
SHORT_RIGHT_OPERAND = r"(?:[A-Za-z][\w/-]*)(?: [A-Za-z][\w/-]*){0,4}"
DOES_NOT_IMPLY_PATTERN = re.compile(
    rf"(?P<claim>(?:(?<=^)|(?<=[.!?]\s)|(?<=\n)){SHORT_LEFT_OPERAND}\s+does not imply\s+{SHORT_RIGHT_OPERAND})(?=[.,;!?]|$)",
    re.MULTILINE,
)
IS_NOT_CLAUSE_PATTERN = re.compile(
    rf"(?P<claim>(?:(?<=^)|(?<=[.!?]\s)|(?<=\n)){SHORT_LEFT_OPERAND}\s+is not\s+{SHORT_RIGHT_OPERAND})(?=[.,;]|$)",
    re.MULTILINE,
)
SHOULD_NOT_PATTERN = re.compile(
    rf"(?P<claim>(?:(?<=^)|(?<=[.!?]\s)|(?<=\n)){SHORT_LEFT_OPERAND}\s+should not be\s+(?:treated as|confused with)\s+{SHORT_RIGHT_OPERAND})(?=[.,;]|$)",
    re.MULTILINE | re.IGNORECASE,
)
CLAIM_LABEL_TYPE_MAP = {
    "supported": "supported",
    "falsified": "falsified",
    "blocked": "blocked",
    "conclusion": "conclusion",
    "verdict": "other",
    "finding": "other",
}


def first_nonempty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        return value
    return None


def normalize_db_path(db_path):
    return Path(db_path)


def connect_db(db_path):
    resolved_db_path = normalize_db_path(db_path)
    conn = sqlite3.connect(resolved_db_path)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations(
            conv_id TEXT PRIMARY KEY,
            title TEXT,
            source_app TEXT,
            started_at TEXT,
            turn_count INTEGER
        );
        CREATE TABLE IF NOT EXISTS turns(
            turn_id TEXT PRIMARY KEY,
            conv_id TEXT,
            role TEXT,
            position INTEGER,
            timestamp TEXT,
            text TEXT
        );
        CREATE TABLE IF NOT EXISTS claims(
            claim_id TEXT PRIMARY KEY,
            conv_id TEXT,
            turn_id TEXT,
            char_start INTEGER,
            char_end INTEGER,
            claim_text TEXT,
            claim_type TEXT,
            extraction_method TEXT,
            status TEXT,
            timestamp TEXT,
            cluster_id INTEGER,
            cluster_similarity REAL
        );
        """
    )
    conn.commit()


def reset_schema(conn):
    conn.executescript(
        """
        DROP TABLE IF EXISTS claims;
        DROP TABLE IF EXISTS turns;
        DROP TABLE IF EXISTS conversations;
        """
    )
    ensure_schema(conn)


def stable_fallback_id(prefix, payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{prefix}-{hashlib.sha1(encoded).hexdigest()[:12]}"


def normalize_role(raw_role):
    role = (raw_role or "unknown").strip().lower()
    if role == "human":
        return "user"
    if role == "assistant":
        return "assistant"
    return role or "unknown"


def extract_message_text(message):
    direct_text = message.get("text")
    if isinstance(direct_text, str):
        return direct_text

    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "".join(parts)
    return ""


def is_conversation_payload(payload):
    if not isinstance(payload, dict):
        return False

    messages = payload.get("chat_messages")
    if not isinstance(messages, list):
        messages = payload.get("messages")
    if not isinstance(messages, list):
        return False

    return any(field in payload for field in ("uuid", "id", "name", "title", "created_at", "updated_at"))


def iter_json_files(input_path):
    path = Path(input_path)
    if not path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    if path.is_dir():
        for file_path in sorted(p for p in path.rglob("*.json") if p.is_file()):
            yield file_path
        return

    yield path


def load_json_payload(file_path):
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        print(f"Skipping unreadable JSON: {file_path}")
        return []

    if isinstance(payload, list):
        return [item for item in payload if is_conversation_payload(item)]

    if is_conversation_payload(payload):
        return [payload]

    return []


def collect_conversation_payloads(input_path):
    payloads = []
    scanned_file_count = 0

    for file_path in iter_json_files(input_path):
        scanned_file_count += 1
        payloads.extend(load_json_payload(file_path))

    return payloads, scanned_file_count, len(payloads)


def ingest_export(input_path, db_path, reset=False):
    conn = connect_db(db_path)
    resolved_db_path = normalize_db_path(db_path)
    inserted_conversations = 0
    inserted_turns = 0
    payloads, scanned_file_count, accepted_payload_count = collect_conversation_payloads(input_path)

    with conn:
        if reset:
            reset_schema(conn)
        for conversation in payloads:
            conv_id = first_nonempty(
                conversation.get("uuid"),
                conversation.get("id"),
                stable_fallback_id("conv", conversation),
            )
            title = first_nonempty(conversation.get("name"), conversation.get("title"), conv_id)
            messages = conversation.get("chat_messages") or conversation.get("messages") or []
            turn_rows = []

            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue
                turn_text = extract_message_text(message)
                if not turn_text:
                    continue
                turn_id = first_nonempty(
                    message.get("uuid"),
                    message.get("id"),
                    f"{conv_id}-turn-{len(turn_rows)}",
                )
                turn_rows.append(
                    (
                        turn_id,
                        conv_id,
                        normalize_role(message.get("sender") or message.get("role")),
                        len(turn_rows),
                        first_nonempty(message.get("created_at"), message.get("timestamp")),
                        turn_text,
                    )
                )

            started_at = first_nonempty(
                conversation.get("created_at"),
                turn_rows[0][4] if turn_rows else None,
            )
            conn.execute("DELETE FROM turns WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM claims WHERE conv_id = ?", (conv_id,))
            conn.execute(
                """
                INSERT OR REPLACE INTO conversations(conv_id, title, source_app, started_at, turn_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conv_id, title, "claude", started_at, len(turn_rows)),
            )
            conn.executemany(
                """
                INSERT INTO turns(turn_id, conv_id, role, position, timestamp, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                turn_rows,
            )
            inserted_conversations += 1
            inserted_turns += len(turn_rows)

    print(f"Scanned {scanned_file_count} JSON files.")
    print(f"Accepted {accepted_payload_count} conversation payloads.")
    print(f"Ingested {inserted_conversations} conversations and {inserted_turns} turns into {resolved_db_path}.")


def spans_overlap(candidate_start, candidate_end, accepted_spans):
    for start, end in accepted_spans:
        if candidate_start < end and candidate_end > start:
            return True
    return False


def build_claim_id(conv_id, turn_id, char_start, char_end, claim_text, extraction_method, occurrence_index=0):
    raw = f"{conv_id}|{turn_id}|{char_start}|{char_end}|{claim_text}|{extraction_method}|{occurrence_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_claim_row(turn, char_start, char_end, claim_text, claim_type, extraction_method, occurrence_index=0):
    return (
        build_claim_id(
            turn["conv_id"],
            turn["turn_id"],
            char_start,
            char_end,
            claim_text,
            extraction_method,
            occurrence_index,
        ),
        turn["conv_id"],
        turn["turn_id"],
        char_start,
        char_end,
        claim_text,
        claim_type,
        extraction_method,
        "candidate",
        turn["timestamp"],
        None,
        None,
    )


def match_start_for_group(match, group_name):
    return match.start(group_name)


def match_end_for_group(match, group_name):
    return match.end(group_name)


def append_deterministic_claim(turn_matches, turn, char_start, char_end, claim_type):
    if char_start == char_end:
        return
    claim_text = (turn["text"] or "")[char_start:char_end]
    if not claim_text.strip():
        return
    turn_matches.append(
        (
            char_start,
            build_claim_row(
                turn,
                char_start,
                char_end,
                claim_text,
                claim_type,
                "deterministic",
            ),
        )
    )


def extract_deterministic_claims_for_turn(turn):
    accepted_spans = []
    turn_matches = []
    text = turn["text"] or ""

    for claim_type, pattern in SIMPLE_CLAUSE_SPECS:
        for match in pattern.finditer(text):
            char_start, char_end = match.span()
            if char_start == char_end:
                continue
            if spans_overlap(char_start, char_end, accepted_spans):
                continue
            accepted_spans.append((char_start, char_end))
            append_deterministic_claim(turn_matches, turn, char_start, char_end, claim_type)

    for pattern in (NON_COLLAPSE_OPERATOR_PATTERN, DOES_NOT_IMPLY_PATTERN, IS_NOT_CLAUSE_PATTERN, SHOULD_NOT_PATTERN):
        for match in pattern.finditer(text):
            char_start = match_start_for_group(match, "claim")
            char_end = match_end_for_group(match, "claim")
            if spans_overlap(char_start, char_end, accepted_spans):
                continue
            accepted_spans.append((char_start, char_end))
            append_deterministic_claim(turn_matches, turn, char_start, char_end, "non_collapse")

    for match in CLAIM_LABEL_LINE_PATTERN.finditer(text):
        char_start, char_end = match.span()
        if spans_overlap(char_start, char_end, accepted_spans):
            continue
        accepted_spans.append((char_start, char_end))
        claim_type = CLAIM_LABEL_TYPE_MAP.get(match.group("label").lower(), "other")
        append_deterministic_claim(turn_matches, turn, char_start, char_end, claim_type)

    for match in LINE_INITIAL_STATUS_PATTERN.finditer(text):
        char_start, char_end = match.span()
        if spans_overlap(char_start, char_end, accepted_spans):
            continue
        accepted_spans.append((char_start, char_end))
        claim_type = CLAIM_LABEL_TYPE_MAP.get(match.group("label").lower(), "other")
        append_deterministic_claim(turn_matches, turn, char_start, char_end, claim_type)

    turn_matches.sort(key=lambda item: item[0])
    return [row for _, row in turn_matches]


def strip_markdown_fences(text):
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return stripped


def parse_json_array_response(text):
    stripped = strip_markdown_fences(text)
    candidates = [stripped]
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end >= start:
        bracketed = stripped[start : end + 1]
        if bracketed not in candidates:
            candidates.append(bracketed)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def find_verbatim_span(source_text, claim_text):
    match = re.search(re.escape(claim_text), source_text, re.IGNORECASE)
    if match is None:
        return None
    return match.span()


def normalize_llm_claim_type(raw_claim_type):
    claim_type = (raw_claim_type or "other").strip().lower()
    if claim_type not in LLM_CLAIM_TYPES:
        return "other"
    return claim_type


def extract_content_from_chat_response(body):
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        if text_parts:
            return "".join(text_parts)
    return None


def extract_llm_claims(turn_rows, llm_url, llm_model):
    try:
        import requests
    except ImportError as exc:
        raise SystemExit(
            "requests is required for `extract --method llm|both`. Install dependencies from requirements.txt."
        ) from exc

    llm_rows = []
    turns_with_claims = set()
    dropped_non_verbatim = 0

    for turn in turn_rows:
        payload = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": LLM_EXTRACTION_INSTRUCTIONS},
                {"role": "user", "content": f"Message text:\n{turn['text'] or ''}"},
            ],
        }

        try:
            response = requests.post(llm_url, json=payload, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"LLM extraction skipped: unable to reach {llm_url} ({exc})")
            return [], set(), 0

        try:
            body = response.json()
        except ValueError:
            print(f"Skipping LLM claims for turn_id={turn['turn_id']}: response was not valid JSON.")
            continue

        content = extract_content_from_chat_response(body)
        if not isinstance(content, str):
            print(f"Skipping LLM claims for turn_id={turn['turn_id']}: response missing message content.")
            continue

        parsed_claims = parse_json_array_response(content)
        if parsed_claims is None:
            print(f"Skipping LLM claims for turn_id={turn['turn_id']}: could not parse JSON array.")
            continue

        accepted_rows = []
        for occurrence_index, item in enumerate(parsed_claims):
            if not isinstance(item, dict):
                continue
            claim_text = item.get("claim_text")
            if not isinstance(claim_text, str) or not claim_text.strip():
                continue

            claim_type = normalize_llm_claim_type(item.get("claim_type"))
            span = find_verbatim_span(turn["text"] or "", claim_text)
            if span is None:
                dropped_non_verbatim += 1
                continue

            char_start, char_end = span
            accepted_rows.append(
                build_claim_row(
                    turn,
                    char_start,
                    char_end,
                    (turn["text"] or "")[char_start:char_end],
                    claim_type,
                    "llm",
                    occurrence_index,
                )
            )

        if accepted_rows:
            turns_with_claims.add(turn["turn_id"])
            llm_rows.extend(accepted_rows)

    return llm_rows, turns_with_claims, dropped_non_verbatim


def extract_claims(db_path, method, llm_url, llm_model):
    conn = connect_db(db_path)
    resolved_db_path = normalize_db_path(db_path)
    turn_rows = conn.execute(
        """
        SELECT turn_id, conv_id, timestamp, text
        FROM turns
        ORDER BY COALESCE(timestamp, ''), conv_id, position, turn_id
        """
    ).fetchall()

    deterministic_rows = []
    deterministic_turns = set()
    if method in {"deterministic", "both"}:
        for turn in turn_rows:
            rows = extract_deterministic_claims_for_turn(turn)
            if rows:
                deterministic_rows.extend(rows)
                deterministic_turns.add(turn["turn_id"])

    llm_rows = []
    llm_turns = set()
    dropped_non_verbatim = 0
    if method in {"llm", "both"}:
        llm_rows, llm_turns, dropped_non_verbatim = extract_llm_claims(turn_rows, llm_url, llm_model)

    claims_to_insert = deterministic_rows + llm_rows

    with conn:
        conn.execute("DELETE FROM claims")
        conn.executemany(
            """
            INSERT INTO claims(
                claim_id, conv_id, turn_id, char_start, char_end,
                claim_text, claim_type, extraction_method, status, timestamp,
                cluster_id, cluster_similarity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            claims_to_insert,
        )

    print(f"Extracted {len(claims_to_insert)} claims into {resolved_db_path}.")
    print(f"deterministic claims: {len(deterministic_rows)}")
    print(f"llm claims: {len(llm_rows)}")
    print(f"llm dropped (non-verbatim): {dropped_non_verbatim}")
    print(f"turns with claims (det): {len(deterministic_turns)} / {len(turn_rows)}")
    print(f"turns with claims (llm): {len(llm_turns)} / {len(turn_rows)}")


def embed(texts, backend):
    if backend == "local":
        return embed_local(texts)
    if backend == "lmstudio":
        return embed_lmstudio(texts)
    raise SystemExit(f"Unsupported backend: {backend}")


def embed_local(texts):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required for --backend local. Install it with: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(LOCAL_MODEL_NAME, device="cpu")
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def embed_lmstudio(texts):
    payload = json.dumps({"model": LOCAL_MODEL_NAME, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        LMSTUDIO_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"LM Studio embeddings request failed: {exc}") from exc

    data = body.get("data")
    if not isinstance(data, list):
        raise SystemExit("LM Studio response did not contain a data list.")

    ordered = sorted(data, key=lambda item: item.get("index", 0))
    vectors = [item.get("embedding") for item in ordered]
    if any(not isinstance(vector, list) for vector in vectors):
        raise SystemExit("LM Studio response did not contain embeddings in every data item.")
    return vectors


def cosine_similarity(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def comparison_text_for_embedding(claim_type, claim_text):
    text = claim_text.strip()
    if claim_type == "belief":
        core = re.sub(r"(?i)^I think\b\s*", "", text, count=1)
    elif claim_type == "decision":
        core = re.sub(r"(?i)^we decided\b\s*", "", text, count=1)
    elif claim_type == "conclusion":
        core = re.sub(r"(?i)^the conclusion is\b\s*", "", text, count=1)
    elif claim_type == "inference":
        core = re.sub(r"(?i)^this means\b\s*", "", text, count=1)
    elif claim_type == "plan":
        core = re.sub(r"(?i)^the next move is\b\s*", "", text, count=1)
    elif claim_type in {"falsified", "supported", "blocked"}:
        core = re.sub(rf"(?i)^{claim_type}\b(?:\s*:\s*|\s+)", "", text, count=1)
    else:
        core = text
    core = core.strip()
    return core or text


def cluster_claims(db_path, backend, threshold):
    conn = connect_db(db_path)
    claim_rows = conn.execute(
        """
        SELECT claim_id, claim_text, claim_type, timestamp
        FROM claims
        ORDER BY COALESCE(timestamp, ''), claim_id
        """
    ).fetchall()
    if not claim_rows:
        print("No claims found. Run `extract` first.")
        return

    texts = [comparison_text_for_embedding(row["claim_type"], row["claim_text"]) for row in claim_rows]
    vectors = embed(texts, backend)
    clusters = []
    assignments = []

    for row, vector in zip(claim_rows, vectors):
        best_cluster = None
        best_score = -1.0
        for cluster in clusters:
            score = cosine_similarity(vector, cluster["seed_vector"])
            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster is not None and best_score >= threshold:
            cluster_id = best_cluster["cluster_id"]
            cluster_similarity = best_score
        else:
            cluster_id = len(clusters) + 1
            cluster_similarity = 1.0
            clusters.append({"cluster_id": cluster_id, "seed_vector": vector})

        assignments.append((cluster_id, cluster_similarity, row["claim_id"]))
        print(
            f'cluster={cluster_id} similarity={cluster_similarity:.3f} claim_id={row["claim_id"]} text="{row["claim_text"]}"'
        )

    with conn:
        conn.executemany(
            "UPDATE claims SET cluster_id = ?, cluster_similarity = ? WHERE claim_id = ?",
            assignments,
        )

    print(f"Clustered {len(assignments)} claims with backend={backend} threshold={threshold:.2f}.")


def negation_tokens(text):
    lowered = text.lower()
    tokens = set()
    if "!=" in text:
        tokens.add("!=")
    if "≠" in text:
        tokens.add("≠")
    if "n't" in lowered:
        tokens.add("n't")
    if re.search(r"\bnot\b", lowered):
        tokens.add("not")
    return tokens


def timestamp_key(value):
    return value or ""


def format_claim_line(claim):
    title = claim["title"] or claim["conv_id"]
    timestamp = claim["timestamp"] or "unknown-time"
    return (
        f'[{timestamp}] ({claim["claim_type"]}) "{claim["claim_text"]}" '
        f'<{title}|{claim["conv_id"]}|{claim["turn_id"]}>'
    )


def find_reversal(claims):
    ordered = sorted(claims, key=lambda claim: (timestamp_key(claim["timestamp"]), claim["claim_id"]))

    for index, earlier in enumerate(ordered):
        for later in ordered[index + 1 :]:
            if timestamp_key(later["timestamp"]) <= timestamp_key(earlier["timestamp"]):
                continue
            earlier_positive = earlier["claim_type"] in POSITIVE_TYPES
            earlier_negative = earlier["claim_type"] in NEGATIVE_TYPES
            later_positive = later["claim_type"] in POSITIVE_TYPES
            later_negative = later["claim_type"] in NEGATIVE_TYPES
            if (earlier_positive and later_negative) or (earlier_negative and later_positive):
                return ("typed-polarity", earlier, later)

    for index, earlier in enumerate(ordered):
        earlier_negations = negation_tokens(earlier["claim_text"])
        for later in ordered[index + 1 :]:
            if timestamp_key(later["timestamp"]) <= timestamp_key(earlier["timestamp"]):
                continue
            later_negations = negation_tokens(later["claim_text"])
            if later_negations and later_negations - earlier_negations:
                return ("explicit-negation", earlier, later)

    return None


def view_clusters(db_path, min_count):
    conn = connect_db(db_path)
    claim_rows = conn.execute(
        """
        SELECT
            claims.claim_id,
            claims.conv_id,
            claims.turn_id,
            claims.claim_text,
            claims.claim_type,
            claims.timestamp,
            claims.cluster_id,
            conversations.title
        FROM claims
        LEFT JOIN conversations ON conversations.conv_id = claims.conv_id
        ORDER BY COALESCE(claims.cluster_id, 0), COALESCE(claims.timestamp, ''), claims.claim_id
        """
    ).fetchall()

    if not claim_rows:
        print("No claims found. Run `extract` first.")
        return

    grouped = {}
    for row in claim_rows:
        cluster_key = row["cluster_id"] if row["cluster_id"] is not None else f"singleton:{row['claim_id']}"
        grouped.setdefault(cluster_key, []).append(row)

    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: (-len(group), timestamp_key(group[0]["timestamp"]), str(group[0]["cluster_id"])),
    )

    displayed = 0
    for group in ordered_groups:
        if len(group) < min_count:
            continue
        displayed += 1
        ordered_claims = sorted(group, key=lambda claim: (timestamp_key(claim["timestamp"]), claim["claim_id"]))
        representative = ordered_claims[0]
        first_seen = ordered_claims[0]["timestamp"] or "unknown-time"
        last_seen = ordered_claims[-1]["timestamp"] or "unknown-time"
        cluster_label = representative["cluster_id"] if representative["cluster_id"] is not None else "unclustered"

        print(f"=== Cluster {cluster_label} ===")
        print(f'Representative: "{representative["claim_text"]}"')
        print(f"Count: {len(ordered_claims)}")
        print(f"First: {first_seen}")
        print(f"Last: {last_seen}")

        reversal = find_reversal(ordered_claims)
        if reversal is not None:
            _, earlier, later = reversal
            print(f"{REVERSAL_FLAG} (heuristic signal only)")
            print(f"Earlier: {format_claim_line(earlier)}")
            print(f"Later:   {format_claim_line(later)}")

        for claim in ordered_claims:
            print(format_claim_line(claim))
        print()

    if displayed == 0:
        print(f"No clusters matched min-count={min_count}.")


def stats_db(db_path):
    conn = connect_db(db_path)
    resolved_db_path = normalize_db_path(db_path)

    conversation_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    turn_count = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    claim_count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    clustered_claim_count = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE cluster_id IS NOT NULL"
    ).fetchone()[0]
    distinct_cluster_count = conn.execute(
        "SELECT COUNT(DISTINCT cluster_id) FROM claims WHERE cluster_id IS NOT NULL"
    ).fetchone()[0]
    claim_counts_by_type = conn.execute(
        """
        SELECT claim_type, COUNT(*) AS claim_count
        FROM claims
        GROUP BY claim_type
        ORDER BY claim_count DESC, claim_type ASC
        """
    ).fetchall()
    top_conversations = conn.execute(
        """
        SELECT
            conversations.title,
            conversations.conv_id,
            COUNT(claims.claim_id) AS claim_count
        FROM conversations
        LEFT JOIN claims ON claims.conv_id = conversations.conv_id
        GROUP BY conversations.conv_id, conversations.title
        ORDER BY claim_count DESC, conversations.title ASC, conversations.conv_id ASC
        LIMIT 10
        """
    ).fetchall()
    conversation_rows = conn.execute("SELECT conv_id, title FROM conversations ORDER BY conv_id").fetchall()

    print(f"database path: {resolved_db_path}")
    print(f"conversation count: {conversation_count}")
    print(f"turn count: {turn_count}")
    print(f"claim count: {claim_count}")
    print(f"clustered claim count: {clustered_claim_count}")
    print(f"distinct cluster count: {distinct_cluster_count}")
    print("claim counts by type:")
    if claim_counts_by_type:
        for row in claim_counts_by_type:
            print(f"  {row['claim_type']}: {row['claim_count']}")
    else:
        print("  (none)")

    print("top 10 conversations by claim count:")
    if top_conversations:
        for row in top_conversations:
            title = row["title"] or row["conv_id"]
            print(f"  {row['claim_count']}: {title} <{row['conv_id']}>")
    else:
        print("  (none)")

    if conversation_count > 0 and claim_count == 0:
        print("warning: conversations > 0 but claims == 0")
    if claim_count > 0 and clustered_claim_count == 0:
        print("warning: claims > 0 but clustered_claims == 0")
    if conversation_count > 0 and all(
        row["conv_id"] in SAMPLE_CONVERSATION_IDS or (row["title"] or "") in SAMPLE_CONVERSATION_TITLES
        for row in conversation_rows
    ):
        print("warning: only sample_export conversations appear")


def build_parser():
    parser = argparse.ArgumentParser(description="Conversation Ledger")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("input_path")
    ingest_parser.add_argument("--reset", action="store_true")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--method", choices=["deterministic", "llm", "both"], default="deterministic")
    extract_parser.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    extract_parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)

    cluster_parser = subparsers.add_parser("cluster")
    cluster_parser.add_argument("--backend", choices=["local", "lmstudio"], default="local")
    cluster_parser.add_argument("--threshold", type=float, default=0.75)

    subparsers.add_parser("stats")

    view_parser = subparsers.add_parser("view")
    view_parser.add_argument("--min-count", type=int, default=1)

    return parser


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        ingest_export(args.input_path, args.db, reset=args.reset)
    elif args.command == "extract":
        extract_claims(args.db, args.method, args.llm_url, args.llm_model)
    elif args.command == "cluster":
        cluster_claims(args.db, args.backend, args.threshold)
    elif args.command == "stats":
        stats_db(args.db)
    elif args.command == "view":
        view_clusters(args.db, args.min_count)
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
