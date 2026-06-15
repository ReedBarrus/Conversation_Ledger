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


DB_PATH = Path("ledger.db")
LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
LMSTUDIO_URL = "http://localhost:1234/v1/embeddings"
REVERSAL_FLAG = "⚠ POSSIBLE REVERSAL"
POSITIVE_TYPES = {"belief", "conclusion", "inference", "supported", "decision"}
NEGATIVE_TYPES = {"falsified", "blocked"}


PATTERN_SPECS = [
    ("belief", re.compile(r"\bI think\b[^.!?\r\n]*", re.IGNORECASE)),
    ("decision", re.compile(r"\bwe decided\b[^.!?\r\n]*", re.IGNORECASE)),
    ("conclusion", re.compile(r"\bthe conclusion is\b[^.!?\r\n]*", re.IGNORECASE)),
    ("inference", re.compile(r"\bthis means\b[^.!?\r\n]*", re.IGNORECASE)),
    ("plan", re.compile(r"\bthe next move is\b[^.!?\r\n]*", re.IGNORECASE)),
    ("falsified", re.compile(r"\bfalsified\b(?:\s*:\s*|\s+)[^.!?\r\n]+", re.IGNORECASE)),
    ("supported", re.compile(r"\bsupported\b(?:\s*:\s*|\s+)[^.!?\r\n]+", re.IGNORECASE)),
    ("blocked", re.compile(r"\bblocked\b(?:\s*:\s*|\s+)[^.!?\r\n]+", re.IGNORECASE)),
    (
        "non_collapse",
        re.compile(r"\b[^.!?\r\n]*?(?:!=|≠|is not|does not imply)[^.!?\r\n]*", re.IGNORECASE),
    ),
]


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


def connect_db():
    conn = sqlite3.connect(DB_PATH)
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


def load_conversation_payloads(input_path):
    path = Path(input_path)
    if not path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".json")
        for file_path in files:
            yield from load_json_payload(file_path)
        return

    yield from load_json_payload(path)


def load_json_payload(file_path):
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(payload, dict):
        yield payload
        return

    raise SystemExit(f"Unsupported JSON shape in {file_path}")


def ingest_export(input_path):
    conn = connect_db()
    inserted_conversations = 0
    inserted_turns = 0

    with conn:
        for conversation in load_conversation_payloads(input_path):
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

    print(f"Ingested {inserted_conversations} conversations and {inserted_turns} turns into {DB_PATH}.")


def spans_overlap(candidate_start, candidate_end, accepted_spans):
    for start, end in accepted_spans:
        if candidate_start < end and candidate_end > start:
            return True
    return False


def build_claim_id(conv_id, turn_id, char_start, char_end, claim_text):
    raw = f"{conv_id}|{turn_id}|{char_start}|{char_end}|{claim_text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def extract_claims():
    conn = connect_db()
    turn_rows = conn.execute(
        """
        SELECT turn_id, conv_id, timestamp, text
        FROM turns
        ORDER BY COALESCE(timestamp, ''), conv_id, position, turn_id
        """
    ).fetchall()

    claims_to_insert = []
    for turn in turn_rows:
        accepted_spans = []
        turn_matches = []
        text = turn["text"] or ""
        for claim_type, pattern in PATTERN_SPECS:
            for match in pattern.finditer(text):
                char_start, char_end = match.span()
                if char_start == char_end:
                    continue
                if spans_overlap(char_start, char_end, accepted_spans):
                    continue
                claim_text = text[char_start:char_end]
                if not claim_text.strip():
                    continue
                accepted_spans.append((char_start, char_end))
                turn_matches.append(
                    (
                        char_start,
                        (
                            build_claim_id(turn["conv_id"], turn["turn_id"], char_start, char_end, claim_text),
                            turn["conv_id"],
                            turn["turn_id"],
                            char_start,
                            char_end,
                            claim_text,
                            claim_type,
                            "deterministic",
                            "candidate",
                            turn["timestamp"],
                            None,
                            None,
                        ),
                    )
                )
        turn_matches.sort(key=lambda item: item[0])
        claims_to_insert.extend(row for _, row in turn_matches)

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

    print(f"Extracted {len(claims_to_insert)} claims into {DB_PATH}.")


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


def cluster_claims(backend, threshold):
    conn = connect_db()
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


def view_clusters(min_count):
    conn = connect_db()
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


def build_parser():
    parser = argparse.ArgumentParser(description="Conversation Ledger")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("input_path")

    subparsers.add_parser("extract")

    cluster_parser = subparsers.add_parser("cluster")
    cluster_parser.add_argument("--backend", choices=["local", "lmstudio"], default="local")
    cluster_parser.add_argument("--threshold", type=float, default=0.75)

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
        ingest_export(args.input_path)
    elif args.command == "extract":
        extract_claims()
    elif args.command == "cluster":
        cluster_claims(args.backend, args.threshold)
    elif args.command == "view":
        view_clusters(args.min_count)
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
