import json
import re
from pathlib import Path


SOURCE = Path("dataset.jsonl")
CLEANED = Path("dataset.clean.jsonl")

MIN_REPEAT_WORDS = 30  # Shortest run of repeated words worth removing.
MIN_VALUE_CHARS = 25   # Spec values shorter than this are kept even when repeated.
LIST_ASINS = 50        # How many changed ASINs to name in the report.

SPEC_PAIR = re.compile(r"([^:;]{1,45}):\s*(.+)", re.DOTALL)
WORDS = re.compile(r"[a-z0-9]+")


def normalise(text):
    """Lowercased, punctuation-free form used only for comparison, never for output."""
    return " ".join(WORDS.findall(text.lower()))


def drop_repeated_specs(text):
    """Drop a spec pair whose value already appeared, under this key or a prefix of it."""
    kept, seen = [], {}
    for piece in text.split("; "):
        spec = SPEC_PAIR.fullmatch(piece.strip())
        if spec:
            key, value = normalise(spec.group(1)), normalise(spec.group(2))
            first = seen.get(value)
            if first is not None and (len(value) >= MIN_VALUE_CHARS
                                      or key.startswith(first) or first.startswith(key)):
                continue
            seen.setdefault(value, key)
        kept.append(piece)
    return "; ".join(kept)


def drop_repeated_runs(text):
    """Remove every run of words long enough to be boilerplate that already appeared."""
    words = text.split()
    span = MIN_REPEAT_WORDS
    if len(words) < span * 2:
        return text

    keys = [normalise(word) for word in words]
    seen, repeated = set(), [False] * len(words)
    for start in range(len(words) - span + 1):
        window = tuple(keys[start:start + span])
        if window in seen:
            repeated[start:start + span] = [True] * span
        else:
            seen.add(window)
    return " ".join(word for word, drop in zip(words, repeated) if not drop)


def deduplicate(text):
    """Return the input with repeated spec values and repeated word runs removed."""
    return drop_repeated_runs(drop_repeated_specs(text)).strip()


def main():
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found")

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    before = sum(len(row.get("input", "")) for row in rows)

    first_seen, cleaned, changed, duplicates, emptied = {}, [], [], [], 0
    for row in rows:
        original = row.get("input", "")
        row["input"] = deduplicate(original)  # Only "input" is ever rewritten.
        if not row["input"]:
            emptied += 1
            continue

        fingerprint = (row["input"], row.get("output", ""))
        if fingerprint in first_seen:
            duplicates.append((row.get("asin"), first_seen[fingerprint]))
            continue
        first_seen[fingerprint] = row.get("asin")

        cleaned.append(row)
        if len(original) > len(row["input"]):
            changed.append((len(original) - len(row["input"]), row.get("asin"),
                            len(original), len(row["input"])))

    CLEANED.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cleaned),
        encoding="utf-8")

    after = sum(len(row["input"]) for row in cleaned)
    removed = sum(item[0] for item in changed)
    changed.sort(reverse=True)

    print(f"rows {len(rows)} -> {len(cleaned)}")
    print(f"characters {before:,} -> {after:,}  ({1 - after / max(before, 1):.1%} removed)")
    print(f"rows changed {len(changed)} of {len(rows)} "
          f"({len(changed) / max(len(rows), 1):.1%}), {removed:,} characters cut")

    if duplicates:
        print(f"\n{len(duplicates)} exact duplicate rows dropped:")
        for asin, original_asin in duplicates:
            print(f"{asin} (same input and output as {original_asin})")
    if emptied:
        print(f"\n{emptied} rows dropped for having no content left")

    if changed:
        print(f"\nChanged rows, biggest first (showing {min(len(changed), LIST_ASINS)}):")
        for cut, asin, was, now in changed[:LIST_ASINS]:
            print(f"  {asin}  {was:>7,} -> {now:>7,}  (-{cut:,}, -{cut / was:.0%})")


if __name__ == "__main__":
    main()
