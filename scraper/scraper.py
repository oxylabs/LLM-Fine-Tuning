import json
import logging
import os
import random
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DOMAIN = "com"
GEO_LOCATION = "11249"
PAGES_PER_CATEGORY = 2

TARGET_PRODUCTS = 10000  # Stop once the dataset holds this many rows.

BATCH_LIMIT = 5000   # Max queries per batch request. API limit is 5k.
POLL_INTERVAL = 30
WORKERS = 20
TIMEOUT = 180
BATCH_URL = "https://data.oxylabs.io/v1/queries/batch"
OUTPUT = Path("dataset.jsonl")
SKIPPED = Path("skipped.txt")   # ASINs already rejected, so they are never re-scraped.


# Best seller slugs or IDs, plus their display names for readable logs.
with open("categories.json", encoding="utf-8") as file:
    CATEGORY_TREE = json.load(file)

def iter_leaves(nodes):
    """Yield the deepest node of every branch; a childless node is its own leaf."""
    for node in nodes:
        children = node.get("children")
        if children:
            yield from iter_leaves(children)
        else:
            yield node

CATEGORY_NAMES = {leaf["id"]: leaf["name"] for leaf in iter_leaves(CATEGORY_TREE)}
CATEGORIES = list(CATEGORY_NAMES)

# Product details to leave out of the dataset.
SKIP_DETAILS = ("asin", "rank", "review", "upc", "gtin", "identification",
                "part_number", "model_number", "date_first", "department",
                "warranty", "item_type_name")

GREEN, YELLOW, RED, GREY, OFF = "\033[92m", "\033[93m", "\033[91m", "\033[90m", "\033[0m"


def label(query):
    """Show a category's name instead of its ID; ASINs are returned unchanged."""
    return CATEGORY_NAMES.get(query, query)


def make_session():
    load_dotenv()
    username, password = os.getenv("OXYLABS_USERNAME"), os.getenv("OXYLABS_PASSWORD")
    if not (username and password):
        sys.exit("Missing credentials in .env file")

    session = requests.Session()
    session.auth = (username, password)
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "POST"], respect_retry_after_header=False)
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)  # Job links come back as HTTP.
    return session


def submit_batch(session, source, queries, pages=1):
    """Send one batch of queries and return the jobs the API created."""
    response = session.post(BATCH_URL, timeout=TIMEOUT, json={
        "source": source,
        "domain": DOMAIN,
        "geo_location": GEO_LOCATION,
        "query": queries,
        "start_page": 1,
        "pages": pages,
        "parse": True,
    })
    if response.status_code == 401:
        sys.exit("Oxylabs rejected the credentials - check your .env file.")
    response.raise_for_status()

    jobs = response.json()["queries"]
    logging.info(f"{GREEN}Submitted %d %s jobs{OFF}", len(jobs), source)
    return jobs


def job_urls(job, rel):
    """Return the URLs for a link relation, e.g. 'self' or 'results-content'."""
    for link in job["_links"]:
        if link["rel"] == rel:
            return link.get("href_list") or [link["href"]]
    return []


def poll_jobs(session, jobs):
    """Poll until every job is done or faulted, yielding (job, contents) for the good ones."""

    def get_json(url):
        response = session.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()

    def status(job):
        return get_json(job_urls(job, "self")[0])["status"]

    def contents(job):
        return [get_json(url) for url in job_urls(job, "results-content")]

    pending = list(jobs)
    with ThreadPoolExecutor(WORKERS) as pool:
        while pending:
            time.sleep(POLL_INTERVAL)
            states = list(pool.map(status, pending))

            for job, state in zip(pending, states):
                if state == "faulted":
                    logging.warning(f"{RED}Job %s ('%s') faulted{OFF}",
                                    job["id"], label(job["query"]))

            done = [job for job, state in zip(pending, states) if state == "done"]
            yield from zip(done, pool.map(contents, done))

            pending = [j for j, s in zip(pending, states) if s not in ("done", "faulted")]
            logging.info(f"{YELLOW}%d jobs still running{OFF}", len(pending))


def find_asins(session, categories):
    """Scrape every category in one batch and return {asin: category}, duplicates removed."""
    found = {}
    jobs = submit_batch(session, "amazon_bestsellers", categories, pages=PAGES_PER_CATEGORY)
    for job, pages in poll_jobs(session, jobs):
        for content in pages:
            listings = content.get("results") or []
            for item in listings:
                if item.get("asin"):
                    found.setdefault(item["asin"], job["query"])  # Lists overlap a lot.
            logging.info("'%s' page %s: %d listings", label(job["query"]),
                         content.get("page"), len(listings))
    return found


def clean_text(value):
    """Normalise fancy unicode (mathematical bold letters -> plain) and flatten whitespace."""
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def end_sentence(text):
    """End prose with a full stop so it does not run into the specs that follow."""
    text = text.rstrip(" ,;:-")
    return text + "." if text and not text.endswith((".", "!", "?")) else text


def format_bullets(raw):
    """Flatten Amazon's newline-separated bullets into sentences."""
    lines = (clean_text(line) for line in str(raw or "").split("\n"))
    return " ".join(end_sentence(line) for line in lines if line)


def format_specs(details):
    """Flatten product_details into 'Key: value; Key: value', dropping noise keys."""
    if not isinstance(details, dict):
        return ""
    pairs = [f"{key.replace('_', ' ').capitalize()}: {clean_text(value)}"
             for key, value in details.items()
             if not any(part in key for part in SKIP_DETAILS)
             and isinstance(value, (str, int, float)) and not isinstance(value, bool)
             and clean_text(value)]
    return "; ".join(pairs)


def build_example(content, category):
    """Turn one product into an input/output pair, or None if too little is there."""
    title = clean_text(content.get("title") or content.get("product_name"))
    bullets = format_bullets(content.get("bullet_points"))
    description = end_sentence(clean_text(content.get("description")))
    specs = format_specs(content.get("product_details"))

    # Some listings repeat one field verbatim in the other; keep it once.
    if bullets and description and (description in bullets or bullets in description):
        description = ""

    # Specs are required: without them a title is mostly guesswork.
    if not (title and specs and (bullets or description)):
        missing = [name for name, text in (("title", title), ("bullet_points", bullets),
                   ("description", description), ("product_details", specs)) if not text]
        logging.info(f"{GREY}Skipped %s (missing: %s){OFF}",
                     content.get("asin"), ", ".join(missing))
        return None

    brand = clean_text(content.get("brand") or content.get("manufacturer"))
    if brand and brand.lower() not in specs.lower():
        specs = f"Brand: {brand}; {specs}"

    return {"asin": content.get("asin"), "category": category,
            "input": " ".join(text for text in (bullets, description, specs) if text),
            "output": title}


def interleave(asins):
    """Order ASINs one per category in turn, so any cut-off still spans every category."""
    by_category = defaultdict(list)
    for asin, category in asins.items():
        by_category[category].append(asin)
    for group in by_category.values():
        random.shuffle(group)
    return [asin for row in zip_longest(*by_category.values())
            for asin in row if asin is not None]


def scrape_products(session, asins, needed):
    """Scrape batch by batch until `needed` rows are written or the ASINs run out.

    Each batch asks for exactly the rows still missing, then adds the measured reject rate
    from prior batches so later requests overshoot only by as much as experience justifies.
    A clean run (nothing filtered out) never overshoots at all.
    """
    queue, written, submitted = interleave(asins), 0, 0
    with OUTPUT.open("a", encoding="utf-8") as out, SKIPPED.open("a", encoding="utf-8") as skip:
        while queue and written < needed:
            reject_rate = (submitted - written) / submitted if submitted else 0
            reject_rate = min(reject_rate, 0.95)  # Guard against dividing by ~zero.
            missing = needed - written
            size = min(BATCH_LIMIT, len(queue), round(missing / (1 - reject_rate)))
            chunk, queue = queue[:size], queue[size:]
            submitted += len(chunk)

            for job, pages in poll_jobs(session, submit_batch(session, "amazon_product", chunk)):
                example = build_example(pages[0], asins[job["query"]]) if pages else None
                if example:
                    out.write(json.dumps(example, ensure_ascii=False) + "\n")
                    written += 1
                else:
                    skip.write(job["query"] + "\n")  # Never scrape this ASIN again.
                out.flush()   # A crash keeps everything scraped so far.
                skip.flush()
            logging.info(f"{GREEN}%d of %d new products written{OFF}", written, needed)


def load_progress():
    """Read what previous runs finished: known ASINs, finished categories, row count."""
    known, finished, rows = set(), set(), 0
    if OUTPUT.exists():
        for line in OUTPUT.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            known.add(row["asin"])
            finished.add(row["category"])
            rows += 1
    if SKIPPED.exists():
        known.update(SKIPPED.read_text().split())
    return known, finished, rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    session = make_session()

    known, finished, rows = load_progress()
    if known:
        logging.info("Already done: %d rows, %d ASINs known, %d/%d categories",
                     rows, len(known), len(finished), len(CATEGORIES))

    needed = TARGET_PRODUCTS - rows
    categories = [c for c in CATEGORIES if c not in finished]
    if needed <= 0 or not categories:
        logging.info(f"{GREEN}Nothing left to scrape{OFF}")
        return

    try:
        asins = {a: c for a, c in find_asins(session, categories).items() if a not in known}
        logging.info("%d ASINs available, %d products needed", len(asins), needed)
        scrape_products(session, asins, needed)
    except requests.RequestException as error:
        logging.info(f"{RED}%s{OFF}", error)

    total = len(OUTPUT.read_text(encoding="utf-8").splitlines()) if OUTPUT.exists() else 0
    logging.info(f"{GREEN}%d products in %s (+%d this run){OFF}", total, OUTPUT, total - rows)


if __name__ == "__main__":
    main()
