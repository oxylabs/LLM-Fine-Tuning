# LLM Fine-Tuning
[![Oxylabs LLM fine-tuning](https://github.com/oxylabs/LLM-Fine-Tuning/blob/main/LLM%20Fine-Tuning%20GitHub%20repository%20banner.png)](https://oxylabs.io/products/scraper-api/web)

[![](https://dcbadge.limes.pink/api/server/Pds3gBmKMH?style=for-the-badge&theme=discord)](https://discord.gg/Pds3gBmKMH) [![YouTube](https://img.shields.io/badge/YouTube-Oxylabs-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@oxylabs) 


# LLM Fine-Tuning: Amazon Product Title Generator

This repository is a hands-on example of LLM fine-tuning applied to e-commerce data. It uses an Amazon scraper to collect ~10,000 real product listings, cleans the resulting dataset, and fine-tunes `Qwen3-1.7B-Base` — an open base LLM — so it learns to write Amazon-style product titles. It's built as a developer resource for anyone researching how fine-tuning an LLM works in practice, from raw scraped data to a trained model.

The workflow follows one pipeline end to end: **Amazon scraper → clean → fine-tune LLM → test**. Everything below is covered in the order it happens in that pipeline, plus two explainer sections at the end for context.

- [What is LLM fine-tuning?](#what-is-llm-fine-tuning)
- [Key features](#key-features)
- [What's in here](#whats-in-here)
  - [Requirements](#requirements)
  - [Setup](#setup)
- [Fine-tuning an LLM vs. prompting an LLM or AI scraper](#fine-tuning-an-llm-vs-prompting-an-llm-or-ai-scraper)
- [Practical use cases](#practical-use-cases)
- [Related Oxylabs AI agent tooling](#related-oxylabs-ai-agent-tooling)
- [Learn more](#learn-more)

## What is LLM fine-tuning?

LLM fine-tuning is the process of taking a pre-trained base model and continuing its training on a narrower, task-specific dataset, so its outputs adapt to that task instead of relying only on general knowledge from pretraining.

In this repository, fine-tuning the LLM means:

- Starting from `Qwen3-1.7B-Base`, an open base model.
- Training it with LoRA (Low-Rank Adaptation), which updates a small set of additional weights instead of the full model. This is why the training step in this repo fits on Kaggle's free 2× T4 GPU tier instead of requiring large-scale computation.
- Using ~10,000 scraped and cleaned Amazon product listings as the training data, so the model learns the pattern of turning product attributes into an Amazon-style title.
- Running for a small number of epochs (three, in the reference run) while watching `eval_loss` to know when the model has learned the pattern versus when it's starting to overfit.

The result is a small, specialized model rather than a general-purpose one: it's trained specifically to write product titles in the style found in the training data, not to hold a general conversation.

## Key features

- **Amazon scraper** — single-request scripts for a best-sellers page or a single product, plus a full scraper that collects ~10,000 products in about 5 minutes via the Oxylabs Web Scraper API.
- **Resumable scraping** — the scraper can be stopped at any time and picks up where it left off when run again.
- **Category collection** — a browser console script that walks Amazon's best-sellers categories and writes `categories.json`; a ready-made version (15 root categories, 194 sub-categories) is included.
- **Dataset cleaning** — a cleaning script that removes repeated spec values, repeated runs of 30+ words, and duplicate rows, and reports what it removed.
- **LoRA fine-tuning script** — trains `Qwen3-1.7B-Base` on the cleaned dataset, designed for Kaggle's free GPU tier, with a documented path for running on smaller local hardware.
- **Evaluation tooling** — test scripts that run the fine-tuned model against held-out products and score each generated title by how many of its words actually appear in the source text, to catch invented content.

## What's in here

| Path | What it is |
| --- | --- |
| `scraper/single-request/` | One API call each: a best sellers page, a product page |
| `scraper/get-categories/` | Browser script that collects category IDs into `categories.json` |
| `scraper/scraper.py` | Scrapes 10,000 products into `dataset.jsonl` |
| `scraper/clean-dataset.py` | Removes repeated text, writes `dataset.clean.jsonl` |
| `trainer/train.py` | LoRA fine-tuning script |
| `trainer/dataset.clean.jsonl` | Example of the cleaned dataset (150 lines only) |
| `testing/` | Test scripts, held-out products, and the results from the video |

## Requirements

- Python 3.10+
- Oxylabs Web Scraper API credentials (Free Trial available)
- A GPU for training. Free option: Kaggle with 2× T4.

## Setup

```bash
git clone <this-repo>
cd fine-tune-llm
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

For scraping:

```bash
pip install requests python-dotenv
```

For training and testing:

```bash
pip install transformers datasets peft torch
```

Or everything at once: `pip install -r requirements.txt`

Add your credentials to `scraper/.env`:

```
OXYLABS_USERNAME="your_username"
OXYLABS_PASSWORD="your_password"
```

> [!TIP]
> **No credentials yet?** Web Scraper API has a free trial and no credit card is required. Create an account at [dashboard.oxylabs.io](https://dashboard.oxylabs.io/).

### Step 1 — Try a single Amazon scraper request

```bash
cd scraper/single-request
python best-sellers.py     # category page  → best-sellers.json
python product.py          # one ASIN       → product.json
```

Change `query` in either file to use a different category or ASIN. `parse: True` returns clean JSON instead of raw HTML.

### Step 2 — Get the category IDs

`scraper/get-categories/categories.json` is ready to use: 15 root categories, 194 sub-categories. It's a cleaned up version, and doesn't include all categories.

To collect your own:

1. Open [amazon.com/gp/bestsellers](https://www.amazon.com/gp/bestsellers).
2. Open Chrome DevTools → Console.
3. Paste in `category-scraper.js` and press Enter.
4. It downloads `categories.json`. Raise `MAX_DEPTH` to walk deeper.

### Step 3 — Scrape the Amazon dataset

```bash
cd scraper
cp get-categories/categories.json .
python scraper.py
```

Writes `dataset.jsonl`. Around 10,000 products in about 5 minutes.

> [!WARNING]
> Limited to up to 2,000 requests with a free API trial.

Stop the script anytime. Run it again and it continues where it left off.

Settings at the top of the file: `TARGET_PRODUCTS`, `PAGES_PER_CATEGORY`, `DOMAIN`, `GEO_LOCATION`.

### Step 4 — Clean the dataset

```bash
python clean-dataset.py
```

Reads `dataset.jsonl`, writes `dataset.clean.jsonl`, and prints what it removed: repeated spec values, repeated runs of 30+ words, duplicate rows.

Before training, hold a few categories out of the dataset for testing. You can use `testing/data-for-testing.jsonl`.

### Step 5 — Fine-tune the LLM

On Kaggle, as in the video:

1. Create a notebook and upload `dataset.clean.jsonl` as a private dataset.
2. First cell: `!pip uninstall -y torchao`
3. Second cell: paste `trainer/train.py` and set the two paths:

```python
DATA_PATH = "/kaggle/input/<your-dataset>/dataset.clean.jsonl"
OUTPUT_DIR = "/kaggle/working/Amazon-title-gen"
```

4. Settings → Accelerator → **GPU T4 x2**. Turn **Internet on**.
5. Save Version → Save & Run All (Commit).
6. Follow the Logs tab. Three epochs took about 5.5 hours.
7. Download the output folder when the run finishes.

Locally:

```bash
cd trainer
python train.py
```

On a small machine, switch `MODEL_NAME` to `Qwen/Qwen3-0.6B-Base` or similar and lower `MAX_LENGTH`. The base model downloads on first run.

Reading the logs:

- `eval_loss` falling → keep training.
- Flat over several checks → training is done.
- Rising, while training loss keeps falling → overfitting.
- `load_best_model_at_end` restores the best checkpoint, not the last one.

### Step 6 — Test the fine-tuned model

Put the trained `Amazon-title-gen` folder inside `testing/`, or point `MODEL_DIR` at `../Amazon-title-gen/final`.

```bash
cd testing
python test-model-single.py    # one product
python test-model.py           # all held-out products → test-results.json
```

`test-model.py` prints the generated title, the real title, and a word-match count: how many words in the generated title appear in the source text. A low count means the model invented something.

> [!IMPORTANT]
> The prompt must end with a newline (`\n`). That is the format the model was trained on.

## Fine-tuning an LLM vs. prompting an LLM or AI scraper

AI-powered scraping and title generation can mean a few different things in practice. This section lays out where fine-tuning an LLM — the approach used in this repo — sits relative to two other common patterns: prompting a general-purpose LLM at inference time, and AI scrapers that use an LLM to interpret pages during scraping.

| Criteria | Fine-tuning an LLM | Prompting a general-purpose LLM | AI Scraper (LLM-based extraction) |
| --- | --- | --- | --- |
| **Adaptation** | Model weights, via LoRA training | Nothing in the model — only the prompt, per call | Nothing in the model — an LLM parses each page as it's scraped |
| **Processing stage** | Once, during training | At inference time, every time a title is generated | During data collection, while parsing the page |
| **Purpose of data** | Training data for fine-tuning | Optional context passed into a prompt | The content being extracted or interpreted |
| **Format source** | Learned into the model from training examples | Through prompt instructions each call | Through the extraction prompt/schema each call |
| **Ideal use** | A repeatable, narrow task, like generating titles in one consistent style | Ad hoc or varied tasks that don't justify training a model | Turning unstructured pages into structured data |

In short: this is a fine-tuning pipeline that uses a scraper for its dataset, rather than a scraper that uses an LLM for extraction.

## Practical use cases

1. **Standardizing product titles across a catalog** — Sellers or marketplaces with large catalogs can use a fine-tuned model like this one to generate consistent, Amazon-style titles from structured attributes instead of writing each one manually.
2. **Learning the LLM fine-tuning workflow end to end** — Because the pipeline is small enough to run on free-tier hardware, it works as a teaching example for developers who want to see scraping, cleaning, LoRA fine-tuning, and evaluation connected in one place.
3. **Turning scraped category data into a specialized text generator** — The same scrape → clean → fine-tune pattern can be pointed at other structured, repetitive text (descriptions, bullet points, meta titles) by swapping in a different target field.
4. **Comparing a fine-tuned small model against prompting** — The included test scripts and word-match scoring make it possible to benchmark a fine-tuned `Qwen3-1.7B-Base` against simply prompting a general-purpose LLM for the same title-generation task.

## Related Oxylabs AI agent tooling

This LLM fine-tuning pipeline pairs an Amazon scraper with a trained model. If you are looking to connect Oxylabs products to AI coding assistants or AI agents more broadly, rather than fine-tuning a model yourself, these related repositories are built for that:

* [Oxylabs Agent Skills](https://github.com/oxylabs/agent-skills) – ready-made `skills.md` instructions that teach AI coding agents how to configure and call Oxylabs products, including the Web Scraper API, without guessing parameters.
* [Oxylabs MCP](https://github.com/oxylabs/oxylabs-mcp) – an MCP server that exposes Oxylabs [Web Scraper API](https://oxylabs.io/products/scraper-api/web) and [AI Studio](https://aistudio.oxylabs.io/) tools (including Amazon-specific scrapers) so AI models and agents can call them directly.

## Learn more

- [Web Scraper API documentation](https://developers.oxylabs.io/scraper-apis/web-scraper-api) — full reference for the parameters used by the Amazon scraper in this repository.
- [Oxylabs dashboard](https://dashboard.oxylabs.io/) — create an account and get Web Scraper API credentials.
- [Qwen3-1.7B-Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base) — details on the base model fine-tuned in this repo.
- [PEFT / LoRA documentation](https://huggingface.co/docs/peft/index) — background on the LoRA method used for fine-tuning.
- [Kaggle notebooks documentation](https://www.kaggle.com/docs/notebooks) — for running the fine-tuning step on free GPUs.

## Contact us

If you have questions or need support, reach out at [support@oxylabs.io](mailto:support@oxylabs.io), or via live chat in the [Oxylabs Dashboard](https://dashboard.oxylabs.io/). For enterprise inquiries, contact your dedicated account manager.
