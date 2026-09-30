"""Cache zero-shot API emotion scores for SemEval-2007 Affective Text."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import EMOTION_NAMES, load_affective_text, load_prediction_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    'Rate the following news headline on 6 emotions: anger, disgust, fear, joy, sadness, surprise. '
    'Return only 6 numbers from 0 to 100, comma-separated, in that order.\n'
    'Headline: "{headline}"\n'
    "Scores:"
)


def parse_scores(text: str) -> list[float]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) < 6:
        raise ValueError(f"Could not parse 6 scores from response: {text!r}")
    scores = [max(float(x), 0.0) for x in nums[:6]]
    if sum(scores) <= 0:
        raise ValueError(f"Parsed zero-sum scores from response: {text!r}")
    return scores


def call_openai_chat_completions(
    headline: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout_sec: float,
) -> tuple[str, dict]:
    prompt = PROMPT_TEMPLATE.format(headline=headline)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise annotation model."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    text = body["choices"][0]["message"]["content"]
    return text, body


def call_gemini_generate_content(
    headline: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout_sec: float,
) -> tuple[str, dict]:
    prompt = PROMPT_TEMPLATE.format(headline=headline)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
        },
    }
    url = (
        base_url.rstrip("/")
        + f"/models/{model}:generateContent?key={urllib.parse.quote(api_key)}"
    )
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    candidates = body.get("candidates", [])
    if not candidates:
        raise KeyError(f"No Gemini candidates in response: {body}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    if not text:
        raise KeyError(f"No text parts in Gemini response: {body}")
    return text, body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/AffectiveText.Semeval.2007")
    parser.add_argument("--output", default="data/processed/affective_text_predictions.jsonl")
    parser.add_argument("--provider", choices=["openai", "gemini"], default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.model is None:
        if args.provider == "gemini":
            args.model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")
        else:
            args.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
    if args.base_url is None:
        if args.provider == "gemini":
            args.base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        else:
            args.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if args.api_key_env is None:
        args.api_key_env = "GEMINI_API_KEY" if args.provider == "gemini" else "OPENAI_API_KEY"

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key in env var {args.api_key_env}")

    data = load_affective_text(args.data_dir)
    ids = data["ids"]
    headlines = data["headlines"]
    if args.limit is not None:
        ids = ids[:args.limit]
        headlines = headlines[:args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists() and not args.overwrite:
        existing = load_prediction_cache(out_path)
        log.info(f"Loaded {len(existing)} cached predictions from {out_path}")

    n_done = 0
    with open(out_path, "a" if existing and not args.overwrite else "w", encoding="utf-8") as f:
        for idx, headline in zip(ids, headlines):
            if idx in existing and not args.overwrite:
                continue
            try:
                if args.provider == "gemini":
                    raw_text, raw_json = call_gemini_generate_content(
                        headline=headline,
                        model=args.model,
                        api_key=api_key,
                        base_url=args.base_url,
                        timeout_sec=args.timeout_sec,
                    )
                else:
                    raw_text, raw_json = call_openai_chat_completions(
                        headline=headline,
                        model=args.model,
                        api_key=api_key,
                        base_url=args.base_url,
                        timeout_sec=args.timeout_sec,
                    )
                scores = parse_scores(raw_text)
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as exc:
                log.error(f"Failed on id={idx}: {exc}")
                continue

            row = {
                "id": idx,
                "headline": headline,
                "emotions": EMOTION_NAMES,
                "scores": scores,
                "provider": args.provider,
                "model": args.model,
                "base_url": args.base_url,
                "prompt_template": PROMPT_TEMPLATE,
                "raw_text": raw_text,
                "raw_response": raw_json,
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            f.flush()
            n_done += 1
            if n_done % 50 == 0:
                log.info(f"Cached {n_done} new predictions")
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    log.info(f"Finished. Predictions cached at {out_path}")


if __name__ == "__main__":
    main()
