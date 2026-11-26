"""Run independent cached stages; use --stage all for the complete course exercise."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, cast

from common import CONFIG, DATA, MODEL, ROOT, setup, sha256, write_json

setup()

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

LABELS = ["positive", "neutral", "negative"]
torch.set_num_threads(CONFIG["torch_threads"])
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAW = ROOT / "data/raw/dataset_youtube-comments.csv"


def save_table(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(DATA / f"{name}.csv", index=False, encoding="utf-8-sig")


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", keep_default_na=False)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")).strip()


def prepare() -> dict:
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = CONFIG["seed"]
    frame = pd.read_csv(RAW, encoding="utf-8-sig", keep_default_na=False)
    assert {"comment", "type"}.issubset(frame.columns)
    frame.insert(0, "comment_id", [f"YT{i:04d}" for i in range(1, len(frame) + 1)])
    frame["text_raw"] = frame["comment"]
    frame["text_clean"] = frame["comment"].map(clean)
    frame["text_model"] = frame["text_clean"].map(lambda x: re.sub(r"https?://\S+|www\.\S+", "http", re.sub(r"@[^\s]+", "@user", x)))
    frame["text_topic"] = frame["text_clean"].map(lambda x: clean(re.sub(r"https?://\S+|www\.\S+|@[^\s]+", " ", x)))
    duplicate = frame["text_raw"].duplicated(keep=False)
    groups = {text: f"D{i:03d}" for i, text in enumerate(frame.loc[duplicate, "text_raw"].unique(), 1)}
    frame["duplicate_group"] = frame["text_raw"].map(groups).fillna("")
    languages = []
    for text in tqdm(frame["text_topic"], desc="language"):
        try:
            predictions = detect_langs(text)
            language, confidence = predictions[0].lang, float(predictions[0].prob)
        except LangDetectException:
            language, confidence = "und", 0.0
        words = re.findall(r"[A-Za-z]+", text)
        if not text:
            reason = "empty"
        elif re.search(r"[\u3400-\u9fff]", text):
            reason = "cjk_or_mixed"
        elif len(words) < 3:
            reason = "short_uncertain"
        elif language != "en":
            reason = "other_language"
        elif confidence < CONFIG["language_min_probability"]:
            reason = "language_uncertain"
        else:
            reason = "included"
        languages.append((language, confidence, reason == "included", reason))
    frame[["language", "language_probability", "included", "exclusion_reason"]] = pd.DataFrame(languages, index=frame.index)
    save_table("comments_prepared", frame)
    return {"original_count": len(frame), "english_count": int(frame.included.sum()),
            "excluded_count": int((~frame.included).sum()), "exclusion_reasons": frame.exclusion_reason.value_counts().to_dict(),
            "type_counts": frame.type.value_counts().to_dict(), "duplicate_groups": len(groups),
            "included_types": frame.loc[frame.included, "type"].value_counts().to_dict()}


class Classifier:
    def __init__(self, key: str, max_length: int):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = MODEL / key
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        assert self.tokenizer is not None
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
        self.model.to(DEVICE).eval()
        self.labels = {int(index): label.lower() for index, label in self.model.config.id2label.items()}
        self.max_length = min(max_length, self.tokenizer.model_max_length)
        print(f"{key}: device={DEVICE}; labels={self.labels}; max_length={self.max_length}", flush=True)

    def predict(self, texts: list[str], pairs: list[str] | None = None, desc: str = "predict") -> pd.DataFrame:
        assert self.tokenizer is not None
        rows = [{} for _ in texts]
        # Length buckets reduce padding without changing output order or IDs.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]) + (len(pairs[i]) if pairs is not None else 0))
        batch_size = CONFIG["batch_size"]
        for start in tqdm(range(0, len(order), batch_size), desc=desc):
            indices = order[start:start + batch_size]
            batch = [texts[i] for i in indices]
            paired = [pairs[i] for i in indices] if pairs is not None else None
            try:
                raw = self.tokenizer(batch, text_pair=paired, truncation=False)["input_ids"]
                inputs = self.tokenizer(batch, text_pair=paired, padding=True, truncation=True,
                                        max_length=self.max_length, return_tensors="pt").to(DEVICE)
                with torch.inference_mode():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for j, i in enumerate(indices):
                    row: dict[str, Any] = {f"p_{label}": float(probs[j, k]) for k, label in self.labels.items()}
                    row.update(status="ok", error="", confidence=float(probs[j].max()),
                               sentiment_label=self.labels[int(probs[j].argmax())],
                               token_count=len(raw[j]), truncated=len(raw[j]) > self.max_length)
                    rows[i] = row
            except Exception as exc:
                # Retain failed identities explicitly; never substitute a neutral prediction.
                for i in indices:
                    rows[i] = {"status": "failed", "error": str(exc), "sentiment_label": "", "truncated": False}
        columns = [f"p_{label}" for label in self.labels.values()] + ["status", "error", "confidence", "sentiment_label", "token_count", "truncated"]
        return pd.DataFrame(rows, columns=columns)

    def close(self) -> None:
        del self.model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def sentiment() -> dict:
    frame = read_table("comments_prepared")
    subset = frame[frame.included].copy()
    classifier = Classifier("sentiment", CONFIG["max_length"])
    assert set(classifier.labels.values()) == set(LABELS)
    predictions = classifier.predict(subset.text_model.tolist(), desc="RoBERTa")
    classifier.close()
    predictions.insert(0, "comment_id", subset.comment_id.tolist())
    predictions["sentiment_score"] = predictions.p_positive - predictions.p_negative
    predictions["low_confidence"] = predictions.confidence < 0.6
    save_table("sentiment", predictions)
    return {"success": int((predictions.status == "ok").sum()), "failed": int((predictions.status == "failed").sum()),
            "truncated": int(predictions.truncated.sum()), "labels": predictions.sentiment_label.value_counts().to_dict()}


def topics() -> dict:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    frame = read_table("comments_prepared")
    subset = frame[frame.included].copy()
    texts = subset.text_topic.tolist()
    encoder = SentenceTransformer(str(MODEL / "embedding"), device=DEVICE, local_files_only=True)
    vectors = encoder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    np.save(DATA / "embeddings.npy", vectors)
    lengths = encoder.tokenizer(texts, truncation=False)["input_ids"]
    reducer = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=CONFIG["seed"])
    clusterer = HDBSCAN(min_cluster_size=20, min_samples=5, metric="euclidean", prediction_data=True)
    topic_model = BERTopic(embedding_model=encoder, umap_model=reducer, hdbscan_model=clusterer,
                           vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1),
                           ctfidf_model=ClassTfidfTransformer(), representation_model=KeyBERTInspired(top_n_words=8, random_state=CONFIG["seed"]),
                           top_n_words=8, nr_topics=None, verbose=True)
    assignments, strengths = topic_model.fit_transform(texts, embeddings=vectors)
    coordinates = UMAP(n_neighbors=15, n_components=2, min_dist=0.15, metric="cosine", random_state=CONFIG["seed"]).fit_transform(vectors)
    mapped = pd.DataFrame({"comment_id": subset.comment_id.tolist(), "topic_id": assignments,
                           "cluster_strength": strengths, "umap_x": coordinates[:, 0], "umap_y": coordinates[:, 1]})
    save_table("topics", mapped)
    details = []
    for topic_id in sorted(set(assignments)):
        keywords = cast(list[tuple[str, float]], topic_model.get_topic(topic_id))
        representatives = list(dict.fromkeys(topic_model.get_representative_docs(topic_id) or []))
        ids = [subset.loc[subset.text_topic == text, "comment_id"].iloc[0] for text in representatives]
        details.append({"topic_id": int(topic_id), "count": int(assignments.count(topic_id)),
                        "keywords": [{"term": term, "score": float(score)} for term, score in keywords],
                        "keyword_score_type": "KeyBERTInspired cosine similarity", "representative_ids": ids,
                        "representative_texts": representatives})
    write_json(DATA / "topic_details.json", details)
    topic_model.save(str(MODEL / "bertopic_result"), serialization="safetensors", save_ctfidf=True, save_embedding_model=False)
    return {"count": len(subset), "topic_count": len(set(assignments) - {-1}), "unassigned": assignments.count(-1),
            "embedding_dimension": int(vectors.shape[1]), "embedding_max_length": encoder.max_seq_length,
            "embedding_truncated": sum(len(x) > (encoder.max_seq_length or 256) for x in lengths),
            "parameters": {"umap_neighbors": 15, "umap_components": 5, "min_cluster_size": 20, "min_samples": 5}}


def mentions() -> dict:
    import spacy
    frame = read_table("comments_prepared")
    subset = frame[frame.included].sample(n=min(CONFIG["sample_size"], int(frame.included.sum())), random_state=CONFIG["seed"])
    write_json(DATA / "sample_ids.json", subset.comment_id.tolist())
    segmenter = spacy.blank("en")
    segmenter.add_pipe("sentencizer")
    sentences = []
    for row in subset.itertuples():
        for i, sentence in enumerate(segmenter(row.text_clean).sents):
            evidence = sentence.text.strip()
            if not evidence:
                continue
            pronoun = re.search(r"\b(they|them|their|it|these|those)\b", evidence, re.I)
            named = re.search(r"\b(chin\w*|americ\w*|US|USA|BYD|NIO|Xiaomi|Geely|Tesla|Ford|GM|Chevy|Chevrolet|Dodge|tariff\w*)\b", evidence, re.I)
            sentences.append({"comment_id": row.comment_id, "sentence_id": f"{row.comment_id}_S{i+1:02d}",
                              "evidence": evidence, "context_needed": bool(pronoun and not named)})
    sentences_df = pd.DataFrame(sentences)
    save_table("sentences", sentences_df)
    records = [dict(sentence, aspect_id=aspect["id"], target=aspect["target"], aspect=aspect["aspect"],
                    hypothesis=aspect["hypothesis"]) for sentence in sentences for aspect in CONFIG["aspects"]]
    pairs = pd.DataFrame(records)
    classifier = Classifier("mention", CONFIG["aspect_max_length"])
    assert set(classifier.labels.values()) == {"contradiction", "neutral", "entailment"}
    predictions = classifier.predict(pairs.evidence.tolist(), pairs.hypothesis.tolist(), desc="BART mentions")
    classifier.close()
    predictions = predictions.drop(columns=["sentiment_label", "confidence"])
    pairs = pd.concat([pairs.reset_index(drop=True), predictions], axis=1)
    # Multi-label NLI normalizes entailment against contradiction for EACH aspect.
    pairs["mention_score"] = pairs.p_entailment / (pairs.p_entailment + pairs.p_contradiction)
    pairs["detected"] = (pairs.status == "ok") & (pairs.mention_score >= CONFIG["mention_threshold"])
    pairs["eligible"] = pairs.detected & ~pairs.context_needed
    save_table("mention_scores", pairs)
    # Fixed, non-labelled inspection sample, retained for transparent quick checks.
    save_table("mention_spotcheck", pairs[pairs.sentence_id.isin(sentences_df.sentence_id.head(8))])
    return {"sample_size": len(subset), "sentence_count": len(sentences), "candidate_pairs": len(pairs),
            "success": int((pairs.status == "ok").sum()), "failed": int((pairs.status == "failed").sum()),
            "detected_pairs": int(pairs.detected.sum()), "context_deferred": int((pairs.detected & pairs.context_needed).sum()),
            "eligible_pairs": int(pairs.eligible.sum()), "truncated_pairs": int(pairs.truncated.sum()),
            "threshold": CONFIG["mention_threshold"]}


def aggregate_labels(labels: list[str]) -> str:
    values = set(labels)
    if {"positive", "negative"}.issubset(values):
        return "mixed"
    if "positive" in values:
        return "positive"
    if "negative" in values:
        return "negative"
    return "neutral" if "neutral" in values else "failed"


def aspects() -> dict:
    pairs = read_table("mention_scores")
    eligible = pairs[pairs.eligible].drop(columns=["status", "error", "truncated", "token_count", "p_neutral"]).reset_index(drop=True)
    classifier = Classifier("absa", CONFIG["aspect_max_length"])
    assert set(classifier.labels.values()) == set(LABELS), "ABSA labels require explicit publisher mapping"
    predictions = classifier.predict(eligible.evidence.tolist(), eligible.aspect.tolist(), desc="DeBERTa ABSA")
    classifier.close()
    evidence = pd.concat([eligible, predictions], axis=1)
    save_table("aspect_evidence", evidence)
    records = []
    for group_key, group in evidence.groupby(["comment_id", "aspect_id"]):
        comment_id, aspect_id = cast(tuple[str, str], group_key)
        successful = group[group.status == "ok"]
        records.append({"comment_id": comment_id, "aspect_id": aspect_id,
                        "sentiment_label": aggregate_labels(successful.sentiment_label.tolist()),
                        "status": "ok" if len(successful) else "failed", "partial_failure": len(successful) != len(group),
                        "sentence_count": len(group), "sentence_ids": "|".join(group.sentence_id),
                        "mention_score_max": float(group.mention_score.max())})
    aggregated = pd.DataFrame(records, columns=["comment_id", "aspect_id", "sentiment_label", "status", "partial_failure", "sentence_count", "sentence_ids", "mention_score_max"])
    save_table("comment_aspects", aggregated)
    return {"success": int((evidence.status == "ok").sum()), "failed": int((evidence.status == "failed").sum()),
            "truncated": int(evidence.truncated.sum()), "comment_aspect_records": len(aggregated),
            "matched_comments": int(aggregated.comment_id.nunique()), "mixed_records": int((aggregated.sentiment_label == "mixed").sum())}


def summarize() -> dict:
    comments = read_table("comments_prepared")
    comments = comments.merge(read_table("sentiment"), on="comment_id", how="left", validate="one_to_one")
    comments = comments.merge(read_table("topics"), on="comment_id", how="left", validate="one_to_one")
    comments.loc[~comments.included, "status"] = "excluded"
    names_path = ROOT / "topic_names.json"
    names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else {}
    details = json.loads((DATA / "topic_details.json").read_text(encoding="utf-8"))
    topic_rows = []
    for topic in details:
        topic_id = topic["topic_id"]
        words = [x["term"] for x in topic["keywords"]]
        name = names.get(str(topic_id), "未归类" if topic_id == -1 else " / ".join(words[:3]))
        comments.loc[comments.topic_id == topic_id, "topic_name"] = name
        group = comments[(comments.topic_id == topic_id) & (comments.status == "ok")]
        row = {"topic_id": topic_id, "topic_name": name, "comment_count": topic["count"], "valid_sentiment_count": len(group),
               "keywords": " | ".join(words), "keyword_score_type": topic["keyword_score_type"],
               "representative_ids": "|".join(topic["representative_ids"])}
        for label in LABELS:
            count = int((group.sentiment_label == label).sum())
            row[f"{label}_count"] = count
            row[f"{label}_ratio"] = count / len(group) if len(group) else np.nan
        topic_rows.append(row)
    save_table("comments_analysis", comments)
    save_table("topics_summary", pd.DataFrame(topic_rows).sort_values("comment_count", ascending=False))
    aggregated = read_table("comment_aspects")
    sample_ids = json.loads((DATA / "sample_ids.json").read_text(encoding="utf-8"))
    summaries = []
    for aspect in CONFIG["aspects"]:
        group = aggregated[(aggregated.aspect_id == aspect["id"]) & (aggregated.status == "ok")]
        row = {"aspect_id": aspect["id"], "aspect_name": aspect["name"], "sample_comment_count": len(sample_ids),
               "mentioned_comment_count": len(group)}
        for label in LABELS + ["mixed"]:
            count = int((group.sentiment_label == label).sum())
            row[f"{label}_count"] = count
            row[f"{label}_ratio"] = count / len(group) if len(group) else np.nan
        summaries.append(row)
    save_table("aspect_summary", pd.DataFrame(summaries).sort_values("mentioned_comment_count", ascending=False))
    valid = comments[comments.status == "ok"]
    assert comments.comment_id.is_unique and len(comments) == len(pd.read_csv(RAW))
    assert np.allclose(valid[["p_positive", "p_neutral", "p_negative"]].astype(float).sum(axis=1), 1, atol=1e-5)
    assert not aggregated.duplicated(["comment_id", "aspect_id"]).any()
    assert set(aggregated.comment_id).issubset(sample_ids)
    assert int(sum(t["comment_count"] for t in topic_rows)) == int(comments.included.sum())
    evidence = read_table("aspect_evidence")
    assert np.allclose(evidence.loc[evidence.status == "ok", ["p_positive", "p_neutral", "p_negative"]].astype(float).sum(axis=1), 1, atol=1e-5)
    return {"validation": "passed", "records": len(comments), "valid_sentiments": len(valid),
            "unmatched_sample_comments": len(sample_ids) - aggregated.comment_id.nunique()}


STAGES = {"prepare": prepare, "sentiment": sentiment, "topics": topics, "mentions": mentions, "aspects": aspects, "summarize": summarize}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    signature = hashlib.sha256((sha256(RAW) + json.dumps(CONFIG, sort_keys=True)).encode()).hexdigest()
    path = DATA / "run_config.json"
    run: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    if run.get("signature", signature) != signature:
        raise ValueError("Input/config changed: move data/processed aside before recomputing all stages.")
    run.update(signature=signature, input_sha256=sha256(RAW), config=CONFIG, device=DEVICE,
               python=sys.version, updated_at=datetime.now(timezone.utc).isoformat())
    packages = ["torch", "transformers", "sentence-transformers", "bertopic", "umap-learn", "hdbscan", "plotly", "langdetect", "spacy", "pandas", "numpy", "matplotlib"]
    run["packages"] = {key: importlib.metadata.version(key) for key in packages}
    run["models"] = {key: json.loads((MODEL / key / "download_manifest.json").read_text(encoding="utf-8"))
                     for key in CONFIG["models"] if (MODEL / key / "download_manifest.json").exists()}
    selected = list(STAGES) if args.stage == "all" else [args.stage]
    for name in selected:
        if name != "summarize" and name in run["stages"] and run["stages"][name]["status"] == "complete" and not args.force:
            print(f"{name}: cached", flush=True)
            continue
        start = time.monotonic()
        print(f"START {name}", flush=True)
        try:
            stats = STAGES[name]()
            run["stages"][name] = {"status": "complete", "seconds": round(time.monotonic() - start, 2), **stats}
        except Exception as exc:
            run["stages"][name] = {"status": "failed", "error": str(exc)}
            write_json(path, run)
            raise
        write_json(path, run)
        print(json.dumps(run["stages"][name], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
