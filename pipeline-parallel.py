import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html import unescape

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


FOOMATIC_REPO = "https://github.com/OpenPrinting/foomatic-db.git"
DRIVERLESS_URL = (
    "https://raw.githubusercontent.com/OpenPrinting/openprinting.github.io/"
    "master/assets/json/driverless.json"
)

DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"

FOOMATIC_DIR = os.path.join(DATA_DIR, "foomatic-db")
PRINTER_XML_DIR = os.path.join(FOOMATIC_DIR, "db/source/printer")
DRIVER_XML_DIR = os.path.join(FOOMATIC_DIR, "db/source/driver")

DRIVERLESS_FILE = os.path.join(DATA_DIR, "driverless.json")

TOP_K = 5
NEIGHBOR_POOL = 12
FEATURE_WEIGHT = 0.65
TEXT_WEIGHT = 0.35


BRAND_MAP = {
    "hewlettpackard": "hp",
    "hewlettpackard": "hp",
    "hpinc": "hp",
    "hp": "hp",
    "brotherindustries": "brother",
    "brother": "brother",
    "canoninc": "canon",
    "canon": "canon",
    "epsonseiko": "epson",
    "epson": "epson",
    "lexmarkinternational": "lexmark",
}


def normalize_text(value):
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalize_manufacturer(name):
    key = normalize_text(name)
    if not key:
        return "unknown"
    return BRAND_MAP.get(key, key)


def normalize_driver_id(driver_id):
    if not driver_id:
        return None
    driver_id = driver_id.strip()
    if not driver_id:
        return None
    return driver_id if driver_id.startswith("driver/") else f"driver/{driver_id}"


def simplify_driver_name(driver_id):
    if not driver_id:
        return "unknown"
    return driver_id.split("/", 1)[-1].lower()


def clean_text(value):
    if not value:
        return ""
    text = unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def refresh_enabled():
    return os.environ.get("OPENPRINTING_REFRESH", "").lower() in {"1", "true", "yes"}


def ensure_foomatic():
    if os.path.exists(PRINTER_XML_DIR):
        if refresh_enabled():
            print("Refreshing foomatic-db")
            subprocess.run(["git", "-C", FOOMATIC_DIR, "pull"], check=True)
        return

    print("Cloning foomatic-db")
    subprocess.run(["git", "clone", FOOMATIC_REPO, FOOMATIC_DIR], check=True)


def ensure_driverless():
    if os.path.exists(DRIVERLESS_FILE) and not refresh_enabled():
        return

    print("Downloading driverless dataset")
    response = requests.get(DRIVERLESS_URL, timeout=60)
    response.raise_for_status()
    with open(DRIVERLESS_FILE, "w") as handle:
        handle.write(response.text)


def extract_connectivity(root):
    connectivity = []
    autodetect = root.find("autodetect")

    if autodetect is None:
        return connectivity

    if autodetect.find("usb") is not None:
        connectivity.append("USB")
    if autodetect.find("network") is not None:
        connectivity.append("Network")
    if autodetect.find("parallel") is not None:
        connectivity.append("Parallel")

    return connectivity


def extract_type(root):
    mechanism = root.find("mechanism")
    if mechanism is None:
        return "unknown"

    for tag in ("laser", "inkjet", "dotmatrix", "led", "thermal", "sublimation", "impact"):
        if mechanism.find(tag) is not None:
            return tag

    return "unknown"


def extract_color(root):
    mechanism = root.find("mechanism")
    return mechanism is not None and mechanism.find("color") is not None


def extract_resolution(root):
    mechanism = root.find("mechanism")
    if mechanism is None:
        return 0

    best = 0
    for dpi in mechanism.findall("./resolution/dpi"):
        x = dpi.findtext("x")
        y = dpi.findtext("y")
        for value in (x, y):
            if value and value.isdigit():
                best = max(best, int(value))
    return best


def extract_languages(root):
    languages = []
    lang_node = root.find("lang")
    if lang_node is None:
        return languages

    for child in list(lang_node):
        languages.append(child.tag)

    return sorted(set(languages))


def extract_inline_driver_ids(root):
    ids = []
    drivers_node = root.find("drivers")
    if drivers_node is None:
        return ids

    for driver in drivers_node.findall("driver"):
        driver_id = driver.findtext("id") or driver.get("id") or (driver.text or "").strip()
        normalized = normalize_driver_id(driver_id)
        if normalized:
            ids.append(normalized)

    return sorted(set(ids))


def build_make_model_key(manufacturer, model):
    return normalize_text(f"{manufacturer} {model}")


def split_driverless_model(full_model):
    parts = full_model.strip().split(None, 1)
    if not parts:
        return "unknown", ""
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], parts[1]


def load_printers():
    printers = {}

    for file_name in tqdm(sorted(os.listdir(PRINTER_XML_DIR)), desc="Parsing printers"):
        if not file_name.endswith(".xml"):
            continue

        path = os.path.join(PRINTER_XML_DIR, file_name)

        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue

        printer_id = root.attrib.get("id")
        model = (root.findtext("model") or "").strip()
        raw_manufacturer = (root.findtext("make") or "").strip()

        if not printer_id or not model:
            continue

        manufacturer = normalize_manufacturer(raw_manufacturer)
        recommended_driver = normalize_driver_id(root.findtext("driver"))
        notes = clean_text(root.findtext("comments/en") or root.findtext("comments"))
        functionality = (root.findtext("functionality") or "unknown").strip()

        printers[printer_id] = {
            "id": printer_id,
            "manufacturer": manufacturer,
            "manufacturer_display": raw_manufacturer or manufacturer,
            "model": model,
            "canonical_name": f"{raw_manufacturer or manufacturer} {model}".strip(),
            "normalized_model": normalize_text(model),
            "normalized_make_model": build_make_model_key(raw_manufacturer or manufacturer, model),
            "type": extract_type(root),
            "color": extract_color(root),
            "max_resolution_dpi": extract_resolution(root),
            "connectivity": extract_connectivity(root),
            "languages": extract_languages(root),
            "recommended_driver": recommended_driver,
            "inline_drivers": extract_inline_driver_ids(root),
            "drivers": [],
            "functionality": functionality,
            "notes": notes,
            "legacy_supported": True,
            "airprint": False,
            "ipp_everywhere": False,
            "driverless_supported": False,
            "support_summary": "legacy-only",
            "source_presence": {
                "foomatic": True,
                "driverless": False,
            },
        }

    return printers


def load_drivers():
    printer_to_drivers = {}
    driver_names = {}

    for file_name in tqdm(sorted(os.listdir(DRIVER_XML_DIR)), desc="Parsing drivers"):
        if not file_name.endswith(".xml"):
            continue

        path = os.path.join(DRIVER_XML_DIR, file_name)

        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue

        driver_id = normalize_driver_id(root.attrib.get("id"))
        if not driver_id:
            continue

        driver_names[driver_id] = root.findtext("name") or driver_id.split("/", 1)[-1]
        printers_node = root.find("printers")
        if printers_node is None:
            continue

        for printer in printers_node.findall("printer"):
            printer_id = (
                printer.get("id")
                or printer.findtext("id")
                or (printer.text or "").strip()
            )
            if not printer_id:
                continue
            printer_to_drivers.setdefault(printer_id, []).append(driver_id)

    for printer_id, driver_ids in printer_to_drivers.items():
        printer_to_drivers[printer_id] = sorted(set(driver_ids))

    return printer_to_drivers, driver_names


def attach_drivers(printers, printer_to_drivers):
    for printer_id, printer in printers.items():
        merged = set(printer.get("inline_drivers", []))
        merged.update(printer_to_drivers.get(printer_id, []))
        if printer.get("recommended_driver"):
            merged.add(printer["recommended_driver"])
        printer["drivers"] = sorted(merged)
        printer.pop("inline_drivers", None)

    return printers


def load_driverless():
    with open(DRIVERLESS_FILE) as handle:
        rows = json.load(handle)

    records = {}

    for row in rows:
        model = row.get("model")
        if not model or model == "_dummy_":
            continue

        key = normalize_text(model)
        records[key] = {
            "model": model,
            "airprint": row.get("airprt") == "1",
            "ipp_everywhere": row.get("ippeve") == "1",
        }

    return records


def merge_driverless(printers, driverless):
    matched = 0
    unmatched = {}

    for printer in printers.values():
        proto = driverless.get(printer["normalized_make_model"])
        if not proto:
            continue

        matched += 1
        printer["airprint"] = proto["airprint"]
        printer["ipp_everywhere"] = proto["ipp_everywhere"]
        printer["driverless_supported"] = proto["airprint"] or proto["ipp_everywhere"]
        printer["support_summary"] = (
            "both" if printer["legacy_supported"] and printer["driverless_supported"] else "legacy-only"
        )
        printer["source_presence"]["driverless"] = True

    matched_keys = {
        printer["normalized_make_model"]
        for printer in printers.values()
        if printer["source_presence"]["driverless"]
    }

    for key, proto in driverless.items():
        if key in matched_keys:
            continue
        manufacturer_display, model = split_driverless_model(proto["model"])
        manufacturer = normalize_manufacturer(manufacturer_display)
        unmatched_id = f"driverless/{key}"
        unmatched[unmatched_id] = {
            "id": unmatched_id,
            "manufacturer": manufacturer,
            "manufacturer_display": manufacturer_display,
            "model": model,
            "canonical_name": proto["model"].strip(),
            "normalized_model": normalize_text(model),
            "normalized_make_model": key,
            "type": "unknown",
            "color": False,
            "max_resolution_dpi": 0,
            "connectivity": [],
            "languages": [],
            "recommended_driver": None,
            "drivers": [],
            "functionality": "unknown",
            "notes": "Present in the driverless printer list but not matched to the current Foomatic snapshot.",
            "legacy_supported": False,
            "airprint": proto["airprint"],
            "ipp_everywhere": proto["ipp_everywhere"],
            "driverless_supported": proto["airprint"] or proto["ipp_everywhere"],
            "support_summary": "driverless-only",
            "source_presence": {
                "foomatic": False,
                "driverless": True,
            },
        }

    printers.update(unmatched)
    return printers, matched, len(unmatched)


def build_features(dataset):
    frame = pd.DataFrame(dataset)

    frame["driver_count"] = frame["drivers"].apply(len)
    frame["conn_usb"] = frame["connectivity"].apply(lambda values: int("USB" in values))
    frame["conn_network"] = frame["connectivity"].apply(lambda values: int("Network" in values))
    frame["conn_parallel"] = frame["connectivity"].apply(lambda values: int("Parallel" in values))
    frame["supports_postscript"] = frame["languages"].apply(lambda values: int("postscript" in values))
    frame["supports_pcl"] = frame["languages"].apply(
        lambda values: int(any(value.startswith("pcl") for value in values))
    )
    frame["supports_pdf"] = frame["languages"].apply(lambda values: int("pdf" in values))
    frame["supports_text"] = frame["languages"].apply(lambda values: int("text" in values))
    frame["legacy_supported_flag"] = frame["legacy_supported"].astype(int)
    frame["protocol_airprint"] = frame["airprint"].astype(int)
    frame["protocol_ipp_everywhere"] = frame["ipp_everywhere"].astype(int)
    frame["protocol_driverless"] = frame["driverless_supported"].astype(int)
    frame["color_capable"] = frame["color"].astype(int)
    frame["has_notes"] = frame["notes"].apply(lambda value: int(bool(value)))
    frame["resolution_bucket"] = frame["max_resolution_dpi"].apply(
        lambda value: 0 if value == 0 else min((value // 300) * 300, 2400)
    )
    frame["recommended_driver_family"] = frame["recommended_driver"].apply(simplify_driver_name)

    feature_frame = pd.get_dummies(
        frame[
            [
                "manufacturer",
                "type",
                "functionality",
                "recommended_driver_family",
                "driver_count",
                "conn_usb",
                "conn_network",
                "conn_parallel",
                "supports_postscript",
                "supports_pcl",
                "supports_pdf",
                "supports_text",
                "legacy_supported_flag",
                "protocol_airprint",
                "protocol_ipp_everywhere",
                "protocol_driverless",
                "color_capable",
                "has_notes",
                "resolution_bucket",
            ]
        ],
        columns=["manufacturer", "type", "functionality", "recommended_driver_family"],
    )

    return frame, feature_frame


def build_descriptions(dataset, driver_name_map):
    descriptions = []

    for printer in dataset:
        driver_names = [
            driver_name_map.get(driver_id, driver_id.split("/", 1)[-1])
            for driver_id in printer["drivers"][:8]
        ]
        description = " ".join(
            [
                printer["manufacturer_display"],
                printer["model"],
                printer["type"],
                "color" if printer["color"] else "monochrome",
                "legacy supported" if printer["legacy_supported"] else "no legacy support",
                "driverless supported" if printer["driverless_supported"] else "no driverless support",
                printer["support_summary"],
                "airprint" if printer["airprint"] else "",
                "ipp everywhere" if printer["ipp_everywhere"] else "",
                " ".join(printer["connectivity"]),
                " ".join(printer["languages"]),
                "functionality " + printer["functionality"],
                "recommended " + (printer["recommended_driver"] or "none"),
                "drivers " + " ".join(driver_names),
                printer["notes"][:240],
            ]
        )
        descriptions.append(re.sub(r"\s+", " ", description).strip())

    return descriptions


def compute_record_confidence(printer):
    score = 0.0

    if printer["legacy_supported"]:
        score += 0.2
    if printer["driverless_supported"]:
        score += 0.2
    if printer["recommended_driver"]:
        score += 0.15
    if printer["drivers"]:
        score += min(len(printer["drivers"]) / 8.0, 1.0) * 0.15
    if printer["languages"]:
        score += min(len(printer["languages"]) / 4.0, 1.0) * 0.1
    if printer["connectivity"]:
        score += 0.05
    if printer["notes"]:
        score += 0.05
    if printer["max_resolution_dpi"] > 0:
        score += 0.05
    if printer["functionality"] == "A":
        score += 0.05
    elif printer["functionality"] == "B":
        score += 0.025
    elif printer["functionality"] == "F":
        score -= 0.1

    return max(0.0, min(score, 1.0))


def confidence_label(score):
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def compute_recommendation_confidence(source, candidate, similarity_score):
    confidence = 0.5 * max(0.0, min(similarity_score, 1.0))
    confidence += 0.2 * compute_record_confidence(source)
    confidence += 0.2 * compute_record_confidence(candidate)

    if source["support_summary"] == candidate["support_summary"]:
        confidence += 0.05
    elif source["legacy_supported"] == candidate["legacy_supported"]:
        confidence += 0.025

    if (
        source["recommended_driver"]
        and candidate["recommended_driver"]
        and simplify_driver_name(source["recommended_driver"]) == simplify_driver_name(candidate["recommended_driver"])
    ):
        confidence += 0.05

    if source["manufacturer"] == candidate["manufacturer"]:
        confidence += 0.03

    if source["functionality"] == "F" or candidate["functionality"] == "F":
        confidence -= 0.12

    if not source["drivers"] and not source["driverless_supported"]:
        confidence -= 0.1

    if not candidate["drivers"] and not candidate["driverless_supported"]:
        confidence -= 0.05

    confidence = max(0.0, min(confidence, 1.0))
    return round(confidence, 3), confidence_label(confidence)


def build_recommendation_entry(source, candidate, score):
    confidence_score, confidence_label_value = compute_recommendation_confidence(source, candidate, score)
    return {
        "id": candidate["id"],
        "manufacturer": candidate["manufacturer_display"],
        "model": candidate["model"],
        "canonical_name": candidate["canonical_name"],
        "score": round(float(score), 6),
        "confidence_score": confidence_score,
        "confidence_label": confidence_label_value,
        "legacy_supported": candidate["legacy_supported"],
        "driverless_supported": candidate["driverless_supported"],
        "airprint": candidate["airprint"],
        "ipp_everywhere": candidate["ipp_everywhere"],
        "support_summary": candidate["support_summary"],
        "recommended_driver": candidate["recommended_driver"],
    }


def compute_recommendation_map(dataset, distances, indices):
    recs = {}

    for i, (printer, row_distances, row_indices) in enumerate(zip(dataset, distances, indices)):
        recommendations = []
        for distance, index in zip(row_distances, row_indices):
            if index == i:
                continue
            candidate = dataset[index]
            score = 1.0 - float(distance)
            recommendations.append(build_recommendation_entry(printer, candidate, score))
            if len(recommendations) == TOP_K:
                break

        recs[printer["id"]] = {
            "id": printer["id"],
            "manufacturer": printer["manufacturer_display"],
            "model": printer["model"],
            "canonical_name": printer["canonical_name"],
            "record_confidence": round(compute_record_confidence(printer), 3),
            "record_confidence_label": confidence_label(compute_record_confidence(printer)),
            "recommendations": recommendations,
        }

    return recs


def compute_feature_recommendations(dataset, features):
    model = NearestNeighbors(
        n_neighbors=min(NEIGHBOR_POOL + 1, len(dataset)),
        metric="cosine",
        algorithm="brute",
    )
    model.fit(features)
    distances, indices = model.kneighbors(features)
    return compute_recommendation_map(dataset, distances, indices), distances, indices


def compute_text_recommendations(dataset, driver_name_map):
    descriptions = build_descriptions(dataset, driver_name_map)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(descriptions)
    model = NearestNeighbors(
        n_neighbors=min(NEIGHBOR_POOL + 1, len(dataset)),
        metric="cosine",
        algorithm="brute",
    )
    model.fit(matrix)
    distances, indices = model.kneighbors(matrix)
    return compute_recommendation_map(dataset, distances, indices), distances, indices


def compute_combined_recommendations(dataset, feature_distances, feature_indices, text_distances, text_indices):
    recs = {}

    for i, printer in enumerate(dataset):
        candidate_scores = {}

        for distance, index in zip(feature_distances[i], feature_indices[i]):
            if index == i:
                continue
            candidate_scores.setdefault(index, 0.0)
            candidate_scores[index] += FEATURE_WEIGHT * (1.0 - float(distance))

        for distance, index in zip(text_distances[i], text_indices[i]):
            if index == i:
                continue
            candidate_scores.setdefault(index, 0.0)
            candidate_scores[index] += TEXT_WEIGHT * (1.0 - float(distance))

        ordered = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[:TOP_K]
        recs[printer["id"]] = {
            "id": printer["id"],
            "manufacturer": printer["manufacturer_display"],
            "model": printer["model"],
            "canonical_name": printer["canonical_name"],
            "record_confidence": round(compute_record_confidence(printer), 3),
            "record_confidence_label": confidence_label(compute_record_confidence(printer)),
            "recommendations": [
                build_recommendation_entry(printer, dataset[index], score)
                for index, score in ordered
            ],
        }

    return recs


def export_artifacts(
    dataset,
    features,
    feature_recommendations,
    text_recommendations,
    combined_recommendations,
    matched_driverless,
    unmatched_driverless,
):
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v6",
        "printer_count": len(dataset),
        "driverless_match_count": matched_driverless,
        "driverless_only_count": unmatched_driverless,
        "driver_linked_count": sum(1 for printer in dataset if printer["drivers"]),
        "legacy_supported_count": sum(1 for printer in dataset if printer["legacy_supported"]),
        "driverless_capable_count": sum(1 for printer in dataset if printer["driverless_supported"]),
        "both_supported_count": sum(
            1 for printer in dataset if printer["legacy_supported"] and printer["driverless_supported"]
        ),
    }

    with open(f"{ARTIFACT_DIR}/metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    with open(f"{ARTIFACT_DIR}/printers.json", "w") as handle:
        json.dump(dataset, handle, indent=2)

    with open(f"{ARTIFACT_DIR}/unified_printers.json", "w") as handle:
        json.dump(dataset, handle, indent=2)

    feature_rows = []
    for printer, feature_row in zip(dataset, features.to_dict(orient="records")):
        feature_rows.append({"id": printer["id"], **feature_row})

    with open(f"{ARTIFACT_DIR}/features.json", "w") as handle:
        json.dump(feature_rows, handle, indent=2)

    with open(f"{ARTIFACT_DIR}/feature_recommendations.json", "w") as handle:
        json.dump(feature_recommendations, handle, indent=2)

    with open(f"{ARTIFACT_DIR}/embedding_recommendations.json", "w") as handle:
        json.dump(text_recommendations, handle, indent=2)

    with open(f"{ARTIFACT_DIR}/combined_recommendations.json", "w") as handle:
        json.dump(combined_recommendations, handle, indent=2)


def main():
    ensure_dirs()
    ensure_foomatic()
    ensure_driverless()

    printers = load_printers()
    printer_to_drivers, driver_name_map = load_drivers()
    printers = attach_drivers(printers, printer_to_drivers)

    driverless = load_driverless()
    printers, matched_driverless, unmatched_driverless = merge_driverless(printers, driverless)

    dataset = sorted(printers.values(), key=lambda printer: printer["id"])
    frame, features = build_features(dataset)

    feature_recommendations, feature_distances, feature_indices = compute_feature_recommendations(dataset, features)
    text_recommendations, text_distances, text_indices = compute_text_recommendations(dataset, driver_name_map)
    combined_recommendations = compute_combined_recommendations(
        dataset,
        feature_distances,
        feature_indices,
        text_distances,
        text_indices,
    )

    export_artifacts(
        dataset,
        features,
        feature_recommendations,
        text_recommendations,
        combined_recommendations,
        matched_driverless,
        unmatched_driverless,
    )

    print("Pipeline completed successfully")
    print(f"Printers: {len(dataset)}")
    print(f"Driver links: {sum(1 for printer in dataset if printer['drivers'])}")
    print(f"Driverless matches: {matched_driverless}")
    print(f"Driverless only: {unmatched_driverless}")


if __name__ == "__main__":
    main()
