import os
import json
import subprocess
import requests
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


FOOMATIC_REPO = "https://github.com/OpenPrinting/foomatic-db.git"
DRIVERLESS_URL = "https://raw.githubusercontent.com/OpenPrinting/openprinting.github.io/master/assets/json/driverless.json"

DATA_DIR = "data"
FOOMATIC_DIR = os.path.join(DATA_DIR, "foomatic-db")

PRINTER_XML_DIR = os.path.join(FOOMATIC_DIR, "db/source/printer")
DRIVER_XML_DIR = os.path.join(FOOMATIC_DIR, "db/source/driver")

DRIVERLESS_FILE = os.path.join(DATA_DIR, "driverless.json")

ARTIFACT_DIR = "artifacts"

TOP_K = 5


BRAND_MAP = {
    "hewlett-packard": "hp",
    "hp inc.": "hp",
    "hp": "hp",
    "canon inc.": "canon",
    "canon": "canon",
    "brother industries": "brother",
    "brother": "brother",
    "epson": "epson"
}

def normalize_manufacturer(name):

    if not name:
        return "unknown"

    key = name.lower().strip()

    return BRAND_MAP.get(key, key)


def normalize_model(model):

    if not model:
        return ""

    return re.sub(r'[^a-z0-9]', '', model.lower())


def ensure_dirs():

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def setup_foomatic():

    if not os.path.exists(FOOMATIC_DIR):

        print("Cloning foomatic-db...")

        subprocess.run(
            ["git", "clone", FOOMATIC_REPO, FOOMATIC_DIR],
            check=True
        )

    else:

        print("Updating foomatic-db...")

        subprocess.run(
            ["git", "-C", FOOMATIC_DIR, "pull"],
            check=True
        )


def download_driverless():

    print("Downloading driverless dataset...")

    r = requests.get(DRIVERLESS_URL)

    with open(DRIVERLESS_FILE, "w") as f:
        f.write(r.text)


def extract_connectivity(root):

    conn = []

    autodetect = root.find("autodetect")

    if autodetect is None:
        return conn

    if autodetect.find("usb") is not None:
        conn.append("USB")

    if autodetect.find("network") is not None:
        conn.append("Network")

    if autodetect.find("parallel") is not None:
        conn.append("Parallel")

    return conn

def extract_type(root):

    mech = root.find("mechanism")

    if mech is None:
        return "unknown"

    if mech.find("laser") is not None:
        return "laser"

    if mech.find("inkjet") is not None:
        return "inkjet"

    if mech.find("dotmatrix") is not None:
        return "dot-matrix"

    return "unknown"


def load_printers():

    printers = {}

    files = os.listdir(PRINTER_XML_DIR)

    for file in tqdm(files, desc="Parsing printers"):

        if not file.endswith(".xml"):
            continue

        path = os.path.join(PRINTER_XML_DIR, file)

        try:

            tree = ET.parse(path)
            root = tree.getroot()

            pid = root.attrib.get("id")

            model = root.findtext("model")
            make = normalize_manufacturer(root.findtext("make"))

            if not model:
                continue

            printers[pid] = {
                "id": pid,
                "manufacturer": make,
                "model": model,
                "normalized_model": normalize_model(model),
                "type": extract_type(root),
                "connectivity": extract_connectivity(root),
                "recommended_driver": root.findtext("driver"),
                "drivers": [],
                "functionality": root.findtext("functionality"),
                "notes": root.findtext("comments"),
            }

        except:
            continue

    return printers


def load_drivers():

    drivers = {}
    printer_to_drivers = {}

    files = os.listdir(DRIVER_XML_DIR)

    for file in tqdm(files, desc="Parsing drivers"):

        if not file.endswith(".xml"):
            continue

        path = os.path.join(DRIVER_XML_DIR, file)

        try:

            tree = ET.parse(path)
            root = tree.getroot()

            did = root.attrib.get("id")
            name = root.findtext("name")

            drivers[did] = {
                "id": did,
                "name": name
            }

            printers_node = root.find("printers")

            if printers_node is None:
                continue

            for p in printers_node.findall("printer"):

                pid = p.attrib.get("id") or p.text

                if not pid:
                    continue

                if pid not in printer_to_drivers:
                    printer_to_drivers[pid] = []

                printer_to_drivers[pid].append(did)

        except:
            continue

    return drivers, printer_to_drivers


def attach_drivers(printers, printer_to_drivers):

    for pid in printers:

        if pid in printer_to_drivers:
            printers[pid]["drivers"] = printer_to_drivers[pid]

    return printers


def load_driverless():

    with open(DRIVERLESS_FILE) as f:
        data = json.load(f)

    records = {}

    for r in data:

        model = r.get("model")

        if not model or model == "_dummy_":
            continue

        key = normalize_model(model)

        records[key] = {
            "airprint": r.get("airprt") == "1",
            "ipp_everywhere": r.get("ippeve") == "1",
        }

    return records


def merge_driverless(printers, driverless):

    for p in printers.values():

        key = p["normalized_model"]

        proto = driverless.get(key)

        p["airprint"] = proto["airprint"] if proto else False
        p["ipp_everywhere"] = proto["ipp_everywhere"] if proto else False
        p["driverless"] = p["airprint"] or p["ipp_everywhere"]

    return printers


def build_features(dataset):

    df = pd.DataFrame(dataset)

    df["manufacturer"] = df["manufacturer"].fillna("unknown")
    df["type"] = df["type"].fillna("unknown")

    df["driver_count"] = df["drivers"].apply(len)

    df["conn_usb"] = df["connectivity"].apply(lambda x: 1 if "USB" in x else 0)
    df["conn_network"] = df["connectivity"].apply(lambda x: 1 if "Network" in x else 0)

    df["protocol_airprint"] = df["airprint"].astype(int)
    df["protocol_ipp"] = df["ipp_everywhere"].astype(int)
    df["protocol_driverless"] = df["driverless"].astype(int)

    df = pd.get_dummies(df, columns=["manufacturer", "type"])

    features = df.drop(columns=[
        "model",
        "id",
        "connectivity",
        "drivers",
        "normalized_model",
        "notes"
    ], errors="ignore")

    features = features.replace({True: 1, False: 0}).infer_objects(copy=False)

    features = features.select_dtypes(include=["number"])

    return df, features


def compute_similarity(df, features):

    sim = cosine_similarity(features)

    models = df["model"].tolist()

    recommendations = {}

    for i, model in enumerate(models):

        scores = list(enumerate(sim[i]))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        top = []

        for idx, score in scores[1:TOP_K+1]:

            top.append({
                "model": models[idx],
                "score": float(score)
            })

        recommendations[model] = top

    return recommendations


def compute_clusters(features, n_clusters=8):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    labels = kmeans.fit_predict(features)

    score = silhouette_score(features, labels)

    clusters = {}

    for idx, label in enumerate(labels):

        clusters.setdefault(int(label), []).append(int(idx))

    return clusters, score


def export_artifacts(dataset, features, recommendations, clusters, cluster_score):

    metadata = {
        "generated_at": datetime.utcnow().isoformat(),
        "pipeline_version": "v1.1",
        "printer_count": len(dataset),
        "cluster_quality_score": cluster_score
    }

    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "unified_printers.json"), "w") as f:
        json.dump(dataset, f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "feature_vectors.json"), "w") as f:
        json.dump(features.to_dict(orient="records"), f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "recommendations.json"), "w") as f:
        json.dump(recommendations, f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "printer_clusters.json"), "w") as f:
        json.dump(clusters, f, indent=2)


def main():

    print("\nStarting Printer Intelligence Pipeline\n")

    ensure_dirs()

    setup_foomatic()

    download_driverless()

    printers = load_printers()

    drivers, printer_to_drivers = load_drivers()

    printers = attach_drivers(printers, printer_to_drivers)

    driverless = load_driverless()

    printers = merge_driverless(printers, driverless)

    dataset = list(printers.values())

    df, features = build_features(dataset)

    recommendations = compute_similarity(df, features)

    clusters, cluster_score = compute_clusters(features)

    export_artifacts(dataset, features, recommendations, clusters, cluster_score)

    print("\nPipeline completed successfully\n")

if __name__ == "__main__":
    main()
