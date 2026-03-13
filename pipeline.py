import os
import json
import subprocess
import requests
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


FOOMATIC_REPO = "https://github.com/OpenPrinting/foomatic-db.git"
DRIVERLESS_URL = "https://raw.githubusercontent.com/OpenPrinting/openprinting.github.io/master/assets/json/driverless.json"

DATA_DIR = "data"
FOOMATIC_DIR = os.path.join(DATA_DIR, "foomatic-db")
DRIVERLESS_FILE = os.path.join(DATA_DIR, "driverless.json")

PRINTER_XML_DIR = os.path.join(FOOMATIC_DIR, "db/source/printer")

ARTIFACT_DIR = "artifacts"

TOP_K = 5



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



def load_foomatic_printers():

    printers = []

    files = os.listdir(PRINTER_XML_DIR)

    for file in tqdm(files, desc="Parsing foomatic printers"):

        if not file.endswith(".xml"):
            continue

        path = os.path.join(PRINTER_XML_DIR, file)

        try:

            tree = ET.parse(path)
            root = tree.getroot()

            model = root.findtext("model")
            make = root.findtext("make")

            if not model:
                continue

            printers.append({
                "manufacturer": make,
                "model": model,
                "type": extract_type(root),
                "connectivity": extract_connectivity(root),
                "drivers": []
            })

        except:
            continue

    return printers



def load_driverless():

    with open(DRIVERLESS_FILE) as f:
        data = json.load(f)

    records = []

    for r in data:

        model = r.get("model")

        if not model or model == "_dummy_":
            continue

        manufacturer = model.split()[0]

        records.append({
            "manufacturer": manufacturer,
            "model": model,
            "airprint": r.get("airprt") == "1",
            "ipp_everywhere": r.get("ippeve") == "1"
        })

    return records



def merge_datasets(foomatic, driverless):

    merged = []

    for p in foomatic:

        record = p.copy()

        record["airprint"] = False
        record["ipp_everywhere"] = False

        for d in driverless:

            if d["model"].lower() == (p["model"] or "").lower():

                record["airprint"] = d["airprint"]
                record["ipp_everywhere"] = d["ipp_everywhere"]

        merged.append(record)

    return merged



def build_features(dataset):

    df = pd.DataFrame(dataset)

    df["manufacturer"] = df["manufacturer"].fillna("unknown")
    df["type"] = df["type"].fillna("unknown")

    df["conn_usb"] = df["connectivity"].apply(lambda x: 1 if "USB" in x else 0)
    df["conn_network"] = df["connectivity"].apply(lambda x: 1 if "Network" in x else 0)

    df["driver_count"] = df["drivers"].apply(len)

    df = pd.get_dummies(df, columns=["manufacturer", "type"])

    features = df.drop(columns=["model", "connectivity", "drivers"])

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



def export_artifacts(dataset, features, recommendations):

    with open(os.path.join(ARTIFACT_DIR, "unified_printers.json"), "w") as f:
        json.dump(dataset, f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "feature_vectors.json"), "w") as f:
        json.dump(features.to_dict(orient="records"), f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "recommendations.json"), "w") as f:
        json.dump(recommendations, f, indent=2)



def main():

    print("\nStarting OpenPrinting Printer Intelligence Pipeline\n")

    ensure_dirs()

    setup_foomatic()

    download_driverless()

    foomatic = load_foomatic_printers()

    driverless = load_driverless()

    dataset = merge_datasets(foomatic, driverless)

    df, features = build_features(dataset)

    recommendations = compute_similarity(df, features)

    export_artifacts(dataset, features, recommendations)

    print("\nPipeline completed successfully\n")


if __name__ == "__main__":
    main()
