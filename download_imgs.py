import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm

import torch
from transformers import pipeline


def download_one(url: str, out_path: Path, timeout=(5, 60)):
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            return True, ""

        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp_path, out_path)

        return True, ""
    except Exception as e:
        return False, str(e)

def download_all(df: pd.DataFrame, root_dir: str = ".", max_workers: int = 16):
    root = Path(root_dir)
    results = [None] * len(df)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, row in df.reset_index(drop=True).iterrows():
            url = row["url"]
            rel_path = Path(str(row["processed_path"])) if 'processed_path' in row else f'test/processed/{i}.jpg'
            out_path = root / rel_path
            futures.append((i, ex.submit(download_one, url, out_path)))

        for i, fut in tqdm(futures, total=len(futures), desc="Downloading"):
            ok, err = fut.result()
            results[i] = (ok, err)

    df2 = df.reset_index(drop=True).copy()
    df2["download_ok"] = [x[0] for x in results]
    df2["download_err"] = [x[1] for x in results]
    return df2

root_dir = "."

df = pd.read_csv("pseudo_labels.csv")
df_dl = download_all(df, root_dir=root_dir, max_workers=128)

df_dl.to_csv("df_dl.csv", index=False)

df1 = pd.read_excel("model_check.xlsx")
df1['processed_path'] = df1.index.map(lambda x: f'test/processed/{x}.jpg')
df1['url'] = df1['downloadUrl']

df_dl1 = download_all(df1, root_dir=root_dir, max_workers=128)

df_dl1.to_csv("df_dl1.csv", index=False)
