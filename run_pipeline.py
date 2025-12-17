import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


ROOT_DIR = "."

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

BATCH_SIZE = 256
SAVE_EVERY = 300

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

MAX_NEW_TOKENS = 96
MAX_NEW_TOKENS_RETRY = 96


BAD_RE = [
    re.compile(r"^\s*\d+\)", re.IGNORECASE),
    re.compile(r"\bnone\b", re.IGNORECASE),
    re.compile(r"\bnull\b", re.IGNORECASE),
]

GEN1 = {
    "do_sample": True,
    "temperature": 0.75,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
}
GEN2 = {"do_sample": False}

ADD_TREE_TAG_FROM_TEXT = True


PROMPT = (
    "Опиши изображение естественным русским языком одним коротким абзацем (1–2 предложения).\n"
    "Если на изображении есть дерево — упомяни его и скажи, на что оно больше похоже: хвойное или лиственное "
    "(или что не уверен). Если деревьев не видно — скажи об этом.\n"
    "Не используй нумерацию, списки, '1)', '2)', слова 'None', 'null', и не делай служебных пометок."
)

PROMPT_RETRY = (
    "Сделай одно связное описание (1–2 предложения) естественным русским языком.\n"
    "Если есть дерево — коротко поясни, скорее хвойное или лиственное (или не уверен). Если деревьев нет — скажи.\n"
    "Никаких списков, нумерации, 'None', 'null', кавычек вокруг всего текста."
)


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["vlm_text_ru", "vlm_raw", "vlm_err", "vlm_tree_tag"]:
        if c not in df.columns:
            df[c] = pd.Series([None] * len(df), dtype="object")
        else:
            df[c] = df[c].astype("object")

    if "vlm_ok" not in df.columns:
        df["vlm_ok"] = False
    df["vlm_ok"] = df["vlm_ok"].fillna(False).astype(bool)
    return df


def build_inputs(processor, image_paths, prompt_text: str):
    batch_messages = []
    for p in image_paths:
        img_uri = Path(p).resolve().as_uri()
        batch_messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_uri},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        )

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch_messages
    ]
    image_inputs, video_inputs = process_vision_info(batch_messages)

    return processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def qwen_infer_batch(
    model,
    processor,
    image_paths,
    prompt_text: str,
    max_new_tokens: int,
    gen_kwargs=None,
):
    if gen_kwargs is None:
        gen_kwargs = {}
    if not image_paths:
        return []
    try:
        inputs = build_inputs(processor, image_paths, prompt_text).to(model.device)
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, **gen_kwargs
            )
        trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)]
        return processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    except RuntimeError as e:
        raise


def clean_text(txt: str) -> str:
    if txt is None:
        return None
    t = txt.strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_bad_text(txt: str) -> bool:
    if txt is None:
        return True
    t = txt.strip()
    if len(t) < 8:
        return True
    for r in BAD_RE:
        if r.search(t):
            return True
    return False


def tree_tag_from_text_freeform(txt: str) -> str:
    if not txt:
        return "unknown"
    t = txt.lower()

    if re.search(
        r"(дерев(ьев|а)\s+не\s+(видно|заметно)|без\s+дерев(ьев|а)|дерев(ьев|а)\s+нет)",
        t,
    ):
        return "none"

    if "пальм" in t:
        return "palm"

    if re.search(r"(хвойн|ель|сосн|пихт|туя|можжевел)", t):
        return "conifer"

    if re.search(r"(листвен|берез|дуб|клен|топол|осин|рябин)", t):
        return "deciduous"

    if "дерев" in t:
        return "unknown"

    return "unknown"


def run_pipeline(input_csv):
    df = pd.read_csv(input_csv)
    df = ensure_cols(df)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype="auto"
    ).eval()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    root = Path(ROOT_DIR)

    mask = (df.get("download_ok", True) == True) & (df["vlm_ok"] != True)

    idxs = df.index[mask].tolist()
    print("To process:", len(idxs))

    processed = 0
    for s in tqdm(range(0, len(idxs), BATCH_SIZE), desc="Qwen2-VL free caption"):
        batch_idxs = idxs[s : s + BATCH_SIZE]

        paths, good = [], []
        for i in batch_idxs:
            p = root / Path(str(df.at[i, "processed_path"]))
            if p.exists():
                paths.append(str(p))
                good.append(i)
            else:
                df.at[i, "vlm_ok"] = False
                df.at[i, "vlm_err"] = f"missing_file: {p}"

        if not paths:
            continue

        try:
            outs = qwen_infer_batch(
                model, processor, paths, PROMPT, MAX_NEW_TOKENS, gen_kwargs=GEN1
            )
        except Exception as e:
            outs = [None] * len(good)
            for i in good:
                df.at[i, "vlm_ok"] = False
                df.at[i, "vlm_err"] = f"generate_error: {e}"

        retry_paths, retry_idxs = [], []
        for i, txt in zip(good, outs):
            df.at[i, "vlm_raw"] = txt
            txt2 = clean_text(txt)

            if is_bad_text(txt2):
                df.at[i, "vlm_err"] = "bad_text_pass1"
                retry_idxs.append(i)
                retry_paths.append(str(root / Path(str(df.at[i, "processed_path"]))))
                continue

            df.at[i, "vlm_text_ru"] = txt2
            df.at[i, "vlm_ok"] = True
            df.at[i, "vlm_err"] = ""
            if ADD_TREE_TAG_FROM_TEXT:
                df.at[i, "vlm_tree_tag"] = tree_tag_from_text_freeform(txt2)

        if retry_paths:
            try:
                outs2 = qwen_infer_batch(
                    model,
                    processor,
                    retry_paths,
                    PROMPT_RETRY,
                    MAX_NEW_TOKENS_RETRY,
                    gen_kwargs=GEN2,
                )
            except Exception as e:
                outs2 = [None] * len(retry_idxs)
                for i in retry_idxs:
                    df.at[i, "vlm_ok"] = False
                    df.at[i, "vlm_err"] = f"retry_generate_error: {e}"

            for i, txt in zip(retry_idxs, outs2):
                df.at[i, "vlm_raw"] = txt
                txt2 = clean_text(txt)

                if is_bad_text(txt2):
                    df.at[i, "vlm_ok"] = False
                    df.at[i, "vlm_err"] = "bad_text_pass2"
                    continue

                df.at[i, "vlm_text_ru"] = txt2
                df.at[i, "vlm_ok"] = True
                df.at[i, "vlm_err"] = ""
                if ADD_TREE_TAG_FROM_TEXT:
                    df.at[i, "vlm_tree_tag"] = tree_tag_from_text_freeform(txt2)

        processed += len(good)

    return df


df_out_train = run_pipeline("df_dl.csv")
df_out_test = run_pipeline("df_dl1.csv")

df_out_train.to_csv("vlm_caption_train.csv")
df_out_test.to_csv("vlm_caption_test.csv")
