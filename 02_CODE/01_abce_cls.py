import pandas as pd
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
THRESHOLD = 0.4  # Threshold for "Others" Class;

def preprocess_text(text: str):
    if pd.isna(text):
        return []
    raw_keywords = str(text).split(",")
    clean_keywords = []
    for kw in raw_keywords:
        kw = kw.strip().lower()

        kw = re.sub(r"[^a-zA-Z0-9가-힣\s]", "", kw)
        kw = re.sub(r"\s+", " ", kw).strip()
        if kw:
            clean_keywords.append(kw)
    return clean_keywords

def load_abce_keywords(xlsx_path: Path, sheet_name: str = "ABCE_CRI_KW"):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Optional;
    if {"Category", "Keyword"}.issubset(df.columns):
        df_ck = df[["Category", "Keyword"]].copy()
    else:
        df_ck = df.iloc[:, :2].copy()
        df_ck.columns = ["Category", "Keyword"]
    df_ck = df_ck.dropna(subset=["Category", "Keyword"])
    df_ck["Category"] = df_ck["Category"].astype(str).str.strip()
    df_ck["Keyword"] = df_ck["Keyword"].astype(str).str.strip()
    cat2keywords = {}
    for _, row in df_ck.iterrows():
        cat = row["Category"]
        kw = row["Keyword"]
        cat2keywords.setdefault(cat, []).append(kw)
    return cat2keywords

def build_category_embeddings(cat2keywords, model):
    cat2embeds = {}
    for cat, kw_list in cat2keywords.items():
        unique_kw = list(dict.fromkeys(kw_list))  # Eliminating duplicated KW;
        embeds = model.encode(unique_kw, convert_to_tensor=True)
        cat2embeds[cat] = embeds
    return cat2embeds

def classify_each_keyword(df_sheet2, col_idx, cat2embeds, model):
    categories = list(cat2embeds.keys())
    rows_out = []
    for row_idx, row in df_sheet2.iterrows():
        raw_text = row.iloc[col_idx]
        keywords = preprocess_text(raw_text)
        if not keywords:
            continue
        for kw in keywords:
            emb_q = model.encode(kw, convert_to_tensor=True)
            cat_mean_sims = {} # optional calculations (mean value), not used this study;
            cat_max_sims = {}
            for cat, cat_emb in cat2embeds.items():
                cos_scores = util.cos_sim(emb_q, cat_emb)[0]

                # optional calculations (mean value), not used in this study;
                cat_mean_sims[cat] = float(cos_scores.mean())
                cat_max_sims[cat] = float(cos_scores.max())

            # optional calculations (mean value), not used in this study;
            best_cat_mean = max(cat_mean_sims, key=cat_mean_sims.get)
            best_sim_mean = cat_mean_sims[best_cat_mean]

            best_cat_max = max(cat_max_sims, key=cat_max_sims.get)
            best_sim_max = cat_max_sims[best_cat_max]

            # optional calculations (mean value), not used in this study;
            label_mean = best_cat_mean

            label_max = best_cat_max
            if best_sim_max < THRESHOLD:
                label_mean = "Others" # optional calculations (mean value), not used in this study;
                label_max = "Others"
            row_out = {
                "RowIndex_in_41Papers": row_idx+1,
                "Framework Elements Cell": raw_text,
                "Keywords": kw,
                #"BestCategory_mean": label_mean, # optional calculations (mean value), not used in this study;
                #"BestSim_mean": best_sim_mean, # optional calculations (mean value), not used in this study;
                "BestCategory_max": label_max,
                "BestSim_max": best_sim_max,
            }
            for cat in categories:
                #row_out[f"Sim_mean_{cat}"] = cat_mean_sims[cat] # optional calculations (mean value), not used in this study;
                row_out[f"Sim_max_{cat}"] = cat_max_sims[cat]
            rows_out.append(row_out)
    return pd.DataFrame(rows_out)

def main():
    input_path = Path("/home/usr/SynologyDrive/00_Processing.xlsx") # Required step: Modify the file path to verify or replicate this code;
    output_path = Path("/home/usr/SynologyDrive/01_ABCE_CLS_RES.xlsx") # Required step: Modify the file path to verify or replicate this code;

    cat2keywords = load_abce_keywords(input_path, sheet_name="ABCE_CRI_KW")

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    cat2embeds = build_category_embeddings(cat2keywords, model)
    df_sheet2 = pd.read_excel(input_path, sheet_name="41Papers") # Sheet2 is the initial name of the sheet '41Papers';
    df_result = classify_each_keyword(
        df_sheet2=df_sheet2,
        col_idx=2,
        cat2embeds=cat2embeds,
        model=model
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_result.to_excel(writer, sheet_name="ABCE_Result", index=False)
    print(f"Complete: The ABCE category classification results have been successfully saved in the 'ABCE_Result' worksheet of the '{output_path}' file.")

if __name__ == "__main__":
    main()
