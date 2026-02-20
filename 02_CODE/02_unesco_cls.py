import pandas as pd
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
THRESHOLD = 0.4

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

def load_unesco_keywords(xlsx_path: Path, sheet_name: str = "UNESCO_CRI_KW"):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    df = df.iloc[:, :2].copy()
    df.columns = ["Category", "Keyword"]
    header_candidates = {"UNESCO AI CFS", "Framework components", "Category", "Keyword"}
    non_categories = {"Demension", "Dimension"}
    df = df[~df["Category"].isin(header_candidates.union(non_categories))]
    df = df.dropna(subset=["Category", "Keyword"])
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Keyword"] = df["Keyword"].astype(str).str.strip()
    cat2keywords = {}
    for _, row in df.iterrows():
        cat = row["Category"]
        kw = row["Keyword"]
        cat2keywords.setdefault(cat, []).append(kw)
    return cat2keywords

def build_category_embeddings(cat2keywords, model):
    cat2embeds = {}
    for cat, kw_list in cat2keywords.items():
        unique_kw = list(dict.fromkeys(kw_list))
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
            #cat_mean_sims = {} # optional calculations (mean value), not used this study;
            cat_max_sims = {}
            for cat, cat_emb in cat2embeds.items():
                cos_scores = util.cos_sim(emb_q, cat_emb)[0]
                #cat_mean_sims[cat] = float(cos_scores.mean()) # optional calculations (mean value), not used this study;
                cat_max_sims[cat] = float(cos_scores.max())

            #best_cat_mean = max(cat_mean_sims, key=cat_mean_sims.get) # optional calculations (mean value), not used this study;
            #best_sim_mean = cat_mean_sims[best_cat_mean] # optional calculations (mean value), not used this study;

            best_cat_max = max(cat_max_sims, key=cat_max_sims.get)
            best_sim_max = cat_max_sims[best_cat_max]

            #label_mean = best_cat_mean # optional calculations (mean value), not used this study;
            label_max = best_cat_max
            if best_sim_max < THRESHOLD:
                label_mean = "Others"
                label_max = "Others"
            row_out = {
                "RowIndex_in_41Papers": row_idx+1,
                "OriginalCell": raw_text,
                "Keyword": kw,
                #"BestCategory_mean": label_mean, # optional calculations (mean value), not used this study;
                #"BestSim_mean": best_sim_mean, # optional calculations (mean value), not used this study;
                "BestCategory_max": label_max,
                "BestSim_max": best_sim_max,
            }
            for cat in categories:
                #row_out[f"Sim_mean_{cat}"] = cat_mean_sims[cat] # optional calculations (mean value), not used this study;
                row_out[f"Sim_max_{cat}"] = cat_max_sims[cat]
            rows_out.append(row_out)
    return pd.DataFrame(rows_out)

def main():
    input_path = Path("/home/usr/SynologyDrive/00_Processing.xlsx") # Required step: Modify the file path to verify or replicate this code;
    output_path = Path("/home/usr/SynologyDrive/02_UNESCO_CLS_RES.xlsx") # Required step: Modify the file path to verify or replicate this code;

    cat2keywords = load_unesco_keywords(input_path, sheet_name="UNESCO_CRI_KW")

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
        df_result.to_excel(writer, sheet_name="UNESCO_Result", index=False)
    print(f"Complete: The UNESCO category classification results have been successfully saved in the 'UNESCO_Result' worksheet of the '{output_path}' file.")

if __name__ == "__main__":
    main()
