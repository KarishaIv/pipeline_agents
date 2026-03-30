import os
import json
from typing import Any, Dict, List, Optional

import pandas as pd

CHANNEL_MAPPING = {
    # 1. MACRO_OFFICIAL: Фундаментальные данные, ЦБ, Министерства
    'centralbank_russia': 'MACRO_OFFICIAL',
    'minec_russia': 'MACRO_OFFICIAL',
    'russianmacro': 'MACRO_OFFICIAL',
    'ecworld': 'MACRO_OFFICIAL',
    'ecworldtech': 'MACRO_OFFICIAL',
    'economika': 'MACRO_OFFICIAL',
    'visual_capitalist_rus': 'MACRO_OFFICIAL',
    'Econsonline': 'MACRO_OFFICIAL',
    'gosuslugi': 'MACRO_OFFICIAL',

    # 2. NEWS_SOCIAL: Общий фон, Паника, Регионы 
    'mash': 'NEWS_SOCIAL',
    'interfaxonline': 'NEWS_SOCIAL',
    'tass_agency': 'NEWS_SOCIAL',
    'vedomosti': 'NEWS_SOCIAL',
    'rian_ru': 'NEWS_SOCIAL',
    'gazetaru': 'NEWS_SOCIAL',
    'izvestia': 'NEWS_SOCIAL',
    'bbcrussian': 'NEWS_SOCIAL',
    'kommersant': 'NEWS_SOCIAL',
    'rt_russian': 'NEWS_SOCIAL',
    # Регионы
    'moscowach': 'NEWS_SOCIAL',
    'spb_smi': 'NEWS_SOCIAL',
    'ngs_news': 'NEWS_SOCIAL',
    'kazan': 'NEWS_SOCIAL',
    'podslushano_chat52': 'NEWS_SOCIAL',
    'news_74ru': 'NEWS_SOCIAL',
    'yug_24_ru': 'NEWS_SOCIAL',
    'samara_smi': 'NEWS_SOCIAL',

    # 3. FINANCE_MARKETS: Банки, Биржа, Инвестиции, Валюта
    # Банки
    'alfabank': 'FINANCE_MARKETS',
    'sberbank': 'FINANCE_MARKETS',
    'bankvtb': 'FINANCE_MARKETS',
    'tbank': 'FINANCE_MARKETS',
    'gazprombank': 'FINANCE_MARKETS',
    # Рынки
    'MoscowExchangeOfficial': 'FINANCE_MARKETS',
    'cbonds': 'FINANCE_MARKETS',
    'dohod': 'FINANCE_MARKETS',
    'Bonds_lab': 'FINANCE_MARKETS',
    'russianjunkbonds': 'FINANCE_MARKETS',
    # Инвест-каналы и Аналитика
    'smartlabnews': 'FINANCE_MARKETS',
    'investfundsru': 'FINANCE_MARKETS',
    'CFA_RF': 'FINANCE_MARKETS',
    'tb_invest_official': 'FINANCE_MARKETS',
    'investnique': 'FINANCE_MARKETS',
    'banksta': 'FINANCE_MARKETS', 
    'ex_fin': 'FINANCE_MARKETS',
    'bitkogan': 'FINANCE_MARKETS',
    'd_code': 'FINANCE_MARKETS',
    # Валюта
    'sovcomrates_msk': 'FINANCE_MARKETS',
    'sberometer_kurs': 'FINANCE_MARKETS',
    'currency_rates': 'FINANCE_MARKETS',

    # 4. REAL_ESTATE: Ипотека и Недвижка
    'rusipoteka': 'REAL_ESTATE',
    'ipotekahouse': 'REAL_ESTATE',
    'regcik': 'REAL_ESTATE',
    'ipotekacenter': 'REAL_ESTATE',
    'cian_official': 'REAL_ESTATE'
}


# --- АГЕНТЫ ---
CURRENCY_CHANNELS = {
    'sovcomrates_msk',
    'sberometer_kurs',
    'currency_rates',
}

AGENT_BY_CATEGORY = {
    'MACRO_OFFICIAL': 'macroeconomy',
    'NEWS_SOCIAL': 'social_news',
    'REAL_ESTATE': 'real_estate',
    'FINANCE_MARKETS': 'banks',
    'OTHER': 'social_news',
}


def agent_for_channel(channel_name: str, category: str) -> str:
    if category == 'FINANCE_MARKETS' and channel_name in CURRENCY_CHANNELS:
        return 'currency'
    return AGENT_BY_CATEGORY.get(category, 'social_news')


def _is_ad(text: str) -> bool:
    t = text.lower()
    return ("erid:" in t) or ("#реклама" in t)


def build_posts_df(source_folder: str) -> pd.DataFrame:
    all_posts: List[Dict[str, Any]] = []
    files = [f for f in os.listdir(source_folder) if f.endswith('.json')]

    for filename in files:
        channel_name = filename[:-5]  # strip .json
        category = CHANNEL_MAPPING.get(channel_name, 'OTHER')
        agent = agent_for_channel(channel_name, category)

        file_path = os.path.join(source_folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, list):
            continue

        for post in data:
            text = str(post.get('text', '') or '').strip()
            if not text:
                continue
            if _is_ad(text):
                continue

            all_posts.append({
                "date": post.get("date"),
                "channel": channel_name,
                "category": category,
                "agent": agent,
                "text": text,
                "views": post.get("views", 0),
                "post_id": post.get("id"),
            })

    df = pd.DataFrame(all_posts)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date", ascending=False)
    return df


def build_rag_docs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "doc_id", "text", "metadata_json",
            "source", "channel", "category", "agent", "date", "views", "post_id",
        ])

    out = df.copy()

    def make_doc_id(row: pd.Series) -> str:
        post_id = row.get("post_id")
        channel = row.get("channel")
        if pd.notna(post_id) and str(post_id) != "":
            return f"telegram:{channel}:{post_id}"
        return f"telegram:{channel}:row:{row.name}"

    out["doc_id"] = out.apply(make_doc_id, axis=1)
    out["source"] = "telegram"

    def to_iso(val: Any) -> Optional[str]:
        if pd.isna(val):
            return None
        return pd.Timestamp(val).isoformat()

    out["_date_iso"] = out["date"].apply(to_iso)

    def make_metadata(row: pd.Series) -> str:
        meta = {
            "source": row.get("source"),
            "channel": row.get("channel"),
            "category": row.get("category"),
            "agent": row.get("agent"),
            "date": row.get("_date_iso"),
            "views": int(row.get("views", 0) or 0),
            "post_id": row.get("post_id"),
        }
        return json.dumps(meta, ensure_ascii=False)

    out["metadata_json"] = out.apply(make_metadata, axis=1)
    out = out.drop(columns=["_date_iso"])

    out = out[[
        "doc_id",
        "text",
        "metadata_json",
        "source",
        "channel",
        "category",
        "agent",
        "date",
        "views",
        "post_id",
    ]]
    return out


def save_parquet(df: pd.DataFrame, output_parquet: str) -> None:
    if not output_parquet:
        return
    print(f"Saving to {output_parquet}")
    df.to_parquet(output_parquet, index=False)


def save_jsonl(df_rag: pd.DataFrame, output_jsonl: str) -> None:
    if not output_jsonl:
        return
    print(f"Saving to {output_jsonl}")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in df_rag.iterrows():
            f.write(json.dumps({
                "id": row["doc_id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata_json"]),
            }, ensure_ascii=False) + "\n")


def prepare_telegram_rag(
    source_folder: str,
    output_parquet: str = "rag_docs.parquet",
    output_jsonl: str = "",
) -> pd.DataFrame:
    posts = build_posts_df(source_folder)
    rag = build_rag_docs(posts)
    save_parquet(rag, output_parquet)
    save_jsonl(rag, output_jsonl)
    return rag


if __name__ == "__main__":
    prepare_telegram_rag("telegram_data_last_day")
