import time
import os
import pandas as pd
import json

from scripts import (
    fetch_steam_top_sellers,
    fetch_steam_reviews,
    init_language_topic_models,
    compute_topic_features,
    compute_sentiment_features,
    compute_emotion_features,
    compute_length_features,
    LANGUAGE_LDAS
)

NUM_TOPICS = 5  # Number of topics for each LDA
TOP_N = 20      # Top-N words per topic for inspection

def main():
    # Step 1: Fetch top-selling games
    top_sellers = fetch_steam_top_sellers(
        pages=2,
        enrich_details=False,
        save_to_csv=False,
        sleep_time=1.5
    )
    top_25 = top_sellers.head(25)
    os.makedirs('data', exist_ok=True)
    top_25.to_csv('data/top_25_sellers.csv', index=False)
    print("Fetched top 5 sellers:\n", top_25[['appid','title']])

    # Step 2: Download reviews per language for each game
    os.makedirs('data/reviews', exist_ok=True)
    for _, row in top_25.iterrows():
        appid, title = row['appid'], row['title']
        print(f"\nFetching reviews for {title} (AppID {appid})…")
        try:
            # languages defaults to ['english','schinese','russian']
            reviews_df = fetch_steam_reviews(
                appid,
                max_reviews=100,
                save_to_csv=True,
                output_folder='data/reviews',
                sleep_time=1
            )
            print(f"  → Retrieved {len(reviews_df)} total reviews (all langs).")
        except Exception as e:
            print(f"  ✖ Failed to fetch reviews for {appid}: {e}")
            continue
        time.sleep(2)

    # Step 3: Build one LDA per language across all games
    # collect all reviews per language into a dict of DataFrames
    languages = ['english', 'schinese', 'russian']
    dfs_by_lang = {}
    for lang in languages:
        frames = []
        for appid in top_25['appid']:
            path = f'data/reviews/steam_reviews_{appid}.csv'
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df_lang = df[df['language'] == lang]
            if not df_lang.empty:
                frames.append(df_lang)
        dfs_by_lang[lang] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # initialize per-language LDA models
    init_language_topic_models(dfs_by_lang, num_topics=NUM_TOPICS)
    print(f"Built {len(languages)} LDA models (topics={NUM_TOPICS}).")

    # Save top-N terms for each language’s LDA
    os.makedirs('data/features', exist_ok=True)
    TOP_N = 20
    for lang in languages:
        lda_model = LANGUAGE_LDAS.get(lang)
        if lda_model is None:
            print(f"No LDA model for {lang}, skipping topic dump.")
            continue

        topic_terms = {}
        for topic_id in range(NUM_TOPICS):
            terms = lda_model.show_topic(topic_id, topn=TOP_N)
            topic_terms[f"topic_{topic_id}"] = [
                {"word": w, "weight": float(p)} for w, p in terms
            ]

        json_path = f"data/features/{lang}_test_topic_terms.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(topic_terms, jf, indent=2, ensure_ascii=False)
        print(f"Saved {lang} topic terms to {json_path}")

    # Step 4: Extract features per game & per language, then merge
    all_records = []
    for _, row in top_25.iterrows():
        appid, title = row['appid'], row['title']
        path = f'data/reviews/steam_reviews_{appid}.csv'
        if not os.path.exists(path):
            print(f"  ✖ No reviews for {appid}, skipping.")
            continue

        # load the mixed-language reviews
        df_all = pd.read_csv(path)

        # for each language, filter, extract features, prefix keys
        record = {'appid': appid, 'title': title}
        for lang in languages:
            df_lang = df_all[df_all['language'] == lang]
            if df_lang.empty:
                continue
            # basic features
            sent = compute_sentiment_features(df_lang).to_dict()
            emo  = compute_emotion_features(df_lang).to_dict()
            leng = compute_length_features(df_lang).to_dict()
            # topic features for this lang
            topf = compute_topic_features(df_lang, language=lang).to_dict()
            # merge with language prefix
            for k,v in {**sent, **emo, **leng, **topf}.items():
                record[f"{lang}_{k}"] = v

        all_records.append(record)

    # save the merged feature matrix
    combined = pd.DataFrame(all_records)
    combined.to_csv('data/features/games_multilang_features.csv', index=False)
    print("Saved multilingual feature matrix to data/features/games_multilang_features.csv")

if __name__ == "__main__":
    main()
