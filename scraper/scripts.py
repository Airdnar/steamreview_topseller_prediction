import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
import os
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from transformers import AutoTokenizer
import stopwordsiso as stopwordsiso


GLOBAL_DICTIONARY = None
GLOBAL_LDA        = None

EN_STOPWORDS = set(stopwords.words('english'))
CH_STOPWORDS = stopwordsiso.stopwords("zh")
RU_STOPWORDS = set(stopwords.words('russian'))
CUSTOM_STOPWORDS = {
    'english': {
        'game', 'games', 'gaming', 'gamed',
        'play', 'plays', 'playing', 'played',
    },
    'russian': {
        'игра', 'игры', 'игре', 'игру', 'игрой', 'играх',
        'играть', 'поиграть',
        'играю', 'играешь', 'играем', 'играете', 'играют',
        'играл', 'играла', 'играло', 'играли',
    },
    'zh': {
        '游戏',
        '玩', '玩耍', '嬉戏', '游玩',
    }
}
EN_STOPWORDS.update(CUSTOM_STOPWORDS['english'])
RU_STOPWORDS.update(CUSTOM_STOPWORDS['russian'])
CH_STOPWORDS.update(EN_STOPWORDS)  # add English stopwords to Chinese as they were found quite often
CH_STOPWORDS.update(CUSTOM_STOPWORDS['zh'])

def fetch_steam_reviews(
    appid: int,
    max_reviews: int = 100,
    save_to_csv: bool = True,
    output_folder: str = None,
    sleep_time: float = 1.0,
    languages: list[str] = None
) -> pd.DataFrame:
    """
    Fetch up to `max_reviews` recent Steam reviews per language in `languages`.
    Returns a DataFrame of all reviews with columns:
      - author_steamid
      - recommended
      - review_text
      - timestamp_created (as datetime)
      - language
    """
    # default to English, Simplified Chinese, and Russian if not specified
    if languages is None:
        languages = ['english', 'schinese', 'russian']

    all_reviews = []
    headers = {'User-Agent': 'Mozilla/5.0'}

    for lang in languages:
        cursor = '*'
        collected = []
        while len(collected) < max_reviews:
            params = {
                'json': 1,
                'filter': 'recent',
                'language': lang,
                'cursor': cursor,
                'num_per_page': 20
            }
            resp = requests.get(
                f'https://store.steampowered.com/appreviews/{appid}',
                params=params,
                headers=headers
            )
            if resp.status_code != 200:
                print(f"[{lang}] HTTP {resp.status_code}, stopping fetch for this language.")
                break

            data = resp.json()
            batch = data.get('reviews', [])
            if not batch:
                break

            collected.extend(batch)
            cursor = data.get('cursor', '')
            if not cursor:
                break

            time.sleep(sleep_time)

        # truncate to max_reviews and normalize
        for r in collected[:max_reviews]:
            all_reviews.append({
                'author_steamid':     r['author']['steamid'],
                'recommended':        r['voted_up'],
                'review_text':        r['review'],
                'timestamp_created':  r['timestamp_created'],
                'language':           lang
            })

    # build DataFrame and dedupe
    df = pd.DataFrame(all_reviews)
    if not df.empty:
        df.drop_duplicates(subset=['author_steamid','timestamp_created'], inplace=True)
        df['timestamp_created'] = pd.to_datetime(df['timestamp_created'], unit='s')

        if save_to_csv:
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                fname = os.path.join(output_folder, f'steam_reviews_{appid}.csv')
            else:
                fname = f'steam_reviews_{appid}.csv'
            df.to_csv(fname, index=False)
            print(f"Saved {len(df)} reviews (langs={languages}) to {fname}")

    return df


import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

def fetch_steam_top_sellers(pages=4,
                            enrich_details=False,
                            save_to_csv=True,
                            output_folder=None,
                            sleep_time=1.5):
    """
    Fetches top-selling games from Steam, with an overall ranking.

    Parameters:
        pages (int): Number of pages to fetch (25 games per page).
        enrich_details (bool): Whether to scrape tags and genres from individual game pages.
        save_to_csv (bool): Whether to save the result as a CSV file.
        output_folder (str or None): Folder to save the CSV file into.
        sleep_time (float): Time in seconds to wait between requests.

    Returns:
        pd.DataFrame: DataFrame containing top-selling game data, including a 'rank' column.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    games_data = []
    rank_counter = 1

    # Step 1: Fetch top-selling games, page by page
    for page in range(1, pages + 1):
        print(f"Fetching page {page}...")
        url = f"https://store.steampowered.com/search/?filter=topsellers&page={page}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        game_rows = soup.find_all("a", class_="search_result_row")

        for row in game_rows:
            title_tag = row.find("span", class_="title")
            release_tag = row.find("div", class_="search_released")
            price_tag = row.find("div", class_="search_price")
            review_tag = row.find("span", class_="search_review_summary")
            href = row.get('href', '')

            games_data.append({
                "rank": rank_counter,
                "title": title_tag.text.strip() if title_tag else None,
                "release_date": release_tag.text.strip() if release_tag else None,
                "price": price_tag.text.strip() if price_tag else None,
                "review_summary": review_tag['data-tooltip-html'].strip()
                                   if review_tag and review_tag.has_attr('data-tooltip-html')
                                   else None,
                "url": href,
                "appid": href.split("/app/")[1].split("/")[0]
                         if "/app/" in href else None
            })
            rank_counter += 1

        time.sleep(sleep_time)

    steam_top_sellers = pd.DataFrame(games_data)

    # Optionally save to CSV
    if save_to_csv:
        filename = "steam_top_sellers.csv"
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            filename = f"{output_folder}/{filename}"
        steam_top_sellers.to_csv(filename, index=False)
        print(f"Saved top sellers to {filename}")

    return steam_top_sellers


    # Step 2: Optionally enrich each game with tags and genres
    def extract_game_details(url):
        try:
            game_resp = requests.get(url, headers=headers)
            game_soup = BeautifulSoup(game_resp.text, "html.parser")

            tags = [tag.text.strip() for tag in game_soup.find_all("a", class_="app_tag")]

            genres = []
            details = game_soup.find("div", class_="details_block")
            if details:
                for a in details.find_all("a", href=True):
                    if "/genre/" in a["href"]:
                        genres.append(a.text.strip())

            return {
                "tags": ", ".join(tags),
                "genres": ", ".join(genres)
            }
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return {"tags": None, "genres": None}

    if enrich_details:
        for i, row in steam_top_sellers.iterrows():
            print(f"Enriching {i+1}/{len(steam_top_sellers)}: {row['title']}")
            extra = extract_game_details(row["url"])
            steam_top_sellers.at[i, "tags"] = extra["tags"]
            steam_top_sellers.at[i, "genres"] = extra["genres"]
            time.sleep(sleep_time)

    # Step 3: Optionally save to CSV
    if save_to_csv:
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            filename = os.path.join(output_folder, 'steam_top_sellers.csv')
        else:
            filename = 'steam_top_sellers.csv'

        steam_top_sellers.to_csv(filename, index=False)
        print(f"Saved data to {filename}")

    return steam_top_sellers

def tabulsai_analysis(df, max_reviews=100):
    """
    Perform sentiment analysis on a sample of Steam reviews using tabularisai's multilingual-sentiment-analysis model.
    Parameters:
        df (pd.DataFrame): DataFrame containing Steam reviews with a 'review_text' column.
        max_reviews (int): Maximum number of reviews to analyze.
    """
   # sets up the sentiment analysis pipeline using a pre-trained model
    pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    
    # Prepare result list
    results = []

    for comment in df['review_text'][:max_reviews]:
        try:
            result = pipe(comment)[0]  # Get the first result (a dict with label and score)
            results.append({
                'review_text': comment,
                'label': result['label'],
                'score': result['score']
            })
        except Exception as e:
            print(f"Error processing comment: {comment[:30]}... -> {e}")

    return pd.DataFrame(results)

def bertemotion_analysis(df: pd.DataFrame,
                         text_column: str = 'review_text',
                         max_texts: int = 100) -> pd.DataFrame:
    """
    Perform emotion classification on a sample of texts using boltuix/bert-emotion,
    with sliding-window chunking to handle texts longer than 512 tokens.

    Returns a DataFrame with columns:
      - text: the original text
      - label: the predicted emotion label
      - score: the model's confidence score for that label
    """
    # 1) load a fast tokenizer and the pipeline (return full score distributions)
    fast_tok = AutoTokenizer.from_pretrained("boltuix/bert-emotion", use_fast=True)
    pipe     = pipeline(
        "text-classification",
        model="boltuix/bert-emotion",
        tokenizer=fast_tok,
        return_all_scores=True
    )

    results = []
    for txt in df[text_column].astype(str)[:max_texts]:
        try:
            # 2) split into overlapping 512-token chunks
            enc = fast_tok(
                txt,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_overflowing_tokens=True,
                stride=256
            )
            # 3) decode chunks back to plain text
            chunk_texts = fast_tok.batch_decode(enc["input_ids"], skip_special_tokens=True)

            # 4) classify each chunk (pipeline returns list-of-lists of {label, score})
            chunk_outs = pipe(chunk_texts)

            # 5) aggregate chunk-level scores into one final prediction
            agg = {}
            for chunk in chunk_outs:
                for entry in chunk:
                    agg.setdefault(entry["label"], []).append(entry["score"])
            avg_scores    = {lbl: sum(scores)/len(scores) for lbl, scores in agg.items()}
            chosen_label  = max(avg_scores, key=avg_scores.get)
            chosen_score  = avg_scores[chosen_label]

            results.append({
                'text':  txt,
                'label': chosen_label,
                'score': chosen_score
            })
        except Exception as e:
            print(f"Error processing text: {txt[:30]}… -> {e}")

    return pd.DataFrame(results)

def distillbert_analysis(df: pd.DataFrame,
                         text_column: str = 'review_text',
                         max_texts: int = 100) -> pd.DataFrame:
    """
    Perform sentiment analysis on a sample of texts using distilbert-base-uncased-finetuned-sst-2-english.

    Returns a DataFrame with columns:
      - text: the original text
      - label: the predicted sentiment label (e.g., POSITIVE/NEGATIVE)
      - score: the model’s confidence score for that label
    """
    # load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english", use_fast=True
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model.eval()

    results = []
    for txt in df[text_column].astype(str)[:max_texts]:
        try:
            # sliding-window tokenization into overlapping 512-token chunks
            enc = tokenizer(
                txt,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
                return_overflowing_tokens=True,
                stride=256
            )
            input_ids    = enc["input_ids"]        # shape: (num_chunks, 512)
            attention_mask = enc["attention_mask"]

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs  = F.softmax(logits, dim=1)    # shape: (num_chunks, 2)

            # average probabilities across chunks
            avg_probs = probs.mean(dim=0)          # shape: (2,)
            pred_id   = int(avg_probs.argmax())
            label     = model.config.id2label[pred_id]
            score     = float(avg_probs[pred_id])

            results.append({
                'text':  txt,
                'label': label,
                'score': score
            })
        except Exception as e:
            print(f"Error processing text: {txt[:30]}… -> {e}")

    return pd.DataFrame(results)

def compute_sentiment_features(df: pd.DataFrame, method: str = 'distilbert') -> pd.Series:
    """
    Compute sentiment-based features for a single game's reviews.
    Parameters:
      - df: DataFrame with a 'review_text' column
      - method: 'vader' for NLTK VADER, 'distilbert' for DistilBERT SST-2
    Returns:
      Series with average, median, and distribution of sentiment scores.
    """
    if method == 'vader':
        sia = SentimentIntensityAnalyzer()
        scores = df['review_text'].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])
    elif method == 'distilbert':
        out = distillbert_analysis(df, text_column='review_text', max_texts=len(df))
        scores = out['score'] * out['label'].map({'POSITIVE': 1, 'NEGATIVE': -1})
    else:
        raise ValueError("Unknown sentiment method")

    return pd.Series({
        'avg_sentiment': scores.mean(),
        'median_sentiment': scores.median(),
        'positive_frac': (scores > 0).mean(),
        'negative_frac': (scores < 0).mean(),
        'neutral_frac': (scores == 0).mean(),
    })

def compute_emotion_features(df: pd.DataFrame, max_texts: int = None) -> pd.Series:
    """
    Compute emotion category distribution from reviews.
    Parameters:
      - df: DataFrame with 'review_text'
      - max_texts: optional cap on number of reviews to analyze
    Returns:
      Series with fraction of each emotion label.
    """
    em_df = bertemotion_analysis(df, text_column='review_text', max_texts=max_texts or len(df))
    dist = em_df['label'].value_counts(normalize=True)
    # Prefix emotion_ to each
    return dist.rename(lambda x: f'emotion_{x}')

def compute_length_features(df: pd.DataFrame) -> pd.Series:
    """
    Compute average review length and fraction of long-form reviews.
    """
    lengths = df['review_text'].astype(str).apply(lambda x: len(x.split()))
    return pd.Series({
        'avg_review_length': lengths.mean(),
        'long_review_frac': (lengths > 100).mean(),
    })



# Globals to hold per‐language models
LANGUAGE_DICTS = {}
LANGUAGE_LDAS  = {}

def init_language_topic_models(
    dfs_by_lang: dict[str, pd.DataFrame],
    num_topics: int = 5,
    passes: int = 10,
    workers: int = None,
    no_below: int = 5,
    no_above: float = 0.5,
    keep_n: int = 10000
) -> None:
    """
    Given a mapping of language code -> DataFrame of all reviews in that language,
    trains one LDA per language and stores them in LANGUAGE_DICTS / LANGUAGE_LDAS.
    """
    global LANGUAGE_DICTS, LANGUAGE_LDAS
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    for lang, df in dfs_by_lang.items():
        # choose stop-words for this language
        if lang == 'english':
            sw = EN_STOPWORDS
        elif lang in ('schinese', 'chinese'):
            sw = CH_STOPWORDS
        elif lang == 'russian':
            sw = RU_STOPWORDS
        else:
            raise ValueError(f"Unsupported language for topic modeling: {lang}")

        # tokenize all reviews for this language
        texts = []
        for txt in df['review_text'].astype(str):
            tokens = [
                w for w in word_tokenize(txt.lower())
                if w.isalpha() and w not in sw
            ]
            texts.append(tokens)

        # build dictionary & corpus
        dictionary = Dictionary(texts)

        dictionary.filter_extremes(
            no_below=no_below,    # keep tokens in ≥ no_below docs
            no_above=no_above,    # drop tokens in > no_above fraction of docs
            keep_n=keep_n         # cap vocab size to keep_n most frequent tokens
        )

        bow_corpus = [dictionary.doc2bow(t) for t in texts]

        # train LDA
        lda = LdaMulticore(
            bow_corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=passes,
            workers=workers
        )

        # store
        LANGUAGE_DICTS[lang] = dictionary
        LANGUAGE_LDAS[lang]  = lda


def compute_topic_features(
    df: pd.DataFrame,
    language: str,
    num_topics: int = 5
) -> pd.Series:
    """
    Compute topic fraction features for a single game's reviews in one language.
    Must have called init_language_topic_models(...) first.
    """
    if language not in LANGUAGE_LDAS:
        raise RuntimeError(f"No LDA found for language '{language}'. Did you call init_language_topic_models?")

    dictionary = LANGUAGE_DICTS[language]
    lda        = LANGUAGE_LDAS[language]

    # choose stop-words
    if language == 'english':
        sw = EN_STOPWORDS
    elif language in ('schinese', 'chinese'):
        sw = CH_STOPWORDS
    else:  # 'russian'
        sw = RU_STOPWORDS

    # count dominant topic per review
    counts = {i: 0 for i in range(num_topics)}
    for txt in df['review_text'].astype(str):
        tokens = [
            w for w in word_tokenize(txt.lower())
            if w.isalpha() and w not in sw and w in dictionary.token2id
        ]
        bow    = dictionary.doc2bow(tokens)
        topics = lda.get_document_topics(bow)
        if topics:
            main = max(topics, key=lambda x: x[1])[0]
            counts[main] += 1

    total = sum(counts.values()) or 1
    return pd.Series({f'topic_{i}_frac': counts[i] / total for i in range(num_topics)})

def extract_reg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given all reviews for a single game, compute a full feature vector.
    Returns a single-row DataFrame ready for regression input.
    """
    feats = pd.concat([
        compute_sentiment_features(df),
        compute_emotion_features(df),
        compute_length_features(df),
        compute_topic_features(df)
    ])
    return feats.to_frame().T
