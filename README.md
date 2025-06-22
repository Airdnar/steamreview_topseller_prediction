To gather the data, first scrape Steam's website by running scraper/feature_extraction_batch.py. This will go through Steam's top seller page and gather the top 100 best selling games as well as their respective reviews.
This script will scrape reviews in English, Russiam, and simplified Chinese as those are the languages with the most users on the platform. If you want to explore it for other languages you will need to modify scraper/scripts.py.
The feature extraction will also run the reviews through distillbert and bertemotion right away, as well as perform an LDA topic modelling per language.

Afterwards, run correlation_checker.py. To clean the data and remove features with high correlation. By default, the models will be trained on a dataset that has the features with high correlation removed (80% cutoff) but if you look into correlation_checker.py, you will see there are 2 other types of data: compact and trimmed.
Compact consolidates the emotion scores per languages into one value and trimmed then removes those compacted values that have high correlation between each other. I did not put it as default as scores between languages did not show strong correlation between each other and led to overfitted models.

Once the data is scraped and cleaned, you can run the model that interests you, the model will be compared to a baseline model (for example, in the case of the classifiers they will be compared to a "most frequent" guess.)

Credits to:
 - Sinan Talha Hascelik (2019)

Repo: http://www.apache.org/licenses/LICENSE-2.0 

- boltuix/bert-emotion

Website: https://huggingface.co/boltuix/bert-emotion 

- distilbert/distilbert-base-uncased-finetuned-sst-2-english

Website: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
