# Pakistan Social Audience & Campaign Intelligence Engine

Clustering-based engine that groups audience comments, flags unusual posts and generates marketing recommendations for a Pakistani women's lawn-suit brand (Instagram, TikTok, Facebook).

**Live web app:** https://ahmad-f2023376355.github.io/Big-Data-Project-3/  (GitHub Pages, serves `docs/index.html`)

> **Data note:** the demo uses SIMULATED data (`data_source = simulated`), not real brand measurements.

## Five questions answered
1. Which content attracts attention? (engagement rate by content type and platform)
2. What does the audience want? (topic tagging in English / Roman Urdu / Urdu)
3. Which posts are unusual? (robust z-score within peer groups + Isolation Forest)
4. Which audience needs should the brand address? (KMeans on TF-IDF word + character n-grams)
5. What should marketing do next? (evidence-based recommendations)

## Run locally
```
pip install -r requirements.txt
python pipeline.py            # data/ and output/
python build_web.py           # rebuilds the static web app (set the output path inside the script)
streamlit run app.py          # optional interactive dashboard
```
Real data: `python pipeline.py --real posts.csv comments.csv` (columns as in `data/posts_simulated.csv`, `data/comments_simulated.csv`). Collect data only as platform terms allow.

## Limitations
Simulated data shows the pipeline works, not real-world accuracy. Roman Urdu spelling varies widely; a multilingual embedding model (e.g. LaBSE) is a natural upgrade.
