"""Project 3 - Pakistan Social Audience & Campaign Intelligence Engine.
Run:  python pipeline.py            (uses simulated demo data)
      python pipeline.py --real posts.csv comments.csv   (your own collected data)
All simulated rows carry data_source="simulated" - never present them as real measurements."""
import sys, re, json, random, os
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.ensemble import IsolationForest
from scipy.sparse import hstack

OUT = "output"; os.makedirs(OUT, exist_ok=True); os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(42); random.seed(42)

# ---------------------------------------------------------------- 1. DATA
CONTENT = {  # base views, like-rate, comment-rate, share-rate
 "Reel": (30000, .045, .006, .004), "Product photo": (9000, .035, .004, .001),
 "Influencer post": (22000, .040, .007, .003), "Styling video": (18000, .042, .005, .003),
 "Sale announcement": (14000, .030, .012, .002), "Eid campaign": (26000, .050, .008, .004)}
PLATFORMS = ["Instagram", "TikTok", "Facebook"]
BRANDS = ["Brand A", "Brand B", "Brand C"]
CAMPAIGNS = ["Eid Collection", "11.11 Sale", "12.12 Sale", "Summer Lawn"]
T = {  # topic -> (templates, language_type)
 "price_inquiry": [("Price please?","English"),("Price kya hai?","Roman Urdu"),("Rate bataein","Roman Urdu"),("What is the price of this suit?","English"),("قیمت کیا ہے؟","Urdu"),("Price?","English")],
 "price_complaint": [("Price bohat zyada hai","Roman Urdu"),("Too expensive for lawn","English"),("Itni mehngi kyun hai","Roman Urdu"),("Quality ke hisaab se price high hai","Roman Urdu"),("بہت مہنگا ہے","Urdu")],
 "discount": [("Any discount on this suit?","English"),("Sale kab lag rahi hai?","Roman Urdu"),("Discount milega?","Roman Urdu"),("Please give a promo code","English"),("کوئی ڈسکاؤنٹ ہے؟","Urdu")],
 "delivery": [("Delivery kitne din mein hogi?","Roman Urdu"),("Ordered last week, still waiting","English"),("Order abhi tak nahi aaya","Roman Urdu"),("How long does delivery take?","English"),("Tracking id kahan hai?","Roman Urdu")],
 "availability": [("Is this available in Karachi?","English"),("Stock mein hai?","Roman Urdu"),("Lahore store mein available hai?","Roman Urdu"),("Size M available?","English"),("Kya ye Islamabad mein milega?","Roman Urdu")],
 "fabric_quality": [("Fabric kaisa hai?","Roman Urdu"),("Is it pure lawn?","English"),("Colour fade to nahi hota?","Roman Urdu"),("Quality was bad last time","English"),("کپڑا کیسا ہے؟","Urdu")],
 "sizing": [("Size chart please","English"),("Large size ka length kitna hai?","Roman Urdu"),("Is it true to size?","English"),("Plus size available?","English")],
 "purchase_intent": [("Inbox me details send kar dein","Roman Urdu"),("Want to order, how?","English"),("Order karna hai, link bhejein","Roman Urdu"),("DM sent, please reply","English"),("I need 2 pieces","English")],
 "praise": [("Very beautiful dress!","English"),("Bohat khoobsurat ❤️","Roman Urdu"),("Love the colour","English"),("Mashallah lajawab","Roman Urdu"),("بہت پیارا","Urdu")]}
# topic mix depends on campaign type (what makes the data "have structure")
BASE = dict(price_inquiry=.14,price_complaint=.06,discount=.08,delivery=.09,availability=.10,fabric_quality=.07,sizing=.06,purchase_intent=.10,praise=.30)
BOOST = {"11.11 Sale":{"discount":2.5,"price_inquiry":1.5,"delivery":1.6},"12.12 Sale":{"discount":2.5,"delivery":1.8},
         "Eid Collection":{"availability":1.8,"sizing":1.6,"praise":1.3},"Summer Lawn":{"fabric_quality":2.0,"price_complaint":1.4}}
NOISE = ["", "", " plz", " 🙏", " sis", " ??", " asap", " 😍"]

def simulate(n_posts=2500):
    rows = []
    for i in range(n_posts):
        ct = random.choice(list(CONTENT)); v0, lr, cr, sr = CONTENT[ct]
        plat = random.choice(PLATFORMS); camp = random.choice(CAMPAIGNS)
        views = int(rng.lognormal(np.log(v0), .6)) * (1.4 if plat == "TikTok" else 1)
        likes = int(views*lr*rng.lognormal(0,.25)); comm = int(views*cr*rng.lognormal(0,.3)); sh = int(views*sr*rng.lognormal(0,.3))
        rows.append(dict(post_id=f"P{i+1:04d}", brand=random.choice(BRANDS), platform=plat, campaign=camp, content_type=ct,
                         product_category="Lawn suit", likes=likes, comments_count=comm, shares=sh, views=int(views),
                         days_since_post=int(rng.integers(1,30)), data_source="simulated"))
    posts = pd.DataFrame(rows)
    # inject unusual posts (viral / controversial / suspiciously low) so the detector has something to find
    for idx in rng.choice(len(posts), 25, replace=False):
        kind = random.choice(["viral","controversy","flop"])
        if kind=="viral": posts.loc[idx,["likes","shares","views"]] = (posts.loc[idx,["likes","shares","views"]]*6).astype(int)
        elif kind=="controversy": posts.loc[idx,"comments_count"] = int(posts.loc[idx,"comments_count"]*9+300)
        else: posts.loc[idx,["likes","comments_count","shares"]] = (posts.loc[idx,["likes","comments_count","shares"]]*0.08).astype(int)
        posts.loc[idx,"planted_anomaly"] = kind
    com = []
    for _, p in posts.iterrows():
        w = {t: BASE[t]*BOOST.get(p.campaign,{}).get(t,1) for t in BASE}
        if p.get("planted_anomaly")=="controversy": w["price_complaint"]*=5; w["delivery"]*=3
        s = sum(w.values()); topics = list(w); pr = [w[t]/s for t in topics]
        for _ in range(int(np.clip(p.comments_count/4, 3, 14))):
            t = rng.choice(topics, p=pr); txt, lang = random.choice(T[t])
            sent = "Positive" if t in("praise","purchase_intent") else "Negative" if t in("price_complaint",) or (t=="delivery" and "still" in txt) else "Neutral"
            com.append(dict(post_id=p.post_id, comment_text=txt+random.choice(NOISE), language_type=lang, topic_true=t,
                            sentiment=sent, purchase_intent="High" if t=="purchase_intent" else "Possible inquiry" if t in("price_inquiry","availability","sizing","discount") else "No intent",
                            data_source="simulated"))
    return posts, pd.DataFrame(com)

# ---------------------------------------------------------------- 2. TEXT CLUSTERING
def clean(t):
    t = t.lower(); t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t); t = re.sub(r"\b(plz|please|pls|asap|sis|bhai|hai|kya|ka|ki|ko|is|the|this|to|of|a)\b", " ", t); return re.sub(r"\s+"," ",t).strip()

RULES = {  # transparent keyword tagging (Q2) - works on real data without labels
 "Price": r"price|rate|qeemat|قیمت|mehng|مہنگ|expensive|costly", "Discount": r"discount|sale|promo|offer|ڈسکاؤنٹ",
 "Delivery": r"deliver|tracking|order.*(wait|nahi|aaya)|still waiting|courier", "Availability": r"availab|stock|store|karachi|lahore|islamabad|milega",
 "Fabric/quality": r"fabric|lawn|colour|color|fade|quality|کپڑا", "Sizing": r"size|length|plus",
 "Purchase intent": r"inbox|dm|want to order|order karna|details|need \d", "Praise": r"beautiful|khoobsurat|love|mashallah|lajawab|پیارا"}
def rule_topic(t):
    for k, rx in RULES.items():
        if re.search(rx, t): return k
    return "Other"

def cluster_comments(com):
    com = com.copy(); com["clean"] = com.comment_text.map(clean)
    w = TfidfVectorizer(ngram_range=(1,2), min_df=3); c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=5, max_features=3000)
    X = hstack([w.fit_transform(com.clean), c.fit_transform(com.clean)]).tocsr()
    samp = rng.choice(X.shape[0], min(4000, X.shape[0]), replace=False); best = (-1, None)
    for k in range(6, 13):
        km = KMeans(k, n_init=5, random_state=42).fit(X[samp]); sc = silhouette_score(X[samp], km.labels_)
        if sc > best[0]: best = (sc, k)
    km = KMeans(best[1], n_init=10, random_state=42).fit(X); com["cluster"] = km.labels_
    terms = np.array(list(w.get_feature_names_out()) + list(c.get_feature_names_out()))
    com["rule_topic"] = com.clean.map(rule_topic)
    summ = []
    for k in range(best[1]):
        m = com.cluster == k
        top_terms = terms[np.argsort(-km.cluster_centers_[k][:len(w.get_feature_names_out())])[:6]]
        topic = com[m].rule_topic.value_counts()
        summ.append(dict(cluster=k, size=int(m.sum()), share_pct=round(100*m.mean(),1), dominant_topic=topic.index[0],
                         topic_purity_pct=round(100*topic.iloc[0]/m.sum()), top_terms=", ".join(top_terms),
                         example=com[m].comment_text.iloc[0], high_intent_pct=round(100*(com[m].purchase_intent=="High").mean(),1)))
    metrics = dict(k=best[1], silhouette=round(best[0],3))
    if "topic_true" in com: metrics["ARI_vs_simulated_labels"] = round(adjusted_rand_score(com.topic_true, com.cluster),3)
    return com, pd.DataFrame(summ), metrics

# ---------------------------------------------------------------- 3. CONTENT PERFORMANCE + ANOMALIES
def performance(posts):
    p = posts.copy(); p["engagement_rate"] = (p.likes+p.comments_count+p.shares)/p.views.clip(lower=1)
    p["comment_rate"] = p.comments_count/p.views.clip(lower=1); return p

def anomalies(p):
    feats = pd.DataFrame({"log_views":np.log1p(p.views),"eng":p.engagement_rate,"comment_rate":p.comment_rate,
                          "share_rate":p.shares/p.views.clip(lower=1),"like_rate":p.likes/p.views.clip(lower=1)})
    z = feats.groupby([p.platform, p.content_type]).transform(lambda s: (s-s.median())/(1.4826*(s-s.median()).abs().median()+1e-9))
    p = p.copy(); p["max_abs_z"] = z.abs().max(axis=1).round(1); p["driver"] = z.abs().idxmax(axis=1)
    p["direction"] = [("high" if z.loc[i,d]>0 else "low") for i,d in zip(z.index,p.driver)]
    iso = IsolationForest(contamination=0.03, random_state=42).fit(z.clip(-10,10)); p["iso_flag"] = iso.predict(z.clip(-10,10))==-1
    p["is_anomaly"] = p.iso_flag & (p.max_abs_z>=4)
    return p

# ---------------------------------------------------------------- 4. RECOMMENDATIONS
def recommend(p, com, summ):
    R = []; g = p.groupby("content_type").engagement_rate.mean().sort_values(ascending=False)
    R.append(("Content", f"'{g.index[0]}' has the highest average engagement rate ({g.iloc[0]*100:.1f}%) vs '{g.index[-1]}' ({g.iloc[-1]*100:.1f}%). Shift more of the next campaign's budget toward {g.index[0]} and A/B test '{g.index[-1]}' with a clearer call to action."))
    cc = p.groupby("content_type").comment_rate.mean().idxmax(); R.append(("Content", f"'{cc}' drives the most comments per view - use it for questions/polls, but staff the replies (see customer service)."))
    m = com.merge(p[["post_id","campaign"]], on="post_id"); sh = pd.crosstab(m.campaign, m.rule_topic, normalize="index")*100
    for camp in sh.index:
        top = sh.loc[camp].drop(["Praise","Other"],errors="ignore").sort_values(ascending=False)
        R.append(("Promotion", f"{camp}: top audience need is {top.index[0]} ({top.iloc[0]:.0f}% of comments), then {top.index[1]} ({top.iloc[1]:.0f}%). Put that answer in the first line of the caption/pinned comment."))
    d = (com.rule_topic=="Delivery").mean()*100; R.append(("Customer service", f"{d:.0f}% of comments are delivery questions/complaints. Add an auto-reply with delivery times and a tracking link; target a <2h first response on those threads."))
    pi = (com.purchase_intent=="High").mean()*100; R.append(("Sales", f"{pi:.0f}% of comments show clear purchase intent ('inbox me', 'order karna hai'). Create a DM template with price, sizes and order link to convert them."))
    a = (com.rule_topic=="Availability").mean()*100; R.append(("Operations", f"{a:.0f}% ask about city/store/size availability - publish a stock & delivery-city list and show size availability on posts."))
    an = p[p.is_anomaly]; hi=(an.direction=="high").sum()
    R.append(("Investigate", f"{len(an)} posts are unusual vs comparable posts ({hi} unusually high, {len(an)-hi} unusually low). Review them manually: viral reach to replicate, comment spikes (possible complaints) or low reach (possible algorithm/timing issue)."))
    R.append(("Testing", "Run a controlled test: same product, same day, Reel vs Styling video, with and without a price in the caption; compare engagement rate and DM count."))
    return pd.DataFrame(R, columns=["area","recommendation"])

if __name__ == "__main__":
    if "--real" in sys.argv:
        i = sys.argv.index("--real"); posts = pd.read_csv(sys.argv[i+1]); com = pd.read_csv(sys.argv[i+2])
    else:
        posts, com = simulate(); posts.to_csv("data/posts_simulated.csv", index=False); com.to_csv("data/comments_simulated.csv", index=False)
    p = anomalies(performance(posts)); com2, summ, metrics = cluster_comments(com); rec = recommend(p, com2, summ)
    p.to_csv(f"{OUT}/posts_scored.csv", index=False); com2.to_csv(f"{OUT}/comments_clustered.csv", index=False)
    summ.to_csv(f"{OUT}/cluster_summary.csv", index=False); rec.to_csv(f"{OUT}/recommendations.csv", index=False)
    p[p.is_anomaly].sort_values("max_abs_z", ascending=False).to_csv(f"{OUT}/anomalous_posts.csv", index=False)
    if "planted_anomaly" in p: metrics["planted_anomalies_found"] = f'{int(p[p.is_anomaly].planted_anomaly.notna().sum())}/{int(p.planted_anomaly.notna().sum())}'
    json.dump(metrics, open(f"{OUT}/metrics.json","w"), indent=1)
    print(metrics); print(summ[["cluster","size","dominant_topic","topic_purity_pct","top_terms"]].to_string(index=False)); print(rec.to_string(index=False))
