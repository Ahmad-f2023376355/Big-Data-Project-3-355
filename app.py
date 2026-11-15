"""Streamlit dashboard.  Run:  pip install streamlit pandas scikit-learn plotly  ->  python pipeline.py  ->  streamlit run app.py"""
import json, pandas as pd, streamlit as st, plotly.express as px
st.set_page_config(page_title="Audience & Campaign Intelligence", layout="wide")
P = pd.read_csv("output/posts_scored.csv"); C = pd.read_csv("output/comments_clustered.csv")
S = pd.read_csv("output/cluster_summary.csv"); R = pd.read_csv("output/recommendations.csv"); M = json.load(open("output/metrics.json"))
st.title("Pakistan Social Audience & Campaign Intelligence Engine")
if "data_source" in P and (P.data_source == "simulated").any():
    st.warning("Demo uses SIMULATED data. Replace with real, permitted data via `python pipeline.py --real posts.csv comments.csv`.")
f = st.sidebar.multiselect("Platform", sorted(P.platform.unique()), default=sorted(P.platform.unique()))
camp = st.sidebar.multiselect("Campaign", sorted(P.campaign.unique()), default=sorted(P.campaign.unique()))
P = P[P.platform.isin(f) & P.campaign.isin(camp)]; C = C[C.post_id.isin(P.post_id)]
t1, t2, t3, t4, t5 = st.tabs(["1 Content", "2 Audience wants", "3 Unusual posts", "4 Segments", "5 Actions"])
with t1:
    g = P.groupby(["content_type", "platform"]).engagement_rate.mean().reset_index()
    st.plotly_chart(px.bar(g, x="content_type", y="engagement_rate", color="platform", barmode="group", title="Engagement rate = (likes+comments+shares)/views"), use_container_width=True)
with t2:
    m = C.merge(P[["post_id", "campaign"]], on="post_id"); x = pd.crosstab(m.campaign, m.rule_topic, normalize="index").mul(100).round(1)
    st.plotly_chart(px.imshow(x, text_auto=True, aspect="auto", title="% of comments by topic and campaign"), use_container_width=True)
    st.plotly_chart(px.pie(C, names="language_type", title="Comment language mix"), use_container_width=True)
with t3:
    A = P[P.is_anomaly].sort_values("max_abs_z", ascending=False)
    st.plotly_chart(px.scatter(P, x="views", y="engagement_rate", color="is_anomaly", log_x=True, hover_data=["post_id", "content_type", "platform"]), use_container_width=True)
    st.dataframe(A[["post_id", "brand", "platform", "content_type", "campaign", "views", "likes", "comments_count", "driver", "direction", "max_abs_z"]])
with t4:
    st.caption(f"KMeans on TF-IDF (word + character n-grams): k={M['k']}, silhouette={M['silhouette']}")
    st.dataframe(S)
    c = st.selectbox("Cluster", S.cluster); st.write(C[C.cluster == c].comment_text.sample(min(15, (C.cluster == c).sum()), random_state=1).tolist())
with t5:
    for _, r in R.iterrows(): st.markdown(f"**{r.area}** — {r.recommendation}")
