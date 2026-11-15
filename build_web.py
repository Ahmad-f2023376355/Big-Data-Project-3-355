import pandas as pd, json
P=pd.read_csv("output/posts_scored.csv"); C=pd.read_csv("output/comments_clustered.csv"); S=pd.read_csv("output/cluster_summary.csv")
R=pd.read_csv("output/recommendations.csv"); M=json.load(open("output/metrics.json"))
g=P.groupby(["content_type","platform"]).engagement_rate.mean().mul(100).round(2).unstack()
m=C.merge(P[["post_id","campaign"]],on="post_id"); x=(pd.crosstab(m.campaign,m.rule_topic,normalize="index")*100).round(1)
sc=pd.concat([P[P.is_anomaly],P[~P.is_anomaly].sample(500,random_state=1)])
A=P[P.is_anomaly].sort_values("max_abs_z",ascending=False).head(12)
D=dict(eng=dict(labels=list(g.index),series={k:g[k].tolist() for k in g.columns}),
 heat=dict(rows=list(x.index),cols=list(x.columns),vals=x.values.tolist()),
 lang=C.language_type.value_counts().to_dict(),
 sc=[[int(r.views),round(r.engagement_rate*100,2),bool(r.is_anomaly)] for r in sc.itertuples()],
 anom=A[["post_id","platform","content_type","campaign","views","likes","comments_count","driver","direction","max_abs_z"]].values.tolist(),
 clus=S[["cluster","size","share_pct","dominant_topic","topic_purity_pct","top_terms","example"]].values.tolist(),
 rec=R.values.tolist(), met=M, n=[len(P),len(C)])
html=open("web_template.html").read().replace("__DATA__",json.dumps(D,ensure_ascii=False))
open("docs/index.html","w").write(html)
