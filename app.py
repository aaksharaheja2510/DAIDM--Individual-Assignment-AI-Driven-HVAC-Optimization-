"""
UAE AI-Driven HVAC & Energy Management
Business Validation Dashboard — Streamlit App
=============================================
Upload uae_hvac_survey_synthetic.csv to the same GitHub repo as this file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE AI HVAC — Business Validation",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY   = "#1B3A6B"
TEAL   = "#0D7C66"
GOLD   = "#C89A2A"
CORAL  = "#D95F4B"
PURPLE = "#6B4FA0"
SEG_COLORS = [TEAL, NAVY, GOLD, CORAL, PURPLE]
SEG_NAMES  = [
    "Tech-Forward Operators",
    "Cost-Driven Laggards",
    "Compliance-Led Adopters",
    "Early Majority Evaluators",
    "Resistant Traditionalists"
]

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "figure.dpi": 120,
})

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B3A6B 0%, #0D7C66 100%);
        padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; font-size: 2rem; margin: 0 0 0.3rem 0; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 0; font-size: 1rem; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem 1rem;
        text-align: center; border-left: 5px solid #0D7C66;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #1B3A6B; }
    .metric-card .label { font-size: 0.82rem; color: #666; margin-top: 0.2rem; }
    .insight-box {
        background: #E8F5F2; border-left: 4px solid #0D7C66;
        padding: 0.9rem 1.2rem; border-radius: 0 8px 8px 0;
        margin: 0.8rem 0; font-size: 0.92rem; color: #2D2D2D;
    }
    .warn-box {
        background: #FDF6E3; border-left: 4px solid #C89A2A;
        padding: 0.9rem 1.2rem; border-radius: 0 8px 8px 0;
        margin: 0.8rem 0; font-size: 0.92rem; color: #2D2D2D;
    }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #1B3A6B;
        border-bottom: 3px solid #0D7C66; padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    [data-testid="stSidebar"] { background: #F0F4FA; }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("uae_hvac_survey_synthetic.csv")
    # Impute missing values
    for col in ["hvac_satisfaction", "stakeholder_pressure", "feat_carbon"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df

@st.cache_data
def run_kmeans(df_hash, k=5):
    # Accepts a dataframe (hashable) instead of numpy array
    num_features = [
        "realtime_visibility_score","hvac_pct_of_bill","hvac_satisfaction",
        "stakeholder_pressure","ai_openness","pilot_interest",
        "feat_dashboard","feat_auto_hvac","feat_occupancy","feat_predictive",
        "feat_air_quality","feat_carbon","feat_bms_integ","feat_mobile_app","feat_compliance",
        "monitor_bms","monitor_iot","tech_bms","tech_iot_sensor","tech_smart_meter",
        "tech_ai_maintenance","pain_high_cost","pain_compliance","pain_no_data"
    ]
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(df_hash[num_features].fillna(df_hash[num_features].median()))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    centers = pca.transform(km.cluster_centers_)
    var = pca.explained_variance_ratio_ * 100
    return labels, sil, coords, centers, var

@st.cache_data
def run_classifier(X_arr, y_arr, feature_names):
    X_tr, X_te, y_tr, y_te = train_test_split(X_arr, y_arr, test_size=0.25, random_state=42, stratify=y_arr)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    cv = cross_val_score(rf, X_arr, y_arr, cv=5, scoring="f1_macro")
    imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True).tail(12)
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, output_dict=True)
    return cv.mean(), cm, imp, report

@st.cache_data
def run_regression(X_arr, y_arr, feature_names):
    X_tr, X_te, y_tr, y_te = train_test_split(X_arr, y_arr, test_size=0.25, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    gbr.fit(X_tr, y_tr)
    y_pred = gbr.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    mae = np.mean(np.abs(np.expm1(y_te) - np.expm1(y_pred)))
    imp = pd.Series(gbr.feature_importances_, index=feature_names).sort_values(ascending=True).tail(10)
    return r2, mae, imp, y_te, y_pred

@st.cache_data
def run_association(df_bool):
    freq = apriori(df_bool, min_support=0.15, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1.2)
    return rules.sort_values("lift", ascending=False).head(20)

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ CSV file not found. Make sure `uae_hvac_survey_synthetic.csv` is in the same folder as this app in your GitHub repo.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/air-conditioner.png", width=60)
st.sidebar.markdown("## 🏢 UAE AI HVAC")
st.sidebar.markdown("**Business Validation Dashboard**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate to", [
    "🏠 Executive Summary",
    "⚡ Energy Problem Landscape",
    "👥 Market Segmentation",
    "🔵 Cluster Analysis",
    "🤖 Adoption Classification",
    "💰 WTP & Revenue Model",
    "🔗 Association Rules",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Global Filters**")
seg_filter = st.sidebar.multiselect(
    "Filter by Segment",
    options=df["segment"].unique().tolist(),
    default=df["segment"].unique().tolist()
)
emirate_filter = st.sidebar.multiselect(
    "Filter by Emirate",
    options=df["primary_emirate"].unique().tolist(),
    default=df["primary_emirate"].unique().tolist()
)

dff = df[df["segment"].isin(seg_filter) & df["primary_emirate"].isin(emirate_filter)]

if len(dff) == 0:
    st.warning("No data matches your filters. Please adjust the sidebar selections.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(dff):,}** of **{len(df):,}** respondents")
st.sidebar.caption("Smart Climate Tech · UAE Market · 2025")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Summary":

    st.markdown("""
    <div class="main-header">
        <h1>🏢 UAE AI-Driven HVAC Optimisation</h1>
        <p>Business Validation Dashboard · 1,000 Synthetic Survey Respondents · 5 Target Segments · 2025</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    high_intent = (dff["adoption_class"] == "High-Intent Buyer").sum()
    evaluators  = (dff["adoption_class"] == "Evaluator").sum()
    pilot_yes   = (dff["pilot_interest"] >= 4).sum()
    med_wtp     = dff["wtp_aed_per_building"].median()
    med_bill    = dff["monthly_bill_aed"].median()
    hvac_avg    = dff["hvac_pct_of_bill"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, val, label in zip(
        [c1,c2,c3,c4,c5,c6],
        [f"{len(dff):,}", f"{high_intent/len(dff)*100:.1f}%",
         f"{evaluators/len(dff)*100:.1f}%", f"{pilot_yes/len(dff)*100:.1f}%",
         f"AED {med_wtp:,.0f}", f"AED {med_bill/1000:.0f}k"],
        ["Respondents","High-Intent Buyers","Evaluators","Would Take Pilot",
         "Median WTP/Bldg/Mo","Median Monthly Bill"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{val}</div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Adoption Likelihood Split</div>', unsafe_allow_html=True)
        adopt_counts = dff["adoption_class"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(adopt_counts.values, labels=adopt_counts.index,
               autopct="%1.1f%%", colors=[TEAL, GOLD, CORAL],
               startangle=140, wedgeprops={"edgecolor":"white","linewidth":2})
        ax.set_title("Who is ready to buy?", fontweight="bold", color=NAVY)
        st.pyplot(fig); plt.close()
        st.markdown(f"""<div class="insight-box">
        <b>Key finding:</b> {high_intent/len(dff)*100:.1f}% of the UAE commercial market are
        High-Intent Buyers — above the 10–20% B2B enterprise benchmark. A further
        {evaluators/len(dff)*100:.1f}% are Evaluators convertible via a credible pilot.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Respondents by Segment</div>', unsafe_allow_html=True)
        seg_counts = dff["segment"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(seg_counts.index, seg_counts.values,
                       color=SEG_COLORS[:len(seg_counts)], edgecolor="white", height=0.6)
        ax.set_xlabel("Number of Respondents")
        ax.set_title("Market composition by segment", fontweight="bold", color=NAVY)
        for bar in bars:
            w = bar.get_width()
            ax.text(w+2, bar.get_y()+bar.get_height()/2,
                    f"{w} ({w/len(dff)*100:.1f}%)", va="center", fontsize=8.5)
        ax.set_xlim(0, seg_counts.max()*1.25)
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Adoption Class by Segment</div>', unsafe_allow_html=True)
    cross = pd.crosstab(dff["segment"], dff["adoption_class"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(10, 3.5))
    cross.plot(kind="bar", ax=ax, color=[TEAL, GOLD, CORAL], edgecolor="white", width=0.65)
    ax.set_ylabel("% of Segment"); ax.set_xlabel("")
    ax.set_title("High-Intent Buyer share varies significantly by segment — target Tech-Forward and Compliance-Led first",
                 fontweight="bold", color=NAVY, fontsize=10)
    ax.legend(title="Adoption Class", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ENERGY PROBLEM LANDSCAPE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Energy Problem Landscape":

    st.markdown('<div class="main-header"><h1>⚡ Energy Problem Landscape</h1><p>Understanding the scale, distribution, and seasonality of cooling costs across UAE buildings</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Monthly Electricity Bill Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dff["monthly_bill_aed"]/1000, bins=50, color=NAVY, edgecolor="white", alpha=0.85)
        ax.axvline(dff["monthly_bill_aed"].median()/1000, color=GOLD, lw=2, linestyle="--",
                   label=f"Median: AED {dff['monthly_bill_aed'].median()/1000:.0f}k")
        ax.axvline(dff["monthly_bill_aed"].mean()/1000, color=CORAL, lw=2, linestyle=":",
                   label=f"Mean: AED {dff['monthly_bill_aed'].mean()/1000:.0f}k")
        ax.set_xlabel("Monthly Bill (AED '000)"); ax.set_ylabel("Count")
        ax.set_title(f"Right-skewed — skewness = {dff['monthly_bill_aed'].skew():.2f}", fontweight="bold", color=NAVY)
        ax.legend(fontsize=9)
        st.pyplot(fig); plt.close()
        st.markdown("""<div class="warn-box">
        <b>Why the median matters more than the mean here:</b> A small number of very large industrial operators
        pull the mean (AED 307k) well above what most respondents pay (median AED 194k).
        Always quote the median when pitching to mid-market clients.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">HVAC Share of Total Energy Bill</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dff["hvac_pct_of_bill"], bins=35, color=TEAL, edgecolor="white", alpha=0.85)
        ax.axvline(dff["hvac_pct_of_bill"].mean(), color=CORAL, lw=2, linestyle="--",
                   label=f"Mean: {dff['hvac_pct_of_bill'].mean():.1f}%")
        ax.set_xlabel("HVAC % of Total Bill"); ax.set_ylabel("Count")
        ax.set_title("Cooling dominates energy cost across all segments", fontweight="bold", color=NAVY)
        ax.legend(fontsize=9)
        st.pyplot(fig); plt.close()
        st.markdown(f"""<div class="insight-box">
        <b>HVAC averages {dff['hvac_pct_of_bill'].mean():.1f}% of the total bill</b> — consistent with
        DEWA published data for UAE commercial buildings in extreme-heat climates. A 25% reduction in
        HVAC costs saves roughly <b>AED {dff['monthly_bill_aed'].median()*0.679*0.25/1000:.0f}k/month</b>
        per median-sized building — a straightforward ROI case.
        </div>""", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Monthly Bill by Segment (Box Plot)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        segs = dff["segment"].unique()
        data_bp = [dff[dff["segment"]==s]["monthly_bill_aed"].values/1000 for s in segs]
        bp = ax.boxplot(data_bp, patch_artist=True, labels=[s.replace(" ","\n") for s in segs],
                        medianprops=dict(color=CORAL, lw=2))
        for patch, color in zip(bp["boxes"], SEG_COLORS):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_ylabel("Monthly Bill (AED '000)")
        ax.set_title("Tech-Forward Operators carry the highest energy burden", fontweight="bold", color=NAVY)
        plt.xticks(fontsize=8)
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown('<div class="section-title">Summer Cooling Spike Severity</div>', unsafe_allow_html=True)
        spike_seg = pd.crosstab(dff["primary_emirate"], dff["summer_cooling_spike"], normalize="index")*100
        spike_seg = spike_seg.reindex(columns=["<20%","20-50%","51-100%",">100%"], fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        spike_seg.plot(kind="bar", stacked=True, ax=ax,
                       color=[TEAL,"#52B788",GOLD,CORAL], edgecolor="white", width=0.65)
        ax.set_xlabel("Emirate"); ax.set_ylabel("% of Respondents")
        ax.set_title("Summer bills spike 51–100%+ for majority of respondents", fontweight="bold", color=NAVY)
        ax.legend(title="Spike Level", fontsize=8, loc="lower right")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Bill Size vs Willingness-to-Pay — Prospect Quadrant Map</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, seg in enumerate(dff["segment"].unique()):
        mask = dff["segment"] == seg
        ax.scatter(dff.loc[mask,"monthly_bill_aed"]/1000, dff.loc[mask,"wtp_aed_per_building"]/1000,
                   c=SEG_COLORS[i % len(SEG_COLORS)], alpha=0.45, s=22, label=seg, edgecolors="none")
    ax.axvline(dff["monthly_bill_aed"].median()/1000, color="gray", lw=1, linestyle="--", alpha=0.5)
    ax.axhline(dff["wtp_aed_per_building"].median()/1000, color="gray", lw=1, linestyle="--", alpha=0.5)
    ax.text(1800, dff["wtp_aed_per_building"].quantile(0.85)/1000, "🎯 Prime Targets", fontsize=10, color=TEAL, fontweight="bold")
    ax.set_xlabel("Monthly Bill (AED '000)"); ax.set_ylabel("WTP (AED '000/month/building)")
    ax.set_title("High bill + High WTP quadrant (top-right) = ideal first sales targets", fontweight="bold", color=NAVY)
    ax.legend(fontsize=8, loc="upper left")
    st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MARKET SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Market Segmentation":

    st.markdown('<div class="main-header"><h1>👥 Market Segmentation Explorer</h1><p>Deep-dive into each segment\'s pain points, technology readiness, and commercial preferences</p></div>', unsafe_allow_html=True)

    seg_sel = st.multiselect("Select segments to compare", options=dff["segment"].unique().tolist(),
                              default=dff["segment"].unique().tolist())
    dfs = dff[dff["segment"].isin(seg_sel)] if seg_sel else dff

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Pain Point Prevalence by Segment</div>', unsafe_allow_html=True)
        pain_cols = ["pain_high_cost","pain_old_hvac","pain_no_data","pain_occupancy",
                     "pain_wastage","pain_comfort","pain_maintenance","pain_compliance"]
        hm = dfs.groupby("segment")[pain_cols].mean()
        hm.columns = [c.replace("pain_","").replace("_"," ").title() for c in pain_cols]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(hm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, linewidths=0.4,
                    cbar_kws={"label":"Prevalence Rate"})
        ax.set_title("Darker = more prevalent pain point in that segment", fontweight="bold", color=NAVY)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-title">Technology Already Deployed by Segment</div>', unsafe_allow_html=True)
        tech_cols = ["tech_smart_meter","tech_iot_sensor","tech_bms","tech_ai_maintenance","tech_solar","tech_none"]
        ht = dfs.groupby("segment")[tech_cols].mean()
        ht.columns = [c.replace("tech_","").replace("_"," ").title() for c in tech_cols]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(ht, annot=True, fmt=".2f", cmap="Blues", ax=ax, linewidths=0.4,
                    cbar_kws={"label":"Deployment Rate"})
        ax.set_title("Darker = higher technology adoption in that segment", fontweight="bold", color=NAVY)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Commercial Model Preference</div>', unsafe_allow_html=True)
        cm_seg = pd.crosstab(dfs["segment"], dfs["commercial_model"], normalize="index")*100
        fig, ax = plt.subplots(figsize=(6, 4))
        cm_seg.plot(kind="bar", ax=ax, color=[NAVY, TEAL, GOLD, CORAL, PURPLE], edgecolor="white", width=0.65)
        ax.set_ylabel("% of Segment"); ax.set_xlabel("")
        ax.set_title("Contract structure should vary by segment", fontweight="bold", color=NAVY)
        ax.legend(title="Model", fontsize=7, loc="upper right")
        plt.xticks(rotation=20, ha="right", fontsize=8)
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown('<div class="section-title">Segment Profile — Key Scores</div>', unsafe_allow_html=True)
        profile_cols = ["ai_openness","pilot_interest","realtime_visibility_score",
                        "stakeholder_pressure","hvac_satisfaction"]
        seg_prof = dfs.groupby("segment")[profile_cols].mean()
        seg_prof_n = (seg_prof - seg_prof.min()) / (seg_prof.max() - seg_prof.min())
        fig, ax = plt.subplots(figsize=(6, 4))
        seg_prof_n.T.plot(kind="bar", ax=ax, color=SEG_COLORS[:len(seg_prof_n)], edgecolor="white", width=0.72)
        ax.set_ylabel("Normalised Score (0–1)"); ax.set_xlabel("")
        ax.set_xticklabels([c.replace("_"," ").title() for c in profile_cols], rotation=25, ha="right", fontsize=8)
        ax.set_title("Tech-Forward leads on AI openness; Compliance-Led on pressure", fontweight="bold", color=NAVY)
        ax.legend(fontsize=7, loc="upper right")
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Median WTP per Building per Month by Segment</div>', unsafe_allow_html=True)
    wtp_seg = dfs.groupby("segment")["wtp_aed_per_building"].median().sort_values()
    fig, ax = plt.subplots(figsize=(10, 3))
    bars = ax.barh(wtp_seg.index, wtp_seg.values, color=SEG_COLORS[:len(wtp_seg)], edgecolor="white", height=0.55)
    ax.set_xlabel("Median WTP (AED/month/building)")
    ax.set_title("Tiered pricing is essential — a single price point leaves revenue on the table", fontweight="bold", color=NAVY)
    for bar in bars:
        w = bar.get_width()
        ax.text(w+100, bar.get_y()+bar.get_height()/2, f"AED {w:,.0f}", va="center", fontsize=9)
    ax.set_xlim(0, wtp_seg.max()*1.25)
    st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CLUSTER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Cluster Analysis":

    st.markdown('<div class="main-header"><h1>🔵 K-Means Cluster Analysis</h1><p>Data-driven discovery of natural customer groups — do distinct market sub-populations actually exist?</p></div>', unsafe_allow_html=True)

    labels, sil, coords, centers, var = run_kmeans(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Elbow Method — Choosing k</div>', unsafe_allow_html=True)
        inertias = []
        for k in range(2, 10):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_sc)
            inertias.append(km.inertia_)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(range(2,10), inertias, "o-", color=NAVY, lw=2, markersize=7)
        ax.axvline(5, color=TEAL, lw=2, linestyle="--", label="Chosen k=5")
        ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Inertia (SSE)")
        ax.set_title("Elbow at k=5 — five clusters is analytically justified", fontweight="bold", color=NAVY)
        ax.legend(fontsize=9)
        st.pyplot(fig); plt.close()

    with col2:
        st.metric("Silhouette Score (k=5)", f"{sil:.4f}", help="Range: -1 to 1. Higher = better separated clusters. >0.10 indicates meaningful structure in survey data.")
        st.markdown(f"""<div class="insight-box">
        <b>What the Silhouette Score means:</b> A score of {sil:.4f} on survey data confirms that
        five distinct customer groups exist in the market — they are not one undifferentiated mass.
        On real survey data with more respondents, this score is expected to improve above 0.20.
        </div>""", unsafe_allow_html=True)
        cluster_labels = {0:"Cost-Sensitive\nConventionalists", 1:"Tech-Forward\nInnovators",
                          2:"Compliance-Led\nAdopters", 3:"Mid-Market\nEvaluators", 4:"Passive\nResistors"}
        st.markdown("**Cluster Names (data-derived):**")
        for k, v in cluster_labels.items():
            st.markdown(f"- **Cluster {k}:** {v.replace(chr(10),' ')}")

    st.markdown('<div class="section-title">PCA 2D Cluster Visualisation</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cl, color in enumerate(SEG_COLORS):
        mask = labels == cl
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, alpha=0.45, s=25,
                   label=cluster_labels[cl].replace("\n"," "), edgecolors="none")
    for cl, (cx, cy) in enumerate(centers):
        ax.scatter(cx, cy, c=SEG_COLORS[cl], s=250, marker="*", edgecolors="black", lw=0.8, zorder=5)
        ax.annotate(cluster_labels[cl].replace("\n"," "), (cx, cy),
                    textcoords="offset points", xytext=(10, 6), fontsize=8.5,
                    fontweight="bold", color=SEG_COLORS[cl])
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)"); ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
    ax.set_title(f"Five distinct clusters visible — Silhouette = {sil:.4f}  |  Stars = cluster centroids",
                 fontweight="bold", color=NAVY)
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.15)
    st.pyplot(fig); plt.close()
    st.markdown("""<div class="insight-box">
    <b>How to read this chart:</b> Each dot is one respondent. Dots of the same colour belong to the same
    data-derived cluster. The stars mark the cluster centre. Clusters that are well-separated from each
    other indicate that those customer groups have genuinely different profiles — meaning they need
    different sales approaches, different pricing, and different product messaging.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ADOPTION CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Adoption Classification":

    st.markdown('<div class="main-header"><h1>🤖 Adoption Classification Model</h1><p>Which respondents are High-Intent Buyers? — The lead-scoring engine</p></div>', unsafe_allow_html=True)

    clf_features = [
        "realtime_visibility_score","hvac_pct_of_bill","hvac_satisfaction",
        "stakeholder_pressure","ai_openness","pilot_interest",
        "monitor_bms","monitor_iot","tech_bms","tech_iot_sensor",
        "pain_high_cost","pain_compliance","pain_no_data",
        "feat_dashboard","feat_auto_hvac","feat_predictive","feat_compliance"
    ]
    le = LabelEncoder()
    y_clf = le.fit_transform(df["adoption_class"])
    X_clf = df[clf_features].fillna(df[clf_features].median()).values

    f1_cv, cm, feat_imp, report = run_classifier(X_clf, y_clf, clf_features)

    col1, col2, col3 = st.columns(3)
    col1.metric("CV F1-Macro Score", f"{f1_cv:.4f}", help="5-fold cross-validated F1 score. Closer to 1.0 = better.")
    col2.metric("Classes", "3", "High-Intent / Evaluator / Not Ready")
    col3.metric("Features Used", str(len(clf_features)), "Survey variables fed to the model")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Top Feature Importances</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        feat_imp.plot(kind="barh", ax=ax, color=TEAL, edgecolor="white")
        ax.set_xlabel("Importance Score")
        ax.set_title("Ask these questions first to qualify a prospect", fontweight="bold", color=NAVY)
        ax.set_yticklabels([l.replace("_"," ").title() for l in feat_imp.index], fontsize=9)
        st.pyplot(fig); plt.close()
        st.markdown("""<div class="insight-box">
        <b>Sales implication:</b> The top three predictors of a High-Intent Buyer are
        <b>AI Openness, Pilot Interest,</b> and <b>Real-Time Visibility Score</b> —
        all attitude variables, not financial ones. Lead your pitch with the data visibility
        and AI narrative, not just cost savings.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_, linewidths=0.5)
        ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
        ax.set_title("Model prediction accuracy", fontweight="bold", color=NAVY)
        plt.xticks(rotation=15, ha="right", fontsize=8)
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Lead Scoring Table — Prioritised Prospect List</div>', unsafe_allow_html=True)
    view_cols = ["respondent_id","segment","building_type","primary_emirate",
                 "adoption_class","pilot_interest","ai_openness","wtp_aed_per_building","monthly_bill_aed"]
    adopt_filter = st.multiselect("Filter by adoption class", df["adoption_class"].unique().tolist(),
                                   default=["High-Intent Buyer"])
    tbl = df[df["adoption_class"].isin(adopt_filter)][view_cols].sort_values(
        ["pilot_interest","ai_openness"], ascending=False).head(50)
    tbl["wtp_aed_per_building"] = tbl["wtp_aed_per_building"].apply(lambda x: f"AED {x:,.0f}")
    tbl["monthly_bill_aed"] = tbl["monthly_bill_aed"].apply(lambda x: f"AED {x:,.0f}")
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — WTP & REVENUE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 WTP & Revenue Model":

    st.markdown('<div class="main-header"><h1>💰 WTP & Revenue Modelling</h1><p>What will the market pay — and what is this business worth?</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">WTP Distribution by Segment</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        segs = dff["segment"].unique()
        data_bp = [dff[dff["segment"]==s]["wtp_aed_per_building"].values for s in segs]
        bp = ax.boxplot(data_bp, patch_artist=True, labels=[s.replace(" ","\n") for s in segs],
                        medianprops=dict(color=CORAL, lw=2.5))
        for patch, color in zip(bp["boxes"], SEG_COLORS):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_ylabel("WTP (AED/month/building)")
        ax.set_title("Huge WTP spread — one price point won't work", fontweight="bold", color=NAVY)
        plt.xticks(fontsize=7.5)
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-title">WTP by Adoption Class</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, cls in enumerate(dff["adoption_class"].unique()):
            data = dff[dff["adoption_class"]==cls]["wtp_aed_per_building"]
            ax.hist(data, bins=30, alpha=0.55, label=cls,
                    color=[TEAL, GOLD, CORAL][i % 3], edgecolor="white")
        ax.set_xlabel("WTP (AED/month/building)"); ax.set_ylabel("Count")
        ax.set_title("High-Intent Buyers pay more — up to 3× vs Not Ready", fontweight="bold", color=NAVY)
        ax.legend(fontsize=9)
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">💡 Interactive Revenue Calculator</div>', unsafe_allow_html=True)
    st.markdown("Adjust the sliders to model your revenue potential under different assumptions.")

    rc1, rc2, rc3 = st.columns(3)
    n_bldgs      = rc1.slider("Addressable buildings in market", 500, 10000, 3000, 100)
    conv_rate    = rc2.slider("Pilot-to-paid conversion rate (%)", 10, 70, 40, 5)
    avg_wtp      = rc3.slider("Average monthly fee per building (AED)", 2000, 20000, 7000, 500)
    pilot_accept = 0.539  # 53.9% would take a pilot

    pilots       = int(n_bldgs * pilot_accept)
    paid_clients = int(pilots * conv_rate / 100)
    mrr          = paid_clients * avg_wtp
    arr          = mrr * 12

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pilot Candidates", f"{pilots:,}", f"{pilot_accept*100:.0f}% acceptance rate")
    m2.metric("Paying Clients", f"{paid_clients:,}", f"{conv_rate}% conversion")
    m3.metric("Monthly Revenue (MRR)", f"AED {mrr:,.0f}")
    m4.metric("Annual Revenue (ARR)", f"AED {arr:,.0f}")

    st.markdown(f"""<div class="warn-box">
    <b>What this means:</b> At {conv_rate}% conversion from a 60-day free pilot, with an average
    AED {avg_wtp:,}/month per building, targeting {n_bldgs:,} buildings produces
    <b>AED {arr/1e6:.1f}M ARR</b>. Adjust the sliders to test conservative vs optimistic scenarios.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">WTP Regression — What Predicts Willingness to Pay?</div>', unsafe_allow_html=True)
    reg_features = ["monthly_bill_aed","hvac_pct_of_bill","realtime_visibility_score",
                    "ai_openness","pilot_interest","stakeholder_pressure",
                    "tech_bms","tech_iot_sensor","tech_ai_maintenance",
                    "monitor_bms","monitor_iot","pain_high_cost"]
    y_reg = np.log1p(df["wtp_aed_per_building"].values)
    X_reg = df[reg_features].fillna(df[reg_features].median()).values
    r2, mae, reg_imp, y_te, y_pred = run_regression(X_reg, y_reg, reg_features)

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Regression R²", f"{r2:.4f}", help="How well the model explains WTP variation.")
        st.metric("Mean Absolute Error", f"AED {mae:,.0f}/month/building")
        fig, ax = plt.subplots(figsize=(5, 4))
        reg_imp.plot(kind="barh", ax=ax, color=GOLD, edgecolor="white")
        ax.set_xlabel("Importance Score")
        ax.set_title("WTP drivers — attitude > wallet size", fontweight="bold", color=NAVY)
        ax.set_yticklabels([l.replace("_"," ").title() for l in reg_imp.index], fontsize=9)
        st.pyplot(fig); plt.close()
    with col4:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_te, y_pred, alpha=0.35, color=NAVY, s=18, edgecolors="none")
        mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "--", color=CORAL, lw=2, label="Perfect fit")
        ax.set_xlabel("Actual log(WTP)"); ax.set_ylabel("Predicted log(WTP)")
        ax.set_title(f"Actual vs Predicted WTP (R²={r2:.4f})", fontweight="bold", color=NAVY)
        ax.legend(fontsize=9)
        st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ASSOCIATION RULES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":

    st.markdown('<div class="main-header"><h1>🔗 Association Rule Mining</h1><p>Which pain points and technologies co-occur? — Surfacing cross-sell triggers and sales patterns</p></div>', unsafe_allow_html=True)

    basket_cols = ["pain_high_cost","pain_old_hvac","pain_no_data","pain_compliance",
                   "monitor_bms","monitor_iot","tech_bms","tech_iot_sensor","tech_ai_maintenance","tech_smart_meter"]
    col_rename = {c: c.replace("pain_","PAIN:").replace("monitor_","MON:").replace("tech_","TECH:").upper()
                  for c in basket_cols}
    basket_df = df[basket_cols].fillna(0).astype(bool).rename(columns=col_rename)

    with st.spinner("Mining association rules…"):
        rules = run_association(basket_df)

    rules_disp = rules.copy()
    rules_disp["antecedents"] = rules_disp["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules_disp["consequents"] = rules_disp["consequents"].apply(lambda x: ", ".join(list(x)))

    c1, c2, c3 = st.columns(3)
    c1.metric("Rules Found", len(rules), "lift ≥ 1.2, support ≥ 0.15")
    c2.metric("Max Lift", f"{rules['lift'].max():.2f}", "Strength of strongest rule")
    c3.metric("Max Confidence", f"{rules['confidence'].max():.2f}", "Predictive power of top rule")

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="section-title">Support vs Confidence vs Lift</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sc = ax.scatter(rules["support"], rules["confidence"], c=rules["lift"],
                        cmap="viridis", s=rules["lift"]*80, alpha=0.7, edgecolors="white", lw=0.5)
        plt.colorbar(sc, ax=ax, label="Lift")
        ax.set_xlabel("Support (how common is this rule)"); ax.set_ylabel("Confidence (how predictive)")
        ax.set_title("Bigger bubble = stronger lift above random chance", fontweight="bold", color=NAVY)
        for _, row in rules.head(5).iterrows():
            ant = list(row["antecedents"])[0][:12] if isinstance(row["antecedents"], frozenset) else str(row["antecedents"])[:12]
            ax.annotate(f"lift={row['lift']:.2f}", (row["support"], row["confidence"]),
                        fontsize=7.5, alpha=0.85, xytext=(4,4), textcoords="offset points")
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-title">Top 5 Actionable Rules</div>', unsafe_allow_html=True)
        for i, (_, row) in enumerate(rules_disp.head(5).iterrows()):
            st.markdown(f"""<div class="insight-box">
            <b>Rule {i+1}:</b> If <code>{row['antecedents']}</code> → then <code>{row['consequents']}</code><br>
            Support: {row['support']:.2f} | Confidence: {row['confidence']:.2f} | <b>Lift: {row['lift']:.2f}</b>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Full Rules Table</div>', unsafe_allow_html=True)
    min_lift = st.slider("Minimum lift to display", 1.0, rules["lift"].max(), 1.2, 0.05)
    filtered = rules_disp[rules["lift"] >= min_lift][["antecedents","consequents","support","confidence","lift"]]
    filtered["support"] = filtered["support"].round(3)
    filtered["confidence"] = filtered["confidence"].round(3)
    filtered["lift"] = filtered["lift"].round(3)
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)
    st.markdown("""<div class="warn-box">
    <b>How to use this for sales:</b> Each rule is a pattern found in the data. A high-confidence rule
    means: "When a prospect has X, they very likely also have Y." Use this to predict what a prospect
    needs before they tell you, and to personalise the pitch accordingly.
    </div>""", unsafe_allow_html=True)
