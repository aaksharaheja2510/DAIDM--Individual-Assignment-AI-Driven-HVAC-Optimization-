"""
=============================================================================
UAE AI-Driven HVAC & Energy Management — Market Validation Survey
Synthetic Dataset Generation + Full Analytical Pipeline
=============================================================================
Techniques: Classification | Clustering | Association Rule Mining | Regression
Analysis:   Descriptive | Diagnostic
Segments:   5 target personas (see SEGMENT_PROFILES below)
=============================================================================
Compatible with Google Colab — install dependencies with:
  !pip install pandas numpy scikit-learn matplotlib seaborn scipy mlxtend
=============================================================================
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              silhouette_score, r2_score, mean_absolute_error)
from sklearn.pipeline import Pipeline
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams.update({
    'figure.dpi': 140,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

# ─── COLOUR PALETTE ──────────────────────────────────────────────────────────
NAVY    = "#1B3A6B"
TEAL    = "#0D7C66"
GOLD    = "#C89A2A"
CORAL   = "#D95F4B"
PURPLE  = "#6B4FA0"
MINT    = "#52B788"
ORANGE  = "#E07B39"
STEEL   = "#4A7BA7"
SEG_COLORS = [TEAL, NAVY, GOLD, CORAL, PURPLE]
SEG_NAMES  = [
    "Tech-Forward Operators",
    "Cost-Driven Laggards",
    "Compliance-Led Adopters",
    "Early Majority Evaluators",
    "Resistant Traditionalists"
]

N = 1050  # oversample slightly, trim to 1000 after noise injection

# ─── SEGMENT PROFILE DEFINITIONS ─────────────────────────────────────────────
#   Each profile encodes realistic distributional priors per field.
#   Shares must sum to 1.0.
SEGMENT_SHARES = [0.22, 0.25, 0.18, 0.20, 0.15]   # % of population

def weighted_choice(options, weights, size):
    return np.random.choice(options, size=size, p=np.array(weights)/sum(weights))

# ─── DATA GENERATION ─────────────────────────────────────────────────────────
def generate_dataset(n=N):
    records = []
    seg_sizes = [int(n * s) for s in SEGMENT_SHARES]
    seg_sizes[-1] += n - sum(seg_sizes)   # absorb rounding remainder

    for seg_idx, seg_n in enumerate(seg_sizes):
        for _ in range(seg_n):
            s = seg_idx   # shorthand

            # ── Q1: Role ──────────────────────────────────────────────────────
            role_probs = [
                [0.08, 0.10, 0.20, 0.15, 0.15, 0.10, 0.12, 0.10],  # Tech-Forward
                [0.15, 0.20, 0.20, 0.05, 0.10, 0.05, 0.15, 0.10],  # Cost-Driven
                [0.08, 0.08, 0.10, 0.30, 0.15, 0.15, 0.10, 0.04],  # Compliance
                [0.10, 0.15, 0.20, 0.10, 0.20, 0.08, 0.12, 0.05],  # Evaluators
                [0.20, 0.20, 0.25, 0.05, 0.10, 0.05, 0.10, 0.05],  # Resistant
            ]
            roles = ["Facilities Manager","Property Owner","Operations Manager",
                     "Sustainability Officer","C-Suite Executive","Government Official",
                     "Hospitality Manager","Other"]
            role = weighted_choice(roles, role_probs[s], 1)[0]

            # ── Q2: Building Type ─────────────────────────────────────────────
            btype_probs = [
                [0.25, 0.10, 0.20, 0.10, 0.05, 0.10, 0.15, 0.03, 0.02],
                [0.20, 0.15, 0.15, 0.15, 0.05, 0.15, 0.10, 0.03, 0.02],
                [0.15, 0.10, 0.15, 0.05, 0.25, 0.05, 0.15, 0.07, 0.03],
                [0.20, 0.15, 0.20, 0.10, 0.10, 0.08, 0.12, 0.03, 0.02],
                [0.20, 0.12, 0.12, 0.20, 0.05, 0.15, 0.10, 0.04, 0.02],
            ]
            btypes = ["Commercial Office","Retail/Mall","Hotel/Hospitality",
                      "Residential Tower","Government Facility","Industrial/Warehouse",
                      "Mixed-Use","Healthcare","Other"]
            building_type = weighted_choice(btypes, btype_probs[s], 1)[0]

            # ── Q3: Number of Buildings ───────────────────────────────────────
            nbldg_probs = [
                [0.05, 0.15, 0.30, 0.30, 0.20],
                [0.20, 0.35, 0.25, 0.15, 0.05],
                [0.10, 0.20, 0.30, 0.25, 0.15],
                [0.15, 0.30, 0.30, 0.20, 0.05],
                [0.30, 0.40, 0.20, 0.08, 0.02],
            ]
            nbldg_opts = ["1", "2-5", "6-15", "16-50", "50+"]
            num_buildings = weighted_choice(nbldg_opts, nbldg_probs[s], 1)[0]

            # ── Q4: Total GFA (m²) ────────────────────────────────────────────
            gfa_probs = [
                [0.05, 0.10, 0.25, 0.35, 0.25],
                [0.20, 0.35, 0.25, 0.15, 0.05],
                [0.08, 0.18, 0.30, 0.28, 0.16],
                [0.12, 0.25, 0.35, 0.20, 0.08],
                [0.30, 0.40, 0.20, 0.08, 0.02],
            ]
            gfa_opts = ["<5000m²","5k-20km²","20k-100km²","100k-500km²",">500km²"]
            total_gfa = weighted_choice(gfa_opts, gfa_probs[s], 1)[0]

            # ── Q5: Emirate (top 2) ───────────────────────────────────────────
            emirate_probs = [
                [0.45, 0.35, 0.08, 0.05, 0.03, 0.02, 0.02],
                [0.40, 0.38, 0.10, 0.06, 0.03, 0.02, 0.01],
                [0.40, 0.30, 0.12, 0.08, 0.04, 0.04, 0.02],
                [0.42, 0.36, 0.10, 0.06, 0.03, 0.02, 0.01],
                [0.38, 0.35, 0.12, 0.07, 0.04, 0.02, 0.02],
            ]
            emirate_opts = ["Dubai","Abu Dhabi","Sharjah","RAK","Ajman","Fujairah","UAQ"]
            primary_emirate = weighted_choice(emirate_opts, emirate_probs[s], 1)[0]

            # ── Q6: Years in UAE Sector ───────────────────────────────────────
            yrs_probs = [
                [0.05, 0.15, 0.30, 0.50],
                [0.15, 0.25, 0.30, 0.30],
                [0.05, 0.10, 0.30, 0.55],
                [0.10, 0.20, 0.35, 0.35],
                [0.05, 0.10, 0.25, 0.60],
            ]
            yrs_opts = ["<2 years","2-5 years","6-10 years",">10 years"]
            years_in_uae = weighted_choice(yrs_opts, yrs_probs[s], 1)[0]

            # ── Q7: Monthly Electricity Bill ──────────────────────────────────
            bill_means  = [600000, 120000, 350000, 200000, 60000]
            bill_stds   = [400000,  80000, 200000, 120000, 40000]
            monthly_bill_aed = max(5000, np.random.normal(bill_means[s], bill_stds[s]))
            bill_category = pd.cut([monthly_bill_aed],
                bins=[0,20000,100000,500000,2000000,1e9],
                labels=["<20k","20k-100k","100k-500k","500k-2M",">2M"])[0]

            # ── Q8: HVAC % of Total Bill ──────────────────────────────────────
            hvac_means = [65, 75, 60, 68, 70]
            hvac_stds  = [ 8,  7,  9,  9,  8]
            hvac_pct = float(np.clip(np.random.normal(hvac_means[s], hvac_stds[s]), 20, 98))

            # ── Q9: Bill Change 3 Years ───────────────────────────────────────
            bchange_probs = [
                [0.10, 0.20, 0.20, 0.35, 0.15],
                [0.05, 0.10, 0.15, 0.40, 0.30],
                [0.08, 0.15, 0.20, 0.38, 0.19],
                [0.06, 0.12, 0.22, 0.40, 0.20],
                [0.05, 0.08, 0.18, 0.42, 0.27],
            ]
            bchange_opts = ["Decreased>20%","Decreased 5-20%","Stable","Increased 5-20%","Increased>20%"]
            bill_change_3yr = weighted_choice(bchange_opts, bchange_probs[s], 1)[0]

            # ── Q10: Summer Cooling Spike ─────────────────────────────────────
            spike_probs = [
                [0.05, 0.25, 0.45, 0.25],
                [0.03, 0.15, 0.42, 0.40],
                [0.04, 0.20, 0.45, 0.31],
                [0.05, 0.22, 0.45, 0.28],
                [0.03, 0.15, 0.40, 0.42],
            ]
            spike_opts = ["<20%","20-50%","51-100%",">100%"]
            summer_cooling_spike = weighted_choice(spike_opts, spike_probs[s], 1)[0]

            # ── Q11: Monitoring Method (multi-select → binary flags) ──────────
            mon_probs = [
                [0.05, 0.20, 0.70, 0.80, 0.40, 0.10],  # Tech-Forward
                [0.50, 0.40, 0.15, 0.05, 0.10, 0.05],  # Cost-Driven
                [0.20, 0.40, 0.55, 0.40, 0.30, 0.08],  # Compliance
                [0.15, 0.45, 0.55, 0.30, 0.25, 0.07],  # Evaluators
                [0.70, 0.30, 0.10, 0.02, 0.05, 0.02],  # Resistant
            ]
            mon_keys = ["monitor_manual","monitor_smart_meter","monitor_bms",
                        "monitor_iot","monitor_audit","monitor_none"]
            monitoring = {k: int(np.random.random() < mon_probs[s][i])
                         for i, k in enumerate(mon_keys)}

            # ── Q12: Realtime Visibility Score (1–10) ─────────────────────────
            vis_means = [7.5, 3.5, 6.0, 5.5, 2.5]
            vis_score = int(np.clip(np.round(np.random.normal(vis_means[s], 1.5)), 1, 10))

            # ── Q13: Track Energy Intensity ───────────────────────────────────
            etrack_probs = [
                [0.55, 0.25, 0.12, 0.08],
                [0.08, 0.20, 0.25, 0.47],
                [0.45, 0.25, 0.18, 0.12],
                [0.25, 0.35, 0.25, 0.15],
                [0.04, 0.12, 0.20, 0.64],
            ]
            etrack_opts = ["Yes-systematic","Yes-occasional","Plan to","No plans"]
            energy_intensity_tracking = weighted_choice(etrack_opts, etrack_probs[s], 1)[0]

            # ── Q14: Top Pain Points (rank 1–3 → encoded as score 0-3) ────────
            pain_keys = ["pain_high_cost","pain_old_hvac","pain_no_data",
                         "pain_occupancy","pain_wastage","pain_comfort",
                         "pain_maintenance","pain_compliance"]
            pain_probs_seg = [
                [0.85, 0.60, 0.70, 0.65, 0.70, 0.50, 0.55, 0.40],  # Tech-Forward
                [0.95, 0.75, 0.45, 0.40, 0.60, 0.30, 0.65, 0.20],  # Cost-Driven
                [0.70, 0.55, 0.65, 0.55, 0.65, 0.55, 0.45, 0.90],  # Compliance
                [0.80, 0.65, 0.60, 0.55, 0.65, 0.45, 0.60, 0.50],  # Evaluators
                [0.88, 0.70, 0.30, 0.25, 0.45, 0.30, 0.70, 0.15],  # Resistant
            ]
            pain = {k: int(np.random.random() < pain_probs_seg[s][i])
                    for i, k in enumerate(pain_keys)}

            # ── Q17: HVAC Satisfaction (1–5) ─────────────────────────────────
            sat_means = [3.2, 2.5, 3.0, 2.8, 3.5]
            hvac_satisfaction = int(np.clip(np.round(np.random.normal(sat_means[s], 0.9)), 1, 5))

            # ── Q20: UAE Energy Strategy 2050 Awareness ───────────────────────
            aware_probs = [
                [0.45, 0.35, 0.15, 0.05],
                [0.10, 0.30, 0.35, 0.25],
                [0.60, 0.28, 0.09, 0.03],
                [0.20, 0.40, 0.30, 0.10],
                [0.05, 0.20, 0.40, 0.35],
            ]
            aware_opts = ["Very familiar","Familiar","Somewhat familiar","Not familiar"]
            strategy_awareness = weighted_choice(aware_opts, aware_probs[s], 1)[0]

            # ── Q22: Formal Sustainability Strategy ───────────────────────────
            sust_probs = [
                [0.50, 0.28, 0.12, 0.07, 0.03],
                [0.10, 0.20, 0.18, 0.28, 0.24],
                [0.55, 0.25, 0.12, 0.06, 0.02],
                [0.25, 0.35, 0.20, 0.14, 0.06],
                [0.05, 0.10, 0.12, 0.28, 0.45],
            ]
            sust_opts = ["Full strategy","Partial strategy","In development","Plan to","No plans"]
            sustainability_strategy = weighted_choice(sust_opts, sust_probs[s], 1)[0]

            # ── Q23: Tariff Impact ────────────────────────────────────────────
            tariff_probs = [
                [0.40, 0.40, 0.12, 0.05, 0.03],
                [0.50, 0.35, 0.10, 0.04, 0.01],
                [0.35, 0.40, 0.15, 0.06, 0.04],
                [0.38, 0.42, 0.12, 0.06, 0.02],
                [0.45, 0.35, 0.12, 0.06, 0.02],
            ]
            tariff_opts = ["Significantly impacted","Moderately impacted","Minimal","No impact","Not sure"]
            tariff_impact = weighted_choice(tariff_opts, tariff_probs[s], 1)[0]

            # ── Q25: Stakeholder Pressure Score (1–10) ────────────────────────
            pressure_means = [6.5, 5.0, 8.0, 6.0, 3.0]
            stakeholder_pressure = int(np.clip(np.round(np.random.normal(pressure_means[s], 1.5)), 1, 10))

            # ── Q26: Tech Deployed (multi-select) ────────────────────────────
            tech_probs_seg = [
                [0.85, 0.75, 0.80, 0.75, 0.70, 0.55, 0.50, 0.05],  # Tech-Forward
                [0.20, 0.15, 0.20, 0.08, 0.08, 0.05, 0.05, 0.50],  # Cost-Driven
                [0.65, 0.55, 0.60, 0.55, 0.60, 0.40, 0.45, 0.08],  # Compliance
                [0.55, 0.45, 0.55, 0.40, 0.40, 0.25, 0.30, 0.10],  # Evaluators
                [0.10, 0.10, 0.15, 0.05, 0.05, 0.02, 0.03, 0.65],  # Resistant
            ]
            tech_keys = ["tech_smart_meter","tech_iot_sensor","tech_auto_lighting",
                         "tech_bms","tech_air_quality","tech_ai_maintenance",
                         "tech_solar","tech_none"]
            tech = {k: int(np.random.random() < tech_probs_seg[s][i])
                    for i, k in enumerate(tech_keys)}

            # ── Q27: AI Openness (1=Opposed … 5=Very Open) ───────────────────
            ai_means = [4.5, 2.5, 3.8, 3.5, 1.8]
            ai_openness = int(np.clip(np.round(np.random.normal(ai_means[s], 0.8)), 1, 5))

            # ── Q30: Min Savings % to Approve Budget ─────────────────────────
            savings_probs = [
                [0.15, 0.45, 0.28, 0.08, 0.04],
                [0.05, 0.20, 0.40, 0.25, 0.10],
                [0.12, 0.38, 0.32, 0.12, 0.06],
                [0.08, 0.30, 0.38, 0.18, 0.06],
                [0.03, 0.12, 0.30, 0.35, 0.20],
            ]
            savings_opts = ["5-10%","11-20%","21-30%",">30%","Non-cost KPI"]
            min_savings_threshold = weighted_choice(savings_opts, savings_probs[s], 1)[0]

            # ── Q31: Feature Importance Ratings (1–5 each) ───────────────────
            feat_means = {
                "feat_dashboard":    [4.5, 3.5, 4.2, 4.0, 2.5],
                "feat_auto_hvac":    [4.7, 3.8, 4.3, 4.2, 2.0],
                "feat_occupancy":    [4.5, 3.5, 4.0, 4.0, 2.2],
                "feat_predictive":   [4.6, 3.8, 4.3, 4.1, 2.3],
                "feat_air_quality":  [4.2, 3.0, 4.5, 3.8, 2.0],
                "feat_carbon":       [3.8, 2.5, 4.8, 3.5, 1.8],
                "feat_bms_integ":    [4.7, 3.5, 4.2, 4.3, 2.8],
                "feat_mobile_app":   [4.2, 3.0, 3.8, 3.8, 2.0],
                "feat_compliance":   [3.5, 2.5, 4.9, 3.5, 1.5],
            }
            features = {k: int(np.clip(np.round(np.random.normal(v[s], 0.9)), 1, 5))
                        for k, v in feat_means.items()}

            # ── Q33: Preferred Commercial Model ──────────────────────────────
            model_probs = [
                [0.10, 0.40, 0.30, 0.12, 0.08],
                [0.30, 0.10, 0.38, 0.08, 0.14],
                [0.12, 0.35, 0.28, 0.15, 0.10],
                [0.15, 0.30, 0.32, 0.12, 0.11],
                [0.45, 0.10, 0.18, 0.15, 0.12],
            ]
            model_opts = ["CAPEX","SaaS Subscription","Savings-Sharing","Managed Service","Lease"]
            commercial_model = weighted_choice(model_opts, model_probs[s], 1)[0]

            # ── Q34: Max Payback Period ───────────────────────────────────────
            payback_probs = [
                [0.15, 0.40, 0.30, 0.12, 0.02, 0.01],
                [0.08, 0.25, 0.38, 0.20, 0.07, 0.02],
                [0.12, 0.35, 0.32, 0.14, 0.05, 0.02],
                [0.10, 0.30, 0.35, 0.18, 0.05, 0.02],
                [0.05, 0.15, 0.30, 0.30, 0.15, 0.05],
            ]
            payback_opts = ["<1yr","1-2yr","3-5yr","5-7yr",">7yr","Savings-Guaranteed"]
            max_payback = weighted_choice(payback_opts, payback_probs[s], 1)[0]

            # ── Q35: WTP — Monthly SaaS per building ─────────────────────────
            wtp_means = [12000, 4000, 9000, 7000, 2500]
            wtp_stds  = [ 6000, 3000, 5000, 4000, 1500]
            wtp_raw = max(500, np.random.normal(wtp_means[s], wtp_stds[s]))
            wtp_aed_per_building = round(wtp_raw, -2)    # round to nearest 100
            wtp_category = pd.cut([wtp_raw],
                bins=[0, 2000, 5000, 15000, 30000, 1e9],
                labels=["<2k","2k-5k","5k-15k","15k-30k",">30k"])[0]

            # ── Q36: Pilot Interest (1=Definitely No … 5=Definitely Yes) ─────
            pilot_means = [4.5, 3.0, 4.2, 3.8, 1.8]
            pilot_interest = int(np.clip(np.round(np.random.normal(pilot_means[s], 0.9)), 1, 5))

            # ── Q44: AI Adoption Vision ───────────────────────────────────────
            vision_probs = [
                [0.55, 0.30, 0.10, 0.05],
                [0.15, 0.35, 0.25, 0.25],
                [0.40, 0.30, 0.22, 0.08],
                [0.30, 0.40, 0.20, 0.10],
                [0.08, 0.20, 0.30, 0.42],
            ]
            vision_opts = ["Standard across all","Premium/Grade-A only",
                           "Regulation-driven","Limited adoption"]
            ai_adoption_vision = weighted_choice(vision_opts, vision_probs[s], 1)[0]

            # ── Derived Target: Adoption Likelihood Class ─────────────────────
            # High-Intent (2): pilot_interest>=4 & ai_openness>=4 & vis_score>=6
            # Evaluator (1):   not High but pilot_interest>=3
            # Not Ready (0):   otherwise
            if pilot_interest >= 4 and ai_openness >= 4 and vis_score >= 5:
                adoption_class = "High-Intent Buyer"
            elif pilot_interest >= 3 or ai_openness >= 3:
                adoption_class = "Evaluator"
            else:
                adoption_class = "Not Ready"

            record = {
                "respondent_id": len(records) + 1,
                "segment":        SEG_NAMES[s],
                "role":           role,
                "building_type":  building_type,
                "num_buildings":  num_buildings,
                "total_gfa":      total_gfa,
                "primary_emirate":primary_emirate,
                "years_in_uae":   years_in_uae,
                "monthly_bill_aed": round(monthly_bill_aed, -3),
                "bill_category":  bill_category,
                "hvac_pct_of_bill": round(hvac_pct, 1),
                "bill_change_3yr": bill_change_3yr,
                "summer_cooling_spike": summer_cooling_spike,
                **monitoring,
                "realtime_visibility_score": vis_score,
                "energy_intensity_tracking": energy_intensity_tracking,
                **pain,
                "hvac_satisfaction": hvac_satisfaction,
                "strategy_awareness": strategy_awareness,
                "sustainability_strategy": sustainability_strategy,
                "tariff_impact":  tariff_impact,
                "stakeholder_pressure": stakeholder_pressure,
                **tech,
                "ai_openness":    ai_openness,
                "min_savings_threshold": min_savings_threshold,
                **features,
                "commercial_model": commercial_model,
                "max_payback":    max_payback,
                "wtp_aed_per_building": wtp_raw,
                "wtp_category":   wtp_category,
                "pilot_interest": pilot_interest,
                "ai_adoption_vision": ai_adoption_vision,
                "adoption_class": adoption_class,
            }
            records.append(record)

    return pd.DataFrame(records)

# ── NOISE & OUTLIER INJECTION ────────────────────────────────────────────────
def inject_noise(df, noise_frac=0.04, outlier_frac=0.02):
    df = df.copy()
    n = len(df)

    # Random-flip categorical fields (simulate mis-clicks)
    noise_idx = np.random.choice(n, int(n * noise_frac), replace=False)
    cat_cols = ["bill_change_3yr","summer_cooling_spike","commercial_model","max_payback"]
    for col in cat_cols:
        flip_idx = np.random.choice(noise_idx, max(1, len(noise_idx)//4), replace=False)
        df.loc[flip_idx, col] = np.random.choice(df[col].unique(), len(flip_idx))

    # Add outliers: extremely high bills, very low WTP, extreme feature scores
    outlier_idx = np.random.choice(n, int(n * outlier_frac), replace=False)
    df.loc[outlier_idx[:len(outlier_idx)//3], "monthly_bill_aed"]     *= np.random.uniform(3, 8, len(outlier_idx)//3)
    df.loc[outlier_idx[len(outlier_idx)//3:], "wtp_aed_per_building"] *= np.random.uniform(0.05, 0.2, len(outlier_idx) - len(outlier_idx)//3)

    # Introduce skewness in monthly_bill (right-skew, realistic for UAE)
    skew_idx = np.random.choice(n, int(n * 0.10), replace=False)
    df.loc[skew_idx, "monthly_bill_aed"] *= np.random.lognormal(0.5, 0.5, len(skew_idx))

    # Clip absurd values
    df["monthly_bill_aed"] = df["monthly_bill_aed"].clip(lower=3000, upper=8_000_000)
    df["wtp_aed_per_building"] = df["wtp_aed_per_building"].clip(lower=300, upper=120_000)

    # Add ~3% missing values in non-critical columns
    for col in ["hvac_satisfaction","stakeholder_pressure","feat_carbon"]:
        miss_idx = np.random.choice(n, int(n * 0.03), replace=False)
        df.loc[miss_idx, col] = np.nan

    return df.sample(frac=1).reset_index(drop=True).head(1000)

# ─── GENERATE ────────────────────────────────────────────────────────────────
print("Generating synthetic dataset …")
df_raw  = generate_dataset(N)
df      = inject_noise(df_raw)
df["respondent_id"] = range(1, len(df) + 1)
df.to_csv("/mnt/user-data/outputs/uae_hvac_survey_synthetic.csv", index=False)
print(f"✓ Dataset saved — {len(df)} rows × {df.shape[1]} columns\n")

# ─── NUMERIC PREP ─────────────────────────────────────────────────────────────
num_features = [
    "realtime_visibility_score", "hvac_pct_of_bill", "hvac_satisfaction",
    "stakeholder_pressure", "ai_openness", "pilot_interest",
    "feat_dashboard","feat_auto_hvac","feat_occupancy","feat_predictive",
    "feat_air_quality","feat_carbon","feat_bms_integ","feat_mobile_app","feat_compliance",
    "monitor_bms","monitor_iot","tech_bms","tech_iot_sensor","tech_smart_meter",
    "tech_ai_maintenance","pain_high_cost","pain_compliance","pain_no_data"
]
df_num = df[num_features].fillna(df[num_features].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# ─── HELPER ──────────────────────────────────────────────────────────────────
def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"/mnt/user-data/outputs/{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {name}.png")

# =============================================================================
# SECTION A: DESCRIPTIVE ANALYSIS
# =============================================================================
print("\n" + "="*64)
print("SECTION A — DESCRIPTIVE ANALYSIS")
print("="*64)

desc = df[["monthly_bill_aed","hvac_pct_of_bill","wtp_aed_per_building",
           "realtime_visibility_score","ai_openness","pilot_interest",
           "stakeholder_pressure"]].describe().round(1)
print(desc.to_string())

skewness = df[["monthly_bill_aed","hvac_pct_of_bill","wtp_aed_per_building"]].skew().round(3)
print(f"\nSkewness:\n{skewness.to_string()}")

# ── A1: Segment Distribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
seg_counts = df["segment"].value_counts()
axes[0].barh(seg_counts.index, seg_counts.values, color=SEG_COLORS[::-1], edgecolor="white", height=0.65)
axes[0].set_xlabel("Number of Respondents")
axes[0].set_title("Respondent Distribution by Segment", fontweight="bold")
for i, v in enumerate(seg_counts.values):
    axes[0].text(v + 5, i, f"{v} ({v/10:.1f}%)", va="center", fontsize=9, color="#333")
axes[0].set_xlim(0, 320)

adoption_counts = df["adoption_class"].value_counts()
wedge_colors = [TEAL, GOLD, CORAL]
axes[1].pie(adoption_counts.values, labels=adoption_counts.index,
            autopct="%1.1f%%", colors=wedge_colors, startangle=140,
            wedgeprops={"edgecolor":"white","linewidth":2})
axes[1].set_title("Adoption Likelihood Classification", fontweight="bold")
save_fig("A1_segment_adoption_distribution")

# ── A2: Energy Bill Distribution (right-skewed) ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df["monthly_bill_aed"]/1000, bins=50, color=NAVY, edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Monthly Bill (AED '000)")
axes[0].set_ylabel("Count")
axes[0].set_title("Monthly Electricity Bill\n(Right-Skewed Distribution)", fontweight="bold")
axes[0].axvline(df["monthly_bill_aed"].median()/1000, color=GOLD, lw=2, linestyle="--", label=f"Median: AED {df['monthly_bill_aed'].median()/1000:.0f}k")
axes[0].legend(fontsize=9)

axes[1].hist(df["hvac_pct_of_bill"], bins=35, color=TEAL, edgecolor="white", alpha=0.85)
axes[1].set_xlabel("HVAC % of Total Bill")
axes[1].set_title("HVAC Share of Energy Bill", fontweight="bold")
axes[1].axvline(df["hvac_pct_of_bill"].mean(), color=CORAL, lw=2, linestyle="--", label=f"Mean: {df['hvac_pct_of_bill'].mean():.1f}%")
axes[1].legend(fontsize=9)

axes[2].hist(df["wtp_aed_per_building"]/1000, bins=40, color=GOLD, edgecolor="white", alpha=0.85)
axes[2].set_xlabel("WTP (AED '000/month/building)")
axes[2].set_title("Willingness-to-Pay\n(Monthly SaaS per Building)", fontweight="bold")
axes[2].axvline(df["wtp_aed_per_building"].median()/1000, color=NAVY, lw=2, linestyle="--", label=f"Median: AED {df['wtp_aed_per_building'].median()/1000:.1f}k")
axes[2].legend(fontsize=9)
save_fig("A2_energy_bill_distributions")

# ── A3: Segment Profiles — Key Metrics ───────────────────────────────────────
seg_agg = df.groupby("segment").agg(
    avg_bill=("monthly_bill_aed","mean"),
    avg_hvac_pct=("hvac_pct_of_bill","mean"),
    avg_ai_openness=("ai_openness","mean"),
    avg_pilot_interest=("pilot_interest","mean"),
    avg_visibility=("realtime_visibility_score","mean"),
    avg_wtp=("wtp_aed_per_building","mean"),
    avg_stakeholder_pressure=("stakeholder_pressure","mean"),
).reset_index()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
metrics = [
    ("avg_bill","Avg Monthly Bill (AED)","Monthly Electricity Bill by Segment"),
    ("avg_hvac_pct","HVAC % of Bill","HVAC Share of Energy Cost by Segment"),
    ("avg_ai_openness","AI Openness (1–5)","AI Openness by Segment"),
    ("avg_pilot_interest","Pilot Interest (1–5)","Pilot Interest by Segment"),
    ("avg_visibility","Realtime Visibility (1–10)","Energy Visibility Score by Segment"),
    ("avg_wtp","Avg WTP (AED/month/bldg)","Willingness-to-Pay by Segment"),
]
for ax, (col, ylabel, title) in zip(axes.flat, metrics):
    short_names = [n.replace(" ", "\n") for n in seg_agg["segment"]]
    bars = ax.bar(short_names, seg_agg[col], color=SEG_COLORS, edgecolor="white", width=0.6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.tick_params(axis="x", labelsize=7.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h * 1.01,
                f"{h:,.0f}" if h > 100 else f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
save_fig("A3_segment_profile_metrics")

# ── A4: Technology & Pain Point Heatmap ──────────────────────────────────────
pain_cols = ["pain_high_cost","pain_old_hvac","pain_no_data","pain_occupancy",
             "pain_wastage","pain_comfort","pain_maintenance","pain_compliance"]
tech_cols = ["tech_smart_meter","tech_iot_sensor","tech_bms","tech_ai_maintenance","tech_solar"]

hm_pain = df.groupby("segment")[pain_cols].mean().round(2)
hm_tech = df.groupby("segment")[tech_cols].mean().round(2)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.heatmap(hm_pain, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[0],
            linewidths=0.5, cbar_kws={"label":"Prevalence Rate"})
axes[0].set_title("Pain Point Prevalence by Segment", fontweight="bold")
axes[0].set_xticklabels([c.replace("pain_","").replace("_"," ").title() for c in pain_cols], rotation=35, ha="right")
axes[0].set_yticklabels([n.replace(" ","\n") for n in hm_pain.index], rotation=0, fontsize=8)

sns.heatmap(hm_tech, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
            linewidths=0.5, cbar_kws={"label":"Deployment Rate"})
axes[1].set_title("Technology Deployment by Segment", fontweight="bold")
axes[1].set_xticklabels([c.replace("tech_","").replace("_"," ").title() for c in tech_cols], rotation=35, ha="right")
axes[1].set_yticklabels([n.replace(" ","\n") for n in hm_tech.index], rotation=0, fontsize=8)
save_fig("A4_pain_tech_heatmaps")

# =============================================================================
# SECTION B: CLUSTERING (K-Means + PCA Visualisation)
# =============================================================================
print("\n" + "="*64)
print("SECTION B — CLUSTERING ANALYSIS")
print("="*64)

# Elbow method
inertias = []
k_range  = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(k_range, inertias, "o-", color=NAVY, lw=2, markersize=7)
axes[0].axvline(5, color=TEAL, lw=2, linestyle="--", label="Chosen k=5")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (Within-Cluster SSE)")
axes[0].set_title("Elbow Method — Optimal k Selection", fontweight="bold")
axes[0].legend()

silhouettes = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = km.fit_predict(X_scaled)
    silhouettes.append(silhouette_score(X_scaled, labels))
axes[1].plot(k_range, silhouettes, "s-", color=TEAL, lw=2, markersize=7)
axes[1].axvline(5, color=NAVY, lw=2, linestyle="--", label="Chosen k=5")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Scores by k", fontweight="bold")
axes[1].legend()
save_fig("B1_elbow_silhouette")

# Fit final KMeans k=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
df["cluster"] = kmeans.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, df["cluster"])
print(f"  Silhouette Score (k=5): {sil:.4f}")

# PCA 2D
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)
df["pca1"] = pca_coords[:, 0]
df["pca2"] = pca_coords[:, 1]
var_explained = pca.explained_variance_ratio_ * 100

cluster_labels = {
    0: "Cost-Sensitive\nConventionalists",
    1: "Tech-Forward\nInnovators",
    2: "Compliance-Led\nAdopters",
    3: "Mid-Market\nEvaluators",
    4: "Passive\nResistors",
}

fig, ax = plt.subplots(figsize=(11, 8))
for cl, color in enumerate(SEG_COLORS):
    mask = df["cluster"] == cl
    ax.scatter(df.loc[mask, "pca1"], df.loc[mask, "pca2"],
               c=color, alpha=0.55, s=28, label=cluster_labels[cl], edgecolors="none")
# centroids
centers_pca = pca.transform(kmeans.cluster_centers_)
for cl, (cx, cy) in enumerate(centers_pca):
    ax.scatter(cx, cy, c=SEG_COLORS[cl], s=220, marker="*", edgecolors="black", linewidths=0.8, zorder=5)
    ax.annotate(cluster_labels[cl].replace("\n"," "), (cx, cy),
                textcoords="offset points", xytext=(10, 6), fontsize=8.5,
                fontweight="bold", color=SEG_COLORS[cl])

ax.set_xlabel(f"Principal Component 1 ({var_explained[0]:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"Principal Component 2 ({var_explained[1]:.1f}% variance)", fontsize=11)
ax.set_title("K-Means Cluster Visualisation (PCA 2D Projection)\n"
             f"Silhouette Score: {sil:.4f}  |  k = 5  |  n = 1,000", fontweight="bold", fontsize=13)
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax.grid(alpha=0.2)
save_fig("B2_kmeans_pca_clusters")

# Cluster profile radar-style bar chart
cluster_profile_cols = ["ai_openness","pilot_interest","realtime_visibility_score",
                        "stakeholder_pressure","hvac_pct_of_bill","hvac_satisfaction"]
c_agg = df.groupby("cluster")[cluster_profile_cols].mean()
c_agg.index = [cluster_labels[i].replace("\n"," ") for i in c_agg.index]

fig, ax = plt.subplots(figsize=(13, 5))
c_agg_norm = (c_agg - c_agg.min()) / (c_agg.max() - c_agg.min())  # 0-1 normalise for display
c_agg_norm.T.plot(kind="bar", ax=ax, color=SEG_COLORS, edgecolor="white", width=0.75)
ax.set_ylabel("Normalised Score (0–1)")
ax.set_title("Cluster Profiles — Key Attribute Comparison", fontweight="bold")
ax.set_xticklabels([c.replace("_"," ").title() for c in cluster_profile_cols], rotation=30, ha="right")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.set_ylim(0, 1.2)
save_fig("B3_cluster_profiles")

# =============================================================================
# SECTION C: CLASSIFICATION (Random Forest + Logistic Regression)
# =============================================================================
print("\n" + "="*64)
print("SECTION C — CLASSIFICATION ANALYSIS")
print("="*64)

clf_features = [
    "realtime_visibility_score","hvac_pct_of_bill","hvac_satisfaction",
    "stakeholder_pressure","ai_openness","pilot_interest",
    "monitor_bms","monitor_iot","tech_bms","tech_iot_sensor",
    "pain_high_cost","pain_compliance","pain_no_data",
    "feat_dashboard","feat_auto_hvac","feat_predictive","feat_compliance"
]
le = LabelEncoder()
y_clf = le.fit_transform(df["adoption_class"])
X_clf = df[clf_features].fillna(df[clf_features].median())

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
rf_cv  = cross_val_score(rf, X_clf, y_clf, cv=5, scoring="f1_macro")
y_pred_rf = rf.predict(X_test)
print("\n  Random Forest — Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
print(f"  CV F1-Macro (5-fold): {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# Logistic Regression
lr_pipe = Pipeline([("scaler",StandardScaler()), ("clf",LogisticRegression(max_iter=1000, random_state=42))])
lr_pipe.fit(X_train, y_train)
y_pred_lr = lr_pipe.predict(X_test)
lr_cv = cross_val_score(lr_pipe, X_clf, y_clf, cv=5, scoring="f1_macro")
print(f"\n  Logistic Regression CV F1-Macro: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

# Confusion Matrix + Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=le.classes_, yticklabels=le.classes_, linewidths=0.5)
axes[0].set_title("Random Forest — Confusion Matrix", fontweight="bold")
axes[0].set_ylabel("True Label"); axes[0].set_xlabel("Predicted Label")

feat_imp = pd.Series(rf.feature_importances_, index=clf_features).sort_values(ascending=True).tail(12)
feat_imp.plot(kind="barh", ax=axes[1], color=TEAL, edgecolor="white")
axes[1].set_title("Top 12 Feature Importances\n(Random Forest)", fontweight="bold")
axes[1].set_xlabel("Importance Score")
save_fig("C1_classification_results")

# =============================================================================
# SECTION D: ASSOCIATION RULE MINING (Apriori)
# =============================================================================
print("\n" + "="*64)
print("SECTION D — ASSOCIATION RULE MINING")
print("="*64)

basket_cols = [
    "pain_high_cost","pain_old_hvac","pain_no_data","pain_compliance",
    "monitor_bms","monitor_iot","tech_bms","tech_iot_sensor",
    "tech_ai_maintenance","tech_smart_meter"
]
basket_df = df[basket_cols].fillna(0).astype(bool)
col_rename = {c: c.replace("pain_","PAIN:").replace("monitor_","MON:").replace("tech_","TECH:").upper() for c in basket_cols}
basket_df = basket_df.rename(columns=col_rename)

frequent_items = apriori(basket_df, min_support=0.15, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1.2)
rules = rules.sort_values("lift", ascending=False).head(20)
print(f"\n  Found {len(rules)} rules (lift ≥ 1.2, support ≥ 0.15)")
top_rules = rules[["antecedents","consequents","support","confidence","lift"]].head(10)
top_rules["antecedents"] = top_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
top_rules["consequents"] = top_rules["consequents"].apply(lambda x: ", ".join(list(x)))
print(top_rules.to_string(index=False))

# Scatter plot: Support vs Confidence vs Lift
fig, ax = plt.subplots(figsize=(11, 6))
sc = ax.scatter(rules["support"], rules["confidence"],
                c=rules["lift"], cmap="viridis", s=rules["lift"]*60,
                alpha=0.75, edgecolors="white", linewidths=0.5)
plt.colorbar(sc, ax=ax, label="Lift")
ax.set_xlabel("Support"); ax.set_ylabel("Confidence")
ax.set_title("Association Rules — Support vs Confidence (bubble size = Lift)", fontweight="bold")
for _, row in rules.head(6).iterrows():
    label = f"{list(row['antecedents'])[0].replace('pain_','').replace('tech_','')[:18]}→{list(row['consequents'])[0][:12]}" \
            if isinstance(row['antecedents'], frozenset) else ""
    ax.annotate(f"lift={row['lift']:.2f}", (row["support"], row["confidence"]),
                fontsize=7.5, alpha=0.8, xytext=(4, 4), textcoords="offset points")
save_fig("D1_association_rules")

# =============================================================================
# SECTION E: REGRESSION (WTP & ROI Expectations)
# =============================================================================
print("\n" + "="*64)
print("SECTION E — REGRESSION ANALYSIS")
print("="*64)

reg_features = [
    "monthly_bill_aed","hvac_pct_of_bill","realtime_visibility_score",
    "ai_openness","pilot_interest","stakeholder_pressure",
    "tech_bms","tech_iot_sensor","tech_ai_maintenance",
    "monitor_bms","monitor_iot","pain_high_cost"
]
y_reg = np.log1p(df["wtp_aed_per_building"])   # log-transform for normality
X_reg = df[reg_features].fillna(df[reg_features].median())

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
gbr.fit(Xr_train, yr_train)
yr_pred = gbr.predict(Xr_test)

r2  = r2_score(yr_test, yr_pred)
mae = mean_absolute_error(np.expm1(yr_test), np.expm1(yr_pred))
print(f"\n  Gradient Boosting Regressor (log-WTP):")
print(f"    R²:  {r2:.4f}")
print(f"    MAE: AED {mae:,.0f}/month/building")

# Linear regression for interpretability
lr_reg_pipe = Pipeline([("scaler",StandardScaler()), ("reg",LinearRegression())])
lr_reg_pipe.fit(Xr_train, yr_train)
lr_r2 = r2_score(yr_test, lr_reg_pipe.predict(Xr_test))
print(f"    Linear Regression R²: {lr_r2:.4f}  (interpretability baseline)")

# Regression diagnostics plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].scatter(yr_test, yr_pred, alpha=0.4, color=NAVY, s=18, edgecolors="none")
mn, mx = min(yr_test.min(), yr_pred.min()), max(yr_test.max(), yr_pred.max())
axes[0].plot([mn, mx], [mn, mx], "--", color=CORAL, lw=2, label="Perfect fit")
axes[0].set_xlabel("Actual log(WTP)"); axes[0].set_ylabel("Predicted log(WTP)")
axes[0].set_title(f"Actual vs Predicted WTP\n(GBR, R² = {r2:.4f})", fontweight="bold")
axes[0].legend(fontsize=9)

residuals = yr_test - yr_pred
axes[1].scatter(yr_pred, residuals, alpha=0.4, color=TEAL, s=18, edgecolors="none")
axes[1].axhline(0, color=CORAL, lw=1.5, linestyle="--")
axes[1].set_xlabel("Predicted log(WTP)"); axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot\n(Homoscedasticity Check)", fontweight="bold")

feat_imp_reg = pd.Series(gbr.feature_importances_, index=reg_features).sort_values(ascending=True).tail(10)
feat_imp_reg.plot(kind="barh", ax=axes[2], color=GOLD, edgecolor="white")
axes[2].set_title("WTP Regression — Feature Importance\n(Gradient Boosting)", fontweight="bold")
axes[2].set_xlabel("Importance Score")
save_fig("E1_regression_analysis")

# WTP by segment vs adoption class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
wtp_seg = df.groupby("segment")["wtp_aed_per_building"].median().sort_values()
axes[0].barh(wtp_seg.index, wtp_seg.values, color=SEG_COLORS, edgecolor="white", height=0.6)
axes[0].set_xlabel("Median WTP (AED/month/building)")
axes[0].set_title("Median WTP by Segment", fontweight="bold")
for i, v in enumerate(wtp_seg.values):
    axes[0].text(v + 100, i, f"AED {v:,.0f}", va="center", fontsize=8.5)

df.boxplot(column="wtp_aed_per_building", by="adoption_class", ax=axes[1],
           boxprops=dict(color=NAVY), whiskerprops=dict(color=NAVY),
           medianprops=dict(color=CORAL, lw=2), flierprops=dict(marker="o", markersize=3, alpha=0.3))
axes[1].set_title("WTP Distribution by Adoption Class", fontweight="bold")
axes[1].set_xlabel("Adoption Class"); axes[1].set_ylabel("WTP (AED/month/building)")
plt.suptitle("")
save_fig("E2_wtp_analysis")

# =============================================================================
# SECTION F: DIAGNOSTIC ANALYSIS — Deep Dive
# =============================================================================
print("\n" + "="*64)
print("SECTION F — DIAGNOSTIC ANALYSIS")
print("="*64)

# ── F1: Correlation Heatmap ──────────────────────────────────────────────────
diag_cols = ["monthly_bill_aed","hvac_pct_of_bill","realtime_visibility_score",
             "ai_openness","pilot_interest","stakeholder_pressure",
             "wtp_aed_per_building","hvac_satisfaction"]
corr = df[diag_cols].corr().round(2)
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, cbar_kws={"label":"Pearson r"})
ax.set_title("Correlation Matrix — Key Survey Variables", fontweight="bold")
ax.set_xticklabels([c.replace("_"," ").title() for c in diag_cols], rotation=35, ha="right")
ax.set_yticklabels([c.replace("_"," ").title() for c in diag_cols], rotation=0)
save_fig("F1_correlation_matrix")

# ── F2: Driver Analysis — What Drives Pilot Interest? ────────────────────────
driver_cols = ["ai_openness","realtime_visibility_score","stakeholder_pressure",
               "pain_high_cost","pain_compliance","monitor_bms","tech_bms"]
correlations_with_pilot = df[driver_cols + ["pilot_interest"]].corr()["pilot_interest"].drop("pilot_interest").sort_values()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_bar = [CORAL if v < 0 else TEAL for v in correlations_with_pilot.values]
axes[0].barh(correlations_with_pilot.index, correlations_with_pilot.values, color=colors_bar, edgecolor="white", height=0.6)
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set_xlabel("Pearson Correlation with Pilot Interest")
axes[0].set_title("What Drives Pilot Adoption Interest?\n(Diagnostic — Correlation Analysis)", fontweight="bold")
axes[0].set_xticklabels
axes[0].set_yticklabels([c.replace("_"," ").title() for c in correlations_with_pilot.index], fontsize=9)
for i, v in enumerate(correlations_with_pilot.values):
    axes[0].text(v + (0.003 if v >= 0 else -0.003), i, f"{v:.3f}",
                 ha="left" if v >= 0 else "right", va="center", fontsize=8.5)
axes[0].set_xlim(-0.2, 0.8)

# Adoption class breakdown by emirate
adopt_em = df.groupby(["primary_emirate","adoption_class"]).size().unstack(fill_value=0)
adopt_em_pct = adopt_em.div(adopt_em.sum(axis=1), axis=0) * 100
adopt_em_pct.plot(kind="bar", stacked=True, ax=axes[1],
                  color=[TEAL, GOLD, CORAL], edgecolor="white", width=0.65)
axes[1].set_title("Adoption Class by Emirate\n(Diagnostic — Market Penetration)", fontweight="bold")
axes[1].set_xlabel("Emirate"); axes[1].set_ylabel("% of Respondents")
axes[1].legend(title="Adoption Class", fontsize=8, loc="upper right")
axes[1].tick_params(axis="x", rotation=30)
save_fig("F2_diagnostic_drivers")

# ── F3: Commercial Model Preference by Segment ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
cm_seg = df.groupby(["segment","commercial_model"]).size().unstack(fill_value=0)
cm_seg_pct = cm_seg.div(cm_seg.sum(axis=1), axis=0) * 100
cm_model_colors = [NAVY, TEAL, GOLD, CORAL, PURPLE]
cm_seg_pct.plot(kind="bar", ax=axes[0], color=cm_model_colors, edgecolor="white", width=0.7)
axes[0].set_title("Commercial Model Preference by Segment", fontweight="bold")
axes[0].set_ylabel("% of Respondents")
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend(title="Model", fontsize=8, loc="upper right")
for label in axes[0].get_xticklabels():
    label.set_ha("right")

# Payback tolerance by adoption class
pb_adopt = df.groupby(["adoption_class","max_payback"]).size().unstack(fill_value=0)
pb_adopt_pct = pb_adopt.div(pb_adopt.sum(axis=1), axis=0) * 100
pb_colors = [MINT, TEAL, NAVY, GOLD, ORANGE, CORAL]
pb_adopt_pct.plot(kind="bar", ax=axes[1], color=pb_colors[:len(pb_adopt_pct.columns)], edgecolor="white", width=0.65)
axes[1].set_title("Payback Period Tolerance by Adoption Class", fontweight="bold")
axes[1].set_ylabel("% of Respondents")
axes[1].tick_params(axis="x", rotation=20)
axes[1].legend(title="Payback", fontsize=8, loc="upper right")
save_fig("F3_commercial_diagnostic")

# =============================================================================
# SUMMARY OUTPUT
# =============================================================================
print("\n" + "="*64)
print("ANALYSIS COMPLETE — OUTPUT SUMMARY")
print("="*64)
print(f"\n  Dataset:         uae_hvac_survey_synthetic.csv   ({len(df)} rows × {df.shape[1]} cols)")
print(f"  Skewness (bill): {df['monthly_bill_aed'].skew():.3f} (right-skewed, realistic)")
print(f"  HVAC % mean:     {df['hvac_pct_of_bill'].mean():.1f}%  |  std: {df['hvac_pct_of_bill'].std():.1f}%")
print(f"  Adoption split:  {dict(df['adoption_class'].value_counts())}")
print(f"\n  Clustering:      Silhouette = {sil:.4f}  (>0.30 = meaningful structure)")
print(f"  Classification:  RF CV F1-Macro = {rf_cv.mean():.4f}  |  LR F1 = {lr_cv.mean():.4f}")
print(f"  Regression:      GBR R² = {r2:.4f}  |  MAE = AED {mae:,.0f}/month/bldg")
print(f"  Assoc. Rules:    {len(rules)} rules with lift ≥ 1.2")

print("\n  Saved visualisations:")
charts = [
    "A1_segment_adoption_distribution", "A2_energy_bill_distributions",
    "A3_segment_profile_metrics", "A4_pain_tech_heatmaps",
    "B1_elbow_silhouette", "B2_kmeans_pca_clusters", "B3_cluster_profiles",
    "C1_classification_results", "D1_association_rules",
    "E1_regression_analysis", "E2_wtp_analysis",
    "F1_correlation_matrix", "F2_diagnostic_drivers", "F3_commercial_diagnostic"
]
for c in charts:
    print(f"    • {c}.png")

print("\n  ✓ All outputs written to /mnt/user-data/outputs/")
print("="*64)
