import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

# ── TITLE ──────────────────────────────────────────────────────────────────────
cells.append(md("""# 🛒 Retail Customer Segmentation & KPI Analytics
## A Python Workflow for CRM Decision-Making
**Course:** IS513E – Database for Direct Marketing & E-CRM  
**Dataset:** Retail Transactional Dataset (Kaggle – bhavikjikadara)  
**Group:** Yasmine CHAKER · Mohamed HADJI · Hiba RHARS · Niranjan SAWANT  
**Date:** February 2026

---
> **Run order:** Execute all cells from top to bottom. Each section is self-contained.  
> **Dataset:** Upload `retail_transactional_dataset.csv` to your Colab session or mount Google Drive before running Phase 1."""))

# ── INSTALL ────────────────────────────────────────────────────────────────────
cells.append(md("## ⚙️ 0 · Environment Setup\nInstall/upgrade required libraries (Colab-safe cell)."))
cells.append(code("""\
# Run once to install/upgrade dependencies
import subprocess, sys
pkgs = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "plotly", "kaleido"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + pkgs)
print("✅ All packages ready.")"""))

cells.append(code("""\
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime

# ── Global aesthetics ──
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({
    "figure.facecolor": "#f9f9f9",
    "axes.facecolor":   "#f9f9f9",
    "axes.edgecolor":   "#cccccc",
    "grid.color":       "#e0e0e0",
})
PALETTE = sns.color_palette("Set2", 10)
print("✅ Libraries imported.")"""))

# ── PHASE 1 ────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 📥 Phase 1 · Data Import & Cleaning
**Owner:** Niranjan SAWANT

Steps:
1. Load the CSV file
2. Inspect structure (dtypes, shape, head)
3. Handle missing values
4. Remove duplicates
5. Parse and standardise data types"""))

cells.append(code("""\
# ── 1.1  Load CSV ──────────────────────────────────────────────────────────────
# Option A – local upload (works in Colab)
# from google.colab import files
# uploaded = files.upload()
# import io
# FILE = io.BytesIO(list(uploaded.values())[0])

# Option B – direct path (set FILE to the CSV path)
FILE = "retail_transactional_dataset.csv"

df_raw = pd.read_csv(FILE)
print(f"Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
df_raw.head(3)"""))

cells.append(code("""\
# ── 1.2  Structural inspection ─────────────────────────────────────────────────
print("=== DATA TYPES ===")
print(df_raw.dtypes)
print()
print("=== MISSING VALUES ===")
miss = df_raw.isnull().sum()
miss_pct = (miss / len(df_raw) * 100).round(2)
print(pd.DataFrame({"Missing": miss, "Pct%": miss_pct})[miss > 0])"""))

cells.append(code("""\
# ── 1.3  Column name normalisation ────────────────────────────────────────────
df_raw.columns = (
    df_raw.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[\\s/]+", "_", regex=True)
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)
print("Normalised columns:", df_raw.columns.tolist())"""))

cells.append(code("""\
# ── 1.4  Column mapping  (adjust keys if your CSV differs) ──────────────────
COL = {
    "transaction_id": "transaction_id",
    "customer_id":    "customer_id",
    "date":           "date",
    "category":       "product_category",   # adjust if needed
    "quantity":       "quantity",
    "price":          "price_per_unit",
    "total":          "total_amount",
    "payment":        "payment_method",
    "location":       "store_location",
    "discount":       "discount_applied",
    "age":            "age",
    "gender":         "gender",
}

# Auto-detect columns by substring match
actual = {}
for key, hint in COL.items():
    found = [c for c in df_raw.columns if hint.split("_")[0] in c]
    actual[key] = found[0] if found else None

print("Column mapping detected:")
for k, v in actual.items():
    print(f"  {k:15s} → {v}")"""))

cells.append(code("""\
# ── 1.5  Build clean working dataframe ────────────────────────────────────────
needed = {k: v for k, v in actual.items() if v is not None}
df = df_raw[[v for v in needed.values()]].copy()
df.columns = list(needed.keys())

# Date parsing
df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")

# Numeric coercion
for col in ["quantity", "price", "total"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows where core fields are null
core = [c for c in ["transaction_id", "customer_id", "date", "total"] if c in df.columns]
before = len(df)
df.dropna(subset=core, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)
after = len(df)
print(f"Rows after cleaning: {after:,}  (removed {before - after:,})")
df.head(3)"""))

cells.append(code("""\
# ── 1.6  Compute total_amount if missing ─────────────────────────────────────
if "total" not in df.columns and "quantity" in df.columns and "price" in df.columns:
    df["total"] = df["quantity"] * df["price"]
    print("total_amount computed from quantity × price_per_unit.")

# Sanity check – negative amounts
neg = (df["total"] < 0).sum()
print(f"Negative total_amount rows: {neg}")
df = df[df["total"] >= 0]
print(f"Final clean shape: {df.shape}")"""))

# ── PHASE 2 ────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 🔒 Phase 2 · Data Diagnosis & GDPR Review
**Owner:** Niranjan SAWANT

This section assesses data quality and flags GDPR-sensitive fields in line with **Regulation (EU) 2016/679**."""))

cells.append(code(
r"""# ── 2.2  GDPR field tagging ───────────────────────────────────────────────────
GDPR_MAP = {
print("║                  GDPR FIELD RISK ASSESSMENT                         ║")
print("╠══════════════════╦═══════════════════════╦═══════════════════════════╣")
print(f"║ {'Field':<16s} ║ {'Classification':<21s} ║ {'Risk Note':<25s} ║")
print("╠══════════════════╬═══════════════════════╬═══════════════════════════╣")
for field, (cls, risk) in GDPR_MAP.items():
    if field in df.columns:
        print(f"║ {field:<16s} ║ {cls:<21s} ║ {risk:<25s} ║")
print("╚══════════════════╩═══════════════════════╩═══════════════════════════╝")

print("""
GDPR Mitigations applied in this notebook:
  ✔ Customer IDs are treated as opaque tokens – never decoded.
  ✔ Age is kept as a numeric range proxy; no real names or DOBs present.
  ✔ City/location is used for aggregate visualisation only.
  ✔ No join to external datasets that could enable re-identification.
  ✔ Data minimisation: only fields necessary for the analysis are retained.
  ✔ Legal basis: academic research under Art. 89 GDPR (publicly available Kaggle dataset).
""")"""))

cells.append(code("""\
# ── 2.3  Optional: anonymise age into bands ───────────────────────────────────
if "age" in df.columns:
    bins   = [0, 25, 35, 45, 55, 65, 120]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    print("Age binned into bands:", df["age_band"].value_counts().to_dict())"""))

# ── PHASE 3 ────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 📊 Phase 3 · Exploratory Data Analysis (EDA)
**Owner:** Yasmine CHAKER

Visualise spending patterns, identify top customers, and uncover seasonal trends."""))

cells.append(code("""\
# ── 3.1  Revenue over time (monthly) ─────────────────────────────────────────
df["year_month"] = df["date"].dt.to_period("M")
monthly = df.groupby("year_month")["total"].sum().reset_index()
monthly["year_month"] = monthly["year_month"].astype(str)

fig, ax = plt.subplots(figsize=(13, 4))
ax.fill_between(monthly["year_month"], monthly["total"], alpha=0.25, color="#4C72B0")
ax.plot(monthly["year_month"], monthly["total"], marker="o", markersize=4, color="#4C72B0", linewidth=2)
ax.set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Month"); ax.set_ylabel("Total Revenue ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
step = max(1, len(monthly) // 10)
ax.set_xticks(ax.get_xticks()[::step])
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()"""))

cells.append(code("""\
# ── 3.2  Revenue by product category ─────────────────────────────────────────
if "category" in df.columns:
    cat_rev = df.groupby("category")["total"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cat_rev.index, cat_rev.values, color=PALETTE[:len(cat_rev)])
    ax.set_title("Revenue by Product Category (Top 10)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Revenue ($)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar in bars:
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                f"${bar.get_width():,.0f}", va="center", fontsize=9)
    plt.tight_layout(); plt.show()"""))

cells.append(code("""\
# ── 3.3  Payment method distribution ─────────────────────────────────────────
if "payment" in df.columns:
    pay_counts = df["payment"].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie
    ax1.pie(pay_counts, labels=pay_counts.index, autopct="%1.1f%%",
            colors=PALETTE[:len(pay_counts)], startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax1.set_title("Payment Method Share", fontsize=13, fontweight="bold")

    # Bar
    ax2.bar(pay_counts.index, pay_counts.values, color=PALETTE[:len(pay_counts)], edgecolor="white")
    ax2.set_title("Payment Method Count", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Transactions")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout(); plt.show()"""))

cells.append(code("""\
# ── 3.4  Revenue by city / store location ────────────────────────────────────
if "location" in df.columns:
    city_rev = df.groupby("location")["total"].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(city_rev.index, city_rev.values, color=PALETTE[:len(city_rev)], edgecolor="white")
    ax.set_title("Revenue by Store Location (Top 15)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()"""))

cells.append(code("""\
# ── 3.5  Discount Applied impact ─────────────────────────────────────────────
if "discount" in df.columns:
    df["discount_flag"] = df["discount"].astype(str).str.upper().isin(["YES","TRUE","1","Y"])
    disc_group = df.groupby("discount_flag")["total"].agg(["mean", "count"]).reset_index()
    disc_group.columns = ["Discount Applied", "Avg Order Value", "# Transactions"]
    disc_group["Discount Applied"] = disc_group["Discount Applied"].map({True: "Yes", False: "No"})
    print(disc_group.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(disc_group["Discount Applied"], disc_group["Avg Order Value"],
           color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.4)
    ax.set_title("Avg Order Value: Discount vs No Discount", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Order Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
    plt.tight_layout(); plt.show()"""))

cells.append(code("""\
# ── 3.6  Top 10 customers by total spend ──────────────────────────────────────
top10 = (df.groupby("customer_id")["total"]
           .sum().sort_values(ascending=False).head(10).reset_index())
top10.columns = ["Customer ID", "Total Spend"]
top10["Customer ID"] = top10["Customer ID"].astype(str).str[-6:]  # last 6 chars for readability

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(top10["Customer ID"], top10["Total Spend"],
              color=PALETTE[:10], edgecolor="white")
ax.set_title("Top 10 Customers by Total Spend", fontsize=14, fontweight="bold")
ax.set_ylabel("Total Spend ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()"""))

cells.append(code("""\
# ── 3.7  Spending by day-of-week and hour (if datetime has time) ──────────────
df["day_of_week"] = df["date"].dt.day_name()
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_rev = df.groupby("day_of_week")["total"].mean().reindex(dow_order)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(dow_rev.index, dow_rev.values, color=PALETTE[:7], edgecolor="white")
ax.set_title("Average Transaction Value by Day of Week", fontsize=13, fontweight="bold")
ax.set_ylabel("Avg Transaction Value ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
plt.tight_layout(); plt.show()"""))

cells.append(code("""\
# ── 3.8  Demographics – Age band & Gender (if columns present) ────────────────
demo_cols = [c for c in ["age_band", "gender"] if c in df.columns]
if len(demo_cols) == 2:
    piv = df.groupby(["age_band","gender"])["total"].mean().unstack()
    piv.plot(kind="bar", figsize=(10,5), color=PALETTE[:2], edgecolor="white",
             title="Avg Spend by Age Band & Gender")
    plt.ylabel("Avg Transaction Value ($)"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.show()
elif "age_band" in df.columns:
    age_rev = df.groupby("age_band")["total"].mean()
    age_rev.plot(kind="bar", color=PALETTE, edgecolor="white",
                 title="Avg Spend by Age Band", figsize=(8,4))
    plt.ylabel("Avg ($)"); plt.xticks(rotation=0); plt.tight_layout(); plt.show()"""))

# ── PHASE 4 ────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 🎯 Phase 4 · Customer Segmentation – RFM Analysis & K-Means Clustering
**Owner:** Mohamed HADJI

### 4A – RFM Scoring
**R**ecency: days since last purchase  
**F**requency: number of transactions  
**M**onetary: total spend  

### 4B – K-Means Clustering
Apply K-Means on standardised RFM features to discover distinct customer segments."""))

cells.append(code("""\
# ── 4.1  Compute RFM table ────────────────────────────────────────────────────
SNAPSHOT = df["date"].max() + pd.Timedelta(days=1)   # day after last transaction

rfm = df.groupby("customer_id").agg(
    Recency   = ("date",  lambda x: (SNAPSHOT - x.max()).days),
    Frequency = ("transaction_id", "nunique"),
    Monetary  = ("total", "sum")
).reset_index()

print(f"RFM table: {len(rfm):,} customers")
rfm.describe().round(2)"""))

cells.append(code("""\
# ── 4.2  RFM score quintiles (1–5 per dimension) ─────────────────────────────
rfm["R_score"] = pd.qcut(rfm["Recency"],   5, labels=[5,4,3,2,1])   # lower recency ➜ higher score
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  5, labels=[1,2,3,4,5])

rfm["RFM_score"] = (rfm["R_score"].astype(int)
                  + rfm["F_score"].astype(int)
                  + rfm["M_score"].astype(int))
rfm["RFM_segment"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)

print("RFM Score distribution:")
print(rfm["RFM_score"].describe().round(2))
rfm.head()"""))

cells.append(code("""\
# ── 4.3  RFM distributions ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, color in zip(axes, ["Recency","Frequency","Monetary"], ["#4C72B0","#DD8452","#55A868"]):
    ax.hist(rfm[col], bins=30, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(f"Distribution of {col}", fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("# Customers")
    if col == "Monetary":
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
plt.suptitle("RFM Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(); plt.show()"""))

cells.append(code("""\
# ── 4.4  K-Means: determine optimal k (Elbow + Silhouette) ──────────────────
features = ["Recency", "Frequency", "Monetary"]
X = rfm[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia, silhouettes = [], []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(K_range), inertia, "bo-", linewidth=2); ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method", fontweight="bold")
ax2.plot(list(K_range), silhouettes, "go-", linewidth=2); ax2.set_xlabel("k")
ax2.set_ylabel("Silhouette Score"); ax2.set_title("Silhouette Score", fontweight="bold")
plt.tight_layout(); plt.show()

best_k = list(K_range)[silhouettes.index(max(silhouettes))]
print(f"Best k by silhouette: {best_k}  (score = {max(silhouettes):.3f})")"""))

cells.append(code("""\
# ── 4.5  Final K-Means model ─────────────────────────────────────────────────
K = best_k   # override here if you prefer a specific k, e.g. K = 4
km_final = KMeans(n_clusters=K, random_state=42, n_init=10)
rfm["Cluster"] = km_final.fit_predict(X_scaled)

# Cluster profile
profile = rfm.groupby("Cluster")[features].mean().round(2)
profile["Count"] = rfm.groupby("Cluster").size().values
profile["Pct%"]  = (profile["Count"] / len(rfm) * 100).round(1)
print("=== CLUSTER PROFILES ===")
print(profile.to_string())"""))

cells.append(code("""\
# ── 4.6  Assign business labels ───────────────────────────────────────────────
# Sort clusters by Monetary desc to assign intuitive labels
cluster_order = profile.sort_values("Monetary", ascending=False).index.tolist()
label_map = {}
label_names = ["Champions", "Loyal Customers", "At-Risk Customers",
               "Lost Customers", "New Customers", "Potential Loyalists",
               "Cannot-Lose", "Hibernating"]
for i, cl in enumerate(cluster_order):
    label_map[cl] = label_names[i] if i < len(label_names) else f"Segment {i+1}"

rfm["Segment"] = rfm["Cluster"].map(label_map)
print("Segment distribution:")
print(rfm["Segment"].value_counts().to_string())"""))

cells.append(code("""\
# ── 4.7  3-D scatter: R / F / M coloured by segment ─────────────────────────
fig_3d = px.scatter_3d(
    rfm, x="Recency", y="Frequency", z="Monetary",
    color="Segment", symbol="Segment",
    title="Customer Segments in RFM Space (3-D View)",
    labels={"Recency":"Recency (days)","Frequency":"Frequency","Monetary":"Monetary ($)"},
    opacity=0.75, height=600
)
fig_3d.update_traces(marker=dict(size=3))
fig_3d.show()"""))

cells.append(code("""\
# ── 4.8  Segment sizes bar chart ──────────────────────────────────────────────
seg_count = rfm["Segment"].value_counts().sort_values(ascending=False)
colors_seg = px.colors.qualitative.Set2[:len(seg_count)]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(seg_count.index, seg_count.values,
               color=colors_seg[:len(seg_count)], edgecolor="white")
ax.set_title("Customer Count per Segment", fontsize=14, fontweight="bold")
ax.set_xlabel("# Customers")
for bar in bars:
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(int(bar.get_width())), va="center", fontsize=9)
plt.tight_layout(); plt.show()"""))

# ── PHASE 5 ────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 📈 Phase 5 · KPI Development
**Owner:** Mohamed HADJI

KPIs computed **globally** and **per segment**:
| KPI | Formula |
|-----|---------|
| **CLV** | Total revenue per customer over observation period |
| **AOV** | Mean transaction amount |
| **Purchase Frequency** | Avg # transactions per customer |
| **Avg Basket Size** | Avg quantity per transaction |
| **Churn Proxy** | % customers inactive > threshold days |
| **Retention Rate** | % customers with ≥ 2 purchases |"""))

cells.append(code("""\
# ── 5.1  Global KPIs ──────────────────────────────────────────────────────────
OBS_DAYS = (df["date"].max() - df["date"].min()).days or 1

aov = df["total"].mean()
total_customers = df["customer_id"].nunique()
total_transactions = df["transaction_id"].nunique()
freq = total_transactions / total_customers
basket_size = df["quantity"].mean() if "quantity" in df.columns else None

# CLV estimate (simple: avg revenue per customer)
clv_global = rfm["Monetary"].mean()

# Churn proxy – recency > 90 days
CHURN_THRESHOLD = 90
churn_customers = rfm[rfm["Recency"] > CHURN_THRESHOLD]
churn_rate = len(churn_customers) / len(rfm) * 100

# Retention rate – customers with Frequency >= 2
repeat_customers = rfm[rfm["Frequency"] >= 2]
retention_rate = len(repeat_customers) / len(rfm) * 100

print("╔══════════════════════════════════════════════════════╗")
print("║              GLOBAL CRM KPI DASHBOARD               ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Dataset period        : {OBS_DAYS:>6.0f} days                ║")
print(f"║  Total customers       : {total_customers:>8,}              ║")
print(f"║  Total transactions    : {total_transactions:>8,}              ║")
print(f"║  Avg Order Value (AOV) : ${aov:>10,.2f}              ║")
print(f"║  Purchase Frequency    : {freq:>10.2f} txns/customer    ║")
if basket_size:
    print(f"║  Avg Basket Size       : {basket_size:>10.2f} units/txn       ║")
print(f"║  Customer LTV (avg)    : ${clv_global:>10,.2f}              ║")
print(f"║  Churn Proxy (>{CHURN_THRESHOLD}d)    : {churn_rate:>9.1f}%               ║")
print(f"║  Retention Rate        : {retention_rate:>9.1f}%               ║")
print("╚══════════════════════════════════════════════════════╝")"""))

cells.append(code("""\
# ── 5.2  KPIs per segment ──────────────────────────────────────────────────────
# Merge rfm segment back to transactions
df_seg = df.merge(rfm[["customer_id","Segment","Frequency","Recency","Monetary"]],
                  on="customer_id", how="left")

seg_kpi = df_seg.groupby("Segment").agg(
    Customers      = ("customer_id", "nunique"),
    Transactions   = ("transaction_id", "nunique"),
    Total_Revenue  = ("total", "sum"),
    AOV            = ("total", "mean"),
).round(2)

seg_kpi["Avg_Frequency"] = (seg_kpi["Transactions"] / seg_kpi["Customers"]).round(2)
seg_kpi["CLV_avg"]       = (seg_kpi["Total_Revenue"] / seg_kpi["Customers"]).round(2)

# Churn proxy per segment
churn_per_seg = (rfm[rfm["Recency"] > CHURN_THRESHOLD]
                 .groupby("Segment").size()
                 .rename("Churned"))
seg_kpi = seg_kpi.join(churn_per_seg, how="left").fillna(0)
seg_kpi["Churn_Rate%"] = (seg_kpi["Churned"] / seg_kpi["Customers"] * 100).round(1)

# Retention per segment
ret_per_seg = (rfm[rfm["Frequency"] >= 2]
               .groupby("Segment").size()
               .rename("Repeat"))
seg_kpi = seg_kpi.join(ret_per_seg, how="left").fillna(0)
seg_kpi["Retention%"] = (seg_kpi["Repeat"] / seg_kpi["Customers"] * 100).round(1)

print("=== KPI TABLE BY SEGMENT ===")
seg_kpi.sort_values("Total_Revenue", ascending=False)"""))

cells.append(code("""\
# ── 5.3  KPI visualisations ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

kpi_plots = [
    ("CLV_avg",     "Avg Customer Lifetime Value ($)", "CLV ($)"),
    ("AOV",         "Avg Order Value by Segment ($)",  "AOV ($)"),
    ("Churn_Rate%", "Churn Rate by Segment (%)",       "Churn %"),
    ("Retention%",  "Retention Rate by Segment (%)",   "Retention %"),
]
cmap = plt.cm.get_cmap("Set2", len(seg_kpi))

for ax, (metric, title, ylabel) in zip(axes, kpi_plots):
    data = seg_kpi[metric].sort_values(ascending=False)
    bars = ax.barh(data.index, data.values,
                   color=[cmap(i) for i in range(len(data))], edgecolor="white")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel(ylabel)
    for bar in bars:
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.1f}", va="center", fontsize=8)

plt.suptitle("CRM KPI Dashboard by Customer Segment", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout(); plt.show()"""))

# ── PHASE 6 ────────────────────────────────────────────────────────────────────
cells.append(md("""---
## 💡 Phase 6 · CRM Recommendations & Action Plan
**Owner:** Hiba RHARS

Based on the segmentation and KPI analysis, we map each cluster to concrete CRM actions aligned with **Activity-based Customer Lifecycle Management (ACLM)**."""))

cells.append(code("""\
# ── 6.1  CRM Action Matrix ────────────────────────────────────────────────────
crm_actions = {
    "Champions": {
        "Lifecycle Stage": "Active – High Value",
        "Strategy": "Reward & Retain",
        "Tactics": [
            "Exclusive VIP loyalty tier with early-access sales",
            "Personalised thank-you emails with loyalty points update",
            "Referral programme: reward for bringing new customers",
            "Co-creation: invite to product feedback panels",
        ],
        "Channel": "Email (personalised) + Mobile App push",
        "KPI Target": "Maintain retention ≥ 90%, upsell rate +15%",
    },
    "Loyal Customers": {
        "Lifecycle Stage": "Active – Medium-High Value",
        "Strategy": "Increase Frequency & Basket Size",
        "Tactics": [
            "Cross-sell complementary product categories",
            "Bundle discount: 'buy 3, save 10%' on top categories",
            "Points multiplier weekends",
            "Monthly curated newsletter with personalised picks",
        ],
        "Channel": "Email + SMS",
        "KPI Target": "Increase AOV +10%, frequency +1 txn/quarter",
    },
    "At-Risk Customers": {
        "Lifecycle Stage": "Declining – Formerly Active",
        "Strategy": "Win-Back",
        "Tactics": [
            "Re-engagement email: 'We miss you' + 15% discount",
            "Personalised product recommendations based on past purchases",
            "Limited-time offer expiring in 7 days (urgency trigger)",
            "Survey to understand reasons for inactivity",
        ],
        "Channel": "Email + Retargeting Ads",
        "KPI Target": "Recover 20% of churned revenue within 60 days",
    },
    "Lost Customers": {
        "Lifecycle Stage": "Churned",
        "Strategy": "Reactivation or GDPR-compliant suppression",
        "Tactics": [
            "Last-chance reactivation offer (one-time 20% off)",
            "If no response: move to suppression list (data minimisation)",
            "Analyse why lost: price, experience, competition",
        ],
        "Channel": "Email (1 final campaign)",
        "KPI Target": "5–10% reactivation rate",
    },
    "New Customers": {
        "Lifecycle Stage": "Onboarding",
        "Strategy": "Engage & Convert to Loyal",
        "Tactics": [
            "Welcome series: 3-email onboarding sequence",
            "First-purchase follow-up: satisfaction survey + cross-sell",
            "Introduce loyalty programme with sign-up bonus points",
            "Educational content: product usage tips",
        ],
        "Channel": "Email sequence + In-app notifications",
        "KPI Target": "Convert 30% to repeat purchasers within 90 days",
    },
    "Potential Loyalists": {
        "Lifecycle Stage": "Growing",
        "Strategy": "Nurture toward Champions",
        "Tactics": [
            "Targeted promotions on highest-affinity categories",
            "Progress tracker: 'You are X points away from VIP status'",
            "Double-points events on slow sales days",
        ],
        "Channel": "Email + Mobile App",
        "KPI Target": "Increase frequency +2 txn over 6 months",
    },
}

print("CRM ACTION MATRIX GENERATED – see table below")
rows = []
for seg, info in crm_actions.items():
    rows.append({
        "Segment":        seg,
        "Lifecycle Stage":info["Lifecycle Stage"],
        "Strategy":       info["Strategy"],
        "Channel":        info["Channel"],
        "KPI Target":     info["KPI Target"],
        "# Tactics":      len(info["Tactics"]),
    })
action_df = pd.DataFrame(rows)
action_df"""))

cells.append(code("""\
# ── 6.2  Print full tactic details ──────────────────────────────────────────
for seg, info in crm_actions.items():
    if seg in rfm["Segment"].unique():
        n = len(rfm[rfm["Segment"]==seg])
        print(f"\\n{'━'*65}")
        print(f"  {seg.upper()}  ({n:,} customers | {n/len(rfm)*100:.1f}%)")
        print(f"  Stage   : {info['Lifecycle Stage']}")
        print(f"  Strategy: {info['Strategy']}")
        print(f"  Channel : {info['Channel']}")
        print(f"  KPI     : {info['KPI Target']}")
        print("  Tactics :")
        for t in info["Tactics"]:
            print(f"    • {t}")
print(f"{'━'*65}")"""))

cells.append(code("""\
# ── 6.3  CRM Priority matrix (Monetary vs Churn Risk) ─────────────────────────
# Bubble chart: x=Churn Rate, y=CLV, size=# Customers, colour=Segment
seg_bubble = seg_kpi.reset_index()
seg_bubble.columns = [c if c != "index" else "Segment" for c in seg_bubble.columns]
# Ensure Segment column
if "Segment" not in seg_bubble.columns:
    seg_bubble = seg_bubble.rename(columns={seg_bubble.columns[0]: "Segment"})

fig_bub = px.scatter(
    seg_bubble,
    x="Churn_Rate%", y="CLV_avg",
    size="Customers", color="Segment",
    text="Segment",
    title="CRM Priority Matrix: Churn Risk vs Customer Lifetime Value",
    labels={"Churn_Rate%": "Churn Rate (%)", "CLV_avg": "Avg CLV ($)"},
    size_max=60, height=500
)
fig_bub.update_traces(textposition="top center", textfont_size=10)
fig_bub.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
    text="↑ High CLV, Low Churn = PROTECT<br>→ Low CLV, High Churn = DECIDE",
    showarrow=False, align="left", bordercolor="#888", borderwidth=1,
    bgcolor="white", font=dict(size=10))
fig_bub.show()"""))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
cells.append(md("""---
## 📋 Executive Summary

This notebook has delivered a **complete end-to-end CRM analytics pipeline**:

| Phase | Output |
|-------|--------|
| **Data Cleaning** | Clean, typed, duplicate-free dataset |
| **GDPR Review** | Risk-tagged fields, anonymisation applied |
| **EDA** | Revenue trends, category/city/payment insights |
| **RFM + K-Means** | Customer segments with business labels |
| **KPI Dashboard** | CLV, AOV, Frequency, Churn, Retention per segment |
| **CRM Actions** | Personalised strategy per segment aligned with ACLM |

### Key Takeaways
- **Champions** and **Loyal Customers** drive the majority of revenue → invest in retention.
- **At-Risk / Lost** clusters require targeted win-back campaigns with urgency triggers.
- **New Customers** need a structured onboarding journey to convert them to loyal buyers.
- Combining **RFM scoring** with **K-Means** produces richer, more actionable segments than quintiles alone.

### GDPR Reminder
All analysis treats Customer IDs as opaque tokens. No real PII is present. City-level data is used only for aggregate geographic visualisation. Under Art. 89 GDPR, academic research on publicly available data is permissible with appropriate safeguards.

---
*Generated for IS513E – Database for Direct Marketing & E-CRM | Group: Chaker · Hadji · Rhars · Sawant*"""))

# ── BUILD NOTEBOOK ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"name": "Retail_CRM_Analytics_IS513E.ipynb", "provenance": []}
    },
    "cells": cells
}

OUT = "retail_crm_analytics.ipynb"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Count cell types
md_count   = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"✅ Notebook written to: {OUT}")
print(f"   Markdown cells : {md_count}")
print(f"   Code cells     : {code_count}")
print(f"   Total cells    : {len(cells)}")
