"""
Build retail_crm_analytics.ipynb for IS513E CRM project.
Avoids nested triple-quoted strings by defining each cell's source
as a plain list of strings (each line ending in \\n).
"""
import json, textwrap

def md(*lines):
    src = "\n".join(lines)
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": textwrap.dedent(src).lstrip("\n")}

# ═══════════════════════════════════════════════════════════════════════════════
cells = []

# ── TITLE
cells.append(md(
    "# Retail Customer Segmentation & KPI Analytics",
    "## A Python Workflow for CRM Decision-Making",
    "**Course:** IS513E – Database for Direct Marketing & E-CRM  ",
    "**Dataset:** Retail Transactional Dataset (Kaggle – bhavikjikadara)  ",
    "**Group:** Yasmine CHAKER · Mohamed HADJI · Hiba RHARS · Niranjan SAWANT  ",
    "**Date:** February 2026",
    "",
    "---",
    "> **Run order:** Execute all cells top-to-bottom. Each section is self-contained.",
    ">",
    "> **Dataset:** Upload `retail_transactional_dataset.csv` to your Colab session",
    "> (or mount Google Drive) before running Phase 1.",
))

# ── PHASE 0: SETUP
cells.append(md("## 0 · Environment Setup",
                "Install / upgrade required libraries (Colab-safe)."))

cells.append(code("""
    import subprocess, sys
    pkgs = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "plotly", "kaleido"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + pkgs)
    print("All packages ready.")
"""))

cells.append(code("""
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

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
    plt.rcParams.update({
        "figure.facecolor": "#f9f9f9",
        "axes.facecolor":   "#f9f9f9",
        "axes.edgecolor":   "#cccccc",
        "grid.color":       "#e0e0e0",
    })
    PALETTE = sns.color_palette("Set2", 10)
    print("Libraries imported OK.")
"""))

# ── PHASE 1: DATA IMPORT & CLEANING
cells.append(md(
    "---",
    "## Phase 1 · Data Import & Cleaning",
    "**Owner:** Niranjan SAWANT",
    "",
    "Steps: load CSV · inspect structure · handle nulls · remove duplicates · fix dtypes.",
))

cells.append(code("""
    # 1.1  Load CSV
    # Uncomment the two lines below to upload interactively in Colab:
    # from google.colab import files
    # uploaded = files.upload()   # then set FILE to the key name

    FILE = "retail_transactional_dataset.csv"   # adjust path if needed

    df_raw = pd.read_csv(FILE)
    print(f"Shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    df_raw.head(3)
"""))

cells.append(code("""
    # 1.2  Structural inspection
    print("DATA TYPES")
    print(df_raw.dtypes)
    print()
    miss     = df_raw.isnull().sum()
    miss_pct = (miss / len(df_raw) * 100).round(2)
    missing  = pd.DataFrame({"Missing": miss, "Pct%": miss_pct})
    print("MISSING VALUES")
    print(missing[missing["Missing"] > 0])
"""))

cells.append(code("""
    # 1.3  Normalise column names  (lower-snake-case)
    df_raw.columns = (
        df_raw.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\\s/]+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    print("Columns:", df_raw.columns.tolist())
"""))

cells.append(code("""
    # 1.4  Auto-detect column mapping
    HINTS = {
        "transaction_id": "transaction",
        "customer_id":    "customer",
        "date":           "date",
        "category":       "category",
        "quantity":       "quantity",
        "price":          "price",
        "total":          "total",
        "payment":        "payment",
        "location":       "location",
        "discount":       "discount",
        "age":            "age",
        "gender":         "gender",
    }
    actual = {}
    for key, hint in HINTS.items():
        found = [c for c in df_raw.columns if hint in c]
        actual[key] = found[0] if found else None

    print("Column mapping:")
    for k, v in actual.items():
        print(f"  {k:20s} -> {v}")
"""))

cells.append(code("""
    # 1.5  Build clean working dataframe
    needed = {k: v for k, v in actual.items() if v is not None}
    df = df_raw[[v for v in needed.values()]].copy()
    df.columns = list(needed.keys())

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")

    # Numeric coercion
    for col in ["quantity", "price", "total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing core fields
    core = [c for c in ["transaction_id", "customer_id", "date", "total"] if c in df.columns]
    before = len(df)
    df.dropna(subset=core, inplace=True)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"Records after cleaning: {after:,}  (removed {before - after:,})")
    df.head(3)
"""))

cells.append(code("""
    # 1.6  Compute total if missing
    if "total" not in df.columns and "quantity" in df.columns and "price" in df.columns:
        df["total"] = df["quantity"] * df["price"]
        print("total computed from quantity x price.")

    # Remove negative totals
    df = df[df["total"] >= 0]
    print(f"Final clean shape: {df.shape}")
"""))

# ── PHASE 2: DIAGNOSIS & GDPR
cells.append(md(
    "---",
    "## Phase 2 · Data Diagnosis & GDPR Review",
    "**Owner:** Niranjan SAWANT",
    "",
    "Assess data quality and flag GDPR-sensitive fields (Reg. EU 2016/679).",
))

cells.append(code("""
    # 2.1  Overall data quality report
    print("OVERALL DATA QUALITY REPORT")
    print(f"  Total records       : {len(df):,}")
    print(f"  Unique customers    : {df['customer_id'].nunique():,}")
    print(f"  Unique transactions : {df['transaction_id'].nunique():,}")
    print(f"  Date range          : {df['date'].min().date()} to {df['date'].max().date()}")
    print()
    print("DESCRIPTIVE STATISTICS")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    display(df[num_cols].describe().round(2))
"""))

cells.append(code("""
    # 2.2  GDPR field risk assessment
    GDPR_MAP = {
        "customer_id": ("Pseudonymous ID",   "LOW  - opaque token, no direct identity"),
        "age":         ("Quasi-identifier",  "MED  - combined with gender/city raises re-ID risk"),
        "gender":      ("Quasi-identifier",  "MED  - as above"),
        "location":    ("Quasi-identifier",  "MED  - city-level geographic data"),
    }

    SEP = "=" * 70
    print(SEP)
    print("  GDPR FIELD RISK ASSESSMENT")
    print(SEP)
    print(f"  {'Field':<16}  {'Classification':<22}  Risk Note")
    print("-" * 70)
    for field, (cls, risk) in GDPR_MAP.items():
        if field in df.columns:
            print(f"  {field:<16}  {cls:<22}  {risk}")
    print(SEP)

    mitigations = [
        "Customer IDs treated as opaque tokens - never decoded.",
        "Age kept as numeric range; no real names or DOBs present.",
        "City/location used for aggregate visualisation only.",
        "No join to external datasets that could enable re-identification.",
        "Data minimisation: only fields required for analysis are retained.",
        "Legal basis: academic research under Art. 89 GDPR.",
    ]
    print("\\nGDPR Mitigations applied:")
    for m in mitigations:
        print(f"  [OK] {m}")
"""))

cells.append(code("""
    # 2.3  Age band anonymisation
    if "age" in df.columns:
        bins   = [0, 25, 35, 45, 55, 65, 120]
        labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
        df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
        print("Age bands:", df["age_band"].value_counts().sort_index().to_dict())
"""))

# ── PHASE 3: EDA
cells.append(md(
    "---",
    "## Phase 3 · Exploratory Data Analysis (EDA)",
    "**Owner:** Yasmine CHAKER",
    "",
    "Visualise spending patterns, top customers, and seasonal trends.",
))

cells.append(code("""
    # 3.1  Monthly revenue trend
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("year_month")["total"].sum().reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(monthly["year_month"], monthly["total"], alpha=0.25, color="#4C72B0")
    ax.plot(monthly["year_month"], monthly["total"],
            marker="o", markersize=4, color="#4C72B0", linewidth=2)
    ax.set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    step = max(1, len(monthly) // 10)
    tick_pos = list(range(0, len(monthly), step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([monthly["year_month"].iloc[i] for i in tick_pos], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
"""))

cells.append(code("""
    # 3.2  Revenue by product category
    if "category" in df.columns:
        cat_rev = df.groupby("category")["total"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(cat_rev.index, cat_rev.values, color=PALETTE[:len(cat_rev)])
        ax.set_title("Revenue by Product Category (Top 10)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Total Revenue ($)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for bar in bars:
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"${bar.get_width():,.0f}", va="center", fontsize=9)
        plt.tight_layout()
        plt.show()
"""))

cells.append(code("""
    # 3.3  Payment method split
    if "payment" in df.columns:
        pay_counts = df["payment"].value_counts()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.pie(pay_counts, labels=pay_counts.index, autopct="%1.1f%%",
                colors=PALETTE[:len(pay_counts)], startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax1.set_title("Payment Method Share", fontsize=13, fontweight="bold")
        ax2.bar(pay_counts.index, pay_counts.values,
                color=PALETTE[:len(pay_counts)], edgecolor="white")
        ax2.set_title("Payment Method Count", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Transactions")
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        plt.show()
"""))

cells.append(code("""
    # 3.4  Revenue by store location (top 15)
    if "location" in df.columns:
        city_rev = df.groupby("location")["total"].sum().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(city_rev.index, city_rev.values, color=PALETTE[:len(city_rev)], edgecolor="white")
        ax.set_title("Revenue by Store Location (Top 15)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Total Revenue ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
"""))

cells.append(code("""
    # 3.5  Discount impact on AOV
    if "discount" in df.columns:
        df["discount_flag"] = df["discount"].astype(str).str.upper().isin(["YES", "TRUE", "1", "Y"])
        disc = df.groupby("discount_flag")["total"].agg(["mean", "count"]).reset_index()
        disc.columns = ["Discount Applied", "Avg Order Value", "Transactions"]
        disc["Discount Applied"] = disc["Discount Applied"].map({True: "Discounted", False: "Full Price"})
        print(disc.to_string(index=False))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(disc["Discount Applied"], disc["Avg Order Value"],
               color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.4)
        ax.set_title("AOV: Discounted vs Full Price", fontsize=13, fontweight="bold")
        ax.set_ylabel("Avg Order Value ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
        plt.tight_layout()
        plt.show()
"""))

cells.append(code("""
    # 3.6  Top 10 customers by spend
    top10 = (df.groupby("customer_id")["total"]
               .sum().sort_values(ascending=False).head(10).reset_index())
    top10.columns = ["Customer ID", "Total Spend"]
    top10["Customer ID"] = top10["Customer ID"].astype(str).str[-6:]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(top10["Customer ID"], top10["Total Spend"],
                  color=PALETTE[:10], edgecolor="white")
    ax.set_title("Top 10 Customers by Total Spend", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Spend ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
"""))

cells.append(code("""
    # 3.7  Average spend by day of week
    df["day_of_week"] = df["date"].dt.day_name()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_rev = df.groupby("day_of_week")["total"].mean().reindex(dow_order)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(dow_rev.index, dow_rev.values, color=PALETTE[:7], edgecolor="white")
    ax.set_title("Avg Transaction Value by Day of Week", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Transaction ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
    plt.tight_layout()
    plt.show()
"""))

cells.append(code("""
    # 3.8  Demographics heatmap (age band x gender)
    demo_cols = [c for c in ["age_band", "gender"] if c in df.columns]
    if len(demo_cols) == 2:
        piv = df.groupby(["age_band", "gender"])["total"].mean().unstack()
        piv.plot(kind="bar", figsize=(10, 5), color=PALETTE[:2], edgecolor="white",
                 title="Avg Spend by Age Band & Gender")
        plt.ylabel("Avg Transaction ($)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    elif "age_band" in df.columns:
        age_rev = df.groupby("age_band")["total"].mean()
        age_rev.plot(kind="bar", color=PALETTE, edgecolor="white",
                     title="Avg Spend by Age Band", figsize=(8, 4))
        plt.ylabel("Avg ($)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
"""))

# ── PHASE 4: RFM + KMEANS
cells.append(md(
    "---",
    "## Phase 4 · RFM Analysis & K-Means Customer Segmentation",
    "**Owner:** Mohamed HADJI",
    "",
    "### 4A – RFM Scoring",
    "| Dimension | Definition |",
    "|-----------|-----------|",
    "| **Recency (R)** | Days since the customer's last purchase |",
    "| **Frequency (F)** | Number of distinct transactions |",
    "| **Monetary (M)** | Total spend over the observation period |",
    "",
    "### 4B – K-Means Clustering",
    "K-Means is applied on standardised RFM features to uncover data-driven segments.",
))

cells.append(code("""
    # 4.1  Build RFM table
    SNAPSHOT = df["date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        Recency   = ("date",           lambda x: (SNAPSHOT - x.max()).days),
        Frequency = ("transaction_id", "nunique"),
        Monetary  = ("total",          "sum")
    ).reset_index()

    print(f"RFM table: {len(rfm):,} customers")
    display(rfm.describe().round(2))
"""))

cells.append(code("""
    # 4.2  RFM quintile scores (1 = worst, 5 = best per dimension)
    rfm["R_score"] = pd.qcut(rfm["Recency"],
                              5, labels=[5, 4, 3, 2, 1])   # lower recency = recent = better
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"),
                              5, labels=[1, 2, 3, 4, 5])
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"),
                              5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_score"]   = rfm[["R_score","F_score","M_score"]].astype(int).sum(axis=1)
    rfm["RFM_segment"] = (rfm["R_score"].astype(str)
                         + rfm["F_score"].astype(str)
                         + rfm["M_score"].astype(str))
    print("RFM Score summary:")
    print(rfm["RFM_score"].describe().round(2))
    rfm.head()
"""))

cells.append(code("""
    # 4.3  RFM distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, color in zip(axes,
                               ["Recency", "Frequency", "Monetary"],
                               ["#4C72B0", "#DD8452", "#55A868"]):
        ax.hist(rfm[col], bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"Distribution of {col}", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("# Customers")
        if col == "Monetary":
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.suptitle("RFM Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
"""))

cells.append(code("""
    # 4.4  Optimal k: Elbow + Silhouette
    features = ["Recency", "Frequency", "Monetary"]
    X_raw    = rfm[features].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    inertia_vals, sil_vals = [], []
    K_range = range(2, 10)
    for k in K_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertia_vals.append(km.inertia_)
        sil_vals.append(silhouette_score(X_scaled, lbl))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(K_range), inertia_vals, "bo-", linewidth=2)
    ax1.set_title("Elbow Method", fontweight="bold")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
    ax2.plot(list(K_range), sil_vals, "go-", linewidth=2)
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.set_xlabel("k"); ax2.set_ylabel("Score")
    plt.tight_layout(); plt.show()

    best_k = list(K_range)[sil_vals.index(max(sil_vals))]
    print(f"Best k by silhouette: {best_k}  (score = {max(sil_vals):.3f})")
"""))

cells.append(code("""
    # 4.5  Final K-Means model
    K = best_k   # change this to override, e.g. K = 4
    km_final       = KMeans(n_clusters=K, random_state=42, n_init=10)
    rfm["Cluster"] = km_final.fit_predict(X_scaled)

    profile = rfm.groupby("Cluster")[features].mean().round(2)
    profile["Count"] = rfm.groupby("Cluster").size().values
    profile["Pct%"]  = (profile["Count"] / len(rfm) * 100).round(1)
    print("CLUSTER PROFILES")
    display(profile.sort_values("Monetary", ascending=False))
"""))

cells.append(code("""
    # 4.6  Assign business segment labels (ranked by Monetary value)
    cluster_order = profile.sort_values("Monetary", ascending=False).index.tolist()
    LABEL_POOL = [
        "Champions", "Loyal Customers", "At-Risk Customers",
        "Lost Customers", "New Customers", "Potential Loyalists",
        "Cannot-Lose", "Hibernating",
    ]
    label_map = {cl: LABEL_POOL[i] if i < len(LABEL_POOL) else f"Segment {i+1}"
                 for i, cl in enumerate(cluster_order)}

    rfm["Segment"] = rfm["Cluster"].map(label_map)
    print("Segment sizes:")
    print(rfm["Segment"].value_counts().to_string())
"""))

cells.append(code("""
    # 4.7  Interactive 3-D RFM scatter (Plotly)
    fig_3d = px.scatter_3d(
        rfm, x="Recency", y="Frequency", z="Monetary",
        color="Segment", symbol="Segment",
        title="Customer Segments in RFM Space",
        labels={"Recency":"Recency (days)", "Frequency":"# Transactions", "Monetary":"Spend ($)"},
        opacity=0.75, height=600,
    )
    fig_3d.update_traces(marker=dict(size=3))
    fig_3d.show()
"""))

cells.append(code("""
    # 4.8  Segment size bar chart
    seg_count = rfm["Segment"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(seg_count.index, seg_count.values,
                   color=PALETTE[:len(seg_count)], edgecolor="white")
    ax.set_title("Customer Count per Segment", fontsize=14, fontweight="bold")
    ax.set_xlabel("# Customers")
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontsize=9)
    plt.tight_layout(); plt.show()
"""))

# ── PHASE 5: KPIs
cells.append(md(
    "---",
    "## Phase 5 · KPI Development",
    "**Owner:** Mohamed HADJI",
    "",
    "| KPI | Description |",
    "|-----|-------------|",
    "| **CLV** | Total revenue per customer over the observation period |",
    "| **AOV** | Mean transaction amount |",
    "| **Purchase Frequency** | Avg number of transactions per customer |",
    "| **Avg Basket Size** | Avg quantity of items per transaction |",
    "| **Churn Proxy** | % customers inactive beyond a threshold |",
    "| **Retention Rate** | % customers with 2+ purchases |",
))

cells.append(code("""
    # 5.1  Global KPIs
    OBS_DAYS          = max((df["date"].max() - df["date"].min()).days, 1)
    aov               = df["total"].mean()
    total_customers   = df["customer_id"].nunique()
    total_txns        = df["transaction_id"].nunique()
    freq              = total_txns / total_customers
    basket_size       = df["quantity"].mean() if "quantity" in df.columns else None
    clv_global        = rfm["Monetary"].mean()

    CHURN_DAYS  = 90
    churn_rate  = (rfm["Recency"] > CHURN_DAYS).mean() * 100
    ret_rate    = (rfm["Frequency"] >= 2).mean() * 100

    line = "=" * 54
    print(line)
    print("  GLOBAL CRM KPI DASHBOARD")
    print(line)
    print(f"  Observation period  : {OBS_DAYS} days")
    print(f"  Unique customers    : {total_customers:,}")
    print(f"  Unique transactions : {total_txns:,}")
    print(f"  Avg Order Value     : ${aov:,.2f}")
    print(f"  Purchase Frequency  : {freq:.2f} txns/customer")
    if basket_size is not None:
        print(f"  Avg Basket Size     : {basket_size:.2f} units/txn")
    print(f"  Customer LTV (avg)  : ${clv_global:,.2f}")
    print(f"  Churn Proxy (>{CHURN_DAYS}d)  : {churn_rate:.1f}%")
    print(f"  Retention Rate      : {ret_rate:.1f}%")
    print(line)
"""))

cells.append(code("""
    # 5.2  KPIs per segment
    df_seg = df.merge(rfm[["customer_id", "Segment", "Recency", "Frequency"]],
                      on="customer_id", how="left")

    seg_kpi = df_seg.groupby("Segment").agg(
        Customers     = ("customer_id",    "nunique"),
        Transactions  = ("transaction_id", "nunique"),
        Total_Revenue = ("total",          "sum"),
        AOV           = ("total",          "mean"),
    ).round(2)

    seg_kpi["Avg_Frequency"] = (seg_kpi["Transactions"] / seg_kpi["Customers"]).round(2)
    seg_kpi["CLV_avg"]       = (seg_kpi["Total_Revenue"] / seg_kpi["Customers"]).round(2)

    churn_s = rfm[rfm["Recency"] > CHURN_DAYS].groupby("Segment").size().rename("Churned")
    repeat_s = rfm[rfm["Frequency"] >= 2].groupby("Segment").size().rename("Repeat")

    seg_kpi = seg_kpi.join(churn_s, how="left").fillna(0)
    seg_kpi = seg_kpi.join(repeat_s, how="left").fillna(0)
    seg_kpi["Churn_Rate_pct"]  = (seg_kpi["Churned"] / seg_kpi["Customers"] * 100).round(1)
    seg_kpi["Retention_pct"]   = (seg_kpi["Repeat"]  / seg_kpi["Customers"] * 100).round(1)

    display(seg_kpi.sort_values("Total_Revenue", ascending=False))
"""))

cells.append(code("""
    # 5.3  KPI dashboard charts (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    cmap = plt.colormaps["Set2"].resampled(len(seg_kpi))

    panels = [
        ("CLV_avg",         "Avg Customer Lifetime Value ($)", "CLV ($)"),
        ("AOV",             "Avg Order Value by Segment ($)",  "AOV ($)"),
        ("Churn_Rate_pct",  "Churn Rate by Segment (%)",       "Churn %"),
        ("Retention_pct",   "Retention Rate by Segment (%)",   "Retention %"),
    ]
    for ax, (metric, title, xlabel) in zip(axes, panels):
        data = seg_kpi[metric].sort_values(ascending=False)
        bars = ax.barh(data.index, data.values,
                       color=[cmap(i) for i in range(len(data))], edgecolor="white")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel(xlabel)
        for bar in bars:
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}", va="center", fontsize=8)

    plt.suptitle("CRM KPI Dashboard by Customer Segment",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.show()
"""))

# ── PHASE 6: CRM RECOMMENDATIONS
cells.append(md(
    "---",
    "## Phase 6 · CRM Recommendations & Action Plan",
    "**Owner:** Hiba RHARS",
    "",
    "Each segment is mapped to a tailored CRM strategy aligned with",
    "**Activity-based Customer Lifecycle Management (ACLM)**.",
))

cells.append(code("""
    # 6.1  CRM Action Matrix definition
    CRM_ACTIONS = {
        "Champions": {
            "stage":    "Active – High Value",
            "strategy": "Reward & Retain",
            "channel":  "Personalised email + Mobile App push",
            "kpi_goal": "Retention >= 90%, upsell rate +15%",
            "tactics":  [
                "VIP loyalty tier: early access to sales & new arrivals",
                "Personalised thank-you emails with points balance update",
                "Referral programme: bonus credits for successful referrals",
                "Co-creation invites: product feedback panels & beta testing",
            ],
        },
        "Loyal Customers": {
            "stage":    "Active – Medium-High Value",
            "strategy": "Increase Frequency & Basket Size",
            "channel":  "Email + SMS",
            "kpi_goal": "AOV +10%, frequency +1 txn/quarter",
            "tactics":  [
                "Cross-sell complementary product categories",
                "Bundle discount: Buy 3 save 10% on top-purchased categories",
                "Double-points weekends to boost purchase cadence",
                "Monthly curated newsletter with personalised top picks",
            ],
        },
        "At-Risk Customers": {
            "stage":    "Declining – Formerly Active",
            "strategy": "Win-Back Campaign",
            "channel":  "Email + Retargeting Ads",
            "kpi_goal": "Recover 20% of at-risk revenue within 60 days",
            "tactics":  [
                "Re-engagement email: 'We miss you' with 15% discount code",
                "Personalised product recommendations based on past purchases",
                "Limited-time offer (7-day countdown) to create urgency",
                "Short survey: understand reasons for disengagement",
            ],
        },
        "Lost Customers": {
            "stage":    "Churned",
            "strategy": "Reactivation or GDPR-compliant suppression",
            "channel":  "Email (single final campaign)",
            "kpi_goal": "5–10% reactivation rate",
            "tactics":  [
                "Last-chance offer: one-time 20% discount",
                "If no response: suppress from further campaigns (GDPR Art. 7)",
                "Post-mortem analysis: price? experience? competitor?",
            ],
        },
        "New Customers": {
            "stage":    "Onboarding",
            "strategy": "Engage & Convert to Loyals",
            "channel":  "Email sequence + In-app notifications",
            "kpi_goal": "30% convert to repeat buyers within 90 days",
            "tactics":  [
                "3-email welcome series: brand story, best sellers, loyalty intro",
                "Post-first-purchase survey + cross-sell suggestion",
                "Sign-up bonus points to incentivise second purchase",
                "Product usage tips and how-to content",
            ],
        },
        "Potential Loyalists": {
            "stage":    "Growing",
            "strategy": "Nurture toward Champions tier",
            "channel":  "Email + Mobile App",
            "kpi_goal": "Frequency +2 txns over next 6 months",
            "tactics":  [
                "Targeted promotions on highest-affinity product category",
                "Progress bar: 'X points away from VIP status'",
                "Exclusive flash sales on slow sales days",
            ],
        },
    }
    print("CRM Action Matrix loaded –", len(CRM_ACTIONS), "segments defined.")
"""))

cells.append(code("""
    # 6.2  Tabular summary of CRM actions
    rows = []
    for seg, info in CRM_ACTIONS.items():
        rows.append({
            "Segment":        seg,
            "Lifecycle Stage": info["stage"],
            "Strategy":       info["strategy"],
            "Channel":        info["channel"],
            "KPI Goal":       info["kpi_goal"],
            "# Tactics":      len(info["tactics"]),
        })
    action_df = pd.DataFrame(rows)
    display(action_df)
"""))

cells.append(code("""
    # 6.3  Detailed tactic printout per detected segment
    detected = rfm["Segment"].unique().tolist() if "Segment" in rfm.columns else []
    SEP = "=" * 65
    for seg, info in CRM_ACTIONS.items():
        if seg in detected:
            n = (rfm["Segment"] == seg).sum()
            pct = n / len(rfm) * 100
            print(SEP)
            print(f"  {seg.upper():50s} ({n:,} customers | {pct:.1f}%)")
            print(f"  Stage    : {info['stage']}")
            print(f"  Strategy : {info['strategy']}")
            print(f"  Channel  : {info['channel']}")
            print(f"  KPI Goal : {info['kpi_goal']}")
            print("  Tactics  :")
            for t in info["tactics"]:
                print(f"    * {t}")
    print(SEP)
"""))

cells.append(code("""
    # 6.4  CRM Priority Bubble Chart: Churn Risk vs CLV
    seg_bubble = seg_kpi.reset_index()
    # Ensure the index column (Segment) is named correctly
    if seg_bubble.columns[0] != "Segment":
        seg_bubble = seg_bubble.rename(columns={seg_bubble.columns[0]: "Segment"})

    fig_bub = px.scatter(
        seg_bubble,
        x="Churn_Rate_pct",
        y="CLV_avg",
        size="Customers",
        color="Segment",
        text="Segment",
        title="CRM Priority Matrix: Churn Risk vs Customer Lifetime Value",
        labels={"Churn_Rate_pct": "Churn Rate (%)", "CLV_avg": "Avg CLV ($)"},
        size_max=60,
        height=520,
    )
    fig_bub.update_traces(textposition="top center", textfont_size=10)
    fig_bub.add_annotation(
        x=0.01, y=0.98, xref="paper", yref="paper",
        text="^ High CLV, Low Churn = PROTECT | > Low CLV, High Churn = DECIDE",
        showarrow=False, align="left",
        bordercolor="#888", borderwidth=1,
        bgcolor="white", font=dict(size=10),
    )
    fig_bub.show()
"""))

# ── EXECUTIVE SUMMARY
cells.append(md(
    "---",
    "## Executive Summary",
    "",
    "| Phase | Deliverable |",
    "|-------|-------------|",
    "| **Data Cleaning** | Clean, typed, duplicate-free transaction dataset |",
    "| **GDPR Review** | Risk-tagged fields; anonymisation strategy documented |",
    "| **EDA** | Revenue trends, category/city/payment/demographics insights |",
    "| **RFM + K-Means** | Data-driven customer segments with business labels |",
    "| **KPI Dashboard** | CLV, AOV, Frequency, Churn %, Retention % per segment |",
    "| **CRM Actions** | Segment-specific strategies aligned with ACLM |",
    "",
    "### Key Takeaways",
    "- **Champions** and **Loyal Customers** are the revenue engine — prioritise retention investments.",
    "- **At-Risk** and **Lost** clusters require time-sensitive win-back campaigns.",
    "- **New Customers** need a structured onboarding journey to build loyalty.",
    "- Combining quintile RFM scores with K-Means yields richer segments than quintiles alone.",
    "",
    "### GDPR Reminder",
    "Customer IDs are opaque tokens throughout. City data is used only for aggregates.",
    "Under Art. 89 GDPR, academic research using publicly available data is permissible",
    "with appropriate technical and organisational safeguards.",
    "",
    "---",
    "*IS513E – Database for Direct Marketing & E-CRM |",
    "Group: Chaker · Hadji · Rhars · Sawant | February 2026*",
))

# ═══════════════════════════════════════════════════════════════════════════════
# Assemble and write notebook
# ═══════════════════════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "colab": {
            "name": "Retail_CRM_Analytics_IS513E.ipynb",
            "provenance": [],
        },
    },
    "cells": cells,
}

OUT = "retail_crm_analytics.ipynb"
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, ensure_ascii=False, indent=1)

md_n   = sum(1 for c in cells if c["cell_type"] == "markdown")
code_n = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook written: {OUT}")
print(f"  Markdown cells : {md_n}")
print(f"  Code cells     : {code_n}")
print(f"  Total cells    : {len(cells)}")
