import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def set_aesthetics():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.facecolor": "#BCE3FB",
        "axes.facecolor":   "#BCE3FB",
        "axes.edgecolor":   "#cccccc",
        "grid.color":       "#A8D0E6",
        "text.color":       "#1f4287",
        "axes.labelcolor":  "#1f4287",
        "xtick.color":      "#1f4287",
        "ytick.color":      "#1f4287",
    })

def decorate_fig(fig):
    fig.patch.set_linewidth(3)
    fig.patch.set_edgecolor('#1f4287')

def main():
    os.makedirs("assets", exist_ok=True)
    set_aesthetics()
    
    # Custom colors for bars/pies
    colors = ["#224584", "#48539B", "#71659C", "#917992", "#A6AFAE", "#3A5C8C", "#5D7B9D"]
    
    # ── PHASE 1: LOAD & CLEAN ACTUAL DATA
    FILE = "retail_transactional_dataset.csv"
    if not os.path.exists(FILE):
        print(f"Error: {FILE} not found.")
        return
        
    df = pd.read_csv(FILE)
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s/]+", "_", regex=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
    df.dropna(subset=["transaction_id", "customer_id", "date", "total_amount"], inplace=True)
    
    # Rename mapped columns for consistency with notebook
    df = df.rename(columns={"total_amount": "total", "payment_method": "payment"})
    
    # ── PHASE 3: AUTHENTIC EDA CHARTS
    
    # Chart 1: Monthly Revenue Trend
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("year_month")["total"].sum().reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    decorate_fig(fig)
    ax.fill_between(monthly["year_month"], monthly["total"], alpha=0.3, color="#224584")
    ax.plot(monthly["year_month"], monthly["total"], marker="o", color="#224584", linewidth=2)
    ax.set_title("Actual Monthly Revenue Trend", fontsize=14, pad=15)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Total Revenue ($)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    step = max(1, len(monthly) // 10)
    tick_pos = list(range(0, len(monthly), step))
    if tick_pos:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([monthly["year_month"].iloc[i] for i in tick_pos], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("assets/revenue_trend.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 2: Payment Method distribution
    if "payment" in df.columns:
        pay_counts = df["payment"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        decorate_fig(fig)
        ax.pie(pay_counts, labels=pay_counts.index, autopct="%1.1f%%",
                colors=colors[:len(pay_counts)], startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title("Payment Method Share", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig("assets/payment_methods.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── PHASE 4: RFM & KMEANS
    SNAPSHOT = df["date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        Recency   = ("date", lambda x: (SNAPSHOT - x.max()).days),
        Frequency = ("transaction_id", "nunique"),
        Monetary  = ("total", "sum")
    ).reset_index()

    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run K-Means with K=5 for visual consistency
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(X_scaled)
    
    # Assign names
    profile = rfm.groupby("Cluster")[features].mean()
    cluster_order = profile.sort_values("Monetary", ascending=False).index.tolist()
    labels = ["Champions", "Loyal Customers", "At Risk", "New Customers", "Hibernating"]
    label_map = {cl: labels[i] for i, cl in enumerate(cluster_order)}
    rfm["Segment"] = rfm["Cluster"].map(label_map)

    # Chart 3: Segment Sizes
    seg_count = rfm["Segment"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    decorate_fig(fig)
    bars = ax.barh(seg_count.index, seg_count.values, color=colors[:len(seg_count)], edgecolor="none")
    ax.set_title("Customer Count per Segment", fontsize=14, pad=15)
    ax.set_xlabel("# Customers", fontsize=12)
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontsize=10, color="#1f4287")
    plt.tight_layout()
    plt.savefig("assets/segment_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── PHASE 5: KPI DEVELOPMENT
    df_seg = df.merge(rfm[["customer_id", "Segment"]], on="customer_id", how="left")
    seg_kpi = df_seg.groupby("Segment").agg(
        Customers     = ("customer_id", "nunique"),
        Total_Revenue = ("total", "sum")
    )
    seg_kpi["CLV_avg"] = seg_kpi["Total_Revenue"] / seg_kpi["Customers"]
    
    # Chart 4: CLV by Segment
    clv_data = seg_kpi["CLV_avg"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    decorate_fig(fig)
    bars = ax.bar(clv_data.index, clv_data.values, color=colors[:len(clv_data)], edgecolor="none")
    ax.set_title("Average Customer Lifetime Value (CLV) by Segment", fontsize=14, pad=15)
    ax.set_xlabel("Customer Segment", fontsize=12)
    ax.set_ylabel("Average CLV ($)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    plt.savefig("assets/clv_by_segment.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Authentic charts generated in assets/ directory.")

if __name__ == "__main__":
    main()
