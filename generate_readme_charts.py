import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_charts():
    os.makedirs("assets", exist_ok=True)
    
    # 1. Setup Aesthetics
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
    
    # 2. Mock some summarized data based on the usual segments (to match the requested image style)
    segments = ["Champions", "At Risk", "Loyal Customers", "New Customers", "Hibernating"]
    clv = [58000, 44500, 44000, 22800, 19500]
    
    df_clv = pd.DataFrame({"Customer Segment": segments, "Average CLV ($)": clv})
    
    # Chart 1: CLV by Segment
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Custom colors mimicking the image
    colors = ["#224584", "#48539B", "#71659C", "#917992", "#A6AFAE"]
    
    bars = ax.bar(df_clv["Customer Segment"], df_clv["Average CLV ($)"], color=colors, edgecolor="none")
    ax.set_title("Average Customer Lifetime Value (CLV) by Segment", fontsize=14, pad=15)
    ax.set_xlabel("Customer Segment", fontsize=12)
    ax.set_ylabel("Average CLV ($)", fontsize=12)
    
    # Add a thin border around the figure like the image
    fig.patch.set_linewidth(3)
    fig.patch.set_edgecolor('#1f4287')
    
    plt.tight_layout()
    plt.savefig("assets/clv_by_segment.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Chart 2: Revenue Trend
    np.random.seed(42)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rev = np.random.normal(150000, 20000, 12).cumsum()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_linewidth(3)
    fig.patch.set_edgecolor('#1f4287')
    
    ax.fill_between(months, rev, alpha=0.3, color="#224584")
    ax.plot(months, rev, marker="o", color="#224584", linewidth=2)
    ax.set_title("Annual Revenue Trend Projection", fontsize=14, pad=15)
    ax.set_ylabel("Total Revenue ($)", fontsize=12)
    plt.tight_layout()
    plt.savefig("assets/revenue_trend.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    create_charts()
    print("Static charts generated in assets/ directory.")
