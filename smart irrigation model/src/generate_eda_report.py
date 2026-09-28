import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations(data_path="dataset/cleaned_binary_irrigation.csv", fig_dir="reports/figures"):
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    features = ["MOI", "temp", "humidity"]

    # 1. Target Class Distribution Bar Chart
    plt.figure(figsize=(6, 4))
    counts = df["need_irrigation"].value_counts().sort_index()
    labels = ["Class 0: Pump OFF", "Class 1: Pump ON"]
    colors = ["#2ca02c", "#d62728"]
    bars = plt.bar(labels, counts, color=colors, width=0.5, edgecolor="black")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150, f"{count} ({count/len(df)*100:.1f}%)", ha="center", fontweight="bold")
    plt.title("Target Distribution: Binary Irrigation Decision", fontsize=12, fontweight="bold")
    plt.ylabel("Sample Count")
    plt.ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/01_class_distribution.png", dpi=200)
    plt.close()

    # 2. Feature Distributions by Class (Boxplots)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pal = {"0": "#66c2a5", "1": "#fc8d62", 0: "#66c2a5", 1: "#fc8d62"}
    for idx, feat in enumerate(features):
        sns.boxplot(x="need_irrigation", y=feat, hue="need_irrigation", data=df, ax=axes[idx], palette=pal, legend=False)
        axes[idx].set_title(f"{feat} Distribution by Decision", fontweight="bold")
        axes[idx].set_xticklabels(["0: Pump OFF", "1: Pump ON"])
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/02_feature_boxplots.png", dpi=200)
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(6, 5))
    corr = df[features + ["need_irrigation"]].corr()
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True, square=True)
    plt.title("Feature & Target Pearson Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/03_correlation_matrix.png", dpi=200)
    plt.close()

    # 4. Temperature vs Humidity vs Irrigation Decision Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="temp", y="humidity", hue="need_irrigation", data=df.sample(2000, random_state=42), palette={0: "#2ca02c", 1: "#d62728"}, alpha=0.6)
    plt.title("Atmospheric Evaporative Demand: Temp vs Humidity", fontweight="bold")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Relative Humidity (%)")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/04_temp_humidity_scatter.png", dpi=200)
    plt.close()

    # 5. Moisture (MOI) Cumulative Depletion KDE
    plt.figure(figsize=(7, 4))
    sns.kdeplot(df[df["need_irrigation"]==0]["MOI"], label="Class 0: Pump OFF", fill=True, color="#2ca02c", alpha=0.4)
    sns.kdeplot(df[df["need_irrigation"]==1]["MOI"], label="Class 1: Pump ON", fill=True, color="#d62728", alpha=0.4)
    plt.title("Soil Moisture (MOI) Probability Density by Class", fontweight="bold")
    plt.xlabel("MOI (Soil Moisture %)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/05_moi_density.png", dpi=200)
    plt.close()

    print("[+] All 5 Phase 1 figures successfully saved to", fig_dir)

if __name__ == "__main__":
    generate_visualizations()
