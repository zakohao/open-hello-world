import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\vscode\open-hello-world\master"
CSV_PATH = os.path.join(BASE_DIR, "scatter.csv")
OUT_PATH = os.path.join(BASE_DIR, "scatter_ratio_variance.png")

# 1) 明确分隔符（大部分scatter.csv是逗号）
df = pd.read_csv(CSV_PATH, header=None, sep=",", encoding="utf-8")
df.columns = ["ratio", "variance", "order_m_type"]

# 2) 清洗 & 类型转换
df = df.dropna(subset=["ratio", "variance", "order_m_type"]).copy()
df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
df["variance"] = pd.to_numeric(df["variance"], errors="coerce")
df["order_m_type"] = pd.to_numeric(df["order_m_type"], errors="coerce")
df = df.dropna(subset=["ratio", "variance", "order_m_type"])
df["order_m_type"] = df["order_m_type"].astype(int)

marker_map = {1: r"$S$", 0: r"$M$", 2: r"$L$"}

fig, ax = plt.subplots(figsize=(7, 5))

for _, row in df.iterrows():
    marker = marker_map.get(row["order_m_type"], "o")  # 未知类型用圆点
    ax.scatter(row["ratio"], row["variance"], marker=marker)

ax.set_xlabel("Ratio")
ax.set_ylabel("Variance")
ax.set_title("Scatter Plot (S/M/L by Order Size)")

os.makedirs(BASE_DIR, exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PATH)
