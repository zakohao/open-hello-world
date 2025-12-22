import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\vscode\open-hello-world\master"
CSV_NAME = "scatter.csv"
CSV_PATH = os.path.join(BASE_DIR, CSV_NAME)

df = pd.read_csv(CSV_PATH, header=None, sep=None, engine="python")
df.columns = ["ratio", "variance", "order_m_type"]

marker_map = {1: "$S$", 0: "$M$", 2: "$L$"}

fig, ax = plt.subplots(figsize=(7, 5))

for _, row in df.iterrows():
    ax.scatter(
        row["ratio"],
        row["variance"],
        marker=marker_map[int(row["order_m_type"])]
    )

ax.set_xlabel("Ratio")
ax.set_ylabel("Variance")
ax.set_title("Scatter Plot (S/M/L by Order Size)")

plt.savefig(
    os.path.join(BASE_DIR, "scatter_ratio_variance.png"),
    dpi=300,
    bbox_inches="tight"
)
