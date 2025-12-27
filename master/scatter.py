import os
import pandas as pd
import plotly.graph_objects as go

BASE_DIR = r"D:\vscode\open-hello-world\master"
CSV_PATH = os.path.join(BASE_DIR, "scatter.csv")
OUT_PATH = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "scatter_ratio_variance.png"
)

df = pd.read_csv(CSV_PATH, header=None)
df.columns = ["ratio", "variance", "order_m_type"]
df["order_m_type"] = df["order_m_type"].astype(int)

# ===== 映射定义 =====
type_name_map = {
    0: "normal",
    1: "small",
    2: "large",
}

symbol_map = {
    0: "circle",        # normal
    1: "square",        # small
    2: "triangle-up",   # large
}

fig = go.Figure()

for t in sorted(df["order_m_type"].unique()):
    sub = df[df["order_m_type"] == t]
    fig.add_trace(
        go.Scatter(
            x=sub["ratio"],
            y=sub["variance"],
            mode="markers",
            marker=dict(
                symbol=symbol_map[t],
                size=8
            ),
            name=type_name_map[t]   # ← 这里就是你要的名字
        )
    )

fig.update_layout(
    title="Scatter Plot (Order Size)",
    xaxis_title="Ratio",
    yaxis_title="Variance",
    template="simple_white",
    width=700,
    height=500,
    legend_title_text="Order type"
)

fig.write_image(OUT_PATH, scale=2)

print("Saved to:", OUT_PATH)
