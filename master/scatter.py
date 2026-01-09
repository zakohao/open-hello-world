import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = r"D:\vscode\open-hello-world\master"
CSV_PATH = os.path.join(BASE_DIR, "scatter_1.csv")

OUT_PATH = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "makespan_scatter_ga_dqn.png"
)

# ===== 读取 CSV =====
df = pd.read_csv(CSV_PATH, header=None)
df.columns = ["ga_makespan", "dqn_makespan", "order_m_type"]

# ===== 核心修正：安全数值化（不会炸）=====
df["order_m_type"] = pd.to_numeric(df["order_m_type"], errors="coerce")

# 如果你【确定】NaN 行是无效数据，直接删
df = df.dropna(subset=["order_m_type"])

# 用 pandas 的 nullable int（关键）
df["order_m_type"] = df["order_m_type"].astype("Int64")

# ===== jobset id =====
df["jobset_id"] = range(1, len(df) + 1)

# ===== 类别映射 =====
type_name_map = {
    0: "small",
    1: "regular",
    2: "large",
}
df["order_type"] = df["order_m_type"].map(type_name_map)

# ===== 颜色映射 =====
palette = px.colors.qualitative.Plotly
color_map = {
    "small": palette[0],
    "regular": palette[1],
    "large": palette[2],
}

fig = go.Figure()

for t in [ "small","regular", "large"]:
    sub = df[df["order_type"] == t]
    if sub.empty:
        continue

    c = color_map[t]

    # GA
    fig.add_trace(
        go.Scatter(
            x=sub["ga_makespan"],
            y=sub["jobset_id"],
            mode="markers",
            marker=dict(symbol="circle", size=8, color=c),
            name=f"{t} - GA"
        )
    )

    # DQN
    fig.add_trace(
        go.Scatter(
            x=sub["dqn_makespan"],
            y=sub["jobset_id"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=8, color=c),
            name=f"{t} - DQN"
        )
    )

# ===== 连接同一 jobset 的 GA 和 DQN（虚线）=====
for _, row in df.iterrows():
    fig.add_trace(
        go.Scatter(
            x=[row["ga_makespan"], row["dqn_makespan"]],
            y=[row["jobset_id"], row["jobset_id"]],
            mode="lines",
            line=dict(
                dash="dash",
                width=1,
                color=color_map[row["order_type"]]
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )
    
# ===== 統計：order_type別に「DQN < GA」の件数をカウントして右側に表示 =====
better = df[df["dqn_makespan"] < df["ga_makespan"]]

counts = better["order_type"].value_counts()
total_counts = df["order_type"].value_counts()

# 表示順を固定
order_list = [ "small","regular","large"]

lines = ["<b>num. of jobsets</b> ( DQN makespan <NA )"]
for t in order_list:
    b = int(counts.get(t, 0))
    n = int(total_counts.get(t, 0))
    lines.append(f"{t}: {b}/{n}")

stats_text = "<br>".join(lines)

fig.add_annotation(
    x=1.35, y=1.2,                 # 右側＆下寄せ（凡例の下に来やすい）
    xref="paper", yref="paper",
    text=stats_text,
    showarrow=False,
    align="left",
    bordercolor="rgba(0,0,0,0.2)",
    borderwidth=1,
    borderpad=6,
    bgcolor="rgba(255,255,255,0.9)",
)

# 右側に余白を作る（注釈がはみ出ないように）
fig.update_layout(margin=dict(r=220))



fig.update_layout(
    xaxis_title="Makespan",
    yaxis_title="Jobset ID",
    template="simple_white",
    width=900,
    height=650,
    legend_title_text="Order Variance / Method",
    
    # === X轴 ===
    xaxis=dict(
        title_font=dict(size=22),  # X轴标题
        tickfont=dict(size=16)     # X轴刻度
    ),

    # === Y轴 ===
    yaxis=dict(
        title_font=dict(size=22),  # Y轴标题
        tickfont=dict(size=16)     # Y轴刻度
    )
)

fig.write_image(OUT_PATH, scale=2)
print("Saved to:", OUT_PATH)
