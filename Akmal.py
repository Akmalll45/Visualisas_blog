
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# ─────────────────────────────────────────
# 1. LOAD & PERSIAPAN DATA
# ─────────────────────────────────────────
df = pd.read_csv("Data_visualisasi_blog.csv", sep=";")
 
# Label lebih deskriptif
df["depression_label_str"] = df["depression_label"].map({0: "Tidak Depresi", 1: "Depresi"})
df["gender_id"] = df["gender"].map({"male": "Laki-laki", "female": "Perempuan"})
df["platform_id"] = df["platform_usage"].map({
    "Instagram": "Instagram",
    "TikTok": "TikTok",
    "Both": "Keduanya"
})
df["social_id"] = df["social_interaction_level"].map({
    "low": "Rendah", "medium": "Sedang", "high": "Tinggi"
})

# Palet warna tema gelap elegan
COLORS = {
    "primary":    "#6C63FF",
    "secondary":  "#F7971E",
    "accent1":    "#FF6584",
    "accent2":    "#43E97B",
    "accent3":    "#38F9D7",
    "male":       "#6C63FF",
    "female":     "#FF6584",
    "instagram":  "#E1306C",
    "tiktok":     "#69C9D0",
    "both":       "#F7971E",
    "depresi":    "#FF6584",
    "tidak":      "#43E97B",
    "bg":         "#0D0D1A",
    "card":       "#14142B",
    "text":       "#E8E8F0",
    "grid":       "rgba(255,255,255,0.07)",
}

LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card"],
    font=dict(family="'Segoe UI', Arial, sans-serif", color=COLORS["text"], size=13),
    margin=dict(l=50, r=30, t=60, b=50),
)
 
AXIS_STYLE = dict(
    gridcolor=COLORS["grid"],
    linecolor="rgba(255,255,255,0.15)",
    tickcolor="rgba(255,255,255,0.3)",
    showgrid=True,
)
 
# ─────────────────────────────────────────
# CHART 1 – Distribusi Jam Sosmed per Platform
# ─────────────────────────────────────────
fig1 = go.Figure()
platform_colors = {
    "Instagram": COLORS["instagram"],
    "TikTok":    COLORS["tiktok"],
    "Keduanya":  COLORS["both"],
}
for platform, color in platform_colors.items():
    subset = df[df["platform_id"] == platform]["daily_social_media_hours"]
    fig1.add_trace(go.Violin(
        y=subset,
        name=platform,
        box_visible=True,
        meanline_visible=True,
        fillcolor=color,
        opacity=0.7,
        line_color=color,
        marker=dict(color=color),
    ))
fig1.update_layout(
    **LAYOUT_BASE,
    title=dict(text="📱 Distribusi Jam Penggunaan Media Sosial per Platform", font=dict(size=16, color=COLORS["secondary"])),
    yaxis=dict(title="Jam/Hari", **AXIS_STYLE),
    xaxis=dict(title="Platform", **AXIS_STYLE),
    showlegend=False,
    height=420,
)
 
# ─────────────────────────────────────────
# CHART 2 – Jam Sosmed vs Performa Akademik (Scatter)
# ─────────────────────────────────────────
fig2 = px.scatter(
    df,
    x="daily_social_media_hours",
    y="academic_performance",
    color="platform_id",
    color_discrete_map={"Instagram": COLORS["instagram"], "TikTok": COLORS["tiktok"], "Keduanya": COLORS["both"]},
    opacity=0.55,
    trendline="ols",
    trendline_scope="overall",
    trendline_color_override=COLORS["accent3"],
    labels={
        "daily_social_media_hours": "Jam Sosmed / Hari",
        "academic_performance": "Performa Akademik (GPA)",
        "platform_id": "Platform",
    },
    hover_data=["age", "gender_id", "sleep_hours"],
    title="📉 Jam Sosmed vs Performa Akademik",
)
fig2.update_layout(
    **LAYOUT_BASE,
    height=420,
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=COLORS["grid"]),
    xaxis=AXIS_STYLE,
    yaxis=AXIS_STYLE,
)
fig2.update_traces(marker=dict(size=5))
 
# ─────────────────────────────────────────
# CHART 3 – Heatmap Korelasi Variabel Numerik
# ─────────────────────────────────────────
num_cols = ["daily_social_media_hours", "sleep_hours", "screen_time_before_sleep",
            "academic_performance", "physical_activity",
            "stress_level", "anxiety_level", "addiction_level", "depression_label"]
labels_id = ["Jam Sosmed", "Jam Tidur", "Screen Sebelum Tidur",
             "Performa Akademik", "Aktivitas Fisik",
             "Stres", "Kecemasan", "Kecanduan", "Depresi"]
corr = df[num_cols].corr().round(2)
 
fig3 = go.Figure(go.Heatmap(
    z=corr.values,
    x=labels_id,
    y=labels_id,
    colorscale=[
        [0.0,  "#FF6584"],
        [0.5,  "#14142B"],
        [1.0,  "#43E97B"],
    ],
    zmid=0,
    text=corr.values.round(2),
    texttemplate="%{text}",
    textfont=dict(size=11),
    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Korelasi: %{z}<extra></extra>",
))
fig3.update_layout(
    **LAYOUT_BASE,
    title=dict(text="🔥 Heatmap Korelasi Antar Variabel", font=dict(size=16, color=COLORS["secondary"])),
    height=480,
    xaxis=dict(side="bottom", tickangle=-35, tickcolor="rgba(255,255,255,0.3)"),
    yaxis=dict(autorange="reversed", tickcolor="rgba(255,255,255,0.3)"),
)
 
# ─────────────────────────────────────────
# CHART 4 – Tingkat Stres, Kecemasan & Kecanduan per Platform (Bar Grouped)
# ─────────────────────────────────────────
grp = df.groupby("platform_id")[["stress_level", "anxiety_level", "addiction_level"]].mean().reset_index()
fig4 = go.Figure()
bar_metrics = {"stress_level": ("Stres", COLORS["accent1"]),
               "anxiety_level": ("Kecemasan", COLORS["secondary"]),
               "addiction_level": ("Kecanduan", COLORS["primary"])}
for col, (label, color) in bar_metrics.items():
    fig4.add_trace(go.Bar(
        x=grp["platform_id"],
        y=grp[col].round(2),
        name=label,
        marker_color=color,
        opacity=0.85,
        text=grp[col].round(2),
        textposition="outside",
        textfont=dict(size=11),
    ))
fig4.update_layout(
    **LAYOUT_BASE,
    barmode="group",
    title=dict(text="😰 Rata-rata Stres, Kecemasan & Kecanduan per Platform", font=dict(size=16, color=COLORS["secondary"])),
    yaxis=dict(title="Skor (1–10)", **AXIS_STYLE, range=[0, 12]),
    xaxis=dict(title="Platform", **AXIS_STYLE),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=COLORS["grid"]),
    height=420,
)
 
# ─────────────────────────────────────────
# CHART 5 – Proporsi Depresi per Gender (Donut)
# ─────────────────────────────────────────
gender_dep = df[df["depression_label"] == 1]["gender_id"].value_counts()
fig5 = go.Figure(go.Pie(
    labels=gender_dep.index,
    values=gender_dep.values,
    hole=0.55,
    marker=dict(colors=[COLORS["male"], COLORS["female"]],
                line=dict(color=COLORS["bg"], width=3)),
    textinfo="label+percent",
    textfont=dict(size=14, color=COLORS["text"]),
    hovertemplate="<b>%{label}</b><br>%{value} orang (%{percent})<extra></extra>",
))
fig5.update_layout(
    **LAYOUT_BASE,
    title=dict(text="🧠 Proporsi Depresi berdasarkan Gender", font=dict(size=16, color=COLORS["secondary"])),
    height=380,
    annotations=[dict(text="Depresi", x=0.5, y=0.5, font_size=18,
                       font_color=COLORS["text"], showarrow=False)],
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0.4)"),
)
 
# ─────────────────────────────────────────
# CHART 6 – Jam Tidur vs Jam Sosmed (Bubble by Addiction)
# ─────────────────────────────────────────
fig6 = px.scatter(
    df,
    x="daily_social_media_hours",
    y="sleep_hours",
    size="addiction_level",
    color="depression_label_str",
    color_discrete_map={"Depresi": COLORS["depresi"], "Tidak Depresi": COLORS["tidak"]},
    opacity=0.6,
    labels={
        "daily_social_media_hours": "Jam Sosmed / Hari",
        "sleep_hours": "Jam Tidur / Malam",
        "addiction_level": "Tingkat Kecanduan",
        "depression_label_str": "Status Depresi",
    },
    hover_data=["age", "platform_id", "stress_level"],
    title="😴 Jam Tidur vs Jam Sosmed (Ukuran = Kecanduan)",
)
fig6.update_layout(
    **LAYOUT_BASE,
    height=420,
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=COLORS["grid"]),
    xaxis=AXIS_STYLE,
    yaxis=AXIS_STYLE,
)
 
# ─────────────────────────────────────────
# CHART 7 – Distribusi Usia Pengguna per Platform (Histogram)
# ─────────────────────────────────────────
fig7 = px.histogram(
    df,
    x="age",
    color="platform_id",
    barmode="overlay",
    nbins=7,
    opacity=0.75,
    color_discrete_map={"Instagram": COLORS["instagram"], "TikTok": COLORS["tiktok"], "Keduanya": COLORS["both"]},
    labels={"age": "Usia (tahun)", "count": "Jumlah", "platform_id": "Platform"},
    title="👥 Distribusi Usia Pengguna per Platform",
)
fig7.update_layout(
    **LAYOUT_BASE,
    height=400,
    bargap=0.05,
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=COLORS["grid"]),
    xaxis=AXIS_STYLE,
    yaxis=dict(title="Jumlah Responden", **AXIS_STYLE),
)
 
# ─────────────────────────────────────────
# CHART 8 – Radar: Profil Rata-rata per Platform
# ─────────────────────────────────────────
radar_metrics = ["daily_social_media_hours", "sleep_hours", "academic_performance",
                 "stress_level", "anxiety_level", "addiction_level"]
radar_labels  = ["Jam Sosmed", "Jam Tidur", "Performa\nAkademik",
                 "Stres", "Kecemasan", "Kecanduan"]
radar_colors  = {"Instagram": COLORS["instagram"], "TikTok": COLORS["tiktok"], "Keduanya": COLORS["both"]}
 
# Normalisasi 0–10 untuk tampilan radar yang seragam
df_norm = df.copy()
for col in radar_metrics:
    mn, mx = df[col].min(), df[col].max()
    df_norm[col] = (df[col] - mn) / (mx - mn) * 10
 
fig8 = go.Figure()
for platform, color in radar_colors.items():
    vals = df_norm[df_norm["platform_id"] == platform][radar_metrics].mean().tolist()
    vals_closed = vals + [vals[0]]
    labels_closed = radar_labels + [radar_labels[0]]
    fig8.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=labels_closed,
        fill="toself",
        name=platform,
        line_color=color,
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)",
        opacity=0.9,
    ))
fig8.update_layout(
    **LAYOUT_BASE,
    title=dict(text="📊 Profil Perilaku Rata-rata per Platform (Dinormalisasi)", font=dict(size=16, color=COLORS["secondary"])),
    polar=dict(
        bgcolor=COLORS["card"],
        radialaxis=dict(visible=True, range=[0, 10], gridcolor=COLORS["grid"],
                        tickfont=dict(size=10, color=COLORS["text"])),
        angularaxis=dict(gridcolor=COLORS["grid"], tickfont=dict(size=12, color=COLORS["text"])),
    ),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=COLORS["grid"]),
    height=450,
)
 
# ─────────────────────────────────────────
# CHART 9 – Aktivitas Fisik vs Tingkat Stres (Box per Level Interaksi Sosial)
# ─────────────────────────────────────────
fig9 = go.Figure()
interaction_colors = {"Rendah": COLORS["accent1"], "Sedang": COLORS["secondary"], "Tinggi": COLORS["accent2"]}
for level, color in interaction_colors.items():
    subset = df[df["social_id"] == level]
    fig9.add_trace(go.Box(
        x=subset["social_id"],
        y=subset["stress_level"],
        name=level,
        marker_color=color,
        boxmean=True,
        line_color=color,
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.35)",
        jitter=0.4,
        pointpos=0,
        marker=dict(size=3, opacity=0.4),
    ))
fig9.update_layout(
    **LAYOUT_BASE,
    title=dict(text="🏃 Tingkat Stres berdasarkan Interaksi Sosial", font=dict(size=16, color=COLORS["secondary"])),
    yaxis=dict(title="Tingkat Stres (1–10)", **AXIS_STYLE),
    xaxis=dict(title="Level Interaksi Sosial", **AXIS_STYLE,
               categoryorder="array", categoryarray=["Rendah", "Sedang", "Tinggi"]),
    showlegend=False,
    height=400,
)
 
# ─────────────────────────────────────────
# CHART 10 – Sunburst: Gender > Platform > Depresi
# ─────────────────────────────────────────
sun_df = df.groupby(["gender_id", "platform_id", "depression_label_str"]).size().reset_index(name="count")
fig10 = px.sunburst(
    sun_df,
    path=["gender_id", "platform_id", "depression_label_str"],
    values="count",
    color="platform_id",
    color_discrete_map={"Instagram": COLORS["instagram"], "TikTok": COLORS["tiktok"], "Keduanya": COLORS["both"]},
    title="🌐 Hierarki: Gender → Platform → Status Depresi",
)
fig10.update_traces(
    textfont=dict(size=12),
    insidetextorientation="radial",
)
fig10.update_layout(
    **LAYOUT_BASE,
    height=480,
)
 
# ─────────────────────────────────────────
# STATISTIK RINGKAS
# ─────────────────────────────────────────
total        = len(df)
depresi_n    = df["depression_label"].sum()
avg_sosmed   = df["daily_social_media_hours"].mean()
avg_tidur    = df["sleep_hours"].mean()
avg_akademik = df["academic_performance"].mean()
avg_stres    = df["stress_level"].mean()
