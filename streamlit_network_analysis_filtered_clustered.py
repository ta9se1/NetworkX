
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.title("📊 人物・所属・テーマのネットワーク分析＆クラスタリングツール")

uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("ファイルを読み込みました！")

    st.subheader("カラムの選択")
    columns = df.columns.tolist()
    person_col = st.selectbox("人物名カラムを選んでください", columns)
    org_col = st.selectbox("所属名カラムを選んでください", columns)
    theme_col = st.selectbox("テーマカラムを選んでください", columns)

    df_clean = df.dropna(subset=[person_col, org_col, theme_col]).copy()

    def split_items(text):
        if pd.isna(text) or not isinstance(text, str):
            return []
        return [x.strip() for x in re.split(r'[;,、]', text) if x.strip()]

    df_clean["人物リスト"] = df_clean[person_col].apply(split_items)
    df_clean["所属リスト"] = df_clean[org_col].apply(split_items)
    df_clean["テーマリスト"] = df_clean[theme_col].apply(split_items)

    # フィルタ
    all_people = sorted(set(p for sublist in df_clean["人物リスト"] for p in sublist))
    all_orgs = sorted(set(o for sublist in df_clean["所属リスト"] for o in sublist))

    selected_people = st.multiselect("人物でフィルタ（空欄で全て）", all_people)
    selected_orgs = st.multiselect("所属でフィルタ（空欄で全て）", all_orgs)

    def filter_rows(row):
        if selected_people and not any(p in row["人物リスト"] for p in selected_people):
            return False
        if selected_orgs and not any(o in row["所属リスト"] for o in selected_orgs):
            return False
        return True

    df_filtered = df_clean[df_clean.apply(filter_rows, axis=1)]

    if st.button("ネットワーク分析を実行"):
        G = nx.Graph()

        for row in df_filtered.itertuples():
            for person in row.人物リスト:
                G.add_node(person, type="人物")
                for org in row.所属リスト:
                    G.add_node(org, type="所属")
                    G.add_edge(person, org)
                for theme in row.テーマリスト:
                    G.add_node(theme, type="テーマ")
                    G.add_edge(person, theme)
            for org in row.所属リスト:
                for theme in row.テーマリスト:
                    G.add_edge(org, theme)

        pos = nx.spring_layout(G, seed=42)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x, node_y, node_text, node_color = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            ntype = G.nodes[node]['type']
            node_color.append("gold" if ntype == "人物" else "lightblue" if ntype == "所属" else "lightgreen")

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                line=dict(width=0.5, color='#888'), hoverinfo='none')
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                text=node_text, textposition="top center",
                                marker=dict(color=node_color, size=10, line_width=1),
                                hoverinfo='text')

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title='人物・所属・テーマのネットワーク図（フィルタ適用）',
                                         showlegend=False,
                                         hovermode='closest',
                                         margin=dict(b=20, l=5, r=5, t=40),
                                         xaxis=dict(showgrid=False, zeroline=False),
                                         yaxis=dict(showgrid=False, zeroline=False)))
        st.plotly_chart(fig, use_container_width=True)

        
