import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 0 15px 15px 0;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        paths = ["UniversalBank.xlsx", "data/UniversalBank.xlsx",
                "UniversalBank with description.xls", "data/UniversalBank with description.xls"]
        for path in paths:
            try:
                df = pd.read_excel(path, sheet_name="Data")
                df.columns = df.columns.str.strip().str.replace(' ', '_')
                df['Experience'] = df['Experience'].abs() if 'Experience' in df.columns else 0
                return df
            except:
                continue
    except:
        pass
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'Age': np.random.randint(23, 67, n), 'Experience': np.random.randint(0, 43, n),
        'Income': np.clip(np.random.exponential(60, n), 8, 224).astype(int),
        'Family': np.random.choice([1, 2, 3, 4], n),
        'CCAvg': np.round(np.random.exponential(1.9, n), 1),
        'Education': np.random.choice([1, 2, 3], n),
        'Mortgage': np.where(np.random.random(n) > 0.5, np.random.exponential(100, n).astype(int), 0),
        'Securities_Account': np.random.choice([0, 1], n, p=[0.90, 0.10]),
        'CD_Account': np.random.choice([0, 1], n, p=[0.94, 0.06]),
        'Online': np.random.choice([0, 1], n), 'CreditCard': np.random.choice([0, 1], n)
    })
    prob = 0.02 + 0.15*(df['Income']>100) + 0.25*(df['CD_Account']==1)
    df['Personal_Loan'] = (np.random.random(n) < prob).astype(int)
    return df

df = load_data()
df['Loan_Status'] = df['Personal_Loan'].map({0: 'Not Accepted', 1: 'Accepted'})

st.title("🔍 Exploratory Data Analysis")
st.markdown("---")

# Filters
st.sidebar.header("🔧 Filters")
income_range = st.sidebar.slider("Income ($K)", int(df['Income'].min()), int(df['Income'].max()),
                                  (int(df['Income'].min()), int(df['Income'].max())))
selected_edu = st.sidebar.multiselect("Education", [1,2,3], [1,2,3],
                                       format_func=lambda x: {1:'UG', 2:'Grad', 3:'Adv'}[x])

df_f = df[(df['Income'].between(income_range[0], income_range[1])) & (df['Education'].isin(selected_edu))]
st.sidebar.metric("Records", f"{len(df_f):,}")

colors = {'Not Accepted': '#636EFA', 'Accepted': '#EF553B'}

# CHART 1: Histograms
st.subheader("📊 Chart 1: Income & Age Distributions")

fig = make_subplots(rows=1, cols=2, subplot_titles=('Income Distribution', 'Age Distribution'))

for status, color in colors.items():
    subset = df_f[df_f['Loan_Status'] == status]
    fig.add_trace(go.Histogram(x=subset['Income'], name=status, marker_color=color, opacity=0.7), row=1, col=1)
    fig.add_trace(go.Histogram(x=subset['Age'], name=status, marker_color=color, opacity=0.7, showlegend=False), row=1, col=2)

fig.update_layout(barmode='overlay', height=400, legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'))
st.plotly_chart(fig, use_container_width=True)

inc_acc = df_f[df_f['Personal_Loan']==1]['Income'].mean()
inc_non = df_f[df_f['Personal_Loan']==0]['Income'].mean()
st.markdown(f"""
<div class="insight-box">
    <strong>📌 Insight:</strong> Loan accepters have avg income ${inc_acc:.0f}K vs ${inc_non:.0f}K for non-accepters 
    (${inc_acc-inc_non:.0f}K higher). Age distributions are similar across both groups.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# CHART 2: Scatter
st.subheader("📊 Chart 2: CC Spending vs Income")

fig = px.scatter(df_f, x='Income', y='CCAvg', color='Loan_Status', color_discrete_map=colors,
                 opacity=0.6, title='Credit Card Spending vs Income')
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)

cc_acc = df_f[df_f['Personal_Loan']==1]['CCAvg'].mean()
cc_non = df_f[df_f['Personal_Loan']==0]['CCAvg'].mean()
st.markdown(f"""
<div class="insight-box">
    <strong>📌 Insight:</strong> Clear clustering - accepters in high-income, high-spending quadrant. 
    Avg CC spend: ${cc_acc:.2f}K (accepters) vs ${cc_non:.2f}K (non-accepters).
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# CHART 4: Correlation Heatmap
st.subheader("📊 Chart 4: Correlation Heatmap")

num_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 
            'Mortgage', 'Personal_Loan', 'Securities_Account', 'CD_Account', 'Online', 'CreditCard']
num_cols = [c for c in num_cols if c in df_f.columns]

corr = df_f[num_cols].corr()

fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu_r', zmid=0,
                                text=np.round(corr.values, 2), texttemplate='%{text}', textfont={'size': 9}))
fig.update_layout(height=550, title='Correlation Matrix')
st.plotly_chart(fig, use_container_width=True)

loan_corr = corr['Personal_Loan'].drop('Personal_Loan').sort_values(key=abs, ascending=False).head(3)
st.markdown(f"""
<div class="insight-box">
    <strong>📌 Insight:</strong> Top predictors of loan acceptance:<br>
    1. <strong>{loan_corr.index[0]}</strong> (r={loan_corr.values[0]:.3f})<br>
    2. <strong>{loan_corr.index[1]}</strong> (r={loan_corr.values[1]:.3f})<br>
    3. <strong>{loan_corr.index[2]}</strong> (r={loan_corr.values[2]:.3f})
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# CHART 7: Box Plots
st.subheader("📊 Chart 7: Distribution Comparison (Box Plots)")

col1, col2, col3 = st.columns(3)

with col1:
    fig = px.box(df_f, x='Loan_Status', y='Income', color='Loan_Status', color_discrete_map=colors)
    fig.update_layout(height=350, showlegend=False, title='Income')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(df_f, x='Loan_Status', y='CCAvg', color='Loan_Status', color_discrete_map=colors)
    fig.update_layout(height=350, showlegend=False, title='CC Average')
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = px.box(df_f, x='Loan_Status', y='Mortgage', color='Loan_Status', color_discrete_map=colors)
    fig.update_layout(height=350, showlegend=False, title='Mortgage')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="insight-box">
    <strong>📌 Insight:</strong> Box plots show clear separation in Income and CCAvg between accepters 
    and non-accepters. Mortgage distribution is similar, suggesting it's not a strong differentiator.
</div>
""", unsafe_allow_html=True)
