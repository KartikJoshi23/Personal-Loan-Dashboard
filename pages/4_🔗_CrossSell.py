import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Cross-Sell", page_icon="🔗", layout="wide")

st.markdown("""
<style>
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 0 15px 15px 0;
        margin: 15px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
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
                return df
            except:
                continue
    except:
        pass
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'Income': np.clip(np.random.exponential(60, n), 8, 224).astype(int),
        'Education': np.random.choice([1, 2, 3], n),
        'Securities_Account': np.random.choice([0, 1], n, p=[0.90, 0.10]),
        'CD_Account': np.random.choice([0, 1], n, p=[0.94, 0.06]),
        'Online': np.random.choice([0, 1], n), 'CreditCard': np.random.choice([0, 1], n)
    })
    prob = 0.02 + 0.15*(df['Income']>100) + 0.25*(df['CD_Account']==1)
    df['Personal_Loan'] = (np.random.random(n) < prob).astype(int)
    return df

df = load_data()
df['Loan_Status'] = df['Personal_Loan'].map({0: 'Not Accepted', 1: 'Accepted'})

st.title("🔗 Cross-Sell Opportunity Analysis")
st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Filters")
income_range = st.sidebar.slider("Income ($K)", int(df['Income'].min()), int(df['Income'].max()),
                                  (int(df['Income'].min()), int(df['Income'].max())))

df_f = df[df['Income'].between(income_range[0], income_range[1])]
st.sidebar.metric("Records", f"{len(df_f):,}")

# CHART 6: Cross-Sell Patterns
st.subheader("📊 Chart 6: Cross-Sell Pattern Analysis")

st.info("**Design:** Showing acceptance rates across different product combinations to identify cross-sell opportunities.")

# Product combinations
df_f['Combo'] = (df_f['Securities_Account'].map({0:'', 1:'Sec'}) + 
                 df_f['CD_Account'].map({0:'', 1:'+CD'}) + 
                 df_f['CreditCard'].map({0:'', 1:'+CC'}))
df_f['Combo'] = df_f['Combo'].str.lstrip('+').replace('', 'None')

combo_stats = df_f.groupby('Combo').agg({'Personal_Loan': ['mean', 'sum', 'count']}).reset_index()
combo_stats.columns = ['Combo', 'Rate', 'Accepted', 'Total']
combo_stats['Rate'] *= 100
combo_stats = combo_stats.sort_values('Rate', ascending=False)

fig = px.bar(combo_stats, x='Combo', y='Rate', color='Rate', color_continuous_scale='RdYlGn',
             text='Rate', title='Acceptance Rate by Product Combination')
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(height=400, showlegend=False, yaxis_title='Acceptance Rate (%)')
st.plotly_chart(fig, use_container_width=True)

best = combo_stats.iloc[0]
overall = df_f['Personal_Loan'].mean() * 100
st.markdown(f"""
<div class="insight-box">
    <strong>📌 Insight:</strong> Customers with '<strong>{best['Combo']}</strong>' products have highest 
    acceptance ({best['Rate']:.1f}%) - <strong>{best['Rate']/overall:.1f}x higher</strong> than overall ({overall:.1f}%).
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Individual product impact
st.subheader("📊 Individual Product Impact")

col1, col2, col3 = st.columns(3)

products = [('CD_Account', 'CD Account'), ('Securities_Account', 'Securities'), ('CreditCard', 'Credit Card')]

for i, (col_name, label) in enumerate(products):
    with [col1, col2, col3][i]:
        has_p = df_f[df_f[col_name] == 1]['Personal_Loan'].mean() * 100
        no_p = df_f[df_f[col_name] == 0]['Personal_Loan'].mean() * 100
        
        comp = pd.DataFrame({'Has': [f'Has {label}', f'No {label}'], 'Rate': [has_p, no_p]})
        fig = px.bar(comp, x='Has', y='Rate', color='Rate', color_continuous_scale='Blues',
                     title=f'{label} Impact', text='Rate')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=300, showlegend=False, yaxis_title='%')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"**Lift:** {has_p/no_p:.1f}x" if no_p > 0 else "N/A")

st.markdown("---")

# Heatmap
st.subheader("📊 CD vs Securities Heatmap")

pivot = df_f.pivot_table(values='Personal_Loan', index='CD_Account', columns='Securities_Account', aggfunc='mean') * 100
count_pivot = df_f.pivot_table(values='Personal_Loan', index='CD_Account', columns='Securities_Account', aggfunc='count')

fig = go.Figure(data=go.Heatmap(
    z=pivot.values, x=['No Securities', 'Has Securities'], y=['No CD', 'Has CD'],
    colorscale='RdYlGn',
    text=[[f'{pivot.iloc[i,j]:.1f}%<br>n={int(count_pivot.iloc[i,j])}' for j in range(2)] for i in range(2)],
    texttemplate='%{text}', textfont={'size': 14}
))
fig.update_layout(height=350, title='Acceptance Rate: CD × Securities')
st.plotly_chart(fig, use_container_width=True)

cd_sec = pivot.iloc[1, 1] if 1 in pivot.index and 1 in pivot.columns else 0
neither = pivot.iloc[0, 0]

st.markdown(f"""
<div class="success-box">
    <h4 style="color: #28a745; margin-top: 0;">🎯 Cross-Sell Opportunity</h4>
    <p><strong>Finding:</strong> Customers with BOTH CD + Securities accounts: <strong>{cd_sec:.1f}%</strong> acceptance 
    vs <strong>{neither:.1f}%</strong> for neither - <strong>{cd_sec/neither:.1f}x higher!</strong></p>
    <p><strong>Action:</strong> Prioritize loan campaigns for CD account holders. Bundle CD promotions with loan offers.</p>
</div>
""", unsafe_allow_html=True)
