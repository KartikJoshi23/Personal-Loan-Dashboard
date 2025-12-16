import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Universal Bank - Loan Analytics", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { color: #FAFAFA !important; font-weight: 700 !important; }
    
    .insight-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
        border: 1px solid #7B68EE;
        border-left: 4px solid #7B68EE;
        padding: 25px 30px;
        border-radius: 12px;
        margin: 20px 0;
    }
    .insight-box h4 { color: #7B68EE !important; margin: 0 0 15px 0 !important; }
    .insight-box p { color: #E2E8F0 !important; line-height: 1.7; margin: 8px 0 !important; }
    .insight-box strong { color: #FAFAFA !important; }
    
    .success-box {
        background: linear-gradient(135deg, #1a2e1a 0%, #1f3d1f 100%);
        border: 1px solid #2ECC71;
        border-left: 4px solid #2ECC71;
        padding: 25px 30px;
        border-radius: 12px;
        margin: 20px 0;
    }
    .success-box h4 { color: #2ECC71 !important; margin: 0 0 15px 0 !important; }
    .success-box p { color: #E2E8F0 !important; line-height: 1.7; margin: 8px 0 !important; }
    
    .section-header {
        color: #FAFAFA;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #3D4663;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1E2130 100%);
        border-right: 1px solid #3D4663;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        paths = ["UniversalBank.xlsx", "data/UniversalBank.xlsx", "UniversalBank with description.xls", "data/UniversalBank with description.xls"]
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
        'Online': np.random.choice([0, 1], n),
        'CreditCard': np.random.choice([0, 1], n)
    })
    prob = 0.02 + 0.15*(df['Income']>100) + 0.25*(df['CD_Account']==1)
    df['Personal_Loan'] = (np.random.random(n) < prob).astype(int)
    return df

df = load_data()
df['Loan_Status'] = df['Personal_Loan'].map({0: 'Not Accepted', 1: 'Accepted'})

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 3rem;">🏦</div>
        <h2 style="color: #FAFAFA; margin: 10px 0;">Universal Bank</h2>
        <p style="color: #A0AEC0; font-size: 0.9rem;">Personal Loan Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔧 Filters")
    
    income_range = st.slider("Income Range ($K)", int(df['Income'].min()), int(df['Income'].max()), (int(df['Income'].min()), int(df['Income'].max())))
    cd_filter = st.multiselect("CD Account", options=[0, 1], default=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    sec_filter = st.multiselect("Securities Account", options=[0, 1], default=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

df_f = df[(df['Income'].between(income_range[0], income_range[1])) & (df['CD_Account'].isin(cd_filter)) & (df['Securities_Account'].isin(sec_filter))]

st.sidebar.metric("Filtered Records", f"{len(df_f):,}")

st.markdown("""
<div style="text-align: center; padding: 10px 0 30px 0;">
    <h1 style="color: #FAFAFA;">🔗 Cross-Sell Analysis</h1>
    <p style="color: #A0AEC0;">Identify product combinations that drive loan acceptance</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Cross-Sell Pattern Analysis
# =============================================================================
st.markdown('<p class="section-header">📊 Cross-Sell Pattern Analysis</p>', unsafe_allow_html=True)

st.info("💡 Analyzing how combinations of existing products impact loan acceptance rates.")

df_f['Combo'] = (
    df_f['Securities_Account'].map({0: '', 1: 'Sec'}) + 
    df_f['CD_Account'].map({0: '', 1: '+CD'}) + 
    df_f['CreditCard'].map({0: '', 1: '+CC'})
)
df_f['Combo'] = df_f['Combo'].str.lstrip('+').replace('', 'None')

combo_stats = df_f.groupby('Combo').agg({'Personal_Loan': ['mean', 'sum', 'count']}).reset_index()
combo_stats.columns = ['Combo', 'Rate', 'Accepted', 'Total']
combo_stats['Rate'] *= 100
combo_stats = combo_stats.sort_values('Rate', ascending=True)

fig = go.Figure()
fig.add_trace(go.Bar(
    y=combo_stats['Combo'],
    x=combo_stats['Rate'],
    orientation='h',
    marker_color=['#E74C3C', '#F39C12', '#3498DB', '#9B59B6', '#1ABC9C', '#2ECC71', '#27AE60', '#16A085'][:len(combo_stats)],
    text=[f"{v:.1f}%" for v in combo_stats['Rate']],
    textposition='outside',
    textfont=dict(color='#FAFAFA', size=12)
))

fig.update_layout(
    height=450,
    title=dict(text='Loan Acceptance Rate by Product Combination', font=dict(color='#FAFAFA', size=16)),
    xaxis_title='Acceptance Rate (%)',
    showlegend=False,
    paper_bgcolor='#0E1117',
    plot_bgcolor='#0E1117',
    font=dict(color='#FAFAFA'),
    xaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0')),
    yaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0'))
)

st.plotly_chart(fig, use_container_width=True)

best = combo_stats.iloc[-1]
overall = df_f['Personal_Loan'].mean() * 100

st.markdown(f"""
<div class="insight-box">
    <h4>📌 Key Insight</h4>
    <p><strong>Finding:</strong> Customers with '<strong>{best['Combo']}</strong>' products have the highest 
    acceptance rate at <strong>{best['Rate']:.1f}%</strong> ({int(best['Accepted'])} of {int(best['Total'])} customers).</p>
    <p><strong>Implication:</strong> This is <strong>{best['Rate']/overall:.1f}x higher</strong> than the overall rate ({overall:.1f}%). 
    Target these customers first!</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Individual Product Impact
st.markdown('<p class="section-header">📊 Individual Product Impact on Loan Acceptance</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

products = [
    ('CD_Account', 'CD Account', ['#5A5F72', '#F39C12']),
    ('Securities_Account', 'Securities Account', ['#5A5F72', '#3498DB']),
    ('CreditCard', 'Credit Card', ['#5A5F72', '#E74C3C'])
]

for i, (col_name, label, colors) in enumerate(products):
    with [col1, col2, col3][i]:
        has_p = df_f[df_f[col_name] == 1]['Personal_Loan'].mean() * 100
        no_p = df_f[df_f[col_name] == 0]['Personal_Loan'].mean() * 100
        lift = has_p / no_p if no_p > 0 else 0
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['No', 'Yes'],
            y=[no_p, has_p],
            marker_color=colors,
            text=[f"{no_p:.1f}%", f"{has_p:.1f}%"],
            textposition='outside',
            textfont=dict(color='#FAFAFA', size=12)
        ))
        
        fig.update_layout(
            height=350,
            title=dict(text=f'{label} Impact', font=dict(color='#FAFAFA', size=14)),
            yaxis_title='Acceptance Rate (%)',
            showlegend=False,
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            font=dict(color='#FAFAFA'),
            xaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0')),
            yaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0'))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Lift: {lift:.1f}x** more likely to accept")

st.markdown("---")

# Heatmap
st.markdown('<p class="section-header">📊 Cross-Sell Heatmap: CD Account vs Securities Account</p>', unsafe_allow_html=True)

pivot = df_f.pivot_table(values='Personal_Loan', index='CD_Account', columns='Securities_Account', aggfunc='mean') * 100
count_pivot = df_f.pivot_table(values='Personal_Loan', index='CD_Account', columns='Securities_Account', aggfunc='count')

fig = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=['No Securities', 'Has Securities'],
    y=['No CD Account', 'Has CD Account'],
    colorscale=[[0, '#5A5F72'], [0.5, '#4FD1C5'], [1, '#2ECC71']],
    text=[[f'{pivot.iloc[i,j]:.1f}%\nn={int(count_pivot.iloc[i,j])}' for j in range(2)] for i in range(2)],
    texttemplate='%{text}',
    textfont={'size': 14, 'color': '#FAFAFA'},
    hovertemplate='%{y} × %{x}<br>Rate: %{z:.1f}%<extra></extra>'
))

fig.update_layout(
    height=400,
    title=dict(text='Loan Acceptance Rate: CD Account × Securities Account', font=dict(color='#FAFAFA', size=16)),
    paper_bgcolor='#0E1117',
    plot_bgcolor='#0E1117',
    font=dict(color='#FAFAFA'),
    xaxis=dict(tickfont=dict(color='#A0AEC0')),
    yaxis=dict(tickfont=dict(color='#A0AEC0'))
)

st.plotly_chart(fig, use_container_width=True)

cd_sec = pivot.iloc[1, 1] if 1 in pivot.index and 1 in pivot.columns else 0
neither = pivot.iloc[0, 0]

st.markdown(f"""
<div class="success-box">
    <h4>🎯 Cross-Sell Opportunity Identified!</h4>
    <p><strong>Finding:</strong> Customers with <strong>BOTH CD Account AND Securities Account</strong> show 
    <strong>{cd_sec:.1f}%</strong> acceptance rate — this is <strong>{cd_sec/neither:.1f}x higher</strong> 
    than customers with neither product ({neither:.1f}%).</p>
    <p><strong>Recommendation:</strong> Prioritize personal loan campaigns for existing CD account holders. 
    Consider bundling CD account promotions with personal loan offers for maximum conversion.</p>
</div>
""", unsafe_allow_html=True)

# Summary Table
st.markdown("---")
st.markdown('<p class="section-header">📋 Cross-Sell Summary Table</p>', unsafe_allow_html=True)

summary_data = []
for combo, group in df_f.groupby(['Securities_Account', 'CD_Account', 'CreditCard']):
    summary_data.append({
        'Securities': '✓' if combo[0] == 1 else '✗',
        'CD Account': '✓' if combo[1] == 1 else '✗',
        'Credit Card': '✓' if combo[2] == 1 else '✗',
        'Customers': len(group),
        'Accepters': int(group['Personal_Loan'].sum()),
        'Acceptance Rate': f"{group['Personal_Loan'].mean()*100:.1f}%",
        'Avg Income': f"${group['Income'].mean():.0f}K"
    })

summary_df = pd.DataFrame(summary_data).sort_values('Accepters', ascending=False)
st.dataframe(summary_df, use_container_width=True, hide_index=True)
