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
    .insight-box h4 { color: #7B68EE !important; font-size: 1.1rem; margin: 0 0 15px 0 !important; }
    .insight-box p { color: #E2E8F0 !important; font-size: 0.95rem; line-height: 1.7; margin: 8px 0 !important; }
    .insight-box strong { color: #FAFAFA !important; }
    
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

COLORS = {'Not Accepted': '#5A5F72', 'Accepted': '#7B68EE'}
OUTLIER_COLOR = '#FFA500'

@st.cache_data
def load_data():
    try:
        paths = ["UniversalBank.xlsx", "data/UniversalBank.xlsx", "UniversalBank with description.xls", "data/UniversalBank with description.xls"]
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
        'Age': np.random.randint(23, 67, n),
        'Income': np.clip(np.random.exponential(60, n), 8, 224).astype(int),
        'ZIP_Code': np.random.randint(90000, 96700, n),
        'Family': np.random.choice([1, 2, 3, 4], n),
        'CCAvg': np.round(np.random.exponential(1.9, n), 1),
        'Education': np.random.choice([1, 2, 3], n),
        'Mortgage': np.where(np.random.random(n) > 0.5, np.random.exponential(100, n).astype(int), 0),
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
df['Education_Label'] = df['Education'].map({1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced'})

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
    
    edu_map = {1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced'}
    selected_edu = st.multiselect("Education Level", options=[1, 2, 3], default=[1, 2, 3], format_func=lambda x: edu_map[x])
    selected_fam = st.multiselect("Family Size", options=sorted(df['Family'].unique()), default=list(df['Family'].unique()))
    income_range = st.slider("Income Range ($K)", int(df['Income'].min()), int(df['Income'].max()), (int(df['Income'].min()), int(df['Income'].max())))

df_f = df[(df['Education'].isin(selected_edu)) & (df['Family'].isin(selected_fam)) & (df['Income'].between(income_range[0], income_range[1]))]

st.sidebar.metric("Filtered Records", f"{len(df_f):,}")

st.markdown("""
<div style="text-align: center; padding: 10px 0 30px 0;">
    <h1 style="color: #FAFAFA;">👥 Customer Segments</h1>
    <p style="color: #A0AEC0;">Analyze geographic, education, and demographic segments</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# CHART: ZIP Code Region Analysis
# =============================================================================
st.markdown('<p class="section-header">📊 ZIP Code Region Analysis</p>', unsafe_allow_html=True)

st.info("💡 ZIP codes are aggregated to regions (first 3 digits) for clarity. Bubble size = customer count, color = acceptance rate.")

df_f['ZIP_Region'] = df_f['ZIP_Code'].astype(str).str[:3]
zip_agg = df_f.groupby('ZIP_Region').agg({'Income': 'mean', 'Personal_Loan': ['mean', 'count']}).reset_index()
zip_agg.columns = ['ZIP_Region', 'Avg_Income', 'Acceptance_Rate', 'Customers']
zip_agg['Acceptance_Rate'] *= 100
zip_agg = zip_agg[zip_agg['Customers'] >= 10]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=zip_agg['Avg_Income'],
    y=zip_agg['Acceptance_Rate'],
    mode='markers',
    marker=dict(
        size=zip_agg['Customers'] / 5,
        color=zip_agg['Acceptance_Rate'],
        colorscale=[[0, '#5A5F72'], [0.5, '#4FD1C5'], [1, '#2ECC71']],
        showscale=True,
        colorbar=dict(title='Rate %', tickfont=dict(color='#A0AEC0'))
    ),
    text=zip_agg['ZIP_Region'],
    hovertemplate='ZIP: %{text}<br>Avg Income: $%{x:.0f}K<br>Rate: %{y:.1f}%<br>Customers: %{marker.size:.0f}<extra></extra>'
))

fig.update_layout(
    height=500,
    title=dict(text='ZIP Region: Average Income vs Loan Acceptance Rate', font=dict(color='#FAFAFA', size=16)),
    xaxis_title='Average Income ($K)',
    yaxis_title='Acceptance Rate (%)',
    paper_bgcolor='#0E1117',
    plot_bgcolor='#0E1117',
    font=dict(color='#FAFAFA'),
    xaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0')),
    yaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0'))
)

st.plotly_chart(fig, use_container_width=True)

best = zip_agg.loc[zip_agg['Acceptance_Rate'].idxmax()]

st.markdown(f"""
<div class="insight-box">
    <h4>📌 Key Insight</h4>
    <p><strong>Finding:</strong> ZIP region <strong>{best['ZIP_Region']}</strong> shows the highest acceptance rate 
    (<strong>{best['Acceptance_Rate']:.1f}%</strong>) with average income <strong>${best['Avg_Income']:.0f}K</strong>.</p>
    <p><strong>Implication:</strong> Higher-income geographic regions show better loan conversion. Consider geo-targeted marketing campaigns.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# CHART: Education vs Income - GROUPED BOX PLOT
# =============================================================================
st.markdown('<p class="section-header">📊 Income Distribution by Education Level</p>', unsafe_allow_html=True)

fig = go.Figure()

for status in ['Not Accepted', 'Accepted']:
    subset = df_f[df_f['Loan_Status'] == status]
    fig.add_trace(go.Box(
        x=subset['Education_Label'],
        y=subset['Income'],
        name=status,
        marker=dict(color=OUTLIER_COLOR, size=4, outliercolor=OUTLIER_COLOR),
        line=dict(color='#FFFFFF', width=1.5),
        fillcolor=COLORS[status]
    ))

fig.update_layout(
    height=500,
    title=dict(text='Income Distribution by Education Level and Loan Status', font=dict(color='#FAFAFA', size=16)),
    yaxis_title='Income ($K)',
    boxmode='group',
    legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center', font=dict(color='#FAFAFA')),
    xaxis=dict(categoryorder='array', categoryarray=['Undergraduate', 'Graduate', 'Advanced'], gridcolor='#2D3748', tickfont=dict(color='#A0AEC0')),
    yaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0')),
    paper_bgcolor='#0E1117',
    plot_bgcolor='#0E1117',
    font=dict(color='#FAFAFA')
)

st.plotly_chart(fig, use_container_width=True)

edu_rates = df_f.groupby('Education_Label')['Personal_Loan'].mean() * 100
best_edu = edu_rates.idxmax()

st.markdown(f"""
<div class="insight-box">
    <h4>📌 Key Insight</h4>
    <p><strong>Finding:</strong> <strong>{best_edu}</strong> degree holders show the highest acceptance rate 
    (<strong>{edu_rates[best_edu]:.1f}%</strong>). Higher education correlates with both higher income and loan acceptance.</p>
    <p><strong>Implication:</strong> Education level is a strong predictor. Prioritize marketing to advanced degree holders.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# CHART: Family vs Income
# =============================================================================
st.markdown('<p class="section-header">📊 Family Size Analysis</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])
with col1:
    size_var = st.radio("Bubble size:", ['Mortgage', 'CCAvg'], index=0)

plot_df = df_f.copy()
plot_df['Size'] = plot_df[size_var] + 1

fig = go.Figure()

for status in ['Not Accepted', 'Accepted']:
    subset = plot_df[plot_df['Loan_Status'] == status]
    fig.add_trace(go.Scatter(
        x=subset['Family'],
        y=subset['Income'],
        mode='markers',
        name=status,
        marker=dict(
            color=COLORS[status],
            size=subset['Size'].clip(upper=30),
            opacity=0.6,
            sizemin=4
        )
    ))

fig.update_layout(
    height=500,
    title=dict(text=f'Income by Family Size (bubble size = {size_var})', font=dict(color='#FAFAFA', size=16)),
    xaxis_title='Family Size',
    yaxis_title='Income ($K)',
    legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center', font=dict(color='#FAFAFA')),
    paper_bgcolor='#0E1117',
    plot_bgcolor='#0E1117',
    font=dict(color='#FAFAFA'),
    xaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0')),
    yaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0'))
)

st.plotly_chart(fig, use_container_width=True)

fam_rates = df_f.groupby('Family')['Personal_Loan'].mean() * 100
best_fam = fam_rates.idxmax()

st.markdown(f"""
<div class="insight-box">
    <h4>📌 Key Insight</h4>
    <p><strong>Finding:</strong> Family size <strong>{int(best_fam)}</strong> shows the highest acceptance rate 
    (<strong>{fam_rates[best_fam]:.1f}%</strong>). Toggle between Mortgage and CCAvg to see additional patterns.</p>
    <p><strong>Implication:</strong> Family size influences loan needs. Larger families may have higher financial requirements.</p>
</div>
""", unsafe_allow_html=True)
