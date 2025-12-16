import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Universal Bank - Loan Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0E1117; }
    
    h1, h2, h3 { color: #FAFAFA !important; font-weight: 700 !important; }
    
    .kpi-row {
        display: grid;
        gap: 20px;
        margin-bottom: 20px;
    }
    .kpi-row-4 { grid-template-columns: repeat(4, 1fr); }
    .kpi-row-3 { grid-template-columns: repeat(3, 1fr); }
    
    .kpi-card {
        background: linear-gradient(135deg, #1E2130 0%, #2D3348 100%);
        border: 1px solid #3D4663;
        border-radius: 16px;
        padding: 30px 20px;
        text-align: center;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(139, 127, 255, 0.2);
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 15px;
        font-weight: 500;
        line-height: 1.4;
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #8B7FFF;
        line-height: 1;
    }
    
    @media (max-width: 1200px) {
        .kpi-row-4 { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 768px) {
        .kpi-row-4, .kpi-row-3 { grid-template-columns: 1fr; }
        .kpi-value { font-size: 2rem; }
    }
    
    .insight-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
        border: 1px solid #8B7FFF;
        border-left: 4px solid #8B7FFF;
        padding: 25px 30px;
        border-radius: 12px;
        margin: 25px 0;
    }
    .insight-box h4 { color: #8B7FFF !important; font-size: 1.1rem; margin: 0 0 15px 0 !important; }
    .insight-box p { color: #E2E8F0 !important; font-size: 0.95rem; line-height: 1.7; margin: 8px 0 !important; }
    .insight-box strong { color: #FAFAFA !important; }
    
    .section-header {
        color: #FAFAFA;
        font-size: 1.5rem;
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

# Color scheme for charts
COLORS = {
    'Not Accepted': '#636B7C',
    'Accepted': '#8B7FFF'
}
COLOR_SEQUENCE = ['#8B7FFF', '#F6AD55', '#48BB78', '#4FD1C5', '#F687B3', '#63B3ED', '#FC8181']

# Chart template
chart_template = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#FAFAFA', family='Inter'),
    title_font=dict(size=16, color='#FAFAFA', family='Inter'),
    legend=dict(font=dict(color='#FAFAFA')),
    colorway=COLOR_SEQUENCE
)

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
                if 'Experience' in df.columns:
                    df['Experience'] = df['Experience'].abs()
                return df
            except:
                continue
        return create_sample_data()
    except:
        return create_sample_data()

def create_sample_data():
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'ID': range(1, n + 1),
        'Age': np.random.randint(23, 67, n),
        'Experience': np.abs(np.random.randint(-3, 43, n)),
        'Income': np.clip(np.random.exponential(60, n), 8, 224).astype(int),
        'ZIP_Code': np.random.randint(90000, 96700, n),
        'Family': np.random.choice([1, 2, 3, 4], n, p=[0.25, 0.35, 0.25, 0.15]),
        'CCAvg': np.round(np.clip(np.random.exponential(1.9, n), 0, 10), 1),
        'Education': np.random.choice([1, 2, 3], n, p=[0.35, 0.40, 0.25]),
        'Mortgage': np.where(np.random.random(n) > 0.5, np.random.exponential(100, n).astype(int), 0),
        'Securities_Account': np.random.choice([0, 1], n, p=[0.90, 0.10]),
        'CD_Account': np.random.choice([0, 1], n, p=[0.94, 0.06]),
        'Online': np.random.choice([0, 1], n, p=[0.40, 0.60]),
        'CreditCard': np.random.choice([0, 1], n, p=[0.71, 0.29])
    })
    prob = (0.02 + 0.12*(df['Income']>100) + 0.08*(df['Income']>150) + 
            0.08*(df['Education']==3) + 0.25*(df['CD_Account']==1) + 0.04*(df['CCAvg']>3))
    df['Personal_Loan'] = (np.random.random(n) < prob).astype(int)
    return df

df = load_data()

# Sidebar
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
    
    edu_map = {1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced/Professional'}
    selected_edu = st.multiselect("Education Level", options=[1, 2, 3], default=[1, 2, 3],
                                  format_func=lambda x: edu_map[x])
    
    income_range = st.slider("Income Range ($K)", int(df['Income'].min()), int(df['Income'].max()),
                            (int(df['Income'].min()), int(df['Income'].max())))
    
    selected_family = st.multiselect("Family Size", options=sorted(df['Family'].unique()),
                                     default=list(df['Family'].unique()))
    
    st.markdown("---")
    with st.expander("📋 Dataset Info"):
        st.markdown("""
        **Columns:**
        - `Age` - Customer age
        - `Income` - Annual income ($K)
        - `CCAvg` - CC spend ($K/month)
        - `Education` - 1=UG, 2=Grad, 3=Adv
        - `Personal_Loan` - Target (0/1)
        """)

# Apply filters
df_filtered = df[
    (df['Education'].isin(selected_edu)) &
    (df['Income'].between(income_range[0], income_range[1])) &
    (df['Family'].isin(selected_family))
]

# Calculate KPIs
total_customers = len(df_filtered)
loan_accepters = df_filtered['Personal_Loan'].sum()
acceptance_rate = (loan_accepters / total_customers * 100) if total_customers > 0 else 0
accepters_df = df_filtered[df_filtered['Personal_Loan'] == 1]
non_accepters_df = df_filtered[df_filtered['Personal_Loan'] == 0]
avg_income_accepters = accepters_df['Income'].mean() if len(accepters_df) > 0 else 0
avg_income_non = non_accepters_df['Income'].mean() if len(non_accepters_df) > 0 else 0
avg_cc_accepters = accepters_df['CCAvg'].mean() if len(accepters_df) > 0 else 0
cd_penetration = (accepters_df['CD_Account'].sum() / len(accepters_df) * 100) if len(accepters_df) > 0 else 0
sec_penetration = (accepters_df['Securities_Account'].sum() / len(accepters_df) * 100) if len(accepters_df) > 0 else 0

# Header
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 2.5rem; margin-bottom: 10px;">🏦 Universal Bank - Personal Loan Campaign</h1>
    <p style="color: #A0AEC0; font-size: 1.1rem;">Analyze customer data to predict loan acceptance and identify opportunities</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# KPI CARDS
st.markdown('<p class="section-header">📊 Key Performance Indicators</p>', unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-row kpi-row-4">
    <div class="kpi-card">
        <div class="kpi-label">Total Customers</div>
        <div class="kpi-value">{total_customers:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Loan Accepters</div>
        <div class="kpi-value">{loan_accepters:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Acceptance Rate</div>
        <div class="kpi-value">{acceptance_rate:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Avg Income<br>(Accepters)</div>
        <div class="kpi-value">${avg_income_accepters:.0f}K</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-row kpi-row-3">
    <div class="kpi-card">
        <div class="kpi-label">Avg CC Spend<br>(Accepters)</div>
        <div class="kpi-value">${avg_cc_accepters:.2f}K</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">CD Account<br>Penetration</div>
        <div class="kpi-value">{cd_penetration:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Securities<br>Penetration</div>
        <div class="kpi-value">{sec_penetration:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Insight Box
income_diff = avg_income_accepters - avg_income_non
cd_overall = df_filtered['CD_Account'].mean() * 100
cd_lift = cd_penetration / cd_overall if cd_overall > 0 else 0

st.markdown(f"""
<div class="insight-box">
    <h4>📌 Key Insight</h4>
    <p><strong>Finding:</strong> Loan accepters have an average income of <strong>${avg_income_accepters:.0f}K</strong>, 
    which is <strong>${income_diff:.0f}K higher</strong> than non-accepters. CD account holders show 
    <strong>{cd_penetration:.1f}%</strong> penetration among accepters vs <strong>{cd_overall:.1f}%</strong> overall.</p>
    <p><strong>Implication:</strong> Target high-income customers with CD accounts. They show 
    <strong>{cd_lift:.1f}x higher</strong> representation among loan accepters.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Quick Charts
st.markdown('<p class="section-header">📈 Quick Overview Charts</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Acceptance by Education - Using gradient colors
    edu_stats = df_filtered.groupby('Education').agg({'Personal_Loan': ['sum', 'count']}).reset_index()
    edu_stats.columns = ['Education', 'Accepted', 'Total']
    edu_stats['Rate'] = edu_stats['Accepted'] / edu_stats['Total'] * 100
    edu_stats['Education'] = edu_stats['Education'].map(edu_map)
    
    fig = px.bar(
        edu_stats, x='Education', y='Rate',
        title='📚 Acceptance Rate by Education',
        color='Education',
        color_discrete_sequence=['#8B7FFF', '#F6AD55', '#48BB78'],
        text='Rate'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False, yaxis_title='Acceptance Rate (%)', xaxis_title='', **chart_template)
    fig.update_xaxes(gridcolor='#2D3748', zerolinecolor='#2D3748')
    fig.update_yaxes(gridcolor='#2D3748', zerolinecolor='#2D3748')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Acceptance by Family Size - Using gradient colors
    fam_stats = df_filtered.groupby('Family').agg({'Personal_Loan': ['sum', 'count']}).reset_index()
    fam_stats.columns = ['Family', 'Accepted', 'Total']
    fam_stats['Rate'] = fam_stats['Accepted'] / fam_stats['Total'] * 100
    
    fig = px.bar(
        fam_stats, x='Family', y='Rate',
        title='👨‍👩‍👧‍👦 Acceptance Rate by Family Size',
        color='Family',
        color_discrete_sequence=['#4FD1C5', '#63B3ED', '#F687B3', '#FC8181'],
        text='Rate'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False, yaxis_title='Acceptance Rate (%)', xaxis_title='Family Size', **chart_template)
    fig.update_xaxes(gridcolor='#2D3748', zerolinecolor='#2D3748')
    fig.update_yaxes(gridcolor='#2D3748', zerolinecolor='#2D3748')
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    # Pie Chart - Using the reference colors
    loan_dist = df_filtered['Personal_Loan'].value_counts().reset_index()
    loan_dist.columns = ['Status', 'Count']
    loan_dist['Status'] = loan_dist['Status'].map({0: 'Not Accepted', 1: 'Accepted'})
    
    fig = px.pie(
        loan_dist, values='Count', names='Status',
        title='🎯 Loan Acceptance Distribution',
        color='Status',
        color_discrete_map={'Not Accepted': '#636B7C', 'Accepted': '#8B7FFF'},
        hole=0.4
    )
    fig.update_traces(textposition='outside', textinfo='percent+label',
                     textfont=dict(color='#FAFAFA'))
    fig.update_layout(height=400, **chart_template)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # CD Account Impact - Orange gradient like reference
    cd_impact = df_filtered.groupby('CD_Account')['Personal_Loan'].mean() * 100
    cd_df = pd.DataFrame({
        'CD Account': ['No', 'Yes'],
        'Acceptance Rate': [cd_impact.get(0, 0), cd_impact.get(1, 0)]
    })
    
    fig = px.bar(
        cd_df, x='CD Account', y='Acceptance Rate',
        title='🏦 CD Account Impact on Loan Acceptance',
        color='Acceptance Rate',
        color_continuous_scale=[[0, '#636B7C'], [0.5, '#F6AD55'], [1, '#F6AD55']],
        text='Acceptance Rate'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False, yaxis_title='Acceptance Rate (%)', **chart_template)
    fig.update_xaxes(gridcolor='#2D3748', zerolinecolor='#2D3748')
    fig.update_yaxes(gridcolor='#2D3748', zerolinecolor='#2D3748')
    st.plotly_chart(fig, use_container_width=True)

# Data Preview
st.markdown("---")
st.markdown('<p class="section-header">📁 Data Preview</p>', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col2:
    st.download_button(
        label="📥 Download CSV",
        data=df_filtered.to_csv(index=False),
        file_name="loan_data.csv",
        mime="text/csv",
        use_container_width=True
    )

st.dataframe(df_filtered.head(100), use_container_width=True, height=300)
st.caption(f"Showing 100 of {len(df_filtered):,} filtered records")
