import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Universal Bank - Loan Analytics", page_icon="🏦", layout="wide")

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { color: #FAFAFA !important; font-weight: 700 !important; }
    
    .insight-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
        border: 1px solid #8B7FFF;
        border-left: 4px solid #8B7FFF;
        padding: 25px 30px;
        border-radius: 12px;
        margin: 20px 0;
    }
    .insight-box h4 { color: #8B7FFF !important; font-size: 1.1rem; margin: 0 0 15px 0 !important; }
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

# Colors - Updated beautiful scheme
COLORS = {'Not Accepted': '#636B7C', 'Accepted': '#8B7FFF'}
COLOR_SEQUENCE = ['#8B7FFF', '#F6AD55', '#48BB78', '#4FD1C5', '#F687B3', '#63B3ED', '#FC8181']

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
        'Experience': np.random.randint(0, 43, n),
        'Income': np.clip(np.random.exponential(60, n), 8, 224).astype(int),
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

    income_range = st.slider("Income Range ($K)", int(df['Income'].min()), int(df['Income'].max()),
                            (int(df['Income'].min()), int(df['Income'].max())))
    
    edu_map = {1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced'}
    selected_edu = st.multiselect("Education Level", options=[1, 2, 3], default=[1, 2, 3],
                                  format_func=lambda x: edu_map[x])
    
    selected_family = st.multiselect("Family Size", options=sorted(df['Family'].unique()),
                                     default=list(df['Family'].unique()))

# Apply filters
df_f = df[
    (df['Income'].between(income_range[0], income_range[1])) & 
    (df['Education'].isin(selected_edu)) &
    (df['Family'].isin(selected_family))
]

st.sidebar.metric("Filtered Records", f"{len(df_f):,}")

# Header
st.markdown("""
<div style="text-align: center; padding: 10px 0 30px 0;">
    <h1 style="color: #FAFAFA;">📈 Detailed Analysis</h1>
    <p style="color: #A0AEC0;">Explore distributions, correlations, and patterns in the data</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# CHART 1: Income & Age Histograms
# =============================================================================
st.markdown('<p class="section-header">📊 Chart 1: Income & Age Distributions by Loan Status</p>', unsafe_allow_html=True)

fig = make_subplots(rows=1, cols=2, subplot_titles=('Income Distribution', 'Age Distribution'))

for status, color in COLORS.items():
    subset = df_f[df_f['Loan_Status'] == status]
    fig.add_trace(go.Histogram(x=subset['Income'], name=status, marker_color=color, opacity=0.75), row=1, col=1)
    fig.add_trace(go.Histogram(x=subset['Age'], name=status, marker_color=color, opacity=0.75, showlegend=False), row=1, col=2)

fig.update_layout(
    barmode='overlay', height=450,
    legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'),
    **chart_template
)
fig.update_annotations(font=dict(color='#FAFAFA', size=14))
fig.update_xaxes(gridcolor='#2D3748', zerolinecolor='#2D3748', title_font=dict(color='#A0AEC0'))
fig.update_yaxes(gridcolor='#2D3748', zerolinecolor='#2D3748', title_font=dict(color='#A0AEC0'))
fig.update_xaxes(title_text='Income ($K)', row=1, col=1)
fig.update_xaxes(title_text='Age (Years)', row=1, col=2)
fig.update_yaxes(title_text='Count', row=1, col=1)

st.plotly_chart(fig, use_container_width=True)

inc_acc = df_f[df_f['Personal_Loan']==1]['Income'].mean()
inc_non = df_f[df_f['Personal_Loan']==0]['Income'].mean()

st.markdown(f"""
<div class="insight-box">
    <h4>📌 Key Insight</h4>
    <p><strong>Finding:</strong> Loan accepters have an average income of <strong>${inc_acc:.0f}K</strong> compared to 
    <strong>${inc_non:.0f}K</strong> for non-accepters — a difference of <strong>${inc_acc-inc_non:.0f}K</strong>.</p>
    <p><strong>Implication:</strong> Income is a primary driver. Target customers with income above $100K for higher conversion rates.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# CHART 2: Scatter Plot - CCAvg vs Income
# =============================================================================
st.markdown('<p class="section-header">📊 Chart 2: Credit Card Spending vs Income</p>', unsafe_allow_html=True)

fig = px.scatter(
    df_f, x='Income', y='CCAvg', color='Loan_Status',
    color_discrete_map=COLORS, opacity=0.6,
    title='Credit Card Average Spending vs Income by Loan Status'
)
fig.update_layout(height=500, **chart_template)
fig.update_layout(legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'))
fig.update_xaxes(gridcolor='#2D3748', zerolinecolor='#2D3748', title='Income ($K)')
fig.update_yaxes(gridcolor='#2D3748', zerolinecolor='#2D3748', title='CC Avg Spending ($K/month)')
fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='#1E2130')))

st.plotly_chart(fig, use_container_width=True)

cc_acc = df_f[df_f['Personal_Loan']==1]['CCAvg'].mean()
cc_non = df_f[df_f['Personal_Loan']==0]['CCAvg'].mean()

st.markdown(f"""
<div class="insight-box">
    <h4>📌 Key Insight</h4>
    <p><strong>Finding:</strong> Clear clustering pattern — loan accepters concentrate in the <strong>high-income, high-spending</strong> 
    quadrant (upper right). Average CC spend: <strong>${cc_acc:.2f}K/month</strong> for accepters vs <strong>${cc_non:.2f}K/month</strong> for non-accepters.</p>
    <p><strong>Implication:</strong> Customers with both high income AND high credit card spending are prime loan candidates.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# CHART 3: Correlation Heatmap
# =============================================================================
st.markdown('<p class="section-header">📊 Chart 3: Correlation Heatmap</p>', unsafe_allow_html=True)

num_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 
            'Mortgage', 'Personal_Loan', 'Securities_Account', 'CD_Account', 'Online', 'CreditCard']
num_cols = [c for c in num_cols if c in df_f.columns]

corr = df_f[num_cols].corr()

# Beautiful purple-orange diverging colorscale
fig = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale=[
        [0.0, '#FC8181'],   # Red for negative
        [0.25, '#F6AD55'],  # Orange
        [0.5, '#1E2130'],   # Dark center
        [0.75, '#63B3ED'],  # Blue
        [1.0, '#8B7FFF']    # Purple for positive
    ],
    zmid=0,
    text=np.round(corr.values, 2),
    texttemplate='%{text}',
    textfont={'size': 9, 'color': '#FAFAFA'}
))

fig.update_layout(height=600, title='Correlation Matrix - All Numeric Variables', **chart_template)

st.plotly_chart(fig, use_container_width=True)

loan_corr = corr['Personal_Loan'].drop('Personal_Loan').sort_values(key=abs, ascending=False).head(3)

st.markdown(f"""
<div class="insight-box">
    <h4>📌 Key Insight</h4>
    <p><strong>Finding:</strong> Top 3 predictors of Personal Loan acceptance:</p>
    <p>1. <strong>{loan_corr.index[0]}</strong> (correlation: {loan_corr.values[0]:.3f})</p>
    <p>2. <strong>{loan_corr.index[1]}</strong> (correlation: {loan_corr.values[1]:.3f})</p>
    <p>3. <strong>{loan_corr.index[2]}</strong> (correlation: {loan_corr.values[2]:.3f})</p>
    <p><strong>Implication:</strong> Focus predictive models and marketing on these key variables.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# CHART 4: Box Plots - Beautiful colors
# =============================================================================
st.markdown('<p class="section-header">📊 Chart 4: Distribution Comparison (Box Plots)</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    fig = px.box(df_f, x='Loan_Status', y='Income', color='Loan_Status',
                 color_discrete_map=COLORS, title='Income Distribution')
    fig.update_layout(height=400, showlegend=False, **chart_template)
    fig.update_xaxes(gridcolor='#2D3748')
    fig.update_yaxes(gridcolor='#2D3748', title='Income ($K)')
    fig.update_traces(marker=dict(outliercolor='#F6AD55', size=4),
                     line=dict(color='#FAFAFA', width=1))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(df_f, x='Loan_Status', y='CCAvg', color='Loan_Status',
                 color_discrete_map=COLORS, title='CC Average Distribution')
    fig.update_layout(height=400, showlegend=False, **chart_template)
    fig.update_xaxes(gridcolor='#2D3748')
    fig.update_yaxes(gridcolor='#2D3748', title='CC Avg ($K/month)')
    fig.update_traces(marker=dict(outliercolor='#F6AD55', size=4),
                     line=dict(color='#FAFAFA', width=1))
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = px.box(df_f, x='Loan_Status', y='Mortgage', color='Loan_Status',
                 color_discrete_map=COLORS, title='Mortgage Distribution')
    fig.update_layout(height=400, showlegend=False, **chart_template)
    fig.update_xaxes(gridcolor='#2D3748')
    fig.update_yaxes(gridcolor='#2D3748', title='Mortgage ($K)')
    fig.update_traces(marker=dict(outliercolor='#F6AD55', size=4),
                     line=dict(color='#FAFAFA', width=1))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="insight-box">
    <h4>📌 Key Insight</h4>
    <p><strong>Finding:</strong> Box plots reveal significant separation in <strong>Income</strong> and <strong>CCAvg</strong> 
    distributions between accepters and non-accepters. Mortgage shows similar distribution for both groups.</p>
    <p><strong>Implication:</strong> Income and CC spending are strong differentiators; mortgage status is not predictive of loan acceptance.</p>
</div>
""", unsafe_allow_html=True)
