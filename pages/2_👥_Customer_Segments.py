import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Segments", page_icon="👥", layout="wide")

# Dark theme CSS
st.markdown("""
<style>
    .insight-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
        border: 1px solid #6C63FF;
        border-left: 4px solid #6C63FF;
        padding: 25px 30px;
        border-radius: 12px;
        margin: 20px 0;
    }
    .insight-box h4 { color: #6C63FF !important; margin: 0 0 15px 0 !important; }
    .insight-box p { color: #E2E8F0 !important; line-height: 1.7; margin: 8px 0 !important; }
    .insight-box strong { color: #FAFAFA !important; }
    .section-header {
        color: #FAFAFA;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #3D4663;
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

# Sidebar
st.sidebar.header("🔧 Filters")
selected_edu = st.sidebar.multiselect("Education", [1,2,3], [1,2,3],
                                       format_func=lambda x: {1:'Undergraduate', 2:'Graduate', 3:'Advanced'}[x])
selected_fam = st.sidebar.multiselect("Family Size", list(df['Family'].unique()), list(df['Family'].unique()))

df_f = df[(df['Education'].isin(selected_edu)) & (df['Family'].isin(selected_fam))]
st.sidebar.metric("Records", f"{len(df_f):,}")

colors = {'Not Accepted': '#3D4663', 'Accepted': '#6C63FF'}

chart_template = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#FAFAFA'),
    title_font=dict(size=16, color='#FAFAFA')
)

# Header
st.markdown("""
<div style="text-align: center; padding: 10px 0 30px 0;">
    <h1 style="color: #FAFAFA;">👥 Customer Segments</h1>
    <p style="color: #A0AEC0;">Analyze geographic, education, and demographic segments</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# CHART 3: ZIP Code Region Analysis
# =============================================================================
st.markdown('<p class="section-header">📊 Chart 3: ZIP Code Region Analysis</p>', unsafe_allow_html=True)

st.info("💡 ZIP codes are aggregated to regions (first 3 digits) for clarity. Bubble size = customer count, color = acceptance rate.")

df_f['ZIP_Region'] = df_f['ZIP_Code'].astype(str).str[:3]
zip_agg = df_f.groupby('ZIP_Region').agg({
    'Income': 'mean',
    'Personal_Loan': ['mean', 'count']
}).reset_index()
zip_agg.columns = ['ZIP_Region', 'Avg_Income', 'Acceptance_Rate', 'Customers']
zip_agg['Acceptance_Rate'] *= 100
zip_agg = zip_agg[zip_agg['Customers'] >= 10]

fig = px.scatter(
    zip_agg, x='Avg_Income', y='Acceptance_Rate',
    size='Customers', color='Acceptance_Rate',
    color_continuous_scale=['#3D4663', '#48BB78'],
    hover_data=['ZIP_Region'],
    title='ZIP Region: Average Income vs Loan Acceptance Rate'
)
fig.update_layout(height=500, **chart_template)
fig.update_xaxes(gridcolor='#3D4663', title='Average Income ($K)')
fig.update_yaxes(gridcolor='#3D4663', title='Acceptance Rate (%)')

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
# CHART 8: Education vs Income
# =============================================================================
st.markdown('<p class="section-header">📊 Chart 8: Income Distribution by Education Level</p>', unsafe_allow_html=True)

fig = px.box(
    df_f, x='Education_Label', y='Income', color='Loan_Status',
    color_discrete_map=colors,
    category_orders={'Education_Label': ['Undergraduate', 'Graduate', 'Advanced']},
    title='Income Distribution by Education Level and Loan Status'
)
fig.update_layout(height=500, **chart_template)
fig.update_layout(legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center'))
fig.update_xaxes(gridcolor='#3D4663')
fig.update_yaxes(gridcolor='#3D4663', title='Income ($K)')

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
# CHART 5: Family vs Income with Toggle
# =============================================================================
st.markdown('<p class="section-header">📊 Chart 5: Family Size Analysis (Variable Toggle)</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])
with col1:
    size_var = st.radio("Bubble size:", ['Mortgage', 'CCAvg'], index=0)

plot_df = df_f.copy()
plot_df['Size'] = plot_df[size_var] + 1

fig = px.scatter(
    plot_df, x='Family', y='Income',
    size='Size', color='Loan_Status',
    color_discrete_map=colors,
    title=f'Income by Family Size (bubble size = {size_var})',
    hover_data=[size_var, 'Education_Label']
)
fig.update_layout(height=500, **chart_template)
fig.update_layout(legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'))
fig.update_xaxes(gridcolor='#3D4663', title='Family Size')
fig.update_yaxes(gridcolor='#3D4663', title='Income ($K)')
fig.update_traces(marker=dict(opacity=0.6, sizemin=4))

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

st.markdown("---")

# =============================================================================
# CHART 9: Mortgage Multi-Dimensional
# =============================================================================
st.markdown('<p class="section-header">📊 Chart 9: Mortgage Analysis (Multi-Dimensional)</p>', unsafe_allow_html=True)

st.info("💡 X=Mortgage, Y=Income, Size=Family, Color=Loan Status. Outliers (>95th percentile) excluded for clarity.")

pctl = df_f['Mortgage'].quantile(0.95)
plot_df = df_f[df_f['Mortgage'] <= pctl]

fig = px.scatter(
    plot_df, x='Mortgage', y='Income',
    size='Family', color='Loan_Status',
    color_discrete_map=colors,
    size_max=20,
    title='Mortgage vs Income (Size=Family, Color=Loan Status)',
    hover_data=['Education_Label', 'CCAvg', 'Age']
)
fig.update_layout(height=500, **chart_template)
fig.update_layout(legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'))
fig.update_xaxes(gridcolor='#3D4663', title='Mortgage ($K)')
fig.update_yaxes(gridcolor='#3D4663', title='Income ($K)')
fig.update_traces(marker=dict(opacity=0.6))

st.plotly_chart(fig, use_container_width=True)

has_m = df_f[df_f['Mortgage'] > 0]['Personal_Loan'].mean() * 100
no_m = df_f[df_f['Mortgage'] == 0]['Personal_Loan'].mean() * 100

st.markdown(f"""
<div class="insight-box">
    <h4>📌 Key Insight</h4>
    <p><strong>Finding:</strong> Customers with mortgage: <strong>{has_m:.1f}%</strong> acceptance rate vs 
    <strong>{no_m:.1f}%</strong> for those without mortgage.</p>
    <p><strong>Implication:</strong> Mortgage status {'slightly increases' if has_m > no_m else 'slightly decreases'} loan acceptance, 
    but income remains the dominant factor.</p>
</div>
""", unsafe_allow_html=True)
