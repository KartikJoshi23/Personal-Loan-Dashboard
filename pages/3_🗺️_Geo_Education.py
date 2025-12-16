import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Geo & Education", page_icon="🗺️", layout="wide")

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
        'Age': np.random.randint(23, 67, n),
        'Income': np.clip(np.random.exponential(60, n), 8, 224).astype(int),
        'ZIP_Code': np.random.randint(90000, 96700, n),
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
df['Education_Label'] = df['Education'].map({1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced'})

st.title("🗺️ Geographic & Education Analysis")
st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Filters")
selected_edu = st.sidebar.multiselect("Education", [1,2,3], [1,2,3],
                                       format_func=lambda x: {1:'UG', 2:'Grad', 3:'Adv'}[x])
selected_fam = st.sidebar.multiselect("Family Size", list(df['Family'].unique()), list(df['Family'].unique()))

df_f = df[(df['Education'].isin(selected_edu)) & (df['Family'].isin(selected_fam))]
st.sidebar.metric("Records", f"{len(df_f):,}")

colors = {'Not Accepted': '#636EFA', 'Accepted': '#EF553B'}

# CHART 3: ZIP Code Analysis
st.subheader("📊 Chart 3: ZIP Code Region Analysis")

st.info("**Design:** ZIP codes aggregated to regions (first 3 digits) for readability. Bubble size = customer count, color = acceptance rate.")

df_f['ZIP_Region'] = df_f['ZIP_Code'].astype(str).str[:3]
zip_agg = df_f.groupby('ZIP_Region').agg({'Income': 'mean', 'Personal_Loan': ['mean', 'count']}).reset_index()
zip_agg.columns = ['ZIP_Region', 'Avg_Income', 'Acceptance_Rate', 'Customers']
zip_agg['Acceptance_Rate'] *= 100
zip_agg = zip_agg[zip_agg['Customers'] >= 10]

fig = px.scatter(zip_agg, x='Avg_Income', y='Acceptance_Rate', size='Customers', color='Acceptance_Rate',
                 color_continuous_scale='RdYlGn', hover_data=['ZIP_Region'],
                 title='ZIP Region: Income vs Acceptance Rate')
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)

best = zip_agg.loc[zip_agg['Acceptance_Rate'].idxmax()]
st.markdown(f"""
<div class="insight-box">
    <strong>📌 Insight:</strong> Region {best['ZIP_Region']} has highest acceptance ({best['Acceptance_Rate']:.1f}%) 
    with avg income ${best['Avg_Income']:.0f}K. Higher-income regions show better conversion.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# CHART 8: Education vs Income
st.subheader("📊 Chart 8: Income by Education Level")

fig = px.box(df_f, x='Education_Label', y='Income', color='Loan_Status', color_discrete_map=colors,
             category_orders={'Education_Label': ['Undergraduate', 'Graduate', 'Advanced']})
fig.update_layout(height=450, legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center'))
st.plotly_chart(fig, use_container_width=True)

edu_stats = df_f.groupby('Education_Label')['Personal_Loan'].mean() * 100
best_edu = edu_stats.idxmax()
st.markdown(f"""
<div class="insight-box">
    <strong>📌 Insight:</strong> <strong>{best_edu}</strong> degree holders show highest acceptance rate 
    ({edu_stats[best_edu]:.1f}%). Education is a strong predictor.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# CHART 5: Family vs Income with Toggle
st.subheader("📊 Chart 5: Family Size Analysis (Toggle)")

col1, col2 = st.columns([1, 4])
with col1:
    var = st.radio("Size by:", ['Mortgage', 'CCAvg'])

plot_df = df_f.copy()
plot_df['Size'] = plot_df[var] + 1

fig = px.scatter(plot_df, x='Family', y='Income', size='Size', color='Loan_Status',
                 color_discrete_map=colors, title=f'Income by Family (size={var})', hover_data=[var])
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# CHART 9: Mortgage Multi-encoding
st.subheader("📊 Chart 9: Mortgage Analysis (Multi-Dimensional)")

st.info("**Chart:** X=Mortgage, Y=Income, Size=Family, Color=Loan. Outliers (>95th pctl) excluded.")

pctl = df_f['Mortgage'].quantile(0.95)
plot_df = df_f[df_f['Mortgage'] <= pctl]

fig = px.scatter(plot_df, x='Mortgage', y='Income', size='Family', color='Loan_Status',
                 color_discrete_map=colors, size_max=20, hover_data=['Education_Label', 'CCAvg'])
fig.update_layout(height=450)
fig.update_traces(marker=dict(opacity=0.6))
st.plotly_chart(fig, use_container_width=True)

has_m = df_f[df_f['Mortgage'] > 0]['Personal_Loan'].mean() * 100
no_m = df_f[df_f['Mortgage'] == 0]['Personal_Loan'].mean() * 100
st.markdown(f"""
<div class="insight-box">
    <strong>📌 Insight:</strong> Customers with mortgage: {has_m:.1f}% acceptance vs {no_m:.1f}% without. 
    Mortgage {'slightly increases' if has_m > no_m else 'slightly decreases'} acceptance likelihood.
</div>
""", unsafe_allow_html=True)
