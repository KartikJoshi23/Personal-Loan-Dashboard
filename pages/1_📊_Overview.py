import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin: 10px 0;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .kpi-label {
        font-size: 0.95rem;
        opacity: 0.9;
    }
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
                if 'Experience' in df.columns:
                    df['Experience'] = df['Experience'].abs()
                return df
            except:
                continue
    except:
        pass
    
    # Sample data fallback
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'Age': np.random.randint(23, 67, n),
        'Experience': np.random.randint(0, 43, n),
        'Income': np.clip(np.random.exponential(60, n), 8, 224).astype(int),
        'ZIP_Code': np.random.randint(90000, 96700, n),
        'Family': np.random.choice([1, 2, 3, 4], n),
        'CCAvg': np.round(np.random.exponential(1.9, n), 1),
        'Education': np.random.choice([1, 2, 3], n),
        'Mortgage': np.where(np.random.random(n) > 0.5, np.random.exponential(100, n).astype(int), 0),
        'Securities_Account': np.random.choice([0, 1], n, p=[0.90, 0.10]),
        'CD_Account': np.random.choice([0, 1], n, p=[0.94, 0.06]),
        'Online': np.random.choice([0, 1], n, p=[0.40, 0.60]),
        'CreditCard': np.random.choice([0, 1], n, p=[0.71, 0.29])
    })
    prob = 0.02 + 0.15*(df['Income']>100) + 0.08*(df['Education']==3) + 0.25*(df['CD_Account']==1)
    df['Personal_Loan'] = (np.random.random(n) < prob).astype(int)
    return df

df = load_data()

st.title("📊 Overview - Key Performance Indicators")
st.markdown("---")

# Sidebar Filters
st.sidebar.header("🔧 Filters")

edu_map = {1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced/Professional'}
selected_edu = st.sidebar.multiselect("Education Level", options=[1, 2, 3], default=[1, 2, 3],
                                       format_func=lambda x: edu_map[x])

selected_family = st.sidebar.multiselect("Family Size", options=sorted(df['Family'].unique()),
                                         default=list(df['Family'].unique()))

income_range = st.sidebar.slider("Income Range ($K)", 
                                  int(df['Income'].min()), int(df['Income'].max()),
                                  (int(df['Income'].min()), int(df['Income'].max())))

selected_online = st.sidebar.multiselect("Online Banking", options=[0, 1], default=[0, 1],
                                          format_func=lambda x: {0: 'No', 1: 'Yes'}[x])

# Apply filters
df_f = df[(df['Education'].isin(selected_edu)) & 
          (df['Family'].isin(selected_family)) &
          (df['Income'].between(income_range[0], income_range[1])) &
          (df['Online'].isin(selected_online))]

st.sidebar.markdown(f"**Filtered:** {len(df_f):,} / {len(df):,}")

# Calculate KPIs
total = len(df_f)
accepters = df_f[df_f['Personal_Loan'] == 1]
n_acc = len(accepters)
acc_rate = (n_acc / total * 100) if total > 0 else 0
avg_inc = accepters['Income'].mean() if n_acc > 0 else 0
avg_cc = accepters['CCAvg'].mean() if n_acc > 0 else 0
cd_pen = (accepters['CD_Account'].sum() / n_acc * 100) if n_acc > 0 else 0
sec_pen = (accepters['Securities_Account'].sum() / n_acc * 100) if n_acc > 0 else 0

# Display 7 KPIs
st.markdown("### 📈 7 Key Performance Indicators")

# Row 1: 4 KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">👥 Total Customers</div>
        <div class="kpi-value">{total:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">✅ Loan Accepters</div>
        <div class="kpi-value">{n_acc:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">📈 Acceptance Rate</div>
        <div class="kpi-value">{acc_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">💰 Avg Income (Accepters)</div>
        <div class="kpi-value">${avg_inc:.0f}K</div>
    </div>
    """, unsafe_allow_html=True)

# Row 2: 3 KPIs
col5, col6, col7 = st.columns(3)

with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">💳 Avg CC Spend (Accepters)</div>
        <div class="kpi-value">${avg_cc:.2f}K</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">🏦 CD Penetration</div>
        <div class="kpi-value">{cd_pen:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col7:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">📊 Securities Penetration</div>
        <div class="kpi-value">{sec_pen:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# KPI Definitions
with st.expander("📋 KPI Definitions & Formulas"):
    st.markdown("""
    | KPI | Formula | Interpretation |
    |-----|---------|----------------|
    | **Total Customers** | `COUNT(*)` | Total customers in filtered dataset |
    | **Loan Accepters** | `SUM(Personal_Loan)` | Customers who accepted the loan |
    | **Acceptance Rate** | `SUM(Personal_Loan) / COUNT(*) × 100` | Campaign conversion rate |
    | **Avg Income (Accepters)** | `AVG(Income) WHERE Personal_Loan=1` | Income profile of accepters |
    | **Avg CC Spend** | `AVG(CCAvg) WHERE Personal_Loan=1` | Spending of accepters |
    | **CD Penetration** | `SUM(CD_Account) / COUNT(Accepters) × 100` | CD account ownership among accepters |
    | **Securities Penetration** | `SUM(Securities_Account) / COUNT(Accepters) × 100` | Securities ownership |
    """)

# Insight Box
non_acc = df_f[df_f['Personal_Loan'] == 0]
inc_diff = avg_inc - non_acc['Income'].mean() if len(non_acc) > 0 else 0

st.markdown(f"""
<div class="insight-box">
    <h4 style="color: #667eea; margin-top: 0;">📌 Key Insight</h4>
    <p><strong>Finding:</strong> Loan accepters have an average income of ${avg_inc:.0f}K, 
    which is ${inc_diff:.0f}K higher than non-accepters. CD account holders show 
    {cd_pen:.1f}% penetration among accepters vs {df_f['CD_Account'].mean()*100:.1f}% overall.</p>
    <p><strong>Implication:</strong> Target high-income customers with CD accounts. They show 
    {cd_pen/(df_f['CD_Account'].mean()*100):.1f}x higher representation among loan accepters.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Charts
st.markdown("### 📊 Overview Charts")

col1, col2 = st.columns(2)

with col1:
    # Acceptance by Education
    edu_stats = df_f.groupby('Education').agg({'Personal_Loan': ['sum', 'count']}).reset_index()
    edu_stats.columns = ['Education', 'Accepted', 'Total']
    edu_stats['Rate'] = edu_stats['Accepted'] / edu_stats['Total'] * 100
    edu_stats['Education'] = edu_stats['Education'].map(edu_map)
    
    fig = px.bar(edu_stats, x='Education', y='Rate', color='Rate',
                 color_continuous_scale='Blues', text='Rate',
                 title='📚 Acceptance Rate by Education')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False, yaxis_title='Acceptance Rate (%)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Acceptance by Family Size
    fam_stats = df_f.groupby('Family').agg({'Personal_Loan': ['sum', 'count']}).reset_index()
    fam_stats.columns = ['Family', 'Accepted', 'Total']
    fam_stats['Rate'] = fam_stats['Accepted'] / fam_stats['Total'] * 100
    
    fig = px.bar(fam_stats, x='Family', y='Rate', color='Rate',
                 color_continuous_scale='Greens', text='Rate',
                 title='👨‍👩‍👧‍👦 Acceptance Rate by Family Size')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False, yaxis_title='Acceptance Rate (%)')
    st.plotly_chart(fig, use_container_width=True)

# Data Table
st.markdown("---")
st.markdown("### 📁 Filtered Data")

search = st.text_input("🔍 Search", "")
if search:
    mask = df_f.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
    show_df = df_f[mask]
else:
    show_df = df_f

col1, col2 = st.columns([4, 1])
with col2:
    st.download_button("📥 Download", show_df.to_csv(index=False), "filtered_data.csv", "text/csv")

st.dataframe(show_df, use_container_width=True, height=400)
