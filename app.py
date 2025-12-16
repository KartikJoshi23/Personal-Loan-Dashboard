import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Universal Bank - Loan Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1E3A5F;
        font-weight: 700 !important;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    div[data-testid="metric-container"] label {
        color: rgba(255,255,255,0.8) !important;
    }
    
    div[data-testid="metric-container"] div {
        color: white !important;
    }
    
    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 5px solid #667eea;
        padding: 25px;
        border-radius: 0 15px 15px 0;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .insight-box h4 {
        color: #667eea;
        margin: 0 0 15px 0;
        font-size: 1.2rem;
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 25px;
        border-radius: 0 15px 15px 0;
        margin: 20px 0;
    }
    
    /* Cards */
    .custom-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 10px 0;
        border: 1px solid #eee;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A5F 0%, #2D5F8B 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 25px;
        font-weight: 600;
        transition: transform 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load the Universal Bank dataset."""
    try:
        # Try different file paths
        paths = [
            "UniversalBank.xlsx",
            "data/UniversalBank.xlsx", 
            "UniversalBank with description.xls",
            "data/UniversalBank with description.xls"
        ]
        
        for path in paths:
            try:
                df = pd.read_excel(path, sheet_name="Data")
                df.columns = df.columns.str.strip().str.replace(' ', '_')
                if 'Experience' in df.columns:
                    df['Experience'] = df['Experience'].abs()
                return df
            except:
                continue
        
        # If no file found, create sample data
        return create_sample_data()
    except Exception as e:
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration."""
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
        'Mortgage': np.where(np.random.random(n) > 0.5, 
                            np.random.exponential(100, n).astype(int), 0),
        'Securities_Account': np.random.choice([0, 1], n, p=[0.90, 0.10]),
        'CD_Account': np.random.choice([0, 1], n, p=[0.94, 0.06]),
        'Online': np.random.choice([0, 1], n, p=[0.40, 0.60]),
        'CreditCard': np.random.choice([0, 1], n, p=[0.71, 0.29])
    })
    
    # Create realistic Personal Loan acceptance
    prob = (0.02 + 
            0.12 * (df['Income'] > 100).astype(int) +
            0.08 * (df['Income'] > 150).astype(int) +
            0.08 * (df['Education'] == 3).astype(int) +
            0.25 * (df['CD_Account'] == 1).astype(int) +
            0.04 * (df['CCAvg'] > 3).astype(int) +
            0.02 * (df['Family'] >= 3).astype(int))
    
    df['Personal_Loan'] = (np.random.random(n) < prob).astype(int)
    
    return df

# Store in session state
if 'df' not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: white; font-size: 1.8rem;">🏦 Universal Bank</h1>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Personal Loan Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Data info
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0;">
        <p style="color: white; margin: 5px 0;">📊 <strong>Total Records:</strong> {len(df):,}</p>
        <p style="color: white; margin: 5px 0;">✅ <strong>Loan Accepters:</strong> {df['Personal_Loan'].sum():,}</p>
        <p style="color: white; margin: 5px 0;">📈 <strong>Acceptance Rate:</strong> {df['Personal_Loan'].mean()*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Column info expander
    with st.expander("📋 Dataset Columns", expanded=False):
        st.markdown("""
        <div style="font-size: 0.85rem;">
        
        **Demographics:**
        - `Age` - Customer age
        - `Experience` - Work experience (years)
        - `Income` - Annual income ($K)
        - `Family` - Family size
        - `Education` - 1=UG, 2=Grad, 3=Adv
        
        **Banking:**
        - `CCAvg` - Avg CC spend ($K/month)
        - `Mortgage` - Mortgage value ($K)
        - `Securities_Account` - Has securities
        - `CD_Account` - Has CD
        - `Online` - Uses online banking
        - `CreditCard` - Has credit card
        
        **Target:**
        - `Personal_Loan` - Accepted loan (1=Yes)
        </div>
        """, unsafe_allow_html=True)

# Main content
st.markdown("""
<h1 style="text-align: center; margin-bottom: 10px;">
    🏦 Universal Bank - Personal Loan Campaign Analysis
</h1>
<p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 30px;">
    Analyze customer data to predict loan acceptance and identify cross-sell opportunities
</p>
""", unsafe_allow_html=True)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
loan_accepters = df['Personal_Loan'].sum()
acceptance_rate = (loan_accepters / total_customers) * 100
avg_income_accepters = df[df['Personal_Loan'] == 1]['Income'].mean()

with col1:
    st.metric("Total Customers", f"{total_customers:,}")
with col2:
    st.metric("Loan Accepters", f"{loan_accepters:,}")
with col3:
    st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
with col4:
    st.metric("Avg Income (Accepters)", f"${avg_income_accepters:.0f}K")

# Key Insight Box
st.markdown(f"""
<div class="insight-box">
    <h4>📌 Key Insight</h4>
    <p><strong>Finding:</strong> The dataset contains {total_customers:,} customers with a 
    {acceptance_rate:.1f}% loan acceptance rate. Accepters have an average income of 
    ${avg_income_accepters:.0f}K, which is ${avg_income_accepters - df[df['Personal_Loan']==0]['Income'].mean():.0f}K 
    higher than non-accepters.</p>
    <p><strong>Implication:</strong> Focus marketing efforts on high-income customers, particularly 
    those with existing CD accounts, to maximize conversion rates.</p>
</div>
""", unsafe_allow_html=True)

# Quick Stats Cards
st.markdown("### 📊 Quick Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="custom-card">
        <h4 style="color: #667eea;">💰 Income Insights</h4>
        <hr style="margin: 10px 0;">
    """, unsafe_allow_html=True)
    
    st.write(f"**Mean Income:** ${df['Income'].mean():.1f}K")
    st.write(f"**Median Income:** ${df['Income'].median():.1f}K")
    st.write(f"**Max Income:** ${df['Income'].max():.0f}K")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="custom-card">
        <h4 style="color: #667eea;">👨‍👩‍👧‍👦 Demographics</h4>
        <hr style="margin: 10px 0;">
    """, unsafe_allow_html=True)
    
    st.write(f"**Avg Age:** {df['Age'].mean():.0f} years")
    st.write(f"**Avg Family Size:** {df['Family'].mean():.1f}")
    st.write(f"**Avg Experience:** {df['Experience'].mean():.0f} years")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="custom-card">
        <h4 style="color: #667eea;">🏦 Banking Products</h4>
        <hr style="margin: 10px 0;">
    """, unsafe_allow_html=True)
    
    st.write(f"**Online Banking:** {df['Online'].mean()*100:.0f}%")
    st.write(f"**Credit Card:** {df['CreditCard'].mean()*100:.0f}%")
    st.write(f"**CD Account:** {df['CD_Account'].mean()*100:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

# Navigation guide
st.markdown("---")
st.markdown("### 🧭 Dashboard Navigation")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="custom-card" style="text-align: center;">
        <h3>📊</h3>
        <p><strong>Overview</strong></p>
        <p style="font-size: 0.85rem; color: #666;">7 KPIs & Summary Stats</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="custom-card" style="text-align: center;">
        <h3>🔍</h3>
        <p><strong>EDA</strong></p>
        <p style="font-size: 0.85rem; color: #666;">Distributions & Correlations</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="custom-card" style="text-align: center;">
        <h3>🗺️</h3>
        <p><strong>Geo & Education</strong></p>
        <p style="font-size: 0.85rem; color: #666;">Regional & Education Analysis</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="custom-card" style="text-align: center;">
        <h3>🔗</h3>
        <p><strong>Cross-Sell</strong></p>
        <p style="font-size: 0.85rem; color: #666;">Product Opportunities</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="custom-card" style="text-align: center;">
        <h3>🤖</h3>
        <p><strong>Predictor</strong></p>
        <p style="font-size: 0.85rem; color: #666;">ML Predictions</p>
    </div>
    """, unsafe_allow_html=True)

st.info("👈 **Navigate using the sidebar** to explore different sections of the dashboard")

# Data preview
st.markdown("---")
st.markdown("### 📁 Data Preview")

col1, col2 = st.columns([4, 1])
with col2:
    st.download_button(
        label="📥 Download CSV",
        data=df.to_csv(index=False),
        file_name="universal_bank_data.csv",
        mime="text/csv"
    )

st.dataframe(df.head(50), use_container_width=True, height=300)
st.caption(f"Showing 50 of {len(df):,} rows")
