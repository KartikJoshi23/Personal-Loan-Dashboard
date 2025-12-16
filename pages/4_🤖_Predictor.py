import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve

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
    
    .prediction-card {
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .accept-card {
        background: linear-gradient(135deg, #1a3d1a 0%, #2d5a2d 100%);
        border: 2px solid #2ECC71;
    }
    .reject-card {
        background: linear-gradient(135deg, #3d1a1a 0%, #5a2d2d 100%);
        border: 2px solid #E74C3C;
    }
    .pred-title { font-size: 1.5rem; font-weight: 600; margin-bottom: 15px; }
    .pred-prob { font-size: 3.5rem; font-weight: 700; margin: 20px 0; }
    .accept-card .pred-title, .accept-card .pred-prob { color: #2ECC71; }
    .reject-card .pred-title, .reject-card .pred-prob { color: #E74C3C; }
    
    .section-header {
        color: #FAFAFA;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #3D4663;
    }
    
    .metric-card {
        background: #1E2130;
        border: 1px solid #3D4663;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7B68EE; }
    .metric-label { color: #A0AEC0; font-size: 0.9rem; margin-top: 5px; }
    
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

features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities_Account', 'CD_Account', 'Online', 'CreditCard']
features = [f for f in features if f in df.columns]

X = df[features]
y = df['Personal_Loan']

@st.cache_resource
def train_models():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_s, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    return {'lr': lr, 'rf': rf, 'scaler': scaler, 'X_test': X_test_s, 'y_test': y_test, 'features': features}

models = train_models()

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 3rem;">🏦</div>
        <h2 style="color: #FAFAFA; margin: 10px 0;">Universal Bank</h2>
        <p style="color: #A0AEC0; font-size: 0.9rem;">Personal Loan Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    model_choice = st.radio("Select Model", ['Logistic Regression', 'Random Forest'])

model = models['lr'] if model_choice == 'Logistic Regression' else models['rf']

st.markdown("""
<div style="text-align: center; padding: 10px 0 30px 0;">
    <h1 style="color: #FAFAFA;">🤖 Personal Loan Predictor</h1>
    <p style="color: #A0AEC0;">Machine learning model to predict customer loan acceptance</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<p class="section-header">📊 Model Performance: {model_choice}</p>', unsafe_allow_html=True)

y_pred = model.predict(models['X_test'])
y_prob = model.predict_proba(models['X_test'])[:, 1]

acc = accuracy_score(models['y_test'], y_pred)
prec = precision_score(models['y_test'], y_pred)
rec = recall_score(models['y_test'], y_pred)
auc = roc_auc_score(models['y_test'], y_prob)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{acc:.1%}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{prec:.1%}</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{rec:.1%}</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{auc:.3f}</div><div class="metric-label">ROC-AUC</div></div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="insight-box">
    <h4>📌 Model Insight</h4>
    <p><strong>Performance:</strong> {model_choice} achieves <strong>{auc:.1%} ROC-AUC</strong> 
    ({'excellent' if auc > 0.9 else 'good'} discrimination ability).</p>
    <p><strong>Interpretation:</strong> Precision of {prec:.1%} means {prec:.1%} of predicted accepters actually accept. 
    Recall of {rec:.1%} means we identify {rec:.1%} of all actual accepters.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(models['y_test'], y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=['Predicted: No', 'Predicted: Yes'], y=['Actual: No', 'Actual: Yes'],
        colorscale=[[0, '#1E2130'], [0.5, '#4FD1C5'], [1, '#7B68EE']],
        text=cm, texttemplate='%{text}', textfont={'size': 18, 'color': '#FAFAFA'}
    ))
    fig.update_layout(
        height=350,
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='#FAFAFA'),
        xaxis=dict(tickfont=dict(color='#A0AEC0')),
        yaxis=dict(tickfont=dict(color='#A0AEC0'))
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**ROC Curve**")
    fpr, tpr, _ = roc_curve(models['y_test'], y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={auc:.3f})', line=dict(color='#7B68EE', width=3), fill='tozeroy', fillcolor='rgba(123, 104, 238, 0.2)'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', line=dict(dash='dash', color='#5A5F72')))
    fig.update_layout(
        height=350,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(font=dict(color='#FAFAFA')),
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='#FAFAFA'),
        xaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0')),
        yaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0'))
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown('<p class="section-header">🔍 Feature Importance</p>', unsafe_allow_html=True)

if model_choice == 'Logistic Regression':
    coef = model.coef_[0]
    imp_df = pd.DataFrame({'Feature': models['features'], 'Coefficient': coef}).sort_values('Coefficient')
    colors = ['#2ECC71' if c > 0 else '#E74C3C' for c in imp_df['Coefficient']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=imp_df['Coefficient'], y=imp_df['Feature'], orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in imp_df['Coefficient']],
        textposition='outside', textfont=dict(color='#FAFAFA')
    ))
    fig.update_layout(
        height=450,
        title=dict(text='Logistic Regression Coefficients (Standardized)', font=dict(color='#FAFAFA', size=16)),
        paper_bgcolor='#0E1117', plot_bgcolor='#0E1117', font=dict(color='#FAFAFA'),
        xaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0')),
        yaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0'))
    )
else:
    imp = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': models['features'], 'Importance': imp}).sort_values('Importance')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
        marker=dict(color=imp_df['Importance'], colorscale=[[0, '#5A5F72'], [0.5, '#4FD1C5'], [1, '#2ECC71']]),
        text=[f"{v:.3f}" for v in imp_df['Importance']],
        textposition='outside', textfont=dict(color='#FAFAFA')
    ))
    fig.update_layout(
        height=450,
        title=dict(text='Random Forest Feature Importance', font=dict(color='#FAFAFA', size=16)),
        paper_bgcolor='#0E1117', plot_bgcolor='#0E1117', font=dict(color='#FAFAFA'),
        xaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0')),
        yaxis=dict(gridcolor='#2D3748', tickfont=dict(color='#A0AEC0'))
    )

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown('<p class="section-header">🎯 Make a Prediction</p>', unsafe_allow_html=True)

st.markdown("Enter customer details to predict loan acceptance probability:")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    experience = st.number_input("Experience (years)", 0, 50, 10)
    income = st.number_input("Income ($K/year)", 0, 500, 60)
    family = st.selectbox("Family Size", [1, 2, 3, 4], index=1)

with col2:
    ccavg = st.number_input("CC Avg Spending ($K/month)", 0.0, 15.0, 1.5, 0.1)
    education = st.selectbox("Education", [1, 2, 3], index=1, format_func=lambda x: {1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced'}[x])
    mortgage = st.number_input("Mortgage ($K)", 0, 1000, 0)

with col3:
    securities = st.selectbox("Securities Account", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    cd_account = st.selectbox("CD Account", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    online = st.selectbox("Online Banking", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    creditcard = st.selectbox("Credit Card", [0, 1], format_func=lambda x: 'Yes' if x else 'No')

if st.button("🔮 Predict Loan Acceptance", type="primary", use_container_width=True):
    input_data = np.array([[age, experience, income, family, ccavg, education, mortgage, securities, cd_account, online, creditcard]])
    input_scaled = models['scaler'].transform(input_data)
    
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pred == 1:
            st.markdown(f"""
            <div class="prediction-card accept-card">
                <div class="pred-title">✅ LIKELY TO ACCEPT</div>
                <div class="pred-prob">{prob:.1%}</div>
                <p style="color: #A0AEC0;">Probability of Acceptance</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card reject-card">
                <div class="pred-title">❌ UNLIKELY TO ACCEPT</div>
                <div class="pred-prob">{prob:.1%}</div>
                <p style="color: #A0AEC0;">Probability of Acceptance</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': '%', 'font': {'color': '#FAFAFA', 'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#FAFAFA'},
                'bar': {'color': '#2ECC71' if prob > 0.5 else '#E74C3C'},
                'bgcolor': '#1E2130',
                'bordercolor': '#3D4663',
                'steps': [
                    {'range': [0, 30], 'color': '#2d1a1a'},
                    {'range': [30, 70], 'color': '#2d2d1a'},
                    {'range': [70, 100], 'color': '#1a2d1a'}
                ],
                'threshold': {'line': {'color': '#FAFAFA', 'width': 3}, 'thickness': 0.8, 'value': 50}
            }
        ))
        fig.update_layout(height=300, paper_bgcolor='#0E1117', plot_bgcolor='#0E1117', font=dict(color='#FAFAFA'))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**📋 Key Factors for This Prediction:**")
    factors = []
    if income > 100: factors.append(f"✅ **High income** (${income}K) - strongly positive")
    elif income < 50: factors.append(f"⚠️ **Lower income** (${income}K) - may reduce probability")
    if cd_account == 1: factors.append("✅ **Has CD Account** - strongly positive indicator")
    if education == 3: factors.append("✅ **Advanced education** - positive indicator")
    if ccavg > 3: factors.append(f"✅ **High CC spending** (${ccavg}K/month) - positive indicator")
    if securities == 1: factors.append("ℹ️ **Has Securities Account** - moderate positive")
    
    for f in factors if factors else ["ℹ️ No strongly influential factors identified for this profile"]:
        st.markdown(f)

with st.expander("📝 Model Notes & Limitations"):
    st.markdown("""
    ### Model Details
    - **Train/Test Split:** 80/20 with stratification
    - **Preprocessing:** StandardScaler normalization
    - **Logistic Regression:** Interpretable baseline model
    - **Random Forest:** 100 trees for higher accuracy
    
    ### Limitations
    - **Class Imbalance:** ~10% acceptance rate
    - **ZIP Code:** Not used due to high cardinality
    - **Temporal:** Assumes patterns remain stable over time
    """)
