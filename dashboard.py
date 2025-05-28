import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page configuration  
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="👥",
    layout="wide"
)

# Set matplotlib and seaborn style
plt.style.use('default')
sns.set_palette("husl")

# Enhanced CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin: 0;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        margin: 30px 0 10px 0;
        color: #ffffff; /* Ubah warna teks jadi putih */
    }
    .insight-box {
        background-color: #1e1e1e;
        padding: 16px;
        border-radius: 10px;
        color: #f1f1f1;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .risk-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #e9ecef;
        padding: 1.2rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        border-color: #3498db;
    }
    /* Fix warna teks putih di dalam metric card */
    div[data-testid="metric-container"] * {
        color: #212529 !important;
    }
    .stColumn > div {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the CSV files separately"""
    try:
        # Load the CSV files
        df_processed = pd.read_csv('X_processed.csv')
        df_predictions = pd.read_csv('attrition_test_predictions.csv')
        
        return df_processed, df_predictions
    except FileNotFoundError as e:
        st.error(f"CSV files not found: {e}")
        return None, None

def get_hardcoded_metrics():
    """Return metrics from the analysis results"""
    return {
        'total_employees': 1058,
        'total_attritions': 179,
        'attrition_rate': 16.92,
        'f1_score': 60.24,
        'precision': 53.19,
        'recall': 69.44,
        'optimal_threshold': 0.150,
        'test_accuracy': 85.38,  # (154+25)/(154+22+11+25) * 100
        
        # Risk segmentation
        'low_risk_employees': 352,
        'low_risk_attritions': 20,
        'low_risk_rate': 5.7,
        
        'medium_risk_employees': 354,
        'medium_risk_attritions': 41,
        'medium_risk_rate': 11.6,
        
        'high_risk_employees': 352,
        'high_risk_attritions': 118,
        'high_risk_rate': 33.5,
        
        # Confusion matrix values
        'true_negatives': 154,
        'false_positives': 22,
        'false_negatives': 11,
        'true_positives': 25,
        'missed_attritions': 11
    }

def create_characteristics_plots():
    """Create cleaner and smaller plots based on the analysis data"""
    plots = {}
    
    # Set consistent figure parameters for smaller plots
    plt.rcParams.update({
        'figure.figsize': (8, 5),
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8
    })
    
    # 1. Demographic Profile Comparison (Age and Experience)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Age comparison
    categories = ['High Risk\nEmployees', 'Company\nAverage']
    ages = [31.7, 36.9]
    colors = ['#e74c3c', '#27ae60']
    
    bars1 = ax1.bar(categories, ages, color=colors, alpha=0.8, width=0.6)
    ax1.set_ylabel('Age (years)', fontweight='bold')
    ax1.set_title('Average Age Comparison', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(ages) * 1.2)
    
    # Add value labels
    for bar, age in zip(bars1, ages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{age} years', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Work experience comparison
    work_exp = [6.2, 11.3]
    
    bars2 = ax2.bar(categories, work_exp, color=colors, alpha=0.8, width=0.6)
    ax2.set_ylabel('Years of Experience', fontweight='bold')
    ax2.set_title('Work Experience Comparison', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(work_exp) * 1.2)
    
    # Add value labels
    for bar, exp in zip(bars2, work_exp):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{exp} years', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plots['demographic_profile'] = fig1
    
    # 2. Job Characteristics (Job Level and Salary)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Job Level distribution for high-risk
    job_levels = ['Level 1\n(Junior)', 'Other Levels']
    job_percentages = [70.7, 29.3]
    colors_job = ['#e74c3c', '#3498db']
    
    wedges, texts, autotexts = ax1.pie(job_percentages, labels=job_levels, colors=colors_job, 
                                      autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
    ax1.set_title('Job Level Distribution\n(High-Risk Employees)', fontweight='bold', pad=20)
    
    # Salary comparison
    salary_categories = ['High Risk\nEmployees', 'Company\nAverage']
    salaries = [3591, 6503]
    colors_salary = ['#e74c3c', '#27ae60']
    
    bars = ax2.bar(salary_categories, salaries, color=colors_salary, alpha=0.8, width=0.6)
    ax2.set_ylabel('Monthly Salary ($)', fontweight='bold')
    ax2.set_title('Monthly Salary Comparison', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(salaries) * 1.2)
    
    # Add value labels
    for bar, salary in zip(bars, salaries):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'${salary:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plots['job_characteristics'] = fig2
    
    # 3. High-Risk Job Roles Distribution
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    job_roles = ['Research Scientist', 'Laboratory Technician', 'Sales Executive']
    percentages = [31.1, 23.5, 22.5]
    colors_roles = ['#e74c3c', '#f39c12', '#e67e22']
    
    bars = ax3.barh(job_roles, percentages, color=colors_roles, alpha=0.8, height=0.6)
    ax3.set_xlabel('Percentage of High-Risk Employees (%)', fontweight='bold')
    ax3.set_title('Job Roles with Highest Attrition Risk', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax3.set_xlim(0, max(percentages) * 1.2)
    
    # Add value labels
    for bar, pct in zip(bars, percentages):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct}%', ha='left', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plots['job_roles_risk'] = fig3
    
    # 4. Critical Risk Factor Combinations
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    risk_combinations = [
        'Sales Rep +\nOvertime',
        'Sales Rep +\nSingle',
        'Junior Level +\nOvertime', 
        'Single +\nOvertime'
    ]
    attrition_rates = [76.5, 61.5, 53.3, 51.5]
    
    # Create color gradient based on attrition rate
    colors_comb = ['#c0392b', '#e74c3c', '#f39c12', '#f1c40f']
    
    bars = ax4.barh(risk_combinations, attrition_rates, color=colors_comb, alpha=0.8, height=0.6)
    ax4.set_xlabel('Attrition Rate (%)', fontweight='bold')
    ax4.set_title('Critical Risk Factor Combinations', fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax4.set_xlim(0, max(attrition_rates) * 1.2)
    
    # Add value labels and risk indicators
    for bar, rate in zip(bars, attrition_rates):
        ax4.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{rate}%', ha='left', va='center', fontweight='bold', fontsize=10)
        
        # Add risk level indicator
        if rate > 70:
            risk_text = 'CRITICAL'
            risk_color = '#c0392b'
        elif rate > 50:
            risk_text = 'HIGH'
            risk_color = '#e74c3c'
        else:
            risk_text = 'MEDIUM'
            risk_color = '#f39c12'
        
        ax4.text(8, bar.get_y() + bar.get_height()/2,
                risk_text, ha='left', va='center', 
                color=risk_color, fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plots['risk_combinations'] = fig4
    
    return plots

def main():
    # Header
    st.markdown('<h1 class="main-header">👥 Employee Attrition Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Get hardcoded metrics
    metrics = get_hardcoded_metrics()
    
    # ==================== SECTION 1: KPI METRICS ====================
    st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    # Main KPI Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 Total Employees (cleaned)", f"{metrics['total_employees']:,}")
    
    with col2:
        st.metric("📉 Total Attritions (cleaned) ", f"{metrics['total_attritions']:,}")
    
    with col3:
        st.metric("📈 Attrition Rate (cleaned)", f"{metrics['attrition_rate']:.1f}%")
    
    with col4:
        st.metric("🎯 F1-Score", f"{metrics['f1_score']:.1f}%")
    
    with col5:
        st.metric("🔍 Test Accuracy", f"{metrics['test_accuracy']:.1f}%")

    # ==================== SECTION 2: EMPLOYEE CHARACTERISTICS ====================
    st.markdown('<div class="section-header">📈 High-Risk Employee Characteristics</div>', unsafe_allow_html=True)
    
    # Create characteristics plots
    char_plots = create_characteristics_plots()
    
    if char_plots:
        # First row of plots
        st.markdown("#### 👤 Demographic & Job Characteristics")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'demographic_profile' in char_plots:
                st.pyplot(char_plots['demographic_profile'], use_container_width=True)
        
        with col2:
            if 'job_characteristics' in char_plots:
                st.pyplot(char_plots['job_characteristics'], use_container_width=True)
        
        # Second row of plots
        st.markdown("#### 🎯 Risk Analysis by Role & Factors")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'job_roles_risk' in char_plots:
                st.pyplot(char_plots['job_roles_risk'], use_container_width=True)
        
        with col2:
            if 'risk_combinations' in char_plots:
                st.pyplot(char_plots['risk_combinations'], use_container_width=True)
        # Key insights
    st.markdown("""
    <div class="insight-box">
        <h4 style="margin-top: 0;">🔍 Key Findings</h4>
        <p>• High-risk employees are <strong>5.2 years younger</strong> on average (31.7 vs 36.9 years)</p>
        <p>• They have <strong>45% less work experience</strong> (6.2 vs 11.3 years)</p>
        <p>• <strong>70.7%</strong> are in junior-level positions</p>
        <p>• Monthly salary is <strong>45% lower</strong> than company average ($3,591 vs $6,503)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 Precision", f"{metrics['precision']:.1f}%")
    
    with col2:
        st.metric("📊 Recall", f"{metrics['recall']:.1f}%")
    
    with col3:
        st.metric("⚖️ Optimal Threshold", f"{metrics['optimal_threshold']:.3f}")
    
    with col4:
        st.metric("❌ Missed Attritions", f"{metrics['missed_attritions']:,}")
    
    # Risk Segmentation Analysis
    st.markdown("### 📊 Employee Risk Segmentation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h4 style="color: #28a745; margin-top: 0;">🟢 Low Risk Segment</h4>
            <p><strong>Employees:</strong> {metrics['low_risk_employees']:,}</p>
            <p><strong>Attritions:</strong> {metrics['low_risk_attritions']:,}</p>
            <p><strong>Rate:</strong> <span class="risk-low">{metrics['low_risk_rate']}%</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-box">
            <h4 style="color: #fd7e14; margin-top: 0;">🟡 Medium Risk Segment</h4>
            <p><strong>Employees:</strong> {metrics['medium_risk_employees']:,}</p>
            <p><strong>Attritions:</strong> {metrics['medium_risk_attritions']:,}</p>
            <p><strong>Rate:</strong> <span class="risk-medium">{metrics['medium_risk_rate']}%</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="insight-box">
            <h4 style="color: #dc3545; margin-top: 0;">🔴 High Risk Segment</h4>
            <p><strong>Employees:</strong> {metrics['high_risk_employees']:,}</p>
            <p><strong>Attritions:</strong> {metrics['high_risk_attritions']:,}</p>
            <p><strong>Rate:</strong> <span class="risk-high">{metrics['high_risk_rate']}%</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Confusion Matrix Breakdown
    st.markdown("### 🔍 Model Prediction Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("✅ True Negatives", f"{metrics['true_negatives']:,}", help="Correctly predicted to stay")
    
    with col2:
        st.metric("❌ False Positives", f"{metrics['false_positives']:,}", help="Incorrectly predicted to leave")
    
    with col3:
        st.metric("❌ False Negatives", f"{metrics['false_negatives']:,}", help="Missed attrition cases")
    
    with col4:
        st.metric("✅ True Positives", f"{metrics['true_positives']:,}", help="Correctly predicted to leave")
    
    st.markdown("---")
    
    
    # Load data info in expander
    with st.expander("📋 Data & Model Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Dataset Overview:**")
            st.write(f"- Total employees analyzed: {metrics['total_employees']:,}")
            st.write(f"- Historical attrition cases: {metrics['total_attritions']:,}")
            st.write(f"- Overall attrition rate: {metrics['attrition_rate']:.1f}%")
        
        with col2:
            st.markdown("**🤖 Model Performance:**")
            st.write(f"- Optimal threshold: {metrics['optimal_threshold']:.3f}")
            st.write(f"- Test accuracy: {metrics['test_accuracy']:.1f}%")
            st.write(f"- F1-Score: {metrics['f1_score']:.1f}%")

if __name__ == "__main__":
    main()