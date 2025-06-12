import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import SparsePCA

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# ============================================
# Load Data
# ============================================
@st.cache_data

def load_data():
    try:
        df = pd.read_excel('MKG1_Data_Konversi_Repricing.xlsx')
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# ============================================
# Sidebar Navigasi
# ============================================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["Unsupervised Learning", "Supervised Learning"])

# ============================================
# Preprocessing
# ============================================
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

target_col = 'Issued' if 'Issued' in df.columns else df.columns[-1]
X = df.drop(target_col, axis=1)
y = df[target_col]

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# Halaman: Unsupervised Learning
# ============================================
if page == "Unsupervised Learning":
    st.title("Unsupervised Learning - K-Means Clustering")

    st.write("#### Statistik Deskriptif")
    st.dataframe(df.describe())

    st.write("#### Visualisasi Korelasi Fitur")
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm', linewidths=0.5, ax=ax_corr)
    ax_corr.set_title('Korelasi Antar Fitur')
    st.pyplot(fig_corr)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = kmeans_labels

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    jitter_strength = st.slider('Jitter Strength (untuk visualisasi PCA)', 0.0, 1.0, 0.2, step=0.05)
    df['PCA 1'] = df['PCA1'] + np.random.normal(0, jitter_strength, size=len(df))
    df['PCA 2'] = df['PCA2'] + np.random.normal(0, jitter_strength, size=len(df))

    fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='PCA 1', 
        y='PCA 2', 
        hue='Cluster', 
        data=df, 
        palette='Set2',
        s=60,
        alpha=0.6,
        edgecolor='black',
        linewidth=0.5,
        ax=ax_pca
    )

    ax_pca.set_title('Visualisasi Cluster (PCA)', fontsize=14)
    ax_pca.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_pca)

# ============================================
# Halaman: Supervised Learning
# ============================================
elif page == "Supervised Learning":
    st.title("Supervised Learning - Logistic Regression")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else None

    st.write("#### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.write("#### Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

    st.write("Label unik di y_test:", np.unique(y_test))
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)

        st.write("#### ROC Curve")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        st.pyplot(fig_roc)
    else:
        st.warning("ROC Curve hanya tersedia untuk klasifikasi biner.")

    st.write("#### Koefisien Model")
    coef_df = pd.DataFrame({
        'Fitur': X.columns,
        'Koefisien': lr.coef_[0]
    }).sort_values(by='Koefisien', key=abs, ascending=False)
    st.dataframe(coef_df)

    st.markdown("""
    **Interpretasi Koefisien:**
    - Koefisien positif artinya fitur meningkatkan kemungkinan polis berhasil diterbitkan.
    - Koefisien negatif artinya fitur menurunkan kemungkinan polis berhasil diterbitkan.
    - Semakin besar nilai absolut koefisien, semakin besar pengaruhnya terhadap prediksi.
    """)