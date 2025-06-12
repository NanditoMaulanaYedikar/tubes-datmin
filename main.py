import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io 

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# ============================================
# 1. Data Collection (Pengumpulan Data)
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
page = st.sidebar.radio("Pilih halaman:", [
    "Data Understanding", 
    "Unsupervised Learning", 
    "Supervised Learning"])
# ============================================
# 2. Data Understanding (Pemahaman Data)
# ============================================
if page == "Data Understanding":
    st.title("Data Understanding")

    st.write("#### Struktur Data")
    st.dataframe(df.head())

    st.write("#### Info Kolom dan Tipe Data")
    buffer = io.StringIO()  
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("#### Statistik Deskriptif")
    st.dataframe(df.describe())

    st.write("#### Distribusi Variabel Target (Issued)")
    if 'Issued' in df.columns:
        st.bar_chart(df['Issued'].value_counts())
    else:
        st.warning("Kolom 'Issued' tidak ditemukan dalam dataset.")

    st.write("#### Korelasi Antar Variabel Numerik")
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm', linewidths=0.5, ax=ax_corr)
    ax_corr.set_title('Korelasi Antar Fitur')
    st.pyplot(fig_corr)

    st.write("#### Boxplot per Fitur Numerik")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_col = st.selectbox("Pilih fitur untuk ditampilkan boxplot:", num_cols)
    fig_box, ax_box = plt.subplots()
    sns.boxplot(x=df[selected_col], ax=ax_box, color='skyblue')
    ax_box.set_title(f"Boxplot: {selected_col}")
    st.pyplot(fig_box)

# ============================================
# 3. Data Preparation (Persiapan Data)
# ============================================
# Encode nilai target Issued jika masih berupa teks
if 'Issued' in df.columns:
    df['Issued'] = df['Issued'].replace({'Yes': 1, 'No': 0})

# Label Encoding untuk kolom kategorikal lainnya
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Tentukan kolom target
target_col = 'Issued' if 'Issued' in df.columns else df.columns[-1]
X = df.drop(target_col, axis=1)
y = df[target_col]

# Konversi dan scaling
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data untuk supervised learning
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ============================================
# 4A. Unsupervised Learning - K-Means Clustering
# ============================================
if page == "Unsupervised Learning":
    st.title("Unsupervised Learning - K-Means Clustering")

    n_clusters = st.slider("Pilih jumlah cluster:", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.write("#### Distribusi Cluster")
    st.bar_chart(df['Cluster'].value_counts())

    # Visualisasi PCA 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set2', ax=ax_pca)
    ax_pca.set_title('Visualisasi Cluster (PCA)', fontsize=14)
    st.pyplot(fig_pca)

# ============================================
# 4B. Supervised Learning - Logistic Regression
# ============================================
elif page == "Supervised Learning":
    st.title("Supervised Learning - Logistic Regression")

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else None

    st.write("#### Classification Report")
    labels = [0, 1]  # Pastikan hanya 0 dan 1
    report = classification_report(y_test, y_pred, labels=labels, target_names=['No', 'Yes'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.write("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax_cm)
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

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
    Interpretasi Koefisien:
    - Koefisien positif artinya fitur meningkatkan kemungkinan polis berhasil diterbitkan.
    - Koefisien negatif artinya fitur menurunkan kemungkinan polis berhasil diterbitkan.
    - Semakin besar nilai absolut koefisien, semakin besar pengaruhnya terhadap prediksi.
    """)
