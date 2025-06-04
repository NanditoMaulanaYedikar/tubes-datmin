import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    df = pd.read_excel('data/MKG1- Data Konversi & Repricing.xlsx', sheet_name=0)
    return df

df = load_data()

# ============================================
# Sidebar Navigasi (pakai tombol) alif jelek
# ============================================
st.sidebar.title("Navigasi")
if st.sidebar.button("Unsupervised Learning"):
    st.session_state['page'] = "unsupervised"
if st.sidebar.button("Supervised Learning"):
    st.session_state['page'] = "supervised"

# Inisialisasi page default
if 'page' not in st.session_state:
    st.session_state['page'] = "unsupervised"

# ============================================
# Preprocessing
# ============================================
# Label Encoding untuk kolom kategorikal
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Target variabel
target_col = 'Issued' if 'Issued' in df.columns else df.columns[-1]
X = df.drop(target_col, axis=1)
y = df[target_col]

# Konversi kolom ke numerik
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# Halaman: Unsupervised Learning
# ============================================
if st.session_state['page'] == "unsupervised":
    st.title("Unsupervised Learning - K-Means Clustering")

    st.write("#### Statistik Deskriptif")
    st.dataframe(df.describe())

    st.write("#### Visualisasi Korelasi Fitur")
    plt.figure(figsize=(15, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5, square=True)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title('Korelasi Antar Fitur', fontsize=14, fontweight='bold')
    st.pyplot(plt.gcf())

    # K-Means Clustering
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

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set2', ax=ax)
    ax.set_title('Visualisasi Cluster (PCA)', fontsize=14, fontweight='bold')
    st.pyplot(fig)

# ============================================
# Halaman: Supervised Learning
# ============================================
elif st.session_state['page'] == "supervised":
    st.title("Supervised Learning - Logistic Regression")

    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    st.write("#### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.write("#### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    st.pyplot(fig)

    st.write("Label unik di y_test:", np.unique(y_test))
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)

        st.write("#### ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        st.pyplot(fig)
    else:
        st.warning("ROC Curve hanya tersedia untuk binary classification.")

    st.write("#### Koefisien Model")
    coef_df = pd.DataFrame({
        'Fitur': X.columns,
        'Koefisien': lr.coef_[0]
    }).sort_values(by='Koefisien', key=abs, ascending=False)
    st.dataframe(coef_df)
