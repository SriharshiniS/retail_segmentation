import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

# ------------------- Flask App -------------------
app = Flask(__name__)
customer_data = None  # global variable to store dataset

# ------------------- Routes -------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    global customer_data
    show = False
    if request.method == 'POST':
        # Use uploaded CSV if provided
        file = request.files.get('file')
        if file:
            customer_data = pd.read_csv(io.StringIO(file.stream.read().decode('utf8')))
        else:
            # Generate synthetic data
            np.random.seed(42)
            customer_data = pd.DataFrame({
                "Age": np.random.randint(18, 65, 1200),
                "AnnualIncome": np.random.randint(20000, 150000, 1200),
                "SpendingScore": np.random.randint(1, 100, 1200),
                "PurchaseFrequency": np.random.randint(1, 30, 1200),
                "TimeSpent": np.random.randint(5, 120, 1200)
            })
        show = True

    return render_template('index.html', show=show)


@app.route('/analyze')
def analyze():
    global customer_data
    if customer_data is None:
        return "No data available. Please upload a CSV or use synthetic data first."

    data = customer_data.copy()
    features = data.columns

    # ------------------- Scaling -------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])

    # ------------------- K-Means -------------------
    sil_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels))
    optimal_k = np.argmax(sil_scores) + 2

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # ------------------- PCA 2D -------------------
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    data['PCA1'] = X_pca_2d[:, 0]
    data['PCA2'] = X_pca_2d[:, 1]

    # Save 2D plot as base64
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='tab10', ax=ax)
    ax.set_title('Customer Segments (2D PCA)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_2d = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    # ------------------- PCA 3D (Plotly) -------------------
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    data['PCA3'] = X_pca_3d[:, 2]

    fig3d = px.scatter_3d(data, x='PCA1', y='PCA2', z='PCA3',
                          color='Cluster', size='AnnualIncome',
                          title="Customer Segments (3D PCA)")
    plot_3d = fig3d.to_html(full_html=False)

    # ------------------- Cluster Profiling -------------------
    cluster_profile = data.groupby('Cluster').mean().round(2)
    cluster_size = data['Cluster'].value_counts()

    return render_template('result.html',
                           show=True,
                           optimal_k=optimal_k,
                           cluster_profile=cluster_profile.to_html(classes='table table-striped'),
                           cluster_size=cluster_size.to_frame().to_html(classes='table table-striped'),
                           plot_2d=plot_2d,
                           plot_3d=plot_3d)
    

# ------------------- Run App -------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
