import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, send_file
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

# ------------------- Flask App -------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['PLOT_FOLDER'] = "static/plots"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)

# ------------------- Routes -------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Use uploaded CSV if provided
        file = request.files.get('file')
        if file:
            stream = io.StringIO(file.stream.read().decode("utf-8"), newline=None)
            data = pd.read_csv(stream)
        else:
            # Generate synthetic data
            np.random.seed(42)
            data = pd.DataFrame({
                "Age": np.random.randint(18, 65, 1200),
                "AnnualIncome": np.random.randint(20000, 150000, 1200),
                "SpendingScore": np.random.randint(1, 100, 1200),
                "PurchaseFrequency": np.random.randint(1, 30, 1200),
                "TimeSpent": np.random.randint(5, 120, 1200)
            })
        
        # Save data to session (or temp CSV)
        data.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'customers.csv'), index=False)
        return redirect(url_for('result'))
    
    return render_template('index.html')


@app.route('/result')
def result():
    # Load data
    data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'customers.csv'))
    features = data.columns

    # ------------------- Scaling -------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])

    # ------------------- K-Means -------------------
    # Use silhouette to pick K automatically
    sil_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels))
    optimal_k = np.argmax(sil_scores) + 2

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # ------------------- PCA for Visualization -------------------
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    data['PCA1'] = X_pca_2d[:,0]
    data['PCA2'] = X_pca_2d[:,1]

    # Save 2D plot
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='tab10')
    plt.title('Customer Segments (2D PCA)')
    plt.savefig(os.path.join(app.config['PLOT_FOLDER'], 'cluster_2d.png'))
    plt.close()

    # ------------------- 3D Plotly -------------------
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    data['PCA3'] = X_pca_3d[:,2]

    fig = px.scatter_3d(data, x='PCA1', y='PCA2', z='PCA3', 
                        color='Cluster', size='AnnualIncome')
    fig.write_html(os.path.join(app.config['PLOT_FOLDER'], 'cluster_3d.html'))

    # ------------------- Cluster Profiling -------------------
    cluster_profile = data.groupby('Cluster').mean().round(2)
    cluster_size = data['Cluster'].value_counts()

    return render_template('result.html', 
                           cluster_profile=cluster_profile.to_html(classes='table table-striped'),
                           cluster_size=cluster_size.to_frame().to_html(classes='table table-striped'),
                           optimal_k=optimal_k)


@app.route('/download')
def download():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'customers.csv')
    return send_file(path, as_attachment=True)


# ------------------- Run App -------------------
if __name__ == '__main__':
    app.run(debug=True)
