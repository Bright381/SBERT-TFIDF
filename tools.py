import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
from sklearn.manifold import TSNE
import time



def correlation_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlations Heatmap")
    plt.show()

# ---- PCA ----
def plot_circle_of_correlations(pca, feature_names=None):
    """
    Plots the circle of correlations using PCA components and explained variance.

    Parameters:
    - pca: fitted sklearn PCA object
    - feature_names: list of str, optional
    """

    pcs = pca.components_[:2] 
    evr = pca.explained_variance_ratio_[:2]

    loadings = pcs.T * np.sqrt(evr)

    fig, ax = plt.subplots(figsize=(6, 6))

    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax.add_artist(circle)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    for i in range(loadings.shape[0]):
        x, y = loadings[i, 0], loadings[i, 1]
        ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, fc='blue', ec='blue')
        if feature_names:
            ax.text(x * 1.1, y * 1.1, feature_names[i], ha='center', va='center')

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Circle of Correlations")
    ax.grid(True)
    plt.show()


def plot_simple_scree(pca):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, marker='o')
    plt.title("Éboulis des valeurs propres")
    plt.xlabel("Composante principale")
    plt.ylabel("Ratio de variance expliquée")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_scree(pca_model):
    """
    Trace un éboulis des valeurs propres (scree plot) avec style.

    Paramètre :
    - pca_model : un objet PCA déjà entraîné (sklearn.decomposition.PCA)
    """

    evr = pca_model.explained_variance_ratio_
    components = np.arange(1, len(evr) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(components, evr, marker='o', linestyle='-', color='royalblue', label='Variance expliquée')
    plt.bar(components, evr, alpha=0.3, color='skyblue')

    cumulative_evr = np.cumsum(evr)
    plt.plot(components, cumulative_evr, marker='s', linestyle='--', color='darkorange', label='Variance cumulée')

    plt.title("Éboulis des valeurs propres (Scree Plot)", fontsize=14, fontweight='bold')
    plt.xlabel("Composantes principales", fontsize=12)
    plt.ylabel("Ratio de variance expliquée", fontsize=12)
    plt.xticks(components)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()



# ---- plot distributions -------

def plot_quantitative(df, max_cols=20):
    quant_vars = df.select_dtypes(include=['int64', 'float64']).columns[:max_cols]
    if len(quant_vars) == 0:
        print("Aucune variable quantitative à afficher.")
        return

    rows = math.ceil(len(quant_vars) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, rows * 4))
    axes = np.array(axes).flatten()

    fig.suptitle("Distributions des variables quantitatives", fontsize=16, fontweight='bold')
    for i, col in enumerate(quant_vars):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(col)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_qualitative(df, max_cols=20, max_categories=20):
    qual_vars = df.select_dtypes(include=['object', 'category']).columns[:max_cols]
    if len(qual_vars) == 0:
        print("Aucune variable qualitative à afficher.")
        return

    rows = math.ceil(len(qual_vars) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, rows * 4))
    axes = np.array(axes).flatten()

    fig.suptitle("Distributions des variables qualitatives", fontsize=16, fontweight='bold')
    for i, col in enumerate(qual_vars):
        # Limiter le nombre de catégories
        if df[col].nunique() > max_categories:
            top_values = df[col].value_counts().nlargest(max_categories).index
            data = df[df[col].isin(top_values)][col]
        else:
            data = df[col]
        sns.countplot(x=data, ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis='x', rotation=45)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ----- t-SNE -----

def compare_tsne_parameters(features, perplexities=[5, 10, 20, 30, 50, 70], labels=None, init='pca'):
    n_plots = len(perplexities)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()

    for i, perp in enumerate(perplexities):
        print(f"Calcul t-SNE pour perplexity={perp}...")
        start = time.time()
        
        tsne = TSNE(
            n_components=2, 
            perplexity=perp, 
            init=init, 
            random_state=42,
            n_jobs=-1
        )
        embedding = tsne.fit_transform(features)
        end = time.time()
        
        df_viz = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'Category': labels
        })

        scatter = sns.scatterplot(
            data=df_viz,
            x='x', y='y', 
            hue='Category',
            ax=axes[i],
            palette='tab10',  
            alpha=0.7,
            s=40,             # points sizes
            legend=False,
            linewidth=0.5
        )
        
        axes[i].set_title(f"Perplexity: {perp}", fontsize=14)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    plt.tight_layout()
    plt.show()

# perps = [5, 10, 20, 30, 50, 70]
# compare_tsne_parameters(vgg16_features, perplexities=perps)

# ----- confusion matrix -------
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

def conf_mat_transform(y_true, y_pred):
    """
    Aligne les labels de clustering (y_pred) sur les vraies classes (y_true)
    à l'aide de l'algorithme hongrois.
    """
    conf_mat = metrics.confusion_matrix(y_true, y_pred)

    # Algorithme hongrois (maximisation de la diagonale)
    row_ind, col_ind = linear_sum_assignment(-conf_mat)

    cluster_to_class = dict(zip(col_ind, row_ind))
    print("Correspondance des clusters :", cluster_to_class)

    y_pred_aligned = np.array([cluster_to_class[c] for c in y_pred])

    aligned_conf_mat = metrics.confusion_matrix(y_true, y_pred_aligned)

    return aligned_conf_mat, y_pred_aligned


def evaluate_supervised_classifier(y_true, y_pred, id2label_dic):
    # check tout est np.array "plat"
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    unique_ids = np.unique(np.concatenate([y_true, y_pred]))

    display_names = [id2label_dic[i] for i in unique_ids]


    conf_mat = metrics.confusion_matrix(y_true, y_pred, labels=unique_ids)
    
    print(metrics.classification_report(y_true, y_pred), "\n ARI: ", metrics.adjusted_rand_score(y_true, y_pred))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=display_names, 
                yticklabels=display_names)
    
    plt.ylabel('Ground truth')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')
    plt.show()

    return conf_mat, y_pred
