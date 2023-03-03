import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from kmeans import KMeans

# state = random.randint(0, 1500)
state = 0

def make_data(
        kind="blobs",
        n_clusters=4, 
        n_samples=350, 
        cluster_std=0.3, 
        random_state=state
    ):
    assert kind in ["blobs", "moons"], "keyword \"kind\" can be either \"blobs\" or \"moons\""

    if kind == "blobs":
        data, labels = make_blobs(
            n_samples=n_samples,
            centers=n_clusters,  # Specify the number of cluster our data should have
            cluster_std=cluster_std,
            random_state=random_state
        )
    elif (kind == "moons"):
        data, labels = make_moons(
            n_samples=n_samples,
            random_state=random_state
        )
    return data, labels



def plot_unclustered_data(
        data, 
        alpha=0.3,
        fig_title="Unclustered Data",
        savefig=False, 
        fig_name="unclustered_data.png"
    ):
    """
        Plot original data
    """
    plt.title(fig_title)
    plt.scatter(data[:, 0], data[:, 1], s=15, color=(0.1, 0.2, 0.5, alpha))
    # plt.xticks([])
    # plt.yticks([])
    if savefig:
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        plt.clf()
    else:
        plt.show()
        plt.clf()

def plot_clustered_data(
        data, 
        labels, 
        n_clusters,
        centers=None,
        alpha=0.3, 
        randomize_colors=False, 
        fig_title="Clustered Data",
        savefig=False, 
        fig_name="clustered_data.png"
    ):
    """
        Map a distinct color to each data cluster (based on predicted label)
    """
    for cluster_number in range(n_clusters):
        if randomize_colors:
            used_colors = []
            r = random.randint(0, 10) / 10.
            g = random.randint(0, 10) / 10.
            b = random.randint(0, 10) / 10.
            rgba = (r,g,b,alpha)
            if rgba in used_colors:
                print("Color already exist when cluster number is: ", cluster_number)
                cluster_number -= 1
                continue
            else:
                used_colors.append(rgba)
            cluster_data = labels == cluster_number
            plt.scatter(data[cluster_data, 0], data[cluster_data, 1], s=15, color=(r,g,b,alpha))
        else:
            cluster_data = labels == cluster_number
            plt.scatter(data[cluster_data, 0], data[cluster_data, 1], s=15, alpha=alpha)

    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5)

    plt.title(fig_title)
    plt.xticks([])
    plt.yticks([])
    if savefig:
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        plt.clf()
    else:
        plt.show()
        plt.clf()



def main():
    X, true_y = make_data(kind="moons")
    plot_unclustered_data(X, fig_title="Original Unclustered Data", savefig=False)

    # Standardization of features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    plot_unclustered_data(X_std, fig_title="Standardized Unclustered Data", savefig=False)

    # Initialize KMeans
    number_of_clusters = 2
    km_model = KMeans(n_clusters=number_of_clusters)

    # Fit the data and predict cluster labels
    # A label is a number ranging from 0 to n_clusters
    km_model.fit(X_std)
    predict_y = km_model.predict(X_std)  # list of labels, i.e. number in [0, n_clusters]
    centers = km_model.centroids

    plot_clustered_data(
        data = X_std, 
        labels=true_y, 
        n_clusters=number_of_clusters,
        centers=centers,
        fig_title="True Clustered Data", 
        savefig=False, 
        fig_name="true_clustered_data.png"
    )
    
    plot_clustered_data(
        data = X_std, 
        labels=predict_y, 
        n_clusters=number_of_clusters, 
        centers=centers,
        fig_title="Predicted Clustered Data", 
        savefig=False, 
        fig_name="predicted_clustered_data.png"
    )


if __name__ == "__main__":
    main()