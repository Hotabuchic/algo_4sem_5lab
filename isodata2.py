import numpy as np
from sklearn.metrics import adjusted_rand_score, rand_score
from scipy.spatial.distance import cdist
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class FastISODATA:
    def __init__(self, init_k=7, max_iter=200, p=1.3):
        self.init_k = init_k
        self.max_iter = max_iter
        self.p = p
        self.centers = None

    def fit_predict(self, X, y_true=None):
        self.centers = X[np.random.choice(X.shape[0], self.init_k, replace=False)]

        for _ in range(self.max_iter):
            distances = cdist(X, self.centers, metric='minkowski', p=self.p)
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.centers.shape[0])])
            if np.allclose(self.centers, new_centers):
                break
            self.centers = new_centers

        final_labels = np.argmin(cdist(X, self.centers, metric='minkowski', p=self.p), axis=1)

        if y_true is not None:
            self.ari = adjusted_rand_score(y_true, final_labels)
            self.ri = rand_score(y_true, final_labels)
        return final_labels


def optimized_spa(X, y, n_iter=50, n_features=5, p=1.3):
    feature_cache = {}
    best_ari = -1
    best_ri = -1
    best_features = None

    for _ in tqdm(range(n_iter), desc="SPA поиск"):
        while True:
            features = tuple(sorted(np.random.choice(X.shape[1], n_features, replace=False)))
            if features not in feature_cache:
                break

        model = FastISODATA(p=p)
        labels = model.fit_predict(X[:, features])
        ari = adjusted_rand_score(y, labels)
        ri = rand_score(y, labels)
        feature_cache[features] = {'ari': ari, 'ri': ri}

        if ari > best_ari:
            best_ari = ari
            best_ri = ri
            best_features = features

    return list(best_features), best_ari, best_ri
