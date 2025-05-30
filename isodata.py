import numpy as np
from sklearn.metrics import adjusted_rand_score, rand_score
from scipy.spatial.distance import cdist
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class ISODATA:
    def __init__(self, init_k=7, max_iter=100, p=1.3,
                 merge_threshold=0.5, split_threshold=1.5,
                 min_cluster_size=3, min_clusters=3, max_clusters=12):
        self.init_k = init_k
        self.max_iter = max_iter
        self.p = p
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.min_clusters = min_clusters
        self.centers = None
        self.ari = None
        self.ri = None

    def _merge_clusters(self, X, labels):
        """Объединение близких кластеров"""
        merged_centers = []
        to_merge = set()

        for i in range(len(self.centers)):
            if i in to_merge:
                continue

            closest = None
            min_dist = float('inf')

            for j in range(i + 1, len(self.centers)):
                if j in to_merge:
                    continue

                dist = np.linalg.norm(self.centers[i] - self.centers[j])
                if dist < self.merge_threshold and dist < min_dist:
                    min_dist = dist
                    closest = j

            if closest is not None:
                # Объединяем кластеры i и closest
                combined_points = X[np.isin(labels, [i, closest])]
                new_center = combined_points.mean(axis=0)
                merged_centers.append(new_center)
                to_merge.update([i, closest])
            else:
                merged_centers.append(self.centers[i])

        return np.array(merged_centers)

    def _split_clusters(self, X, labels):
        """Разделение кластеров с высокой дисперсией"""
        new_centers = []

        for i in range(len(self.centers)):
            cluster_points = X[labels == i]

            if len(cluster_points) < self.min_cluster_size:
                # Кластер слишком мал - оставляем как есть
                new_centers.append(self.centers[i])
                continue

            # Вычисляем дисперсию внутри кластера
            variances = np.var(cluster_points, axis=0)
            max_var = np.max(variances)

            if max_var > self.split_threshold and len(self.centers) < self.max_clusters:
                # Находим признак с максимальной дисперсией
                split_feature = np.argmax(variances)
                split_value = np.median(cluster_points[:, split_feature])

                # Разделяем кластер по медиане
                mask = cluster_points[:, split_feature] < split_value
                center1 = cluster_points[mask].mean(axis=0)
                center2 = cluster_points[~mask].mean(axis=0)

                new_centers.extend([center1, center2])
            else:
                new_centers.append(self.centers[i])

        return np.array(new_centers)

    def fit_predict(self, X, y_true=None):
        self.centers = X[np.random.choice(X.shape[0], self.init_k, replace=False)]

        for _ in range(self.max_iter):
            # Шаг 1: Назначение точек кластерам
            distances = cdist(X, self.centers, metric='minkowski', p=self.p)
            labels = np.argmin(distances, axis=1)

            # Шаг 2: Удаление маленьких кластеров
            unique, counts = np.unique(labels, return_counts=True)
            valid_clusters = unique[counts >= self.min_cluster_size]
            if len(valid_clusters) < len(self.centers):
                if len(valid_clusters) >= self.min_clusters:
                    self.centers = self.centers[valid_clusters]
                else:
                    # Оставляем min_clusters самых больших кластеров
                    largest_clusters = np.argsort(counts)[-self.min_clusters:]
                    self.centers = self.centers[largest_clusters]
                distances = cdist(X, self.centers, metric='minkowski', p=self.p)
                labels = np.argmin(distances, axis=1)

            # Шаг 3: Пересчет центроидов
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(len(self.centers))])

            # Шаг 4: Слияние кластеров
            if len(self.centers) > self.min_clusters:
                new_centers = self._merge_clusters(X, labels)
                if len(new_centers) < self.min_clusters:
                    new_centers = self.centers

            # Шаг 5: Разделение кластеров
            if len(new_centers) < self.max_clusters:
                new_centers = self._split_clusters(X, labels)
                if len(new_centers) > self.max_clusters:
                    new_centers = new_centers[:self.max_clusters]

            # Проверка сходимости
            if len(new_centers) == len(self.centers) and np.allclose(self.centers, new_centers) and len(new_centers) >= self.min_clusters:
                break

            self.centers = new_centers

        # Финальное назначение меток
        final_labels = np.argmin(cdist(X, self.centers, metric='minkowski', p=self.p), axis=1)

        if len(self.centers) < self.min_clusters:
            # Выбираем min_clusters случайных точек как центры
            self.centers = X[np.random.choice(X.shape[0], self.min_clusters, replace=False)]
            final_labels = np.argmin(cdist(X, self.centers, metric='minkowski', p=self.p), axis=1)

        if y_true is not None:
            self.ari = adjusted_rand_score(y_true, final_labels)
            self.ri = rand_score(y_true, final_labels)
        return final_labels


def optimized_spa(X, y, n_iter=50, n_features=5, p=1.3, init_k=7, min_clusters=3, max_clusters=12):
    feature_cache = {}
    best_ari = -1
    best_ri = -1
    best_features = None

    if n_features >= 17:
        n_iter = 1
    elif n_features == 16 or n_features == 1:
        n_iter = 17

    for _ in tqdm(range(n_iter), desc="SPA поиск"):
        while True:
            features = tuple(sorted(np.random.choice(X.shape[1], n_features, replace=False)))
            if features not in feature_cache:
                break
        model = ISODATA(p=p, init_k=init_k, min_clusters=min_clusters, max_clusters=max_clusters)
        labels = model.fit_predict(X[:, features])
        ari = adjusted_rand_score(y, labels)
        ri = rand_score(y, labels)
        feature_cache[features] = {'ari': ari, 'ri': ri}

        if ri > best_ri:
            best_ari = ari
            best_ri = ri
            best_features = features

    return list(best_features), best_ari, best_ri
