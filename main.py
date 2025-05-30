import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFormLayout, QPlainTextEdit,
    QPushButton, QLineEdit, QFileDialog, QTableView,
    QSizePolicy, QHeaderView, QGridLayout
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from isodata import ISODATA, optimized_spa


class DatasetClusterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Кластеризация")
        self.resize(800, 600)

        self.loadedDataFrame = pd.DataFrame()
        self.initializeUserInterface()

    def initializeUserInterface(self):
        mainLayout = QGridLayout()
        self.setLayout(mainLayout)

        controlPanel = QWidget()
        controlLayout = QFormLayout()
        controlPanel.setLayout(controlLayout)

        self.buttonLoadCsv = QPushButton("Загрузить")
        self.buttonLoadCsv.clicked.connect(self.handleLoad)
        controlLayout.addRow(self.buttonLoadCsv)

        self.buttonDeidentify = QPushButton("Обезличить")
        self.buttonDeidentify.clicked.connect(self.anonymize_dataframe)
        controlLayout.addRow(self.buttonDeidentify)

        self.buttonRunClustering = QPushButton("Запустить")
        self.buttonRunClustering.clicked.connect(self.handleRunClustering)
        controlLayout.addRow(self.buttonRunClustering)

        self.mincluster = QLineEdit("3")
        controlLayout.addRow("Минимальное кол-во кластеров:", self.mincluster)

        self.init_k = QLineEdit("7")
        controlLayout.addRow("Начальное кол-во кластеров:", self.init_k)

        self.maxcluster = QLineEdit("12")
        controlLayout.addRow("Максимальное кол-во кластеров:", self.maxcluster)

        self.inputFeatureCount = QLineEdit("3")
        controlLayout.addRow("Число признаков:", self.inputFeatureCount)

        self.textStatusOutput = QPlainTextEdit()
        self.textStatusOutput.setReadOnly(True)
        controlLayout.addRow(self.textStatusOutput)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        self.figure2 = Figure(figsize=(5, 4), dpi=100)
        self.canvas2 = FigureCanvas(self.figure2)

        self.tableViewData = QTableView()
        self.tableViewData.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mainLayout.addWidget(controlPanel, 0, 0)
        mainLayout.addWidget(self.tableViewData, 1, 0)
        mainLayout.addWidget(self.canvas, 0, 1)
        mainLayout.addWidget(self.canvas2, 1, 1)

    def updateTableView(self):
        if self.loadedDataFrame.empty:
            return
        model = QStandardItemModel()
        model.setColumnCount(len(self.loadedDataFrame.columns))
        model.setRowCount(len(self.loadedDataFrame.index))
        model.setHorizontalHeaderLabels(self.loadedDataFrame.columns.tolist())
        for rowIndex, row in self.loadedDataFrame.iterrows():
            for columnIndex, cellValue in enumerate(row):
                item = QStandardItem(str(cellValue))
                model.setItem(rowIndex, columnIndex, item)
        self.tableViewData.setModel(model)
        header = self.tableViewData.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        for index in range(min(3, len(self.loadedDataFrame.columns))):
            header.setSectionResizeMode(index, QHeaderView.Stretch)


    def handleLoad(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Выбрать файл", "", "CSV Files (*.csv);;")
        if not filePath:
            return
        self.loadedDataFrame = pd.read_csv(filePath)
        self.loadedDataFrame.drop("CUST_ID", axis=1, inplace=True)
        self.updateTableView()
        self.textStatusOutput.setPlainText("Датасет загружен успешно.")

    def handleRunClustering(self):
        n_features = int(self.inputFeatureCount.text())
        min_cluster = int(self.mincluster.text())
        max_cluster = int(self.maxcluster.text())
        init_k =int(self.init_k.text())
        X = self.loadedDataFrame.copy().fillna(0)

        kmeans = KMeans(n_clusters=5, random_state=42)
        y = kmeans.fit_predict(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_full = ISODATA(min_clusters=min_cluster, max_clusters=max_cluster, init_k=init_k)
        features = tuple(sorted(np.random.choice(X.shape[1], n_features, replace=False)))
        labels_full = model_full.fit_predict(X_scaled[:, features], y)

        best_feats, best_ari, best_ri = optimized_spa(X_scaled, y, n_iter=50, n_features=n_features, p=1.3, min_clusters=min_cluster, max_clusters=max_cluster, init_k=init_k)
        X_selected = X_scaled[:, best_feats]
        model_selected = ISODATA(min_clusters=min_cluster, max_clusters=max_cluster, init_k=init_k)
        labels_selected = model_selected.fit_predict(X_selected, y)

        self.plot_clusters(X_scaled[:, features], labels_full, self.figure, self.canvas,
                           "Кластеризация без отбора признаков")
        self.plot_clusters(X_selected, labels_selected, self.figure2, self.canvas2,
                           "Кластеризация с отбором признаков")

        n_clusters_full = len(np.unique(labels_full))
        n_clusters_selected = len(np.unique(labels_selected))

        self.textStatusOutput.setPlainText(
            f"Кластеризация на {n_features} признаках\n"
            f"Кластеров (без отбора): {n_clusters_full}\n"
            f"Кластеров (с отбором): {n_clusters_selected}\n"
            f"Индекс Rand без отбора признаков: {model_full.ri:.4f}\n"
            f"Индекс Rand с отбором признаков: {model_selected.ri:.4f}\n"
        )

    def plot_clusters(self, X, labels, figure, canvas, title):
        figure.clear()
        ax = figure.add_subplot(111)

        if X.shape[1] == 1:
            X_plot = np.zeros((len(X), 2))
            X_plot[:, 0] = X[:, 0]
            scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1],
                                 c=labels, cmap='viridis', alpha=0.6)
            ax.set_xlabel('Единственный признак')
            ax.set_yticks([])
        else:
            if X.shape[1] > 2:
                pca = PCA(n_components=2)
                X = pca.fit_transform(X)
                x_label = 'Главная компонента 1'
                y_label = 'Главная компонента 2'
            else:
                x_label = 'Признак 1'
                y_label = 'Признак 2'

            scatter = ax.scatter(X[:, 0], X[:, 1] if X.shape[1] > 1 else X[:, 0],
                                 c=labels, cmap='viridis', alpha=0.6)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label if X.shape[1] > 1 else '')

        ax.set_title(title)
        figure.colorbar(scatter, ax=ax, label='Кластер')
        canvas.draw()

    def anonymize_dataframe(self):
        self.loadedDataFrame = self.loadedDataFrame.copy().apply(lambda col:
                                       (col // (col.std() / 3)) * (col.std() / 3)
                                       )
        self.updateTableView()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    applicationWindow = DatasetClusterApp()
    applicationWindow.show()
    sys.exit(app.exec_())
