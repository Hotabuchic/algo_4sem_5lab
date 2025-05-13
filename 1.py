import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QFormLayout, QPlainTextEdit,
    QPushButton, QLineEdit, QFileDialog, QTableView,
    QSizePolicy, QHeaderView, QMessageBox
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from isodata import FastISODATA, optimized_spa


class DatasetClusterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Кластеризация")
        self.resize(900, 350)

        self.loadedDataFrame = pd.DataFrame()
        self.initializeUserInterface()

    def initializeUserInterface(self):
        mainLayout = QHBoxLayout()
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

        self.inputFeatureCount = QLineEdit("3")
        controlLayout.addRow("Число признаков:", self.inputFeatureCount)

        self.textStatusOutput = QPlainTextEdit()
        self.textStatusOutput.setReadOnly(True)
        controlLayout.addRow(self.textStatusOutput)

        self.tableViewData = QTableView()
        self.tableViewData.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mainLayout.addWidget(self.tableViewData)
        mainLayout.addWidget(controlPanel)

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
        X = self.loadedDataFrame.copy().fillna(0)

        # Генерация искусственных меток для оценки
        kmeans = KMeans(n_clusters=5, random_state=42)  # Фиксированное число кластеров
        y = kmeans.fit_predict(X)

        # Масштабирование данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_full = FastISODATA()
        model_full.fit_predict(X_scaled, y)

        best_feats, best_ari, best_ri = optimized_spa(X_scaled, y, n_iter=50, n_features=n_features, p=1.3)

        X_selected = X_scaled[:, best_feats]
        model_selected = FastISODATA()
        model_selected.fit_predict(X_selected, y)

        self.textStatusOutput.setPlainText(
            f"Кластеризация на {n_features} признаках\n"
            f"Индекс Rand без отбора признаков: {model_full.ri:.4f}\n"
            f"Индекс ARI без отбора признаков: {model_full.ari:.4f}\n"
            f"Индекс Rand с отбором признаков: {model_selected.ri:.4f}\n"
            f"Индекс ARI без отбора признаков: {model_selected.ari:.4f}\n"
        )

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
