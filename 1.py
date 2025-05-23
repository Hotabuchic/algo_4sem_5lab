import hashlib
import sys
import pandas as pd
import numpy as np
import time
from scipy.io import arff
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QFormLayout, QPlainTextEdit,
    QPushButton, QLineEdit, QFileDialog, QTableView,
    QSizePolicy, QHeaderView, QSpacerItem, QMessageBox
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from sklearn.cluster import KMeans
from sklearn.metrics import matthews_corrcoef, confusion_matrix, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder, StandardScaler

from isodata import FastISODATA, optimized_spa


class DatasetClusterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Кластеризация данных")
        self.resize(900, 350)

        self.loadedDataFrame = pd.DataFrame()
        self.initializeUserInterface()

    def initializeUserInterface(self):
        mainLayout = QHBoxLayout()
        self.setLayout(mainLayout)

        controlPanel = QWidget()
        controlLayout = QFormLayout()
        controlPanel.setLayout(controlLayout)

        self.buttonLoadCsv = QPushButton("Загрузить датасет")
        self.buttonLoadCsv.clicked.connect(self.handleLoad)
        controlLayout.addRow(self.buttonLoadCsv)

        self.buttonDeidentify = QPushButton("Обезличить датасет")
        self.buttonDeidentify.clicked.connect(self.anonymize_dataframe)
        controlLayout.addRow(self.buttonDeidentify)

        self.buttonRunClustering = QPushButton("Запустить кластеризацию")
        self.buttonRunClustering.clicked.connect(self.handleRunClustering)
        controlLayout.addRow(self.buttonRunClustering)

        self.inputFeatureCount = QLineEdit("3")
        controlLayout.addRow("Число признаков:", self.inputFeatureCount)

        self.textStatusOutput = QPlainTextEdit()
        self.textStatusOutput.setReadOnly(True)
        self.textStatusOutput.setStyleSheet("background-color: #f0f0f0;")
        self.textStatusOutput.setFixedHeight(180)
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
        res = model_full.fit_predict(X_scaled, y)
        print(res)

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
        if self.loadedDataFrame.empty:
            QMessageBox.warning(self, "Ошибка", "Датасет не загружен.")
            return

        anonymized = self.loadedDataFrame.copy()

        numeric_cols = anonymized.select_dtypes(include=np.number).columns.tolist()

        for col in numeric_cols:
            # Пропорциональный шум (10% от std)
            std = anonymized[col].std()
            noise = np.random.normal(0, std * 0.1, size=len(anonymized))
            anonymized[col] = anonymized[col] + noise

            # Особые правила для определенных колонок
            if 'BALANCE' in col or 'PURCHASES' in col or 'CASH_ADVANCE' in col:
                # Логарифмирование для больших значений
                anonymized[col] = np.log1p(anonymized[col])

            elif 'FREQUENCY' in col or 'TRX' in col:
                # Округление частотных показателей
                anonymized[col] = anonymized[col].round(0)

            elif 'CREDIT_LIMIT' in col or 'PAYMENTS' in col:
                # Биннинг чувствительных финансовых данных
                anonymized[col] = pd.qcut(anonymized[col], q=5, labels=False, duplicates='drop')

        self.loadedDataFrame = anonymized.fillna(0)
        self.updateTableView()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    applicationWindow = DatasetClusterApp()
    applicationWindow.show()
    sys.exit(app.exec_())
