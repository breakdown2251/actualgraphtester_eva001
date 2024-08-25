import sys
import plotlyplot
from PyQt5.QtWidgets import QApplication, QTableView, QWidget, QVBoxLayout, QPushButton, QComboBox, QSlider, \
    QHBoxLayout, QLabel, QLineEdit, QFileDialog, QCheckBox
from PyQt5.QtWebEngineCore import QWebEngineUrlScheme, QWebEngineUrlSchemeHandler, QWebEngineUrlRequestJob

from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QAbstractTableModel, Qt, QByteArray, QBuffer, QIODevice, QUrl
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import os

HTML_DATA = {}
URL_SCHEME = 'local'


class AuroSera(QWidget):
    def __init__(self):
        super(AuroSera, self).__init__()

        self.base_path_daily = r"C:\Users\User\Desktop\data\binance csvs\python\data\spot\daily\klines\ETHUSDT\5m"
        self.base_path_monthly = r"C:\Users\User\Desktop\data\binance csvs\python\data\spot\monthly\klines\ETHUSDT\5m"
        self.date_string = "YYYY-MM-DD"
        self.current_date: datetime

        self.file_prefix = ["BTCUSDT", "15m"]  # its gonna updated on load anyway

        self.setWindowTitle("AuroSera")
        self.setGeometry(0, 0, 1500, 1000)


        self.top_webview = QWebEngineView()
        self.bottom_webview = QWebEngineView()
        # UrlSchemeHandler to break chromium 2mb local .html limit
        self.top_webview.page().profile().installUrlSchemeHandler(bytes(URL_SCHEME, 'ascii'), UrlSchemeHandler(self))
        self.top_webview.loadFinished.connect(self.handleLoaded)

        self.call_ui()
        self.read_first_date_file()
        self.show()

        self.set_webview_top('htmlplot_cdn.html')
        self.set_webview_bottom('nodata.html')

    def call_ui(self):
        # Create buttons, combo box, and slider
        self.refresh_button = QPushButton("Refresh And Apply", self)
        self.df_call_button = QPushButton("Show Current DF Table", self)
        self.period_back_button = QPushButton("<", self)
        self.period_label = QLabel(f"File: {self.date_string}", self)
        self.period_forward_button = QPushButton(">", self)
        self.graph_mode_monthly = QCheckBox("Monthly Mode", self)

        self.period_back_button.setFixedWidth(30)
        self.period_label.setAlignment(Qt.AlignCenter)
        self.period_forward_button.setFixedWidth(30)

        # Create text boxes for year, month, and day
        self.year_edit = QLineEdit(self)
        self.month_edit = QLineEdit(self)
        self.day_edit = QLineEdit(self)
        self.go_to_date_button = QPushButton('Go', self)

        self.folder_edit_daily = QLineEdit(self)
        self.folder_edit_daily.setReadOnly(True)
        self.folder_edit_daily.setText(self.base_path_daily)
        self.folder_query_button_d = QPushButton('Check Daily Folder', self)

        self.folder_edit_monthly = QLineEdit(self)
        self.folder_edit_monthly.setReadOnly(True)
        self.folder_edit_monthly.setText(self.base_path_monthly)
        self.folder_query_button_m = QPushButton('Check Monthly Folder', self)

        self.folder_query_button_d.setFixedWidth(140)
        self.folder_query_button_m.setFixedWidth(140)

        self.year_edit.setText("2023")
        self.month_edit.setText("01")
        self.day_edit.setText("01")

        self.year_edit.setInputMask("0000")
        self.month_edit.setInputMask("00")
        self.day_edit.setInputMask("00")

        self.year_edit.setFixedWidth(50)
        self.month_edit.setFixedWidth(30)
        self.day_edit.setFixedWidth(30)
        self.go_to_date_button.setFixedWidth(40)

        combo_box = QComboBox()
        slider = QSlider()

        # Linking events
        self.df_call_button.clicked.connect(self.call_df_window)
        self.refresh_button.clicked.connect(self.refresh_webviews)

        self.period_forward_button.clicked.connect(self.read_next_date_file)
        self.period_back_button.clicked.connect(self.read_previous_date_file)

        self.go_to_date_button.clicked.connect(
            lambda: self.go_to_date())
        self.graph_mode_monthly.toggled.connect(
            lambda: self.go_to_date())

        self.folder_query_button_d.clicked.connect(
            lambda : self.open_file_dialog(self.folder_edit_daily, var_edit="d"))
        self.folder_query_button_m.clicked.connect(
            lambda: self.open_file_dialog(self.folder_edit_monthly, var_edit="m"))

        # Create layouts
        date_picker_layout = QHBoxLayout()
        date_picker_layout.addWidget(self.year_edit)
        date_picker_layout.addWidget(self.month_edit)
        date_picker_layout.addWidget(self.day_edit)
        date_picker_layout.addWidget(self.go_to_date_button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.top_webview)
        #left_layout.addWidget(self.bottom_webview)

        date_layout = QHBoxLayout()
        date_layout.addWidget(self.period_back_button)
        date_layout.addWidget(self.period_label)
        date_layout.addWidget(self.period_forward_button)

        right_layout = QVBoxLayout()
        right_layout.addLayout(date_layout)
        right_layout.addWidget(self.refresh_button)
        right_layout.addWidget(self.df_call_button)
        right_layout.addWidget(self.graph_mode_monthly)
        right_layout.addLayout(date_picker_layout)
        right_layout.addWidget(slider)
        
        daily_folder_layout = QHBoxLayout()
        daily_folder_layout.addWidget(self.folder_edit_daily)
        daily_folder_layout.addWidget(self.folder_query_button_d)
        monthly_folder_layout = QHBoxLayout()
        monthly_folder_layout.addWidget(self.folder_edit_monthly)
        monthly_folder_layout.addWidget(self.folder_query_button_m)

        folder_layout = QVBoxLayout()
        folder_layout.addLayout(daily_folder_layout)
        folder_layout.addLayout(monthly_folder_layout)
        
        view_layout = QHBoxLayout()
        view_layout.addLayout(left_layout)
        view_layout.addLayout(right_layout)
        
        base_layout = QVBoxLayout()
        base_layout.addLayout(folder_layout)
        base_layout.addLayout(view_layout)
        
        self.setLayout(base_layout)

    def set_webview_top(self, html_path):
        """html = Path(html_path).read_text(encoding="utf8")
        self.top_webview.setHtml(html)"""
        HTML_DATA[html_path] = open(html_path).read()
        url = QUrl(html_path)
        url.setScheme(URL_SCHEME)
        self.top_webview.setUrl(url)

    def set_webview_bottom(self, html_path):
        html = Path(html_path).read_text(encoding="utf8")
        self.bottom_webview.setHtml(html)

    def call_df_window(self):
        #self.data = plotlyplot.CsvToHtmlPlot(saveThePlot=True)
        #self.refresh_webviews()
        self.df_window = DataFrameWindow(self.df.return_dataframe())
        print("DF WINDOW")

    def refresh_webviews(self):
        self.set_webview_top('htmlplot_cdn.html')
        self.set_webview_bottom('nodata.html')
        print("REFRESH\n")

    def get_file_path(self, target_date):
        if self.graph_mode_monthly.isChecked():
            formatted_date = target_date.strftime("%Y-%m")
            file_name = f"{self.file_prefix[0]}-{self.file_prefix[1]}-{formatted_date}.zip"
            return os.path.join(self.base_path_monthly, file_name)
        else:
            formatted_date = target_date.strftime("%Y-%m-%d")
            file_name = f"{self.file_prefix[0]}-{self.file_prefix[1]}-{formatted_date}.zip"
            return os.path.join(self.base_path_daily, file_name)

    def read_previous_date_file(self):
        current_date = self.current_date
        if self.graph_mode_monthly.isChecked():
            previous_date = current_date - pd.offsets.DateOffset(months=1)
        else:
            previous_date = current_date - timedelta(days=1)
        file_path = self.get_file_path(previous_date)

        if os.path.exists(file_path):
            self.current_date = previous_date
            if self.graph_mode_monthly.isChecked():
                self.period_label.setText(datetime.strftime(self.current_date, '%Y-%m'))
            else:
                self.period_label.setText(datetime.strftime(self.current_date, '%Y-%m-%d'))
            self.df = plotlyplot.CsvToHtmlPlot(saveThePlot=True, filePath=file_path)
            self.refresh_webviews()
            return True
        else:
            print(f"QUERY FALSE:{file_path}")
            return None

    def read_next_date_file(self):
        current_date = self.current_date
        if self.graph_mode_monthly.isChecked():
            next_date = current_date + pd.offsets.DateOffset(months=1)
        else:
            next_date = current_date + timedelta(days=1)
        file_path = self.get_file_path(next_date)

        if os.path.exists(file_path):
            self.current_date = next_date
            if self.graph_mode_monthly.isChecked():
                self.period_label.setText(datetime.strftime(self.current_date, '%Y-%m'))
            else:
                self.period_label.setText(datetime.strftime(self.current_date, '%Y-%m-%d'))
            self.df = plotlyplot.CsvToHtmlPlot(saveThePlot=True, filePath=file_path)
            self.refresh_webviews()
            return True
        else:
            print(f"QUERY FALSE:{file_path}")
            return None

    def read_first_date_file(self):
        print("FILE DIR QUERY")
        # Assuming data files are sorted by date in ascending order
        files = sorted(os.listdir(self.base_path_daily))

        if files:
            print("DATETIME QUERY")
            file = files[0].split(".")
            file = file[0].split("-")
            date = file[2] + "-" + file[3] + "-" + file[4]
            self.file_prefix = [file[0], file[1]]
            print(f"FILE PREFIX {self.file_prefix}")

            self.current_date = datetime.strptime(date, '%Y-%m-%d')

            first_file_path = os.path.join(self.base_path_daily, files[0])
            self.period_label.setText(datetime.strftime(self.current_date, '%Y-%m-%d'))
            print("PLOT QUERY")
            self.df = plotlyplot.CsvToHtmlPlot(saveThePlot=True, filePath=first_file_path)
            self.refresh_webviews()
            return True
        else:
            print("FILE DIR FAILED")
            return None

    def go_to_date(self):
        print("SPECIFIC DATE QUERY")

        date = self.year_edit.text() + "-" + self.month_edit.text() + "-" + self.day_edit.text()
        self.current_date = datetime.strptime(date, '%Y-%m-%d')
        first_file_path = self.get_file_path(self.current_date)
        print(f"\t{self.current_date}")
        if self.graph_mode_monthly.isChecked():
            self.period_label.setText(datetime.strftime(self.current_date, '%Y-%m'))
            if not os.path.isfile(first_file_path):
                raise ValueError(f"FILE NOT EXIST!:requestMonthly{self.graph_mode_monthly.isChecked()}")
            self.df = plotlyplot.CsvToHtmlPlot(saveThePlot=True, filePath=first_file_path)
            self.refresh_webviews()
            return True
        if self.year_edit.hasAcceptableInput() and self.month_edit.hasAcceptableInput() and self.day_edit.hasAcceptableInput():
            self.period_label.setText(datetime.strftime(self.current_date, '%Y-%m-%d'))
            if not os.path.exists(first_file_path):
                raise ValueError(f"FILE NOT EXIST!:requestMonthly{self.graph_mode_monthly.isChecked()}")
            print("PLOT QUERY")
            self.df = plotlyplot.CsvToHtmlPlot(saveThePlot=True, filePath=first_file_path)
            self.refresh_webviews()
            return True
        print("FILE DIR FAILED")
        return None

    def open_file_dialog(self, line_edit, var_edit="d"):
        folder_path = QFileDialog.getExistingDirectory(self, 'Open Folder', self.base_path_daily)
        if folder_path:
            if var_edit is "d":
                self.base_path_daily = folder_path
            if var_edit is "m":
                self.base_path_monthly = folder_path
            line_edit.setText(folder_path)
            self.read_first_date_file()

    def handleLoaded(self, ok):
        if not ok:
            self.view.setHtml('<h3>414: URI Too Long</h3>')


class DataFrameWindow(QWidget):
    def __init__(self, dataframe):
        super(DataFrameWindow, self).__init__()

        self.model = PandasModel(dataframe)
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.resize(1000, 600)
        self.view.setWindowTitle("DATAFRAME OUTPUT")
        self.view.show()


class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class UrlSchemeHandler(QWebEngineUrlSchemeHandler):
    def requestStarted(self, job):
        href = job.requestUrl().path()
        if (data := HTML_DATA.get(href)) is not None:
            if not isinstance(data, bytes):
                data = str(data).encode()
            mime = QByteArray(b'text/html')
            buffer = QBuffer(job)
            buffer.setData(data)
            buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            job.reply(mime, buffer)
        else:
            print(f'ERROR: request job failed: {href!r}')
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)


if __name__ == '__main__':
    scheme = QWebEngineUrlScheme(bytes(URL_SCHEME, 'ascii'))
    scheme.setFlags(QWebEngineUrlScheme.Flag.SecureScheme |
                    QWebEngineUrlScheme.Flag.LocalScheme |
                    QWebEngineUrlScheme.Flag.LocalAccessAllowed)
    QWebEngineUrlScheme.registerScheme(scheme)

    app = QApplication(['Test'])
    app.setStyleSheet(Path('ConsoleStyle.qss').read_text())
    window = AuroSera()
    sys.exit(app.exec_())

"""
# For Kaltsit, my beloved <3
# """
