import numpy as np
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import tanalysis


class CsvToHtmlPlot:
    def __init__(self, saveThePlot=False, filePath="None"):
        self.file_path = filePath
        self.saveThePlot = saveThePlot
        self.df = self.ask_csv_file()
        self.df_1hour = self.resample_to(self.df, 'h')
        self.calculate_dataframe_additions(self.df)
        self.df = self.simulate(self.df)
        self.plotly_run(self.df)
        for index, row in self.df.iterrows():
            if row['dtime']:
                print(f"index:{index}\t"
                      f"date:{row['Date']}\t"
                      f"dtime:{row['dtime']}\t"
                      f"res:{row['deltas']:.2f}\t"
                      )
        print(f"sum: {sum(self.df['deltas'])}")
        #self.testing_algo()

    def testing_algo(self):
        len0 = 7
        len1 = 14
        len2 = 35
        slen = 20
        upColor = 'green'
        downColor = 'red'
        maLengthInput = 14
        maTypeInput = 'simple'
        lbL = 10
        lbR = 1
        plotBullish = True
        plotBear = True

        close = self.df['Close']

        v1 = tanalysis.rsi(close, length=len0)
        v2 = tanalysis.rsi(close, length=len1)
        v3 = tanalysis.rsi(close, length=len2)

        x = v1 - v2
        x1 = v2 - v3

        f = tanalysis.hma(pd.concat([x, x1], axis=1).agg(np.mean, 1), length=slen)
        f2 = tanalysis.ema((v3 - 50) / 2, length=slen)

        col = np.where(f > f.shift(1), upColor, downColor)

        fig = go.Figure()

        # Plot mid and f2
        fig.add_trace(go.Scatter(x=self.df.index, y=[0] * len(self.df), mode='lines', line=dict(color='grey'), name='mid'))
        #fig.add_trace(go.Scatter(x=self.df.index, y=f2, mode='lines', line=dict(color=np.where(f2 > 0, upColor, downColor)),name='f2'))
        fig.add_trace(
            go.Scatter(x=self.df.index, y=f2, mode='lines', line=dict(color='green'),
                       name='f2'))

        # Fill between mid and f2
        fig.add_trace(
            go.Scatter(x=self.df.index, y=f2, mode='lines', fill='tonexty', fillcolor='green',
                       name='Fill'))

        # Plot moving averages
        fig.add_trace(go.Scatter(x=self.df.index, y=tanalysis.ma(f, maLengthInput, maTypeInput), mode='lines',
                                 line=dict(color='#4421f3', width=1), name='Moving Average'))

        # Plot o1 and o2
        fig.add_trace(go.Scatter(x=self.df.index, y=f, mode='lines', line=dict(color='green'), name='o1'))
        fig.add_trace(
            go.Scatter(x=self.df.index, y=f.shift(1), mode='lines', line=dict(color='green'), name='o2', visible='legendonly'))

        # Plotchar for trend shifts and reversals
        fig.add_trace(
            go.Scatter(x=self.df.index, y=np.where((f > f.shift(1)) & (f.shift(1) < -10), -22, np.nan), mode='markers',
                       marker=dict(symbol='triangle-up', size=8, color=upColor), name='MS-RSI Bullish Reversal'))
        fig.add_trace(
            go.Scatter(x=self.df.index, y=np.where((f < f.shift(1)) & (f.shift(1) > 10), 22, np.nan), mode='markers',
                       marker=dict(symbol='triangle-down', size=8, color=downColor), name='MS-RSI Bearish Reversal'))

        # Save the plot to an HTML file
        fig.write_html("plot.html")
    
    def custom_resample(self, group):
        #.apply is depracted by .agg() 
        # function is obsolette
        open_price = group['Open'].iloc[0]
        high_price = group['High'].max()
        low_price = group['Low'].min()
        close_price = group['Close'].iloc[-1]
        return pd.Series([open_price, high_price, low_price, close_price], index=['Open', 'High', 'Low', 'Close'])
    
    def resample_to(self, df, resample_rule='h'):

        df.set_index('Date', inplace=True)
        # Resample to 1-hour intervals and aggregate the data
        df_resampled = df.resample(resample_rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
        })
        # Reset index to make 'Date' a column again
        df.reset_index(inplace=True)
        df_resampled.reset_index(inplace=True)
        return df_resampled

    def check_and_add_header(self, file_path, header):
        try:
            with open(file_path, 'r') as file:
                # Read the first line of the file
                first_line = file.readline().strip()

                # Check if the first line matches the required pattern
                if first_line != header:
                    # If not, add the required line to the top of the file
                    with open(file_path, 'r+') as modified_file:
                        content = modified_file.read()
                        modified_file.seek(0, 0)
                        modified_file.write(f"{header}\n" + content)

            return True  # Operation successful
        except FileNotFoundError:
            return False  # File not found

    def calculate_ma(self, df_data, window_size=7):
        moving_avg = df_data.rolling(window=window_size).mean()
        return moving_avg

    def calculate_atr(self, dataframe, period=14):
        # Ensure the DataFrame has a 'High', 'Low', and 'Close' column
        if not all(col in dataframe.columns for col in ['High', 'Low', 'Close']):
            raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns.")

        print("ATR QUERY")
        # Calculate True Range
        dataframe['High-Low'] = dataframe['High'] - dataframe['Low']
        dataframe['High-PrevClose'] = abs(dataframe['High'] - dataframe['Close'].shift(1))
        dataframe['Low-PrevClose'] = abs(dataframe['Low'] - dataframe['Close'].shift(1))

        dataframe['TrueRange'] = dataframe[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        print("ATR TRUE")
        # Calculate Average True Range
        dataframe['ATR'] = dataframe['TrueRange'].rolling(window=period).mean()

        # Drop intermediate columns
        dataframe.drop(['High-Low', 'High-PrevClose', 'Low-PrevClose', 'TrueRange'], axis=1, inplace=True)
        print("ATR TO DF QUERY")
        return dataframe['ATR']

    def calculate_rsi(self, prices, period=14):
        """
        Calculate the Relative Strength Index (RSI) for a given list of prices.

        Parameters:
        - prices (list): List of closing prices.
        - period (int): Number of periods to consider for RSI calculation. Default is 14.

        Returns:
        - rsi_values (list): List of RSI values for each corresponding price.
        """

        # Ensure there are enough prices to calculate RSI
        if len(prices) <= period:
            raise ValueError("Insufficient data for RSI calculation")

        # Calculate daily price changes
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]

        # Calculate average gains and losses over the specified period
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = []
        for i in range(period):
            rsi_values.append(50.0)

        # Calculate RSI for each subsequent day
        for i in range(period, len(prices)):
            gain = gains[i - 1]
            loss = losses[i - 1]

            # Update average gains and losses using smoothing formula
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

            # Calculate relative strength (RS)
            rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')

            # Calculate RSI using the formula
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)

        return rsi_values

    def ask_csv_file(self):
        if self.file_path == "None":
            print("FILE QUERY")
            self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files, ZIP Files", "*.csv *.zip")])
        if self.file_path:
            print("DATAFRAME QUERY")


            # Read the CSV file
            if self.file_path.endswith(".csv"):
                print("CSV QUERY")
                # Check the CSV file header
                header = "Date,Open,High,Low,Close,Volume,CloseTime,ignore0,NumberOfTrades,ignore1,ignore2,ignore3"
                print("CSV HEADER", self.check_and_add_header(self.file_path, header))

                # New DF
                df = pd.read_csv(
                    self.file_path,
                    usecols=['Date', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'NumberOfTrades'])
                print("CSV TRUE")

            elif self.file_path.endswith(".zip"):
                print(f"{self.file_path} QUERY?")
                # if its zip, its header most likely not edited
                df = pd.read_csv(
                    self.file_path,
                    header=None,
                    names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'CloseTime', 'ignore0', 'NumberOfTrades', 'ignore1', 'ignore2', 'ignore3'],
                    usecols=['Date', 'Open', 'High', 'Low', 'Close'],
                    compression="zip",
                )
                print("ZIP TRUE")

            else:
                print("QUERY FAILED")

            print("DATAFRAME TRUE")
            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
            """ ----UTC0---- """

            return df
        return None

    def standardized_median_proximity(self):
        price_source = self.df['Close']
        lookback_length = 21
        ema_lookback_length = 20
        std_dev_multiplier = 1.0
        upper_val = 2
        lower_val = 1.5
        colored_bars = True
        show_neutral_trend = True
        color_up = '#00ffbb'
        color_down = '#ff1100'

        median_value = price_source.rolling(window=lookback_length).median()
        price_deviation = price_source - median_value
        standard_deviation = price_deviation.rolling(window=45).std()
        normalized_value = price_deviation / (standard_deviation + standard_deviation)  # bar graph

        positive_values = np.where(normalized_value > 0, normalized_value, 0)
        negative_values = np.where(normalized_value < 0, normalized_value, 0)

        upper_boundary = pd.Series(positive_values).ewm(span=lookback_length).mean() + pd.Series(
            positive_values).rolling(window=lookback_length).std() * std_dev_multiplier
        lower_boundary = pd.Series(negative_values).ewm(span=lookback_length).mean() - pd.Series(
            negative_values).rolling(window=lookback_length).std() * std_dev_multiplier
        ema_value = pd.Series(normalized_value).ewm(span=ema_lookback_length).mean()

        # Color Conditions
        color_condition_1 = (normalized_value < 0) & (normalized_value < upper_boundary) & (
                    ema_value > lower_boundary) & (normalized_value > ema_value) & show_neutral_trend
        color_condition_2 = (normalized_value > 0) & (normalized_value > lower_boundary) & (
                    ema_value < upper_boundary) & (normalized_value < ema_value) & show_neutral_trend
        plot_color = np.where(normalized_value > upper_boundary, color_up,
                              np.where(normalized_value < lower_boundary, color_down,
                                       np.where(color_condition_1 | color_condition_2, 'gray',
                                                np.where(normalized_value > 0, color_up, color_down))))

        return normalized_value, upper_boundary, lower_boundary, ema_value, plot_color

    def calculate_dataframe_additions(self, df):
        #df['RSI14'] = self.calculate_rsi(df['Close'], 14)
        #df_1hourly = self.resample_to(df, '1h')
        df['smp_normal'], df['smp_upper'], df['smp_lower'], df['smp_ema'], df['smp_color'] = self.standardized_median_proximity()
        df['RSI14'] = tanalysis.rsi(df['Close'], 14)
        df['smp_upper_ma'] = self.calculate_ma(df['smp_upper'], window_size=2)
        df['smp_lower_ma'] = self.calculate_ma(df['smp_lower'], window_size=2)
        df['MA7'] = self.calculate_ma(df['Close'])
        df['MA5'] = self.calculate_ma(df['Close'], window_size=5)
        df['MA55'] = self.calculate_ma(df['Close'], window_size=55)
        df['MA99'] = self.calculate_ma(df['Close'], window_size=99)
        self.df_1hour['MA99_1H'] = self.calculate_ma(self.df_1hour['Close'], window_size=99)
        df['ATR'] = self.calculate_atr(df,8)

    def simulate(self, data):
        deltas = []
        dtime = []
        buy_nodes = []
        sell_nodes = []
        possible_buys, possible_sells = [], []
        possible_buy, possible_sell = False, False
        bought = pd.Series()
        buy = False
        sell = False
        for this_index, this_row in data.iterrows():
            delta = 0
            passed_time = 0
            buy_nodes.append(np.NaN)
            sell_nodes.append(np.NaN)
            if buy:
                bought = this_row
                buy_nodes[-1] = bought['Close']
                buy = False

            if sell:
                delta = (this_row['Open'] - bought['Close']) / bought['Close']
                passed_time = this_row['Date'] - bought['Date']
                bought = pd.Series()
                sell_nodes[-1] = this_row['Open']
                sell = False

            if deltas:
                """if delta <= 0: #what would happen if none of false positive acts were happened?
                    delta = 0"""
                delta = deltas[-1] * (delta + 1.0)
                deltas.append(delta)

            if not deltas:
                deltas.append(1.0)
            dtime.append(passed_time)
            rsi = this_row['RSI14']
            atr = this_row['ATR']
            open = this_row['Open']
            under_ma, over_ma = False, False
            # TODO: under 99 ma means its falling and not buy!
            #  over 99 means its rising and not sell!

            ###      ###
            ### CORE ###
            ###      ###
            condition_buy = (this_row['smp_ema'] < -0.3 and 
                   this_row['smp_lower'] > -0.9 and
                   #this_row['smp_normal'] < this_row['smp_lower'] and 
                   this_row['Close'] < this_row['MA99'] and
                   (this_row['Close']-this_row['MA99'])/this_row['MA99'] < -0.002
            )
            condition_sell = (this_row['smp_ema'] > 0.2 and 
                    this_row['smp_upper'] > 0.9 and
                    #this_row['smp_normal'] > this_row['smp_upper'] and 
                    this_row['Close'] > this_row['MA99'] and
                   (this_row['Close']-this_row['MA99'])/this_row['MA99'] > 0.002
            )
            ###      ###
            ### CORE ###
            ###      ###


            buy = (condition_buy and 
                   bought.empty)
            sell = (condition_sell and
                    not bought.empty)
            
            possible_buys.append(np.NaN)
            possible_sells.append(np.NaN)

            if possible_buy:
                possible_buys[-1]=this_row['Close']
            if possible_sell:
                possible_sells[-1]=this_row['Close']
            
            possible_buy = (condition_buy and 
                   not buy)
            
            possible_sell = (condition_sell and
                    not sell)
            
        data['possible_buy'], data['possible_sell'] = possible_buys, possible_sells
        data['deltas'], data['dtime'], data['Buy'], data['Sell'] = deltas, dtime, buy_nodes, sell_nodes
        return data

    def plotly_run(self, df):


        fig = make_subplots(rows=3, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02,
                            row_heights=[0.6, 0.2, 0.2])
        ### MAIN FIGURE
        fig.add_trace(go.Candlestick(x=df['Date'],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='Data'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA7'], name='MA7'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA99'], name='MA99'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df_1hour['Date'], y=self.df_1hour['MA99_1H'], name='MA99 1hourly'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA5'], name='MA5'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA55'], name='MA55'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Buy'], name='Buys',
                                 mode='markers',
                                 marker=dict(symbol='star-triangle-down', size=12, color='blue')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Sell'], name='Sells',
                                 mode='markers',
                                 marker=dict(symbol='star-triangle-down', size=12, color='cyan')),
                      row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['possible_buy'], name='Buys(possible)',
                                 mode='markers',
                                 marker=dict(symbol='star-triangle-down', size=12, color='royalblue')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['possible_sell'], name='Sells(possible)',
                                 mode='markers',
                                 marker=dict(symbol='star-triangle-down', size=12, color='darkcyan')),
                      row=1, col=1)
        ### MAIN FIGURE

        ### 2nd FIGURE
        color_up = '#00ffbb'
        color_down = '#ff1100'
        upper_val = 2
        lower_val = 1.5
        """fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI14'], name='RSI14'),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['ATR'], name='ATR14'), row=2, col=1)"""
        fig.add_trace(go.Bar(x=df['Date'], y=df['smp_normal'], marker_color=df['smp_color']), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['smp_upper'], mode='lines', line=dict(color=color_down), name='Upper Boundary'), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['smp_lower'], mode='lines', line=dict(color=color_up), name='Lower Boundary'), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['smp_ema'], mode='lines', line=dict(color='white'), name='Hull MA'), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=df['Date'], y=np.zeros(len(df)), mode='lines', line=dict(color='gray'), name='Zero Line'), row=2, col=1)

        # Fills
        """fig.update_layout(shapes=[
            dict(type='rect', xref='x', yref='y', x0=df.index[0], y0=lower_val, x1=df.index[-1], y1=upper_val,
                 fillcolor='rgba(255, 17, 0, 0.5)', line=dict(color='rgba(255, 17, 0, 0)')),
            dict(type='rect', xref='x', yref='y', x0=df.index[0], y0=-upper_val, x1=df.index[-1], y1=-lower_val,
                 fillcolor='rgba(0, 255, 187, 0.5)', line=dict(color='rgba(0, 255, 187, 0)'))
        ])"""
        ### 2nd FIGURE

        ### 3rd FIGURE
        """fig.add_trace(go.Scatter(x=df['Date'], y=df['deltas'],
                                 name='Simulation',
                                 marker=dict(#color=df['simulation'],
                                 colorscale='Viridis')
                                 ),
                      row=3, col=1)"""
        ### 3rd FIGURE


        fig.update_layout(
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=30, b=5),
            #paper_bgcolor="#b3b6fc",
            template='plotly_dark'
        )
        if self.saveThePlot:
            fig.write_html("htmlplot_cdn.html", include_plotlyjs='cdn')
            return True
        fig.show()
        return True

    def return_dataframe(self):
        return self.df