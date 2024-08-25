import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
from matplotlib.widgets import Widget
import mplfinance as mpf
import matplotlib.dates as mdate

class FigureCursor(Widget):
    """
        Original class from matplotlib.widgets.MultiCursor\n

        Edited for data point snapping and data value printing
    """
    def __init__(self, fig, referenceline, secondplotline, horizOn=True, vertOn=True, numberformat="{0:.4g};{1:.4g}",
                 useblit=True, textprops=None, **lineprops):
        if textprops is None:
            textprops = {}
        self._cidmotion = None
        self._ciddraw = None
        self.background = None
        self.needclear = False
        self.visible = True
        self.visible0 = True
        self.canvas = fig.canvas
        self.fig = fig
        self.referenceline = referenceline
        self.secondplotline = secondplotline
        self.horizOn = horizOn
        self.vertOn = vertOn
        self.useblit = useblit
        self.vline, = fig.axes[0].plot([.5, .5], [0., 1.], visible=vertOn, transform=self.fig.transFigure,
                                       clip_on = False, **lineprops)
        self.hline, = fig.axes[0].plot([0., 1.], [.5, .5], visible=horizOn, transform=self.fig.transFigure,
                                       clip_on=False, **lineprops)
        self.set_position(self.referenceline.get_xdata()[0])
        self.lastdrawnplotpoint = None
        self.offset = np.array((0, 0))
        self.numberformat = numberformat
        self.text = self.fig.axes[0].text(
            .01, .99,
            "0, 0",
            ha='left', va='top',
            transform=self.fig.axes[0].transAxes,
            animated=bool(self.useblit),
            visible=True, **textprops)
        self.connect()

    def connect(self):
        """connect events"""
        self._cidmotion = self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self._ciddraw = self.canvas.mpl_connect('draw_event', self.clear)

    def disconnect(self):
        """disconnect events"""
        self.canvas.mpl_disconnect(self._cidmotion)
        self.canvas.mpl_disconnect(self._ciddraw)

    def clear(self, event):
        """clear the cursor"""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = (
                self.canvas.copy_from_bbox(self.canvas.figure.bbox))
        for line in [self.vline, self.hline]:
            line.set_visible(False)

    def onmove(self, event):
        if self.ignore(event):
            self.lastdrawnplotpoint = None
            return
        if event.inaxes is None:
            self.lastdrawnplotpoint = None
            return
        if not self.canvas.widgetlock.available(self):
            self.lastdrawnplotpoint = None
            return
        if not self.visible:
            return
        plotpoint = None
        if event.xdata is not None and event.ydata is not None:
            # Get plot point related to current x position.
            # These coordinates are displayed in text.
            plotpoint = self.set_position(event.xdata)
            # Modify event, such that the cursor is displayed on the
            # plotted line, not at the mouse pointer,
            # if the returned plot point is valid
            if plotpoint is not None:
                event.xdata = plotpoint[0]
                event.ydata = plotpoint[1]
        if plotpoint is not None and plotpoint == self.lastdrawnplotpoint:
            return
        if plotpoint is not None:
            # Update position and displayed text.
            # Position: Where the event occurred.
            # Text: Determined by set_position() method earlier
            # Position is transformed to pixel coordinates,
            # an offset is added there and this is transformed back.
            temp = [event.xdata, event.ydata]
            temp = self.fig.axes[0].transData.transform(temp)
            temp = temp + self.offset
            temp = self.fig.axes[0].transData.inverted().transform(temp)
            #self.text.set_position(temp)
            self.text.set_text(self.numberformat.format(*plotpoint))
            self.text.set_visible(self.visible)

            # Tell base class, that we have drawn something.
            # Baseclass needs to know, that it needs to restore a clean
            # background, if the cursor leaves our figure context.
            self.needclear = True

            # Remember the recently drawn cursor position, so events for the
            # same position (mouse moves slightly between two plot points)
            # can be skipped
            self.lastdrawnplotpoint = plotpoint
        # otherwise, make text invisible
        else:
            self.text.set_visible(False)

        trans = event.inaxes.transData + self.fig.transFigure.inverted()
        x_fig, y_fig = trans.transform([event.xdata, event.ydata])
        if self.vertOn:
            self.vline.set_xdata([x_fig, x_fig])
            self.vline.set_visible(self.visible)
        if self.horizOn:
            self.hline.set_ydata([y_fig, y_fig])
            self.hline.set_visible(self.visible)
        self._update()

    def set_position(self, xpos):

        xdata = self.referenceline.get_xdata()
        y0data = self.referenceline.get_ydata()
        y1data = self.secondplotline.get_ydata()

        pos = xpos
        data = xdata
        lim = self.fig.axes[0].get_xlim()

        if pos is not None and lim[0] <= pos <= lim[-1]:
            index = np.searchsorted(data, pos)

            if index < 0 or index >= len(data):
                return None

            return (xdata[index], y0data[index], y1data[index])

    def _update(self):
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            if self.vertOn:
                self.fig.draw_artist(self.vline)
            if self.horizOn:
                self.fig.draw_artist(self.hline)
            self.fig.axes[0].draw_artist(self.text)
            self.canvas.blit(self.canvas.figure.bbox)
        else:
            self.canvas.draw_idle()


class TkinterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tkinter Application")

        # Initialize variables for plots
        self.fig1 = Figure(figsize=(5, 4), dpi=100,facecolor='black')
        self.ax1, self.ax2 = self.fig1.subplots(nrows=2, sharex=True)
        self.ax1.set_title("Plot 1")
        self.ax2.set_title("Plot 2")
        self.ax1.set_facecolor('#041105')
        self.ax2.set_facecolor('#041105')
        self.ax1.grid(color='lightgray', linewidth=.5, linestyle=':')
        self.ax2.grid(color='lightgray', linewidth=.5, linestyle=':')

        self.ax1.set_position([0.02, 0.37, 0.88, 0.6])
        self.ax2.set_position([0.02, 0.15, 0.88, 0.22])
        self.ax1.tick_params(axis='both', color='#ffffff', labelcolor='#ffffff')
        self.ax1.yaxis.tick_right()
        self.ax2.tick_params(axis='both', color='#ffffff', labelcolor='#ffffff')
        self.ax2.yaxis.tick_right()
        self.ax1.autoscale_view()
        self.ax2.autoscale_view()

        # Create canvas for matplotlib plots
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=root)
        self.canvas1.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=2, sticky="nsew")

        # Add navigation toolbar
        self.toolbar_frame1 = tk.Frame(master=root)
        self.toolbar_frame1.grid(row=2, column=0, sticky="ew")
        self.toolbar1 = NavigationToolbar2Tk(self.canvas1, self.toolbar_frame1)
        self.toolbar1.update()

        # Button for choosing CSV file
        self.choose_file_button = tk.Button(root, text="Choose CSV File", command=self.load_csv)
        self.choose_file_button.grid(row=0, column=2)

        # Logger in the fourth row
        self.logger = tk.Text(root)
        self.logger.grid(row=4, column=0, columnspan=4, sticky="nsew")
        self.logger.config(state="disabled")

        # Configure row and column weights for resizing
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=0)  # Row with toolbar
        root.grid_rowconfigure(3, weight=0)  # Row with toolbar
        root.grid_rowconfigure(4, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=0)  # Column with button
        root.grid_columnconfigure(3, weight=0)  # Spacer column

    def log(self, message):
        self.logger.config(state="normal")
        self.logger.insert(tk.END, f"{message}\n")
        self.logger.see(tk.END)
        self.logger.config(state="disabled")

    """def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            # Read the CSV file
            data = pd.read_csv(file_path)
            for eachdate in data['Date']:
                eachdate = mdate.num2date(eachdate/86400000).isoformat()
                print(eachdate)
            self.log(f"load_csv data:\n\n{data}")
            # Create a candlestick chart
            ohlc = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', ]]
            ohlc['Date'] = mdate.num2date(ohlc['Date']/86400000)
            #ohlc['Date'] = pd.to_datetime(ohlc['Date'])
            ohlc.set_index('Date', inplace=True)
            self.log(f"load_csv data:\n\n{data}")
            # Update plots with candlestick chart
            self.ax1.clear()
            mpf.plot(ohlc, type='candle', ax=self.ax1, volume=self.ax2, show_nontrading=True)
            self.ax1.set_title("Candlestick Chart")

            # Enable FigureCursor
            self.cursor1 = FigureCursor(self.fig1,
                                        horizOn=False,
                                        vertOn=True,
                                        numberformat="Date: {0}\nOpen: {1:.2f}\nHigh: {2:.2f}\nLow: {3:.2f}\nClose: {4:.2f}",
                                        referenceline=self.ax1.get_lines()[0],
                                        # Assuming only one line in the candlestick chart
                                        secondplotline=self.ax2.get_lines()[0],  # Assuming only one line in the volume chart
                                        color='w',
                                        textprops={'color': 'w', 'fontweight': 'bold'},
                                        lw=1,
                                        ls='dashed')

            # Update logger with file information
            self.log(f"CSV file loaded: {file_path}")
            self.canvas1.draw()
            self.root.update_idletasks()"""

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Update plots with data from CSV
            # Replace the following lines with your specific plotting logic
            self.ax1.clear()
            self.secondplotline = self.ax1.plot(data['X'], data['Y1'])
            self.ax1.set_title("Plot 1")

            self.ax2.clear()
            self.line2 = self.ax2.plot(data['X'], data['Y2'])
            self.ax2.set_title("Plot 2")

            self.ax1.grid(color='lightgray', linewidth=.5, linestyle=':')
            self.ax2.grid(color='lightgray', linewidth=.5, linestyle=':')

            # Enable MultiCursor variant FigureCursor
            self.cursor1 = FigureCursor(self.fig1,
                                        horizOn=False,
                                        vertOn=True,
                                        numberformat="x:{0:.2f}\ny1:{1:.2f}\ny2:{2:.2f}",
                                        referenceline=self.secondplotline[0],
                                        secondplotline=self.line2[0],
                                        color='w',
                                        textprops={'color': 'w', 'fontweight': 'bold'},
                                        lw=1,
                                        ls='dashed')
            # Update logger with file information
            self.log(f"CSV file loaded: {file_path}")
            self.canvas1.draw()
            self.root.update_idletasks()


root = tk.Tk()
app = TkinterApp(root)
root.mainloop()
