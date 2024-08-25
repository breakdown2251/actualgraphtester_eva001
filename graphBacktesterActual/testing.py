import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta

# Example data
# First time series (daily frequency)
dates1 = pd.date_range(start='2023-01-01', periods=100, freq='D')
values1 = [i * 2 for i in range(100)]

# Second time series (weekly frequency)
dates2 = pd.date_range(start='2023-01-01', periods=15, freq='W')
values2 = [i * 3 for i in range(15)]

# Create traces
trace1 = go.Scatter(
    x=dates1,
    y=values1,
    mode='lines+markers',
    name='Daily Data'
)

trace2 = go.Scatter(
    x=dates2,
    y=values2,
    mode='lines+markers',
    name='Weekly Data'
)

# Create the figure and add traces
fig = go.Figure()
fig.add_trace(trace1)
fig.add_trace(trace2)

# Update layout
fig.update_layout(
    title='Time Series with Different Frequencies',
    xaxis_title='Date',
    yaxis_title='Value',
    legend_title='Series',
    hovermode='x'
)

# Show the plot
fig.show()
