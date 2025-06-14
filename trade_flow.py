import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import configparser
from datetime import datetime, timedelta
import time
import ccxt
from typing import Dict, List, Tuple
import os

# -----------------------------
# CACHED DATAFRAME HELPERS
# -----------------------------
@st.cache_data
def ohlcv_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

@st.cache_data
def trades_df(trades):
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['volume'] = df['amount'] * df['price']
    return df

def create_time_volume_chart(trades: list) -> go.Figure:
    """Creates a chart showing trading volume trend over time (last 3 seconds)"""
    if not trades:
        return go.Figure()

    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['volume'] = df['amount'] * df['price']
    
    # Get the last 3 seconds of data
    end_time = df['timestamp'].max()
    start_time = end_time - pd.Timedelta(seconds=3)
    recent_trades = df[df['timestamp'] >= start_time]
    
    # Resample to 100ms intervals and calculate cumulative volume
    volume_by_time = recent_trades.resample('100ms', on='timestamp')['volume'].sum()
    
    # Calculate the trend (difference between consecutive points)
    volume_trend = volume_by_time.diff()
    
    fig = go.Figure()
    
    # Add volume trend line
    fig.add_trace(go.Scatter(
        x=volume_trend.index,
        y=volume_trend.values,
        mode='lines',
        name='Volume Trend',
        line=dict(
            color='rgba(0, 150, 255, 0.8)',
            width=2
        ),
        hovertemplate='Time: %{x}<br>Volume Change: %{y:.2f}<extra></extra>'
    ))
    
    # Add zero line for reference
    fig.add_hline(
        y=0,
        line=dict(
            color='gray',
            width=1,
            dash='dash'
        )
    )
    
    fig.update_layout(
        title='Volume Trend (Last 3 Seconds)',
        xaxis_title='Time',
        yaxis_title='Volume Change',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        height=300,
        showlegend=False
    )
    
    return fig

def create_trade_frequency_time_chart(trades: list) -> go.Figure:
    """Creates a chart showing the number of trades per second over the last 3 seconds"""
    if not trades:
        return go.Figure()

    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Get the last 3 seconds of data
    end_time = df['timestamp'].max()
    start_time = end_time - pd.Timedelta(seconds=3)
    recent_trades = df[df['timestamp'] >= start_time]
    
    # Resample to 1s intervals and count trades
    trade_counts = recent_trades.resample('1S', on='timestamp').size()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trade_counts.index,
        y=trade_counts.values,
        mode='lines',
        name='Trade Frequency',
        line=dict(color='yellow', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 255, 0, 0.5)',
        hovertemplate='Time: %{x}<br>Trades: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Trade Frequency (Last 3 Seconds)',
        xaxis_title='Time (seconds)',
        yaxis_title='Number of Trades',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        height=300,
        showlegend=False
    )
    
    return fig

def create_trade_direction_ratio_chart(trades: list) -> go.Figure:
    """Plots the ratio of buy-initiated to sell-initiated trades over time"""
    if not trades:
        return go.Figure()
        
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp')
    df['is_buy'] = (df['side'] == 'buy').astype(int)
    df['is_sell'] = (df['side'] == 'sell').astype(int)
    
    # Only keep numeric columns for rolling sum
    numeric = df[['timestamp', 'is_buy', 'is_sell']].set_index('timestamp')
    ratio = numeric.rolling('1min', min_periods=1).sum()
    ratio['direction_ratio'] = ratio['is_buy'] / (ratio['is_sell'] + 1e-6)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ratio.index,
        y=ratio['direction_ratio'],
        mode='lines',
        name='Buy/Sell Ratio',
        line=dict(color='orange', width=2),
        hovertemplate='Time: %{x}<br>Buy/Sell Ratio: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Trade Direction Ratio (Buy/Sell, 1-min rolling)',
        xaxis_title='Time',
        yaxis_title='Buy/Sell Ratio',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        height=300
    )
    
    return fig

def fetch_market_data(config: configparser.ConfigParser) -> Tuple[List[Dict], List]:
    """Fetch recent trades and OHLCV data from Binance"""
    try:
        # Initialize exchange
        exchange = ccxt.binance({
            'apiKey': config['BINANCE']['API_KEY'],
            'secret': config['BINANCE']['API_SECRET'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 60000
            }
        })
        
        symbol = config['TRADING']['symbol']
        # Set default limit if not specified
        limit = int(config.get('TRADING', 'limit', fallback='1000'))  # Increased limit for better data
        timeframe = config.get('TRADING', 'timeframe', fallback='1m')
        
        # Only keep this info in the sidebar
        # st.sidebar.info(f"Fetching data for {symbol} with {limit} trades")
        
        # Fetch recent trades
        trades = exchange.fetch_trades(symbol, limit=limit)
        
        # Process trades to include order type
        processed_trades = []
        for trade in trades:
            # Determine if trade is aggressive (taker) or passive (maker)
            is_aggressive = False
            if 'takerOrMaker' in trade:
                is_aggressive = trade['takerOrMaker'] == 'taker'
            elif 'maker' in trade:
                is_aggressive = not trade['maker']
            elif 'taker' in trade:
                is_aggressive = trade['taker']
            else:
                is_aggressive = True
            
            # Set order type based on whether trade was aggressive
            order_type = 'market' if is_aggressive else 'limit'
            
            processed_trade = {
                'timestamp': trade['timestamp'],
                'price': float(trade['price']),
                'amount': float(trade['amount']),
                'side': trade['side'],
                'order_type': order_type,
                'cost': float(trade['cost'])
            }
            processed_trades.append(processed_trade)
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not processed_trades:
            st.warning("No trades received from the exchange")
            return [], []
            
        if not ohlcv:
            st.warning("No OHLCV data received from the exchange")
            return [], []
            
        return processed_trades, ohlcv
        
    except ccxt.AuthenticationError as e:
        st.error(f"Authentication error: {str(e)}")
        st.error("Please check your API credentials in config.ini")
        return [], []
    except ccxt.NetworkError as e:
        st.error(f"Network error: {str(e)}")
        st.error("Please check your internet connection")
        return [], []
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.error("Please check your configuration and try again")
        return [], []

def create_trade_flow_matrix(trades, ohlcv_data):
    """
    Creates a quadrant chart showing different aspects of trade flow analysis
    Volume Clusters (bottom left) will be replaced with Rolling Volatility (20 trades)
    Adds average volatility line and annotation for the latest value.
    """
    if not trades or not ohlcv_data:
        return go.Figure()

    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['volume'] = df['amount'] * df['price']
    df = df.sort_values('timestamp')

    # Calculate VWAP
    df['cumulative_volume'] = df['volume'].cumsum()
    df['volume_price'] = df['volume'] * df['price']
    df['cumulative_volume_price'] = df['volume_price'].cumsum()
    df['vwap'] = df['cumulative_volume_price'] / df['cumulative_volume']

    # Create subplots for quadrant
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'High Impact Trades',
            'Price Deviation',
            'Rolling Volatility (20 trades)',
            'Trade Frequency'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # 1. High Impact Trades (Top Left)
    df['price_change'] = df['price'].diff()
    df['price_change_pct'] = (df['price_change'] / df['price'].shift(1)) * 100
    df['rolling_impact'] = df['price_change'].rolling(window=20).mean()
    df['impact_per_volume'] = df['price_change'].abs() / df['volume']
    
    # Normalize trade sizes for better visualization
    max_size = 30  
    min_size = 5   
    
    def normalize_sizes(sizes):
        if len(sizes) == 0:
            return sizes
        min_val = sizes.min()
        max_val = sizes.max()
        if max_val == min_val:
            return [min_size] * len(sizes)
        return ((sizes - min_val) / (max_val - min_val) * (max_size - min_size) + min_size)
    
    positive_impact = df[df['rolling_impact'] > 0]
    negative_impact = df[df['rolling_impact'] < 0]
    
    # Add positive impact trades
    if not positive_impact.empty:
        normalized_sizes = normalize_sizes(positive_impact['volume'])
        fig.add_trace(
            go.Scatter(
                x=positive_impact['volume'],
                y=positive_impact['rolling_impact'],
                mode='markers',
                name='Positive Impact',
                marker=dict(
                    color='yellow',
                    size=normalized_sizes,
                    line=dict(color='black', width=1),
                    opacity=0.7
                ),
                hovertemplate=(
                    '<b>Positive Impact Trade</b><br>' +
                    'Volume: %{x:.2f}<br>' +
                    'Price Impact: %{y:.8f}<br>' +
                    'Impact per Volume: %{customdata[0]:.8f}<br>' +
                    'Price Change: %{customdata[1]:.2f}%<br>'
                ),
                customdata=list(zip(
                    positive_impact['impact_per_volume'],
                    positive_impact['price_change_pct']
                ))
            ),
            row=1, col=1
        )
    
    # Add negative impact trades
    if not negative_impact.empty:
        normalized_sizes = normalize_sizes(negative_impact['volume'])
        fig.add_trace(
            go.Scatter(
                x=negative_impact['volume'],
                y=negative_impact['rolling_impact'],
                mode='markers',
                name='Negative Impact',
                marker=dict(
                    color='red',
                    size=normalized_sizes,
                    line=dict(color='black', width=1),
                    opacity=0.7
                ),
                hovertemplate=(
                    '<b>Negative Impact Trade</b><br>' +
                    'Volume: %{x:.2f}<br>' +
                    'Price Impact: %{y:.8f}<br>' +
                    'Impact per Volume: %{customdata[0]:.8f}<br>' +
                    'Price Change: %{customdata[1]:.2f}%<br>'
                ),
                customdata=list(zip(
                    negative_impact['impact_per_volume'],
                    negative_impact['price_change_pct']
                ))
            ),
            row=1, col=1
        )
    
    fig.add_hline(
        y=0,
        line=dict(color='gray', width=1, dash='dash'),
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text='Trade Volume',
        row=1, col=1,
        gridcolor='#333333',
        showgrid=True,
        zeroline=True,
        zerolinecolor='#666666',
        zerolinewidth=1
    )
    fig.update_yaxes(
        title_text='Price Impact (Points)',
        row=1, col=1,
        gridcolor='#333333',
        showgrid=True,
        zeroline=True,
        zerolinecolor='#666666',
        zerolinewidth=1
    )

    # 2. Price Deviation (Top Right)
    df['price_deviation'] = ((df['price'] - df['vwap']) / df['vwap']) * 100
    df['time_window'] = list(range(len(df)))
    
    fig.add_trace(
        go.Scatter(
            x=df['time_window'],
            y=df['price_deviation'],
            mode='markers',
            name='Price Deviation',
            marker=dict(
                color=df['price_deviation'],
                colorscale='Viridis',
                size=8,
                showscale=True,
                colorbar=dict(
                    title='% Deviation',
                    thickness=15,
                    len=0.5,
                    y=0.8
                )
            ),
            hovertemplate='Time Window: %{x}<br>Deviation: %{y:.2f}%<br>'
        ),
        row=1, col=2
    )

    # 3. Rolling Volatility (Bottom Left)
    df['price_change'] = df['price'].diff()
    df['volatility'] = df['price_change'].rolling(window=20).std()
    # Average volatility (ignoring NaN)
    avg_vol = df['volatility'].mean(skipna=True)
    # Last volatility value and timestamp
    last_valid_idx = df['volatility'].last_valid_index()
    last_vol = df['volatility'].iloc[-1] if last_valid_idx is not None else None
    last_time = df['timestamp'].iloc[-1] if last_valid_idx is not None else None
    # Volatility line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['volatility'],
            mode='lines',
            name='Volatility',
            line=dict(color='purple', width=2),
            hovertemplate='Time: %{x}<br>Volatility: %{y:.8f}<extra></extra>'
        ),
        row=2, col=1
    )
    # Average volatility horizontal line
    if avg_vol is not None and not np.isnan(avg_vol):
        fig.add_hline(y=avg_vol, line_dash="dot", line_color="orange", line_width=2, row=2, col=1)
    # Annotate last volatility value
    if last_vol is not None and not np.isnan(last_vol) and last_time is not None:
        fig.add_trace(
            go.Scatter(
                x=[last_time],
                y=[last_vol],
                mode='markers+text',
                marker=dict(color='lime', size=12),
                text=[f'{last_vol:.4f}'],
                textposition='middle right',
                showlegend=False
            ),
            row=2, col=1
        )

    # 4. Trade Frequency (Bottom Right)
    trade_freq = df.resample('1min', on='timestamp').size()
    
    fig.add_trace(
        go.Scatter(
            x=trade_freq.index,
            y=trade_freq.values,
            mode='lines',
            name='Trade Frequency',
            line=dict(color='yellow', width=2),
            fill='tozeroy',
            hovertemplate='Time: %{x}<br>Trades: %{y}<br>'
        ),
        row=2, col=2
    )

    fig.update_layout(
        title={
            'text': 'Trade Flow Matrix',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        showlegend=False,
        height=900,
        width=None,
        plot_bgcolor='rgb(0, 0, 0)',
        paper_bgcolor='rgb(0, 0, 0)',
        font=dict(color='white', size=12),
        margin=dict(l=50, r=50, t=100, b=50)
    )

    fig.update_xaxes(
        gridcolor='#333333',
        showgrid=True,
        title_font=dict(size=14),
        tickfont=dict(size=12)
    )
    fig.update_yaxes(
        gridcolor='#333333',
        showgrid=True,
        title_font=dict(size=14),
        tickfont=dict(size=12)
    )

    fig.update_xaxes(title_text='Volume', row=1, col=1)
    fig.update_yaxes(title_text='Price Impact', row=1, col=1)
    
    fig.update_xaxes(title_text='Time Window', row=1, col=2)
    fig.update_yaxes(title_text='% Deviation from VWAP', row=1, col=2)
    
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_yaxes(title_text='Volatility', row=2, col=1)
    
    fig.update_xaxes(title_text='Time', row=2, col=2)
    fig.update_yaxes(title_text='Number of Trades', row=2, col=2)

    return fig

def main():
    st.set_page_config(page_title="Trade Flow Matrix", layout="wide")
    
    # Read config
    config = configparser.ConfigParser()
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.ini')
        
        if not os.path.exists(config_path):
            st.error(f"Config file not found at: {config_path}")
            return
            
        config.read(config_path)
        if not config.has_section('BINANCE') or not config.has_section('TRADING'):
            st.error("Invalid config.ini file. Missing required sections.")
            return
            
        # Display trading pair info in sidebar
        st.sidebar.info(f"Trading pair: {config['TRADING']['symbol']}")
        
    except Exception as e:
        st.error(f"Error reading config.ini: {str(e)}")
        return
    
    st.title("Trade Flow Matrix Analysis")
    
    # Sidebar controls
    st.sidebar.title("Display Settings")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 60, 5)
    
    # Fetch real market data
    trades, ohlcv_data = fetch_market_data(config)
    
    if not trades:
        st.error("No trade data available. Please check your API credentials and connection.")
        return
    
    # Create and display the main chart
    fig = create_trade_flow_matrix(trades, ohlcv_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main() 
