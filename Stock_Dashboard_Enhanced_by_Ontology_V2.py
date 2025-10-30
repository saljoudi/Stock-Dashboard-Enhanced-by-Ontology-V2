#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from yahooquery import Ticker
import plotly.graph_objs as go
import pandas as pd
import ta
import dash_bootstrap_components as dbc
import warnings
from datetime import datetime, timedelta
import numpy as np
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  ENHANCED Stock Analysis Ontology
# ─────────────────────────────────────────────
class EnhancedStockAnalysisOntology:
    def __init__(self):
        # Comprehensive indicator relationships
        self.indicators = {
            'trend': {
                'SMA': {
                    'periods': [5, 20, 50, 200],
                    'relationships': {
                        'golden_cross': {'fast': 50, 'slow': 200},
                        'death_cross': {'fast': 50, 'slow': 200}
                    },
                    'significance': 'Primary trend identification'
                },
                'EMA': {
                    'periods': [8, 21, 55, 200],
                    'significance': 'Trend direction with recent price emphasis'
                },
                'MACD': {
                    'components': {'fast': 12, 'slow': 26, 'signal': 9},
                    'signals': {
                        'bullish_crossover': 'MACD crosses above signal line',
                        'bearish_crossover': 'MACD crosses below signal line',
                        'zero_line_cross': 'MACD crosses zero line',
                        'divergence': 'Price and MACD move in opposite directions'
                    }
                },
                'ADX': {
                    'strength_levels': {
                        'weak': (0, 25),
                        'strong': (25, 50),
                        'very_strong': (50, 75),
                        'extreme': (75, 100)
                    },
                    'components': ['ADX', 'DI+', 'DI-']
                },
                'Ichimoku': {
                    'components': {
                        'Tenkan_sen': 'Conversion Line (9 periods)',
                        'Kijun_sen': 'Base Line (26 periods)',
                        'Senkou_span_a': 'Leading Span A',
                        'Senkou_span_b': 'Leading Span B',
                        'Chikou_span': 'Lagging Span'
                    },
                    'signals': {
                        'cloud_breakout': 'Price breaks through cloud',
                        'cloud_support': 'Price finds support in cloud',
                        'kumo_twist': 'Cloud color change'
                    }
                }
            },
            'momentum': {
                'RSI': {
                    'levels': {'oversold': 30, 'neutral': 50, 'overbought': 70},
                    'divergence_types': ['regular_bearish', 'regular_bullish', 'hidden_bearish', 'hidden_bullish'],
                    'timeframes': [6, 14, 21]
                },
                'Stochastic': {
                    'levels': {'oversold': 20, 'overbought': 80},
                    'signals': {
                        'crossover': '%K crosses %D',
                        'divergence': 'Price and oscillator divergence'
                    }
                },
                'CCI': {
                    'levels': {'oversold': -100, 'overbought': 100},
                    'significance': 'Cyclical trends identification'
                },
                'Williams_R': {
                    'levels': {'oversold': -80, 'overbought': -20},
                    'significance': 'Overbought/oversold levels'
                }
            },
            'volatility': {
                'Bollinger_Bands': {
                    'period': 20,
                    'std_dev': 2,
                    'signals': {
                        'squeeze': 'Bands contract indicating low volatility',
                        'expansion': 'Bands expand indicating high volatility',
                        'walking_the_bands': 'Price rides upper/lower band'
                    }
                },
                'ATR': {
                    'period': 14,
                    'significance': 'Volatility measurement for stop-loss placement'
                },
                'Keltner_Channel': {
                    'components': {'ema': 20, 'atr': 10},
                    'significance': 'Volatility-based channel'
                }
            },
            'volume': {
                'OBV': {
                    'significance': 'Volume-price relationship',
                    'patterns': ['confirmation', 'divergence']
                },
                'VWAP': {
                    'significance': 'Intraday benchmark price',
                    'signals': {
                        'above_vwap_bullish': 'Price above VWAP',
                        'below_vwap_bearish': 'Price below VWAP'
                    }
                },
                'CMF': {
                    'levels': {'bullish': 0.05, 'bearish': -0.05},
                    'significance': 'Money flow strength'
                },
                'MFI': {
                    'levels': {'oversold': 20, 'overbought': 80},
                    'significance': 'Volume-weighted RSI'
                },
                'ADL': {
                    'significance': 'Accumulation/Distribution measurement',
                    'confirmation': 'Confirms price trends'
                }
            },
            'support_resistance': {
                'Pivot_Points': {
                    'levels': ['S3', 'S2', 'S1', 'Pivot', 'R1', 'R2', 'R3'],
                    'calculation_method': 'Standard'
                },
                'Fibonacci': {
                    'levels': [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
                    'significance': 'Natural retracement levels'
                },
                'Volume_Profile': {
                    'concepts': ['Point of Control', 'Value Area', 'High Volume Nodes'],
                    'significance': 'Volume-based support/resistance'
                }
            }
        }

    def enhanced_trend_analysis(self, df):
        """Comprehensive trend analysis with multiple timeframe confirmation"""
        signals = []
        confidence_score = 0
        max_confidence = 10
        
        if len(df) < 20:
            return ["Insufficient data for trend analysis"], 0
            
        # Multi-timeframe MA analysis
        current_price = df['close'].iloc[-1]
        
        # Calculate additional moving averages if needed
        if 'SMA_5' not in df.columns:
            df['SMA_5'] = df['close'].rolling(5).mean()
        if 'EMA_8' not in df.columns:
            df['EMA_8'] = df['close'].ewm(span=8).mean()
            
        short_term_bullish = (current_price > df['SMA_20'].iloc[-1] and
                             current_price > df['EMA_8'].iloc[-1])
        short_term_bearish = (current_price < df['SMA_20'].iloc[-1] and
                             current_price < df['EMA_8'].iloc[-1])
        
        # Medium-term trend
        medium_term_bullish = (current_price > df['SMA_50'].iloc[-1])
        medium_term_bearish = (current_price < df['SMA_50'].iloc[-1])
        
        # Long-term trend
        long_term_bullish = (current_price > df['SMA_200'].iloc[-1])
        long_term_bearish = (current_price < df['SMA_200'].iloc[-1])
        
        # Trend scoring
        if short_term_bullish and medium_term_bullish and long_term_bullish:
            signals.append("🟢 Strong Uptrend: All timeframes aligned bullishly")
            confidence_score += 3
        elif short_term_bearish and medium_term_bearish and long_term_bearish:
            signals.append("🔴 Strong Downtrend: All timeframes aligned bearishly")
            confidence_score += 3
        else:
            mixed_signals = []
            if short_term_bullish: 
                mixed_signals.append("ST ↑")
                confidence_score += 1
            if medium_term_bullish: 
                mixed_signals.append("MT ↑") 
                confidence_score += 1
            if long_term_bullish: 
                mixed_signals.append("LT ↑")
                confidence_score += 1
            if short_term_bearish: 
                mixed_signals.append("ST ↓")
            if medium_term_bearish: 
                mixed_signals.append("MT ↓")
            if long_term_bearish: 
                mixed_signals.append("LT ↓")
                
            signals.append(f"🟡 Mixed Trend: {' '.join(mixed_signals)}")
        
        # ADX trend strength
        if 'ADX' in df.columns:
            adx = df['ADX'].iloc[-1]
            if adx > 25:
                signals.append(f"📈 Strong Trend (ADX: {adx:.1f})")
                confidence_score += 2
            elif adx > 20:
                signals.append(f"↔️ Moderate Trend (ADX: {adx:.1f})")
                confidence_score += 1
            else:
                signals.append(f"⚪ Weak/No Trend (ADX: {adx:.1f})")
        
        return signals, min(confidence_score, max_confidence)

    def enhanced_momentum_analysis(self, df):
        """Multi-indicator momentum analysis with divergence detection"""
        signals = []
        momentum_score = 0
        max_score = 8
        
        if len(df) < 2:
            return ["Insufficient data for momentum analysis"], 0
            
        # RSI analysis
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if rsi > 70:
                signals.append(f"🎯 RSI Overbought: {rsi:.1f}")
                momentum_score -= 1
            elif rsi < 30:
                signals.append(f"🎯 RSI Oversold: {rsi:.1f}")
                momentum_score += 1
            else:
                signals.append(f"⚖️ RSI Neutral: {rsi:.1f}")
        
        # MACD analysis
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            macd = df['MACD'].iloc[-1]
            signal = df['MACD_Signal'].iloc[-1]
            
            if macd > signal:
                signals.append("📊 MACD Bullish")
                momentum_score += 1
            else:
                signals.append("📊 MACD Bearish")
                momentum_score -= 1
            
            if macd > 0:
                signals.append("🟢 MACD Above Zero (Bullish)")
                momentum_score += 1
            else:
                signals.append("🔴 MACD Below Zero (Bearish)")
                momentum_score -= 1
        
        # Stochastic analysis
        if all(col in df.columns for col in ['%K', '%D']):
            k = df['%K'].iloc[-1]
            d = df['%D'].iloc[-1]
            
            if k > 80 and d > 80:
                signals.append("🎯 Stochastic Overbought")
                momentum_score -= 1
            elif k < 20 and d < 20:
                signals.append("🎯 Stochastic Oversold")
                momentum_score += 1
        
        return signals, min(max(momentum_score, 0), max_score)

    def enhanced_volume_analysis(self, df):
        """Comprehensive volume analysis with multi-indicator confirmation"""
        signals = []
        volume_score = 0
        max_score = 6
        
        if len(df) < 20:
            return ["Insufficient data for volume analysis"], 0
        
        # OBV analysis
        if 'OBV' in df.columns:
            obv_current = df['OBV'].iloc[-1]
            obv_prev = df['OBV'].iloc[-5] if len(df) > 5 else obv_current
            price_current = df['close'].iloc[-1]
            price_prev = df['close'].iloc[-5] if len(df) > 5 else price_current
            
            obv_trend = obv_current > obv_prev
            price_trend = price_current > price_prev
            
            if obv_trend and price_trend:
                signals.append("📊 OBV Confirming Uptrend")
                volume_score += 2
            elif not obv_trend and not price_trend:
                signals.append("📊 OBV Confirming Downtrend")
                volume_score -= 2
            elif obv_trend != price_trend:
                signals.append("⚠️ OBV Divergence Detected")
                volume_score -= 1
        
        # CMF analysis
        if 'CMF' in df.columns:
            cmf = df['CMF'].iloc[-1]
            if cmf > 0.05:
                signals.append(f"💰 Strong Buying Pressure (CMF: {cmf:.3f})")
                volume_score += 2
            elif cmf < -0.05:
                signals.append(f"💰 Strong Selling Pressure (CMF: {cmf:.3f})")
                volume_score -= 2
        
        return signals, min(max(volume_score, 0), max_score)

    def volatility_analysis(self, df):
        """Comprehensive volatility assessment"""
        signals = []
        
        # ATR analysis
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
            atr_percent = (atr / df['close'].iloc[-1]) * 100
            
            if atr_percent > 5:
                signals.append(f"🌪️ High Volatility (ATR: {atr_percent:.1f}%)")
            elif atr_percent < 2:
                signals.append(f"🍃 Low Volatility (ATR: {atr_percent:.1f}%)")
            else:
                signals.append(f"⚖️ Moderate Volatility (ATR: {atr_percent:.1f}%)")
        
        return signals

    def support_resistance_analysis(self, df):
        """Advanced support and resistance analysis"""
        signals = []
        
        if len(df) < 20:
            return ["Insufficient data for S/R analysis"]
            
        # Recent price action relative to key levels
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        # Distance from recent extremes
        from_high = ((recent_high - current_price) / recent_high) * 100
        from_low = ((current_price - recent_low) / current_price) * 100
        
        if from_high < 2:
            signals.append(f"🏔️ Near Resistance: {from_high:.1f}% from recent high")
        elif from_low < 2:
            signals.append(f"🛟 Near Support: {from_low:.1f}% from recent low")
        
        return signals

    def generate_advanced_summary(self, df):
        """Generate comprehensive analysis with confidence scoring"""
        if len(df) < 20:
            return self._insufficient_data_summary()
            
        trend_signals, trend_score = self.enhanced_trend_analysis(df)
        momentum_signals, momentum_score = self.enhanced_momentum_analysis(df)
        volume_signals, volume_score = self.enhanced_volume_analysis(df)
        volatility_signals = self.volatility_analysis(df)
        sr_signals = self.support_resistance_analysis(df)
        
        # Overall confidence score (0-10)
        total_score = trend_score + momentum_score + volume_score
        max_possible = 10 + 8 + 6  # Max scores from each category
        confidence_percentage = (total_score / max_possible) * 100
        
        # Market regime detection
        market_regime = self.detect_market_regime(df)
        
        # Risk assessment
        risk_level = self.assess_risk_level(df)
        
        summary = {
            'trend': trend_signals,
            'momentum': momentum_signals,
            'volume': volume_signals,
            'volatility': volatility_signals,
            'support_resistance': sr_signals,
            'overall_bias': self.determine_advanced_bias(trend_score, momentum_score, volume_score),
            'confidence_score': confidence_percentage,
            'market_regime': market_regime,
            'risk_assessment': risk_level,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return summary

    def detect_market_regime(self, df):
        """Detect current market regime"""
        if 'ADX' not in df.columns or len(df) < 20:
            return "Unknown"
            
        adx = df['ADX'].iloc[-1]
        
        if adx > 25:
            if df['close'].iloc[-1] > df['SMA_50'].iloc[-1]:
                return "Trending Bull"
            else:
                return "Trending Bear"
        else:
            return "Ranging/Low Volatility"

    def assess_risk_level(self, df):
        """Comprehensive risk assessment"""
        risk_factors = []
        
        # Volatility risk
        if 'ATR' in df.columns:
            atr_ratio = df['ATR'].iloc[-1] / df['close'].iloc[-1]
            if atr_ratio > 0.05:
                risk_factors.append("High Volatility")
            elif atr_ratio > 0.02:
                risk_factors.append("Medium Volatility")
            else:
                risk_factors.append("Low Volatility")
        
        return risk_factors if risk_factors else ["Normal Market Conditions"]

    def determine_advanced_bias(self, trend_score, momentum_score, volume_score):
        """Determine overall bias with weighted scoring"""
        total = trend_score + momentum_score + volume_score
        max_possible = 10 + 8 + 6
        
        if total > max_possible * 0.6:
            return "Strong Bullish"
        elif total > max_possible * 0.4:
            return "Moderate Bullish"
        elif total > max_possible * 0.2:
            return "Neutral"
        elif total > max_possible * 0.1:
            return "Moderate Bearish"
        else:
            return "Strong Bearish"

    def _insufficient_data_summary(self):
        """Return summary for insufficient data"""
        return {
            'trend': ["Insufficient data for analysis"],
            'momentum': ["Insufficient data for analysis"],
            'volume': ["Insufficient data for analysis"],
            'volatility': ["Insufficient data for analysis"],
            'support_resistance': ["Insufficient data for analysis"],
            'overall_bias': "Neutral",
            'confidence_score': 0,
            'market_regime': "Unknown",
            'risk_assessment': ["Insufficient Data"],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_trading_recommendations(self, summary):
        """Generate trading recommendations based on analysis"""
        bias = summary['overall_bias'].lower()
        regime = summary['market_regime'].lower()
        
        recommendations = []
        
        if "bull" in bias and "trend" in regime:
            recommendations.extend([
                "Consider long positions on pullbacks to support",
                "Use trailing stops to protect profits",
                "Watch for trend exhaustion signals"
            ])
        elif "bear" in bias and "trend" in regime:
            recommendations.extend([
                "Consider short positions on rallies to resistance", 
                "Use tight stops above recent highs",
                "Monitor for potential bottoming patterns"
            ])
        elif "rang" in regime:
            recommendations.extend([
                "Trade range boundaries (buy support, sell resistance)",
                "Use mean reversion strategies",
                "Watch for breakout confirmation"
            ])
        else:
            recommendations.extend([
                "Wait for clearer market direction",
                "Reduce position sizes due to uncertainty",
                "Monitor key support/resistance levels"
            ])
        
        return recommendations

# Initialize enhanced ontology
ontology = EnhancedStockAnalysisOntology()

# ─────────────────────────────────────────────
#  App setup with callback exception suppression
# ─────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR], suppress_callback_exceptions=True)
server = app.server

# ─────────────────────────────────────────────
#  COMPLETE Layout with ALL Chart Components
# ─────────────────────────────────────────────
app.layout = dbc.Container([
    # Header
    dbc.NavbarSimple(
        brand="Advanced Stock Dashboard with Enhanced Ontology",
        color="dark brown",
        dark=True,
    ),

    # Symbol input
    dbc.Row([
        dbc.Col(
            dbc.InputGroup([
                dbc.Input(id='stock-input',
                          placeholder='Enter stock symbol',
                          value='AAPL',
                          debounce=False),
            ]),
            width=4,
        ),
    ], justify='center', className="my-3"),

    # Time-range selector
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Time Range:"),
            dcc.Dropdown(
                id='time-range',
                options=[
                    {'label': '6 months', 'value': '6mo'},
                    {'label': '1 year',   'value': '1y'},
                    {'label': '2 years',  'value': '2y'},
                    {'label': '5 years',  'value': '5y'},
                    {'label': 'All',      'value': 'max'}
                ],
                value='1y',
                clearable=False
            )
        ], width=4),
    ], justify='center', className="my-3"),

    # Interval selector
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Interval:"),
            dcc.Dropdown(
                id='interval',
                options=[
                    {'label': 'Daily',   'value': '1d'},
                    {'label': 'Weekly',  'value': '1wk'},
                    {'label': 'Monthly', 'value': '1mo'},
                ],
                value='1d',
                clearable=False
            )
        ], width=4),
    ], justify='center', className="my-3"),

    # Analyze button
    dbc.Row([
        dbc.Col(
            dbc.Button("Analyze with Enhanced Ontology",
                       id='analyze-button',
                       n_clicks=0,
                       color="primary"),
            width="auto"
        )
    ], justify="center", className="my-3"),

    # === ENHANCED ONTOLOGY INSIGHTS SECTION ===
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Advanced Ontology-Based Analysis", className="bg-primary text-white"),
                dbc.CardBody([
                    html.Div(id="ontology-insights"),
                    html.Div(id="trading-signals"),
                    html.Div(id="risk-assessment"),
                    html.Div(id="trading-recommendations")
                ])
            ]),
            width=12
        )
    ], className="mb-4"),

    # === COMPLETE Chart Layout - ALL 18 Charts ===
    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='candlestick-chart'))), width=12)], className="mb-4"),
    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='sma-ema-chart'))), width=12)], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='support-resistance-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='rsi-chart'))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='bollinger-bands-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='macd-chart'))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='stochastic-oscillator-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='obv-chart'))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='atr-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='cci-chart'))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='mfi-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='cmf-chart'))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='fi-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='fibonacci-retracement-chart'))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='ichimoku-cloud-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='vwap-chart'))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='adl-chart'))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='adx-di-chart'))), width=6),
    ], className="mb-4"),

    # Footer
    dbc.Row([
        dbc.Col(html.Footer("Advanced Stock Dashboard with Enhanced Ontology ©2025 by Abu Sanad", 
                           className="text-center text-muted"))
    ], className="mt-4")
], fluid=True)

# ─────────────────────────────────────────────
#  Enhanced Callback with ALL Outputs
# ─────────────────────────────────────────────
@app.callback(
    [Output('candlestick-chart', 'figure'),
     Output('sma-ema-chart', 'figure'),
     Output('support-resistance-chart', 'figure'),
     Output('rsi-chart', 'figure'),
     Output('bollinger-bands-chart', 'figure'),
     Output('macd-chart', 'figure'),
     Output('stochastic-oscillator-chart', 'figure'),
     Output('obv-chart', 'figure'),
     Output('atr-chart', 'figure'),
     Output('cci-chart', 'figure'),
     Output('mfi-chart', 'figure'),
     Output('cmf-chart', 'figure'),
     Output('fi-chart', 'figure'),
     Output('fibonacci-retracement-chart', 'figure'),
     Output('ichimoku-cloud-chart', 'figure'),
     Output('vwap-chart', 'figure'),
     Output('adl-chart', 'figure'),
     Output('adx-di-chart', 'figure'),
     Output('ontology-insights', 'children'),
     Output('trading-signals', 'children'),
     Output('risk-assessment', 'children'),
     Output('trading-recommendations', 'children')],
    Input('analyze-button', 'n_clicks'),
    State('stock-input', 'value'),
    State('time-range', 'value'),
    State('interval', 'value')
)
def update_graphs(n_clicks, ticker, time_range, interval):
    # Return empty until first click
    if not n_clicks:
        empty = go.Figure().update_layout(
            title="Click 'Analyze Stock' to display the analysis",
            template='plotly_dark')
        empty_insights = html.Div("Click analyze to see enhanced ontology-based insights")
        empty_signals = html.Div()
        empty_risk = html.Div()
        empty_recommendations = html.Div()
        return (empty,) * 18 + (empty_insights, empty_signals, empty_risk, empty_recommendations)

    # Auto-append '.SR' for Saudi tickers
    if ticker.isdigit():
        ticker += '.SR'

    # Fetch data
    try:
        tq = Ticker(ticker)
        df = tq.history(period=time_range, interval=interval)

        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values('date')

        if df.empty:
            empty = go.Figure().update_layout(
                title=f"No data for {ticker} ({time_range}, {interval})",
                template='plotly_dark')
            empty_insights = html.Div(f"No data available for {ticker}")
            return (empty,) * 18 + (empty_insights, html.Div(), html.Div(), html.Div())

    except Exception as e:
        empty = go.Figure().update_layout(
            title=f"Error fetching data for {ticker}: {str(e)}",
            template='plotly_dark')
        empty_insights = html.Div(f"Error fetching data: {str(e)}")
        return (empty,) * 18 + (empty_insights, html.Div(), html.Div(), html.Div())

    # Calculate all indicators
    try:
        # Moving Averages
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['SMA_200'] = df['close'].rolling(200).mean()

        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # Pivot Points
        pivot = (df['high'] + df['low'] + df['close']) / 3
        df['Pivot_Point'] = pivot
        df['Support_1'] = 2 * pivot - df['high']
        df['Resistance_1'] = 2 * pivot - df['low']
        df['Support_2'] = pivot - (df['high'] - df['low'])
        df['Resistance_2'] = pivot + (df['high'] - df['low'])

        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # Bollinger Bands
        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['Upper_band'] = ma20 + 2 * std20
        df['Lower_band'] = ma20 - 2 * std20

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['%K'] = stoch.stoch()
        df['%D'] = stoch.stoch_signal()

        # Volume indicators
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Volatility
        df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Other indicators
        df['CCI'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        df['ADL'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        
        df['MFI'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['FI'] = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()

        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['ADX'] = adx.adx()
        df['DI+'] = adx.adx_pos()
        df['DI-'] = adx.adx_neg()

        # Fibonacci levels
        max_p, min_p = df['high'].max(), df['low'].min()
        diff = max_p - min_p
        fib = {
            '0.0%': max_p,
            '23.6%': max_p - 0.236 * diff,
            '38.2%': max_p - 0.382 * diff,
            '50.0%': max_p - 0.5 * diff,
            '61.8%': max_p - 0.618 * diff,
            '100.0%': min_p,
        }

        # Ichimoku
        df['Tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        df['Kijun_sen'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
        df['Senkou_span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
        df['Chikou_span'] = df['close'].shift(-26)

    except Exception as e:
        empty = go.Figure().update_layout(
            title=f"Error calculating indicators: {str(e)}",
            template='plotly_dark')
        empty_insights = html.Div(f"Error in calculations: {str(e)}")
        return (empty,) * 18 + (empty_insights, html.Div(), html.Div(), html.Div())

    # Generate Enhanced Ontology Insights
    try:
        analysis_summary = ontology.generate_advanced_summary(df)
        trading_recommendations = ontology.get_trading_recommendations(analysis_summary)
        
        # Create enhanced insights display
        insights_content = [
            html.H4(f"Overall Bias: {analysis_summary['overall_bias']}", 
                   className=f"text-{'success' if 'bull' in analysis_summary['overall_bias'].lower() else 'danger' if 'bear' in analysis_summary['overall_bias'].lower() else 'warning'}"),
            html.H5(f"Confidence Score: {analysis_summary['confidence_score']:.1f}%"),
            html.P(f"Market Regime: {analysis_summary['market_regime']}"),
            html.P(f"Analysis Time: {analysis_summary['timestamp']}"),
            
            dbc.Row([
                dbc.Col([
                    html.H5("📈 Trend Analysis"),
                    html.Ul([html.Li(signal) for signal in analysis_summary['trend'][:5]])
                ], width=6),
                
                dbc.Col([
                    html.H5("⚡ Momentum Analysis"), 
                    html.Ul([html.Li(signal) for signal in analysis_summary['momentum'][:5]])
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.H5("📊 Volume Analysis"),
                    html.Ul([html.Li(signal) for signal in analysis_summary['volume'][:5]])
                ], width=6),
                
                dbc.Col([
                    html.H5("🌪️ Volatility & Support/Resistance"),
                    html.Ul([html.Li(signal) for signal in analysis_summary['volatility'] + analysis_summary['support_resistance']])
                ], width=6),
            ])
        ]
        
        # Risk assessment
        risk_content = [
            html.H5("🛡️ Risk Assessment"),
            html.Ul([html.Li(risk) for risk in analysis_summary['risk_assessment']])
        ]
        
        # Trading recommendations
        recommendations_content = [
            html.H5("💡 Trading Recommendations"),
            html.Ul([html.Li(recommendation) for recommendation in trading_recommendations])
        ]
        
        signals_content = html.Div([
            html.H5("📈 Key Signals"),
            html.P("Analysis complete - check individual charts for detailed signals")
        ])
        
    except Exception as e:
        insights_content = html.Div(f"Error in ontology analysis: {str(e)}")
        signals_content = html.Div()
        risk_content = html.Div()
        recommendations_content = html.Div()

    # Generate all figures
    try:
        # Candlestick chart
        candlestick_fig = go.Figure(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Candlestick'))
        candlestick_fig.add_trace(go.Bar(
            x=df.index, y=df['volume'], name='Volume',
            marker_color='rgba(52,152,219,0.5)', yaxis='y2'))
        candlestick_fig.update_layout(
            title=f'{ticker} Candlestick',
            yaxis2=dict(title='Volume', overlaying='y', side='right'),
            template='plotly_dark')

        # SMA/EMA chart
        sma_ema_fig = go.Figure()
        sma_ema_fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close'))
        for col in ['SMA_20','SMA_50','SMA_200','EMA_20','EMA_50','EMA_200']:
            sma_ema_fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
        sma_ema_fig.update_layout(title=f'{ticker} SMA & EMA', template='plotly_dark')

        # Support Resistance chart
        support_resistance_fig = go.Figure()
        for col, style in [('Pivot_Point','dash'),('Support_1','dot'),
                           ('Resistance_1','dot'),('Support_2','dot'),
                           ('Resistance_2','dot')]:
            support_resistance_fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=col.replace('_',' '),
                           line=dict(dash=style)))
        support_resistance_fig.update_layout(
            title=f'{ticker} Support & Resistance', template='plotly_dark')

        # RSI chart
        rsi_fig = go.Figure(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        for lvl,color in [(70,'Red'),(30,'Green')]:
            rsi_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1],
                              y0=lvl, y1=lvl, line=dict(color=color, dash='dash'))
        rsi_fig.update_layout(title=f'{ticker} RSI', template='plotly_dark')
        
        # Bollinger Bands chart
        bollinger_bands_fig = go.Figure()
        for col in ['close','Upper_band','Lower_band']:
            bollinger_bands_fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
        bollinger_bands_fig.update_layout(
            title=f'{ticker} Bollinger Bands', template='plotly_dark')

        # MACD chart
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'))
        macd_fig.update_layout(title=f'{ticker} MACD', template='plotly_dark')

        # Stochastic chart
        stochastic_fig = go.Figure()
        stochastic_fig.add_trace(go.Scatter(x=df.index, y=df['%K'], name='%K'))
        stochastic_fig.add_trace(go.Scatter(x=df.index, y=df['%D'], name='%D'))
        stochastic_fig.update_layout(
            title=f'{ticker} Stochastic Oscillator', template='plotly_dark')

        # OBV chart
        obv_fig  = go.Figure(go.Scatter(x=df.index, y=df['OBV'], name='OBV'))
        obv_fig.update_layout(title=f'{ticker} OBV', template='plotly_dark')

        # ATR chart
        atr_fig  = go.Figure(go.Scatter(x=df.index, y=df['ATR'], name='ATR'))
        atr_fig.update_layout(title=f'{ticker} ATR', template='plotly_dark')

        # CCI chart
        cci_fig  = go.Figure(go.Scatter(x=df.index, y=df['CCI'], name='CCI'))
        for lvl,color in [(100,'Red'),(-100,'Green')]:
            cci_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1],
                              y0=lvl, y1=lvl, line=dict(color=color, dash='dash'))
        cci_fig.update_layout(title=f'{ticker} CCI', template='plotly_dark')

        # MFI chart
        mfi_fig  = go.Figure(go.Scatter(x=df.index, y=df['MFI'], name='MFI'))
        for lvl,color in [(80,'Red'),(20,'Green')]:
            mfi_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1],
                              y0=lvl, y1=lvl, line=dict(color=color, dash='dash'))
        mfi_fig.update_layout(title=f'{ticker} MFI', template='plotly_dark')

        # CMF chart
        cmf_fig  = go.Figure(go.Scatter(x=df.index, y=df['CMF'], name='CMF'))
        cmf_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1],
                          y0=0, y1=0, line=dict(color='Red', dash='dash'))
        cmf_fig.update_layout(title=f'{ticker} CMF', template='plotly_dark')

        # FI chart
        fi_fig   = go.Figure(go.Scatter(x=df.index, y=df['FI'], name='FI'))
        fi_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1],
                         y0=0, y1=0, line=dict(color='Red', dash='dash'))
        fi_fig.update_layout(title=f'{ticker} Force Index', template='plotly_dark')

        # Fibonacci chart
        fib_fig  = go.Figure(go.Scatter(x=df.index, y=df['close'], name='Close'))
        for label,price in fib.items():
            fib_fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]], y=[price, price],
                name=f'Fib {label}', line=dict(dash='dash')))
        fib_fig.update_layout(
            title=f'{ticker} Fibonacci Retracement', template='plotly_dark')

        # Ichimoku chart
        ichimoku_fig = go.Figure(go.Scatter(x=df.index, y=df['close'], name='Close'))
        for col in ['Tenkan_sen','Kijun_sen','Senkou_span_a','Senkou_span_b','Chikou_span']:
            ichimoku_fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
        ichimoku_fig.update_layout(
            title=f'{ticker} Ichimoku Cloud', template='plotly_dark')

        # VWAP chart
        vwap_fig = go.Figure()
        vwap_fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close'))
        vwap_fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'],  name='VWAP'))
        vwap_fig.update_layout(title=f'{ticker} VWAP', template='plotly_dark')

        # ADL chart
        adl_fig = go.Figure()
        adl_fig.add_trace(go.Scatter(x=df.index, y=df['ADL'], name='ADL'))
        adl_fig.update_layout(title=f'{ticker} ADL', template='plotly_dark')

        # ADX chart
        adx_fig = go.Figure()
        adx_fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX'))
        adx_fig.add_trace(go.Scatter(x=df.index, y=df['DI-'], name='DI-'))
        adx_fig.add_trace(go.Scatter(x=df.index, y=df['DI+'], name='DI+'))
        adx_fig.update_layout(title=f'{ticker} ADX & DI', template='plotly_dark')

    except Exception as e:
        empty = go.Figure().update_layout(
            title=f"Error generating charts: {str(e)}",
            template='plotly_dark')
        return (empty,) * 18 + (insights_content, signals_content, risk_content, recommendations_content)

    # Return all 22 outputs
    return (candlestick_fig, sma_ema_fig, support_resistance_fig, rsi_fig,
            bollinger_bands_fig, macd_fig, stochastic_fig, obv_fig,
            atr_fig, cci_fig, mfi_fig, cmf_fig, fi_fig, fib_fig,
            ichimoku_fig, vwap_fig, adl_fig, adx_fig,
            insights_content, signals_content, risk_content, recommendations_content)

# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run_server(debug=False, port=8050)


# In[ ]:





# In[ ]:





# In[ ]:




