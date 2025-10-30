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
        trend_score = 0
        max_score = 15

        if len(df) < 50:
            return ["Insufficient data for trend analysis"], 0

        current_price = df['close'].iloc[-1]

        # Calculate additional moving averages if needed
        for period in [5, 8]:
            if f'SMA_{period}' not in df.columns:
                df[f'SMA_{period}'] = df['close'].rolling(period).mean()
            if f'EMA_{period}' not in df.columns:
                df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Multi-timeframe trend analysis with weighted scoring
        timeframe_scores = {
            'short_term': 0,
            'medium_term': 0, 
            'long_term': 0
        }

        # Short-term trend (5-20 days)
        if current_price > df['SMA_20'].iloc[-1]:
            timeframe_scores['short_term'] += 2
            signals.append("🟢 Short-term: Above SMA 20")
        else:
            timeframe_scores['short_term'] -= 2
            signals.append("🔴 Short-term: Below SMA 20")

        if current_price > df['EMA_8'].iloc[-1]:
            timeframe_scores['short_term'] += 1
            signals.append("🟢 Short-term: Above EMA 8")
        else:
            timeframe_scores['short_term'] -= 1
            signals.append("🔴 Short-term: Below EMA 8")

        # Medium-term trend (50 days)
        if current_price > df['SMA_50'].iloc[-1]:
            timeframe_scores['medium_term'] += 3
            signals.append("🟢 Medium-term: Above SMA 50")
        else:
            timeframe_scores['medium_term'] -= 3
            signals.append("🔴 Medium-term: Below SMA 50")

        # Long-term trend (200 days)
        if current_price > df['SMA_200'].iloc[-1]:
            timeframe_scores['long_term'] += 4
            signals.append("🟢 Long-term: Above SMA 200")
        else:
            timeframe_scores['long_term'] -= 4
            signals.append("🔴 Long-term: Below SMA 200")

        # Trend alignment analysis
        total_alignment = sum(timeframe_scores.values())

        if total_alignment >= 6:
            trend_strength = "🟢 STRONG UPTREND"
            trend_score += 8
        elif total_alignment >= 2:
            trend_strength = "🟡 MODERATE UPTREND" 
            trend_score += 5
        elif total_alignment >= -2:
            trend_strength = "⚪ MIXED/NEUTRAL TREND"
            trend_score += 2
        elif total_alignment >= -6:
            trend_strength = "🟡 MODERATE DOWNTREND"
            trend_score += 1
        else:
            trend_strength = "🔴 STRONG DOWNTREND"

        signals.insert(0, trend_strength)

        # ADX trend strength with DI analysis
        if 'ADX' in df.columns and 'DI+' in df.columns and 'DI-' in df.columns:
            adx = df['ADX'].iloc[-1]
            di_plus = df['DI+'].iloc[-1]
            di_minus = df['DI-'].iloc[-1]

            if adx > 25:
                if di_plus > di_minus:
                    signals.append(f"📈 Strong Bullish Trend (ADX: {adx:.1f}, DI+ > DI-)")
                    trend_score += 3
                else:
                    signals.append(f"📉 Strong Bearish Trend (ADX: {adx:.1f}, DI- > DI+)")
                    trend_score -= 3
            elif adx > 20:
                signals.append(f"↔️ Moderate Trend Strength (ADX: {adx:.1f})")
                trend_score += 1
            else:
                signals.append(f"⚪ Weak Trend (ADX: {adx:.1f})")

        # ICHIMOKU CLOUD ANALYSIS - ORDER MATTERS!
        if all(col in df.columns for col in ['Tenkan_sen', 'Kijun_sen', 'Senkou_span_a', 'Senkou_span_b']):
            ichimoku_signals = self.analyze_ichimoku_cloud(df, current_price)
            signals.extend(ichimoku_signals)
            trend_score += len([s for s in ichimoku_signals if '🟢' in s or '🔴' in s]) * 0.5

        # Moving average cross signals
        cross_signals = self.detect_moving_average_crosses(df)
        signals.extend(cross_signals)
        trend_score += len(cross_signals) * 0.5

        return signals, min(max(trend_score, 0), max_score)

    def detect_moving_average_crosses(self, df):
        """Detect significant moving average crosses"""
        signals = []
        
        # Golden/Death Cross
        if len(df) > 50 and 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            sma_50_current = df['SMA_50'].iloc[-1]
            sma_200_current = df['SMA_200'].iloc[-1]
            sma_50_prev = df['SMA_50'].iloc[-2] if len(df) > 1 else sma_50_current
            sma_200_prev = df['SMA_200'].iloc[-2] if len(df) > 1 else sma_200_current
            
            # Golden Cross
            if (sma_50_current > sma_200_current and sma_50_prev <= sma_200_prev):
                signals.append("💰 GOLDEN CROSS: SMA 50 crossed above SMA 200")
            # Death Cross
            elif (sma_50_current < sma_200_current and sma_50_prev >= sma_200_prev):
                signals.append("💀 DEATH CROSS: SMA 50 crossed below SMA 200")
        
        return signals

    def analyze_ichimoku_cloud(self, df, current_price):
        """Analyze Ichimoku Cloud with emphasis on LINE ORDER importance"""
        signals = []

        tenkan = df['Tenkan_sen'].iloc[-1]  # Conversion Line (9-period)
        kijun = df['Kijun_sen'].iloc[-1]    # Base Line (26-period)
        senkou_a = df['Senkou_span_a'].iloc[-1]  # Leading Span A
        senkou_b = df['Senkou_span_b'].iloc[-1]  # Leading Span B
        chikou = df['Chikou_span'].iloc[-1] if 'Chikou_span' in df.columns else 0  # Lagging Span

        # BULLISH ICHIMOKU ORDER (Strongest to weakest):
        # 1. Price > Cloud > Kijun > Tenkan (Strongest Bullish)
        # 2. Price > Cloud > Tenkan > Kijun 
        # 3. Price > Tenkan > Kijun > Cloud

        # BEARISH ICHIMOKU ORDER (Strongest to weakest):
        # 1. Price < Cloud < Kijun < Tenkan (Strongest Bearish)
        # 2. Price < Cloud < Tenkan < Kijun
        # 3. Price < Tenkan < Kijun < Cloud

        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # Determine line order and generate signals
        price_above_cloud = current_price > cloud_top
        price_below_cloud = current_price < cloud_bottom
        price_in_cloud = cloud_bottom <= current_price <= cloud_top

        tenkan_above_kijun = tenkan > kijun
        price_above_kijun = current_price > kijun
        price_above_tenkan = current_price > tenkan

        # ICHIMOKU LINE ORDER ANALYSIS
        if price_above_cloud:
            if price_above_kijun and tenkan_above_kijun:
                if current_price > tenkan:
                    # STRONGEST BULLISH: Price > Tenkan > Kijun > Cloud
                    signals.append("☁️ ICHIMOKU: STRONG BULLISH (Price > Tenkan > Kijun > Cloud)")
                else:
                    # BULLISH: Price > Cloud > Kijun > Tenkan
                    signals.append("☁️ ICHIMOKU: BULLISH (Price > Cloud > Kijun > Tenkan)")
            else:
                signals.append("☁️ ICHIMOKU: MILD BULLISH (Price above Cloud)")

        elif price_below_cloud:
            if not price_above_kijun and not tenkan_above_kijun:
                if current_price < tenkan:
                    # STRONGEST BEARISH: Price < Tenkan < Kijun < Cloud
                    signals.append("☁️ ICHIMOKU: STRONG BEARISH (Price < Tenkan < Kijun < Cloud)")
                else:
                    # BEARISH: Price < Cloud < Kijun < Tenkan
                    signals.append("☁️ ICHIMOKU: BEARISH (Price < Cloud < Kijun < Tenkan)")
            else:
                signals.append("☁️ ICHIMOKU: MILD BEARISH (Price below Cloud)")

        else:  # Price in cloud
            signals.append("☁️ ICHIMOKU: NEUTRAL (Price inside Cloud)")

        # TENKAN-KIJUN CROSS SIGNALS (Important momentum)
        if tenkan_above_kijun:
            signals.append("🟢 Tenkan > Kijun (Bullish Momentum)")
        else:
            signals.append("🔴 Tenkan < Kijun (Bearish Momentum)")

        # CLOUD COLOR ANALYSIS (Future sentiment)
        if senkou_a > senkou_b:
            signals.append("🟢 Cloud Color: Green (Bullish Future)")
        else:
            signals.append("🔴 Cloud Color: Red (Bearish Future)")

        # CHIKOU SPAN ANALYSIS (Lagging confirmation)
        if chikou != 0:
            if chikou > current_price:
                signals.append("📈 Chikou Span: Above Price (Bullish Confirmation)")
            else:
                signals.append("📉 Chikou Span: Below Price (Bearish Confirmation)")

        return signals

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
        volatility_score = 0
        
        # ATR analysis
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
            atr_percent = (atr / df['close'].iloc[-1]) * 100
            
            if atr_percent > 5:
                signals.append(f"🌪️ High Volatility (ATR: {atr_percent:.1f}%)")
                volatility_score = 3
            elif atr_percent < 2:
                signals.append(f"🍃 Low Volatility (ATR: {atr_percent:.1f}%)")
                volatility_score = 1
            else:
                signals.append(f"⚖️ Moderate Volatility (ATR: {atr_percent:.1f}%)")
                volatility_score = 2
        
        return signals, volatility_score

    def support_resistance_analysis(self, df):
        """Advanced support and resistance analysis"""
        signals = []
        sr_score = 0
        
        if len(df) < 20:
            return ["Insufficient data for S/R analysis"], 0
            
        # Recent price action relative to key levels
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        # Distance from recent extremes
        from_high = ((recent_high - current_price) / recent_high) * 100
        from_low = ((current_price - recent_low) / current_price) * 100
        
        if from_high < 2:
            signals.append(f"🏔️ Near Resistance: {from_high:.1f}% from recent high")
            sr_score -= 2
        elif from_low < 2:
            signals.append(f"🛟 Near Support: {from_low:.1f}% from recent low")
            sr_score += 2
        
        return signals, sr_score

    def generate_advanced_summary(self, df):
        """Generate comprehensive analysis with logical consistency"""
        if len(df) < 50:
            return self._insufficient_data_summary()

        current_price = df['close'].iloc[-1] if len(df) > 0 else 0

        # Get all analysis components
        trend_signals, trend_score = self.enhanced_trend_analysis(df)
        momentum_signals, momentum_score = self.enhanced_momentum_analysis(df)
        volume_signals, volume_score = self.enhanced_volume_analysis(df)
        volatility_signals, volatility_score = self.volatility_analysis(df)
        sr_signals, sr_score = self.support_resistance_analysis(df)

        # ICHIMOKU SPECIFIC ANALYSIS (added to top)
        ichimoku_signals = []
        if all(col in df.columns for col in ['Tenkan_sen', 'Kijun_sen', 'Senkou_span_a', 'Senkou_span_b']):
            ichimoku_signals = self.get_detailed_ichimoku_analysis(df, current_price)

        # Calculate overall bias with logical consistency
        overall_bias, confidence = self.determine_overall_bias(
            trend_score, momentum_score, volume_score, volatility_score, sr_score
        )

        # Market regime detection
        market_regime = self.detect_market_regime(df, trend_score, volatility_score)

        # Risk assessment
        risk_level = self.assess_risk_level(df, momentum_score, volatility_score)

        summary = {
            'ichimoku': ichimoku_signals,  # NEW: Ichimoku-specific signals at top
            'trend': trend_signals[:8],
            'momentum': momentum_signals[:8],
            'volume': volume_signals[:6],
            'volatility': volatility_signals,
            'support_resistance': sr_signals,
            'market_regime': market_regime,
            'overall_bias': overall_bias,
            'confidence_score': confidence,
            'risk_assessment': risk_level,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detailed_scores': {
                'trend': trend_score,
                'momentum': momentum_score,
                'volume': volume_score,
                'volatility': volatility_score,
                'support_resistance': sr_score
            }
        }

        return summary

    def get_detailed_ichimoku_analysis(self, df, current_price):
        """Get detailed Ichimoku analysis focusing on LINE ORDER importance"""
        signals = []

        tenkan = df['Tenkan_sen'].iloc[-1]
        kijun = df['Kijun_sen'].iloc[-1]
        senkou_a = df['Senkou_span_a'].iloc[-1]
        senkou_b = df['Senkou_span_b'].iloc[-1]

        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # LINE ORDER ANALYSIS - THIS IS CRITICAL!
        signals.append("🎯 ICHIMOKU LINE ORDER ANALYSIS:")

        # Determine the current line order
        lines = [
            ("Price", current_price),
            ("Tenkan", tenkan),
            ("Kijun", kijun), 
            ("Cloud Top", cloud_top),
            ("Cloud Bottom", cloud_bottom)
        ]

        # Sort by value to see the actual order
        sorted_lines = sorted(lines, key=lambda x: x[1], reverse=True)
        current_order = " > ".join([f"{name}" for name, val in sorted_lines])

        signals.append(f"   Current Order: {current_order}")

        # STRONG BULLISH PATTERNS
        if (current_price > tenkan > kijun > cloud_top):
            signals.append("   🟢 STRONGEST BULLISH: Price > Tenkan > Kijun > Cloud")
        elif (current_price > cloud_top > kijun > tenkan):
            signals.append("   🟢 BULLISH: Price > Cloud > Kijun > Tenkan")
        elif (current_price > tenkan > cloud_top > kijun):
            signals.append("   🟢 BULLISH: Price > Tenkan > Cloud > Kijun")

        # STRONG BEARISH PATTERNS  
        elif (current_price < tenkan < kijun < cloud_bottom):
            signals.append("   🔴 STRONGEST BEARISH: Price < Tenkan < Kijun < Cloud")
        elif (current_price < cloud_bottom < kijun < tenkan):
            signals.append("   🔴 BEARISH: Price < Cloud < Kijun < Tenkan")
        elif (current_price < tenkan < cloud_bottom < kijun):
            signals.append("   🔴 BEARISH: Price < Tenkan < Cloud < Kijun")

        # NEUTRAL/TRANSITION PATTERNS
        else:
            signals.append("   ⚪ MIXED ORDER: Transition or ranging market")

        # INDIVIDUAL COMPONENT SIGNALS
        signals.append("   📊 Component Analysis:")

        # Price vs Cloud
        if current_price > cloud_top:
            signals.append("     🟢 Price ABOVE Cloud (Bullish)")
        elif current_price < cloud_bottom:
            signals.append("     🔴 Price BELOW Cloud (Bearish)")
        else:
            signals.append("     ⚪ Price IN Cloud (Neutral)")

        # Tenkan vs Kijun
        if tenkan > kijun:
            signals.append("     🟢 Tenkan > Kijun (Bullish Momentum)")
        else:
            signals.append("     🔴 Tenkan < Kijun (Bearish Momentum)")

        # Cloud Color (Future sentiment)
        if senkou_a > senkou_b:
            signals.append("     🟢 Cloud Color: Green (Bullish Future Bias)")
        else:
            signals.append("     🔴 Cloud Color: Red (Bearish Future Bias)")

        # Cloud Thickness (Volatility)
        cloud_thickness = (cloud_top - cloud_bottom) / current_price * 100
        if cloud_thickness > 5:
            signals.append(f"     🌪️ Thick Cloud ({cloud_thickness:.1f}% - High Volatility)")
        elif cloud_thickness < 2:
            signals.append(f"     🍃 Thin Cloud ({cloud_thickness:.1f}% - Low Volatility)")

        return signals

    def determine_overall_bias(self, trend_score, momentum_score, volume_score, volatility_score, sr_score):
        """Determine overall bias with weighted scoring"""
        # Weighted scores (trend is most important)
        total_score = (
            trend_score * 0.35 +
            momentum_score * 0.25 +
            volume_score * 0.20 +
            volatility_score * 0.10 +
            sr_score * 0.10
        )
        
        max_possible = 15 * 0.35 + 8 * 0.25 + 6 * 0.20 + 3 * 0.10 + 2 * 0.10
        confidence = (total_score / max_possible) * 100
        
        # Determine bias based on weighted score
        if total_score > max_possible * 0.7:
            bias = "🟢 STRONG BULLISH"
        elif total_score > max_possible * 0.5:
            bias = "🟡 MODERATE BULLISH"
        elif total_score > max_possible * 0.3:
            bias = "⚪ NEUTRAL"
        elif total_score > max_possible * 0.1:
            bias = "🟡 MODERATE BEARISH"
        else:
            bias = "🔴 STRONG BEARISH"
        
        return bias, min(confidence, 100)

    def detect_market_regime(self, df, trend_score, volatility_score):
        """Detect current market regime"""
        regimes = []
        
        # Trend-based regime
        if trend_score > 10:
            regimes.append("📈 TRENDING BULL")
        elif trend_score > 6:
            regimes.append("📈 MILD UPTREND")
        elif trend_score < 3:
            regimes.append("📉 TRENDING BEAR")
        else:
            regimes.append("🔄 RANGING")
        
        # Volatility-based regime
        if volatility_score >= 3:
            regimes.append("🌪️ HIGH VOLATILITY")
        elif volatility_score <= 1:
            regimes.append("🍃 LOW VOLATILITY")
        else:
            regimes.append("⚖️ MODERATE VOLATILITY")
        
        return regimes

    def assess_risk_level(self, df, momentum_score, volatility_score):
        """Comprehensive risk assessment"""
        risk_factors = []
        
        # Overbought/oversold risk
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if rsi > 75:
                risk_factors.append("🔴 RSI Extremely Overbought")
            elif rsi > 70:
                risk_factors.append("🟡 RSI Overbought")
            elif rsi < 25:
                risk_factors.append("🔴 RSI Extremely Oversold")
            elif rsi < 30:
                risk_factors.append("🟢 RSI Oversold (Opportunity)")
        
        # Volatility risk
        if volatility_score >= 3:
            risk_factors.append("🔴 High Volatility - Larger Stops Needed")
        elif volatility_score <= 1:
            risk_factors.append("🟢 Low Volatility - Tighter Stops Possible")
        
        # Momentum risk
        if momentum_score < 3:
            risk_factors.append("🟡 Weak Momentum")
        
        # Near key levels risk
        if len(df) > 20:
            current_price = df['close'].iloc[-1]
            recent_high = df['high'].tail(20).max()
            from_high = ((recent_high - current_price) / recent_high) * 100
            if from_high < 3:
                risk_factors.append("🟡 Near Resistance - Reversal Risk")
        
        if not risk_factors:
            risk_factors.append("🟢 Normal Market Conditions")
        
        return risk_factors

    def _insufficient_data_summary(self):
        """Return summary for insufficient data"""
        return {
            'trend': ["Insufficient data for analysis"],
            'momentum': ["Insufficient data for analysis"],
            'volume': ["Insufficient data for analysis"],
            'volatility': ["Insufficient data for analysis"],
            'support_resistance': ["Insufficient data for analysis"],
            'ichimoku': ["Insufficient data for analysis"],
            'overall_bias': "NEUTRAL",
            'confidence_score': 0,
            'market_regime': ["Insufficient data"],
            'risk_assessment': ["Insufficient Data"],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detailed_scores': {
                'trend': 0,
                'momentum': 0,
                'volume': 0,
                'volatility': 0,
                'support_resistance': 0
            }
        }

    def get_trading_recommendations(self, summary):
        """Generate trading recommendations based on analysis"""
        bias = summary['overall_bias'].lower()
        regime = summary['market_regime'][0].lower() if summary.get('market_regime') else ''
        confidence = summary['confidence_score']
        
        recommendations = []
        
        if "strong bullish" in bias:
            recommendations.extend([
                "🎯 STRONG BUY SIGNAL - Consider long positions",
                "💰 Add on pullbacks to moving average support",
                "📈 Use trailing stops to protect profits",
                "🔍 Monitor for overbought conditions"
            ])
        elif "moderate bullish" in bias:
            recommendations.extend([
                "🟢 MODERATE BUY - Long with caution",
                "⚖️ Scale into positions gradually", 
                "🛡️ Use tighter stop losses",
                "📊 Wait for confirmation on breakouts"
            ])
        elif "neutral" in bias:
            recommendations.extend([
                "⚖️ NEUTRAL BIAS - Wait for clearer direction",
                "📉 Consider range-bound strategies",
                "🔍 Wait for breakout confirmation",
                "💼 Reduce position sizes"
            ])
        elif "bearish" in bias:
            recommendations.extend([
                "🔴 BEARISH BIAS - Consider short positions",
                "📉 Sell rallies to resistance",
                "🛡️ Use aggressive stop losses",
                "💸 Consider hedging strategies"
            ])
        
        # Confidence-based adjustments
        if confidence > 70:
            recommendations.append("✅ HIGH CONFIDENCE - Consider larger positions")
        elif confidence < 40:
            recommendations.append("⚠️ LOW CONFIDENCE - Reduce position sizes")
        
        # Risk-based recommendations
        risk_factors = summary.get('risk_assessment', [])
        if any("🔴" in risk for risk in risk_factors):
            recommendations.append("🚨 HIGH RISK ENVIRONMENT - Extreme caution advised")
        if any("🟡" in risk for risk in risk_factors):
            recommendations.append("⚠️ MODERATE RISK - Standard position sizing")
        
        # Always include risk management
        recommendations.extend([
            "🎯 RISK MANAGEMENT: Max 2% risk per trade",
            "📝 Always use stop losses based on ATR",
            "🔍 Monitor for changing market conditions"
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
        df['SMA_5'] = df['close'].rolling(5).mean()
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['SMA_200'] = df['close'].rolling(200).mean()

        df['EMA_8'] = df['close'].ewm(span=8, adjust=False).mean()
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
            html.H4(f"🎯 OVERALL BIAS: {analysis_summary['overall_bias']}", 
                   className=f"text-{'success' if 'bull' in analysis_summary['overall_bias'].lower() else 'danger' if 'bear' in analysis_summary['overall_bias'].lower() else 'warning'} font-weight-bold"),
            html.H5(f"📊 CONFIDENCE SCORE: {analysis_summary['confidence_score']:.1f}%"),
            html.P(f"🏛️ MARKET REGIME: {analysis_summary['market_regime'][0] if analysis_summary['market_regime'] else 'Unknown'}"),
            html.P(f"⏰ ANALYSIS TIME: {analysis_summary['timestamp']}"),

            # NEW: ICHIMOKU ANALYSIS AT TOP
            dbc.Row([
                dbc.Col([
                    html.H5("☁️ ICHIMOKU CLOUD ANALYSIS", className="text-info font-weight-bold"),
                    html.Ul([html.Li(signal, className="small") for signal in analysis_summary.get('ichimoku', ['No Ichimoku data'])[:10]])
                ], width=12),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.H5("📈 TREND ANALYSIS", className="text-info"),
                    html.Ul([html.Li(signal, className="small") for signal in analysis_summary['trend'][:6]])
                ], width=6),

                dbc.Col([
                    html.H5("⚡ MOMENTUM ANALYSIS", className="text-warning"), 
                    html.Ul([html.Li(signal, className="small") for signal in analysis_summary['momentum'][:6]])
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.H5("📊 Volume Analysis", className="text-success"),
                    html.Ul([html.Li(signal, className="small") for signal in analysis_summary['volume'][:5]])
                ], width=6),
                
                dbc.Col([
                    html.H5("🌪️ Volatility & Support/Resistance", className="text-primary"),
                    html.Ul([html.Li(signal, className="small") for signal in analysis_summary['volatility'] + analysis_summary['support_resistance']])
                ], width=6),
            ])
        ]
        
        # Risk assessment
        risk_content = [
            html.H5("🛡️ Risk Assessment", className="text-danger"),
            html.Ul([html.Li(risk, className="small") for risk in analysis_summary['risk_assessment']])
        ]
        
        # Trading recommendations
        recommendations_content = [
            html.H5("💡 Trading Recommendations", className="text-success"),
            html.Ul([html.Li(recommendation, className="small") for recommendation in trading_recommendations])
        ]
        
        signals_content = html.Div([
            html.H5("📈 Key Signals", className="text-info"),
            html.P("Analysis complete - check individual charts for detailed signals", className="small")
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
