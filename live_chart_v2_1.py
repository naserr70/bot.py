# live_chart.py - نسخه فیکس شده

import sys
import threading
import pandas as pd
import time
from datetime import datetime
import MetaTrader5 as mt5
import ta
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QComboBox, QLabel, QFrame, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWebEngineWidgets import QWebEngineView
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio


class ChartSignals(QObject):
    update_chart = pyqtSignal(str)
    update_price = pyqtSignal(str)
    update_status = pyqtSignal(str)


class LiveChart:
    def __init__(self, symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, tick_count=200, interval=2):
        self.symbol = symbol
        self.timeframe = timeframe
        self.tick_count = tick_count
        self.interval = interval
        self.running = False
        self.thread = None
        self.signals = ChartSignals()
        
        self.timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

    def fetch_data(self):
        """دریافت داده از MT5"""
        tick = None
        try:
            if not mt5.initialize():
                self.signals.update_status.emit("❌ MT5 not initialized")
                print("❌ MT5 initialize failed")
                return pd.DataFrame(), None
            
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.tick_count)
            tick = mt5.symbol_info_tick(self.symbol)
            mt5.shutdown()
            
            if rates is None or len(rates) == 0:
                self.signals.update_status.emit("❌ No data from MT5")
                print("❌ No rates data")
                return pd.DataFrame(), tick
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # اندیکاتورها
            if len(df) > 20:
                df['RSI'] = ta.momentum.rsi(df['close'], window=14)
                macd = ta.trend.MACD(df['close'])
                df['MACD'] = macd.macd()
                df['MACD_signal'] = macd.macd_signal()
                df['MACD_hist'] = macd.macd_diff()
                df['SMA20'] = df['close'].rolling(20).mean()
                df['SMA50'] = df['close'].rolling(50).mean()
            
            self.signals.update_status.emit(f"✅ {len(df)} candles loaded")
            print(f"✅ Fetched {len(df)} candles")
            return df, tick
            
        except Exception as e:
            self.signals.update_status.emit(f"❌ Error: {str(e)[:30]}")
            print(f"❌ Fetch error: {e}")
            return pd.DataFrame(), tick

    def create_chart(self, df):
        """ساخت چارت Plotly - با Plotly offline"""
        try:
            if df.empty:
                print("⚠️ DataFrame is empty, creating placeholder chart")
                fig = go.Figure()
                fig.add_annotation(
                    text="⏳ Waiting for data...<br>Make sure MT5 is running",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=24, color="white")
                )
                fig.update_layout(
                    template="plotly_dark",
                    height=700,
                    paper_bgcolor='#1e1e1e',
                    plot_bgcolor='#1e1e1e'
                )
                # استفاده از plotly offline به جای CDN
                return pio.to_html(fig, include_plotlyjs=True, full_html=True)
            
            print(f"📊 Creating chart with {len(df)} candles")
            
            # ساخت subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=[f'{self.symbol} Price', 'RSI (14)', 'MACD']
            )
            
            # کندل استیک
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350',
                    increasing_fillcolor='#26a69a',
                    decreasing_fillcolor='#ef5350'
                ),
                row=1, col=1
            )
            
            # SMA
            if 'SMA20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['SMA20'],
                        name='SMA 20',
                        line=dict(color='#ffeb3b', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'SMA50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['SMA50'],
                        name='SMA 50',
                        line=dict(color='#2196f3', width=1)
                    ),
                    row=1, col=1
                )
            
            # RSI
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['RSI'],
                        name='RSI',
                        line=dict(color='#ab47bc', width=2)
                    ),
                    row=2, col=1
                )
                # خطوط 30 و 70
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
            
            # MACD
            if 'MACD' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['MACD'],
                        name='MACD',
                        line=dict(color='#2196f3', width=2)
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['MACD_signal'],
                        name='Signal',
                        line=dict(color='#ff9800', width=2)
                    ),
                    row=3, col=1
                )
                
                # هیستوگرام
                if 'MACD_hist' in df.columns:
                    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in df['MACD_hist']]
                    fig.add_trace(
                        go.Bar(
                            x=df.index, y=df['MACD_hist'],
                            name='Histogram',
                            marker_color=colors,
                            showlegend=False,
                            opacity=0.6
                        ),
                        row=3, col=1
                    )
                
                fig.add_hline(y=0, line_color="gray", opacity=0.5, row=3, col=1)
            
            # Layout
            fig.update_layout(
                template="plotly_dark",
                height=750,
                showlegend=True,
                legend=dict(
                    x=0.01, y=0.99,
                    bgcolor='rgba(0,0,0,0.5)',
                    font=dict(size=10)
                ),
                margin=dict(l=60, r=60, t=50, b=50),
                xaxis_rangeslider_visible=False,
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e'
            )
            
            # تنظیم محورها
            fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(title_text="MACD", row=3, col=1, gridcolor='rgba(128,128,128,0.2)')
            fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
            
            # استفاده از plotly offline (بدون نیاز به اینترنت)
            html = pio.to_html(fig, include_plotlyjs=True, full_html=True)
            print(f"✅ Chart HTML created, length: {len(html)}")
            return html
            
        except Exception as e:
            print(f"❌ Chart creation error: {e}")
            import traceback
            traceback.print_exc()
            return f"""
            <html>
            <body style="background-color:#1e1e1e; color:white; display:flex; justify-content:center; align-items:center; height:100vh;">
                <div style="text-align:center;">
                    <h2>❌ Chart Error</h2>
                    <p>{str(e)}</p>
                </div>
            </body>
            </html>
            """

    def _update_loop(self):
        while self.running:
            try:
                df, tick = self.fetch_data()
                
                if not df.empty:
                    html = self.create_chart(df)
                    self.signals.update_chart.emit(html)
                
                if tick:
                    spread = (tick.ask - tick.bid) * 10000
                    price_text = f"Bid: {tick.bid:.2f} | Ask: {tick.ask:.2f} | Spread: {spread:.1f}"
                    self.signals.update_price.emit(price_text)
                
            except Exception as e:
                print(f"❌ Update loop error: {e}")
            
            time.sleep(self.interval)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print("✅ Chart update thread started")

    def stop(self):
        self.running = False
        print("⏹️ Chart update stopped")

    def set_timeframe(self, tf_name):
        if tf_name in self.timeframes:
            self.timeframe = self.timeframes[tf_name]
            print(f"📊 Timeframe changed to {tf_name}")


class ChartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.live_chart = LiveChart()
        
        self.setWindowTitle(f"📈 {self.live_chart.symbol} - Live Chart")
        self.setGeometry(100, 100, 1300, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: white; font-size: 14px; }
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #404040;
                padding: 8px 15px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #404040; }
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #404040;
                padding: 8px;
                min-width: 80px;
                font-size: 12px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #2196f3;
            }
        """)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("background-color: #2d2d2d; border-radius: 8px; padding: 10px;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setSpacing(15)
        
        # Symbol
        symbol_label = QLabel(f"📈 {self.live_chart.symbol}")
        symbol_label.setFont(QFont("Arial", 16, QFont.Bold))
        symbol_label.setStyleSheet("color: #2196f3;")
        toolbar_layout.addWidget(symbol_label)
        
        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #404040;")
        toolbar_layout.addWidget(sep1)
        
        # Timeframe
        toolbar_layout.addWidget(QLabel("Timeframe:"))
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
        self.tf_combo.setCurrentText("M5")
        self.tf_combo.currentTextChanged.connect(self.on_tf_changed)
        toolbar_layout.addWidget(self.tf_combo)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #404040;")
        toolbar_layout.addWidget(sep2)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh)
        toolbar_layout.addWidget(refresh_btn)
        
        toolbar_layout.addStretch()
        
        # Status label
        self.status_label = QLabel("⏳ Loading...")
        self.status_label.setStyleSheet("color: #90a4ae;")
        toolbar_layout.addWidget(self.status_label)
        
        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("color: #404040;")
        toolbar_layout.addWidget(sep3)
        
        # Price label
        self.price_label = QLabel("Bid: --- | Ask: --- | Spread: ---")
        self.price_label.setStyleSheet("color: #ffeb3b; font-size: 14px; font-weight: bold;")
        toolbar_layout.addWidget(self.price_label)
        
        layout.addWidget(toolbar)
        
        # Chart WebView
        self.web_view = QWebEngineView()
        self.web_view.setStyleSheet("background-color: #1e1e1e;")
        layout.addWidget(self.web_view)
        
        # Connect signals
        self.live_chart.signals.update_chart.connect(self.update_chart)
        self.live_chart.signals.update_price.connect(self.update_price)
        self.live_chart.signals.update_status.connect(self.update_status)
        
        # Show loading message
        self.show_loading()
        
        # Start chart after a delay
        QTimer.singleShot(1000, self.start_chart)

    def show_loading(self):
        """نمایش پیام لودینگ"""
        loading_html = """
        <html>
        <body style="background-color:#1e1e1e; color:white; display:flex; justify-content:center; align-items:center; height:100vh; margin:0;">
            <div style="text-align:center;">
                <h1>📈 Loading Chart...</h1>
                <p style="color:#90a4ae;">Connecting to MetaTrader 5</p>
                <p style="color:#90a4ae;">Please wait...</p>
            </div>
        </body>
        </html>
        """
        self.web_view.setHtml(loading_html)

    def start_chart(self):
        """شروع چارت"""
        print("🚀 Starting chart...")
        
        # اول داده اولیه رو بگیر و نشون بده
        df, tick = self.live_chart.fetch_data()
        
        if df.empty:
            print("⚠️ Initial data is empty")
            self.status_label.setText("⚠️ No data - Check MT5")
        else:
            print(f"📊 Initial data: {len(df)} candles")
            html = self.live_chart.create_chart(df)
            self.web_view.setHtml(html)
            
            if tick:
                spread = (tick.ask - tick.bid) * 10000
                self.price_label.setText(f"Bid: {tick.bid:.2f} | Ask: {tick.ask:.2f} | Spread: {spread:.1f}")
        
        # شروع بروزرسانی خودکار
        self.live_chart.start()

    def update_chart(self, html):
        """بروزرسانی چارت"""
        self.web_view.setHtml(html)

    def update_price(self, text):
        """بروزرسانی قیمت"""
        self.price_label.setText(text)

    def update_status(self, text):
        """بروزرسانی وضعیت"""
        self.status_label.setText(text)

    def on_tf_changed(self, tf):
        """تغییر تایم فریم"""
        self.live_chart.set_timeframe(tf)
        self.refresh()

    def refresh(self):
        """رفرش دستی"""
        self.status_label.setText("🔄 Refreshing...")
        df, tick = self.live_chart.fetch_data()
        if not df.empty:
            html = self.live_chart.create_chart(df)
            self.web_view.setHtml(html)

    def closeEvent(self, event):
        """هنگام بستن پنجره"""
        self.live_chart.stop()
        event.accept()


if __name__ == "__main__":
    print("="*50)
    print("📈 Live Chart Starting...")
    print("="*50)
    
    app = QApplication(sys.argv)
    window = ChartWindow()
    window.show()
    
    print("✅ Window opened")
    sys.exit(app.exec_())
