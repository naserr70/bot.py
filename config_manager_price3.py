"""
مدیریت پویای تمام تنظیمات برنامه
نسخه اصلاح شده - مسیرهای داینامیک
"""

import json
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import Any, Dict


# ═══════════════════════════════════════════════════════════════
# توابع مسیریابی داینامیک
# ═══════════════════════════════════════════════════════════════

def get_app_directory():
    """
    گرفتن مسیر پوشه برنامه

    - اگه EXE باشه: پوشه EXE
    - اگه اسکریپت باشه: پوشه اسکریپت
    """
    if getattr(sys, 'frozen', False):
        # اجرا از EXE
        return os.path.dirname(sys.executable)
    else:
        # اجرا از اسکریپت پایتون
        return os.path.dirname(os.path.abspath(__file__))


def get_default_paths():
    """
    ساخت مسیرهای پیش‌فرض نسبت به پوشه برنامه
    """
    app_dir = get_app_directory()

    # تعریف پوشه‌ها
    data_dir = os.path.join(app_dir, "data")
    models_dir = os.path.join(app_dir, "models")
    logs_dir = os.path.join(app_dir, "logs")

    # ساخت پوشه‌ها اگه نیستن
    for folder in [data_dir, models_dir, logs_dir]:
        os.makedirs(folder, exist_ok=True)

    return {
        'app_dir': app_dir,
        'data_dir': data_dir,
        'models_dir': models_dir,
        'logs_dir': logs_dir,
        'model_path': os.path.join(models_dir, "trading_model.h5"),
        'scaler_path': os.path.join(models_dir, "scaler.pkl"),
        'db_path': os.path.join(data_dir, "trading_data.db"),
        'log_csv': os.path.join(logs_dir, "trade_log.csv"),
        'last_train_file': os.path.join(data_dir, "last_train_status.txt"),
        'config_file': os.path.join(data_dir, "config.json"),
        'users_file': os.path.join(data_dir, "users.json"),
    }


# گرفتن مسیرهای پیش‌فرض (یک بار)
DEFAULT_PATHS = get_default_paths()


# ═══════════════════════════════════════════════════════════════
# کلاس‌های تنظیمات
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradingSettings:
    """تنظیمات معاملاتی"""
    symbol: str = "XAUUSD"
    timeframe: str = "M5"
    secondary_timeframe: str = "M30"
    signal_source: str = "Combined"
    max_positions: int = 1
    lot_size: float = 0.05
    tp_atr_mult: float = 2.5
    sl_atr_mult: float = 1.5
    # لیبل‌گذاری و حد سود/ضرر معامله:
    #   "Fixed" = فاصله‌ی ثابت قیمت (fixed_tp_price / fixed_sl_price، به دلار قیمت طلا)
    #   "ATR"   = ضریب ATR (tp_atr_mult / sl_atr_mult)
    label_mode: str = "ATR"
    fixed_tp_price: float = 7.0
    fixed_sl_price: float = 7.0
    min_confidence_buy: float = 0.1
    min_confidence_sell: float = 0.1
    min_confluence_score: float = 1.2
    check_interval: int = 5
    # تصمیم لحظه‌ای: سیگنال (مدل/اندیکاتور/RL) هر check_interval ثانیه روی کندل جاری حساب می‌شود و بدون
    # صبر برای بسته شدن کندل معامله باز می‌شود. False = فقط یک بار بعد از بسته شدن هر کندل.
    intrabar_decisions: bool = True
    # فیلتر اسپرد مدل/اندیکاتور: ورود وقتی اسپرد > max_spread_atr × ATR تایم‌فریم مدل ممنوع (0 = خاموش)
    max_spread_atr: float = 0.35
    # بستن معامله‌ی مدل بعد از future_steps کندل (همان افق لیبل)
    use_time_exit: bool = True
    trailing_active_atr: float = 1.0
    trailing_distance_mult: float = 0.8


    # ریسک و تریلینگ استاپ
    max_daily_loss_percent: float = 20.0
    max_position_percent: float = 3.0

    trailing_stop_mode: str = "Off"
    fixed_trigger_profit: float = 17.0
    fixed_lock_profit: float = 5.0
    dynamic_trigger_profit: float = 18.0
    dynamic_trail_distance: float = 9.0
    # اندیکاتورها
    use_rsi: bool = True
    use_macd: bool = True
    use_bollinger: bool = True
    use_atr: bool = True
    use_adx: bool = True
    use_stochastic: bool = True


@dataclass
class ModelSettings:
    """تنظیمات مدل ML"""
    model_path: str = ""
    scaler_path: str = ""
    timesteps: int = 60
    n_history: int = 2000
    future_steps: int = 24
    # یادگیری تقویتی (RL): حداکثر کندل نگه‌داری هر معامله و تعداد گام‌های شبیه‌سازی در آموزش
    rl_max_hold_bars: int = 24
    rl_train_steps: int = 500000
    # عامل اسکلپ RL تایم‌فریم‌های خودش را دارد (مستقل از timeframe / secondary_timeframe مدل)
    rl_timeframe: str = "M1"
    rl_secondary_timeframe: str = "M5"
    # حد ضرر عامل: "ATR" = rl_sl_atr_mult × ATR تایم‌فریم RL ، "Fixed" = rl_sl_price دلار حرکت قیمت
    rl_sl_mode: str = "ATR"
    rl_sl_atr_mult: float = 2.0
    rl_sl_price: float = 3.0
    # تعداد عامل‌های مستقل که تصمیمشان میانگین گرفته می‌شود، و حداکثر کندل تاریخچه برای آموزش
    rl_ensemble: int = 5
    rl_data_bars: int = 200000
    # فیلتر اسپرد عامل (× ATR تایم‌فریم RL) و فاصله‌ی آموزش دوباره‌ی عامل (ساعت)
    rl_max_spread_atr: float = 0.35
    rl_retrain_hours: int = 12
    finetune_interval_hours: int = 5
    min_samples_for_training: int = 1000
    batch_size: int = 128
    epochs: int = 40
    thresh_atr_up: float = 0.3
    thresh_atr_down: float = 0.3

    # معماری مدل
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dropout_rate_1: float = 0.4
    dropout_rate_2: float = 0.4
    learning_rate: float = 0.001

    def __post_init__(self):
        # اگه خالی بود، مسیر پیش‌فرض بذار
        if not self.model_path:
            self.model_path = DEFAULT_PATHS['model_path']
        if not self.scaler_path:
            self.scaler_path = DEFAULT_PATHS['scaler_path']


@dataclass
class DatabaseSettings:
    """تنظیمات پایگاه داده"""
    db_path: str = ""
    log_csv: str = ""
    last_train_file: str = ""
    max_rows: int = 8000
    backup_interval_hours: int = 24
    auto_clean_old_data: bool = True
    keep_days: int = 30

    def __post_init__(self):
        # اگه خالی بود، مسیر پیش‌فرض بذار
        if not self.db_path:
            self.db_path = DEFAULT_PATHS['db_path']
        if not self.log_csv:
            self.log_csv = DEFAULT_PATHS['log_csv']
        if not self.last_train_file:
            self.last_train_file = DEFAULT_PATHS['last_train_file']


@dataclass
class UISettings:
    """تنظیمات رابط کاربری"""
    theme: str = "dark"
    language: str = "en"
    font_size: int = 12
    chart_theme: str = "plotly_dark"
    show_grid: bool = True
    animation_enabled: bool = True
    auto_refresh_chart: bool = True
    refresh_interval: int = 5


@dataclass
class BacktestSettings:
    """تنظیمات بک‌تست"""
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_balance: float = 10000.0
    commission_per_trade: float = 0.0
    slippage: float = 0.0


# ═══════════════════════════════════════════════════════════════
# کلاس اصلی مدیریت کانفیگ
# ═══════════════════════════════════════════════════════════════

class ConfigManager:
    """مدیریت تمام تنظیمات برنامه"""

    def __init__(self):
        self.config_file = DEFAULT_PATHS['config_file']

        # ساخت آبجکت‌های تنظیمات
        self.trading = TradingSettings()
        self.model = ModelSettings()
        self.database = DatabaseSettings()
        self.ui = UISettings()
        self.backtest = BacktestSettings()

        # بارگذاری از فایل (اگه وجود داره)
        self.load_config()

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری"""
        return {
            'trading': asdict(self.trading),
            'model': asdict(self.model),
            'database': asdict(self.database),
            'ui': asdict(self.ui),
            'backtest': asdict(self.backtest)
        }

    def from_dict(self, data: Dict[str, Any]):
        """خواندن از دیکشنری"""
        if 'trading' in data:
            for key, value in data['trading'].items():
                if hasattr(self.trading, key):
                    setattr(self.trading, key, value)

        if 'model' in data:
            for key, value in data['model'].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)

        if 'database' in data:
            for key, value in data['database'].items():
                if hasattr(self.database, key):
                    setattr(self.database, key, value)

        if 'ui' in data:
            for key, value in data['ui'].items():
                if hasattr(self.ui, key):
                    setattr(self.ui, key, value)

        if 'backtest' in data:
            for key, value in data['backtest'].items():
                if hasattr(self.backtest, key):
                    setattr(self.backtest, key, value)

    def save_config(self):
        """ذخیره تنظیمات در فایل"""
        try:
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)

            data = self.to_dict()

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)

            print(f"[OK] Config saved: {self.config_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Save config failed: {e}")
            return False

    def load_config(self):
        """بارگذاری تنظیمات از فایل"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.from_dict(data)
                #print(f"[OK] Config loaded: {self.config_file}")
                return True
            except Exception as e:
                print(f"[WARNING] Load config error: {e}")
                print("Using default settings...")
                return False
        else:
            print(f"[INFO] No config file, creating defaults...")
            self.save_config()
            return True

    def reset_to_defaults(self):
        """ریست به مقادیر پیش‌فرض"""
        self.trading = TradingSettings()
        self.model = ModelSettings()
        self.database = DatabaseSettings()
        self.ui = UISettings()
        self.backtest = BacktestSettings()
        self.save_config()
        print("[OK] Config reset to defaults")

    def get_paths_info(self) -> Dict[str, str]:
        """اطلاعات مسیرها برای نمایش در Settings"""
        return {
            'App Directory': get_app_directory(),
            'Config File': self.config_file,
            'Model Path': self.model.model_path,
            'Scaler Path': self.model.scaler_path,
            'Database': self.database.db_path,
            'Trade Log': self.database.log_csv,
            'Training Status': self.database.last_train_file,
        }

    def print_paths(self):
        """چاپ مسیرها برای دیباگ"""
        print("\n" + "=" * 50)
        print("Current Paths:")
        print("=" * 50)
        for name, path in self.get_paths_info().items():
            print(f"  {name}: {path}")
        print("=" * 50 + "\n")


# ═══════════════════════════════════════════════════════════════
# تابع کمکی
# ═══════════════════════════════════════════════════════════════

def get_config():
    """گرفتن یک instance از ConfigManager"""
    return ConfigManager()


# ═══════════════════════════════════════════════════════════════
# تست
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Config Manager Test")
    print("=" * 60)

    config = ConfigManager()
    config.print_paths()

    print("\nTrading Settings:")
    print(f"  Symbol: {config.trading.symbol}")
    print(f"  Timeframe: {config.trading.timeframe}")
    print(f"  Lot Size: {config.trading.lot_size}")

    print("\nModel Settings:")
    print(f"  Model Path: {config.model.model_path}")
    print(f"  Epochs: {config.model.epochs}")

    input("\nPress Enter to exit...")
