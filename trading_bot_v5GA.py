# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys

# Fix Windows Console Encoding
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import ast
import bisect
import gc
import json
import re
import queue
import sqlite3
from collections import deque
import threading
import time
import traceback
import warnings
from datetime import datetime, timezone

import joblib
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import ta
from numpy.lib.stride_tricks import sliding_window_view

# TensorFlow و sklearn فقط در کد قدیمی شبکه‌ی عصبی (GRU) لازم‌اند؛ حالت‌های فعلی (GA / اندیکاتور / RL) از آن‌ها
# استفاده نمی‌کنند. import آن‌ها در شروع برنامه ده‌ها ثانیه و صدها مگابایت حافظه می‌گرفت، پس فقط هنگام نیاز بارگذاری می‌شوند.

warnings.filterwarnings('ignore')

# Event for stopping the bot
bot_stop_event = threading.Event()


# ═══════════════════════════════════════════════════════════════
# CONFIG - از ConfigManager با مسیرهای داینامیک
# ═══════════════════════════════════════════════════════════════
from config_manager_price3 import ConfigManager

config = ConfigManager()

BOT_COMMENT = "AI_bot"

# هر بار که منطق لیبل‌گذاری عوض شود این عدد بالا می‌رود تا لیبل‌های قدیمی دور ریخته شوند
# v4: بریرهای دقیق با اسپرد (مثل اجرای واقعی بروکر) + دو خروجی باینری (برد خرید / برد فروش)
# v5: ذخیره‌ی Open/High/Low/اسپرد/ATR هر کندل (لازم برای شبیه‌ساز یادگیری تقویتی)
LABEL_VERSION = 5

# هر بار که مجموعه‌ی فیچرها عوض شود این عدد بالا می‌رود تا فیچرهای ذخیره‌شده‌ی هر دو جدول دور ریخته شوند
# v2: ۱۷ فیچر فقط پرایس‌اکشن (OHLC)
FEATURE_VERSION = 2

# کندل‌های اول هر پنجره که اندیکاتورها/فیچرهایشان هنوز کامل نشده؛ هیچ‌وقت ذخیره یا پیش‌بینی نمی‌شوند.
# (بلندترین نگاه به عقب فیچرها ۵۱ کندل است؛ بقیه برای همگرا شدن ATR)
WARMUP_BARS = 100

# منبع سیگنال: فقط مدل، فقط اندیکاتورها، یا عامل یادگیری تقویتی — حالت‌های مستقل که هیچ اثری روی هم ندارند
SIGNAL_MODES = ("model", "indicators", "rl")

# قانون تصمیم حالت Model (اگر عوض شود مدل دوباره آموزش می‌بیند)
# ga_patterns_v2: قانون‌های متقارن (خرید + فروش آینه‌ای) الگوی کندلی/کلاسیک که الگوریتم ژنتیک پیدا کرده، فقط روی کندل بسته
# pattern_stats_v4: همه‌ی الگوها، هر کدام با حد سود/ضرر آماری خودش؛ پرسودترین الگوی برقرار معامله می‌شود
# pattern_stats_v5_breakout: + قانون شکست — بعد از کلوز بالای سطح شکست، تا کلوز دوباره زیر آن فروش نه (و برعکس)
# pattern_runs_v6: بدون محدودیت زمان؛ حد ضرر ثابت، حد سود هر الگو = میانگین بیشترین قیمت قبل از استاپ در گذشته‌اش؛
#                  فقط الگوهایی معامله می‌شوند که میانگین سود واقعی هر معامله‌شان به حداقل (PATTERN_MIN_PROFIT_USD) می‌رسد
# pattern_fixed_v7: حد سود/ضرر ثابت (PATTERN_TP_ATR / PATTERN_SL_ATR)؛ الگوهایی با نرخ برد ≥ PATTERN_MIN_WIN_RATE یا
#                  میانگین سود مثبت بعد از هزینه‌ها معامله می‌شوند؛ ۴۶ الگوی کندلی تازه (و نسخه‌ی «بعد از روند»)
# pattern_runs_v7: خروج و انتخاب v6 (حالت runs) + ۴۶ الگوی کندلی تازه و گزارش همه‌ی الگوها
# DECISION_RULE کنار PATTERN_EXIT_MODE تعریف می‌شود (از روی آن)، تا عوض شدن روش خروج مدل قبلی را ناسازگار کند

# برای برآورد محافظه‌کارانه‌ی اسپرد در لیبل‌گذاری: بیشترِ (اسپرد کندل، میانه‌ی اسپرد این تعداد کندل اخیر)
SPREAD_MEDIAN_WINDOW = 100

TF_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}

# تعداد کندل تاریخی برای جمع‌آوری اولیه‌ی داده
HISTORY_BARS = {
    "M1": 90000, "M5": 90000, "M15": 90000, "M30": 45000,
    "H1": 30000, "H4": 8000, "D1": 2000,
}

# تاریخچه‌ی تایم‌فریم اصلی مدل (الگوریتم ژنتیک): هر چه بیشتر، قانون‌های پایدارتر و Validation/Test مطمئن‌تر.
# اگر ترمینال این‌قدر ندهد (Max bars in chart)، بیشترین مقدار موجود گرفته می‌شود. دیتابیس حداکثر MODEL_DB_MAX_ROWS ردیف
# نگه می‌دارد. v6: آمار الگوها از کل تاریخچه گرفته می‌شود (آزمون ۱۱ ساله: با فقط ۲۰۰ هزار کندل آخر نتیجه ضعیف‌تر و
# ناپایدارتر بود)، پس هدف M5 حدود ۱۱ سال است.
# v9 (به خواست کاربر): حداکثر ۲۰۰ هزار کندل (M5 ≈ ۲.۸ سال)؛ مدل الگو هم فقط روی GA_TRAIN_BARS کندل آخر آموزش می‌بیند.
MODEL_HISTORY_BARS = {
    "M1": 200000, "M5": 200000, "M15": 200000, "M30": 130000,
    "H1": 65000, "H4": 16000, "D1": 4000,
}
# سقف همه‌ی داده‌های کندلی ذخیره‌شده (دیتابیس مدل، تایم‌فریم دوم، کندل خام RL) — به خواست کاربر ۲۰۰ هزار:
# هر بار که کندل تازه ذخیره می‌شود، به همان تعداد قدیمی‌ترین کندل‌ها پاک می‌شوند. همه‌ی خواندن‌ها بر اساس زمان و از
# جدیدترین کندل است، پس آموزش و فاین‌تون بعدی همیشه تازه‌ترین داده را دارد.
DB_MAX_BARS = 200000
MODEL_DB_MAX_ROWS = DB_MAX_BARS
HISTORY_BACKFILL_RETRY_DAYS = 3


def _safe_tag(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))


def build_profile_paths_from_base(symbol: str, tf_str: str,
                                  base_model_path: str, base_scaler_path: str,
                                  base_db_path: str, base_last_train_file: str,
                                  base_log_csv: str,
                                  is_dual: bool = False):
    """
    ساخت مسیرهای پروفایل برای یک تایم‌فریم خاص
    - is_dual=True برای ساخت مسیر مدل دوگانه
    """
    tag = f"{_safe_tag(symbol)}_{_safe_tag(tf_str)}"

    model_dir = os.path.dirname(base_model_path) or "."
    model_ext = os.path.splitext(base_model_path)[1] or ".h5"

    if is_dual:
        model_path = os.path.join(model_dir, f"model_dual_{tag}{model_ext}")
    else:
        model_path = os.path.join(model_dir, f"model_{tag}{model_ext}")

    scaler_dir = os.path.dirname(base_scaler_path) or "."
    scaler_path = os.path.join(scaler_dir, f"scaler_{tag}.pkl")

    db_dir = os.path.dirname(base_db_path) or "."
    db_path = os.path.join(db_dir, f"training_{tag}.db")

    lt_dir = os.path.dirname(base_last_train_file) or "."
    last_train_file = os.path.join(lt_dir, f"last_train_{tag}.txt")

    return model_path, scaler_path, db_path, last_train_file, base_log_csv


def get_mt5_timeframe(tf_str):
    """تبدیل string به MT5 timeframe"""
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    return tf_map.get(tf_str, mt5.TIMEFRAME_M5)


def get_secondary_timeframe_default(primary_tf_str):
    """اگه کاربر تایم‌فریم دوم رو انتخاب نکرده، یک پیش‌فرض منطقی بده"""
    defaults = {"M1": "M15", "M5": "M15", "M15": "H1", "M30": "H4", "H1": "H4", "H4": "D1"}
    return defaults.get(primary_tf_str, "M15")


RL_TF_OPTIONS = ("M1", "M5", "M15")


def get_rl_secondary_default(rl_tf):
    """تایم‌فریم زمینه‌ی پیش‌فرض عامل اسکلپ (بزرگ‌تر از تایم‌فریم اصلی‌اش)"""
    return {"M1": "M5", "M5": "M15", "M15": "H1", "M30": "H4", "H1": "H4"}.get(rl_tf, "M5")


def _read_rl_timeframes(m):
    """
    تایم‌فریم‌های عامل RL — مستقل از تایم‌فریم مدل (config.model.rl_timeframe / rl_secondary_timeframe).
    تایم‌فریم زمینه باید بزرگ‌تر از تایم‌فریم اصلی باشد؛ وگرنه پیش‌فرض استفاده می‌شود.
    """
    rl_tf = str(getattr(m, 'rl_timeframe', '') or 'M1').strip().upper()
    if rl_tf not in TF_SECONDS:
        rl_tf = "M1"
    rl_s = str(getattr(m, 'rl_secondary_timeframe', '') or '').strip().upper()
    if rl_s not in TF_SECONDS or TF_SECONDS[rl_s] <= TF_SECONDS[rl_tf]:
        rl_s = get_rl_secondary_default(rl_tf)
    return rl_tf, rl_s


def profile_tag(symbol, primary_tf, secondary_tf):
    return f"{_safe_tag(symbol)}_{_safe_tag(primary_tf)}_{_safe_tag(secondary_tf)}"


def build_dual_profile_paths(symbol: str, primary_tf: str, secondary_tf: str,
                             base_model_path: str, base_scaler_path: str,
                             base_db_path: str, base_last_train_file: str,
                             base_log_csv: str, rl_tf: str = None, rl_secondary_tf: str = None):
    """
    مدل و عامل یادگیری تقویتی هر کدام تایم‌فریم‌های خودشان را دارند و پوشه‌ی هر تنظیم
    (نماد + تایم‌فریم اصلی + تایم‌فریم دوم) جداست؛ هیچ فایلی بین دو تنظیم یا بین مدل و RL مشترک نیست:

      data/<XAUUSD_M5_M15>/model/   primary_M5.db, secondary_M15.db, last_train.txt   (لیبل‌ها و فیچرهای مدل)
      models/<XAUUSD_M5_M15>/model/ model_dual.h5 (+meta), scaler_primary_M5.pkl, scaler_secondary_M15.pkl
      data/<XAUUSD_M1_M5>/rl/       raw_M1.db, raw_M5.db, last_train.txt              (کندل خام عامل RL)
      models/<XAUUSD_M1_M5>/rl/     agent.npz (+meta)
    rl_tf/rl_secondary_tf = None: همان تایم‌فریم‌های مدل.
    """
    tag = profile_tag(symbol, primary_tf, secondary_tf)
    p_tag, s_tag = _safe_tag(primary_tf), _safe_tag(secondary_tf)
    rl_tf = rl_tf or primary_tf
    rl_secondary_tf = rl_secondary_tf or secondary_tf
    rl_tag = profile_tag(symbol, rl_tf, rl_secondary_tf)
    rp_tag, rs_tag = _safe_tag(rl_tf), _safe_tag(rl_secondary_tf)
    data_root = os.path.join(os.path.dirname(base_db_path) or ".", tag)
    models_root = os.path.join(os.path.dirname(base_model_path) or ".", tag)
    rl_data_root = os.path.join(os.path.dirname(base_db_path) or ".", rl_tag)
    rl_models_root = os.path.join(os.path.dirname(base_model_path) or ".", rl_tag)
    dirs = {k: os.path.join(root, sub) for k, root, sub in (
        ('data_model', data_root, 'model'), ('data_rl', rl_data_root, 'rl'),
        ('models_model', models_root, 'model'), ('models_rl', rl_models_root, 'rl'))}
    for d in dirs.values():
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass
    model_ext = os.path.splitext(base_model_path)[1] or ".h5"

    return {
        'profile_tag': tag,
        'profile_data_dir': data_root,
        'profile_models_dir': models_root,
        # مدل (لیبل‌دار)
        'primary_db_path': os.path.join(dirs['data_model'], f"primary_{p_tag}.db"),
        'secondary_db_path': os.path.join(dirs['data_model'], f"secondary_{s_tag}.db"),
        'dual_model_path': os.path.join(dirs['models_model'], f"model_dual{model_ext}"),
        'primary_scaler_path': os.path.join(dirs['models_model'], f"scaler_primary_{p_tag}.pkl"),
        'secondary_scaler_path': os.path.join(dirs['models_model'], f"scaler_secondary_{s_tag}.pkl"),
        'dual_last_train_file': os.path.join(dirs['data_model'], "last_train.txt"),
        # عامل یادگیری تقویتی (کاملاً جدا، با تایم‌فریم‌های خودش)
        'rl_profile_tag': rl_tag,
        'rl_primary_db_path': os.path.join(dirs['data_rl'], f"raw_{rp_tag}.db"),
        'rl_secondary_db_path': os.path.join(dirs['data_rl'], f"raw_{rs_tag}.db"),
        'rl_agent_path': os.path.join(dirs['models_rl'], "agent.npz"),
        'rl_last_train_file': os.path.join(dirs['data_rl'], "last_train.txt"),
        # لاگ معاملات مربوط به حساب است و مشترک می‌ماند
        'log_csv': base_log_csv,
    }


# مقداری که در config.trading.signal_source ذخیره می‌شود
SIGNAL_SOURCE_LABELS = {"model": "Model", "indicators": "Indicators", "rl": "RL"}
SIGNAL_MODE_TEXT = {"model": "MODEL (GA patterns)", "indicators": "INDICATORS only", "rl": "RL agent"}
_unknown_source_warned = None
_bad_fixed_warned = None


def _normalize_signal_mode(value):
    """'Model' → 'model' ، 'Indicators'/'Indicator' → 'indicators' ، 'RL' → 'rl' ؛ بقیه → None"""
    v = str(value or "").strip().lower()
    if v.startswith("indicator"):
        return "indicators"
    if v == "model":
        return "model"
    if v in ("rl", "reinforcement", "reinforcement learning"):
        return "rl"
    return None


def _read_signal_mode():
    """منبع سیگنال از config.trading.signal_source ؛ مقدار ناشناخته → 'model'"""
    global _unknown_source_warned
    raw = getattr(config.trading, 'signal_source', None)
    mode = _normalize_signal_mode(raw)
    if mode is None:
        if raw and _unknown_source_warned != raw:
            print(f"ℹ️ signal_source={raw!r} is not a supported mode (Model / Indicators / RL) — using Model")
            _unknown_source_warned = raw
        mode = "model"
    return mode


def _read_label_mode(t):
    """
    حالت لیبل‌گذاری و حد سود/ضرر:
      'fixed' : فاصله‌ی ثابت قیمت (مثلاً ۷ دلار حرکت قیمت طلا) — config: label_mode="Fixed"
      'atr'   : ضریب ATR (TP_ATR_MULT / SL_ATR_MULT)          — config: label_mode="ATR"
    فاصله‌ی ثابتِ صفر یا منفی معنی ندارد؛ در آن صورت حالت ATR استفاده می‌شود.
    """
    global _bad_fixed_warned
    mode = "atr" if str(getattr(t, 'label_mode', 'Fixed') or 'Fixed').strip().lower() == "atr" else "fixed"
    tp = float(getattr(t, 'fixed_tp_price', 7.0) or 0.0)
    sl = float(getattr(t, 'fixed_sl_price', 7.0) or 0.0)
    if mode == "fixed" and (tp <= 0 or sl <= 0):
        # تنظیمات هر چند ثانیه دوباره خوانده می‌شود؛ هشدار فقط یک بار برای هر مقدار
        if _bad_fixed_warned != (tp, sl):
            print(f"⚠️ Fixed TP/SL must be > 0 (got TP={tp}, SL={sl}) — using ATR mode")
            _bad_fixed_warned = (tp, sl)
        mode = "atr"
    return mode, tp, sl


def load_settings():
    """بارگذاری تنظیمات از config - نسخه‌ی دوگانه (Dual Timeframe)"""
    config.load_config()
    t = config.trading
    m = config.model

    symbol = t.symbol
    primary_tf_str = t.timeframe
    label_mode, fixed_tp, fixed_sl = _read_label_mode(t)
    secondary_tf_str = getattr(t, 'secondary_timeframe', None) \
        or get_secondary_timeframe_default(primary_tf_str)
    rl_tf_str, rl_secondary_tf_str = _read_rl_timeframes(m)

    dual_paths = build_dual_profile_paths(
        symbol, primary_tf_str, secondary_tf_str,
        m.model_path, m.scaler_path, config.database.db_path,
        config.database.last_train_file, config.database.log_csv,
        rl_tf=rl_tf_str, rl_secondary_tf=rl_secondary_tf_str
    )

    return {
        'SYMBOL': symbol,
        'PRIMARY_TF_STR': primary_tf_str,
        'SECONDARY_TF_STR': secondary_tf_str,
        'TIMEFRAME': get_mt5_timeframe(primary_tf_str),
        'SECONDARY_TIMEFRAME': get_mt5_timeframe(secondary_tf_str),
        'PRIMARY_TF_SECONDS': TF_SECONDS.get(primary_tf_str, 300),
        'SECONDARY_TF_SECONDS': TF_SECONDS.get(secondary_tf_str, 900),
        # تایم‌فریم‌های عامل یادگیری تقویتی (مستقل از مدل)
        'RL_TF_STR': rl_tf_str,
        'RL_SECONDARY_TF_STR': rl_secondary_tf_str,
        'RL_TIMEFRAME': get_mt5_timeframe(rl_tf_str),
        'RL_SECONDARY_TIMEFRAME': get_mt5_timeframe(rl_secondary_tf_str),
        'RL_TF_SECONDS': TF_SECONDS[rl_tf_str],
        'RL_SECONDARY_TF_SECONDS': TF_SECONDS[rl_secondary_tf_str],

        'MAX_POSITIONS': int(t.max_positions),
        'LOT': float(t.lot_size),
        'TP_ATR_MULT': float(t.tp_atr_mult),
        'SL_ATR_MULT': float(t.sl_atr_mult),
        # حالت لیبل/خروج: 'fixed' = فاصله‌ی ثابت قیمت (دلار قیمت طلا)، 'atr' = ضریب ATR
        'LABEL_MODE': label_mode,
        'FIXED_TP_PRICE': fixed_tp,
        'FIXED_SL_PRICE': fixed_sl,
        'CHECK_INTERVAL': max(1, int(t.check_interval)),
        'MIN_CONFLUENCE_SCORE': float(t.min_confluence_score),
        'MIN_MODEL_CONFIDENCE': float(t.min_confidence_buy),
        'MIN_CONFIDENCE_BUY': float(t.min_confidence_buy),
        'MIN_CONFIDENCE_SELL': float(getattr(t, 'min_confidence_sell', t.min_confidence_buy)),

        'MAX_DAILY_LOSS_PERCENT': float(t.max_daily_loss_percent),
        'MAX_POSITION_PERCENT': float(t.max_position_percent),
        'TRAILING_STOP_MODE': str(getattr(t, 'trailing_stop_mode', 'Off') or 'Off'),
        'FIXED_TRIGGER_PROFIT': float(getattr(t, 'fixed_trigger_profit', 10.0)),
        'FIXED_LOCK_PROFIT': float(getattr(t, 'fixed_lock_profit', 3.0)),
        'DYNAMIC_TRIGGER_PROFIT': float(getattr(t, 'dynamic_trigger_profit', 15.0)),
        'DYNAMIC_TRAIL_DISTANCE': float(getattr(t, 'dynamic_trail_distance', 5.0)),

        'MAGIC': int(getattr(t, 'magic_number', 123456)),
        'DEVIATION': int(getattr(t, 'deviation_points', 20)),
        'MAX_SPREAD_ATR': float(getattr(t, 'max_spread_atr', 0.35)),
        'USE_TIME_EXIT': bool(getattr(t, 'use_time_exit', True)),
        # تصمیم لحظه‌ای: مدل/اندیکاتور/RL هر CHECK_INTERVAL ثانیه روی کندل در حال شکل‌گیری حساب می‌شوند و
        # به‌محض خوب بودن شرایط معامله باز می‌شود (False = فقط یک بار بعد از بسته شدن هر کندل)
        'INTRABAR_DECISIONS': bool(getattr(t, 'intrabar_decisions', True)),
        # (قدیمی — دیگر استفاده نمی‌شود؛ جایش را SIGNAL_MODE = 'indicators' گرفته)
        'ALLOW_INDICATOR_ONLY': bool(getattr(t, 'allow_indicator_only_trading', False)),
        # هزینه‌ی اضافه به پوینت (کمیسیون رفت‌وبرگشت + اسلیپیج) که از بازده هر معامله کم می‌شود
        'EXTRA_COST_POINTS': float(getattr(t, 'extra_cost_points', 0.0) or 0.0),
        # منبع سیگنال (config.trading.signal_source): 'model' فقط مدل یا 'indicators' فقط اندیکاتورها
        'SIGNAL_MODE': _read_signal_mode(),

        # مسیرهای دوگانه
        'PRIMARY_DB_PATH': dual_paths['primary_db_path'],
        'SECONDARY_DB_PATH': dual_paths['secondary_db_path'],
        'MODEL_PATH': dual_paths['dual_model_path'],
        'PRIMARY_SCALER_PATH': dual_paths['primary_scaler_path'],
        'SECONDARY_SCALER_PATH': dual_paths['secondary_scaler_path'],
        'LAST_TRAIN_FILE': dual_paths['dual_last_train_file'],
        'LOG_CSV': dual_paths['log_csv'],
        'PROFILE_TAG': dual_paths['profile_tag'],
        'PROFILE_DATA_DIR': dual_paths['profile_data_dir'],
        'PROFILE_MODELS_DIR': dual_paths['profile_models_dir'],
        'RL_PROFILE_TAG': dual_paths['rl_profile_tag'],
        'RL_PRIMARY_DB_PATH': dual_paths['rl_primary_db_path'],
        'RL_SECONDARY_DB_PATH': dual_paths['rl_secondary_db_path'],
        'RL_AGENT_PATH': dual_paths['rl_agent_path'],
        'RL_LAST_TRAIN_FILE': dual_paths['rl_last_train_file'],

        'N_HISTORY': int(m.n_history),
        # یادگیری تقویتی: حداکثر تعداد کندل نگه داشتن هر معامله، و تعداد گام‌های شبیه‌سازی در آموزش
        'RL_MAX_HOLD': max(1, int(getattr(m, 'rl_max_hold_bars', 24) or 24)),
        'RL_TRAIN_STEPS': max(10000, int(getattr(m, 'rl_train_steps', 500000) or 500000)),
        # حد ضرر عامل اسکلپ (مستقل از TP/SL مدل): 'atr' = ضریب ATR تایم‌فریم اصلی، 'fixed' = فاصله‌ی ثابت قیمت
        'RL_SL_MODE': ("fixed" if str(getattr(m, 'rl_sl_mode', 'ATR') or 'ATR').strip().lower() == "fixed"
                       and float(getattr(m, 'rl_sl_price', 0) or 0) > 0 else "atr"),
        'RL_SL_ATR_MULT': max(0.1, float(getattr(m, 'rl_sl_atr_mult', 2.0) or 2.0)),
        'RL_SL_PRICE': float(getattr(m, 'rl_sl_price', 0) or 0),
        # تعداد عامل‌های مستقل (seed جدا) که تصمیمشان میانگین گرفته می‌شود، و حداکثر کندل خام برای آموزش
        'RL_ENSEMBLE': min(7, max(1, int(getattr(m, 'rl_ensemble', 3) or 3))),
        'RL_DATA_BARS': min(DB_MAX_BARS, max(20000, int(getattr(m, 'rl_data_bars', 200000) or 200000))),
        # فیلتر اسپرد و فاصله‌ی آموزش دوباره‌ی عامل RL (مستقل از مدل)
        'RL_MAX_SPREAD_ATR': max(0.0, float(getattr(m, 'rl_max_spread_atr',
                                                    getattr(t, 'max_spread_atr', 0.35)) or 0.0)),
        'RL_RETRAIN_HOURS': max(1.0, float(getattr(m, 'rl_retrain_hours', 0) or m.finetune_interval_hours or 12)),
        'TIMESTEPS': int(m.timesteps),
        'TIMESTEPS_SECONDARY': int(getattr(m, 'timesteps_secondary', 30)),
        'FUTURE_STEPS': max(1, int(m.future_steps)),
        'FINETUNE_INTERVAL_HOURS': float(m.finetune_interval_hours),
        'MIN_SAMPLES_FOR_TRAINING': int(m.min_samples_for_training),
        'BATCH_SIZE': int(m.batch_size),
        'EPOCHS': int(m.epochs),
        'THRESH_ATR_UP': float(m.thresh_atr_up),
        'THRESH_ATR_DOWN': float(m.thresh_atr_down),
        'MIN_TEST_TRADES': int(getattr(m, 'min_test_trades', 30)),
        'MIN_TEST_PROFIT_FACTOR': float(getattr(m, 'min_test_profit_factor', 1.15)),
        'MIN_TEST_EXPECTANCY_ATR': float(getattr(m, 'min_test_expectancy_atr', 0.05)),
    }


settings = load_settings()

# متغیرهای سراسری (main_app مستقیماً bot.SYMBOL و bot.MODEL_PATH و ... را می‌خواند)
SYMBOL = settings['SYMBOL']
PRIMARY_TF_STR = settings['PRIMARY_TF_STR']
SECONDARY_TF_STR = settings['SECONDARY_TF_STR']
TIMEFRAME = settings['TIMEFRAME']
SECONDARY_TIMEFRAME = settings['SECONDARY_TIMEFRAME']
PRIMARY_TF_SECONDS = settings['PRIMARY_TF_SECONDS']
SECONDARY_TF_SECONDS = settings['SECONDARY_TF_SECONDS']
RL_TF_STR = settings['RL_TF_STR']
RL_SECONDARY_TF_STR = settings['RL_SECONDARY_TF_STR']
RL_TIMEFRAME = settings['RL_TIMEFRAME']
RL_SECONDARY_TIMEFRAME = settings['RL_SECONDARY_TIMEFRAME']
RL_TF_SECONDS = settings['RL_TF_SECONDS']
RL_SECONDARY_TF_SECONDS = settings['RL_SECONDARY_TF_SECONDS']
MAX_POSITIONS = settings['MAX_POSITIONS']
LOT = settings['LOT']
TP_ATR_MULT = settings['TP_ATR_MULT']
SL_ATR_MULT = settings['SL_ATR_MULT']
LABEL_MODE = settings['LABEL_MODE']
FIXED_TP_PRICE = settings['FIXED_TP_PRICE']
FIXED_SL_PRICE = settings['FIXED_SL_PRICE']
CHECK_INTERVAL = settings['CHECK_INTERVAL']
MIN_CONFLUENCE_SCORE = settings['MIN_CONFLUENCE_SCORE']
MIN_MODEL_CONFIDENCE = settings['MIN_MODEL_CONFIDENCE']
MIN_CONFIDENCE_BUY = settings['MIN_CONFIDENCE_BUY']
MIN_CONFIDENCE_SELL = settings['MIN_CONFIDENCE_SELL']
MAX_DAILY_LOSS_PERCENT = settings['MAX_DAILY_LOSS_PERCENT']
MAX_POSITION_PERCENT = settings['MAX_POSITION_PERCENT']
TRAILING_STOP_MODE = settings['TRAILING_STOP_MODE']
FIXED_TRIGGER_PROFIT = settings['FIXED_TRIGGER_PROFIT']
FIXED_LOCK_PROFIT = settings['FIXED_LOCK_PROFIT']
DYNAMIC_TRIGGER_PROFIT = settings['DYNAMIC_TRIGGER_PROFIT']
DYNAMIC_TRAIL_DISTANCE = settings['DYNAMIC_TRAIL_DISTANCE']
MAGIC = settings['MAGIC']
DEVIATION = settings['DEVIATION']
MAX_SPREAD_ATR = settings['MAX_SPREAD_ATR']
USE_TIME_EXIT = settings['USE_TIME_EXIT']
INTRABAR_DECISIONS = settings['INTRABAR_DECISIONS']
ALLOW_INDICATOR_ONLY = settings['ALLOW_INDICATOR_ONLY']
EXTRA_COST_POINTS = settings['EXTRA_COST_POINTS']
SIGNAL_MODE = settings['SIGNAL_MODE']
PRIMARY_DB_PATH = settings['PRIMARY_DB_PATH']
SECONDARY_DB_PATH = settings['SECONDARY_DB_PATH']
MODEL_PATH = settings['MODEL_PATH']
PRIMARY_SCALER_PATH = settings['PRIMARY_SCALER_PATH']
SECONDARY_SCALER_PATH = settings['SECONDARY_SCALER_PATH']
LAST_TRAIN_FILE = settings['LAST_TRAIN_FILE']
LOG_CSV = settings['LOG_CSV']
PROFILE_TAG = settings['PROFILE_TAG']
PROFILE_DATA_DIR = settings['PROFILE_DATA_DIR']
PROFILE_MODELS_DIR = settings['PROFILE_MODELS_DIR']
RL_PROFILE_TAG = settings['RL_PROFILE_TAG']
RL_PRIMARY_DB_PATH = settings['RL_PRIMARY_DB_PATH']
RL_SECONDARY_DB_PATH = settings['RL_SECONDARY_DB_PATH']
RL_AGENT_PATH = settings['RL_AGENT_PATH']
RL_LAST_TRAIN_FILE = settings['RL_LAST_TRAIN_FILE']
N_HISTORY = settings['N_HISTORY']
RL_MAX_HOLD = settings['RL_MAX_HOLD']
RL_TRAIN_STEPS = settings['RL_TRAIN_STEPS']
RL_SL_MODE = settings['RL_SL_MODE']
RL_SL_ATR_MULT = settings['RL_SL_ATR_MULT']
RL_SL_PRICE = settings['RL_SL_PRICE']
RL_ENSEMBLE = settings['RL_ENSEMBLE']
RL_DATA_BARS = settings['RL_DATA_BARS']
RL_MAX_SPREAD_ATR = settings['RL_MAX_SPREAD_ATR']
RL_RETRAIN_HOURS = settings['RL_RETRAIN_HOURS']
TIMESTEPS = settings['TIMESTEPS']
TIMESTEPS_SECONDARY = settings['TIMESTEPS_SECONDARY']
FUTURE_STEPS = settings['FUTURE_STEPS']
FINETUNE_INTERVAL_HOURS = settings['FINETUNE_INTERVAL_HOURS']
MIN_SAMPLES_FOR_TRAINING = settings['MIN_SAMPLES_FOR_TRAINING']
BATCH_SIZE = settings['BATCH_SIZE']
EPOCHS = settings['EPOCHS']
THRESH_ATR_UP = settings['THRESH_ATR_UP']
THRESH_ATR_DOWN = settings['THRESH_ATR_DOWN']
MIN_TEST_TRADES = settings['MIN_TEST_TRADES']
MIN_TEST_PROFIT_FACTOR = settings['MIN_TEST_PROFIT_FACTOR']
MIN_TEST_EXPECTANCY_ATR = settings['MIN_TEST_EXPECTANCY_ATR']

_symbol_change_warned = False

# تنظیماتی که پروفایل را عوض می‌کنند (دیتابیس، لیبل‌ها، مدل و عامل RL خودشان را دارند)؛
# در ربات در حال اجرا اعمال نمی‌شوند — بعد از Stop (apply_profile_settings) یا ری‌استارت
PROFILE_KEYS = ('SYMBOL', 'PRIMARY_TF_STR', 'SECONDARY_TF_STR', 'RL_TF_STR', 'RL_SECONDARY_TF_STR',
                'TP_ATR_MULT', 'SL_ATR_MULT',
                'LABEL_MODE', 'FIXED_TP_PRICE', 'FIXED_SL_PRICE', 'RL_MAX_HOLD',
                'RL_SL_MODE', 'RL_SL_ATR_MULT', 'RL_SL_PRICE',
                'FUTURE_STEPS', 'TIMESTEPS', 'TIMESTEPS_SECONDARY', 'EXTRA_COST_POINTS')


def pending_profile_changes():
    """
    کدام تنظیم پروفایل (نماد، تایم‌فریم‌ها، حد سود/ضرر، تنظیمات RL، ...) در فایل تنظیمات با تنظیم فعال ربات فرق دارد.
    main_app بعد از ذخیره‌ی Settings: اگر چیزی فرق دارد، پروفایل را با apply_profile_settings عوض می‌کند.
    """
    try:
        new_settings = load_settings()
    except Exception as e:
        print(f"⚠️ Settings read failed: {e}")
        return []
    return [k for k in PROFILE_KEYS if new_settings[k] != settings[k]]


def apply_profile_settings():
    """
    نماد/تایم‌فریم/تنظیمات جدید را همین الان اعمال کن (بدون ری‌استارت): مسیر دیتابیس، مدل و عامل RL مال پروفایل جدید
    می‌شوند و بقیه‌ی توابع (دانلود، آموزش، تصمیم) از همین لحظه با آن کار می‌کنند.
    فقط وقتی حلقه‌ی ربات در حال اجرا نیست (main_app بعد از Stop صدا می‌زند)؛ شروع بعدی ربات با پروفایل جدید است.
    """
    global settings, _symbol_change_warned
    settings = load_settings()
    globals().update(settings)
    _symbol_change_warned = False
    # حافظه‌های موقت مال نماد/مدل قبلی‌اند
    set_active_model_meta({})
    _display_agent_cache.update(stamp=None, agent=None)
    _RL_PAT_CACHE.update(key=None, val=None, now=None, ts=None)
    _last_decision_log['key'] = None
    print(f"🔁 Active setup: {SYMBOL} | Model {PRIMARY_TF_STR}+{SECONDARY_TF_STR} | RL {RL_TF_STR}+{RL_SECONDARY_TF_STR}")
    return settings


def reload_runtime_settings():
    """
    بارگذاری مجدد تنظیمات در حین اجرا.
    تغییر نماد/تایم‌فریم (PROFILE_KEYS) در ربات در حال اجرا اعمال نمی‌شود (مدل و دیتابیس عوض می‌شوند)،
    پس در آن حالت هیچ تنظیمی اعمال نمی‌شود؛ بعد از Stop برنامه پروفایل را عوض می‌کند (apply_profile_settings).
    """
    global settings, _symbol_change_warned, SIGNAL_MODE

    try:
        new_settings = load_settings()
    except Exception as e:
        print(f"⚠️ Settings reload failed, keeping previous settings: {e}")
        return

    changed = [k for k in PROFILE_KEYS if new_settings[k] != settings[k]]
    if changed:
        # منبع سیگنال به مدل/دیتابیس وابسته نیست و همیشه اعمال می‌شود
        SIGNAL_MODE = settings['SIGNAL_MODE'] = new_settings['SIGNAL_MODE']
        if not _symbol_change_warned:
            print(f"⚠️ {', '.join(changed)} changed — applies when the bot is stopped (the app then switches to the "
                  f"new setup, downloads its data and trains it) or after a restart. Other settings are kept as before.")
            _symbol_change_warned = True
        return

    _symbol_change_warned = False
    settings = new_settings
    globals().update(settings)


def set_signal_mode(mode, config_manager=None):
    """
    تغییر منبع سیگنال ('model' یا 'indicators' یا 'rl'): در config.trading.signal_source ذخیره و فوراً اعمال می‌شود.
    main_app باید ConfigManager خودش را بدهد؛ وگرنه ذخیره‌ی بعدی main مقدار قدیمی را روی فایل می‌نویسد.
    ربات در حال اجرا از بررسی بعدی (در حالت تصمیم روی کندل بسته: از کندل بعدی) با حالت جدید تصمیم می‌گیرد.
    """
    global SIGNAL_MODE
    mode = _normalize_signal_mode(mode)
    if mode is None:
        raise ValueError(f"signal mode must be one of {tuple(SIGNAL_SOURCE_LABELS.values())}")
    cm = config_manager or config
    cm.trading.signal_source = SIGNAL_SOURCE_LABELS[mode]
    if cm is not config:
        config.trading.signal_source = SIGNAL_SOURCE_LABELS[mode]
    cm.save_config()
    SIGNAL_MODE = mode
    settings['SIGNAL_MODE'] = mode
    print(f"🔀 Signal source set to: {SIGNAL_MODE_TEXT[mode]}")
    return mode


def barrier_distances(atr):
    """
    فاصله‌ی حد سود و حد ضرر از قیمت ورود (به واحد قیمت) — مشترک بین لیبل‌گذاری و معامله‌ی زنده.
    حالت fixed: FIXED_TP_PRICE / FIXED_SL_PRICE ؛ حالت atr: ضریب × ATR
    """
    if LABEL_MODE == "fixed":
        return FIXED_TP_PRICE, FIXED_SL_PRICE
    return TP_ATR_MULT * atr, SL_ATR_MULT * atr


def _price_digits():
    """تعداد رقم اعشار قیمت نماد فعال (طلا ۲، EURUSD ۵)؛ None اگر هنوز معلوم نیست"""
    pt = float(_point_size.get(SYMBOL, 0.0) or 0.0)
    return max(0, min(8, int(round(-np.log10(pt))))) if pt > 0 else None


def _px_txt(x):
    """فاصله/مقدار قیمت در گزارش‌ها با دقت همان نماد: طلا «$7.00»، EURUSD «0.00160» (نه «$0.00»)"""
    d = _price_digits()
    if d is None:
        return f"{float(x):.5g}"
    return f"${float(x):.{d}f}" if d <= 2 else f"{float(x):.{d}f}"


def barrier_text():
    if LABEL_MODE == "fixed":
        return f"TP={_px_txt(FIXED_TP_PRICE)} SL={_px_txt(FIXED_SL_PRICE)} (fixed price move of {SYMBOL})"
    return f"TP={TP_ATR_MULT}×ATR SL={SL_ATR_MULT}×ATR"


def ret_unit():
    """واحد بازده‌ها در گزارش‌ها: حالت atr «ATR»؛ حالت fixed فاصله‌ی قیمت نماد («$» برای طلا، «price» برای جفت‌ارزها)"""
    if LABEL_MODE != "fixed":
        return "ATR"
    d = _price_digits()
    return "$" if d is None or d <= 2 else "price"


def min_test_expectancy():
    """
    حداقل سود مورد انتظار هر معامله روی Test، به واحد ret_unit().
    حالت atr: همان min_test_expectancy_atr (به ATR).
    حالت fixed: همان کسر، ولی از فاصله‌ی حد ضرر (0.05 × 7$ = 0.35$ حرکت قیمت در هر معامله).
    """
    if LABEL_MODE == "fixed":
        return MIN_TEST_EXPECTANCY_ATR * FIXED_SL_PRICE
    return MIN_TEST_EXPECTANCY_ATR


def feature_history_bars():
    """
    تعداد کندلی که برای اندیکاتورها/فیچرها و ذخیره‌ی داده‌ی تازه لازم است: N_HISTORY، ولی نه کمتر از
    warm-up + طول پنجره‌ی مدل + افق لیبل (وگرنه فیچرهای کندل‌های آخر ناقص می‌شوند و داده‌ی تازه ذخیره نمی‌شود).
    """
    need = WARMUP_BARS + max(TIMESTEPS, TIMESTEPS_SECONDARY) + FUTURE_STEPS + 50
    return max(int(N_HISTORY), int(need))


def live_history_bars():
    """
    تعداد کندل تایم‌فریم اصلی که برای تصمیم زنده لازم است: feature_history_bars، و نه کمتر از پنجره‌ی لازم برای
    الگوهای مدل (ga_warmup: شکست‌های چندروزه تا ۲۲ روز) + حاشیه. اندیکاتورها/فیچرها فقط روی feature_history_bars
    کندل آخر حساب می‌شوند (پنجره‌ی بلند فقط برای تشخیص الگو است؛ تصمیم دیرتر یعنی لغزش بیشتر).
    """
    return max(feature_history_bars(), int(ga_warmup() + 100))


def rl_live_bars():
    """تعداد کندل تایم‌فریم RL که برای تصمیم زنده‌ی عامل لازم است (warm-up ورودی‌ها + پنجره)"""
    return rl_raw_warmup() + RL_WINDOW_P + 100


def rl_raw_warmup(P=None, S=None):
    """کندل‌های اول هر سری که ورودی‌های عامل (و ویژگی‌های زمینه‌ی مدل الگو) هنوز کامل نیستند"""
    if not RL_CONTEXT_FEATURES:
        return RL_RAW_WARMUP
    return max(RL_RAW_WARMUP, ga_warmup(P or RL_TF_SECONDS, S or RL_SECONDARY_TF_SECONDS) + 10)


# ═══════════════════════════════════════════════════════════════
# MT5 CONNECTION
# ═══════════════════════════════════════════════════════════════
def ensure_mt5_connected(symbol=None):
    """اتصال به MT5 (در صورت نیاز) و فعال کردن نماد در Market Watch"""
    try:
        info = mt5.terminal_info()
        if info is None or not info.connected:
            if not mt5.initialize():
                print(f"✗ MT5 initialize failed: {mt5.last_error()}")
                return False
            info = mt5.terminal_info()
            if info is None or not info.connected:
                print("✗ MT5 terminal is not connected to the broker")
                return False

        symbol = symbol or SYMBOL
        if symbol and not mt5.symbol_select(symbol, True):
            print(f"✗ Cannot select symbol {symbol}: {mt5.last_error()}")
            return False
        return True
    except Exception as e:
        print(f"✗ MT5 connection error: {e}")
        return False


def init_mt5():
    if not ensure_mt5_connected():
        raise RuntimeError("MT5 initialization failed")
    print("✓ MT5 initialized")


def fetch_ohlcv(symbol, timeframe, n, closed_only=False):
    """
    گرفتن کندل‌ها از MT5.
    closed_only=True کندل در حال شکل‌گیری را حذف می‌کند (داده‌ی آموزش فقط از کندل‌های بسته ساخته می‌شود؛
    تصمیم لحظه‌ای ربات کندل در حال شکل‌گیری را هم می‌خواند).
    """
    try:
        start_pos = 1 if closed_only else 0
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, int(n))
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'tick_volume': 'Volume'
        }, inplace=True)
        return df
    except Exception as e:
        print(f"✗ fetch_ohlcv error: {e}")
        return None


def fetch_ohlcv_deep(symbol, timeframe, n, closed_only=True):
    """
    تا n کندل؛ اگر ترمینال این‌قدر تاریخچه ندهد (محدودیت «Max bars in chart» در تنظیمات MT5)
    با تعداد کمتر دوباره امتحان می‌کند و بیشترین تعداد موجود را برمی‌گرداند.
    """
    n = int(n)
    tried = []
    for k in (n, 800000, 600000, 400000, 300000, 200000, 100000, 50000, 20000):
        if k > n or k in tried:
            continue
        tried.append(k)
        df = fetch_ohlcv(symbol, timeframe, k, closed_only=closed_only)
        if df is not None and len(df) > 0:
            if k < n:
                print(f"ℹ️ Terminal returned {len(df)} bars (asked {n}); raise 'Max bars in chart' in MT5 "
                      f"options to get more history")
            return df
    return None


def _bar_epochs(df):
    """زمان باز شدن کندل‌ها به ثانیه (زمان سرور)"""
    if 'time' in df.columns:
        return df['time'].to_numpy(dtype='int64')
    return (pd.to_datetime(df['Date']).astype('datetime64[ns]').astype('int64') // 10**9).to_numpy()


def _drop_forming_bar(df, features_df, tf_seconds, symbol):
    """
    اگر آخرین کندل هنوز بسته نشده باشد حذفش کن.
    ذخیره‌ی کندل نیمه‌کاره در دیتابیس یعنی آموزش روی داده‌ای که بعداً عوض می‌شود.
    """
    if df is None or len(df) == 0 or not tf_seconds:
        return df, features_df
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or not tick.time:
            return df, features_df
        last_open = int(_bar_epochs(df)[-1])
        if last_open + tf_seconds > int(tick.time):
            return df.iloc[:-1], features_df.iloc[:-1]
    except Exception:
        pass
    return df, features_df


def _symbol_point(symbol):
    try:
        info = mt5.symbol_info(symbol)
        return float(info.point) if info else 0.0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════
# FIRST RUN DETECTION & AUTO SETUP
# ═══════════════════════════════════════════════════════════════
def check_first_run():
    """
    چک کردن وجود مدل حالت Model (استراتژی الگوریتم ژنتیک)
    """
    ga_path = ga_model_path()
    model_exists = os.path.exists(ga_path)

    print("\n[CHECK] GA Pattern Model Check:")
    print(f"   Model Path     : {ga_path}")
    print(f"   Model Exists   : {model_exists}")

    return not model_exists


def collect_training_data(symbol=None, primary_tf=None, secondary_tf=None,
                          primary_db=None, secondary_db=None,
                          primary_bars=None, secondary_bars=None):
    """
    دانلود + محاسبه‌ی فیچر + ذخیره‌ی داده‌ی هر دو تایم‌فریم (فقط کندل‌های بسته).
    خروجی: (تعداد ذخیره‌شده‌ی primary، تعداد ذخیره‌شده‌ی secondary)
    """
    symbol = symbol or SYMBOL
    primary_tf = primary_tf or PRIMARY_TF_STR
    secondary_tf = secondary_tf or SECONDARY_TF_STR
    primary_db = primary_db or PRIMARY_DB_PATH
    secondary_db = secondary_db or SECONDARY_DB_PATH
    primary_bars = primary_bars or MODEL_HISTORY_BARS.get(primary_tf, 90000)
    secondary_bars = secondary_bars or HISTORY_BARS.get(secondary_tf, 30000)

    saved_p = saved_s = 0

    # فقط داده‌ی مدل؛ کندل خام عامل RL تایم‌فریم‌های خودش را دارد (rl_sync_raw_bars / collect_raw_bars)
    # دانلود حجیم: اگر ترمینال این‌قدر تاریخچه ندهد، با تعداد کمتر دوباره امتحان می‌شود (fetch_ohlcv_deep)
    if primary_bars > live_history_bars() + 1:
        df_p = fetch_ohlcv_deep(symbol, get_mt5_timeframe(primary_tf), primary_bars, closed_only=True)
    else:
        df_p = fetch_ohlcv(symbol, get_mt5_timeframe(primary_tf), primary_bars, closed_only=True)
    if df_p is not None and len(df_p) > max(300, WARMUP_BARS + TIMESTEPS + FUTURE_STEPS + 50):
        df_p = compute_advanced_indicators(df_p)
        features_p = create_features_improved(df_p)
        saved_p = save_training_data_multiclass(
            df_p, features_p, db_path=primary_db, symbol=symbol,
            tf_seconds=TF_SECONDS.get(primary_tf, 300))
    else:
        print(f"[WARNING] Primary data insufficient ({0 if df_p is None else len(df_p)} bars)")

    df_s = fetch_ohlcv(symbol, get_mt5_timeframe(secondary_tf), secondary_bars, closed_only=True)
    if df_s is not None and len(df_s) > WARMUP_BARS + TIMESTEPS_SECONDARY + 50:
        df_s = compute_advanced_indicators(df_s)
        features_s = create_features_improved(df_s)
        saved_s = save_secondary_training_data(
            df_s, features_s, db_path=secondary_db, symbol=symbol,
            tf_seconds=TF_SECONDS.get(secondary_tf, 900))
    else:
        print(f"[WARNING] Secondary data insufficient ({0 if df_s is None else len(df_s)} bars)")

    return saved_p, saved_s


def _bars_since_last_saved(db_path, table, time_col, tf_seconds, server_now):
    """چند کندل از آخرین کندل ذخیره‌شده در این جدول گذشته (None = جدول خالی/ناموجود)"""
    try:
        conn = _db_connect(db_path)
        try:
            row = conn.execute(f"SELECT MAX({time_col}) FROM {table}").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if not row or row[0] is None:
        return None
    return max(0, (int(server_now) - int(row[0])) // int(tf_seconds))


def catch_up_bars():
    """
    تعداد کندل هر تایم‌فریم که بعد از ری‌استارت باید دانلود شود تا بین آخرین کندل ذخیره‌شده و الان شکافی
    نماند. پنجره‌ی زنده (live_history_bars) فقط چند هفته را پوشش می‌دهد؛ اگر ربات بیشتر خاموش بوده،
    ذخیره‌ی فقط همان پنجره در دیتابیس شکاف می‌سازد و پنجره‌های مدل و ATR/شبیه‌ساز RL روی دو کندل
    غیرمتوالی ساخته می‌شوند. خروجی: (تعداد کندل primary، تعداد کندل secondary)
    """
    base = live_history_bars()
    server_now = _server_now(SYMBOL)
    if server_now is None:
        return base, base
    margin = WARMUP_BARS + FUTURE_STEPS + 50

    def need(sources, tf_seconds, full):
        gaps = [_bars_since_last_saved(p, t, col, tf_seconds, server_now) for p, t, col in sources]
        if any(g is None for g in gaps):
            return full                               # یکی از دیتابیس‌ها خالی است: کل تاریخچه
        return int(min(max(base, max(gaps) + margin), full))

    n_p = need(((PRIMARY_DB_PATH, 'training_data_multiclass', 'timestamp_us'),), PRIMARY_TF_SECONDS,
               MODEL_HISTORY_BARS.get(PRIMARY_TF_STR, 90000))
    n_s = need(((SECONDARY_DB_PATH, 'training_data', 'timestamp'),), SECONDARY_TF_SECONDS,
               HISTORY_BARS.get(SECONDARY_TF_STR, 90000))
    return n_p, n_s


def history_backfill_needed(db_path=None, tf_str=None):
    """
    آیا تاریخچه‌ی ذخیره‌شده‌ی مدل از هدف (MODEL_HISTORY_BARS) کمتر است و باید یک بار کل تاریخچه دانلود شود؟
    (مثلاً بعد از بالا رفتن هدف در این نسخه). اگر ترمینال بیشتر از این نمی‌دهد، فقط هر چند روز یک بار دوباره امتحان می‌شود.
    """
    db_path = db_path or PRIMARY_DB_PATH
    tf_str = tf_str or PRIMARY_TF_STR
    target = int(MODEL_HISTORY_BARS.get(tf_str, 90000))
    have = count_labeled_samples(db_path)
    if have >= 0.9 * (target - WARMUP_BARS - FUTURE_STEPS):
        return False
    key = f"history_backfill_{target}"
    now = int(time.time())
    try:
        conn = _db_connect(db_path)
        try:
            conn.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            if row is not None and now - int(float(row[0])) < HISTORY_BACKFILL_RETRY_DAYS * 86400:
                return False
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(now)))
            conn.commit()
        finally:
            conn.close()
    except (sqlite3.Error, ValueError):
        return False
    print(f"📥 Model history has {have:,} bars (target {target:,}) — downloading deeper {tf_str} history once")
    return True


def collect_raw_bars(symbol=None, primary_bars=None, secondary_bars=None):
    """
    فقط کندل‌های خام هر دو تایم‌فریم را از MT5 می‌گیرد و ذخیره می‌کند (بدون فیچر و لیبل).
    داده‌ی عامل یادگیری تقویتی؛ مستقل از تنظیمات مدل. پیش‌فرض: تاریخچه‌ی عمیق (RL_HISTORY_BARS)،
    و برای تایم‌فریم دوم همان بازه‌ی زمانی.
    """
    symbol = symbol or SYMBOL
    init_raw_bars(RL_PRIMARY_DB_PATH)
    init_raw_bars(RL_SECONDARY_DB_PATH)
    n_p_req = int(primary_bars or rl_history_bars())
    df_p = fetch_ohlcv_deep(symbol, RL_TIMEFRAME, n_p_req, closed_only=True)
    got_p = len(df_p) if df_p is not None else n_p_req
    n_s_req = int(secondary_bars or (got_p * RL_TF_SECONDS // max(1, RL_SECONDARY_TF_SECONDS) + 500
                                     + rl_secondary_warmup()))
    df_s = fetch_ohlcv_deep(symbol, RL_SECONDARY_TIMEFRAME, n_s_req, closed_only=True)
    n_p = save_raw_bars(df_p, RL_PRIMARY_DB_PATH, symbol, RL_TF_SECONDS)
    n_s = save_raw_bars(df_s, RL_SECONDARY_DB_PATH, symbol, RL_SECONDARY_TF_SECONDS)
    print(f"[OK] RL raw bars saved: {RL_TF_STR}={n_p} | {RL_SECONDARY_TF_STR}={n_s}")
    return n_p, n_s


def rl_sync_raw_bars():
    """
    کندل‌های خام عامل RL را تا الان کامل کن (از ترد MT5): اگر داده کمتر از RL_DATA_BARS است تاریخچه‌ی عمیق،
    وگرنه فقط از آخرین کندل ذخیره‌شده تا الان — تا بین داده‌ی قبلی و کندل‌های زنده شکافی نماند.
    """
    need_s = int(0.9 * (RL_DATA_BARS * RL_TF_SECONDS // max(1, RL_SECONDARY_TF_SECONDS) + rl_secondary_warmup()))
    if count_raw_bars(RL_PRIMARY_DB_PATH) < RL_DATA_BARS or count_raw_bars(RL_SECONDARY_DB_PATH) < need_s:
        print(f"📥 Downloading deep {RL_TF_STR}/{RL_SECONDARY_TF_STR} history for the RL scalper...")
        return collect_raw_bars()
    server_now = _server_now(SYMBOL)
    gp = _bars_since_last_saved(RL_PRIMARY_DB_PATH, 'raw_bars', 'time', RL_TF_SECONDS, server_now) \
        if server_now else None
    gs = _bars_since_last_saved(RL_SECONDARY_DB_PATH, 'raw_bars', 'time', RL_SECONDARY_TF_SECONDS, server_now) \
        if server_now else None
    if gp is None or gs is None:
        return collect_raw_bars()
    n_p = int(min(max(rl_live_bars(), gp + 50), rl_history_bars()))
    n_s = int(max(RL_WINDOW_S + 100, gs + 50))
    return collect_raw_bars(primary_bars=n_p, secondary_bars=n_s)


def perform_first_run_setup(progress_callback=None):
    """
    اولین اجرا برای Dual Timeframe - جمع‌آوری داده + آموزش
    خروجی True یعنی ربات آماده‌ی کار است (یا باید صبر کند تا داده‌ی کافی جمع شود).
    """
    print("\n" + "=" * 60)
    print("FIRST RUN SETUP - Dual Timeframe")
    print("=" * 60)

    def update_progress(progress, message):
        print(f"[{progress*100:.0f}%] {message}")
        if progress_callback:
            try:
                progress_callback(progress, message)
            except Exception:
                pass

    update_progress(0.05, "Initializing Dual Setup...")

    update_progress(0.10, "Connecting to MT5...")
    if not ensure_mt5_connected():
        update_progress(0.10, "[ERROR] MT5 connection failed!")
        return False

    update_progress(0.15, "Creating databases...")
    init_database(db_path=PRIMARY_DB_PATH)
    init_database(db_path=SECONDARY_DB_PATH)

    update_progress(0.20, f"Downloading & labeling {PRIMARY_TF_STR} + {SECONDARY_TF_STR} history...")
    saved_p, saved_s = collect_training_data()
    print(f"[OK] Primary saved: {saved_p} | Secondary saved: {saved_s}")

    total_samples = count_labeled_samples(PRIMARY_DB_PATH)
    print(f"[INFO] Labeled samples in database: {total_samples}")

    if total_samples < MIN_SAMPLES_FOR_TRAINING:
        update_progress(0.90, f"[WARNING] Not enough data ({total_samples}/{MIN_SAMPLES_FOR_TRAINING})")
        update_progress(1.0, "Bot will collect more data and train later")
        return True

    update_progress(0.60, "Training pattern model (statistics of every pattern + genetic discovery)...")
    update_progress(0.65, "This usually takes a few minutes...")

    model = train_model_multiclass()

    if model is None:
        update_progress(1.0, "[WARNING] Model training failed — nothing saved (see log)")
        return False

    update_progress(1.0, "[OK] Dual Setup Finished Successfully!")
    print("\n" + "=" * 60)
    print("[SUCCESS] FIRST RUN SETUP COMPLETE!")
    print(f"     Symbol          : {SYMBOL}")
    print(f"     Primary TF      : {PRIMARY_TF_STR}")
    print(f"     Secondary TF    : {SECONDARY_TF_STR}")
    print(f"     Model (GA)      : {ga_model_path()}")
    print(f"     Total Samples   : {total_samples}")
    print("=" * 60 + "\n")
    return True


# ═══════════════════════════════════════════════════════════════
# Shared Signal State (برای ارتباط با GUI)
# ═══════════════════════════════════════════════════════════════
class SharedSignalState:
    """کلاس برای اشتراک‌گذاری سیگنال بین ربات و GUI"""
    def __init__(self):
        self.lock = threading.Lock()
        self.signal = 0
        self.confidence = 0.0
        self.model_conf = 0.0
        self.confluence_score = 0.0
        self.model_prob = 0.5
        self.atr = 0.0
        self.last_update = None
        self.reason = ""
        self.indicators = {}

    def update(self, signal, confidence, model_conf=0, confluence_score=0, model_prob=0.5,
               atr=0, reason="", indicators=None):
        with self.lock:
            self.signal = signal
            self.confidence = confidence
            self.model_conf = model_conf
            self.confluence_score = confluence_score
            self.model_prob = model_prob
            self.atr = atr
            self.reason = reason
            self.indicators = indicators or {}
            self.last_update = datetime.now(timezone.utc)

    def get(self):
        with self.lock:
            return {
                'signal': self.signal,
                'confidence': self.confidence,
                'model_conf': self.model_conf,
                'confluence_score': self.confluence_score,
                'model_prob': self.model_prob,
                'atr': self.atr,
                'reason': self.reason,
                'indicators': self.indicators.copy(),
                'last_update': self.last_update
            }

# ایجاد instance سراسری
shared_signal = SharedSignalState()


class TrainingState:
    """وضعیت آموزش برای نمایش در GUI"""
    def __init__(self):
        self.lock = threading.Lock()
        self.is_training = False
        self.training_started = None
        self.training_progress = 0.0  # 0.0 to 1.0
        self.current_epoch = 0
        self.total_epochs = 0

    def start(self, total_epochs):
        with self.lock:
            self.is_training = True
            self.training_started = datetime.now()
            self.total_epochs = total_epochs
            self.current_epoch = 0
            self.training_progress = 0.0

    def update_progress(self, current_epoch):
        with self.lock:
            self.current_epoch = current_epoch
            if self.total_epochs > 0:
                self.training_progress = current_epoch / self.total_epochs

    def finish(self):
        with self.lock:
            self.is_training = False
            self.training_progress = 1.0

    def get(self):
        with self.lock:
            return {
                'is_training': self.is_training,
                'started': self.training_started,
                'progress': self.training_progress,
                'epoch': f"{self.current_epoch}/{self.total_epochs}"
            }

# ایجاد instance سراسری
training_state = TrainingState()

# ========================
# اندیکاتورها: BB + RSI + MACD + ATR + Price Action
# (برای فیلتر Confluence و ATR لیبل/SL/TP؛ ورودی مدل فقط از create_features_improved است)
# ========================
def compute_advanced_indicators(df):
    """محاسبه 5 اندیکاتور اصلی"""
    df = df.copy()

    # === Price Action ===
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Close']
    df['CO_PCT'] = (df['Close'] - df['Open']) / df['Open']
    df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
    df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)

    # Price momentum
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_3'] = df['Close'].pct_change(3)

    # High/Low position
    df['High_5'] = df['High'].rolling(5).max()
    df['Low_5'] = df['Low'].rolling(5).min()
    df['Price_Position'] = (df['Close'] - df['Low_5']) / (df['High_5'] - df['Low_5'] + 1e-8)

    # === RSI ===
    df['RSI_7'] = ta.momentum.rsi(df['Close'], window=7)
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['RSI_21'] = ta.momentum.rsi(df['Close'], window=21)

    # === MACD ===
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    # خط سیگنال مثل MACD خود MT5: میانگین ساده‌ی ۹ تایی (کتابخانه‌ی ta میانگین نمایی می‌گیرد و کراسش با نمودار MT5 فرق دارد)
    df['MACD_signal'] = df['MACD'].rolling(9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # === Bollinger Bands ===
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_width'] + 1e-8)

    # === ATR ===

    df['ATR_5'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=5)
    df['ATR_7'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=7)
    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['ATR_21'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=21)
    df['ATR_25'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=25)
    df['ATR_Ratio'] = df['ATR_5'] / (df['ATR_25'] + 1e-8)

    # Volume
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-8)

    return df


# ========================
#  فیچرهای مدل: ۱۷ فیچر فقط پرایس‌اکشن (OHLC)
# ========================
FEATURE_NAMES = [
    # کالبد کندل فعلی
    'body', 'upper_wick', 'lower_wick', 'range', 'close_loc',
    # رابطه با کندل قبل
    'hh', 'll', 'close_vs_prev',
    # مومنتوم چندافقی
    'ret_5', 'ret_20',
    # ساختار بازار
    'range_pos_20', 'dist_high_50', 'dist_low_50', 'extreme_recency',
    # کیفیت روند و رژیم نوسان
    'efficiency_10', 'streak', 'range_regime',
]


def _bars_since_extreme(values, window, use_max):
    """
    چند کندل از آخرین سقف (use_max) یا کف پنجره‌ی `window` کندلی گذشته (0 = همین کندل).
    در تساوی جدیدترین حساب می‌شود. فقط به گذشته نگاه می‌کند.
    """
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < window:
        return out
    fill = -np.inf if use_max else np.inf
    win = sliding_window_view(np.where(np.isfinite(values), values, fill), window)[:, ::-1]
    pos = win.argmax(axis=1) if use_max else win.argmin(axis=1)
    out[window - 1:] = pos
    return out


def create_features_improved(df):
    """
    ۱۷ فیچر پرایس‌اکشن، همه فقط از Open/High/Low/Close.
    ATR فقط «واحد اندازه‌گیری» است (خودش هم از OHLC حساب می‌شود) تا فیچرها روی هر نماد و
    سطح قیمتی قابل مقایسه باشند. همه‌ی فیچرها تقریباً در بازه‌ی [-1, 1] محدود می‌شوند و
    فقط به کندل‌های فعلی و قبلی نگاه می‌کنند (بدون نگاه به آینده).

    کندل‌هایی که هنوز داده‌ی کافی برای فیچرها ندارند صفر می‌گیرند؛ این ردیف‌ها
    (اولین WARMUP_BARS کندل هر پنجره) هیچ‌وقت ذخیره یا برای پیش‌بینی استفاده نمی‌شوند.
    """
    o = df['Open'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    c = df['Close'].astype(float)
    if 'ATR_14' in df.columns:
        atr = df['ATR_14'].astype(float)
    else:
        atr = ta.volatility.average_true_range(h, l, c, window=14)
    a = atr.where(atr > 0)                     # ATR صفرِ دوره‌ی گرم شدن → NaN

    def by_atr(x, lo, hi):
        return (x / a).clip(lo, hi)

    top = np.maximum(o, c)
    bottom = np.minimum(o, c)
    rng = h - l
    h1, l1 = h.shift(1), l.shift(1)
    rng1 = h1 - l1

    f = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        # ══ 1. کالبد کندل ══
        f['body'] = by_atr(c - o, -3, 3) / 3                      # جهت و قدرت کندل
        f['upper_wick'] = by_atr(h - top, 0, 3) / 3               # فشار فروش در سقف
        f['lower_wick'] = by_atr(bottom - l, 0, 3) / 3            # فشار خرید در کف
        f['range'] = by_atr(rng, 0, 4) / 4                        # اندازه‌ی کندل نسبت به معمول
        f['close_loc'] = 2 * (c - l) / rng.where(rng > 0) - 1     # کلوز نزدیک سقف (+1) یا کف (-1)

        # ══ 2. رابطه با کندل قبل ══
        f['hh'] = by_atr(h - h1, -3, 3) / 3                       # سقف بالاتر/پایین‌تر
        f['ll'] = by_atr(l - l1, -3, 3) / 3                       # کف بالاتر/پایین‌تر
        f['close_vs_prev'] = (((c - l1) / rng1.where(rng1 > 0)).clip(-1, 2) - 0.5) / 1.5  # شکست/engulf

        # ══ 3. مومنتوم ══
        f['ret_5'] = by_atr(c - c.shift(5), -6, 6) / 6
        f['ret_20'] = by_atr(c - c.shift(20), -12, 12) / 12

        # ══ 4. ساختار بازار ══
        hh20 = h.rolling(20).max()
        ll20 = l.rolling(20).min()
        r20 = hh20 - ll20
        f['range_pos_20'] = 2 * (c - ll20) / r20.where(r20 > 0) - 1
        f['dist_high_50'] = by_atr(h.shift(1).rolling(50).max() - c, -3, 10) / 10   # منفی = شکست سقف
        f['dist_low_50'] = by_atr(c - l.shift(1).rolling(50).min(), -3, 10) / 10    # منفی = شکست کف
        since_high = _bars_since_extreme(h.to_numpy(), 20, use_max=True)
        since_low = _bars_since_extreme(l.to_numpy(), 20, use_max=False)
        f['extreme_recency'] = pd.Series((since_low - since_high) / 19.0, index=df.index)

        # ══ 5. کیفیت روند و رژیم نوسان ══
        dc = c.diff()
        path = dc.abs().rolling(10).sum()
        f['efficiency_10'] = ((c - c.shift(10)) / path.where(path > 0)).clip(-1, 1)

        direction = np.sign(dc.fillna(0.0))
        run_id = (direction != direction.shift()).cumsum()
        run_len = direction.groupby(run_id).cumcount() + 1
        f['streak'] = (direction * run_len).clip(-6, 6) / 6

        mean5 = rng.rolling(5).mean()
        mean50 = rng.rolling(50).mean()
        f['range_regime'] = np.log(mean5 / mean50.where(mean50 > 0)).clip(-1, 1)

    feature_df = pd.DataFrame(f, index=df.index)[FEATURE_NAMES]
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return feature_df


# ========================
# Confluence Signals (5 اندیکاتور)
# ========================
def get_confluence_signals(df):
    """محاسبه سیگنال‌های همسو از 5 اندیکاتور"""
    if len(df) < 2:
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = {}

    prev_2 = df.iloc[-3] if len(df) >= 3 else prev
    rsi_prev_2 = prev_2['RSI_14']
    bb_pos_prev_2 = prev_2['BB_position']

    # =======================================================================
    # 1. RSI SIGNALS (با قابلیت ماندگاری تا 1 کندل بعد)
    # =======================================================================
    rsi_now = last['RSI_14']
    rsi_prev = prev['RSI_14']

    inp_rsi_low_level = 30.0
    inp_rsi_high_level = 70.0

    # الف) بررسی کراس در کندل فعلی
    rsi_buy_fresh = (rsi_prev < inp_rsi_low_level and rsi_now >= inp_rsi_low_level)
    rsi_sell_fresh = (rsi_prev > inp_rsi_high_level and rsi_now <= inp_rsi_high_level)

    # ب) بررسی اینکه آیا کندل قبلی لحظه کراس بوده است (ماندگاری سیگنال)
    rsi_buy_delayed = (rsi_prev_2 < inp_rsi_low_level and rsi_prev >= inp_rsi_low_level)
    rsi_sell_delayed = (rsi_prev_2 > inp_rsi_high_level and rsi_prev <= inp_rsi_high_level)

    # تخصیص امتیاز (اگر کراس الان رخ داده یا ۱ کندل قبل رخ داده بود)
    if rsi_buy_fresh or rsi_buy_delayed:
        # برای تعیین قدرت، بر اساس کندلی که شروع‌کننده کراس بوده تصمیم می‌گیریم
        starting_rsi_prev = rsi_prev if rsi_buy_fresh else rsi_prev_2

        if starting_rsi_prev < 20.0:
            signals['RSI'] = +1.5
        else:
            signals['RSI'] = +1.0

    elif rsi_sell_fresh or rsi_sell_delayed:
        starting_rsi_prev = rsi_prev if rsi_sell_fresh else rsi_prev_2

        if starting_rsi_prev > 80.0:
            signals['RSI'] = -1.5
        else:
            signals['RSI'] = -1.0

    else:
        signals['RSI'] = 0.0


    # ═══════════════════════════════════════════════════════════════
    # 2. MACD
    # ═══════════════════════════════════════════════════════════════
    macd_hist = last['MACD_hist']
    macd_hist_prev = prev['MACD_hist']

    crossed_up = macd_hist > 0 and macd_hist_prev <= 0
    crossed_down = macd_hist < 0 and macd_hist_prev >= 0

    if crossed_up:
        signals['MACD'] = +1.4
    elif macd_hist > 0 and macd_hist > macd_hist_prev:
        signals['MACD'] = +0.8
    elif crossed_down:
        signals['MACD'] = -1.4
    elif macd_hist < 0 and macd_hist < macd_hist_prev:
        signals['MACD'] = -0.8
    else:
        signals['MACD'] = 0

    # =======================================================================
    # 2. BOLLINGER BANDS SIGNALS (با قابلیت ماندگاری تا 1 کندل بعد)
    # =======================================================================
    bb_pos_now = last['BB_position']
    bb_pos_prev = prev['BB_position']

    # الف) بررسی کراس در کندل فعلی
    bb_buy_fresh = (bb_pos_prev <= 0.05 and bb_pos_now > 0.05)
    bb_sell_fresh = (bb_pos_prev >= 0.95 and bb_pos_now < 0.95)

    # ب) بررسی کراس در کندل قبلی (ماندگاری سیگنال)
    bb_buy_delayed = (bb_pos_prev_2 <= 0.05 and bb_pos_prev > 0.05)
    bb_sell_delayed = (bb_pos_prev_2 >= 0.95 and bb_pos_prev < 0.95)

    if bb_buy_fresh or bb_buy_delayed:
        signals['BB'] = +1.0

    elif bb_sell_fresh or bb_sell_delayed:
        signals['BB'] = -1.0

    else:
        signals['BB'] = 0.0
    # ═══════════════════════════════════════════════════════════════
    # 5. Price Action
    # ═══════════════════════════════════════════════════════════════
    score = 0.0

    is_bullish = last['Is_Bullish']
    was_bullish = prev['Is_Bullish']
    body_size = last['Body_Size']
    avg_body = df['Body_Size'].rolling(20).mean().iloc[-1]
    big_candle = body_size > avg_body * 1.5

    upper_shadow = last['Upper_Shadow']
    lower_shadow = last['Lower_Shadow']

    # Hammer (bullish)
    if lower_shadow > body_size * 2 and upper_shadow < body_size:
        score += 0.8

    # Shooting Star (bearish)
    if upper_shadow > body_size * 2 and lower_shadow < body_size:
        score -= 0.8

    # Bullish Engulfing
    if is_bullish and not was_bullish:
        if last['Close'] > prev['Open'] and last['Open'] < prev['Close']:
            score += 1.2

    # Bearish Engulfing
    if not is_bullish and was_bullish:
        if last['Close'] < prev['Open'] and last['Open'] > prev['Close']:
            score -= 1.2

    # Strong candle
    if is_bullish and big_candle:
        score += 0.5
    elif not is_bullish and big_candle:
        score -= 0.5

    # Price position
    price_pos = last['Price_Position']
    if price_pos < 0.2:
        score += 0.3
    elif price_pos > 0.8:
        score -= 0.3

    signals['PriceAction'] = np.clip(score, -1.5, 1.5)

    # ═══════════════════════════════════════════════════════════════
    # 4. ATR (Volatility Confirmation)
    # ═══════════════════════════════════════════════════════════════
    atr_ratio = last['ATR_Ratio']

    if atr_ratio > 1.2:
        # مجموع امتیاز چهار اندیکاتور اصلی را محاسبه می‌کنیم
        base_confluence = signals.get('RSI', 0.0) + signals.get('MACD', 0.0) + signals.get('BB', 0.0) + signals.get('PriceAction', 0.0)

        if base_confluence > 0.0:
            # اگر جهت همسویی صعودی بود، امتیاز خرید را تقویت کن
            signals['ATR'] = +0.5
        elif base_confluence < 0.0:
            # اگر جهت همسویی نزولی بود، امتیاز فروش را تقویت کن
            signals['ATR'] = -0.5
        else:
            signals['ATR'] = 0.0

    else:
        # در بازارهای بی‌رمق و رنج (atr_ratio < 0.7) یا معمولی، ATR وزنی اضافه نمی‌کند
        signals['ATR'] = 0.0

    return signals

def calculate_confluence_score(signals):
    """محاسبه امتیاز همسویی - 5 اندیکاتور"""
    if not signals:
        return 0, {}

    total_score = sum(signals.values())
    strong_buy = sum(1 for v in signals.values() if v >= 0.7)
    strong_sell = sum(1 for v in signals.values() if v <= -0.7)

    details = {
        'total_score': total_score,
        'strong_buy': strong_buy,
        'strong_sell': strong_sell,
        'signals': signals
    }

    return total_score, details


def compute_confluence_scores(df):
    """
    نسخه‌ی وکتوریزه‌ی get_confluence_signals + calculate_confluence_score برای همه‌ی کندل‌ها.
    خروجی دقیقاً همان امتیازی است که get_confluence_signals برای df.iloc[:i+1] می‌دهد
    (در تست‌ها با نسخه‌ی تک‌کندلی مقایسه شده). برای ذخیره در دیتابیس و بک‌تست قانون زنده لازم است.
    """
    n = len(df)
    if n < 2:
        return np.zeros(n)

    def col(name):
        return df[name].to_numpy(dtype=float)

    def shifted(a, k):
        out = np.full_like(a, np.nan)
        out[k:] = a[:-k]
        return out

    rsi = col('RSI_14')
    rsi_p, rsi_p2 = shifted(rsi, 1), shifted(rsi, 2)
    rsi_p2[1] = rsi_p[1]   # با فقط دو کندل، prev_2 = prev
    with np.errstate(invalid='ignore'):
        buy_fresh = (rsi_p < 30) & (rsi >= 30)
        sell_fresh = (rsi_p > 70) & (rsi <= 70)
        buy_del = (rsi_p2 < 30) & (rsi_p >= 30)
        sell_del = (rsi_p2 > 70) & (rsi_p <= 70)
        start_b = np.where(buy_fresh, rsi_p, rsi_p2)
        start_s = np.where(sell_fresh, rsi_p, rsi_p2)
        s_rsi = np.where(buy_fresh | buy_del, np.where(start_b < 20, 1.5, 1.0),
                         np.where(sell_fresh | sell_del, np.where(start_s > 80, -1.5, -1.0), 0.0))

        h = col('MACD_hist')
        hp = shifted(h, 1)
        s_macd = np.select(
            [(h > 0) & (hp <= 0), (h > 0) & (h > hp), (h < 0) & (hp >= 0), (h < 0) & (h < hp)],
            [1.4, 0.8, -1.4, -0.8], default=0.0)

        bb = col('BB_position')
        bb_p, bb_p2 = shifted(bb, 1), shifted(bb, 2)
        bb_p2[1] = bb_p[1]
        bb_buy = ((bb_p <= 0.05) & (bb > 0.05)) | ((bb_p2 <= 0.05) & (bb_p > 0.05))
        bb_sell = ((bb_p >= 0.95) & (bb < 0.95)) | ((bb_p2 >= 0.95) & (bb_p < 0.95))
        s_bb = np.where(bb_buy, 1.0, np.where(bb_sell, -1.0, 0.0))

        body = col('Body_Size')
        upper, lower = col('Upper_Shadow'), col('Lower_Shadow')
        is_bull = col('Is_Bullish') != 0
        was_bull = shifted(col('Is_Bullish'), 1) != 0
        close, open_ = col('Close'), col('Open')
        prev_open, prev_close = shifted(open_, 1), shifted(close, 1)
        avg_body = df['Body_Size'].rolling(20).mean().to_numpy(dtype=float)
        big = body > avg_body * 1.5
        pos = col('Price_Position')

        pa = np.zeros(n)
        pa += np.where((lower > body * 2) & (upper < body), 0.8, 0.0)
        pa -= np.where((upper > body * 2) & (lower < body), 0.8, 0.0)
        pa += np.where(is_bull & ~was_bull & (close > prev_open) & (open_ < prev_close), 1.2, 0.0)
        pa -= np.where(~is_bull & was_bull & (close < prev_open) & (open_ > prev_close), 1.2, 0.0)
        pa += np.where(is_bull & big, 0.5, np.where(~is_bull & big, -0.5, 0.0))
        pa += np.where(pos < 0.2, 0.3, np.where(pos > 0.8, -0.3, 0.0))
        s_pa = np.clip(pa, -1.5, 1.5)

        base = s_rsi + s_macd + s_bb + s_pa
        s_atr = np.where(col('ATR_Ratio') > 1.2, np.sign(base) * 0.5, 0.0)

    total = s_rsi + s_macd + s_bb + s_pa + s_atr
    total[0] = 0.0
    return total


def _model_side(p_buy, p_sell, buy_threshold, sell_threshold):
    """
    جهت پیشنهادی مدل: جهتی که احتمال بردش بیشترین فاصله را از آستانه‌ی همان جهت دارد.
    خروجی: (side = 1/-1, احتمال برد همان جهت, آستانه‌ی همان جهت)
    """
    p_buy = np.asarray(p_buy, dtype=float)
    p_sell = np.asarray(p_sell, dtype=float)
    side = np.where(p_buy - buy_threshold >= p_sell - sell_threshold, 1, -1)
    p_side = np.where(side == 1, p_buy, p_sell)
    thr_side = np.where(side == 1, buy_threshold, sell_threshold)
    return side, p_side, thr_side


def decide_model_signals(p_buy, p_sell, buy_threshold, sell_threshold):
    """
    قانون تصمیم حالت «فقط مدل» — مشترک بین معامله‌ی زنده و بک‌تست اعتبارسنجی (وکتوریزه).
    مدل احتمال برد (رسیدن به TP قبل از SL) خرید و فروش را جدا می‌دهد؛ جهتی معامله می‌شود
    که احتمالش از آستانه‌ی همان جهت رد شده باشد. اندیکاتورها هیچ نقشی ندارند.
    خروجی: آرایه‌ی 1 / -1 / 0
    """
    side, p_side, thr_side = _model_side(p_buy, p_sell, buy_threshold, sell_threshold)
    return np.where(p_side >= thr_side, side, 0)


def decide_indicator_signals(confluence_score, min_confluence):
    """
    قانون تصمیم حالت «فقط اندیکاتور»: امتیاز Confluence ≥ MIN_CONFLUENCE_SCORE → خرید،
    ≤ −MIN_CONFLUENCE_SCORE → فروش. مدل هیچ نقشی ندارد.
    خروجی: آرایه‌ی 1 / -1 / 0
    """
    cs = np.asarray(confluence_score, dtype=float)
    thr = float(min_confluence)
    with np.errstate(invalid='ignore'):
        return np.where((cs >= thr) & (cs > 0), 1, np.where((cs <= -thr) & (cs < 0), -1, 0))


# ========================
# Database Functions
# ========================
def _db_connect(db_path):
    """اتصال SQLite با timeout (ترد آموزش و ترد اصلی هم‌زمان به دیتابیس دسترسی دارند)"""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    return conn


def _parse_features(text):
    """فیچرها قبلاً با str(list) و الان با json ذخیره می‌شوند؛ هر دو را پشتیبانی کن"""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return ast.literal_eval(text)


def feature_signature():
    """نسخه‌ی مجموعه‌ی فیچرها؛ فیچرهای ذخیره‌شده فقط با همین نسخه معتبرند"""
    return f"f{FEATURE_VERSION}|n{len(FEATURE_NAMES)}"


def label_signature():
    """پارامترهایی که معنی لیبل و ورودی مدل را تعیین می‌کنند؛ مدل فقط با همین‌ها معتبر است"""
    if LABEL_MODE == "fixed":
        return (f"v{LABEL_VERSION}|{feature_signature()}|fixed|tp$={FIXED_TP_PRICE}|sl$={FIXED_SL_PRICE}"
                f"|h={FUTURE_STEPS}|xc={EXTRA_COST_POINTS}")
    return (f"v{LABEL_VERSION}|{feature_signature()}|tp={TP_ATR_MULT}|sl={SL_ATR_MULT}"
            f"|h={FUTURE_STEPS}|xc={EXTRA_COST_POINTS}")


def training_signature():
    """امضای کامل مدل: معنی لیبل/ورودی + قانون تصمیمی که آستانه‌ها برایش تنظیم شده‌اند"""
    return f"{label_signature()}|rule={DECISION_RULE}"


def init_database(db_path=None):
    """
    ساخت/مهاجرت جدول‌ها.
    - training_data: فیچرهای تایم‌فریم دوم (بدون لیبل)
    - training_data_multiclass: فیچر + لیبل + بازده واقعی معامله‌ی خرید/فروش (برای اعتبارسنجی سودآوری)
    اگر نسخه‌ی فیچرها عوض شده باشد هر دو جدول، و اگر لیبل‌گذاری عوض شده باشد جدول لیبل‌دار
    خالی می‌شود تا از تاریخچه‌ی MT5 دوباره ساخته شوند.
    """
    if db_path is None:
        db_path = PRIMARY_DB_PATH

    conn = _db_connect(db_path)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER UNIQUE,
                features TEXT,
                target INTEGER,
                close_price REAL,
                symbol TEXT DEFAULT NULL,
                timeframe TEXT DEFAULT NULL
            )
        ''')
        for col_sql in ('close_price REAL', 'symbol TEXT', 'timeframe TEXT'):
            try:
                conn.execute(f'ALTER TABLE training_data ADD COLUMN {col_sql}')
            except sqlite3.OperationalError:
                pass  # ستون قبلاً وجود دارد
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ts ON training_data(timestamp)')

        # timestamp_us اسم قدیمی ستون است؛ مقدار آن زمان باز شدن کندل به ثانیه است
        # target: کد سه‌کلاسه برای سازگاری (2 خرید / 0 فروش / 1 هیچ‌کدام / -1 نامعتبر)
        # buy_win/sell_win: خروجی‌های مدل (1 = TP قبل از SL خورده) ؛ buy_exit/sell_exit: چند کندل بعد بسته شده
        conn.execute('''
            CREATE TABLE IF NOT EXISTS training_data_multiclass (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_us INTEGER UNIQUE,
                features TEXT,
                target INTEGER,
                close_price REAL,
                is_synthetic INTEGER DEFAULT 0,
                buy_ret REAL,
                sell_ret REAL,
                confluence REAL,
                buy_win INTEGER,
                sell_win INTEGER,
                buy_exit INTEGER,
                sell_exit INTEGER,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                spread_px REAL,
                atr REAL
            )
        ''')
        # open/high/low/spread_px/atr: برای شبیه‌ساز یادگیری تقویتی (قیمت‌ها Bid، اسپرد به واحد قیمت)
        for col_sql in ('is_synthetic INTEGER DEFAULT 0', 'buy_ret REAL', 'sell_ret REAL', 'confluence REAL',
                        'buy_win INTEGER', 'sell_win INTEGER', 'buy_exit INTEGER', 'sell_exit INTEGER',
                        'open_price REAL', 'high_price REAL', 'low_price REAL', 'spread_px REAL', 'atr REAL'):
            try:
                conn.execute(f'ALTER TABLE training_data_multiclass ADD COLUMN {col_sql}')
            except sqlite3.OperationalError:
                pass
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ts_mc ON training_data_multiclass(timestamp_us)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_target_mc ON training_data_multiclass(target)')

        conn.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
        name = os.path.basename(db_path)

        f_sig = feature_signature()
        row = conn.execute("SELECT value FROM meta WHERE key='feature_version'").fetchone()
        if row is None or str(row[0]) != f_sig:
            removed = conn.execute("DELETE FROM training_data").rowcount
            removed += conn.execute("DELETE FROM training_data_multiclass").rowcount
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('feature_version', ?)", (f_sig,))
            if removed:
                print(f"♻️ Feature set changed ({f_sig}) — discarded {removed} old rows in {name}; "
                      f"they will be rebuilt from MT5 history")

        l_sig = label_signature()
        row = conn.execute("SELECT value FROM meta WHERE key='label_version'").fetchone()
        if row is None or str(row[0]) != l_sig:
            removed = conn.execute("DELETE FROM training_data_multiclass").rowcount
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('label_version', ?)", (l_sig,))
            if removed:
                print(f"♻️ Labeling changed ({l_sig}) — discarded {removed} old labeled rows "
                      f"in {name}; they will be rebuilt from MT5 history")

        conn.commit()
    finally:
        conn.close()

    print(f"✓ Database initialized successfully: {os.path.basename(db_path)}")


# ═══════════════════════════════════════════════════════════════
# کندل‌های خام (مستقل از لیبل‌ها و فیچرها) — داده‌ی عامل یادگیری تقویتی
# این جدول با تغییر تنظیمات لیبل یا نسخه‌ی فیچرها هیچ‌وقت پاک نمی‌شود.
# ═══════════════════════════════════════════════════════════════
def _create_raw_bars_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS raw_bars (
            time INTEGER PRIMARY KEY,
            open REAL, high REAL, low REAL, close REAL,
            spread_px REAL,
            volume REAL
        )
    ''')
    try:
        conn.execute('ALTER TABLE raw_bars ADD COLUMN volume REAL')   # حجم تیکی (دیتابیس‌های قدیمی این ستون را نداشتند)
    except sqlite3.OperationalError:
        pass


def init_raw_bars(db_path):
    """فقط جدول کندل خام را می‌سازد (به جدول‌های لیبل دست نمی‌زند)"""
    conn = _db_connect(db_path)
    try:
        _create_raw_bars_table(conn)
        conn.commit()
    finally:
        conn.close()


RAW_BARS_KEEP = DB_MAX_BARS       # کندل خام RL: همان سقف ۲۰۰ هزار


def save_raw_bars(df, db_path, symbol=None, tf_seconds=None, keep=RAW_BARS_KEEP):
    """
    ذخیره‌ی کندل‌های بسته (Open/High/Low/Close به قیمت Bid، اسپرد به واحد قیمت، حجم تیکی).
    باید از ترد MT5 صدا زده شود (برای تبدیل اسپرد از پوینت به قیمت).
    """
    symbol = symbol or SYMBOL
    try:
        if df is None or len(df) == 0:
            return 0
        df, _ = _drop_forming_bar(df, df, tf_seconds, symbol)
        if len(df) == 0:
            return 0
        point = _symbol_point(symbol)
        _remember_contract_size(symbol)
        sp = (df['spread'].to_numpy(dtype=float) * point) if ('spread' in df.columns and point > 0) \
            else np.zeros(len(df))
        vol = df['Volume'].to_numpy(dtype=float) if 'Volume' in df.columns else np.full(len(df), np.nan)
        t = _bar_epochs(df)
        o, h, l, c = (df[k].to_numpy(dtype=float) for k in ('Open', 'High', 'Low', 'Close'))
        ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
        rows = [(int(t[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i]), float(max(sp[i], 0.0)),
                 float(vol[i]) if np.isfinite(vol[i]) else None)
                for i in np.flatnonzero(ok)]
        if not rows:
            return 0
        conn = _db_connect(db_path)
        try:
            _create_raw_bars_table(conn)
            conn.executemany('''
                INSERT INTO raw_bars (time, open, high, low, close, spread_px, volume) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(time) DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, spread_px=excluded.spread_px,
                    volume=COALESCE(excluded.volume, raw_bars.volume)
            ''', rows)
            conn.execute('DELETE FROM raw_bars WHERE time < (SELECT time FROM raw_bars '
                         'ORDER BY time DESC LIMIT 1 OFFSET ?)', (int(keep) - 1,))
            conn.commit()
        finally:
            conn.close()
        return len(rows)
    except Exception as e:
        print(f"✗ Raw bars save error in {os.path.basename(str(db_path))}: {e}")
        return 0


def count_raw_bars(db_path=None):
    try:
        conn = _db_connect(db_path or RL_PRIMARY_DB_PATH)
        try:
            _create_raw_bars_table(conn)
            return int(conn.execute("SELECT COUNT(*) FROM raw_bars").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def load_raw_bars(db_path, limit=None, since=None):
    """کندل‌های خام به ترتیب زمانی: DataFrame با ستون‌های time, open, high, low, close, spread_px, volume"""
    conn = _db_connect(db_path)
    try:
        _create_raw_bars_table(conn)
        q = "SELECT time, open, high, low, close, spread_px, volume FROM raw_bars"
        params = []
        if since is not None:
            q += " WHERE time >= ?"
            params.append(int(since))
        q += " ORDER BY time DESC"
        if limit:
            q += " LIMIT ?"
            params.append(int(limit))
        df = pd.read_sql_query(q, conn, params=params)
    finally:
        conn.close()
    return df.iloc[::-1].reset_index(drop=True)


def count_labeled_samples(db_path=None):
    """تعداد نمونه‌های واقعی با لیبل معتبر"""
    db_path = db_path or PRIMARY_DB_PATH
    try:
        conn = _db_connect(db_path)
        try:
            return int(conn.execute(
                "SELECT COUNT(*) FROM training_data_multiclass "
                "WHERE target >= 0 AND buy_win >= 0 AND sell_win >= 0 "
                "AND COALESCE(is_synthetic, 0) = 0"
            ).fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def compute_trade_outcomes_vectorized(highs, lows, closes, atrs, horizon,
                                      tp_mult, sl_mult, spreads=None, extra_cost=0.0):
    """
    لیبل‌گذاری Triple-Barrier دقیقاً مطابق اجرای واقعی بروکر.

    قیمت‌های کندل MT5 قیمت Bid هستند؛ Ask = Bid + اسپرد.
      خرید : ورود با Ask = c0 + s0 ؛ SL/TP نسبت به Ask ورود ؛ بسته شدن با Bid
             TP وقتی Bid ≥ c0 + s0 + TP×ATR ، SL وقتی Bid ≤ c0 + s0 − SL×ATR
      فروش : ورود با Bid = c0 ؛ بسته شدن با Ask (= Bid + اسپرد همان کندل)
             TP وقتی Low + s ≤ c0 − TP×ATR ، SL وقتی High + s ≥ c0 + SL×ATR
    پس اسپرد در احتمال برخورد به بریرها دیده می‌شود و بازده برد/باخت دقیقاً +TP / −SL است
    (منهای extra_cost مثل کمیسیون). خروج زمانی بعد از horizon کندل با Close همان کندل.
    اگر TP و SL در یک کندل لمس شوند، ضرر حساب می‌شود (محافظه‌کارانه؛ ترتیب داخل کندل نامعلوم است).

    spreads    : اسپرد هر کندل به واحد قیمت (None = صفر)
    extra_cost : هزینه‌ی اضافه به واحد قیمت برای هر معامله

    خروجی dict:
      labels            : 2 خرید برنده، 0 فروش برنده، 1 هیچ‌کدام، -1 نامعتبر (سازگاری با نسخه‌ی قبل)
      buy_win, sell_win : 1 = TP قبل از SL خورد، 0 = نخورد، -1 = نامعتبر
      buy_ret, sell_ret : بازده بر حسب ATR
      buy_exit, sell_exit : معامله چند کندل بعد از ورود بسته شد (1..horizon)، -1 = نامعتبر
    """
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    atrs = np.asarray(atrs, dtype=float)
    n = len(closes)
    if spreads is None:
        spreads = np.zeros(n)
    spreads = np.clip(np.nan_to_num(np.asarray(spreads, dtype=float), nan=0.0, posinf=0.0), 0.0, None)

    out = {
        'labels': np.full(n, -1, dtype=np.int32),
        'buy_win': np.full(n, -1, dtype=np.int32),
        'sell_win': np.full(n, -1, dtype=np.int32),
        'buy_ret': np.full(n, np.nan),
        'sell_ret': np.full(n, np.nan),
        'buy_exit': np.full(n, -1, dtype=np.int32),
        'sell_exit': np.full(n, -1, dtype=np.int32),
        'buy_timeout': np.zeros(n, dtype=bool),
        'sell_timeout': np.zeros(n, dtype=bool),
    }

    valid_n = n - horizon
    if horizon <= 0 or valid_n <= 0:
        return out

    c0 = closes[:valid_n]
    a0 = atrs[:valid_n]
    s0 = spreads[:valid_n]
    valid = np.isfinite(c0) & np.isfinite(a0) & (a0 > 0)
    a = np.where(valid, a0, 1.0)

    # پنجره‌ی کندل‌های i+1 .. i+horizon
    hw = sliding_window_view(highs[1:], horizon)[:valid_n]
    lw = sliding_window_view(lows[1:], horizon)[:valid_n]
    sw = sliding_window_view(spreads[1:], horizon)[:valid_n]

    never = horizon

    def first_hit(mask):
        return np.where(mask.any(axis=1), mask.argmax(axis=1), never)

    buy_entry = c0 + s0
    b_tp = first_hit(hw >= (buy_entry + tp_mult * a)[:, None])
    b_sl = first_hit(lw <= (buy_entry - sl_mult * a)[:, None])
    s_tp = first_hit(lw + sw <= (c0 - tp_mult * a)[:, None])
    s_sl = first_hit(hw + sw >= (c0 + sl_mult * a)[:, None])

    buy_win = b_tp < b_sl
    buy_loss = (b_sl <= b_tp) & (b_sl < never)
    sell_win = s_tp < s_sl
    sell_loss = (s_sl <= s_tp) & (s_sl < never)

    exit_bid = closes[horizon:horizon + valid_n]
    exit_spread = spreads[horizon:horizon + valid_n]
    b_time = (exit_bid - buy_entry) / a
    s_time = (c0 - (exit_bid + exit_spread)) / a

    cost = float(extra_cost or 0.0) / a
    br = np.where(buy_win, tp_mult,
                  np.where(buy_loss, -sl_mult, np.clip(b_time, -sl_mult, tp_mult))) - cost
    sr = np.where(sell_win, tp_mult,
                  np.where(sell_loss, -sl_mult, np.clip(s_time, -sl_mult, tp_mult))) - cost

    b_exit = np.where(buy_win, b_tp + 1, np.where(buy_loss, b_sl + 1, horizon))
    s_exit = np.where(sell_win, s_tp + 1, np.where(sell_loss, s_sl + 1, horizon))

    lab = np.ones(valid_n, dtype=np.int32)
    lab[buy_win & ~sell_win] = 2
    lab[sell_win & ~buy_win] = 0
    both = buy_win & sell_win
    lab[both & (b_tp < s_tp)] = 2
    lab[both & (s_tp < b_tp)] = 0

    lab[~valid] = -1
    br[~valid] = np.nan
    sr[~valid] = np.nan

    out['labels'][:valid_n] = lab
    out['buy_win'][:valid_n] = np.where(valid, buy_win.astype(np.int32), -1)
    out['sell_win'][:valid_n] = np.where(valid, sell_win.astype(np.int32), -1)
    out['buy_ret'][:valid_n] = br
    out['sell_ret'][:valid_n] = sr
    out['buy_exit'][:valid_n] = np.where(valid, b_exit, -1)
    out['sell_exit'][:valid_n] = np.where(valid, s_exit, -1)
    out['buy_timeout'][:valid_n] = valid & ~buy_win & ~buy_loss
    out['sell_timeout'][:valid_n] = valid & ~sell_win & ~sell_loss
    return out


_spread_warning_shown = False
_contract_size = {}


_point_size = {}
_swap_info = {}


def _swap_per_night(info):
    """
    سواپ هر شب برای ۱ واحد حجم، به واحد قیمت (یا کسری از قیمت در حالت بهره): {'long', 'short', 'frac', 'triple_weekday'}
    triple_weekday: روز سواپ سه‌برابر با دوشنبه = 0. حالت‌های ناشناخته → صفر.
    """
    mode = int(getattr(info, 'swap_mode', 0) or 0)
    s_long, s_short = float(getattr(info, 'swap_long', 0) or 0), float(getattr(info, 'swap_short', 0) or 0)
    point = float(getattr(info, 'point', 0) or 0)
    size = float(getattr(info, 'trade_contract_size', 0) or 0)
    frac = False
    if mode == getattr(mt5, 'SYMBOL_SWAP_MODE_POINTS', 1):
        long_px, short_px = s_long * point, s_short * point
    elif mode in (getattr(mt5, 'SYMBOL_SWAP_MODE_CURRENCY_SYMBOL', 2), getattr(mt5, 'SYMBOL_SWAP_MODE_CURRENCY_MARGIN', 3),
                  getattr(mt5, 'SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT', 4)) and size > 0:
        long_px, short_px = s_long / size, s_short / size        # پول هر لات ÷ اندازه‌ی قرارداد = حرکت قیمت معادل
    elif mode in (getattr(mt5, 'SYMBOL_SWAP_MODE_INTEREST_CURRENT', 5), getattr(mt5, 'SYMBOL_SWAP_MODE_INTEREST_OPEN', 6)):
        long_px, short_px, frac = s_long / 100.0 / 360.0, s_short / 100.0 / 360.0, True
    else:
        long_px = short_px = 0.0
    triple = (int(getattr(info, 'swap_rollover3days', 3) or 3) - 1) % 7     # MT5: یکشنبه = 0
    return {'long': long_px, 'short': short_px, 'frac': frac, 'triple_weekday': triple, 'mode': mode}


def _remember_contract_size(symbol):
    """
    اندازه‌ی قرارداد (طلا: ۱۰۰ اونس)، اندازه‌ی پوینت و سواپ شبانه‌ی نماد را نگه می‌دارد تا ترد آموزش
    بدون دست زدن به MT5 بتواند سود را به دلار حساب، هزینه‌ی اضافه را به قیمت تبدیل و سواپ را شبیه‌سازی کند.
    """
    try:
        info = mt5.symbol_info(symbol)
        size = float(getattr(info, 'trade_contract_size', 0) or 0)
        if size > 0:
            _contract_size[symbol] = size
        point = float(getattr(info, 'point', 0) or 0)
        if point > 0:
            _point_size[symbol] = point
        if info is not None:
            _swap_info[symbol] = _swap_per_night(info)
        _remember_commission(symbol)
    except Exception:
        pass


_commission_frac = {}        # نماد → کمیسیون رفت‌وبرگشت به‌صورت کسری از قیمت (از معاملات واقعی همین حساب)
_commission_checked = {}


def _remember_commission(symbol):
    """
    کمیسیون رفت‌وبرگشت هر معامله (کسری از ارزش معامله) از معاملات ۳۰ روز اخیر همین حساب؛ هر ۶ ساعت یک بار.
    شبیه‌ساز مدل الگو آن را به هزینه‌ی هر معامله اضافه می‌کند. اگر معامله‌ی کافی نباشد MODEL_COMMISSION_FRAC می‌ماند.
    """
    now = time.time()
    if now - _commission_checked.get(symbol, 0.0) < 6 * 3600:
        return
    _commission_checked[symbol] = now
    try:
        size = float(_contract_size.get(symbol) or 0)
        deals = mt5.history_deals_get(datetime.fromtimestamp(now - 30 * 86400, tz=timezone.utc),
                                      datetime.fromtimestamp(now + 86400, tz=timezone.utc)) or []
        com = notional = 0.0
        cnt = 0
        for d_ in deals:
            if d_.symbol == symbol and float(d_.volume) > 0 and float(d_.price) > 0:
                com += abs(float(d_.commission)) + abs(float(getattr(d_, 'fee', 0.0) or 0.0))
                notional += float(d_.volume) * size * float(d_.price)
                cnt += 1
        if size > 0 and notional > 0 and cnt >= 10:
            _commission_frac[symbol] = 2.0 * com / notional
    except Exception:
        pass


def money_per_trade(price_move, symbol=None):
    """حرکت قیمت → دلار حساب با LOT فعلی (فقط اگر اندازه‌ی قرارداد معلوم باشد)"""
    size = _contract_size.get(symbol or SYMBOL)
    return None if not size else price_move * size * LOT


def _conservative_spreads(df, point):
    """
    اسپرد هر کندل به واحد قیمت، محافظه‌کارانه: بیشترِ (اسپرد خود کندل، میانه‌ی اسپرد کندل‌های اخیر).
    ستون spread کندل‌های MT5 یک عدد برای کل کندل است و جهش‌های اسپرد را نشان نمی‌دهد.
    """
    global _spread_warning_shown
    if 'spread' not in df.columns or point <= 0:
        if not _spread_warning_shown:
            print("⚠️ No spread data (or symbol point unknown) — labels are built without spread")
            _spread_warning_shown = True
        return np.zeros(len(df))
    raw = df['spread'].astype(float).clip(lower=0) * point
    med = raw.rolling(SPREAD_MEDIAN_WINDOW, min_periods=1).median()
    return np.maximum(raw.to_numpy(), med.to_numpy())


def _stored_timestamps(db_path, table, time_col, lo, hi, complete_sql=None):
    """
    زمان کندل‌هایی از بازه‌ی [lo, hi] که قبلاً (و کامل) در جدول ذخیره شده‌اند.
    ردیف ذخیره‌شده‌ی یک کندل بسته دیگر عوض نمی‌شود (فیچر فقط به گذشته نگاه می‌کند و لیبل فقط وقتی ذخیره می‌شود
    که افقش کامل باشد)، پس لازم نیست هر کندل کل پنجره‌ی زنده دوباره نوشته شود.
    """
    try:
        conn = _db_connect(db_path)
        try:
            q = f"SELECT {time_col} FROM {table} WHERE {time_col} BETWEEN ? AND ?"
            if complete_sql:
                q += f" AND {complete_sql}"
            return {int(r[0]) for r in conn.execute(q, (int(lo), int(hi)))}
        finally:
            conn.close()
    except sqlite3.Error:
        return set()


def save_training_data_multiclass(df, features_df, db_path=None,
                                  balance_method='none',
                                  max_oversample_ratio=2.0,
                                  symbol=None, tf_seconds=None):
    """
    ذخیره‌ی داده‌ی تایم‌فریم اصلی با لیبل.

    - اولین WARMUP_BARS کندل هر پنجره ذخیره نمی‌شود: فیچرها و ATR آن‌ها هنوز کامل نشده‌اند و
      بازنویسی‌شان داده‌ی سالم دیتابیس را خراب می‌کرد.
    - Balancing عمداً انجام نمی‌شود: نمونه‌ی مصنوعی وسط سری زمانی، پنجره‌های GRU و مجموعه‌ی تست را
      آلوده می‌کند.
    """
    if db_path is None:
        db_path = PRIMARY_DB_PATH
    symbol = symbol or SYMBOL
    if tf_seconds is None:
        tf_seconds = PRIMARY_TF_SECONDS
    if balance_method not in (None, 'none'):
        print(f"ℹ️ balance_method='{balance_method}' ignored")

    try:
        df, features_df = _drop_forming_bar(df, features_df, tf_seconds, symbol)
        n = len(df)
        m = n - FUTURE_STEPS
        if m <= WARMUP_BARS:
            return 0

        closes = df['Close'].to_numpy(dtype=float)
        atrs = df['ATR_14'].to_numpy(dtype=float)
        point = _symbol_point(symbol)
        _remember_contract_size(symbol)
        spreads = _conservative_spreads(df, point)

        # حالت fixed: واحد بریر = ۱ دلار قیمت، پس TP/SL مستقیماً فاصله‌ی قیمت‌اند و بازده‌ها به دلار قیمت
        if LABEL_MODE == "fixed":
            unit, tp_mult, sl_mult = np.ones(n), FIXED_TP_PRICE, FIXED_SL_PRICE
        else:
            unit, tp_mult, sl_mult = atrs, TP_ATR_MULT, SL_ATR_MULT

        out = compute_trade_outcomes_vectorized(
            df['High'].to_numpy(dtype=float), df['Low'].to_numpy(dtype=float),
            closes, unit, FUTURE_STEPS, tp_mult, sl_mult,
            spreads=spreads, extra_cost=EXTRA_COST_POINTS * point
        )

        confluence = compute_confluence_scores(df)
        feat_arr = features_df.to_numpy(dtype=np.float64)[:m]
        ok = np.isfinite(feat_arr).all(axis=1)
        ok[:WARMUP_BARS] = False
        timestamps = _bar_epochs(df)[:m]
        if ok.any():
            # فقط کندل‌هایی که هنوز (کامل) ذخیره نشده‌اند؛ شکاف‌ها و ردیف‌های ناقص نسخه‌های قبل هم پر می‌شوند
            sel_ts = timestamps[ok]
            done = _stored_timestamps(
                db_path, 'training_data_multiclass', 'timestamp_us', sel_ts.min(), sel_ts.max(),
                "COALESCE(is_synthetic, 0) = 0 AND open_price IS NOT NULL AND atr IS NOT NULL "
                "AND buy_exit IS NOT NULL AND features IS NOT NULL")
            if done:
                ok &= ~np.isin(timestamps, np.fromiter(done, dtype=np.int64, count=len(done)))
        if not ok.any():
            return 0

        def opt(x):
            return None if not np.isfinite(x) else float(x)

        opens = df['Open'].to_numpy(dtype=float)
        highs = df['High'].to_numpy(dtype=float)
        lows = df['Low'].to_numpy(dtype=float)

        rows = []
        for i in np.flatnonzero(ok):
            rows.append((
                int(timestamps[i]),
                json.dumps(np.round(feat_arr[i], 6).tolist()),
                int(out['labels'][i]),
                float(closes[i]),
                opt(out['buy_ret'][i]),
                opt(out['sell_ret'][i]),
                float(confluence[i]),
                int(out['buy_win'][i]),
                int(out['sell_win'][i]),
                int(out['buy_exit'][i]),
                int(out['sell_exit'][i]),
                opt(opens[i]), opt(highs[i]), opt(lows[i]), opt(spreads[i]), opt(atrs[i]),
            ))

        if not rows:
            print("⚠️ No valid samples after labeling")
            return 0

        # گزارش فقط برای ذخیره‌ی حجیم (دانلود تاریخچه)، نه هر کندل زنده
        # (پنجره‌ی زنده خودش live_history_bars() کندل است؛ آستانه‌ی ثابت ۱۰۰۰ ردیف هر کندل رد می‌شد)
        if len(rows) >= 1000 and len(df) > live_history_bars():
            sel = np.flatnonzero(ok)
            valid = out['buy_win'][sel] >= 0
            nv = int(valid.sum())
            if nv:
                bw = out['buy_win'][sel][valid]
                sw = out['sell_win'][sel][valid]
                timeouts = np.mean(np.concatenate([out['buy_timeout'][sel][valid],
                                                   out['sell_timeout'][sel][valid]]))
                print(f"📊 Labels ({FUTURE_STEPS} bars, {barrier_text()}, "
                      f"spread-exact): buy wins={bw.mean():.1%} | sell wins={sw.mean():.1%} | "
                      f"neither={np.mean((bw == 0) & (sw == 0)):.1%} | n={nv}")
                print(f"   ⏱️ Trades closed by time exit (neither TP nor SL) ≈ {timeouts:.1%} "
                      f"— if most trades end by time, FUTURE_STEPS is short for this TP")

        conn = _db_connect(db_path)
        try:
            conn.executemany("""
                INSERT INTO training_data_multiclass
                    (timestamp_us, features, target, close_price, buy_ret, sell_ret, confluence,
                     buy_win, sell_win, buy_exit, sell_exit,
                     open_price, high_price, low_price, spread_px, atr, is_synthetic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(timestamp_us) DO UPDATE SET
                    features = excluded.features,
                    target = excluded.target,
                    close_price = excluded.close_price,
                    buy_ret = excluded.buy_ret,
                    sell_ret = excluded.sell_ret,
                    confluence = excluded.confluence,
                    buy_win = excluded.buy_win,
                    sell_win = excluded.sell_win,
                    buy_exit = excluded.buy_exit,
                    sell_exit = excluded.sell_exit,
                    open_price = excluded.open_price,
                    high_price = excluded.high_price,
                    low_price = excluded.low_price,
                    spread_px = excluded.spread_px,
                    atr = excluded.atr,
                    is_synthetic = 0
            """, rows)

            # نمونه‌های مصنوعی نسخه‌های قبلی + نگه داشتن فقط MODEL_DB_MAX_ROWS ردیف جدیدتر
            conn.execute("DELETE FROM training_data_multiclass WHERE is_synthetic = 1")
            conn.execute(f"""
                DELETE FROM training_data_multiclass
                WHERE timestamp_us < (
                    SELECT timestamp_us FROM training_data_multiclass
                    ORDER BY timestamp_us DESC LIMIT 1 OFFSET {int(MODEL_DB_MAX_ROWS) - 1}
                )
            """)
            conn.commit()
        finally:
            conn.close()

        return len(rows)

    except Exception as e:
        print(f"✗ Save error: {e}")
        traceback.print_exc()
        return 0


def save_secondary_training_data(df, features_df, db_path=None, symbol=None, tf_seconds=None):
    """
    ذخیره داده تایم‌فریم دوم (Secondary) - فقط برای Context
    بدون نیاز به لیبل (target=None). اولین WARMUP_BARS کندل پنجره ذخیره نمی‌شود.
    """
    if db_path is None:
        db_path = SECONDARY_DB_PATH
    symbol = symbol or SYMBOL
    if tf_seconds is None:
        tf_seconds = SECONDARY_TF_SECONDS

    try:
        df, features_df = _drop_forming_bar(df, features_df, tf_seconds, symbol)
        if len(df) <= WARMUP_BARS:
            return 0

        feat_arr = features_df.to_numpy(dtype=np.float64)
        finite = np.isfinite(feat_arr).all(axis=1)
        finite[:WARMUP_BARS] = False
        timestamps = _bar_epochs(df)
        closes = df['Close'].to_numpy(dtype=float)
        skipped = len(df) - WARMUP_BARS - int(finite.sum())
        if skipped > 0:
            print(f"⚠️ Secondary: skipped {skipped} NaN/Inf samples")
        if finite.any():
            # فقط کندل‌هایی که هنوز ذخیره نشده‌اند (مثل جدول اصلی)
            sel_ts = timestamps[finite]
            done = _stored_timestamps(db_path, 'training_data', 'timestamp', sel_ts.min(), sel_ts.max(),
                                      "features IS NOT NULL")
            if done:
                finite &= ~np.isin(timestamps, np.fromiter(done, dtype=np.int64, count=len(done)))
                if not finite.any():
                    return 0

        rows = [
            (int(timestamps[i]), json.dumps(np.round(feat_arr[i], 6).tolist()), float(closes[i]))
            for i in np.flatnonzero(finite)
        ]
        if not rows:
            print("⚠️ Secondary: no valid samples to save")
            return 0

        conn = _db_connect(db_path)
        try:
            conn.executemany("""
                INSERT INTO training_data (timestamp, features, target, close_price)
                VALUES (?, ?, NULL, ?)
                ON CONFLICT(timestamp) DO UPDATE SET
                    features = excluded.features,
                    close_price = excluded.close_price
            """, rows)
            conn.execute(f"""
                DELETE FROM training_data
                WHERE timestamp < (
                    SELECT timestamp FROM training_data
                    ORDER BY timestamp DESC LIMIT 1 OFFSET {int(DB_MAX_BARS) - 1}
                )
            """)
            conn.commit()
        finally:
            conn.close()

        return len(rows)

    except Exception as e:
        print(f"✗ Secondary save error in {os.path.basename(str(db_path))}: {e}")
        return 0


# ========================
# Training
# ========================
def _uniqueness_weights(labeled, span):
    """
    وزن هر نمونه = میانگین 1/(تعداد لیبل‌های هم‌زمان) روی کندل‌هایی که معامله‌اش باز بوده.
    لیبل‌های کندل‌های پشت‌سرهم روی کندل‌های مشترک حساب می‌شوند و تقریباً تکراری‌اند؛
    این وزن جلوی بیش‌برازش روی خوشه‌های سیگنال را می‌گیرد (López de Prado).
    """
    n = len(span)
    span = np.where(labeled & np.isfinite(span) & (span >= 1), span, 0).astype(np.int64)
    has = span > 0
    starts = np.arange(n) + 1
    ends = starts + span                       # انحصاری
    diff = np.zeros(n + int(span.max(initial=0)) + 2)
    np.add.at(diff, starts[has], 1.0)
    np.add.at(diff, ends[has], -1.0)
    concurrency = np.cumsum(diff)
    inv = np.where(concurrency > 0, 1.0 / np.maximum(concurrency, 1.0), 0.0)
    csum = np.concatenate([[0.0], np.cumsum(inv)])
    w = np.zeros(n)
    w[has] = (csum[ends[has]] - csum[starts[has]]) / span[has]
    return w


def load_dual_sequences_multiclass(primary_db_path, secondary_db_path,
                                   timesteps_primary, timesteps_secondary,
                                   limit=90000, exclude_neutral=False,
                                   primary_tf_seconds=None, secondary_tf_seconds=None):
    """
    ساخت پنجره‌های هم‌تراز primary/secondary از جدیدترین داده‌ها.

    هم‌ترازی بدون نگاه به آینده: برای کندل primary که در (t + P) بسته می‌شود فقط کندل‌های
    secondary استفاده می‌شوند که تا همان لحظه بسته شده‌اند (s + S <= t + P).

    خروجی: (X_primary, X_secondary, y, extras)
      y      : آرایه‌ی (n, 2) = [برد خرید، برد فروش]
      extras : buy_ret/sell_ret/confluence/timestamp/buy_exit/sell_exit/row/weight
               + close/spread/unit/cost برای شبیه‌سازی بستن با سیگنال مخالف (Reverse) مثل ربات زنده
    """
    P = primary_tf_seconds or PRIMARY_TF_SECONDS
    S = secondary_tf_seconds or SECONDARY_TF_SECONDS
    empty = (None, None, None, None)

    conn = _db_connect(primary_db_path)
    try:
        df_p = pd.read_sql_query(
            "SELECT timestamp_us AS ts, features, target, buy_ret, sell_ret, confluence, "
            "buy_win, sell_win, buy_exit, sell_exit, close_price, spread_px, atr "
            "FROM training_data_multiclass WHERE COALESCE(is_synthetic, 0) = 0 "
            "ORDER BY timestamp_us DESC LIMIT ?",
            conn, params=(int(limit),))
    except Exception as e:
        print(f"✗ Cannot read primary training data: {e}")
        return empty
    finally:
        conn.close()

    if len(df_p) < timesteps_primary + 10:
        print(f"✗ Primary data insufficient ({len(df_p)} rows)")
        return empty
    df_p = df_p.iloc[::-1].reset_index(drop=True)

    min_ts = int(df_p['ts'].iloc[0]) - (timesteps_secondary + 10) * S
    conn = _db_connect(secondary_db_path)
    try:
        df_s = pd.read_sql_query(
            "SELECT timestamp AS ts, features FROM training_data "
            "WHERE timestamp >= ? ORDER BY timestamp ASC",
            conn, params=(min_ts,))
    except Exception as e:
        print(f"✗ Cannot read secondary training data: {e}")
        return empty
    finally:
        conn.close()

    if len(df_s) < timesteps_secondary + 10:
        print(f"✗ Secondary data insufficient ({len(df_s)} rows)")
        return empty

    X_p = np.array([_parse_features(r) for r in df_p['features']], dtype=np.float32)
    X_s = np.array([_parse_features(r) for r in df_s['features']], dtype=np.float32)
    if X_p.ndim != 2 or X_s.ndim != 2 or X_p.shape[1] != X_s.shape[1]:
        print(f"✗ Feature shape mismatch: primary {X_p.shape} vs secondary {X_s.shape}")
        return empty
    if X_p.shape[1] != len(FEATURE_NAMES):
        print(f"✗ Stored features have {X_p.shape[1]} columns, expected {len(FEATURE_NAMES)}")
        return empty

    def num(col):
        return pd.to_numeric(df_p[col], errors='coerce').to_numpy(dtype=float)

    y_p = num('target')
    buy_win, sell_win = num('buy_win'), num('sell_win')
    buy_exit, sell_exit = num('buy_exit'), num('sell_exit')
    ts_p = df_p['ts'].to_numpy(dtype=np.int64)
    ts_s = df_s['ts'].to_numpy(dtype=np.int64)
    buy_ret = num('buy_ret')
    sell_ret = num('sell_ret')
    confluence = num('confluence')

    labeled = ((y_p >= 0) & np.isin(buy_win, (0, 1)) & np.isin(sell_win, (0, 1))
               & np.isfinite(buy_ret) & np.isfinite(sell_ret)
               & (buy_exit >= 1) & (sell_exit >= 1))

    # وزن یکتایی روی کل ردیف‌ها (قبل از انتخاب) حساب می‌شود
    weights = _uniqueness_weights(labeled, np.fmax(buy_exit, sell_exit))

    # آخرین کندل secondary که قبل از بسته شدن کندل primary بسته شده
    idx_s = np.searchsorted(ts_s, ts_p + P - S, side='right') - 1

    idx = np.arange(len(y_p))
    keep = (idx >= timesteps_primary - 1) & (idx_s >= timesteps_secondary - 1) & labeled
    keep &= np.isfinite(confluence) & (weights > 0)
    if exclude_neutral:
        keep &= (buy_win == 1) | (sell_win == 1)

    sel = np.flatnonzero(keep)
    if len(sel) == 0:
        print("✗ No aligned sequences")
        return empty

    win_p = sliding_window_view(X_p, timesteps_primary, axis=0)      # (n-T+1, F, T)
    win_s = sliding_window_view(X_s, timesteps_secondary, axis=0)
    Xp_arr = np.ascontiguousarray(win_p[sel - timesteps_primary + 1].transpose(0, 2, 1))
    Xs_arr = np.ascontiguousarray(win_s[idx_s[sel] - timesteps_secondary + 1].transpose(0, 2, 1))

    y = np.stack([buy_win[sel], sell_win[sel]], axis=1).astype(np.float32)

    # برای بستن معامله با سیگنال مخالف در شبیه‌سازی: Close (Bid) و اسپرد همان کندل، و واحد بازده‌ی
    # کندل ورود (مثل لیبل: ATR در حالت atr، 1 دلار قیمت در حالت fixed) و هزینه‌ی اضافه به واحد قیمت
    unit = np.ones(len(y_p)) if LABEL_MODE == "fixed" else num('atr')
    extra_cost = float(EXTRA_COST_POINTS) * _point_size.get(SYMBOL, 0.0)

    print(f"✅ Created {len(sel)} aligned labeled sequences (from {len(y_p)} primary rows)")
    extras = {
        'buy_ret': buy_ret[sel], 'sell_ret': sell_ret[sel],
        'confluence': confluence[sel], 'timestamp': ts_p[sel],
        'buy_exit': buy_exit[sel].astype(np.int64), 'sell_exit': sell_exit[sel].astype(np.int64),
        'row': idx[sel], 'weight': weights[sel],
        'close': num('close_price')[sel], 'spread': np.nan_to_num(num('spread_px')[sel], nan=0.0),
        'unit': unit[sel], 'cost': np.full(len(sel), extra_cost),
    }
    return Xp_arr, Xs_arr, y, extras


def build_multiclass_model(timesteps_p, timesteps_s, n_features, n_outputs=2):
    """
    مدل Dual با دو خروجی مستقل sigmoid:
      output[0] = احتمال اینکه معامله‌ی خرید به TP برسد قبل از SL
      output[1] = احتمال اینکه معامله‌ی فروش به TP برسد قبل از SL
    (این دو ناسازگار نیستند؛ هر دو ممکن است برنده یا بازنده باشند)
    """
    from tensorflow.keras.layers import (
        GRU, Dense, Dropout, Input, BatchNormalization, LayerNormalization, Conv1D, Concatenate
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l1_l2

    input_primary = Input(shape=(timesteps_p, n_features), name='input_primary')
    input_secondary = Input(shape=(timesteps_s, n_features), name='input_secondary')

    # Primary branch
    p = Conv1D(32, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l1_l2(1e-5, 1e-4))(input_primary)
    p = BatchNormalization()(p)
    p = Dropout(0.3)(p)

    p = GRU(64, return_sequences=True, dropout=0.3,
            kernel_regularizer=l1_l2(1e-5, 1e-4))(p)
    p = LayerNormalization()(p)
    p = Dropout(0.3)(p)

    p = GRU(32, return_sequences=False, dropout=0.3,
            kernel_regularizer=l1_l2(1e-5, 1e-4))(p)
    p = LayerNormalization()(p)
    primary_out = Dropout(0.3)(p)

    # Secondary branch
    s = Conv1D(16, kernel_size=3, padding='same', activation='relu',
               kernel_regularizer=l1_l2(1e-5, 1e-4))(input_secondary)
    s = BatchNormalization()(s)
    s = Dropout(0.3)(s)

    s = GRU(32, return_sequences=False, dropout=0.3,
            kernel_regularizer=l1_l2(1e-5, 1e-4))(s)
    s = LayerNormalization()(s)
    secondary_out = Dropout(0.3)(s)

    merged = Concatenate()([primary_out, secondary_out])

    d1 = Dense(64, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(merged)
    d1 = BatchNormalization()(d1)
    d1 = Dropout(0.4)(d1)

    d2 = Dense(32, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(d1)
    d2 = BatchNormalization()(d2)
    d2 = Dropout(0.3)(d2)

    output = Dense(n_outputs, activation='sigmoid', name='output')(d2)

    model = Model(inputs=[input_primary, input_secondary], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
        loss='binary_crossentropy'
    )
    return model


DISABLED_THRESHOLD = 1.01   # آستانه‌ی غیرقابل‌دسترس = آن جهت معامله نمی‌شود


def _perf_from_returns(rets, n_buys, n_sells):
    n = len(rets)
    if n == 0:
        return {'trades': 0, 'buys': 0, 'sells': 0, 'win_rate': 0.0,
                'expectancy': 0.0, 'profit_factor': 0.0, 'total_atr': 0.0}
    gains = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    sd = float(rets.std(ddof=1)) if n > 1 else 0.0
    return {
        'trades': int(n),
        'buys': int(n_buys),
        'sells': int(n_sells),
        'win_rate': float(np.mean(rets > 0)),
        'expectancy': float(rets.mean()),
        'profit_factor': float(gains / losses) if losses > 0 else float('inf'),
        'total_atr': float(rets.sum()),
        # آماره‌ی t میانگین (≈ چند خطای معیار از صفر): زیر ~۲ یعنی سود می‌تواند شانسی باشد
        't': float(rets.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0,
    }


def simulate_trades(signals, buy_ret, sell_ret, buy_exit, sell_exit, rows,
                    close=None, spread=None, unit=None, cost=None):
    """
    شبیه‌سازی مثل ربات زنده: یک پوزیشن در هر لحظه.
    - سیگنال هم‌جهت وسط یک معامله‌ی باز نادیده گرفته می‌شود؛ ورود بعدی از کندلی ممکن است که
      معامله‌ی قبلی در آن بسته شده (row + exit).
    - سیگنال مخالف وسط یک معامله‌ی باز، مثل حلقه‌ی زنده (close_position با دلیل Reverse)، معامله‌ی باز را
      به Close همان کندل می‌بندد (خرید با Bid، فروش با Ask = Close + اسپرد) و معامله‌ی مخالف را باز می‌کند.
      این کار close/spread/unit (و cost) هر نمونه را لازم دارد؛ بدون آن‌ها سیگنال مخالف هم نادیده گرفته می‌شود.
    خروجی: (بازده معاملات به واحد لیبل، تعداد خرید، تعداد فروش)
    """
    signals = np.asarray(signals)
    can_reverse = close is not None and spread is not None and unit is not None
    rets = []
    n_buys = n_sells = 0
    free_at = -np.inf
    open_k, open_dir = -1, 0          # نمونه‌ی ورود و جهت معامله‌ی باز
    for k in np.flatnonzero(signals != 0):
        r = rows[k]
        side = 1 if signals[k] > 0 else -1
        if r < free_at:
            # معامله‌ی قبلی هنوز باز است (نه به TP/SL خورده، نه به خروج زمانی رسیده)
            if not can_reverse or side == open_dir:
                continue
            c_cost = float(cost[open_k]) if cost is not None else 0.0
            if open_dir == 1:
                exit_ret = (close[k] - (close[open_k] + spread[open_k]) - c_cost) / unit[open_k]
            else:
                exit_ret = (close[open_k] - (close[k] + spread[k]) - c_cost) / unit[open_k]
            if not np.isfinite(exit_ret):
                continue
            rets[-1] = float(exit_ret)
        if side == 1:
            rets.append(buy_ret[k])
            free_at = r + buy_exit[k]
            n_buys += 1
        else:
            rets.append(sell_ret[k])
            free_at = r + sell_exit[k]
            n_sells += 1
        open_k, open_dir = k, side
    return np.asarray(rets, dtype=float), n_buys, n_sells


def _subset(extras, part):
    return {k: v[part] for k, v in extras.items()}


def _simulate_extras(sig, extras):
    return simulate_trades(sig, extras['buy_ret'], extras['sell_ret'],
                           extras['buy_exit'], extras['sell_exit'], extras['row'],
                           extras.get('close'), extras.get('spread'), extras.get('unit'), extras.get('cost'))


def _simulate_signals(sig, extras):
    rets, n_b, n_s = _simulate_extras(sig, extras)
    return _perf_from_returns(rets, n_b, n_s)


def evaluate_trading_performance(probs, extras, buy_threshold, sell_threshold):
    """
    شبیه‌سازی حالت «فقط مدل» روی داده‌ی Out-of-Sample با همان قانون تصمیم ربات زنده
    (decide_model_signals) و یک پوزیشن در هر لحظه. بازده‌ها بر حسب ATR و با اسپرد واقعی داخل بریرها.
    """
    sig = decide_model_signals(probs[:, 0], probs[:, 1], buy_threshold, sell_threshold)
    return _simulate_signals(sig, extras)


def evaluate_indicator_performance(extras, min_confluence=None):
    """همان شبیه‌سازی برای حالت «فقط اندیکاتور» (فقط برای اطلاع؛ روی پذیرش مدل اثری ندارد)"""
    thr = MIN_CONFLUENCE_SCORE if min_confluence is None else min_confluence
    return _simulate_signals(decide_indicator_signals(extras['confluence'], thr), extras)


def tune_thresholds(probs, extras, floor_buy, floor_sell, min_trades):
    """
    انتخاب آستانه‌ی احتمال برد برای هر جهت روی مجموعه‌ی Validation (با شبیه‌سازی یک‌پوزیشنی).
    معیار: کران پایین اطمینان بازده (mean - std/sqrt(n)) تا آستانه‌ای که فقط از شانس خوب
    به نظر می‌رسد انتخاب نشود. آستانه هیچ‌وقت کمتر از تنظیم کاربر (min_confidence) نمی‌شود.
    (کف نظری SL/(TP+SL) عمداً استفاده نمی‌شود: وقتی بخش بزرگی از معاملات با خروج زمانی و
    سود/زیان کوچک بسته می‌شوند، آن فرمول غلط است؛ سودآوری مستقیماً روی بازده واقعی سنجیده می‌شود.)
    جهتی که روی Validation بازده مثبت ندارد غیرفعال می‌شود.
    """
    p_buy, p_sell = probs[:, 0], probs[:, 1]
    min_n = max(5, int(min_trades) // 3)
    chosen = {}

    for side, floor, p_arr in ((1, floor_buy, p_buy), (-1, floor_sell, p_sell)):
        lo = max(float(floor), 0.02)
        p_max = float(np.max(p_arr)) if len(p_arr) else 0.0
        best = None
        for thr in np.round(np.arange(lo, 0.96, 0.01), 2):
            if thr > p_max:
                break
            b_thr, s_thr = (thr, DISABLED_THRESHOLD) if side == 1 else (DISABLED_THRESHOLD, thr)
            sig = decide_model_signals(p_buy, p_sell, b_thr, s_thr)
            r, _, _ = _simulate_extras(sig, extras)
            if len(r) < min_n:
                continue
            lcb = r.mean() - r.std(ddof=1) / np.sqrt(len(r))
            if best is None or lcb > best[1]:
                best = (float(thr), float(lcb), float(r.mean()), len(r))
        if best is not None and best[2] > 0:
            chosen[side] = best + (lo, p_max)
        else:
            chosen[side] = (DISABLED_THRESHOLD, None, None, 0, lo, p_max)

    for side, name in ((1, "BUY "), (-1, "SELL")):
        thr, lcb, mean, n, lo, p_max = chosen[side]
        if thr >= DISABLED_THRESHOLD:
            hint = ""
            if p_max < lo:
                hint = (f" — highest P(win) on validation is {p_max:.2f}, below your min_confidence "
                        f"{lo:.2f}; lower min_confidence_{'buy' if side == 1 else 'sell'}")
            print(f"   🎚️ {name}: disabled (no positive edge on validation){hint}")
        else:
            print(f"   🎚️ {name}: threshold={thr:.2f} | val trades={n} | exp={mean:+.3f} | LCB={lcb:+.3f}")
    return chosen[1][0], chosen[-1][0]


def _format_perf(p):
    u = ret_unit()
    txt = (f"trades={p['trades']} (B{p['buys']}/S{p['sells']}) | win={p['win_rate']:.1%} | "
           f"exp={p['expectancy']:+.3f} {u} | PF={p['profit_factor']:.2f} | total={p['total_atr']:+.1f} {u}")
    if 't' in p:
        txt += f" | t={p['t']:+.2f}"
    if LABEL_MODE == "fixed":
        money = money_per_trade(p['expectancy'])
        if money is not None:
            txt += f" | ≈ ${money:+.2f}/trade at lot {LOT}"
    return txt


def _atomic_save_model(model, path):
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext or '.h5'}"
    model.save(tmp)
    os.replace(tmp, path)


def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def model_meta_path(model_path=None):
    return (model_path or MODEL_PATH) + ".meta.json"


# متادیتای مدل فعال (آستانه‌های تنظیم‌شده روی Validation)
ACTIVE_MODEL_META = {}


def set_active_model_meta(meta):
    global ACTIVE_MODEL_META
    ACTIVE_MODEL_META = dict(meta or {})


def effective_thresholds():
    """آستانه‌ی نهایی = بیشترِ (تنظیم کاربر، آستانه‌ی بهینه‌شده برای مدل فعال)"""
    return (max(MIN_CONFIDENCE_BUY, float(ACTIVE_MODEL_META.get('buy_threshold', 0.0))),
            max(MIN_CONFIDENCE_SELL, float(ACTIVE_MODEL_META.get('sell_threshold', 0.0))))


def _head_metrics(y_true, p):
    """AUC و Brier یک خروجی (اگر فقط یک کلاس باشد AUC تعریف نمی‌شود)"""
    from sklearn.metrics import roc_auc_score, brier_score_loss
    y_true = np.asarray(y_true)
    auc = float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) == 2 else float('nan')
    return {'auc': auc, 'brier': float(brier_score_loss(y_true, p)), 'base_rate': float(np.mean(y_true))}


# آخرین گزارش آموزش (برای نمایش در GUI یا لاگ)
last_training_report = {}

# فقط یک آموزش در هر لحظه (Setup در main_app و آموزش پس‌زمینه‌ی ربات هر دو ممکن است شروع کنند؛
# دو آموزش هم‌زمان می‌توانند مدل یکی را با اسکیلر دیگری روی دیسک بنویسند)
_train_lock = threading.Lock()
# لغو آموزش در حال اجرا (مثلاً وقتی کاربر منبع سیگنال را عوض می‌کند و آن آموزش دیگر لازم نیست)
training_cancel = threading.Event()
_training_kind = {'kind': None}


def current_training_kind():
    """نوع آموزش در حال اجرا: 'model' یا 'rl' یا None"""
    return _training_kind['kind']


def request_training_cancel(kind=None, reason="signal source changed"):
    """
    آموزش در حال اجرا را متوقف کن (اگر kind داده شود، فقط وقتی همان نوع در حال اجراست).
    آموزش مدل بعد از batch جاری و عامل RL در گام بعدی متوقف می‌شود؛ چیزی ذخیره نمی‌شود.
    """
    running = _training_kind['kind']
    if running is None or (kind is not None and running != kind):
        return False
    training_cancel.set()
    print(f"⏹️ Stopping the running {'RL agent' if running == 'rl' else 'model'} training "
          f"({reason})...")
    return True


def _acquire_training_lock(kind, wait=False, timeout=1800, on_wait=None, stop_event=None):
    """
    قفل «یک آموزش در هر لحظه». wait=True: تا timeout ثانیه صبر می‌کند تا آموزش قبلی تمام/لغو شود.
    stop_event: فقط برای آموزشی که مال حلقه‌ی ربات است (bot_stop_event)؛ آموزشی که برنامه شروع می‌کند
    نباید به روشن/خاموش بودن ربات وابسته باشد (bot_stop_event بعد از Stop تا Start بعدی set می‌ماند).
    """
    acquired = _train_lock.acquire(blocking=False)
    if not acquired and wait:
        other = _training_kind['kind']
        print(f"⏳ Waiting for the running {other or ''} training to finish before starting {kind} training...")
        deadline = time.time() + timeout
        while (not acquired and time.time() < deadline
               and not (stop_event is not None and stop_event.is_set())):
            if on_wait:
                try:
                    on_wait()
                except Exception:
                    pass
            acquired = _train_lock.acquire(timeout=2.0)
    if acquired:
        training_cancel.clear()
        _training_kind['kind'] = kind
    return acquired


def _release_training_lock():
    _training_kind['kind'] = None
    training_cancel.clear()
    _train_lock.release()


def train_and_validate(db_path=None, model_path=None, scaler_path=None,
                       last_train_file=None, secondary_db_path=None,
                       secondary_scaler_path=None, exclude_neutral=False,
                       primary_tf=None, secondary_tf=None, wait=False):
    """
    آموزش حالت Model: الگوریتم ژنتیک قانون‌های الگوی کندلی/کلاسیک را روی Train پیدا می‌کند، روی Validation
    انتخاب و ترکیب می‌کند و روی Test می‌سنجد (_train_ga_locked).
    بدون فیلتر ذخیره: هر آموزشی که کامل شود (لغو نشده و بدون خطا) ذخیره و استفاده می‌شود؛
    نتیجه‌ی Validation/Test فقط گزارش می‌شود (quality_warnings).
    (امضای تابع برای سازگاری با main_app عوض نشده؛ پارامترهای اسکیلر/تایم‌فریم دوم برای GA استفاده نمی‌شوند)

    wait=True: اگر آموزش دیگری در حال اجراست، صبر کن (برای Setup برنامه)؛ وگرنه این درخواست رد می‌شود.
    خروجی: dict با کلیدهای accepted, model, scaler_p, scaler_s, metrics (+ skipped / cancelled)
    """
    primary_db = db_path or PRIMARY_DB_PATH
    secondary_db = secondary_db_path or SECONDARY_DB_PATH
    model_save_path = model_path or MODEL_PATH
    primary_scaler_path = scaler_path or PRIMARY_SCALER_PATH
    secondary_scaler_save_path = secondary_scaler_path or SECONDARY_SCALER_PATH
    last_train_file = last_train_file or LAST_TRAIN_FILE
    P = TF_SECONDS.get(primary_tf or PRIMARY_TF_STR, 300)
    # تایم‌فریم روند مدل الگو: M30 (GA_TREND_TF)، نه لزوماً تایم‌فریم دوم تنظیمات
    S = ga_htf_seconds(P, fallback=TF_SECONDS.get(secondary_tf or SECONDARY_TF_STR, 900))

    result = {'accepted': False, 'model': None, 'scaler_p': None, 'scaler_s': None,
              'meta': None, 'metrics': {}, 'skipped': False, 'cancelled': False}

    if not _acquire_training_lock('model', wait=wait):
        print("⏳ Another training is already running — this request is skipped")
        result['skipped'] = True
        return result
    try:
        # حالت Model = استراتژی الگوریتم ژنتیک روی الگوهای کندلی و کلاسیک (شبکه‌ی عصبی دیگر آموزش نمی‌بیند)
        return _train_ga_locked(result, primary_db, model_save_path, last_train_file, P, S)
    finally:
        _release_training_lock()


def _train_and_validate_locked(result, primary_db, secondary_db, model_save_path,
                               primary_scaler_path, secondary_scaler_save_path,
                               last_train_file, exclude_neutral, P, S):
    """(قدیمی — دیگر استفاده نمی‌شود) آموزش شبکه‌ی عصبی GRU؛ حالت Model حالا الگوریتم ژنتیک است"""
    global last_training_report
    import tensorflow as tf
    from sklearn.preprocessing import RobustScaler
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    try:
        print("\n" + "=" * 60)
        print("🎯 TRAINING DUAL MODEL (price-action features, buy/sell win heads)")
        print("=" * 60)

        training_state.start(total_epochs=EPOCHS)
        try:
            tf.keras.utils.set_random_seed(42)
        except Exception:
            pass

        X_primary, X_secondary, y, extras = load_dual_sequences_multiclass(
            primary_db, secondary_db, TIMESTEPS, TIMESTEPS_SECONDARY,
            exclude_neutral=exclude_neutral, primary_tf_seconds=P, secondary_tf_seconds=S
        )
        if X_primary is None:
            save_train_time(False, "Not enough aligned data", last_train_file, model_save_path)
            return result

        n_features = X_primary.shape[2]
        print(f"📐 Primary: {X_primary.shape} | Secondary: {X_secondary.shape}")
        bw, sw = y[:, 0], y[:, 1]
        print(f"📈 Win rates: buy={bw.mean():.1%} | sell={sw.mean():.1%} | "
              f"both={np.mean((bw == 1) & (sw == 1)):.1%} | neither={np.mean((bw == 0) & (sw == 0)):.1%} "
              )

        # تقسیم زمانی با فاصله (gap) تا پنجره‌ها و افق لیبل بین مجموعه‌ها هم‌پوشانی نداشته باشند
        n = len(y)
        gap = TIMESTEPS + FUTURE_STEPS
        train_end = int(n * 0.70)
        val_start = train_end + gap
        val_end = val_start + int(n * 0.15)
        test_start = val_end + gap

        if test_start >= n - 50:
            print("✗ Not enough data for split")
            save_train_time(False, "Not enough data for train/val/test split",
                            last_train_file, model_save_path)
            return result

        tr, va, te = slice(0, train_end), slice(val_start, val_end), slice(test_start, n)
        y_train, y_val, y_test = y[tr], y[va], y[te]
        if all(len(np.unique(y_train[:, k])) < 2 for k in range(2)):
            print("✗ Training labels have a single class for both heads")
            save_train_time(False, "Only one class in data", last_train_file, model_save_path)
            return result
        print(f"📊 Split: Train={len(y_train)} | Val={len(y_val)} | Test={len(y_test)}")

        def scale(scaler, X, fit=False):
            flat = X.reshape(-1, n_features)
            out = scaler.fit_transform(flat) if fit else scaler.transform(flat)
            return out.reshape(X.shape).astype(np.float32)

        scaler_p = RobustScaler()
        scaler_s = RobustScaler()
        Xp_train = scale(scaler_p, X_primary[tr], fit=True)
        Xs_train = scale(scaler_s, X_secondary[tr], fit=True)
        Xp_val, Xs_val = scale(scaler_p, X_primary[va]), scale(scaler_s, X_secondary[va])
        Xp_test, Xs_test = scale(scaler_p, X_primary[te]), scale(scaler_s, X_secondary[te])

        # وزن یکتایی (نرمال‌شده به میانگین 1 در هر مجموعه)
        w_train = extras['weight'][tr] / max(float(np.mean(extras['weight'][tr])), 1e-12)
        w_val = extras['weight'][va] / max(float(np.mean(extras['weight'][va])), 1e-12)
        print(f"⚖️ Uniqueness weights (train): mean concurrency ≈ {1.0 / max(np.mean(extras['weight'][tr]), 1e-12):.1f} "
              f"overlapping labels per bar")

        model = build_multiclass_model(TIMESTEPS, TIMESTEPS_SECONDARY, n_features, n_outputs=2)
        print(f"📊 Parameters: {model.count_params():,}")

        # ── AUC هر epoch برای هر دو خروجی (خرید/فروش)، روی Train و Validation ──
        # Train AUC روی یک زیرمجموعه‌ی ثابت ۱۰ هزارتایی حساب می‌شود تا هر epoch زیاد طول نکشد.
        # پیش‌بینی در حالت inference است (بدون dropout) تا با Validation قابل مقایسه باشد.
        auc_rng = np.random.default_rng(0)
        tr_idx = np.sort(auc_rng.choice(len(y_train), size=min(len(y_train), 10000), replace=False))
        auc_sets = {'train': (Xp_train[tr_idx], Xs_train[tr_idx], y_train[tr_idx]),
                    'val': (Xp_val, Xs_val, y_val)}
        auc_history = []

        def epoch_aucs(name):
            Xp_, Xs_, y_ = auc_sets[name]
            pr = np.asarray(model.predict([Xp_, Xs_], verbose=0, batch_size=2048))
            return [_head_metrics(y_[:, k], pr[:, k])['auc'] for k in range(2)]

        print("📏 AUC per epoch (train/val): 0.50 = coin flip | ~0.55 weak | ≥0.60 good for markets | "
              "train rising while val falls = memorizing noise (overfitting)")

        class EpochCallback(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                if training_cancel.is_set():
                    model.stop_training = True

            def on_epoch_end(self, epoch, logs=None):
                if training_cancel.is_set():
                    model.stop_training = True
                    return
                training_state.update_progress(epoch + 1)
                logs = logs or {}
                try:
                    tr_b, tr_s = epoch_aucs('train')
                    va_b, va_s = epoch_aucs('val')
                    auc_history.append({'epoch': epoch + 1, 'train_buy': tr_b, 'train_sell': tr_s,
                                        'val_buy': va_b, 'val_sell': va_s})
                    auc_txt = f" | AUC buy {tr_b:.3f}/{va_b:.3f} | sell {tr_s:.3f}/{va_s:.3f}"
                except Exception as e:
                    auc_txt = f" | AUC n/a ({e})"
                print(f"   Epoch {epoch+1}: loss={logs.get('loss', 0):.4f} | "
                      f"val_loss={logs.get('val_loss', 0):.4f}{auc_txt}")

        # EpochCallback اول است تا AUC هر epoch قبل از اینکه EarlyStopping وزن‌ها را برگرداند حساب شود
        callbacks = [
            EpochCallback(),
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ]

        print("\n🏋️ Training...")
        model.fit(
            [Xp_train, Xs_train], y_train,
            sample_weight=w_train.astype(np.float32),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=([Xp_val, Xs_val], y_val, w_val.astype(np.float32)),
            shuffle=True,   # ترتیب زمانی داخل هر پنجره حفظ می‌شود؛ شافل بین پنجره‌ها مجاز است
            callbacks=callbacks,
            verbose=0
        )

        if training_cancel.is_set():
            # لغوشده: چیزی ذخیره یا ثبت نمی‌شود تا زمان‌بندی آموزش بعدی عقب نیفتد
            print("⏹️ Model training cancelled — nothing saved")
            result['cancelled'] = True
            return result

        if auc_history:
            best = max(auc_history, key=lambda h: np.nanmean([h['val_buy'], h['val_sell']]))
            print(f"📏 Best validation AUC: epoch {best['epoch']} (buy {best['val_buy']:.3f}, "
                  f"sell {best['val_sell']:.3f}) — final weights are from the best val_loss epoch")

        print("\n📊 Evaluating...")
        probs_train = np.asarray(model.predict([Xp_train, Xs_train], verbose=0, batch_size=1024))
        probs_val = np.asarray(model.predict([Xp_val, Xs_val], verbose=0, batch_size=1024))
        probs_test = np.asarray(model.predict([Xp_test, Xs_test], verbose=0, batch_size=1024))

        head_stats = {}
        print("\n" + "=" * 60)
        for k, name in ((0, 'buy'), (1, 'sell')):
            m_tr = _head_metrics(y_train[:, k], probs_train[:, k])
            m_te = _head_metrics(y_test[:, k], probs_test[:, k])
            head_stats[name] = {'train': m_tr, 'test': m_te}
            print(f"   {name.upper():4s} head: AUC train={m_tr['auc']:.3f} test={m_te['auc']:.3f} | "
                  f"Brier test={m_te['brier']:.4f} (base rate {m_te['base_rate']:.1%})")

        # ── تنظیم آستانه روی Validation، سنجش نهایی روی Test (قانون دقیقاً مثل معامله‌ی زنده) ──
        ex_val, ex_test = _subset(extras, va), _subset(extras, te)
        print("🎚️ Tuning win-probability thresholds on validation set:")
        buy_thr, sell_thr = tune_thresholds(probs_val, ex_val, MIN_CONFIDENCE_BUY, MIN_CONFIDENCE_SELL,
                                            MIN_TEST_TRADES)

        perf_val = evaluate_trading_performance(probs_val, ex_val, buy_thr, sell_thr)
        perf_test = evaluate_trading_performance(probs_test, ex_test, buy_thr, sell_thr)
        perf_test_indicators = evaluate_indicator_performance(ex_test)
        baseline_buy = float(np.mean(ex_test['buy_ret']))
        baseline_sell = float(np.mean(ex_test['sell_ret']))

        print(f"💰 Out-of-sample simulation — MODEL mode ({barrier_text()}, one position at a time, "
              f"{ret_unit()} units, spread-exact):")
        print(f"   Validation: {_format_perf(perf_val)}")
        print(f"   Test      : {_format_perf(perf_test)}")
        print(f"   Indicators-only mode on the same test period (info, min score "
              f"{MIN_CONFLUENCE_SCORE}): {_format_perf(perf_test_indicators)}")
        print(f"   Baseline  : always-buy exp={baseline_buy:+.3f} {ret_unit()} | "
              f"always-sell exp={baseline_sell:+.3f} {ret_unit()}")

        # بدون فیلتر ذخیره: معیارهای کیفیت فقط گزارش می‌شوند و جلوی ذخیره را نمی‌گیرند؛
        # هر آموزشی که کامل شود ذخیره و در ربات استفاده می‌شود
        quality_notes = []
        if buy_thr >= DISABLED_THRESHOLD and sell_thr >= DISABLED_THRESHOLD:
            quality_notes.append("no direction has a positive edge on validation — both sides are disabled, "
                                 "so this model will not open trades")
        if perf_test['trades'] < MIN_TEST_TRADES:
            quality_notes.append(f"few test trades ({perf_test['trades']} < {MIN_TEST_TRADES})")
        if perf_test['expectancy'] < min_test_expectancy():
            quality_notes.append(f"test expectancy {perf_test['expectancy']:+.3f} < "
                                 f"{min_test_expectancy():.3g} {ret_unit()}")
        if perf_test['profit_factor'] < MIN_TEST_PROFIT_FACTOR:
            quality_notes.append(f"test PF {perf_test['profit_factor']:.2f} < {MIN_TEST_PROFIT_FACTOR}")

        metrics = {
            'heads': head_stats,
            'val': perf_val, 'test': perf_test, 'test_indicators': perf_test_indicators,
            'buy_threshold': buy_thr, 'sell_threshold': sell_thr,
            'baseline_buy': baseline_buy, 'baseline_sell': baseline_sell,
            'label_mode': LABEL_MODE, 'ret_unit': ret_unit(),
            'auc_history': auc_history,
            'quality_warnings': quality_notes,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        result['metrics'] = metrics
        last_training_report = metrics

        summary = (f"Test {_format_perf(perf_test)} | Val exp={perf_val['expectancy']:+.3f} | "
                   f"AUC buy={head_stats['buy']['test']['auc']:.3f} sell={head_stats['sell']['test']['auc']:.3f}")
        if quality_notes:
            summary += " | Warnings: " + "; ".join(quality_notes)

        meta = {
            'buy_threshold': buy_thr,
            'sell_threshold': sell_thr,
            'label_signature': label_signature(),
            'decision_rule': DECISION_RULE,
            'label_mode': LABEL_MODE,
            'barriers': barrier_text(),
            'output': 'buy_win,sell_win',
            'features': FEATURE_NAMES,
            'timesteps': TIMESTEPS,
            'timesteps_secondary': TIMESTEPS_SECONDARY,
            'n_features': int(n_features),
            'trained_at': metrics['time'],
            'test': perf_test,
            'val': perf_val,
            'heads': head_stats,
            'quality_warnings': quality_notes,
        }
        _atomic_save_model(model, model_save_path)
        _atomic_dump(scaler_p, primary_scaler_path)
        _atomic_dump(scaler_s, secondary_scaler_save_path)
        _atomic_write_json(meta, model_meta_path(model_save_path))
        if quality_notes:
            print(f"\n⚠️ Quality checks not met (saved anyway — no save filter): {'; '.join(quality_notes)}")
        print(f"\n✅ Model saved and deployed → {model_save_path}")
        save_train_time(True, summary, last_train_file, model_save_path)
        result.update(accepted=True, model=model, scaler_p=scaler_p, scaler_s=scaler_s, meta=meta)

        print("=" * 60 + "\n")
        return result

    except Exception as e:
        print(f"✗ Training error: {e}")
        traceback.print_exc()
        save_train_time(False, f"Training error: {e}", last_train_file, model_save_path)
        return result
    finally:
        training_state.finish()


def train_model_multiclass(db_path=None, model_path=None, scaler_path=None,
                           last_train_file=None, secondary_db_path=None,
                           secondary_scaler_path=None, exclude_neutral=False,
                           primary_tf=None, secondary_tf=None, wait=False):
    """
    آموزش مدل.
    مدل ذخیره‌شده را برمی‌گرداند؛ None فقط وقتی آموزش انجام نشد (داده‌ی کم، خطا، لغو، یا آموزش دیگری در حال اجرا).
    """
    result = train_and_validate(
        db_path=db_path, model_path=model_path, scaler_path=scaler_path,
        last_train_file=last_train_file, secondary_db_path=secondary_db_path,
        secondary_scaler_path=secondary_scaler_path, exclude_neutral=exclude_neutral,
        primary_tf=primary_tf, secondary_tf=secondary_tf, wait=wait
    )
    return result['model'] if result['accepted'] else None


def prepare_symbol_profiles_multiclass(symbol: str, timeframes=("M5", "M15"),
                                       progress_callback=None,
                                       balance_method='none',
                                       exclude_neutral_in_training=False):
    """
    آماده‌سازی پروفایل Dual Timeframe برای یک نماد (جمع‌آوری داده + آموزش)
    """
    def progress(p, msg):
        print(f"[{int(p*100):3d}%] {msg}")
        if progress_callback:
            try:
                progress_callback(p, msg)
            except Exception:
                pass

    config.load_config()
    primary_tf = timeframes[0]
    secondary_tf = timeframes[1] if len(timeframes) > 1 else get_secondary_timeframe_default(primary_tf)

    dual_paths = build_dual_profile_paths(
        symbol, primary_tf, secondary_tf,
        config.model.model_path, config.model.scaler_path, config.database.db_path,
        config.database.last_train_file, config.database.log_csv
    )

    if not ensure_mt5_connected(symbol):
        progress(0.0, "❌ MT5 connection failed")
        return False

    progress(0.05, f"[{symbol}] Initializing databases...")
    init_database(dual_paths['primary_db_path'])
    init_database(dual_paths['secondary_db_path'])

    progress(0.10, f"[{symbol}] Downloading {primary_tf} + {secondary_tf} history...")
    collect_training_data(
        symbol=symbol, primary_tf=primary_tf, secondary_tf=secondary_tf,
        primary_db=dual_paths['primary_db_path'], secondary_db=dual_paths['secondary_db_path']
    )

    n_labeled = count_labeled_samples(dual_paths['primary_db_path'])
    progress(0.60, f"[{symbol}] Labeled samples: {n_labeled}")
    if n_labeled < MIN_SAMPLES_FOR_TRAINING:
        progress(1.0, f"⚠️ Need more data ({n_labeled}/{MIN_SAMPLES_FOR_TRAINING})")
        return False

    progress(0.70, f"[{symbol}] Training GA pattern model ({primary_tf}+{secondary_tf})...")
    model = train_model_multiclass(
        db_path=dual_paths['primary_db_path'],
        secondary_db_path=dual_paths['secondary_db_path'],
        model_path=dual_paths['dual_model_path'],
        scaler_path=dual_paths['primary_scaler_path'],
        secondary_scaler_path=dual_paths['secondary_scaler_path'],
        last_train_file=dual_paths['dual_last_train_file'],
        exclude_neutral=exclude_neutral_in_training,
        primary_tf=primary_tf, secondary_tf=secondary_tf
    )

    if model is None:
        progress(1.0, "⚠️ Model training failed — nothing saved (see log)")
        return False
    progress(1.0, f"✅ GA pattern model trained for {symbol} ({len(model)} rule(s))")
    return True


# ═══════════════════════════════════════════════════════════════
# PATTERN MODEL — استراتژی حالت «Model» (همه‌ی الگوهای کندلی/کلاسیک + الگوهای ترکیبی کشف‌شده با الگوریتم ژنتیک)
# ═══════════════════════════════════════════════════════════════
# هر الگو حد ضرر ثابت و حد سود خودش را دارد که از گذشته‌ی همان الگو می‌آید: بعد از هر وقوع الگو در تاریخچه، معامله بدون
# محدودیت زمان تا خوردن حد ضرر دنبال می‌شود و بیشترین قیمت به نفع معامله «سود» آن وقوع است؛ حد سود = میانگین سودها.
# فقط الگوهایی معامله می‌شوند که میانگین سود واقعی هر معامله‌شان (با همین حدها و بعد از همه‌ی هزینه‌ها) به حداقل برسد.
# اگر چند الگو روی یک کندل برقرار باشند، الگوی با بیشترین میانگین سود معامله می‌شود.
# الگوریتم ژنتیک الگوهای ترکیبی (ماشه‌ی کندلی/کلاسیک یا «شکل کندلی» کشف‌شده + فیلترهای زمینه مثل روند، تایم‌فریم
# بالاتر، نوسان، ساعت سرور + شرط‌های عددی با آستانه‌ی یادگرفته‌شده) پیدا می‌کند که آن‌ها هم با همین قانون معامله می‌شوند.
# برازندگی هر ترکیب = سود شبیه‌سازی‌شده با حد سود/ضرر آماری خودش و اسپرد دقیق، روی ۴ تکه‌ی زمانی Train، هم با روند
# واقعی و هم با «روند برعکس» (آزمون فشار)، با کسر عدم‌قطعیت و جریمه برای هر شرط.
# الگوها فقط روی کندل بسته و فقط با گذشته حساب می‌شوند (سقف/کف چرخشی فقط بعد از تأیید)، و یک تابع
# (ga_primitives) هم در آموزش و هم در تصمیم زنده استفاده می‌شود تا هر دو دقیقاً یکی باشند.

# v2: قانون‌های متقارن (خرید + فروش آینه‌ای)، برازندگی بدون اثر روند، ۱۰۲ الگو، شکل‌های کشف‌شده و کتابخانه
# v3: ۱۳۴ الگو، شرط‌های عددی با آستانه‌ی یادگرفته‌شده، شکل‌های کندلی تا ۴ کندل، سواپ شبانه و گپ در شبیه‌سازی
# v4: همه‌ی الگوها معامله می‌شوند، هر کدام با حد سود/ضرر = میانگین سود/زیان ۲۰ دقیقه بعد از وقوع‌های گذشته‌اش
# v5 (pattern_runs_v6): بدون محدودیت زمان، حد ضرر ثابت، حد سود = میانگین بیشترین قیمت قبل از استاپ، حداقل میانگین سود؛
#     الگوهای بی‌اثر حذف و شکست‌های ۱۰/۲۰ روزه، شکست بعد از فشردگی و پولبک به سطح ۵ روزه اضافه شدند؛ هزینه‌ی واقعی
#     (لغزش ورود + کمیسیون) در شبیه‌ساز؛ ساعت آخر قبل از تعطیلی روزانه بدون ورود
# v6: کشف ژنتیک داخل لایه‌ی Walk-Forward (هر ماه دوباره بهینه‌سازی روی پنجره‌ی اخیر، سنجش روی ماه بعد) و حداکثر ۲ شرط
# v7: ۲۳ جفت الگوی کندلی تازه (با نسخه‌ی «بعد از روند»)، گزارش همه‌ی الگوها (pattern_report.csv)، دو روش خروج:
#     runs (پیش‌فرض، همان v6) و fixed (حد سود/ضرر ثابت با انتخاب بر اساس نرخ برد) — PATTERN_EXIT_MODE
# v8 (2026-10-15، «دست باز» به خواست کاربر): الگوریتم ژنتیک روی کل تاریخچه تا آخرین کندل یاد می‌گیرد (به‌جای پنجره‌ی ۶ ماهه‌ی
#     Walk-Forward) و هر الگو یا قانونی که بعد از همه‌ی هزینه‌ها میانگین سود مثبت داشته معامله می‌شود (حداقل ۹ دلار و ۱۰۰/۶۰
#     معامله برداشته شد؛ فقط کف آماری PATTERN_MIN_TRADES). قانون‌های موفق دوباره در کتابخانه ذخیره و دانه‌ی آموزش بعدی می‌شوند.
#     سقف ۲ شرط هر قانون (GA_MAX_FILTERS) مانده است. برای برگشت به Walk-Forward: GA_WALK_FORWARD = True
# v9 (2026-10-15، به خواست کاربر): ژنتیک روند را روی M30 می‌بیند (GA_TREND_TF، مستقل از تایم‌فریم دوم تنظیمات) و الگو را
#     روی تایم‌فریم اصلی (M5)؛ نشانه‌های روند M30 بیشتر شد (SMA50، ساختار سقف/کف، بازده و کارایی ۱۰ کندلی) و استفاده از آن‌ها
#     اختیاری است. آموزش فقط روی GA_TRAIN_BARS کندل آخر (۲۰۰ هزار). اگر مدل کمتر از GA_MIN_TRADES_PER_DAY معامله در روز
#     بدهد، قانون‌های سودده‌ی دیگر ژنتیک اضافه می‌شوند تا تقریباً هر روز معامله باشد (قانون ضررده هیچ‌وقت اضافه نمی‌شود).
# v10 (2026-10-16، به خواست کاربر): حد سود الگوها خیلی دور بود و بیشتر معامله‌ها قبل از رسیدن به آن حد ضرر می‌خوردند —
#     حد سود هر الگو حالا همانی است که در گذشته‌ی خودش بیشترین میانگین سود را داده (PATTERN_TP_METHOD)
# v11 (2026-10-23): کراس MACD با خط سیگنال SMA(9) مثل MT5 (قبلاً EMA؛ کراس گاهی ۱–۲ کندل دیرتر از نمودار ثبت می‌شد)،
#     و یک کندل دیگر هم‌زمان سیگنال خرید و فروش نمی‌دهد (چکش/مرد آویزان، ستاره‌ی دنباله‌دار/چکش وارونه، ...)
GA_VERSION = 11
GA_TREND_TF = "M30"       # تایم‌فریمی که ژنتیک روند را روی آن می‌بیند؛ باید از تایم‌فریم اصلی بزرگ‌تر باشد
                          # (None یا کوچک‌تر = همان تایم‌فریم دوم تنظیمات)
GA_TRAIN_BARS = 200000    # مدل الگو روی این تعداد کندل آخر آموزش می‌بیند (دیتابیس هم بیشتر از DB_MAX_BARS نگه نمی‌دارد)
GA_MIN_TRADES_PER_DAY = 1.0   # هدف: دست‌کم این‌قدر معامله در هر روز معاملاتی (شبیه‌سازی یک‌پوزیشنی روی کل داده‌ی آموزش)
GA_DAILY_FILL_MAX = 200   # حداکثر قانون سودده‌ی اضافه برای رسیدن به هدف معامله‌ی روزانه
GA_PIVOT_K = 4            # سقف/کف چرخشی: بالاتر/پایین‌تر از ۴ کندل هر طرف (تأیید ۴ کندل بعد، بدون نگاه به آینده)
GA_RECENT = 30            # آخرین سقف/کف یک الگوی کلاسیک حداکثر این‌قدر کندل قبل
GA_SPAN = 150             # کل یک الگوی کلاسیک باید در این تعداد کندل جا شود
GA_HTF_BARS = 30          # کندل بسته‌ی تایم‌فریم بالاتر لازم برای فیلترهای آن
GA_LOOKBACK_CAP = 6400    # سقف نگاه به عقب الگوهای چندروزه (کندل)؛ پنجره‌ی زنده باید از این بلندتر باشد
GA_ISLANDS = 4            # چند جمعیت مستقل (seed جدا) — جست‌وجوی گسترده‌تر، نتیجه‌ی پایدارتر
GA_POP = 240              # اندازه‌ی جمعیت هر جزیره
GA_GENERATIONS = 80       # حداکثر نسل هر جزیره
GA_SHAPE_PROB = 0.3       # سهم قانون‌های تصادفی که ماشه‌شان «شکل کندلی» تازه است (الگوی کشف‌شده توسط خود ربات)
GA_RANDOM_SHAPE_SEEDS = 40
# کتابخانه‌ی الگو: قانون‌های موفق هر آموزش ذخیره می‌شوند و در آموزش‌های بعدی دوباره سنجیده و تکامل داده می‌شوند
GA_LIB_MAX = 150          # حداکثر قانون در کتابخانه
GA_LIB_MAX_FAILS = 3      # قانونی که این تعداد آموزش پشت‌سرهم رد شود از کتابخانه حذف می‌شود
GA_LIB_CANDIDATES = 80    # چند قانون برتر و متنوع هر آموزش برای ورود به کتابخانه سنجیده می‌شوند
GA_ELITE = 10             # بهترین‌های هر نسل بدون تغییر به نسل بعد می‌روند
GA_IMMIGRANTS = 20        # قانون تصادفی تازه در هر نسل (حفظ تنوع)
GA_TOURNAMENT = 3
GA_CROSSOVER = 0.7
GA_MAX_FILTERS = 2        # حداکثر شرط هر قانون (فیلتر بله/خیر + شرط عددی) — جریمه‌ی پیچیدگی: قانون ساده‌تر گذشته را حفظ نمی‌کند
GA_MAX_NUM = 2            # حداکثر شرط عددی هر قانون
GA_PATIENCE = 16          # توقف وقتی این تعداد نسل بهترین قانون بهتر نشد
GA_FOLDS = 4              # تکه‌های زمانی Train برای سنجش پایداری (دوره‌ی ضررده دو برابر جریمه می‌شود)
GA_DRIFT_BLOCKS = 12      # Train به این تعداد بلوک زمانی تقسیم و روند هر بلوک از بازده‌ها کم می‌شود
GA_NEG_FOLD_PENALTY = 1.0 # ضریب جریمه‌ی اضافه برای امتیاز منفی هر دوره
GA_FILTER_PENALTY = 0.5   # جریمه‌ی هر شرط = این ضریب × میانه‌ی |بازده یک معامله|
GA_LOCAL_TOP = 24         # چند قانون برتر و متنوع بعد از تکامل با جست‌وجوی محلی (خروج و آستانه‌ها) صیقل داده می‌شوند
GA_MAX_OVERLAP = 0.6      # دو قانون با هم‌پوشانی سیگنال بیشتر از این تکراری حساب می‌شوند
GA_SEED = 42
GA_POOL_NEW = 100         # سقف عملی تعداد قانون‌های متنوع ژنتیک در مدل (فیلتر کیفیت نیست؛ فقط برای سرعت تصمیم زنده)
GA_SCAN_MAX = 5000        # چند قانون برتر سنجیده‌شده برای پیدا کردن قانون‌های متنوع بررسی می‌شوند
# v8: True = روش v6/v7 (کشف ماهانه روی پنجره‌ی GA_WF_TRAIN_MONTHS ماهه + دروازه‌ی سابقه‌ی Walk-Forward)؛
#     False = «دست باز»: کشف روی کل تاریخچه و معامله‌ی هر قانون سودده
GA_WALK_FORWARD = False

# ── Walk-Forward Optimization: لایه‌ی کنترل دور هسته‌ی ژنتیک (GAWalkForward) ──
# تاریخچه به ماه‌های متوالی تقسیم می‌شود؛ در هر گام هسته‌ی ژنتیک فقط روی GA_WF_TRAIN_MONTHS ماهِ قبل بهینه می‌شود و
# قانون‌هایش روی GA_WF_TEST_MONTHS ماهِ بعد (که ندیده) معامله می‌شوند؛ گام بعد یک ماه جلوتر. قانون‌های ماه جاری فقط وقتی
# معامله می‌شوند که سابقه‌ی همین روش در GA_WF_STEPS گام اخیر به حداقل میانگین سود رسیده باشد.
GA_WF_TRAIN_MONTHS = 6    # پنجره‌ی آموزش هر گام (ماه). آزمون ۴۸ ماهه: ۶ ماه ‎+0.22‎ و ۱۲ ماه ‎+0.27 ATR‎ هر معامله، ۲۴ و ۳۶ ماه منفی
GA_WF_TEST_MONTHS = 1     # پنجره‌ی تست هر گام = فاصله‌ی بهینه‌سازی دوباره (ماه)
GA_WF_STEPS = 24          # چند گام آخر سابقه‌ی Walk-Forward را می‌سازند
GA_WF_ISLANDS = 1         # بودجه‌ی هسته‌ی ژنتیک در هر گام (کمتر از آموزش یک‌باره، چون هر ماه تکرار می‌شود)
GA_WF_POP = 120
GA_WF_GENERATIONS = 25
GA_WF_PATIENCE = 10
GA_WF_TOP = 8             # چند قانون برتر و متنوع هر گام سنجیده می‌شوند
GA_WF_MIN_TRADES = 60     # حداقل معامله‌ی یک قانون در پنجره‌ی آموزش
GA_WF_MIN_OOS_TRADES = 30  # حداقل معامله‌ی ماه‌های تست برای قضاوت درباره‌ی سابقه
GA_WF_MIN_PROFIT_USD = None  # حداقل میانگین سود هر معامله‌ی ماه‌های تست (دلار با LOT فعلی) تا قانون‌های ژنتیک معامله شوند؛
                             # None = همان PATTERN_MIN_PROFIT_USD (قانون کاربر برای همه‌ی الگوها)
GA_WF_TAIL_DAYS = 30      # کندل‌های بعد از ماه تست فقط برای بسته شدن معامله‌هایی که در ماه تست باز شده‌اند

# ── آمار الگو (طرح کاربر، آزموده روی ۱۱ سال طلا با به‌روزرسانی ماهانه) ──
# بعد از هر وقوع الگو در گذشته، معامله بدون محدودیت زمان باز می‌ماند تا حد ضرر ثابت بخورد؛ «سود» آن وقوع = بیشترین حرکت
# قیمت به نفع معامله قبل از خوردن حد ضرر. حد سود الگو = میانگین این سودها.
# سپس الگو با همین حد سود و حد ضرر روی کل گذشته‌اش معامله می‌شود (یک پوزیشن در هر لحظه، بعد از اسپرد + لغزش + کمیسیون +
# سواپ) و فقط اگر میانگین سود واقعی هر معامله‌اش ≥ PATTERN_MIN_PROFIT_USD دلار (با LOT فعلی و نوسان امروز) باشد معامله
# می‌شود. آمار در هر آموزش با همه‌ی داده تا آخرین کندل دوباره حساب می‌شود.
# اگر چند الگو روی یک کندل برقرار باشند، الگویی معامله می‌شود که میانگین سود هر معامله‌اش در گذشته بیشتر بوده.
# v7 (2026-10-15): دو روش خروج — PATTERN_EXIT_MODE:
#   "runs" (روش v6، پیش‌فرض): حد ضرر ثابت PATTERN_SL_ATR، حد سود هر الگو = میانگین بیشترین قیمت قبل از خوردن حد ضرر در
#          گذشته‌اش، بدون محدودیت زمان؛ انتخاب: میانگین سود واقعی هر معامله ≥ PATTERN_MIN_PROFIT_USD (نرخ برد اینجا ~۲۰٪
#          است و معیار نیست). آزمون ۱۲ ساله: فقط شکست‌های چندروزه‌ی صعودی از ورود تصادفی بهترند (z≈7–10).
#   "fixed": حد سود/ضرر ثابت PATTERN_TP_ATR / PATTERN_SL_ATR؛ انتخاب: نرخ برد ≥ PATTERN_MIN_WIN_RATE (و اگر
#          PATTERN_SELECT_BY_PROFIT: یا میانگین ≥ PATTERN_MIN_PROFIT_USD). آزمون ۱۲ ساله با TP 2 / SL 1.5 ATR و حداقل ۲۰
#          معامله: هیچ الگویی روی Train به ۵۵٪ نرسید (برد الگوها ۴۰–۴۳٪، سربه‌سر ≈۴۶٪)؛ ۳۸ ترکیب «الگو + یک فیلتر» که روی
#          Train ≥۵۵٪ بودند روی Validation/Test فقط ۴۲–۴۳٪ بردند؛ همه‌ی ۱۴۶ الگو با هم ≈۲۷ معامله در روز با ‎−0.06 تا −0.12 ATR‎.
PATTERN_EXIT_MODE = "runs"
# v10: حد سود هر الگو در حالت runs — PATTERN_TP_METHOD:
#   "best" (پیش‌فرض): بین چندک‌های ۱۰٪ تا ۹۰٪ «بیشترین حرکت به نفع معامله قبل از خوردن حد ضرر» در گذشته‌ی خود الگو (و
#          میانگینش)، حد سودی که بیشترین میانگین سود هر معامله را داده؛ برای الگویی که سودش از حرکت‌های کوتاه است نزدیک
#          می‌شود و برای الگویی که سودش از روندهای بلند است دور می‌ماند. دقیق است، نه تقریبی: معامله با حد سود x قبل از حد
#          ضرر به x می‌رسد اگر و فقط اگر آن حرکت ≥ x باشد (اگر هر دو در یک کندل لمس شوند ضرر، مثل شبیه‌ساز)
#   "median": حد سودی که نیمی از وقوع‌های گذشته قبل از حد ضرر به آن رسیده‌اند (نرخ برد حدود ۵۰٪)
#   "mean": روش v6 تا v9 — چند حرکت خیلی بلند میانگین را بالا می‌کشند و حدود ۸۰٪ معامله‌ها قبل از رسیدن به آن حد ضرر
#          می‌خورند
PATTERN_TP_METHOD = "best"
PATTERN_TP_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# کف حد سود (ضریب فاصله‌ی حد ضرر) در روش best: بدون آن، روی الگوی بی‌مزیت حد سودهای خیلی کوچک (۹۰٪ برد، سود ریز، گاهی
# یک حد ضرر کامل) برنده می‌شوند — شکننده در برابر لغزش. 0.5 یعنی حد سود دست‌کم نصف حد ضرر (با حد ضرر ۲×ATR: ۱×ATR)
PATTERN_TP_MIN_RR = 0.5
# حداقل «نرخ رسیدن به حد سود» در روش best: فقط حد سودهایی سنجیده می‌شوند که دست‌کم این سهم از وقوع‌های گذشته‌ی الگو قبل از
# حد ضرر به آن‌ها رسیده‌اند (0.4 = دست‌کم ۴۰٪)؛ 0 = بدون محدودیت (فقط بیشترین سود — روی روندهای بلند حد سود دور می‌ماند)
PATTERN_TP_MIN_HIT = 0.4
DECISION_RULE = f"pattern_{PATTERN_EXIT_MODE}_tp-{PATTERN_TP_METHOD}_v10"
PATTERN_SL_ATR = 2.0            # حد ضرر ثابت همه‌ی الگوها (ضریب ATR کندل سیگنال؛ در حالت fixed همان FIXED_SL_PRICE دلار)
                                # (برای حالت fixed آزموده‌شده: 1.5)
# v8 «دست باز»: هر الگو/قانونی با میانگین سود مثبت بعد از همه‌ی هزینه‌ها معامله می‌شود (قبلاً حداقل ۹ دلار و ۱۰۰ معامله)
PATTERN_MIN_PROFIT_USD = 0.0    # حداقل میانگین سود واقعی هر معامله‌ی الگو در گذشته بعد از همه‌ی هزینه‌ها (دلار حساب با LOT فعلی)؛
                                # 0 = هر میانگین مثبت. میانگین منفی یا صفر هیچ‌وقت معامله نمی‌شود
PATTERN_MIN_TRADES = 20         # کف آماری: الگویی که کمتر از این معامله‌ی گذشته داشته آمارش فقط شانس است
# فقط حالت fixed:
PATTERN_TP_ATR = 2.0            # حد سود ثابت (ضریب ATR کندل سیگنال؛ در حالت fixed همان FIXED_TP_PRICE دلار)
PATTERN_MIN_WIN_RATE = 0.55     # الگویی که در گذشته دست‌کم این نرخ برد را داشته انتخاب می‌شود
PATTERN_SELECT_BY_PROFIT = True   # True: الگوی سودده با نرخ برد کمتر هم انتخاب می‌شود (میانگین ≥ PATTERN_MIN_PROFIT_USD)
PATTERN_SL_CHOICES = (1.0,)     # حد ضرر هر الگو: این ضرایبِ PATTERN_SL_ATR که هر الگو بهترینش را از گذشته‌ی خودش برمی‌دارد.
                                # (0.75, 1.0, 1.5) = انتخاب بین 1.5/2/3 ATR آزموده شد: الگوهای ضعیف‌تری وارد شدند و نتیجه‌ی
                                # دوره‌ی دیده‌نشده بدتر شد (Validation ‎+0.88 → −0.16 ATR‎)؛ پس فعلاً فقط یک حد ضرر
GA_FITNESS_FLIP = False         # True: برازندگی ژنتیک بدترِ «روند واقعی» و «روند برعکس» (قانونی که سودش از روند است جریمه می‌شود)؛
                                # False: میانگین سود واقعی هر دوره‌ی Train، هم‌راستا با معیار انتخاب (حداقل میانگین سود) و
                                # Walk-Forward که جلوی بیش‌برازش را می‌گیرد
MODEL_SLIPPAGE_ATR = 0.033      # لغزش ورود (میانگین اندازه‌گیری‌شده از معاملات واقعی: ≈0.14 دلار قیمت با ATR≈4.3)
MODEL_COMMISSION_FRAC = 0.000036  # کمیسیون رفت‌وبرگشت پیش‌فرض (کسری از قیمت؛ از معاملات واقعی حساب به‌روز می‌شود)
PRE_BREAK_MINUTES = 60          # این‌قدر مانده به تعطیلی روزانه، معامله‌ی تازه باز نمی‌شود
# کشف ترکیب‌های «الگو + شرایط بازار» با الگوریتم ژنتیک — به خواست کاربر روشن (2026-10-11). v8: کشف روی کل تاریخچه و هر
# ترکیب با میانگین سود مثبت معامله می‌شود (بدون سنجش روی دوره‌ای که ندیده — نتیجه‌ی گزارش‌شده‌اش درون‌نمونه‌ای است).
# یادداشت آزمون ۱۱ ساله (انتخاب از Train، معامله روی Validation/Test): با ترکیب‌ها تعداد معامله دو برابر و سود هر معامله
# کمتر شد (فقط الگوها: ‎+0.88 / +1.46 ATR؛ با ترکیب‌ها: ‎+0.23 / +0.78‎) و آموزش روی ۸۰۰ هزار کندل از حدود ۱ دقیقه به
# حدود ۳۰ دقیقه می‌رسد. False = فقط الگوهای نام‌دار.
GA_DISCOVERY = True

# (الگوی صعودی، الگوی نزولی) — نام‌ها در فایل مدل ذخیره می‌شوند؛ عوضشان نکنید
# v6: ۲۱ جفت که در آزمون ۱۱ ساله (M5 و M15، به‌روزرسانی ماهانه) حتی یک ماه هم به حداقل میانگین سود نرسیدند حذف شدند
#     (NR7، FVG، RSI2، تقاطع و پولبک میانگین و ...) و ۴ جفت شکست چندروزه اضافه شد. انگالف، چکش/ستاره‌ی دنباله‌دار و
#     پین‌بار به خواست کاربر برگشتند (الگوهای مهم؛ مثل بقیه فقط وقتی معامله می‌شوند که به حداقل میانگین سود برسند).
GA_PATTERN_PAIRS = (
    # ── الگوهای کندلی ──
    ('engulfing_bull', 'engulfing_bear'),
    ('hammer', 'shooting_star'),
    ('harami_bull', 'harami_bear'),
    ('three_soldiers', 'three_crows'),
    ('marubozu_bull', 'marubozu_bear'),
    ('three_inside_up', 'three_inside_down'),
    # ── الگوهای کلاسیک (روی سقف/کف‌های چرخشی تأییدشده) ──
    ('double_bottom', 'double_top'),
    ('inv_head_shoulders', 'head_shoulders'),
    ('triangle_break_up', 'triangle_break_down'),
    ('ascending_triangle', 'descending_triangle'),
    ('falling_wedge', 'rising_wedge'),
    ('bos_up', 'bos_down'),
    ('choch_up', 'choch_down'),
    ('bull_flag', 'bear_flag'),
    ('box_break_up', 'box_break_down'),
    # ── الگوهای کندلی بیشتر ──
    ('dragonfly_doji', 'gravestone_doji'),
    ('rising_three', 'falling_three'),
    ('morning_doji_star', 'evening_doji_star'),
    ('pin_bar_bull', 'pin_bar_bear'),
    ('two_bar_reversal_up', 'two_bar_reversal_down'),
    ('hikkake_up', 'hikkake_down'),
    ('three_line_strike_bull', 'three_line_strike_bear'),
    ('harami_cross_bull', 'harami_cross_bear'),
    # ── الگوهای کلاسیک و Smart Money بیشتر ──
    ('triple_bottom', 'triple_top'),
    ('channel_break_up', 'channel_break_down'),
    ('breakout_retest_up', 'breakout_retest_down'),
    ('spring', 'upthrust'),
    ('trendline_break_up', 'trendline_break_down'),
    ('order_block_up', 'order_block_down'),
    ('equal_lows_sweep', 'equal_highs_sweep'),
    ('fib_bounce', 'fib_reject'),
    ('rev123_up', 'rev123_down'),
    ('donchian_break_up', 'donchian_break_down'),
    ('squeeze_break_up', 'squeeze_break_down'),
    # ── v3: شکست‌های چندروزه، سطوح روز قبل، رنج اول روز، مومنتوم و بازگشت به میانگین ──
    ('brk_1d_up', 'brk_1d_down'),
    ('brk_2d_up', 'brk_2d_down'),
    ('brk_5d_up', 'brk_5d_down'),
    ('pdh_break', 'pdl_break'),
    ('pdl_reclaim', 'pdh_reject'),
    ('orb_up', 'orb_down'),
    ('early_range_break_up', 'early_range_break_down'),
    ('keltner_break_up', 'keltner_break_down'),
    ('vol_expansion_up', 'vol_expansion_down'),
    ('zscore_reversal_up', 'zscore_reversal_down'),
    ('macd_cross_up', 'macd_cross_down'),
    ('impulse_up', 'impulse_down'),
    # ── v6: شکست‌های ۱۰ و ۲۰ روزه، شکست ۲ روزه بعد از فشردگی، پولبک به سطح شکسته‌شده‌ی ۵ روزه ──
    ('brk_10d_up', 'brk_10d_down'),
    ('brk_20d_up', 'brk_20d_down'),
    ('squeeze2d_break_up', 'squeeze2d_break_down'),
    ('brk5_retest_up', 'brk5_retest_down'),
    # ── v7: الگوهای کندلی بیشتر ──
    ('piercing_line', 'dark_cloud'),
    ('tweezer_bottom', 'tweezer_top'),
    ('inverted_hammer', 'hanging_man'),
    ('morning_star', 'evening_star'),
    ('three_outside_up', 'three_outside_down'),
    ('belt_hold_bull', 'belt_hold_bear'),
    ('inside_break_up', 'inside_break_down'),
    ('outside_reversal_up', 'outside_reversal_down'),
    ('homing_pigeon', 'descending_hawk'),
    ('matching_low', 'matching_high'),
)
# v7: الگوهای برگشتی که نسخه‌ی «بعد از روند» (_t) هم دارند: (صعودی، نزولی، تعداد کندل الگو)
GA_TREND_VARIANTS = (
    ('engulfing_bull', 'engulfing_bear', 2),
    ('hammer', 'shooting_star', 1),
    ('harami_bull', 'harami_bear', 2),
    ('pin_bar_bull', 'pin_bar_bear', 1),
    ('dragonfly_doji', 'gravestone_doji', 1),
    ('morning_doji_star', 'evening_doji_star', 3),
    ('two_bar_reversal_up', 'two_bar_reversal_down', 2),
    ('harami_cross_bull', 'harami_cross_bear', 2),
    ('three_inside_up', 'three_inside_down', 3),
    ('piercing_line', 'dark_cloud', 2),
    ('tweezer_bottom', 'tweezer_top', 2),
    ('morning_star', 'evening_star', 3),
    ('three_outside_up', 'three_outside_down', 3),
)
GA_PATTERN_PAIRS += tuple((b + '_t', s + '_t') for b, s, _ in GA_TREND_VARIANTS)
GA_BULL = tuple(p[0] for p in GA_PATTERN_PAIRS)
GA_BEAR = tuple(p[1] for p in GA_PATTERN_PAIRS)
GA_MIRROR = {**{a: b for a, b in GA_PATTERN_PAIRS}, **{b: a for a, b in GA_PATTERN_PAIRS}}

GA_FILTERS = (
    'above_sma20', 'above_sma50', 'above_sma100', 'sma50_rising', 'sma20_above_sma50',
    'structure_up', 'structure_down', 'range_low_50', 'range_high_50',
    'momentum_up', 'momentum_down', 'vol_high', 'vol_low', 'wide_bar',
    'rsi_low', 'rsi_high', 'prev_doji',
    'htf_up', 'htf_rising', 'htf_bull_candle', 'htf_bull_pattern', 'htf_bear_pattern',
    'htf_range_low', 'htf_range_high',
    'hour_00_04', 'hour_04_08', 'hour_08_12', 'hour_12_16', 'hour_16_20', 'hour_20_24',
    # v3
    'above_sma200', 'above_day_open', 'prev_day_bull', 'inside_prev_day', 'monday', 'friday',
    # v9: روند تایم‌فریم بالاتر (M30)
    'htf_above_sma50', 'htf_sma20_above_sma50', 'htf_struct_up', 'htf_struct_down',
)

# آینه‌ی هر فیلتر برای نسخه‌ی فروش یک قانون: نام → (نام آینه، برعکس شود؟). فیلترهای بی‌جهت (نوسان، ساعت) همان می‌مانند.
GA_FILTER_MIRROR = {
    'above_sma20': ('above_sma20', True), 'above_sma50': ('above_sma50', True),
    'above_sma100': ('above_sma100', True), 'sma50_rising': ('sma50_rising', True),
    'sma20_above_sma50': ('sma20_above_sma50', True),
    'structure_up': ('structure_down', False), 'structure_down': ('structure_up', False),
    'range_low_50': ('range_high_50', False), 'range_high_50': ('range_low_50', False),
    'momentum_up': ('momentum_down', False), 'momentum_down': ('momentum_up', False),
    'vol_high': ('vol_high', False), 'vol_low': ('vol_low', False), 'wide_bar': ('wide_bar', False),
    'rsi_low': ('rsi_high', False), 'rsi_high': ('rsi_low', False), 'prev_doji': ('prev_doji', False),
    'htf_up': ('htf_up', True), 'htf_rising': ('htf_rising', True), 'htf_bull_candle': ('htf_bull_candle', True),
    'htf_bull_pattern': ('htf_bear_pattern', False), 'htf_bear_pattern': ('htf_bull_pattern', False),
    'htf_range_low': ('htf_range_high', False), 'htf_range_high': ('htf_range_low', False),
    **{f'hour_{h:02d}_{h + 4:02d}': (f'hour_{h:02d}_{h + 4:02d}', False) for h in range(0, 24, 4)},
    'above_sma200': ('above_sma200', True), 'above_day_open': ('above_day_open', True),
    'prev_day_bull': ('prev_day_bull', True), 'inside_prev_day': ('inside_prev_day', False),
    'monday': ('monday', False), 'friday': ('friday', False),
    'htf_above_sma50': ('htf_above_sma50', True), 'htf_sma20_above_sma50': ('htf_sma20_above_sma50', True),
    'htf_struct_up': ('htf_struct_down', False), 'htf_struct_down': ('htf_struct_up', False),
}

# ویژگی‌های عددی (شرط «ویژگی > آستانه» یا «ویژگی < آستانه»؛ آستانه از چندک‌های Train انتخاب و تکامل داده می‌شود).
# همه بدون واحد (نسبت به ATR یا بازه‌ی [-1, 1]) و فقط از گذشته. نام → (ویژگی آینه برای نسخه‌ی فروش، علامت برعکس شود؟):
# شرط خرید «f > T» در نسخه‌ی فروش «mirror < −T» می‌شود اگر علامت برعکس شود، وگرنه «mirror > T».
GA_NUM_FEATURES = {
    'rsi14': ('rsi14', True), 'rsi2': ('rsi2', True),
    'ret5': ('ret5', True), 'ret20': ('ret20', True), 'ret60': ('ret60', True),
    'sma20_dist': ('sma20_dist', True), 'sma50_dist': ('sma50_dist', True), 'sma200_dist': ('sma200_dist', True),
    'sma20_slope': ('sma20_slope', True), 'sma50_slope': ('sma50_slope', True),
    'pos20': ('pos20', True), 'pos100': ('pos100', True), 'pos_1d': ('pos_1d', True), 'pos_5d': ('pos_5d', True),
    'eff20': ('eff20', True), 'body': ('body', True), 'clv': ('clv', True), 'streak': ('streak', True),
    'pdh_dist': ('pdl_dist', True), 'pdl_dist': ('pdh_dist', True), 'dopen_dist': ('dopen_dist', True),
    'htf_sma_dist': ('htf_sma_dist', True), 'htf_pos': ('htf_pos', True), 'htf_slope': ('htf_slope', True),
    'atr_ratio': ('atr_ratio', False), 'range': ('range', False), 'contraction': ('contraction', False),
    'bb_width': ('bb_width', False), 'adx': ('adx', False), 'day_range': ('day_range', False),
    # v9: روند تایم‌فریم بالاتر (M30)
    'htf_sma50_dist': ('htf_sma50_dist', True), 'htf_ret10': ('htf_ret10', True), 'htf_eff10': ('htf_eff10', True),
}
GA_NUM_NAMES = {
    'rsi14': 'RSI14−50', 'rsi2': 'RSI2−50', 'ret5': '5-bar return/ATR', 'ret20': '20-bar return/ATR',
    'ret60': '60-bar return/ATR', 'sma20_dist': '(close−SMA20)/ATR', 'sma50_dist': '(close−SMA50)/ATR',
    'sma200_dist': '(close−SMA200)/ATR', 'sma20_slope': 'SMA20 slope/ATR', 'sma50_slope': 'SMA50 slope/ATR',
    'pos20': 'pos in 20-bar range', 'pos100': 'pos in 100-bar range', 'pos_1d': 'pos in 1-day range',
    'pos_5d': 'pos in 5-day range', 'eff20': '20-bar efficiency', 'body': 'body/ATR', 'clv': 'close location',
    'streak': 'up/down close streak', 'pdh_dist': '(close−prev-day high)/ATR', 'pdl_dist': '(close−prev-day low)/ATR',
    'dopen_dist': '(close−day open)/ATR', 'htf_sma_dist': 'HTF (close−SMA20)/ATR', 'htf_pos': 'HTF pos in range',
    'htf_slope': 'HTF SMA20 slope/ATR', 'atr_ratio': 'ATR/median ATR', 'range': 'bar range/ATR',
    'contraction': 'range 5/50', 'bb_width': 'stdev20/ATR', 'adx': 'ADX14', 'day_range': "today's range/ATR",
    'htf_sma50_dist': 'HTF (close−SMA50)/ATR', 'htf_ret10': 'HTF 10-bar return/ATR', 'htf_eff10': 'HTF 10-bar efficiency',
}

GA_NAMES = {
    'engulfing_bull': 'Bullish engulfing', 'engulfing_bear': 'Bearish engulfing',
    'hammer': 'Hammer', 'shooting_star': 'Shooting star',
    'harami_bull': 'Bullish harami', 'harami_bear': 'Bearish harami',
    'three_soldiers': 'Three white soldiers', 'three_crows': 'Three black crows',
    'marubozu_bull': 'Bullish marubozu', 'marubozu_bear': 'Bearish marubozu',
    'three_inside_up': 'Three inside up', 'three_inside_down': 'Three inside down',
    'double_bottom': 'Double bottom', 'double_top': 'Double top',
    'inv_head_shoulders': 'Inverse head & shoulders', 'head_shoulders': 'Head & shoulders',
    'triangle_break_up': 'Triangle breakout up', 'triangle_break_down': 'Triangle breakout down',
    'ascending_triangle': 'Ascending triangle', 'descending_triangle': 'Descending triangle',
    'falling_wedge': 'Falling wedge', 'rising_wedge': 'Rising wedge',
    'bos_up': 'Break of structure up', 'bos_down': 'Break of structure down',
    'choch_up': 'Change of character up', 'choch_down': 'Change of character down',
    'bull_flag': 'Bull flag', 'bear_flag': 'Bear flag',
    'box_break_up': 'Range breakout up', 'box_break_down': 'Range breakout down',
    'dragonfly_doji': 'Dragonfly doji', 'gravestone_doji': 'Gravestone doji',
    'rising_three': 'Rising three methods', 'falling_three': 'Falling three methods',
    'morning_doji_star': 'Morning doji star', 'evening_doji_star': 'Evening doji star',
    'pin_bar_bull': 'Bullish pin bar', 'pin_bar_bear': 'Bearish pin bar',
    'two_bar_reversal_up': 'Bullish two-bar reversal', 'two_bar_reversal_down': 'Bearish two-bar reversal',
    'hikkake_up': 'Bullish hikkake', 'hikkake_down': 'Bearish hikkake',
    'three_line_strike_bull': 'Bullish three-line strike', 'three_line_strike_bear': 'Bearish three-line strike',
    'harami_cross_bull': 'Bullish harami cross', 'harami_cross_bear': 'Bearish harami cross',
    'triple_bottom': 'Triple bottom', 'triple_top': 'Triple top',
    'channel_break_up': 'Down-channel breakout up', 'channel_break_down': 'Up-channel breakdown',
    'breakout_retest_up': 'Breakout retest up', 'breakout_retest_down': 'Breakdown retest',
    'spring': 'Wyckoff spring', 'upthrust': 'Wyckoff upthrust',
    'trendline_break_up': 'Trendline break up', 'trendline_break_down': 'Trendline break down',
    'order_block_up': 'Bullish order block', 'order_block_down': 'Bearish order block',
    'equal_lows_sweep': 'Equal lows sweep', 'equal_highs_sweep': 'Equal highs sweep',
    'fib_bounce': 'Fibonacci 50-78.6% bounce', 'fib_reject': 'Fibonacci 50-78.6% rejection',
    'rev123_up': '1-2-3 bottom', 'rev123_down': '1-2-3 top',
    'donchian_break_up': '55-bar Donchian breakout', 'donchian_break_down': '55-bar Donchian breakdown',
    'squeeze_break_up': 'Bollinger squeeze breakout up', 'squeeze_break_down': 'Bollinger squeeze breakout down',
    'above_sma20': 'close>SMA20', 'above_sma50': 'close>SMA50', 'above_sma100': 'close>SMA100',
    'sma50_rising': 'SMA50 rising', 'sma20_above_sma50': 'SMA20>SMA50',
    'structure_up': 'HH+HL structure', 'structure_down': 'LH+LL structure',
    'range_low_50': 'bottom of 50-bar range', 'range_high_50': 'top of 50-bar range',
    'momentum_up': '10-bar momentum up', 'momentum_down': '10-bar momentum down',
    'vol_high': 'high volatility', 'vol_low': 'low volatility', 'wide_bar': 'wide bar',
    'rsi_low': 'RSI<35', 'rsi_high': 'RSI>65', 'prev_doji': 'doji before',
    'htf_up': 'HTF above SMA20', 'htf_rising': 'HTF SMA20 rising', 'htf_bull_candle': 'last HTF candle bullish',
    'htf_bull_pattern': 'HTF bullish engulfing/hammer', 'htf_bear_pattern': 'HTF bearish engulfing/star',
    'htf_range_low': 'HTF bottom of range', 'htf_range_high': 'HTF top of range',
    'hour_00_04': 'server hour 00-04', 'hour_04_08': 'server hour 04-08', 'hour_08_12': 'server hour 08-12',
    'hour_12_16': 'server hour 12-16', 'hour_16_20': 'server hour 16-20', 'hour_20_24': 'server hour 20-24',
    # v3
    'brk_1d_up': '1-day high breakout', 'brk_1d_down': '1-day low breakdown',
    'brk_2d_up': '2-day high breakout', 'brk_2d_down': '2-day low breakdown',
    'brk_5d_up': '5-day high breakout', 'brk_5d_down': '5-day low breakdown',
    'pdh_break': 'Previous-day high break', 'pdl_break': 'Previous-day low break',
    'pdl_reclaim': 'Previous-day low sweep & reclaim', 'pdh_reject': 'Previous-day high sweep & reject',
    'orb_up': 'Opening-range (1st hour) breakout', 'orb_down': 'Opening-range (1st hour) breakdown',
    'early_range_break_up': 'Early-session (8h) range breakout', 'early_range_break_down': 'Early-session (8h) range breakdown',
    'keltner_break_up': 'Keltner (SMA20+2ATR) breakout', 'keltner_break_down': 'Keltner (SMA20−2ATR) breakdown',
    'vol_expansion_up': 'Volatility expansion up', 'vol_expansion_down': 'Volatility expansion down',
    'zscore_reversal_up': 'Z-score < −2.2 reversal up', 'zscore_reversal_down': 'Z-score > 2.2 reversal down',
    'macd_cross_up': 'MACD cross up below zero', 'macd_cross_down': 'MACD cross down above zero',
    'impulse_up': '3-bar impulse up', 'impulse_down': '3-bar impulse down',
    # v6
    'brk_10d_up': '10-day high breakout', 'brk_10d_down': '10-day low breakdown',
    'brk_20d_up': '20-day high breakout', 'brk_20d_down': '20-day low breakdown',
    'squeeze2d_break_up': '2-day high breakout after a tight range',
    'squeeze2d_break_down': '2-day low breakdown after a tight range',
    'brk5_retest_up': 'Retest of the broken 5-day high', 'brk5_retest_down': 'Retest of the broken 5-day low',
    'above_sma200': 'close>SMA200', 'above_day_open': 'close>day open', 'prev_day_bull': 'previous day bullish',
    'inside_prev_day': "inside previous day's range", 'monday': 'Monday', 'friday': 'Friday',
    # v9
    'htf_above_sma50': 'HTF above SMA50', 'htf_sma20_above_sma50': 'HTF SMA20>SMA50',
    'htf_struct_up': 'HTF HH+HL structure', 'htf_struct_down': 'HTF LH+LL structure',
    # v7
    'piercing_line': 'Piercing line', 'dark_cloud': 'Dark cloud cover',
    'tweezer_bottom': 'Tweezer bottom', 'tweezer_top': 'Tweezer top',
    'inverted_hammer': 'Inverted hammer (after downtrend)', 'hanging_man': 'Hanging man (after uptrend)',
    'morning_star': 'Morning star', 'evening_star': 'Evening star',
    'three_outside_up': 'Three outside up', 'three_outside_down': 'Three outside down',
    'belt_hold_bull': 'Bullish belt hold (after downtrend)', 'belt_hold_bear': 'Bearish belt hold (after uptrend)',
    'inside_break_up': 'Inside bar breakout up', 'inside_break_down': 'Inside bar breakout down',
    'outside_reversal_up': 'Bullish outside (key) reversal', 'outside_reversal_down': 'Bearish outside (key) reversal',
    'homing_pigeon': 'Homing pigeon (after downtrend)', 'descending_hawk': 'Descending hawk (after uptrend)',
    'matching_low': 'Matching low (after downtrend)', 'matching_high': 'Matching high (after uptrend)',
}
GA_NAMES.update({nm + '_t': GA_NAMES[nm] + (' after downtrend' if i == 0 else ' after uptrend')
                 for b, s, _ in GA_TREND_VARIANTS for i, nm in enumerate((b, s))})


def _ga_day_bars(P):
    """تعداد کندل یک روز در تایم‌فریم P (ثانیه)"""
    return max(1, 86400 // max(1, int(P)))


def ga_lookback(P):
    """بلندترین نگاه به عقب الگوها/ویژگی‌ها (میانگین ۲۰ روزه‌ی رنج ۲ روزه = ۲۲ روز)، حداکثر GA_LOOKBACK_CAP کندل"""
    return int(min(GA_LOOKBACK_CAP, 22 * _ga_day_bars(P)))


def ga_htf_seconds(primary_tf_seconds=None, fallback=None):
    """
    تایم‌فریم روند مدل الگو (ثانیه): GA_TREND_TF (M30) اگر از تایم‌فریم اصلی بزرگ‌تر باشد، وگرنه fallback یا تایم‌فریم
    دوم تنظیمات. کندل‌هایش جدا دانلود نمی‌شوند؛ از کنار هم گذاشتن کندل‌های اصلی ساخته می‌شوند (۶ کندل M5 = یک M30).
    """
    P = int(primary_tf_seconds or PRIMARY_TF_SECONDS)
    S = TF_SECONDS.get(str(GA_TREND_TF or '').strip().upper())
    if S and int(S) > P:
        return int(S)
    return int(fallback or SECONDARY_TF_SECONDS)


def ga_warmup(primary_tf_seconds=None, secondary_tf_seconds=None):
    """کندل‌های اول هر سری که همه‌ی الگوها/فیلترها هنوز کامل نیستند (پنجره‌ی زنده باید بلندتر باشد)"""
    P = int(primary_tf_seconds or PRIMARY_TF_SECONDS)
    S = int(secondary_tf_seconds or ga_htf_seconds(P))
    ratio = max(1, S // max(1, P))
    return max(300, (GA_HTF_BARS + 2) * ratio + 20, ga_lookback(P) + 60)


def ga_model_path(model_path=None):
    """فایل مدل الگوریتم ژنتیک (کنار مدل قبلی، در پوشه‌ی مدل همین تنظیم)"""
    return os.path.join(os.path.dirname(model_path or MODEL_PATH) or ".", "ga_patterns.json")


class _NoScaler:
    """مدل GA اسکیلر ندارد؛ این جای‌نگهدار فقط برای کدی است که (model, scaler_p, scaler_s) می‌خواهد"""

    def transform(self, x):
        return x


GA_NO_SCALER = _NoScaler()


# ─────────────────────────── تشخیص الگو (مشترک آموزش و زنده) ───────────────────────────
def _ga_shift(x, k=1):
    out = np.full(len(x), np.nan)
    if 0 < k < len(x):
        out[k:] = x[:-k]
    return out


def _ga_shift_bool(x, k=1):
    out = np.zeros(len(x), dtype=bool)
    if 0 < k < len(x):
        out[k:] = x[:-k]
    return out


def _ga_roll(x, w, how):
    """پنجره‌ی ثابت (بدون مقدار اولیه‌ی بازگشتی) تا روی پنجره‌ی کوتاه زنده و تاریخچه‌ی کامل یکی باشد"""
    return getattr(pd.Series(x).rolling(w, min_periods=w), how)().to_numpy(dtype=float)


def _ga_pivots(x, k, use_max):
    """اندیس سقف‌ها/کف‌های چرخشی (k کندل هر طرف). کندل j فقط در کندل j+k معلوم می‌شود."""
    n = len(x)
    if n < 2 * k + 1:
        return np.zeros(0, dtype=np.int64)
    win = sliding_window_view(x, 2 * k + 1)
    center = win[:, k]
    if use_max:
        ok = (center > win[:, :k].max(axis=1)) & (center >= win[:, k + 1:].max(axis=1))
    else:
        ok = (center < win[:, :k].min(axis=1)) & (center <= win[:, k + 1:].min(axis=1))
    ok &= np.isfinite(center)
    return (np.flatnonzero(ok) + k).astype(np.int64)


def _ga_last_pivots(piv, k, n, m=3):
    """برای هر کندل i: اندیس m سقف/کف چرخشی آخری که تا کندل i تأیید شده‌اند (جدیدترین اول؛ -1 = نیست)"""
    out = []
    if len(piv) == 0:
        return [np.full(n, -1, dtype=np.int64) for _ in range(m)]
    cnt = np.searchsorted(piv + k, np.arange(n), side='right')
    for q in range(1, m + 1):
        pos = cnt - q
        out.append(np.where(pos >= 0, piv[np.clip(pos, 0, len(piv) - 1)], -1).astype(np.int64))
    return out


def _ga_line(i1, v1, i2, v2, at):
    """مقدار خط گذرنده از (i1, v1) و (i2, v2) در نقطه‌ی at، و شیب آن"""
    slope = (v2 - v1) / (i2 - i1)
    return v2 + slope * (at - i2), slope


def _ga_htf(o, h, l, c, t, P, S):
    """
    فیلترهای تایم‌فریم بالاتر از روی همان کندل‌های اصلی (گروه‌بندی با زمان؛ مثلاً ۳ کندل M5 = یک M15).
    برای هر کندل اصلی فقط آخرین کندل بالاتری استفاده می‌شود که تا بسته شدن آن کندل کامل شده است.
    خروجی: (فیلترهای بله/خیر، valid، ورودی‌های عددی htf_sma_dist / htf_pos / htf_slope)
    """
    n = len(c)
    names = ('htf_up', 'htf_rising', 'htf_bull_candle', 'htf_bull_pattern', 'htf_bear_pattern',
             'htf_range_low', 'htf_range_high',
             'htf_above_sma50', 'htf_sma20_above_sma50', 'htf_struct_up', 'htf_struct_down')
    num_names = ('htf_sma_dist', 'htf_pos', 'htf_slope', 'htf_sma50_dist', 'htf_ret10', 'htf_eff10')
    if S <= P or n == 0:
        return ({k: np.zeros(n, dtype=bool) for k in names}, np.zeros(n, dtype=bool),
                {k: np.full(n, np.nan) for k in num_names})
    g = t - t % S
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], n] - 1
    Go, Gc, Gt = o[starts], c[ends], g[starts]
    Gh = np.maximum.reduceat(h, starts)
    Gl = np.minimum.reduceat(l, starts)
    Gc1, Go1 = _ga_shift(Gc), _ga_shift(Go)
    tr = np.fmax(Gh - Gl, np.fmax(np.abs(Gh - Gc1), np.abs(Gl - Gc1)))
    Ga = _ga_roll(tr, 14, 'mean')
    sma = _ga_roll(Gc, 20, 'mean')
    Gab, Gab1 = np.abs(Gc - Go), np.abs(Gc1 - Go1)
    Grng = Gh - Gl
    Guw = Gh - np.maximum(Go, Gc)
    Glw = np.minimum(Go, Gc) - Gl
    eng_bull = (Gc1 < Go1) & (Gc > Go) & (Gc >= Go1) & (Go <= Gc1) & (Gab > Gab1)
    eng_bear = (Gc1 > Go1) & (Gc < Go) & (Gc <= Go1) & (Go >= Gc1) & (Gab > Gab1)
    hammer = (Grng >= 0.6 * Ga) & (Glw >= 2 * Gab) & (Glw >= 0.55 * Grng) & (Guw <= 0.2 * Grng)
    star = (Grng >= 0.6 * Ga) & (Guw >= 2 * Gab) & (Guw >= 0.55 * Grng) & (Glw <= 0.2 * Grng)
    hh20, ll20 = _ga_roll(Gh, 20, 'max'), _ga_roll(Gl, 20, 'min')
    pos = (Gc - ll20) / (hh20 - ll20)
    # v9: روند بیشتر روی تایم‌فریم بالاتر — میانگین ۵۰، ساختار سقف/کف چرخشی تأییدشده، بازده و کارایی ۱۰ کندل
    sma50 = _ga_roll(Gc, 50, 'mean')
    Gn = len(Gc)
    kp = GA_PIVOT_K
    jH1, jH2 = _ga_last_pivots(_ga_pivots(Gh, kp, True), kp, Gn, 2)
    jL1, jL2 = _ga_last_pivots(_ga_pivots(Gl, kp, False), kp, Gn, 2)
    at_h = lambda j: np.where(j >= 0, Gh[np.clip(j, 0, max(Gn - 1, 0))], np.nan)
    at_l = lambda j: np.where(j >= 0, Gl[np.clip(j, 0, max(Gn - 1, 0))], np.nan)
    pH1, pH2, pL1, pL2 = at_h(jH1), at_h(jH2), at_l(jL1), at_l(jL2)
    move10 = Gc - _ga_shift(Gc, 10)
    path10 = _ga_roll(np.abs(np.r_[np.nan, np.diff(Gc)]), 10, 'sum')
    G = {
        'htf_up': Gc > sma,
        'htf_rising': sma > _ga_shift(sma, 5),
        'htf_bull_candle': Gc > Go,
        'htf_bull_pattern': eng_bull | hammer,
        'htf_bear_pattern': eng_bear | star,
        'htf_range_low': pos < 0.3,
        'htf_range_high': pos > 0.7,
        'htf_above_sma50': Gc > sma50,
        'htf_sma20_above_sma50': sma > sma50,
        'htf_struct_up': (pH1 > pH2) & (pL1 > pL2),
        'htf_struct_down': (pH1 < pH2) & (pL1 < pL2),
    }
    G_num = {
        'htf_sma_dist': (Gc - sma) / Ga,
        'htf_pos': 2 * pos - 1,
        'htf_slope': (sma - _ga_shift(sma, 5)) / Ga,
        'htf_sma50_dist': (Gc - sma50) / Ga,
        'htf_ret10': move10 / Ga,
        'htf_eff10': move10 / path10,
    }
    idx = np.searchsorted(Gt, t + P - S, side='right') - 1     # آخرین کندل بالاتر بسته‌شده
    valid = idx >= GA_HTF_BARS                                  # اولین گروه (شاید ناقص) هیچ‌وقت استفاده نمی‌شود
    safe = np.clip(idx, 0, len(starts) - 1)
    num = {k: np.where(valid, np.asarray(v, dtype=float)[safe], np.nan) for k, v in G_num.items()}
    return {k: (np.asarray(v, dtype=bool)[safe] & valid) for k, v in G.items()}, valid, num


def _ga_break_levels(out, levels, d, n):
    """سطح شکست هر کندل (NaN = شکستی نبود). چند شکست هم‌جهت روی یک کندل: دورترین سطح از کلوز"""
    lv = np.full(n, np.inf if d == 1 else -np.inf)
    pick = np.fmin if d == 1 else np.fmax
    with np.errstate(invalid='ignore'):
        for name, x in levels.items():
            x = np.asarray(x, dtype=np.float64)
            lv = np.where(out[name] & np.isfinite(x), pick(lv, x), lv)
    return np.where(np.isfinite(lv), lv, np.nan)


def _ga_breakout_state(c, up_lv, dn_lv):
    """
    قانون شکست روی همه‌ی کندل‌ها (فقط گذشته). شکست = کلوز بالای سطح (صعودی) / کلوز زیر سطح (نزولی).
    شکست تا وقتی معتبر است که کندلی دوباره آن سوی سطح بسته نشده؛ شکست تازه‌ی خلاف جهت جای آن را می‌گیرد و شکست‌های
    هم‌جهتِ پشت‌سرهم دورترین سطح را نگه می‌دارند (تا یکی از آن‌ها پابرجاست شکست معتبر است). دو شکست خلاف هم روی یک
    کندل = بدون جهت.
    خروجی: (جهت شکست معتبر بعد از بسته شدن هر کندل: 1 / -1 / 0 ، سطح آن یا NaN)
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    bias = np.zeros(n, dtype=np.int8)
    level = np.full(n, np.nan)
    up_ev, dn_ev = np.isfinite(up_lv).tolist(), np.isfinite(dn_lv).tolist()
    cl, ul, dl = c.tolist(), np.asarray(up_lv).tolist(), np.asarray(dn_lv).tolist()
    b, lv = 0, 0.0
    for i in range(n):
        x = cl[i]
        if b == 1 and not x > lv:          # کلوز دوباره زیر سطح: شکست صعودی باطل شد
            b = 0
        elif b == -1 and not x < lv:
            b = 0
        u, dd = up_ev[i], dn_ev[i]
        if u and not dd:
            lv = min(lv, ul[i]) if b == 1 else ul[i]
            b = 1
        elif dd and not u:
            lv = max(lv, dl[i]) if b == -1 else dl[i]
            b = -1
        elif u and dd:
            b = 0
        if b:
            bias[i] = b
            level[i] = lv
    return bias, level


def _ga_breakout_text(brk):
    """«breakout rule: close above 4130.55 → SELL patterns skipped until a close back below it»"""
    up = brk['dir'] == 1
    return (f"breakout rule: close {'above' if up else 'below'} {brk['level']:.2f} → {'SELL' if up else 'BUY'} "
            f"patterns skipped until a close back {'below' if up else 'above'} it")


def _ga_breakout_ok(prim, d):
    """کندل‌هایی که قانون شکست اجازه‌ی معامله در جهت d را می‌دهد (خرید نه وقتی شکست نزولی معتبر است و برعکس)"""
    ok = prim.get('brk:ok_buy' if int(d) == 1 else 'brk:ok_sell')
    return True if ok is None else ok


def ga_primitives(o, h, l, c, atr, t, P, S):
    """
    همه‌ی الگوهای ماشه و فیلترهای زمینه برای همه‌ی کندل‌ها؛ کندل i = لحظه‌ی بسته شدن آن (فقط گذشته).
    o/h/l/c قیمت Bid کندل‌ها، atr همان ATR_14 ربات، t زمان باز شدن کندل (ثانیه‌ی سرور).
    خروجی: (dict نام → آرایه‌ی bool (الگو/فیلتر/اتم شکل) و 'n:<ویژگی>' → آرایه‌ی float32 ویژگی عددی (NaN = نامعلوم)،
            valid = کندل‌هایی که همه‌ی ورودی‌هایشان کامل است)
    """
    o, h, l, c, atr = (np.asarray(x, dtype=np.float64) for x in (o, h, l, c, atr))
    t = np.asarray(t, dtype=np.int64)
    n = len(c)
    ii = np.arange(n)
    out = {}
    lv_up, lv_dn = {}, {}          # سطح هر الگوی شکست: کلوز بالای lv_up (صعودی) / زیر lv_dn (نزولی) — قانون شکست
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(atr > 0, atr, np.nan)
        body = c - o
        ab = np.abs(body)
        rng = h - l
        top, bot = np.maximum(o, c), np.minimum(o, c)
        uw, lw = h - top, bot - l
        o1, h1, l1, c1 = _ga_shift(o), _ga_shift(h), _ga_shift(l), _ga_shift(c)
        o2, h2, l2, c2 = _ga_shift(o, 2), _ga_shift(h, 2), _ga_shift(l, 2), _ga_shift(c, 2)
        ab1, ab2, rng1 = _ga_shift(ab), _ga_shift(ab, 2), _ga_shift(rng)
        uw1, lw1 = _ga_shift(uw), _ga_shift(lw)
        bull, bear = c > o, c < o
        bull1, bear1 = c1 > o1, c1 < o1
        bull2, bear2 = c2 > o2, c2 < o2

        # ══ ۱) الگوهای کندلی ══
        out['engulfing_bull'] = bear1 & bull & (c >= o1) & (o <= c1) & (ab > ab1) & (ab >= 0.3 * a)
        out['engulfing_bear'] = bull1 & bear & (c <= o1) & (o >= c1) & (ab > ab1) & (ab >= 0.3 * a)
        out['hammer'] = (rng >= 0.6 * a) & (lw >= 2 * ab) & (lw >= 0.55 * rng) & (uw <= 0.2 * rng)
        out['shooting_star'] = (rng >= 0.6 * a) & (uw >= 2 * ab) & (uw >= 0.55 * rng) & (lw <= 0.2 * rng)
        harami_bull = bear1 & (ab1 >= 0.6 * a) & bull & (top <= o1) & (bot >= c1) & (ab <= 0.5 * ab1)
        harami_bear = bull1 & (ab1 >= 0.6 * a) & bear & (top <= c1) & (bot >= o1) & (ab <= 0.5 * ab1)
        out['harami_bull'], out['harami_bear'] = harami_bull, harami_bear
        mid2 = (o2 + c2) / 2
        strong = (ab >= 0.4 * a) & (ab1 >= 0.4 * a) & (ab2 >= 0.4 * a)
        out['three_soldiers'] = (bull & bull1 & bull2 & strong & (c > c1) & (c1 > c2)
                                 & (uw <= 0.3 * rng) & (uw1 <= 0.3 * rng1))
        out['three_crows'] = (bear & bear1 & bear2 & strong & (c < c1) & (c1 < c2)
                              & (lw <= 0.3 * rng) & (lw1 <= 0.3 * rng1))
        out['marubozu_bull'] = bull & (ab >= 0.8 * a) & (ab >= 0.85 * rng)
        out['marubozu_bear'] = bear & (ab >= 0.8 * a) & (ab >= 0.85 * rng)
        out['three_inside_up'] = _ga_shift_bool(harami_bull) & bull & (c > o2)
        out['three_inside_down'] = _ga_shift_bool(harami_bear) & bear & (c < o2)

        # ══ ۲) الگوهای کلاسیک روی سقف/کف‌های چرخشی تأییدشده ══
        k = GA_PIVOT_K
        iH1, iH2, iH3 = _ga_last_pivots(_ga_pivots(h, k, True), k, n)
        iL1, iL2, iL3 = _ga_last_pivots(_ga_pivots(l, k, False), k, n)

        def at(x, idx):
            return np.where(idx >= 0, x[np.clip(idx, 0, max(n - 1, 0))], np.nan)

        H1, H2, H3 = at(h, iH1), at(h, iH2), at(h, iH3)
        L1, L2, L3 = at(l, iL1), at(l, iL2), at(l, iL3)

        def recent(idx):
            return (idx >= 0) & (ii - idx <= GA_RECENT)

        def within(idx):
            return (idx >= 0) & (ii - idx <= GA_SPAN)

        def cross_up(now, prev):
            return (c > now) & (c1 <= prev)

        def cross_dn(now, prev):
            return (c < now) & (c1 >= prev)

        hmax_r, lmin_r = _ga_roll(h, GA_RECENT, 'max'), _ga_roll(l, GA_RECENT, 'min')

        # سقف دوقلو / کف دوقلو: شکست خط گردن (کف/سقف بین دو قله)
        neck_t = np.where((iL1 > iH2) & (iL1 < iH1), L1, np.where((iL2 > iH2) & (iL2 < iH1), L2, np.nan))
        tops = np.maximum(H1, H2)
        dt = ((np.abs(H1 - H2) <= 0.5 * a) & (iH1 - iH2 >= 6) & recent(iH1) & within(iH2)
              & (tops - neck_t >= 1.5 * a) & (hmax_r <= tops + 0.2 * a))
        out['double_top'] = dt & cross_dn(neck_t, neck_t)
        lv_dn['double_top'] = neck_t
        neck_b = np.where((iH1 > iL2) & (iH1 < iL1), H1, np.where((iH2 > iL2) & (iH2 < iL1), H2, np.nan))
        bots = np.minimum(L1, L2)
        db = ((np.abs(L1 - L2) <= 0.5 * a) & (iL1 - iL2 >= 6) & recent(iL1) & within(iL2)
              & (neck_b - bots >= 1.5 * a) & (lmin_r >= bots - 0.2 * a))
        out['double_bottom'] = db & cross_up(neck_b, neck_b)
        lv_up['double_bottom'] = neck_b

        # سر و شانه: سر بالاتر از دو شانه‌ی هم‌اندازه، شکست خط گردن (خط بین دو کف میانی)
        sel = iL1 > iH1                                    # کف تازه‌تر بعد از شانه‌ی راست → دو کف قبلی
        jL1, jL2 = np.where(sel, iL2, iL1), np.where(sel, iL3, iL2)
        NL1, NL2 = at(l, jL1), at(l, jL2)
        hs = ((iH3 >= 0) & (jL2 > iH3) & (jL2 < iH2) & (jL1 > iH2) & (jL1 < iH1)
              & (H2 - np.maximum(H1, H3) >= 0.3 * a) & (np.abs(H1 - H3) <= 1.0 * a)
              & recent(iH1) & within(iH3) & (hmax_r <= H2))
        neck_now, _ = _ga_line(jL2, NL2, jL1, NL1, ii)
        neck_prev, _ = _ga_line(jL2, NL2, jL1, NL1, ii - 1)
        out['head_shoulders'] = hs & (H2 - neck_now >= 1.5 * a) & cross_dn(neck_now, neck_prev)
        lv_dn['head_shoulders'] = neck_now
        sel = iH1 > iL1
        jH1, jH2 = np.where(sel, iH2, iH1), np.where(sel, iH3, iH2)
        NH1, NH2 = at(h, jH1), at(h, jH2)
        ihs = ((iL3 >= 0) & (jH2 > iL3) & (jH2 < iL2) & (jH1 > iL2) & (jH1 < iL1)
               & (np.minimum(L1, L3) - L2 >= 0.3 * a) & (np.abs(L1 - L3) <= 1.0 * a)
               & recent(iL1) & within(iL3) & (lmin_r >= L2))
        neck_now, _ = _ga_line(jH2, NH2, jH1, NH1, ii)
        neck_prev, _ = _ga_line(jH2, NH2, jH1, NH1, ii - 1)
        out['inv_head_shoulders'] = ihs & (neck_now - L2 >= 1.5 * a) & cross_up(neck_now, neck_prev)
        lv_up['inv_head_shoulders'] = neck_now

        # مثلث‌ها و گوه‌ها: خط سقف‌ها و خط کف‌ها از دو سقف/کف آخر
        four = (iH2 >= 0) & (iL2 >= 0) & within(np.minimum(iH2, iL2)) & recent(np.maximum(iH1, iL1))
        up_now, sl_h = _ga_line(iH2, H2, iH1, H1, ii)
        up_prev, _ = _ga_line(iH2, H2, iH1, H1, ii - 1)
        lo_now, sl_l = _ga_line(iL2, L2, iL1, L1, ii)
        lo_prev, _ = _ga_line(iL2, L2, iL1, L1, ii - 1)
        contracting = four & (H1 < H2 - 0.2 * a) & (L1 > L2 + 0.2 * a)
        out['triangle_break_up'] = contracting & cross_up(up_now, up_prev)
        out['triangle_break_down'] = contracting & cross_dn(lo_now, lo_prev)
        lv_up['triangle_break_up'], lv_dn['triangle_break_down'] = up_now, lo_now
        res, sup = np.maximum(H1, H2), np.minimum(L1, L2)
        out['ascending_triangle'] = (four & (np.abs(H1 - H2) <= 0.3 * a) & (L1 > L2 + 0.3 * a)
                                     & cross_up(res, res))
        out['descending_triangle'] = (four & (np.abs(L1 - L2) <= 0.3 * a) & (H1 < H2 - 0.3 * a)
                                      & cross_dn(sup, sup))
        lv_up['ascending_triangle'], lv_dn['descending_triangle'] = res, sup
        out['rising_wedge'] = (four & (H1 > H2) & (L1 > L2) & (sl_l > sl_h) & (sl_h > 0)
                               & cross_dn(lo_now, lo_prev))
        out['falling_wedge'] = (four & (H1 < H2) & (L1 < L2) & (sl_h < sl_l) & (sl_l < 0)
                                & cross_up(up_now, up_prev))
        lv_dn['rising_wedge'], lv_up['falling_wedge'] = lo_now, up_now

        # ساختار بازار: BOS (شکست در جهت روند) و CHoCH (شکست خلاف ساختار قبلی = برگشت)
        struct_up = (iH2 >= 0) & (iL2 >= 0) & (H1 > H2) & (L1 > L2)
        struct_dn = (iH2 >= 0) & (iL2 >= 0) & (H1 < H2) & (L1 < L2)
        brk_up = within(iH1) & cross_up(H1, H1)
        brk_dn = within(iL1) & cross_dn(L1, L1)
        out['bos_up'], out['bos_down'] = struct_up & brk_up, struct_dn & brk_dn
        out['choch_up'], out['choch_down'] = struct_dn & brk_up, struct_up & brk_dn
        lv_up['bos_up'] = lv_up['choch_up'] = H1
        lv_dn['bos_down'] = lv_dn['choch_down'] = L1

        # پرچم: حرکت قوی (۱۰ کندل) + استراحت فشرده (۵ کندل) + شکست در جهت حرکت
        c6, c16 = _ga_shift(c, 6), _ga_shift(c, 16)
        pole = c6 - c16
        hh5p, ll5p = _ga_shift(_ga_roll(h, 5, 'max')), _ga_shift(_ga_roll(l, 5, 'min'))
        tight = (hh5p - ll5p) <= 1.5 * a
        out['bull_flag'] = (pole >= 3 * a) & tight & (ll5p >= c16 + 0.5 * pole) & (c > hh5p)
        out['bear_flag'] = (-pole >= 3 * a) & tight & (hh5p <= c16 + 0.5 * pole) & (c < ll5p)
        lv_up['bull_flag'], lv_dn['bear_flag'] = hh5p, ll5p

        # شکست رنج فشرده‌ی ۲۰ کندلی
        hh20p, ll20p = _ga_shift(_ga_roll(h, 20, 'max')), _ga_shift(_ga_roll(l, 20, 'min'))
        box = (hh20p - ll20p) <= 3.0 * a
        out['box_break_up'] = box & (c > hh20p)
        out['box_break_down'] = box & (c < ll20p)
        lv_up['box_break_up'], lv_dn['box_break_down'] = hh20p, ll20p

        # ══ ۳) فیلترهای زمینه ══
        sma20, sma50, sma100 = _ga_roll(c, 20, 'mean'), _ga_roll(c, 50, 'mean'), _ga_roll(c, 100, 'mean')
        out['above_sma20'], out['above_sma50'], out['above_sma100'] = c > sma20, c > sma50, c > sma100
        out['sma50_rising'] = sma50 > _ga_shift(sma50, 10)
        out['sma20_above_sma50'] = sma20 > sma50
        out['structure_up'], out['structure_down'] = struct_up, struct_dn
        hh50, ll50 = _ga_roll(h, 50, 'max'), _ga_roll(l, 50, 'min')
        pos50 = (c - ll50) / (hh50 - ll50)
        out['range_low_50'], out['range_high_50'] = pos50 < 0.25, pos50 > 0.75
        mom = (c - _ga_shift(c, 10)) / a
        out['momentum_up'], out['momentum_down'] = mom > 1.0, mom < -1.0
        vr = a / _ga_roll(a, 200, 'median')
        out['vol_high'], out['vol_low'] = vr > 1.25, vr < 0.8
        out['wide_bar'] = rng >= 1.5 * a
        d = np.r_[np.nan, np.diff(c)]
        up_m = _ga_roll(np.where(d > 0, d, np.where(np.isnan(d), np.nan, 0.0)), 14, 'mean')
        dn_m = _ga_roll(np.where(d < 0, -d, np.where(np.isnan(d), np.nan, 0.0)), 14, 'mean')
        rsi = 100.0 * up_m / (up_m + dn_m)
        out['rsi_low'], out['rsi_high'] = rsi < 35, rsi > 65
        out['prev_doji'] = (ab1 <= 0.1 * rng1) & (rng1 >= 0.3 * a)
        htf, htf_valid, htf_num = _ga_htf(o, h, l, c, t, int(P), int(S))
        out.update(htf)
        hour = (t % 86400) // 3600
        for h0 in range(0, 24, 4):
            out[f'hour_{h0:02d}_{h0 + 4:02d}'] = (hour >= h0) & (hour < h0 + 4)
        # v6: PRE_BREAK_MINUTES دقیقه‌ی آخر قبل از تعطیلی روزانه/آخر هفته (ورود تازه ممنوع: گپ بازگشایی از روی حد ضرر رد
        # می‌شود). ساعت تعطیلی = ساعتِ آخرین تعطیلی‌ای که تا همین کندل دیده شده (فقط گذشته؛ در آموزش و زنده یکی است).
        if n > 1:
            gap_before = np.r_[False, (t[1:] - t[:-1]) > max(1800, 4 * int(P))]
            last_gap = np.maximum.accumulate(np.where(gap_before, ii, -1))
            tod_prev_close = np.r_[0, (t[:-1] + int(P)) % 86400]
            tod_brk = np.where(last_gap >= 0, tod_prev_close[np.clip(last_gap, 0, n - 1)], -1)
            out['pre_break'] = (tod_brk >= 0) & ((tod_brk - (t + int(P)) % 86400) % 86400 < PRE_BREAK_MINUTES * 60)
        else:
            out['pre_break'] = np.zeros(n, dtype=bool)

        # ══ ۴) الگوهای کندلی بیشتر ══
        o3, h3, l3, c3 = _ga_shift(o, 3), _ga_shift(h, 3), _ga_shift(l, 3), _ga_shift(c, 3)
        o4, h4, l4, c4 = _ga_shift(o, 4), _ga_shift(h, 4), _ga_shift(l, 4), _ga_shift(c, 4)
        bull3, bear3 = c3 > o3, c3 < o3
        c6 = _ga_shift(c, 6)
        doji = (ab <= 0.1 * rng) & (rng >= 0.3 * a)
        doji1 = (ab1 <= 0.1 * rng1) & (rng1 >= 0.3 * a)
        out['dragonfly_doji'] = (ab <= 0.1 * rng) & (rng >= 0.6 * a) & (lw >= 0.65 * rng) & (uw <= 0.1 * rng)
        out['gravestone_doji'] = (ab <= 0.1 * rng) & (rng >= 0.6 * a) & (uw >= 0.65 * rng) & (lw <= 0.1 * rng)
        small3 = ((ab1 <= 0.5 * a) & (ab2 <= 0.5 * a) & (_ga_shift(ab, 3) <= 0.5 * a)
                  & (np.fmax(np.fmax(h1, h2), h3) <= h4) & (np.fmin(np.fmin(l1, l2), l3) >= l4))
        out['rising_three'] = (c4 > o4) & (c4 - o4 >= 0.8 * a) & small3 & bull & (ab >= 0.5 * a) & (c > c4)
        out['falling_three'] = (c4 < o4) & (o4 - c4 >= 0.8 * a) & small3 & bear & (ab >= 0.5 * a) & (c < c4)
        out['morning_doji_star'] = bear2 & (ab2 >= 0.6 * a) & doji1 & bull & (c > mid2)
        out['evening_doji_star'] = bull2 & (ab2 >= 0.6 * a) & doji1 & bear & (c < mid2)
        out['pin_bar_bull'] = (lw >= 0.66 * rng) & (rng >= 0.8 * a) & (l < l1) & (c > l1)
        out['pin_bar_bear'] = (uw >= 0.66 * rng) & (rng >= 0.8 * a) & (h > h1) & (c < h1)
        out['two_bar_reversal_up'] = (bear1 & (ab1 >= 0.7 * a) & bull & (ab >= 0.7 * a)
                                      & (np.abs(ab - ab1) <= 0.3 * ab1) & (c - l >= 0.75 * rng))
        out['two_bar_reversal_down'] = (bull1 & (ab1 >= 0.7 * a) & bear & (ab >= 0.7 * a)
                                        & (np.abs(ab - ab1) <= 0.3 * ab1) & (h - c >= 0.75 * rng))
        inside2 = (h2 <= h3) & (l2 >= l3)
        out['hikkake_up'] = inside2 & (l1 < l2) & (c > h2)
        out['hikkake_down'] = inside2 & (h1 > h2) & (c < l2)
        lv_up['hikkake_up'], lv_dn['hikkake_down'] = h2, l2
        inside0 = (h <= h1) & (l >= l1)
        out['three_line_strike_bull'] = bear1 & bear2 & bear3 & (c1 < c2) & (c2 < c3) & bull & (c > o3)
        out['three_line_strike_bear'] = bull1 & bull2 & bull3 & (c1 > c2) & (c2 > c3) & bear & (c < o3)
        out['harami_cross_bull'] = bear1 & (ab1 >= 0.6 * a) & doji & inside0
        out['harami_cross_bear'] = bull1 & (ab1 >= 0.6 * a) & doji & inside0

        # ══ ۵) الگوهای کلاسیک و Smart Money بیشتر ══
        # سقف/کف سه‌قلو: شکست خط گردن (بالاترین سقف چرخشی بین سه کف / پایین‌ترین کف بین سه سقف)
        lows3 = np.vstack([L1, L2, L3])
        neck3b = np.fmax(np.where((iH1 > iL3) & (iH1 < iL1), H1, np.nan),
                         np.where((iH2 > iL3) & (iH2 < iL1), H2, np.nan))
        tb = ((iL3 >= 0) & within(iL3) & recent(iL1) & (np.nanmax(lows3, axis=0) - np.nanmin(lows3, axis=0) <= 0.6 * a)
              & (neck3b - np.nanmin(lows3, axis=0) >= 1.5 * a) & (lmin_r >= np.nanmin(lows3, axis=0) - 0.2 * a))
        out['triple_bottom'] = tb & cross_up(neck3b, neck3b)
        lv_up['triple_bottom'] = neck3b
        highs3 = np.vstack([H1, H2, H3])
        neck3t = np.fmin(np.where((iL1 > iH3) & (iL1 < iH1), L1, np.nan),
                         np.where((iL2 > iH3) & (iL2 < iH1), L2, np.nan))
        tt = ((iH3 >= 0) & within(iH3) & recent(iH1) & (np.nanmax(highs3, axis=0) - np.nanmin(highs3, axis=0) <= 0.6 * a)
              & (np.nanmax(highs3, axis=0) - neck3t >= 1.5 * a) & (hmax_r <= np.nanmax(highs3, axis=0) + 0.2 * a))
        out['triple_top'] = tt & cross_dn(neck3t, neck3t)
        lv_dn['triple_top'] = neck3t
        # کانال موازی: شکست خلاف جهت کانال
        par = four & (np.abs(sl_h - sl_l) <= 0.25 * np.maximum(np.abs(sl_h), np.abs(sl_l)))
        out['channel_break_up'] = par & (sl_h < 0) & (sl_l < 0) & cross_up(up_now, up_prev)
        out['channel_break_down'] = par & (sl_h > 0) & (sl_l > 0) & cross_dn(lo_now, lo_prev)
        lv_up['channel_break_up'], lv_dn['channel_break_down'] = up_now, lo_now
        # شکست و پولبک (Retest) به سطح شکسته‌شده
        lvl_h = _ga_shift(_ga_roll(h, 40, 'max'), 10)
        lvl_l = _ga_shift(_ga_roll(l, 40, 'min'), 10)
        broke_up = _ga_shift(_ga_roll(c, 10, 'max')) > lvl_h + 0.3 * a
        broke_dn = _ga_shift(_ga_roll(c, 10, 'min')) < lvl_l - 0.3 * a
        out['breakout_retest_up'] = broke_up & (l <= lvl_h + 0.3 * a) & (l >= lvl_h - 0.5 * a) & bull & (c > lvl_h)
        out['breakout_retest_down'] = broke_dn & (h >= lvl_l - 0.3 * a) & (h <= lvl_l + 0.5 * a) & bear & (c < lvl_l)
        lv_up['breakout_retest_up'], lv_dn['breakout_retest_down'] = lvl_h, lvl_l
        # Spring / Upthrust (وایکاف): شکست کاذب کف/سقف رنج ۳۰ کندلی و برگشت به داخل
        hh30p, ll30p = _ga_shift(_ga_roll(h, 30, 'max'), 3), _ga_shift(_ga_roll(l, 30, 'min'), 3)
        box30 = (hh30p - ll30p) <= 4.0 * a
        out['spring'] = box30 & (_ga_roll(l, 3, 'min') < ll30p) & (c > ll30p + 0.1 * a) & bull
        out['upthrust'] = box30 & (_ga_roll(h, 3, 'max') > hh30p) & (c < hh30p - 0.1 * a) & bear
        # شکست خط روند: خط سقف‌های نزولی به بالا / خط کف‌های صعودی به پایین
        out['trendline_break_up'] = ((iH2 >= 0) & within(iH2) & recent(iH1) & (H1 < H2 - 0.3 * a)
                                     & cross_up(up_now, up_prev))
        out['trendline_break_down'] = ((iL2 >= 0) & within(iL2) & recent(iL1) & (L1 > L2 + 0.3 * a)
                                       & cross_dn(lo_now, lo_prev))
        lv_up['trendline_break_up'], lv_dn['trendline_break_down'] = up_now, lo_now

        def last_event(ev, *vals):
            j = np.maximum.accumulate(np.where(ev, ii, -1))
            js = np.clip(j, 0, max(n - 1, 0))
            return (j,) + tuple(np.where(j >= 0, v[js], np.nan) for v in vals)

        # Order Block: آخرین کندل خلاف‌جهت قبل از یک حرکت قوی (≥2.5 ATR در ۳ کندل)؛ برگشت به آن و رد شدن
        imp_up = bear3 & (c - c3 >= 2.5 * a)
        jo, ob_lo, ob_hi = last_event(imp_up, l3, o3)
        ageo = ii - jo
        out['order_block_up'] = ((jo >= 0) & (ageo >= 3) & (ageo <= 60) & (l <= ob_hi) & (l >= ob_lo - 0.3 * a)
                                 & bull & (c >= ob_hi))
        imp_dn = bull3 & (c3 - c >= 2.5 * a)
        jd, os_lo, os_hi = last_event(imp_dn, o3, h3)
        aged = ii - jd
        out['order_block_down'] = ((jd >= 0) & (aged >= 3) & (aged <= 60) & (h >= os_lo) & (h <= os_hi + 0.3 * a)
                                   & bear & (c <= os_lo))
        # کف‌ها/سقف‌های برابر (استخر نقدینگی) و شکار آن‌ها
        out['equal_lows_sweep'] = ((iL2 >= 0) & within(iL2) & (np.abs(L1 - L2) <= 0.2 * a) & (ii - iL1 >= 3)
                                   & (l < np.fmin(L1, L2) - 0.05 * a) & (c > np.fmax(L1, L2)) & bull)
        out['equal_highs_sweep'] = ((iH2 >= 0) & within(iH2) & (np.abs(H1 - H2) <= 0.2 * a) & (ii - iH1 >= 3)
                                    & (h > np.fmax(H1, H2) + 0.05 * a) & (c < np.fmin(H1, H2)) & bear)
        # برگشت فیبوناچی ۵۰٪–۷۸.۶٪ آخرین موج و کندل تأیید
        swing_up = (iL1 >= 0) & (iL1 < iH1) & recent(iH1) & (H1 - L1 >= 3 * a)
        ret_up = (H1 - l) / (H1 - L1)
        out['fib_bounce'] = swing_up & (ret_up >= 0.5) & (ret_up <= 0.786) & (l > L1) & bull
        swing_dn = (iH1 >= 0) & (iH1 < iL1) & recent(iL1) & (H1 - L1 >= 3 * a)
        ret_dn = (h - L1) / (H1 - L1)
        out['fib_reject'] = swing_dn & (ret_dn >= 0.5) & (ret_dn <= 0.786) & (h < H1) & bear
        # الگوی ۱-۲-۳: کف، سقف، کف بالاتر بعد از روند نزولی، سپس شکست سقف (و برعکس)
        out['rev123_up'] = ((iH2 >= 0) & within(iH2) & (iH2 < iL2) & (iL2 < iH1) & (iH1 < iL1)
                            & (L1 > L2 + 0.2 * a) & (H2 > H1 + 0.5 * a) & cross_up(H1, H1))
        out['rev123_down'] = ((iL2 >= 0) & within(iL2) & (iL2 < iH2) & (iH2 < iL1) & (iL1 < iH1)
                              & (H1 < H2 - 0.2 * a) & (L2 < L1 - 0.5 * a) & cross_dn(L1, L1))
        lv_up['rev123_up'], lv_dn['rev123_down'] = H1, L1
        # شکست کانال دونچیان ۵۵ کندلی (لاک‌پشت‌ها)
        dh55, dl55 = _ga_shift(_ga_roll(h, 55, 'max')), _ga_shift(_ga_roll(l, 55, 'min'))
        out['donchian_break_up'] = (c > dh55) & (c1 <= _ga_shift(dh55))
        out['donchian_break_down'] = (c < dl55) & (c1 >= _ga_shift(dl55))
        lv_up['donchian_break_up'], lv_dn['donchian_break_down'] = dh55, dl55
        # فشردگی بولینگر و شکست
        std20 = pd.Series(c).rolling(20, min_periods=20).std(ddof=0).to_numpy(dtype=float)
        squeeze = _ga_shift(std20) <= 1.1 * _ga_shift(_ga_roll(std20, 100, 'min'))
        out['squeeze_break_up'] = squeeze & (c > sma20 + 2 * std20)
        out['squeeze_break_down'] = squeeze & (c < sma20 - 2 * std20)
        lv_up['squeeze_break_up'], lv_dn['squeeze_break_down'] = sma20 + 2 * std20, sma20 - 2 * std20

        # ══ ۶) اتم‌های شکل کندل — مواد خام الگوهایی که الگوریتم ژنتیک خودش کشف می‌کند ══
        out['_D:u'], out['_D:d'] = bull, bear
        out['_B:s'], out['_B:m'], out['_B:l'] = ab < 0.25 * a, (ab >= 0.25 * a) & (ab < 0.75 * a), ab >= 0.75 * a
        out['_U:s'], out['_U:l'] = uw <= 0.15 * rng, (uw >= 0.4 * rng) & (rng > 0)
        out['_W:s'], out['_W:l'] = lw <= 0.15 * rng, (lw >= 0.4 * rng) & (rng > 0)
        out['_R:h'], out['_R:l'] = c > h1, c < l1
        out['_R:i'] = (c <= h1) & (c >= l1)
        # v3: جای کلوز در کندل (بالا/وسط/پایین یک‌سوم) و اندازه‌ی کندل نسبت به ATR (کوچک/معمولی/بزرگ)
        clv01 = (c - l) / rng
        out['_C:t'], out['_C:m'], out['_C:b'] = clv01 >= 2 / 3, (clv01 > 1 / 3) & (clv01 < 2 / 3), clv01 <= 1 / 3
        out['_X:s'], out['_X:n'], out['_X:l'] = rng < 0.6 * a, (rng >= 0.6 * a) & (rng <= 1.4 * a), rng > 1.4 * a

        # ══ ۷) v3: شکست‌های چندروزه، سطوح روز قبل، رنج اول روز، مومنتوم و بازگشت به میانگین ══
        bpd = _ga_day_bars(P)

        def nday(k_):
            return int(min(GA_LOOKBACK_CAP, k_ * bpd))

        n_1d, n_5d = nday(1), nday(5)
        day_hl = {}
        for tag, nb in (('1d', n_1d), ('2d', nday(2)), ('5d', n_5d), ('10d', nday(10)), ('20d', nday(20))):
            hi_n, lo_n = _ga_shift(_ga_roll(h, nb, 'max')), _ga_shift(_ga_roll(l, nb, 'min'))
            day_hl[tag] = (hi_n, lo_n)
            out[f'brk_{tag}_up'] = (c > hi_n) & (c1 <= _ga_shift(hi_n))
            out[f'brk_{tag}_down'] = (c < lo_n) & (c1 >= _ga_shift(lo_n))
            lv_up[f'brk_{tag}_up'], lv_dn[f'brk_{tag}_down'] = hi_n, lo_n
        # v6: شکست ۲ روزه بعد از فشردگی (رنج ۲ روز اخیر کمتر از ۷۰٪ میانگین ۲۰ روزه‌ی همین رنج)
        hi2, lo2 = day_hl['2d']
        rng2d = hi2 - lo2
        tight2d = rng2d < 0.7 * _ga_roll(rng2d, nday(20), 'mean')
        out['squeeze2d_break_up'] = out['brk_2d_up'] & tight2d
        out['squeeze2d_break_down'] = out['brk_2d_down'] & tight2d
        lv_up['squeeze2d_break_up'], lv_dn['squeeze2d_break_down'] = hi2, lo2
        # v6: پولبک به سطح شکسته‌شده‌ی ۵ روزه و برگشت (از ۳ کندل تا ۶ ساعت بعد از شکست)
        hi5, lo5 = day_hl['5d']
        j5u, lv5u = last_event(out['brk_5d_up'], hi5)
        j5d, lv5d = last_event(out['brk_5d_down'], lo5)
        span5 = max(4, 6 * 3600 // max(1, int(P)))
        out['brk5_retest_up'] = ((j5u >= 0) & (ii - j5u >= 3) & (ii - j5u <= span5) & (l <= lv5u + 0.5 * a)
                                 & (c > lv5u) & bull & (c > c1))
        out['brk5_retest_down'] = ((j5d >= 0) & (ii - j5d >= 3) & (ii - j5d <= span5) & (h >= lv5d - 0.5 * a)
                                   & (c < lv5d) & bear & (c < c1))
        lv_up['brk5_retest_up'], lv_dn['brk5_retest_down'] = lv5u, lv5d

        # روزهای سرور: سقف/کف/باز/بسته‌ی روز قبل و باز امروز (روز اول سری شاید ناقص است → استفاده نمی‌شود)
        day = t // 86400
        new_day = np.r_[True, day[1:] != day[:-1]] if n else np.zeros(0, dtype=bool)
        gid = np.cumsum(new_day) - 1
        gstart = np.flatnonzero(new_day)
        nan_n = np.full(n, np.nan)
        pdh = pdl = pdo = pdc = dopen = nan_n
        t_in_day = np.zeros(n)
        if n:
            gend = np.r_[gstart[1:], n] - 1
            dH, dL = np.maximum.reduceat(h, gstart), np.minimum.reduceat(l, gstart)
            dO, dC = o[gstart], c[gend]
            prev_ok = gid >= 2
            pg = np.clip(gid - 1, 0, None)
            pdh, pdl = np.where(prev_ok, dH[pg], np.nan), np.where(prev_ok, dL[pg], np.nan)
            pdo, pdc = np.where(prev_ok, dO[pg], np.nan), np.where(prev_ok, dC[pg], np.nan)
            dopen = np.where(gid >= 1, dO[gid], np.nan)
            t_in_day = (t - t[gstart][gid]).astype(float)
        out['pdh_break'] = (c > pdh) & (c1 <= pdh)
        out['pdl_break'] = (c < pdl) & (c1 >= pdl)
        lv_up['pdh_break'], lv_dn['pdl_break'] = pdh, pdl
        out['pdl_reclaim'] = (l < pdl) & (c > pdl) & bull
        out['pdh_reject'] = (h > pdh) & (c < pdh) & bear

        def day_level(in_window, x, use_max):
            """سقف/کف کندل‌های یک بازه‌ی اول هر روز (فقط کندل‌های داخل بازه؛ برای کندل‌های بعد از بازه گذشته است)"""
            if not n:
                return nan_n
            arr = np.where(in_window, x, -np.inf if use_max else np.inf)
            red = (np.maximum if use_max else np.minimum).reduceat(arr, gstart)[gid]
            return np.where(np.isfinite(red) & (gid >= 1), red, np.nan)

        for name, secs in (('orb', 3600), ('early_range_break', 8 * 3600)):
            inside_w = t_in_day < secs
            rh, rl = day_level(inside_w, h, True), day_level(inside_w, l, False)
            out[f'{name}_up'] = ~inside_w & (c > rh) & (c1 <= rh)
            out[f'{name}_down'] = ~inside_w & (c < rl) & (c1 >= rl)
            lv_up[f'{name}_up'], lv_dn[f'{name}_down'] = rh, rl

        sma200 = _ga_roll(c, 200, 'mean')
        up2 = _ga_roll(np.where(d > 0, d, np.where(np.isnan(d), np.nan, 0.0)), 2, 'mean')
        dn2 = _ga_roll(np.where(d < 0, -d, np.where(np.isnan(d), np.nan, 0.0)), 2, 'mean')
        rsi2 = 100.0 * up2 / (up2 + dn2)
        k_up, k_dn = sma20 + 2 * a, sma20 - 2 * a
        out['keltner_break_up'] = (c > k_up) & (c1 <= _ga_shift(k_up))
        out['keltner_break_down'] = (c < k_dn) & (c1 >= _ga_shift(k_dn))
        lv_up['keltner_break_up'], lv_dn['keltner_break_down'] = k_up, k_dn
        calm = _ga_shift(_ga_roll(rng, 5, 'mean')) <= 0.7 * a
        out['vol_expansion_up'] = calm & (rng >= 1.8 * a) & bull & (c - l >= 0.75 * rng)
        out['vol_expansion_down'] = calm & (rng >= 1.8 * a) & bear & (h - c >= 0.75 * rng)
        z1 = _ga_shift((c - sma20) / std20)
        out['zscore_reversal_up'] = (z1 <= -2.2) & bull & (c > h1)
        out['zscore_reversal_down'] = (z1 >= 2.2) & bear & (c < l1)
        cs_ = pd.Series(c)
        macd = (cs_.ewm(span=12, adjust=False).mean() - cs_.ewm(span=26, adjust=False).mean()).to_numpy()
        # خط سیگنال مثل MACD خود MT5: میانگین «ساده»ی ۹ تایی خط MACD (با میانگین نمایی، کراس در نیمی از موارد
        # یک تا چند کندل دیرتر از چیزی که روی نمودار MT5 دیده می‌شود ثبت می‌شد)
        hist = macd - pd.Series(macd).rolling(9).mean().to_numpy()
        out['macd_cross_up'] = (hist > 0) & (_ga_shift(hist) <= 0) & (macd < 0)
        out['macd_cross_down'] = (hist < 0) & (_ga_shift(hist) >= 0) & (macd > 0)
        out['impulse_up'] = (c - c3 >= 3 * a) & bull & bull1 & (c - l >= 0.6 * rng)
        out['impulse_down'] = (c3 - c >= 3 * a) & bear & bear1 & (h - c >= 0.6 * rng)

        # فیلترهای زمینه‌ی v3
        out['above_sma200'] = c > sma200
        out['above_day_open'] = c > dopen
        out['prev_day_bull'] = pdc > pdo
        out['inside_prev_day'] = (c < pdh) & (c > pdl)
        weekday = (day + 3) % 7                       # دوشنبه = 0 (۱۹۷۰/۰۱/۰۱ پنجشنبه بود)
        out['monday'], out['friday'] = weekday == 0, weekday == 4

        # ══ v7: الگوهای کندلی بیشتر + نسخه‌ی «بعد از روند» الگوهای برگشتی (تعریف کتابی: برگشت فقط بعد از روند معنی دارد) ══
        # طلا ۲۴ساعته است و Open هر کندل تقریباً همان Close قبلی است؛ الگوهایی که گپ لازم دارند (کیکر، بچه‌ی رهاشده، تاسوکی،
        # خطوط جداشونده) تقریباً هیچ‌وقت رخ نمی‌دهند و اضافه نشدند، و بقیه نسخه‌ی بدون گپ دارند.
        def trend_before(k_, d_):
            """
            روند قبل از الگوی k_ کندلی (کندل‌های i−k_+1…i): d_=-1 نزولی = Close کندل قبل از الگو دست‌کم ۱ ATR زیر Close ده
            کندل قبل‌ترش و کف الگو کف ۲۰ کندل قبل از الگو را شکسته (یا برابر شده)؛ d_=1 آینه‌ی صعودی آن
            """
            ck, ck10 = _ga_shift(c, k_), _ga_shift(c, k_ + 10)
            if d_ < 0:
                return ((ck10 - ck) >= a) & (_ga_roll(l, k_, 'min') <= _ga_shift(_ga_roll(l, 20, 'min'), k_))
            return ((ck - ck10) >= a) & (_ga_roll(h, k_, 'max') >= _ga_shift(_ga_roll(h, 20, 'max'), k_))

        dn1, dn2, dn3 = trend_before(1, -1), trend_before(2, -1), trend_before(3, -1)
        up1, up2, up3 = trend_before(1, 1), trend_before(2, 1), trend_before(3, 1)
        top1, bot1 = _ga_shift(top), _ga_shift(bot)
        mid1 = (o1 + c1) / 2
        # نفوذی / ابر تیره: کندل دوم تا بالای/پایین نیمه‌ی بدنه‌ی کندل بلند قبلی برمی‌گردد ولی آن را نمی‌بلعد
        out['piercing_line'] = (bear1 & (ab1 >= 0.6 * a) & bull & (o <= c1 + 0.05 * a) & (c > mid1) & (c < o1))
        out['dark_cloud'] = (bull1 & (ab1 >= 0.6 * a) & bear & (o >= c1 - 0.05 * a) & (c < mid1) & (c > o1))
        # موچین کف/سقف: دو کندل خلاف هم با کف/سقف تقریباً برابر که کف/سقف ۱۰ کندل قبل از خودشان است
        out['tweezer_bottom'] = (bear1 & bull & (rng1 >= 0.5 * a) & (rng >= 0.5 * a) & (np.abs(l - l1) <= 0.1 * a)
                                 & (np.fmin(l, l1) <= _ga_shift(_ga_roll(l, 10, 'min'), 2)))
        out['tweezer_top'] = (bull1 & bear & (rng1 >= 0.5 * a) & (rng >= 0.5 * a) & (np.abs(h - h1) <= 0.1 * a)
                              & (np.fmax(h, h1) >= _ga_shift(_ga_roll(h, 10, 'max'), 2)))
        # چکش وارونه (شکل ستاره‌ی دنباله‌دار بعد از روند نزولی) / مرد آویزان (شکل چکش بعد از روند صعودی)
        out['inverted_hammer'] = out['shooting_star'] & dn1
        out['hanging_man'] = out['hammer'] & up1
        # ستاره‌ی صبحگاهی/شامگاهی (بدنه‌ی وسطی کوچک، نه لزوماً دوجی): کندل سوم از نیمه‌ی بدنه‌ی کندل اول رد می‌شود
        out['morning_star'] = (bear2 & (ab2 >= 0.6 * a) & (ab1 <= 0.3 * a) & (top1 <= c2 + 0.1 * a)
                               & bull & (ab >= 0.4 * a) & (c > mid2))
        out['evening_star'] = (bull2 & (ab2 >= 0.6 * a) & (ab1 <= 0.3 * a) & (bot1 >= c2 - 0.1 * a)
                               & bear & (ab >= 0.4 * a) & (c < mid2))
        # سه‌تایی بیرونی: پوشای کندل قبل و تأیید با کلوز بالاتر/پایین‌تر
        out['three_outside_up'] = _ga_shift_bool(out['engulfing_bull']) & bull & (c > c1)
        out['three_outside_down'] = _ga_shift_bool(out['engulfing_bear']) & bear & (c < c1)
        # کمربند: کندل بلندی که از کف خودش باز (صعودی) / از سقف خودش باز (نزولی) شده، بعد از روند مخالف
        out['belt_hold_bull'] = bull & (ab >= 0.6 * a) & (lw <= 0.05 * rng) & (uw <= 0.25 * rng) & dn1
        out['belt_hold_bear'] = bear & (ab >= 0.6 * a) & (uw <= 0.05 * rng) & (lw <= 0.25 * rng) & up1
        # شکست اینساید بار: کندل قبلی داخل کندل قبل‌ترش، کلوز بالای سقف/زیر کف کندل مادر
        inside1 = (h1 <= h2) & (l1 >= l2)
        out['inside_break_up'] = inside1 & (c > h2)
        out['inside_break_down'] = inside1 & (c < l2)
        lv_up['inside_break_up'], lv_dn['inside_break_down'] = h2, l2
        # برگشت کلیدی (اوت‌ساید بار): کف/سقف کندل قبل شکار و کلوز آن سوی سقف/کف آن
        out['outside_reversal_up'] = (l < l1) & (h > h1) & bull & (c > h1)
        out['outside_reversal_down'] = (h > h1) & (l < l1) & bear & (c < l1)
        # کبوتر خانگی / شاهین نزولی: کندل کوچک هم‌رنگ داخل بدنه‌ی کندل بلند قبلی، بعد از روند
        out['homing_pigeon'] = (bear1 & (ab1 >= 0.6 * a) & bear & (top <= o1) & (bot >= c1) & (ab <= 0.6 * ab1) & dn2)
        out['descending_hawk'] = (bull1 & (ab1 >= 0.6 * a) & bull & (top <= c1) & (bot >= o1) & (ab <= 0.6 * ab1) & up2)
        # کف/سقف هم‌تراز: دو کندل هم‌رنگ پشت‌سرهم با کلوز تقریباً برابر، بعد از روند
        out['matching_low'] = bear1 & bear & (ab1 >= 0.4 * a) & (np.abs(c - c1) <= 0.05 * a) & dn2
        out['matching_high'] = bull1 & bull & (ab1 >= 0.4 * a) & (np.abs(c - c1) <= 0.05 * a) & up2
        # همان شکل کندل بعد از روند مخالف معنای برعکس دارد (چکش بعد از روند صعودی = مرد آویزان، ستاره‌ی دنباله‌دار
        # بعد از روند نزولی = چکش وارونه)؛ تا یک کندل هم‌زمان سیگنال خرید و فروش ندهد، از الگوی مخالف حذف می‌شود
        for nm in ('hammer', 'dragonfly_doji', 'pin_bar_bull'):
            out[nm] = out[nm] & ~out['hanging_man']
        for nm in ('shooting_star', 'gravestone_doji', 'pin_bar_bear'):
            out[nm] = out[nm] & ~out['inverted_hammer']
        out['matching_low'] = out['matching_low'] & ~out['three_crows']
        out['matching_high'] = out['matching_high'] & ~out['three_soldiers']
        # نسخه‌ی «بعد از روند» الگوهای برگشتی (_t): همان الگو، فقط وقتی قبلش روند مخالف بوده و کف/سقف تازه ساخته شده
        for bull_nm, bear_nm, k_ in GA_TREND_VARIANTS:
            out[bull_nm + '_t'] = out[bull_nm] & (dn1 if k_ == 1 else dn2 if k_ == 2 else dn3)
            out[bear_nm + '_t'] = out[bear_nm] & (up1 if k_ == 1 else up2 if k_ == 2 else up3)

        # ══ ۸) ویژگی‌های عددی برای شرط‌های «ویژگی > آستانه» (آستانه‌ها را الگوریتم ژنتیک یاد می‌گیرد) ══
        def rpos(nb):
            hh_, ll_ = _ga_roll(h, nb, 'max'), _ga_roll(l, nb, 'min')
            return 2 * (c - ll_) / (hh_ - ll_) - 1

        sgn = np.sign(np.nan_to_num(d))
        run_id = np.cumsum(np.r_[True, sgn[1:] != sgn[:-1]]) if n else np.zeros(0, dtype=np.int64)
        run_len = (pd.Series(np.ones(n)).groupby(run_id).cumsum().to_numpy() if n else np.zeros(0))
        tr_ = np.fmax(h - l, np.fmax(np.abs(h - c1), np.abs(l - c1)))
        up_mv, dn_mv = h - h1, l1 - l
        pdm = np.where((up_mv > dn_mv) & (up_mv > 0), up_mv, 0.0)
        mdm = np.where((dn_mv > up_mv) & (dn_mv > 0), dn_mv, 0.0)
        tr14 = _ga_roll(tr_, 14, 'mean')
        pdi, mdi = 100 * _ga_roll(pdm, 14, 'mean') / tr14, 100 * _ga_roll(mdm, 14, 'mean') / tr14
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
        day_hi = pd.Series(h).groupby(gid).cummax().to_numpy() if n else nan_n
        day_lo = pd.Series(l).groupby(gid).cummin().to_numpy() if n else nan_n
        num = {
            'rsi14': rsi - 50, 'rsi2': rsi2 - 50,
            'ret5': (c - _ga_shift(c, 5)) / a, 'ret20': (c - _ga_shift(c, 20)) / a, 'ret60': (c - _ga_shift(c, 60)) / a,
            'sma20_dist': (c - sma20) / a, 'sma50_dist': (c - sma50) / a, 'sma200_dist': (c - sma200) / a,
            'sma20_slope': (sma20 - _ga_shift(sma20, 5)) / a, 'sma50_slope': (sma50 - _ga_shift(sma50, 10)) / a,
            'pos20': rpos(20), 'pos100': rpos(100), 'pos_1d': rpos(n_1d), 'pos_5d': rpos(n_5d),
            'eff20': (c - _ga_shift(c, 20)) / _ga_roll(np.abs(d), 20, 'sum'),
            'body': body / a, 'clv': 2 * clv01 - 1, 'streak': np.clip(sgn * run_len, -8, 8),
            'pdh_dist': (c - pdh) / a, 'pdl_dist': (c - pdl) / a, 'dopen_dist': (c - dopen) / a,
            'atr_ratio': vr, 'range': rng / a,
            'contraction': _ga_roll(rng, 5, 'mean') / _ga_roll(rng, 50, 'mean'),
            'bb_width': std20 / a, 'adx': _ga_roll(dx, 14, 'mean'),
            'day_range': np.where(gid >= 1, (day_hi - day_lo) / a, np.nan),
            **htf_num,
        }

    out = {k: np.asarray(v, dtype=bool) for k, v in out.items()}
    for k, v in num.items():
        x = np.asarray(v, dtype=np.float64)
        out['n:' + k] = np.where(np.isfinite(x), x, np.nan).astype(np.float32)
    # قانون شکست: جهت شکستِ هنوز معتبر روی هر کندل و سطح آن؛ قانون‌ها خلاف این جهت معامله نمی‌کنند (_ga_breakout_ok)
    bias, level = _ga_breakout_state(c, _ga_break_levels(out, lv_up, 1, n), _ga_break_levels(out, lv_dn, -1, n))
    out['brk:dir'], out['brk:level'] = bias, level
    out['brk:ok_buy'], out['brk:ok_sell'] = bias != -1, bias != 1
    valid = (ii >= ga_warmup(P, S)) & np.isfinite(a) & htf_valid
    if int(S) <= int(P):
        valid = (ii >= ga_warmup(P, S)) & np.isfinite(a)
    return out, valid


def _ga_frame_arrays(df):
    """کندل‌های ربات (DataFrame با Open/High/Low/Close و ATR_14) → آرایه‌های ورودی ga_primitives"""
    o = df['Open'].to_numpy(dtype=np.float64)
    h = df['High'].to_numpy(dtype=np.float64)
    l = df['Low'].to_numpy(dtype=np.float64)
    c = df['Close'].to_numpy(dtype=np.float64)
    if 'ATR_14' in df.columns:
        atr = df['ATR_14'].to_numpy(dtype=np.float64)
    else:
        atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14).to_numpy(dtype=float)
    return o, h, l, c, atr, _bar_epochs(df)


# ─────────────────────────── شکل‌های کندلی کشف‌شده (الگوهای خود ربات) ───────────────────────────
# شکل = ۱ تا ۴ کندل پشت‌سرهم، هر کندل با ۷ ویژگی (یا «هر چه باشد» = *):
#   D جهت: u صعودی / d نزولی       B اندازه‌ی بدنه نسبت به ATR: s کوچک / m متوسط / l بزرگ
#   U سایه‌ی بالا: s کوتاه / l بلند  W سایه‌ی پایین: s کوتاه / l بلند
#   R کلوز نسبت به کندل قبل: h بالای سقفش / l زیر کفش / i داخل رنجش
#   C جای کلوز در کندل: t یک‌سوم بالا / m وسط / b یک‌سوم پایین   X اندازه‌ی کندل نسبت به ATR: s کوچک / n معمولی / l بزرگ
# کد: «S:» + کندل‌ها از قدیمی به جدید با «|» — مثلاً S:dl*****|u*sli** = کندل نزولی بزرگ، سپس کندل صعودی با سایه‌ی بالای
# کوتاه و سایه‌ی پایین بلند که داخل رنج قبلی بسته شده. شکل‌های موفق در کتابخانه اسم می‌گیرند («Learned #7»).
# (شکل‌های نسخه‌ی قبل با ۵ ویژگی برای هر کندل هنوز معتبرند.)
GA_SHAPE_CATS = 'DBUWRCX'
GA_SHAPE_VALUES = {'D': 'ud', 'B': 'sml', 'U': 'sl', 'W': 'sl', 'R': 'hli', 'C': 'tmb', 'X': 'snl'}
GA_SHAPE_MAX_BARS = 4
GA_SHAPE_NAMES = {}       # کد شکل → اسم (از کتابخانه یا فایل مدل)


def _ga_is_shape(trig):
    return isinstance(trig, str) and trig.startswith('S:')


def _ga_shape_bars(code):
    return code[2:].split('|')


def _ga_valid_shape(code):
    if not _ga_is_shape(code):
        return False
    bars = _ga_shape_bars(code)
    if not 1 <= len(bars) <= GA_SHAPE_MAX_BARS:
        return False
    if len({len(b) for b in bars}) != 1 or len(bars[0]) not in (5, len(GA_SHAPE_CATS)):
        return False
    spec = 0
    for bar in bars:
        for cat, ch in zip(GA_SHAPE_CATS, bar):
            if ch != '*' and ch not in GA_SHAPE_VALUES[cat]:
                return False
            spec += ch != '*'
    return spec >= 2


def _ga_shape_mask(prim, code):
    bars = _ga_shape_bars(code)
    L = len(bars)
    m = None
    for j, bar in enumerate(bars):
        k = L - 1 - j
        for cat, ch in zip(GA_SHAPE_CATS, bar):
            if ch == '*':
                continue
            atom = prim[f'_{cat}:{ch}']
            atom = _ga_shift_bool(atom, k) if k else atom
            m = atom.copy() if m is None else (m & atom)
    return m


def _ga_mirror_shape(code):
    """آینه‌ی شکل برای فروش: جهت، سایه‌ی بالا↔پایین، کلوز بالای سقف↔زیر کف، جای کلوز بالا↔پایین (طول کد حفظ می‌شود)"""
    swap = {'D': {'u': 'd', 'd': 'u'}, 'R': {'h': 'l', 'l': 'h'}, 'C': {'t': 'b', 'b': 't'}}
    out = []
    for bar in _ga_shape_bars(code):
        ch = list(bar)
        ch[2], ch[3] = ch[3], ch[2]
        out.append("".join(swap.get(cat, {}).get(x, x) for cat, x in zip(GA_SHAPE_CATS, ch)))
    return 'S:' + '|'.join(out)


def _ga_shape_text(code):
    txt = {'D': {'u': '▲', 'd': '▼'}, 'B': {'s': 'small', 'm': 'mid', 'l': 'big'},
           'U': {'s': 'uw-', 'l': 'uw+'}, 'W': {'s': 'lw-', 'l': 'lw+'}, 'R': {'h': '>H', 'l': '<L', 'i': 'in'},
           'C': {'t': 'close@top', 'm': 'close@mid', 'b': 'close@low'},
           'X': {'s': 'tiny', 'n': 'normal', 'l': 'wide'}}
    bars = []
    for bar in _ga_shape_bars(code):
        parts = [txt[cat][ch] for cat, ch in zip(GA_SHAPE_CATS, bar) if ch != '*']
        bars.append(" ".join(parts) or "any")
    return "[" + " → ".join(bars) + "]"


def _ga_trig_mask(prim, trig):
    return _ga_shape_mask(prim, trig) if _ga_is_shape(trig) else prim[trig]


def _ga_mirror_trig(trig):
    return _ga_mirror_shape(trig) if _ga_is_shape(trig) else GA_MIRROR[trig]


def _ga_valid_trig(trig):
    return _ga_valid_shape(trig) if _ga_is_shape(trig) else trig in GA_MIRROR


def _ga_trig_dir(trig):
    """جهت الگو برای نمایش: 1 صعودی، -1 نزولی (شکل کشف‌شده: نسخه‌ی اصلی صعودی، «mirror» آینه‌ی نزولی)"""
    if _ga_is_shape(trig):
        return -1 if str(GA_SHAPE_NAMES.get(trig, '')).endswith('mirror') else 1
    return 1 if trig in GA_BULL else -1


def _ga_trig_name(trig):
    if _ga_is_shape(trig):
        name = GA_SHAPE_NAMES.get(trig)
        return f"{name} {_ga_shape_text(trig)}" if name else f"Shape {_ga_shape_text(trig)}"
    return GA_NAMES.get(trig, trig)


def _ga_round(x):
    """آستانه با ۴ رقم معنی‌دار (کلید پایدار برای حافظه‌ی قانون‌ها و کتابخانه)"""
    return float(f"{float(x):.4g}")


# ─────────────────────────── شرط‌های عددی ───────────────────────────
def _ga_cond_mask(prim, cond):
    """شرط عددی (ویژگی، '>' یا '<'، آستانه) روی همه‌ی کندل‌ها؛ مقدار نامعلوم (NaN) = برقرار نیست"""
    f, op, thr = cond
    x = prim['n:' + f]
    with np.errstate(invalid='ignore'):
        return (x > thr) if op == '>' else (x < thr)


def _ga_mirror_cond(cond):
    """نسخه‌ی فروش یک شرط خرید: ویژگی آینه؛ اگر علامت برعکس می‌شود، جهت مقایسه و علامت آستانه هم برعکس"""
    f, op, thr = cond
    mf, neg = GA_NUM_FEATURES[f]
    if neg:
        return (mf, '<' if op == '>' else '>', _ga_round(-thr))
    return (mf, op, thr)


def _ga_valid_cond(cond):
    try:
        f, op, thr = cond
        return f in GA_NUM_FEATURES and op in ('>', '<') and bool(np.isfinite(float(thr)))
    except (TypeError, ValueError):
        return False


def _ga_norm_conds(conds):
    return tuple(sorted((str(f), str(op), _ga_round(thr)) for f, op, thr in (conds or ())))


def _ga_cond_text(cond):
    f, op, thr = cond
    return f"{GA_NUM_NAMES.get(f, f)} {op} {thr:.3g}"


# ─────────────────────────── آمار هر الگو (حد سود و ضرر از گذشته‌ی خود الگو) ───────────────────────────
def pattern_sl_units():
    """حد ضرر ثابت همه‌ی الگوها به واحد بازده‌ی مدل: ضریب ATR (حالت atr) یا دلار قیمت (حالت fixed)"""
    return float(FIXED_SL_PRICE) if LABEL_MODE == "fixed" else float(PATTERN_SL_ATR)


def pattern_sl_choices():
    """حد ضررهای مجاز هر الگو به واحد بازده‌ی مدل (ATR یا دلار قیمت در حالت fixed)"""
    base = pattern_sl_units()
    return tuple(round(base * k, 6) for k in PATTERN_SL_CHOICES)


def pattern_fixed_exits():
    """v7: همه‌ی الگوها با حد سود/ضرر ثابت؟ (False = حد سود هر الگو از گذشته‌اش، روش v6)"""
    return PATTERN_EXIT_MODE == "fixed"


def pattern_tp_units():
    """حد سود ثابت الگوها به واحد بازده‌ی مدل: ضریب ATR (حالت atr) یا دلار قیمت (حالت fixed)"""
    return float(FIXED_TP_PRICE) if LABEL_MODE == "fixed" else float(PATTERN_TP_ATR)


def pattern_qualifies(st, min_avg, min_n=None):
    """
    الگو معامله می‌شود؟ معاملات گذشته‌ی کافی (min_n، پیش‌فرض PATTERN_MIN_TRADES)، میانگین سود مثبت بعد از همه‌ی هزینه‌ها
    (همیشه)، و — حالت runs: میانگین سود واقعی
    هر معامله بعد از همه‌ی هزینه‌ها ≥ min_avg؛ حالت fixed: نرخ برد ≥ PATTERN_MIN_WIN_RATE (یا اگر
    PATTERN_SELECT_BY_PROFIT: میانگین ≥ min_avg)
    """
    min_n = PATTERN_MIN_TRADES if min_n is None else int(min_n)
    if not (bool(st) and st['n'] >= min_n):
        return False
    if not st['avg'] > 0:
        return False                                          # بعد از هزینه‌ها ضررده یا سربه‌سر: هیچ‌وقت معامله نمی‌شود
    if not pattern_fixed_exits():
        return st['avg'] >= min_avg                          # runs: نرخ برد ~۲۰٪ است و معیار نیست
    return (float(st.get('win', 0.0)) >= PATTERN_MIN_WIN_RATE
            or (PATTERN_SELECT_BY_PROFIT and st['avg'] >= min_avg))


def pattern_exit_text():
    """«fixed exits: take-profit 2×ATR, stop-loss 1.5×ATR, no time limit»"""
    u = '' if LABEL_MODE == 'fixed' else '×ATR'
    sl = ' / '.join(f'{x:g}' for x in pattern_sl_choices())
    if pattern_fixed_exits():
        return f"fixed exits: take-profit {pattern_tp_units():g}{u}, stop-loss {sl}{u}, no time limit"
    return f"no time limit: stop-loss {sl}{u}, take-profit = {pattern_tp_how()}"


def pattern_tp_how():
    """متن روش حد سود الگوها (حالت runs)"""
    hit = f", reached before the stop by ≥{PATTERN_TP_MIN_HIT:.0%} of its past signals" if PATTERN_TP_MIN_HIT > 0 else ""
    return {'best': f"the most profitable take-profit in its own past (≥{PATTERN_TP_MIN_RR:g}× the stop{hit})",
            'median': "the move half of its past signals reached before the stop",
            }.get(PATTERN_TP_METHOD, "average highest price reached before the stop in its past")


def _pattern_pick_tp(mfe_u, loss_u, sl_u=None):
    """
    حد سود یک الگو (واحد ATR/دلار) از وقوع‌های گذشته‌اش که بدون حد سود تا خوردن حد ضرر دنبال شده‌اند:
    mfe_u = بیشترین حرکت به نفع معامله قبل از کندل حد ضرر، loss_u = نتیجه‌ی همان معامله با حد ضرر (با گپ).
    با حد سود x هر وقوع یا x می‌برد (mfe ≥ x) یا همان loss را می‌دهد — پس میانگین سود هر x دقیقاً از همین دو آرایه حساب
    می‌شود و شبیه‌سازی دوباره لازم نیست. هزینه‌ها برای همه‌ی x یکسان‌اند و در انتخاب اثری ندارند.
    """
    mfe_u = np.asarray(mfe_u, dtype=np.float64)
    loss_u = np.asarray(loss_u, dtype=np.float64)
    ok = np.isfinite(mfe_u) & np.isfinite(loss_u)
    mfe_u, loss_u = mfe_u[ok], loss_u[ok]
    if not len(mfe_u):
        return 1e-4
    if PATTERN_TP_METHOD == 'mean':
        return max(float(np.mean(mfe_u)), 1e-4)
    if PATTERN_TP_METHOD == 'median':
        return max(float(np.median(mfe_u)), 1e-4)
    floor = max(1e-4, float(PATTERN_TP_MIN_RR) * float(sl_u or 0.0))
    cands = np.unique(np.r_[np.quantile(mfe_u, PATTERN_TP_QUANTILES), np.mean(mfe_u), floor])
    cands = cands[cands >= floor]
    if PATTERN_TP_MIN_HIT > 0 and len(cands):
        hit = (mfe_u[None, :] >= cands[:, None]).mean(axis=1)
        reach = cands[hit >= float(PATTERN_TP_MIN_HIT) - 1e-12]
        cands = reach if len(reach) else cands[:1]           # هیچ‌کدام: نزدیک‌ترین (کف)؛ الگو احتمالاً سودده نمی‌شود
    if not len(cands):
        return floor
    gain = np.where(mfe_u[None, :] >= cands[:, None], cands[:, None], loss_u[None, :]).mean(axis=1)
    return float(cands[int(np.argmax(gain))])                # برابر: نزدیک‌ترین حد سود


def pattern_selection_text():
    """runs: «positive average after all costs, ≥ 20 past trades»؛ fixed: «win rate ≥ 55% or …, ≥ 20 past trades»"""
    if PATTERN_MIN_PROFIT_USD > 0:
        profit = f"average ≥ ${PATTERN_MIN_PROFIT_USD:g}/trade after all costs (lot {LOT:g})"
    else:
        profit = "positive average per trade after all costs"
    if not pattern_fixed_exits():
        return f"{profit}, ≥ {PATTERN_MIN_TRADES} past trades"
    profit = f" or {profit}" if PATTERN_SELECT_BY_PROFIT else ""
    return f"win rate ≥ {PATTERN_MIN_WIN_RATE:.0%}{profit}, ≥ {PATTERN_MIN_TRADES} past trades"


def pattern_min_avg_units(atr_now):
    """
    حداقل میانگین سود هر معامله (PATTERN_MIN_PROFIT_USD دلار حساب با LOT فعلی) به واحد بازده‌ی مدل:
    ضریب ATR امروز (حالت atr) یا دلار قیمت (حالت fixed). اندازه‌ی قرارداد نامعلوم → ۱۰۰ (طلا).
    """
    size = float(_contract_size.get(SYMBOL) or 100.0)
    px = float(PATTERN_MIN_PROFIT_USD) / max(float(LOT) * size, 1e-9)
    if LABEL_MODE == "fixed":
        return px
    return px / max(float(atr_now or 0.0), 1e-9)


def ga_stats_text(st, atr_now=None):
    """«TP 10.6×ATR (≈$48), SL 2.00×ATR (≈$9.1), no time limit | past trades +0.64/trade ≈ $+15 at lot 0.05, 25% wins | 990 trades»"""
    if not st:
        return "no statistics"
    if LABEL_MODE == "fixed":
        def fmt(x):
            return _px_txt(x)
        avg_px = float(st['avg'])
    else:
        def fmt(x):
            return f"{x:.2f}×ATR" + (f" (≈{_px_txt(x * atr_now)})" if atr_now else "")
        avg_px = float(st['avg']) * atr_now if atr_now else None
    money = money_per_trade(avg_px) if avg_px is not None else None
    money_txt = f" ≈ ${money:+.1f} at lot {LOT:g}" if money is not None else ""
    tp_how = "" if pattern_fixed_exits() else {'best': " (best in its past)", 'median': " (half its past reached it)"}.get(
        PATTERN_TP_METHOD, " (avg highest price before the stop)")
    return (f"TP {fmt(st['tp'])}{tp_how}, SL {fmt(st['sl'])}, no time limit "
            f"| past trades: {st['avg']:+.3f}/trade after costs{money_txt}, {st['win']:.0%} wins | {int(st['n'])} trades")


# ─────────────────────────── قانون و ژن ───────────────────────────
def _ga_rule_mask(prim, rule, breakout=True):
    """کندل‌هایی که قانون برقرار است؛ breakout=True: با قانون شکست (خلاف جهت شکستِ معتبر معامله نمی‌شود)"""
    m = _ga_trig_mask(prim, rule['trig']).copy()
    for name, neg in rule['filters']:
        m &= ~prim[name] if neg else prim[name]
    for cond in rule.get('conds', ()):
        m &= _ga_cond_mask(prim, cond)
    if breakout:
        m &= _ga_breakout_ok(prim, rule['dir'])
    return m


def _ga_rule_text(rule, with_stats=True):
    parts = [_ga_trig_name(rule['trig'])]
    parts += [("NOT " if neg else "") + GA_NAMES.get(name, name) for name, neg in rule['filters']]
    parts += [_ga_cond_text(cd) for cd in rule.get('conds', ())]
    txt = f"{'BUY ' if rule['dir'] == 1 else 'SELL'} ← " + " + ".join(parts)
    if with_stats and rule.get('stats'):
        txt += f" ⇒ {ga_stats_text(rule['stats'])}"
    return txt


def _ga_key(rule):
    return (int(rule['dir']), rule['trig'], tuple(sorted((str(n), bool(g)) for n, g in rule['filters'])),
            _ga_norm_conds(rule.get('conds', ())))


def _ga_mirror_filters(filters):
    out = []
    for nm, neg in filters:
        mnm, flip = GA_FILTER_MIRROR[nm]
        out.append((mnm, bool(neg) != flip))
    return tuple(sorted(out))


def _ga_side(gene):
    return int(gene.get('side', 0))


def _ga_norm_gene(gene):
    """ژن (همیشه در جهت صعودی): side (1 خرید، -1 فروش آینه‌ای، 0 هر دو)، ماشه، فیلترها، شرط‌های عددی"""
    return {'side': _ga_side(gene), 'trig': gene['trig'],
            'filters': tuple(sorted((str(n), bool(g)) for n, g in gene.get('filters', ()))),
            'conds': _ga_norm_conds(gene.get('conds', ()))}


def _ga_pair(gene):
    """ژن → قانون‌ها: side=1 فقط خرید، side=-1 فقط فروش (نسخه‌ی آینه‌ای)، side=0 هر دو"""
    g = _ga_norm_gene(gene)
    out = []
    if g['side'] in (0, 1):
        out.append({'dir': 1, 'trig': g['trig'], 'filters': g['filters'], 'conds': g['conds']})
    if g['side'] in (0, -1):
        out.append({'dir': -1, 'trig': _ga_mirror_trig(g['trig']), 'filters': _ga_mirror_filters(g['filters']),
                    'conds': _ga_norm_conds(_ga_mirror_cond(cd) for cd in g['conds'])})
    return tuple(out)


def _ga_gene_ok(gene):
    """ژن در حد پیچیدگی مجاز است؟ (حداکثر GA_MAX_FILTERS شرط، که حداکثر GA_MAX_NUM تای آن عددی)"""
    g = _ga_norm_gene(gene)
    return len(g['filters']) + len(g['conds']) <= GA_MAX_FILTERS and len(g['conds']) <= GA_MAX_NUM


def _ga_gene_key(gene):
    g = _ga_norm_gene(gene)
    return (g['side'], g['trig'], g['filters'], g['conds'])


def _ga_gene_text(gene):
    rules = _ga_pair(gene)
    if len(rules) == 1:
        return _ga_rule_text(rules[0])
    g = _ga_norm_gene(gene)
    parts = [_ga_trig_name(g['trig'])]
    parts += [("NOT " if neg else "") + GA_NAMES.get(nm, nm) for nm, neg in g['filters']]
    parts += [_ga_cond_text(cd) for cd in g['conds']]
    return "BUY + mirrored SELL ← " + " + ".join(parts)


# ─────────────────────────── مدل (همه‌ی الگوها، هر کدام با حد سود و ضرر خودش) ───────────────────────────
class GAPatternModel:
    """
    مدل الگوها: فقط الگوهایی (نام‌دار یا ترکیبیِ کشف‌شده با الگوریتم ژنتیک) که در گذشته شرط انتخاب را داشته‌اند
    (pattern_qualifies: نرخ برد ≥ PATTERN_MIN_WIN_RATE یا میانگین سود ≥ PATTERN_MIN_PROFIT_USD). حد سود/ضرر هر الگو در
    آمارش است: v7 ثابت (PATTERN_TP_ATR / PATTERN_SL_ATR)، در حالت runs حد سود = میانگین بیشترین قیمتی که بعد از
    وقوع‌های گذشته‌اش، قبل از خوردن حد ضرر، دیده شده. معامله محدودیت زمان ندارد.
    اگر چند الگو روی یک کندل برقرار باشند، الگویی معامله می‌شود که در گذشته بیشترین سود هر معامله را داده (قانون‌ها به
    همین ترتیب مرتب‌اند)؛ معامله تا حد سود یا حد ضرر باز می‌ماند و سیگنال مخالف آن را نمی‌بندد.
    """
    kind = 'ga_patterns'

    def __init__(self, rules, meta=None):
        self.meta = dict(meta or {})
        self.unit = 'fixed' if self.meta.get('label_mode') == 'fixed' else 'atr'
        self.P = int(self.meta.get('primary_tf_seconds') or PRIMARY_TF_SECONDS)
        self.S = int(self.meta.get('secondary_tf_seconds') or ga_htf_seconds(self.P))
        rules_ = []
        for r in rules:
            rule = dict(r)
            rule['dir'] = int(rule['dir'])
            rule['filters'] = tuple((str(n), bool(g)) for n, g in rule.get('filters', ()))
            rule['conds'] = _ga_norm_conds(rule.get('conds', ()))
            rule['stats'] = dict(rule.get('stats') or {})
            rules_.append(rule)
        rules_.sort(key=lambda r: (-float(r['stats'].get('avg', 0.0)), -int(r['stats'].get('n', 0))))
        self.rules = rules_
        # اسم شکل‌های کشف‌شده و شکل‌های کتابخانه (برای نمایش «الگوهای دیده‌شده روی این کندل»)
        GA_SHAPE_NAMES.update(self.meta.get('shape_names', {}) or {})
        self.library_shapes = [cd for cd in (self.meta.get('library_shapes') or []) if _ga_valid_shape(cd)]

    def __len__(self):
        return len(self.rules)

    def __bool__(self):
        # مدل بدون قانون هم «مدل» است (فقط معامله‌ای باز نمی‌کند)؛ if model: نباید آن را نبودِ مدل حساب کند
        return True

    def exit_params(self, rule):
        """حد سود/ضرر (واحد ATR در حالت atr، دلار قیمت در حالت fixed)، hold=0 یعنی بدون محدودیت زمان، و «بدون بستن با سیگنال مخالف»"""
        st = rule['stats']
        return {'tp_units': float(st['tp']), 'sl_units': float(st['sl']), 'unit': self.unit,
                'hold': int(st.get('hold') or 0), 'reverse': False,
                'avg': float(st.get('avg', 0.0))}

    def signals(self, prim, valid):
        """سیگنال همه‌ی کندل‌ها (1 / -1 / 0) و اندیس قانون انتخاب‌شده (پرسودترین الگوی برقرار؛ -1 = هیچ)"""
        n = len(valid)
        pick = np.full(n, -1, dtype=np.int64)
        for k in range(len(self.rules) - 1, -1, -1):           # پراولویت‌تر آخر نوشته می‌شود
            pick[_ga_rule_mask(prim, self.rules[k])] = k
        pick = np.where(valid & ~prim['pre_break'], pick, -1)
        dirs = np.array([r['dir'] for r in self.rules] + [0], dtype=np.int8)
        return dirs[pick], pick

    def decide(self, df):
        """
        تصمیم روی آخرین کندل df (باید کندل بسته باشد). خروجی dict یا None اگر کندل کافی نیست:
        signal، p_buy/p_sell (سهم دفعات سودده‌ی الگوهای برقرار)، buy_rules/sell_rules، patterns (الگوهای دیده‌شده)،
        rule (الگویی که معامله می‌شود)، exit (حد سود/ضرر و فرصت همان الگو)، breakout (شکست معتبر: dir و level، یا None)
        و blocked_rules (الگوهای برقرار خلاف جهت شکست معتبر که طبق قانون شکست معامله نمی‌شوند)
        """
        if df is None or len(df) < ga_warmup(self.P, self.S) + 1:
            return None
        # همان کندل بسته دوباره پرسیده می‌شود (ربات و GUI): نتیجه‌ی قبلی، بدون محاسبه‌ی دوباره‌ی پنجره‌ی بلند
        key = (int(_bar_epochs(df)[-1]), float(df['Close'].iloc[-1]), len(df) >= live_history_bars())
        hit = getattr(self, '_decide_cache', None)
        if hit is not None and hit[0] == key:
            return hit[1]
        res = self._decide(df)
        self._decide_cache = (key, res)
        return res

    def _decide(self, df):
        prim, valid = ga_primitives(*_ga_frame_arrays(df), self.P, self.S)
        if not valid[-1]:
            return None
        if prim['pre_break'][-1]:
            # ساعت آخر قبل از تعطیلی روزانه: معامله‌ی تازه باز نمی‌شود (مثل آموزش)
            return {'signal': 0, 'p_buy': 0.0, 'p_sell': 0.0, 'buy_rules': [], 'sell_rules': [], 'patterns': [],
                    'rule': None, 'exit': None, 'breakout': None, 'blocked_rules': [], 'pre_break': True}
        brk = int(prim['brk:dir'][-1])
        matched = [r for r in self.rules if _ga_rule_mask(prim, r, breakout=False)[-1]]   # به ترتیب اولویت
        firing = [r for r in matched if r['dir'] != -brk]          # همان _ga_rule_mask با قانون شکست
        blocked = [r for r in matched if r['dir'] == -brk]
        buys = [r for r in firing if r['dir'] == 1]
        sells = [r for r in firing if r['dir'] == -1]
        rule = firing[0] if firing else None
        signal = int(rule['dir']) if rule else 0
        p_buy = max((float(r['stats'].get('win', 0.0)) for r in buys), default=0.0)
        p_sell = max((float(r['stats'].get('win', 0.0)) for r in sells), default=0.0)
        patterns = [nm for nm in GA_BULL + GA_BEAR if prim[nm][-1]]
        patterns += [cd for cd in self.library_shapes if cd in GA_SHAPE_NAMES and _ga_shape_mask(prim, cd)[-1]]
        return {'signal': signal, 'p_buy': p_buy, 'p_sell': p_sell, 'buy_rules': buys, 'sell_rules': sells,
                'patterns': patterns, 'rule': rule, 'exit': self.exit_params(rule) if rule else None,
                'breakout': {'dir': brk, 'level': float(prim['brk:level'][-1])} if brk else None,
                'blocked_rules': blocked}

    def to_dict(self):
        rules = []
        for r in self.rules:
            d = dict(r)
            d['filters'] = [[n, bool(g)] for n, g in r['filters']]
            d['conds'] = [[f, op, float(thr)] for f, op, thr in r['conds']]
            rules.append(d)
        return {'meta': self.meta, 'rules': rules}

    @classmethod
    def from_dict(cls, data):
        return cls(data.get('rules', []), data.get('meta', {}))


# ─────────────────────────── شبیه‌ساز: آمار الگو و معامله با حد سود/ضرر خود الگو ───────────────────────────
def _ga_greedy_rows(rows, e):
    """
    یک پوزیشن در هر لحظه: اندیس معاملاتی که گرفته می‌شوند (ورود بعدی از کندلی که معامله‌ی قبلی در آن بسته شده).
    «معامله‌ی بعدی» همه‌ی ردیف‌ها یک‌جا (برداری) پیدا می‌شود و فقط زنجیره‌ی معاملات گرفته‌شده دنبال می‌شود.
    """
    n = len(rows)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    nxt = np.searchsorted(rows, rows + e, side='left').tolist()
    taken = []
    k = 0
    while k < n:
        taken.append(k)
        k = nxt[k]
    return np.asarray(taken, dtype=np.int64)


def _ga_score(rets, min_n):
    """سود کل با کسر عدم‌قطعیت = Σr − std×√n ؛ با معامله‌ی خیلی کم، مثبت حساب نمی‌شود"""
    n = len(rets)
    if n == 0:
        return 0.0
    s = float(np.sum(rets))
    if n < max(2, min_n):
        return min(0.0, s)
    var = max(0.0, (float(np.dot(rets, rets)) - s * s / n) / (n - 1))
    return s - float(np.sqrt(var * n))


def _ga_fold_scores(rets, fold, n_folds, min_n=2):
    """_ga_score هر دوره‌ی زمانی، یک‌جا (bincount به‌جای ماسک جدا برای هر دوره)"""
    cnt = np.bincount(fold, minlength=n_folds)
    s = np.bincount(fold, weights=rets, minlength=n_folds)
    sq = np.bincount(fold, weights=rets * rets, minlength=n_folds)
    out = []
    for f in range(n_folds):
        n_ = int(cnt[f])
        if n_ == 0:
            out.append(0.0)
        elif n_ < max(2, min_n):
            out.append(min(0.0, float(s[f])))
        else:
            var = max(0.0, (float(sq[f]) - float(s[f]) ** 2 / n_) / (n_ - 1))
            out.append(float(s[f]) - float(np.sqrt(var * n_)))
    return out


def _ga_stats(rets):
    n = len(rets)
    if n == 0:
        return {'trades': 0, 'win_rate': 0.0, 'expectancy': 0.0, 'total': 0.0}
    return {'trades': int(n), 'win_rate': float(np.mean(rets > 0)), 'expectancy': float(np.mean(rets)),
            'total': float(np.sum(rets))}


def _ga_runs_kernel(o, h, l, c, s, rows, d, tp_px, sl_px):
    """
    هر ردیف مستقل از بقیه: معامله‌ی جهت d از Close کندل ردیف (خرید با Ask = Close + اسپرد، فروش با Bid) تا حد سود یا حد
    ضرر، بدون محدودیت زمان. اگر هر دو در یک کندل لمس شوند ضرر حساب می‌شود؛ گپ از روی حد ضرر با Open بدتر پر می‌شود.
    خروجی هر ردیف: سود/زیان (قیمت)، کندل تا خروج، نوع خروج (1 حد سود، -1 حد ضرر، 0 = تا پایان داده باز مانده) و بیشترین
    حرکت قیمت به نفع معامله قبل از خروج (کندل خروج حساب نمی‌شود).
    """
    m = len(rows)
    n = len(c)
    pnl = np.zeros(m)
    bars = np.zeros(m, dtype=np.int64)
    kind = np.zeros(m, dtype=np.int64)
    mfe = np.zeros(m)
    for k in range(m):
        i = rows[k]
        tp = tp_px[k]
        sl = sl_px[k]
        best = 0.0
        x = 0
        e = n - 1 - i
        p = 0.0
        if d == 1:
            entry = c[i] + s[i]
            for j in range(i + 1, n):
                if l[j] <= entry - sl:
                    p = min(o[j], entry - sl) - entry
                    x = -1
                    e = j - i
                    break
                if h[j] >= entry + tp:
                    p = tp
                    x = 1
                    e = j - i
                    break
                v = h[j] - entry
                if v > best:
                    best = v
        else:
            entry = c[i]
            for j in range(i + 1, n):
                aj = s[j]
                if h[j] + aj >= entry + sl:
                    p = entry - max(o[j] + aj, entry + sl)
                    x = -1
                    e = j - i
                    break
                if l[j] + aj <= entry - tp:
                    p = tp
                    x = 1
                    e = j - i
                    break
                v = entry - (l[j] + aj)
                if v > best:
                    best = v
        pnl[k] = p
        bars[k] = e
        kind[k] = x
        mfe[k] = best
    return pnl, bars, kind, mfe


def _ga_runs_numpy(o, h, l, c, s, rows, d, tp_px, sl_px):
    """همان _ga_runs_kernel با NumPy (وقتی numba نصب نیست): کندل‌ها تکه‌تکه جلو می‌روند تا همه‌ی معامله‌ها بسته شوند"""
    m, n = len(rows), len(c)
    pnl = np.zeros(m)
    bars = np.maximum(n - 1 - rows, 0).astype(np.int64)
    kind = np.zeros(m, dtype=np.int64)
    mfe = np.zeros(m)
    if m == 0:
        return pnl, bars, kind, mfe
    entry = c[rows] + (s[rows] if d == 1 else 0.0)
    act = np.arange(m)
    off = np.ones(m, dtype=np.int64)
    while len(act):
        start = rows[act] + off[act]
        live = start < n
        act, start = act[live], start[live]
        if not len(act):
            break
        w = int(min(512, max(16, 4_000_000 // len(act))))
        cols = np.arange(w)[None, :]
        idx = start[:, None] + cols
        inb = idx < n
        idc = np.minimum(idx, n - 1)
        en = entry[act][:, None]
        with np.errstate(invalid='ignore'):
            if d == 1:
                sl_hit = inb & (l[idc] <= en - sl_px[act][:, None])
                tp_hit = inb & (h[idc] >= en + tp_px[act][:, None])
                fav = h[idc] - en
            else:
                sp_ = s[idc]
                sl_hit = inb & (h[idc] + sp_ >= en + sl_px[act][:, None])
                tp_hit = inb & (l[idc] + sp_ <= en - tp_px[act][:, None])
                fav = en - (l[idc] + sp_)
        hit = sl_hit | tp_hit
        has = hit.any(axis=1)
        first = np.where(has, hit.argmax(axis=1), w)
        best = np.where(inb & (cols < first[:, None]), fav, 0.0).max(axis=1)
        mfe[act] = np.maximum(mfe[act], best)
        ra = np.flatnonzero(has)
        if len(ra):
            a_ = act[ra]
            j = idc[ra, first[ra]]
            is_sl = sl_hit[ra, first[ra]]
            e0 = entry[a_]
            if d == 1:
                p_sl = np.minimum(o[j], e0 - sl_px[a_]) - e0
            else:
                p_sl = e0 - np.maximum(o[j] + s[j], e0 + sl_px[a_])
            pnl[a_] = np.where(is_sl, p_sl, tp_px[a_])
            kind[a_] = np.where(is_sl, -1, 1)
            bars[a_] = j - rows[a_]
        off[act] += w
        act = act[~has]
    return pnl, bars, kind, mfe


_GA_RUNS_JIT = {'fn': None, 'tried': False}


def _ga_runs(o, h, l, c, s, rows, d, tp_px, sl_px):
    """
    هسته‌ی شبیه‌ساز بدون محدودیت زمان: کامپایل‌شده با numba اگر نصب باشد (اولین بار چند ثانیه)، وگرنه NumPy (کندتر،
    همان نتیجه). کش دیسکی numba عمداً استفاده نمی‌شود (مثل ارزیابی RL).
    """
    if not _GA_RUNS_JIT['tried']:
        _GA_RUNS_JIT['tried'] = True
        try:
            import numba
            _GA_RUNS_JIT['fn'] = numba.njit(cache=False, fastmath=False, nogil=True)(_ga_runs_kernel)
        except Exception as e:
            print(f"ℹ️ numba not available ({e}) — pattern simulation runs with NumPy (slower)")
    fn = _GA_RUNS_JIT['fn']
    if fn is not None:
        try:
            return fn(o, h, l, c, s, rows, int(d), tp_px, sl_px)
        except Exception as e:
            _GA_RUNS_JIT['fn'] = None
            print(f"ℹ️ numba simulation failed ({e}) — falling back to NumPy")
    return _ga_runs_numpy(o, h, l, c, s, rows, int(d), tp_px, sl_px)


class _GAMarket:
    """
    آمار الگو و شبیه‌سازی معامله روی تاریخچه، مثل اجرای ربات — بدون محدودیت زمان.
      معامله: ورود در Close کندل سیگنال (خرید با Ask = Close + اسپرد، فروش با Bid)، حد ضرر و حد سود روی بروکر (خروج خرید
            با Bid، فروش با Ask)، اگر هر دو در یک کندل لمس شوند ضرر، گپ از روی حد ضرر با Open بدتر پر می‌شود، هزینه‌ی
            اضافه‌ی هر معامله (لغزش ورود + کمیسیون) و سواپ هر رول‌اور (۰۰:۰۰ سرور، روز سه‌برابر، آخر هفته صفر).
      آمار الگو (rule_stats): حد ضرر ثابت (sl_units)؛ حد سود = میانگین بیشترین حرکت قیمت به نفع معامله قبل از خوردن حد
            ضرر؛ avg/win/n = نتیجه‌ی معامله با همین حدها، یک پوزیشن در هر لحظه.
    واحد بازده: ATR کندل سیگنال (حالت atr) یا دلار قیمت (حالت fixed).
    """

    def __init__(self, o, h, l, c, unit, spread, ts, P, extra_cost=0.0, swap=None, sl_units=None):
        self.o, self.h, self.l, self.c = (np.ascontiguousarray(x, dtype=np.float64) for x in (o, h, l, c))
        self.u = np.ascontiguousarray(unit, dtype=np.float64)
        self.s = np.ascontiguousarray(np.clip(np.nan_to_num(np.asarray(spread, dtype=np.float64), nan=0.0), 0.0, None))
        self.ts = np.asarray(ts, dtype=np.int64)
        self.P = int(P)
        self.n = n = len(self.c)
        self.k = 1                                            # هر معامله دست‌کم یک کندل بعد از سیگنال لازم دارد
        self.sl_units = float(pattern_sl_units() if sl_units is None else sl_units)
        # هزینه‌ی اضافه‌ی هر معامله به قیمت (عدد یا آرایه‌ی هر کندل): لغزش ورود + کمیسیون + EXTRA_COST_POINTS
        self.extra = np.array(np.broadcast_to(np.nan_to_num(np.asarray(extra_cost, dtype=np.float64), nan=0.0), (n,)))
        kk = max(1, int(round(3600.0 / max(1, self.P))))      # حرکت Close به Close یک‌ساعته (فقط برای گزارش و مقیاس)
        mv = np.full(n, np.nan)
        if n > kk:
            with np.errstate(divide='ignore', invalid='ignore'):
                mv[:n - kk] = (self.c[kk:] - self.c[:n - kk]) / self.u[:n - kk]
        self.move = {1: mv, -1: -mv}
        with np.errstate(divide='ignore', invalid='ignore'):
            self.cost = (self.s + self.extra) / self.u        # هزینه‌ی یک معامله به همان واحد
        sw = swap or {}
        self.swap_px = {1: float(sw.get('long', 0.0) or 0.0), -1: float(sw.get('short', 0.0) or 0.0)}
        self.swap_frac = bool(sw.get('frac', False))          # True: سواپ کسری از قیمت است (حالت بهره)
        # سواپِ «واحد قیمت» مال قیمت امروز است؛ برای کندل‌های قدیمی به نسبت قیمت کوچک می‌شود (همان درصد بهره)
        self.swap_ref = float(sw.get('ref_price', 0.0) or 0.0)
        self.has_swap = self.swap_px[1] != 0.0 or self.swap_px[-1] != 0.0
        triple = int(sw.get('triple_weekday', 2))             # دوشنبه = 0
        d0 = int(self.ts.min() // 86400) if n else 0
        d1 = int((self.ts.max() + self.P) // 86400) + 1 if n else 1
        wd = (np.arange(d0, d1 + 1) + 3) % 7
        w = np.where(wd < 5, np.where(wd == triple, 3, 1), 0)
        self._d0 = d0
        self._cum = np.concatenate([[0], np.cumsum(w)])       # _cum[d − d0] = رول‌اورهای قبل از شروع روز d
        self._swap_cfg = {'long': self.swap_px[1], 'short': self.swap_px[-1], 'frac': self.swap_frac,
                          'triple_weekday': triple, 'ref_price': self.swap_ref}

    def slice(self, lo, hi):
        """بخش [lo, hi) تاریخچه با همان هزینه‌ها، سواپ و حد ضرر (برای گام‌های Walk-Forward؛ اندیس‌ها از lo شروع می‌شوند)"""
        lo, hi = int(lo), int(hi)
        return _GAMarket(self.o[lo:hi], self.h[lo:hi], self.l[lo:hi], self.c[lo:hi], self.u[lo:hi], self.s[lo:hi],
                         self.ts[lo:hi], self.P, self.extra[lo:hi], self._swap_cfg if self.has_swap else None,
                         self.sl_units)

    def _nights(self, i, j):
        """تعداد رول‌اور (با وزن روز سه‌برابر) بین ورود (Close کندل i) و خروج (داخل کندل j)"""
        e_day = (self.ts[i] + self.P) // 86400
        x_day = np.maximum(self.ts[j] // 86400, e_day)
        return self._cum[x_day - self._d0] - self._cum[e_day - self._d0]

    def _run(self, rows, d, tp_u, sl_u):
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        u = self.u[rows]
        tp_px = np.ascontiguousarray(np.asarray(tp_u, dtype=np.float64) * u)      # آرایه‌ی معمولی (نه نمای فقط‌خواندنی):
        sl_px = np.ascontiguousarray(np.asarray(sl_u, dtype=np.float64) * u)      # numba برای هر نوع ورودی جدا کامپایل می‌کند
        pnl, bars, kind, mfe = _ga_runs(self.o, self.h, self.l, self.c, self.s, rows, int(d), tp_px, sl_px)
        return rows, u, pnl, bars, kind, mfe

    def best_rule_stats(self, rows, d, end=None, min_n=1):
        """
        آمار الگو با بهترین حد ضرر از pattern_sl_choices (بیشترین میانگین سود هر معامله در همین گذشته، با دست‌کم min_n
        معامله اگر ممکن باشد). حد سود همیشه همان میانگین بیشترین قیمت قبل از خوردن همان حد ضرر است.
        """
        best = None
        for sl in pattern_sl_choices():
            st = self.rule_stats(rows, d, end=end, sl_units=sl)
            if st and (best is None or (st['n'] >= min_n, st['avg']) > (best['n'] >= min_n, best['avg'])):
                best = st
        return best

    def exits(self, rows, d, tp_u, sl_u, hold=None):
        """(کندل تا خروج، معتبر) هر ردیف مستقل — برای شبیه‌سازی یک‌پوزیشنی؛ معتبر = معامله تا پایان داده بسته شده"""
        _, _, _, bars, kind, _ = self._run(rows, d, tp_u, sl_u)
        return np.maximum(bars, 1).astype(np.int64), kind != 0

    def outcome(self, rows, d, tp_u, sl_u, hold=None):
        """(بازده به واحد ATR/دلار، کندل تا خروج، معتبر) معامله‌ی جهت d با حد سود/ضرر tp_u/sl_u (به همان واحد)، هر ردیف مستقل"""
        rows, u, pnl, bars, kind, _ = self._run(rows, d, tp_u, sl_u)
        pnl = pnl - self.extra[rows]
        if self.has_swap and len(rows):
            px = self.swap_px[d]
            if self.swap_frac:
                px = px * self.c[rows]
            elif self.swap_ref > 0:
                px = px * self.c[rows] / self.swap_ref
            pnl = pnl + self._nights(rows, np.minimum(rows + bars, self.n - 1)) * px
        with np.errstate(divide='ignore', invalid='ignore'):
            r = pnl / u
        ok = (kind != 0) & np.isfinite(r)
        return np.where(ok, r, np.nan), np.maximum(bars, 1).astype(np.int64), ok

    def rule_stats(self, rows, d, end=None, sl_units=None):
        """
        آمار الگو روی وقوع‌های rows (جهت d) با حد ضرر ثابت و بدون محدودیت زمان:
          tp  = از «بیشترین حرکت قیمت به نفع معامله قبل از خوردن حد ضرر» وقوع‌های گذشته، طبق PATTERN_TP_METHOD
                (پیش‌فرض: حد سودی که بیشترین میانگین سود را داده — _pattern_pick_tp)،
          avg / win / n = نتیجه‌ی معامله با همین حد سود و حد ضرر، یک پوزیشن در هر لحظه، بعد از همه‌ی هزینه‌ها — معیار انتخاب
                          و اولویت الگو،
          cost = میانگین هزینه، bars = میانگین کندل نگه‌داری، occ = تعداد وقوع، hold = 0 (بدون محدودیت زمان).
        end: فقط نتیجه‌هایی که قبل از این کندل معلوم شده‌اند (سنجش منصفانه: آمار Train از بعدش خبر ندارد).
        """
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows):
            rows = rows[(rows + 1 < self.n) & np.isfinite(self.cost[rows]) & (self.u[rows] > 0)]
        if not len(rows):
            return None
        lim = self.n if end is None else int(end)
        sl = self.sl_units if sl_units is None else float(sl_units)
        if pattern_fixed_exits():
            tp = pattern_tp_units()                           # v7: حد سود ثابت برای همه‌ی الگوها
        else:
            _, u, pnl0, bars0, kind0, mfe = self._run(rows, d, np.inf, sl)
            known = (kind0 == -1) & (rows + bars0 < lim)
            if not known.any():
                return None
            tp = _pattern_pick_tp(mfe[known] / u[known], pnl0[known] / u[known], sl)
        r, e, ok = self.outcome(rows, d, tp, sl)
        ok &= rows + e < lim
        if not ok.any():
            return None
        rows_k, r_k, e_k = rows[ok], r[ok], e[ok]
        take = _ga_greedy_rows(rows_k, e_k)
        rr = r_k[take]
        return {'n': int(len(rr)), 'tp': tp, 'sl': float(sl), 'avg': float(np.mean(rr)), 'win': float(np.mean(rr > 0)),
                'cost': float(np.mean(self.cost[rows])), 'move': float(np.nanmean(self.move[d][rows])),
                'hold': 0, 'bars': float(np.mean(e_k[take])), 'occ': int(len(rows))}


# ─────────────────────────── الگوریتم ژنتیک (کشف الگوهای ترکیبی) ───────────────────────────
class _GAEngine:
    """
    الگوریتم ژنتیک الگوهای ترکیبی (ژن = ماشه + فیلترها + شرط‌های عددی؛ فروش = آینه‌ی خرید). هر قانون با حد سود و ضرر
    آماری خودش (حد ضرر ثابت، حد سود = میانگین بیشترین قیمت قبل از استاپ روی Train) معامله و سنجیده می‌شود: برازندگی بدون اثر
    روند روی چند تکه‌ی زمانی Train (با روند واقعی و برعکس)، به‌علاوه‌ی جست‌وجوی محلی روی بهترین‌ها.
    """

    def __init__(self, prim, tradable, mkt, tr_end, scale, min_trades, seed=GA_SEED, drift_end=None, grid=None,
                 best_sl=True):
        self.prim = prim
        # best_sl: آمار هر قانون با بهترین حد ضرر از pattern_sl_choices؛ False = حد ضرر پیش‌فرض (جست‌وجوی ژنتیک سریع‌تر؛
        # حد ضرر قانون‌های نهایی بعد از جست‌وجو انتخاب می‌شود)
        self.best_sl = bool(best_sl)
        self.tradable = tradable
        self.mkt = mkt
        self.tr_end = int(tr_end)
        self.penalty = GA_FILTER_PENALTY * float(scale)
        self.min_trades = int(min_trades)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.cache = {}           # key → (fitness, gene, trades)
        self._stats = {}
        self.drift_end = int(drift_end or tr_end)
        self.grid = dict(grid or {})
        self.num_feats = sorted(self.grid)
        n = len(tradable)
        rows = np.arange(n)
        self.ok_rows = tradable & (rows + mkt.k < n)
        self.blen = max(1, self.tr_end // GA_DRIFT_BLOCKS)       # بلوک‌های هم‌طول تا پایان Validation (Test هرگز)
        self.block = rows // self.blen
        self.nblocks = int(self.block[min(self.drift_end, n) - 1]) + 1 if n else 0
        self.fold = np.minimum((rows * GA_FOLDS) // max(self.tr_end, 1), GA_FOLDS - 1)
        # روند هر بلوک زمانی به‌ازای هر کندل (همان واحد بازده)؛ اثر روند روی یک معامله = روند × کندل‌های نگه‌داری
        step = np.full(n, np.nan)
        if n > 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                step[:-1] = (mkt.c[1:] - mkt.c[:-1]) / mkt.u[:-1]
        self.drift_bar = np.zeros(max(1, self.nblocks) + 1)
        for b in range(self.nblocks):
            m = (self.block == b) & (rows < self.drift_end) & tradable & np.isfinite(step)
            if m.sum() >= 50:
                self.drift_bar[b] = float(np.mean(step[m]))
        self._shape_cache = {}

    # ── روند ──
    def drift_at(self, rows, e):
        """اثر روند دوره روی معامله‌های rows با e کندل نگه‌داری (صفر بعد از Validation)"""
        rows = np.asarray(rows, dtype=np.int64)
        b = np.minimum(self.block[rows], len(self.drift_bar) - 1)
        return np.where(rows < self.drift_end, self.drift_bar[b] * e, 0.0)

    # ── ماسک، آمار و معاملات ──
    def trig_mask(self, trig):
        if not _ga_is_shape(trig):
            return self.prim[trig]
        m = self._shape_cache.get(trig)
        if m is None:
            m = _ga_shape_mask(self.prim, trig)
            if len(self._shape_cache) >= 256:
                self._shape_cache.pop(next(iter(self._shape_cache)))
            self._shape_cache[trig] = m
        return m

    def mask(self, rule):
        m = self.trig_mask(rule['trig']) & self.ok_rows & _ga_breakout_ok(self.prim, rule['dir'])
        for name, neg in rule['filters']:
            m &= ~self.prim[name] if neg else self.prim[name]
        for cond in rule['conds']:
            m &= _ga_cond_mask(self.prim, cond)
        return m

    def stats(self, rule, mask=None):
        """آمار قانون روی Train (حد سود/ضرر و میانگین سود) — فقط از گذشته‌ی Train، برای سنجش منصفانه‌ی Validation/Test"""
        key = _ga_key(rule)
        if key not in self._stats:
            m = self.mask(rule) if mask is None else mask
            rows = np.flatnonzero(m[:self.tr_end])
            if not len(rows):
                self._stats[key] = None
            elif self.best_sl:
                self._stats[key] = self.mkt.best_rule_stats(rows, int(rule['dir']), end=self.tr_end, min_n=self.min_trades)
            else:
                self._stats[key] = self.mkt.rule_stats(rows, int(rule['dir']), end=self.tr_end)
        return self._stats[key]

    def trades_in(self, rule, lo, hi, mask=None, src='raw', st=None):
        """
        معاملات یک قانون یک‌طرفه (یک پوزیشن در هر لحظه) روی بازه‌ی [lo, hi) با حد سود/ضرر آماری (st؛ پیش‌فرض آمار Train).
        src: 'raw' بازده واقعی، 'adj' بدون روند، 'flip' با روند برعکس. خروجی: (بازده، ردیف ورود، کندل نگه‌داری)
        """
        empty = (np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        m = self.mask(rule) if mask is None else mask
        st = self.stats(rule, m) if st is None else st
        if not st:
            return empty
        rows = np.flatnonzero(m[lo:hi]) + lo
        if len(rows) == 0:
            return empty
        d = int(rule['dir'])
        e, ok = self.mkt.exits(rows, d, st['tp'], st['sl'], st['hold'])
        if not ok.all():
            rows, e = rows[ok], e[ok]
        rows = rows[_ga_greedy_rows(rows, e)]
        r, e, ok = self.mkt.outcome(rows, d, st['tp'], st['sl'], st['hold'])
        if not ok.all():
            rows, r, e = rows[ok], r[ok], e[ok]
        if src != 'raw' and len(rows):
            r = r - d * (2.0 if src == 'flip' else 1.0) * self.drift_at(rows, e)
        return r, rows, e

    def pair_train_trades(self, gene, src='adj'):
        """(بازده، ردیف، جهت) معاملات قانون (خرید و/یا فروش آینه‌ای) روی Train؛ src مثل trades_in"""
        rets, rows, dirs = [], [], []
        for rule in _ga_pair(gene):
            r, rw, _ = self.trades_in(rule, 0, self.tr_end, src=src)
            rets.append(r)
            rows.append(rw)
            dirs.append(np.full(len(rw), rule['dir'], dtype=np.int8))
        return np.concatenate(rets), np.concatenate(rows).astype(np.int64), np.concatenate(dirs)

    def fitness(self, gene):
        key = _ga_gene_key(gene)
        hit = self.cache.get(key)
        if hit is not None:
            return hit[0]
        g = _ga_norm_gene(gene)
        raws, flips, rws = [], [], []
        for rule in _ga_pair(g):
            r, rw, e = self.trades_in(rule, 0, self.tr_end)
            raws.append(r)
            rws.append(rw)
            flips.append(r - rule['dir'] * 2.0 * self.drift_at(rw, e) if len(rw) else r)
        rets = np.concatenate(raws)
        n = len(rets)
        if n < self.min_trades:
            fit = -1e6 + n                       # معامله‌ی کم: نامعتبر، ولی بیشتر بهتر (جهت تکامل)
        else:
            # هر دوره: بدترِ (روند واقعی، روند برعکس) — قانون باید در هر دو سودده باشد، پس فقط سوار روند نیست
            flip = np.concatenate(flips)
            fold = self.fold[np.concatenate(rws).astype(np.int64)]
            if GA_FITNESS_FLIP:
                scores = [min(a, b) for a, b in zip(_ga_fold_scores(rets, fold, GA_FOLDS),
                                                    _ga_fold_scores(flip, fold, GA_FOLDS))]
            else:
                scores = _ga_fold_scores(rets, fold, GA_FOLDS)
            fit = sum(scores) - GA_NEG_FOLD_PENALTY * sum(max(0.0, -x) for x in scores)
            fit -= self.penalty * (len(g['filters']) + len(g['conds']))
        self.cache[key] = (float(fit), g, n)
        return float(fit)

    # ── ساخت و تغییر قانون (ژن همیشه در جهت صعودی) ──
    def random_shape(self):
        L = int(self.rng.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15]))
        while True:
            bars = ["".join(str(self.rng.choice(list(GA_SHAPE_VALUES[cat]))) if self.rng.random() < 0.35 else '*'
                            for cat in GA_SHAPE_CATS) for _ in range(L)]
            code = 'S:' + '|'.join(bars)
            if _ga_valid_shape(code):
                return code

    def mutate_shape(self, code):
        W = len(GA_SHAPE_CATS)
        bars = [list(b) + ['*'] * (W - len(b)) for b in _ga_shape_bars(code)]     # شکل نسخه‌ی قبل → ۷ ویژگی
        for _ in range(20):
            nb = [b[:] for b in bars]
            x = self.rng.random()
            if x < 0.15 and len(nb) < GA_SHAPE_MAX_BARS:
                nb.insert(0, ['*'] * W)
                k = int(self.rng.integers(0, W))
                nb[0][k] = str(self.rng.choice(list(GA_SHAPE_VALUES[GA_SHAPE_CATS[k]])))
            elif x < 0.25 and len(nb) > 1:
                nb.pop(0)
            else:
                j, k = int(self.rng.integers(0, len(nb))), int(self.rng.integers(0, W))
                opts = list(GA_SHAPE_VALUES[GA_SHAPE_CATS[k]]) + ['*']
                nb[j][k] = str(self.rng.choice([v for v in opts if v != nb[j][k]]))
            new = 'S:' + '|'.join("".join(b) for b in nb)
            if _ga_valid_shape(new):
                return new
        return code

    def random_trig(self):
        return self.random_shape() if self.rng.random() < GA_SHAPE_PROB else str(self.rng.choice(GA_BULL))

    def random_side(self):
        return int(self.rng.choice([1, -1, 0], p=[0.4, 0.4, 0.2]))

    def random_cond(self, exclude=()):
        feats = [f for f in self.num_feats if f not in exclude]
        if not feats:
            return None
        f = str(self.rng.choice(feats))
        g = self.grid[f]
        return (f, '>' if self.rng.random() < 0.5 else '<', float(g[int(self.rng.integers(len(g)))]))

    def nudge_cond(self, cond, steps=(-2, -1, 1, 2)):
        f, op, thr = cond
        g = self.grid.get(f)
        if g is None:
            return cond
        k = int(np.argmin(np.abs(g - thr)))
        k = int(np.clip(k + int(self.rng.choice(steps)), 0, len(g) - 1))
        return (f, op, float(g[k]))

    def random_rule(self):
        n_items = min(GA_MAX_FILTERS, int(self.rng.choice([0, 1, 2, 3, 4], p=[0.15, 0.35, 0.3, 0.15, 0.05])))
        filters, conds = {}, {}
        for _ in range(n_items):
            if self.num_feats and len(conds) < GA_MAX_NUM and self.rng.random() < 0.4:
                cd = self.random_cond(exclude=conds)
                if cd:
                    conds[cd[0]] = cd
            else:
                free = [nm for nm in GA_FILTERS if nm not in filters]
                filters[str(self.rng.choice(free))] = bool(self.rng.random() < 0.3)
        return {'side': self.random_side(), 'trig': self.random_trig(), 'filters': tuple(sorted(filters.items())),
                'conds': tuple(sorted(conds.values()))}

    def crossover(self, a, b):
        trig = a['trig'] if self.rng.random() < 0.5 else b['trig']
        items = [('f', nm, neg) for nm, neg in list(a['filters']) + list(b['filters'])]
        items += [('c', cd[0], cd) for cd in list(a.get('conds', ())) + list(b.get('conds', ()))]
        order = self.rng.permutation(len(items))
        uniq = {}
        for k in order:
            kind, key, val = items[int(k)]
            uniq.setdefault((kind, key), val)
        size = int(self.rng.integers(0, min(GA_MAX_FILTERS, len(uniq)) + 1)) if uniq else 0
        filters, conds = {}, {}
        for (kind, key), val in uniq.items():
            if len(filters) + len(conds) >= size:
                break
            if kind == 'f':
                filters[key] = val
            elif len(conds) < GA_MAX_NUM:
                conds[key] = val
        side = _ga_side(a) if self.rng.random() < 0.5 else _ga_side(b)
        return {'side': side, 'trig': trig, 'filters': tuple(sorted(filters.items())),
                'conds': tuple(sorted(conds.values()))}

    def mutate(self, gene):
        g = _ga_norm_gene(gene)
        trig, side = g['trig'], g['side']
        f = dict(g['filters'])
        cds = {cd[0]: cd for cd in g['conds']}
        for _ in range(1 if self.rng.random() < 0.7 else 2):
            x = self.rng.random()
            room = len(f) + len(cds) < GA_MAX_FILTERS
            free = [nm for nm in GA_FILTERS if nm not in f]
            if x < 0.07:
                side = int(self.rng.choice([v for v in (1, -1, 0) if v != side]))
            elif x < 0.27:
                trig = (self.mutate_shape(trig) if (_ga_is_shape(trig) and self.rng.random() < 0.7)
                        else self.random_trig())
            elif x < 0.42 and (f or (room and free)):
                if room and free:
                    f[str(self.rng.choice(free))] = bool(self.rng.random() < 0.3)
                else:
                    f.pop(str(self.rng.choice(list(f))))
            elif x < 0.52 and f:
                f.pop(str(self.rng.choice(list(f))))
            elif x < 0.60 and f:
                nm = str(self.rng.choice(list(f)))
                f[nm] = not f[nm]
            elif x < 0.75 and self.num_feats and (cds or room):
                if room and len(cds) < GA_MAX_NUM:
                    cd = self.random_cond(exclude=cds)
                    if cd:
                        cds[cd[0]] = cd
                elif cds:
                    cds.pop(str(self.rng.choice(list(cds))))
            elif x < 0.82 and cds:
                cds.pop(str(self.rng.choice(list(cds))))
            elif x < 0.97 and cds:
                k = str(self.rng.choice(list(cds)))
                cds[k] = self.nudge_cond(cds[k])
            elif cds:
                k = str(self.rng.choice(list(cds)))
                ff, op, thr = cds[k]
                cds[k] = (ff, '<' if op == '>' else '>', thr)
            else:
                trig = self.random_trig()
        return {'side': side, 'trig': trig, 'filters': tuple(sorted(f.items())), 'conds': tuple(sorted(cds.values()))}

    def tournament(self, pop, fits):
        idx = self.rng.integers(0, len(pop), GA_TOURNAMENT)
        return pop[max(idx, key=lambda i: fits[i])]

    def run(self, cancelled, on_generation, island=0, extra_seeds=(), pop_size=None, generations=None, patience=None,
            verbose=True):
        """
        تکامل یک جزیره (seed جدا؛ حافظه‌ی قانون‌های سنجیده‌شده مشترک)؛ خروجی False اگر لغو شد.
        extra_seeds: قانون‌هایی که نسل اول از آن‌ها شروع می‌کند (در Walk-Forward: قانون‌های گام قبل).
        pop_size / generations / patience: بودجه‌ی این اجرا (پیش‌فرض GA_POP / GA_GENERATIONS / GA_PATIENCE).
        """
        pop_size = int(pop_size or GA_POP)
        generations = int(generations or GA_GENERATIONS)
        patience = int(patience or GA_PATIENCE)
        self.rng = np.random.default_rng(self.seed + 7919 * island)
        seeds = [{'side': sd, 'trig': p, 'filters': (), 'conds': ()} for p in GA_BULL for sd in (1, -1)]
        seeds += [_ga_norm_gene(g) for g in extra_seeds if _ga_gene_ok(g)]
        seeds += [{'side': self.random_side(), 'trig': self.random_shape(), 'filters': (), 'conds': ()}
                  for _ in range(GA_RANDOM_SHAPE_SEEDS)]
        pop = seeds + [self.random_rule() for _ in range(max(0, pop_size - len(seeds)))]
        best, stall = -np.inf, 0
        for gen in range(generations):
            if cancelled():
                return False
            fits = [self.fitness(r) for r in pop]
            order = np.argsort(fits)[::-1]
            top = fits[order[0]]
            if top > best + 1e-9:
                best, stall = top, 0
            else:
                stall += 1
            if on_generation is not None:
                on_generation(island, gen + 1, pop[order[0]], top)
            if stall >= patience:
                if verbose:
                    print(f"   ⏹️ island {island + 1}: converged (no better rule in {patience} generations)")
                break
            nxt = [pop[i] for i in order[:GA_ELITE]]
            nxt += [self.random_rule() for _ in range(GA_IMMIGRANTS)]
            while len(nxt) < pop_size:
                p1 = self.tournament(pop, fits)
                child = (self.crossover(p1, self.tournament(pop, fits)) if self.rng.random() < GA_CROSSOVER
                         else dict(p1))
                nxt.append(self.mutate(child))
            pop = nxt
        return True

    def refine(self, gene, cancelled):
        """جست‌وجوی محلی روی یک قانون خوب: آستانه‌های همسایه و حذف شرط‌ها (تا وقتی برازندگی Train بهتر می‌شود)"""
        best = _ga_norm_gene(gene)
        best_fit = self.fitness(best)
        for _ in range(4):
            improved = False
            cands = []
            conds = list(best['conds'])
            for i, cd in enumerate(conds):
                g_ = self.grid.get(cd[0])
                if g_ is not None:
                    k = int(np.argmin(np.abs(g_ - cd[2])))
                    for dk in (-2, -1, 1, 2):
                        if 0 <= k + dk < len(g_):
                            cc = conds[:i] + [(cd[0], cd[1], float(g_[k + dk]))] + conds[i + 1:]
                            cands.append(dict(best, conds=tuple(sorted(cc))))
                cands.append(dict(best, conds=tuple(conds[:i] + conds[i + 1:])))
            fl = list(best['filters'])
            for i in range(len(fl)):
                cands.append(dict(best, filters=tuple(fl[:i] + fl[i + 1:])))
            for g in cands:
                if cancelled():
                    return best, best_fit
                f = self.fitness(g)
                if f > best_fit + 1e-9:
                    best, best_fit, improved = _ga_norm_gene(g), f, True
            if not improved:
                break
        return best, best_fit

    def hall_of_fame(self):
        """همه‌ی قانون‌های معتبر سنجیده‌شده، بهترین اول"""
        good = [(fit, gene, n) for fit, gene, n in self.cache.values() if fit > -1e5]
        return sorted(good, key=lambda x: -x[0])

    def portfolio(self, items, lo, hi, src='raw', with_rows=False):
        """
        اجرای مجموعه‌ی الگوها مثل ربات زنده روی [lo, hi): items = [(rule, mask, stats)] به ترتیب اولویت (میانگین سود)؛ روی
        هر کندل پراولویت‌ترین الگوی برقرار با حد سود/ضرر خودش معامله می‌شود؛ یک پوزیشن در هر لحظه و بدون بستن با سیگنال
        مخالف. src مثل trades_in. خروجی: (بازده معاملات، جهت معاملات) و با with_rows کندل ورود هر معامله هم
        """
        n_ = int(hi) - int(lo)
        if n_ <= 0 or not items:
            empty = (np.zeros(0), np.zeros(0, dtype=np.int8))
            return empty + (np.zeros(0, dtype=np.int64),) if with_rows else empty
        pick = np.full(n_, -1, dtype=np.int64)
        for k in range(len(items) - 1, -1, -1):                 # پراولویت‌تر آخر نوشته می‌شود
            pick[items[k][1][lo:hi]] = k
        pos = np.flatnonzero(pick >= 0)
        rows, ks = pos + lo, pick[pos]
        r = np.full(len(rows), np.nan)
        e = np.ones(len(rows), dtype=np.int64)
        dirs = np.zeros(len(rows), dtype=np.int8)
        for k, (rule, _, st) in enumerate(items):
            sel = ks == k
            if sel.any():
                d = int(rule['dir'])
                rr, ee, ok = self.mkt.outcome(rows[sel], d, st['tp'], st['sl'], st['hold'])
                if src != 'raw':
                    rr = rr - d * (2.0 if src == 'flip' else 1.0) * self.drift_at(rows[sel], ee)
                r[sel] = np.where(ok, rr, np.nan)
                e[sel] = ee
                dirs[sel] = d
        rets, out_dirs, out_rows = [], [], []
        free_at = -1
        for k in range(len(rows)):
            if not np.isfinite(r[k]) or rows[k] < free_at:
                continue                                      # پوزیشن قبلی هنوز باز است
            rets.append(float(r[k]))
            out_dirs.append(int(dirs[k]))
            out_rows.append(int(rows[k]))
            free_at = int(rows[k]) + int(e[k])
        out = (np.asarray(rets, dtype=float), np.asarray(out_dirs, dtype=np.int8))
        return out + (np.asarray(out_rows, dtype=np.int64),) if with_rows else out


def ga_walkforward_path(model_path=None):
    """نتیجه‌ی گام‌های Walk-Forward ژنتیک (کنار مدل): هر ماه فقط یک بهینه‌سازی تازه لازم است"""
    return os.path.join(os.path.dirname(model_path or MODEL_PATH) or ".", "ga_walkforward.json")


class GAWalkForward:
    """
    لایه‌ی کنترل Walk-Forward دور هسته‌ی ژنتیک (_GAEngine) — جلوگیری از بیش‌برازش و سازگاری با رژیم بازار.
      • تاریخچه به ماه‌های متوالی تقسیم می‌شود. در هر گام، هسته‌ی ژنتیک فقط روی GA_WF_TRAIN_MONTHS ماهِ قبل از ماه تست
        بهینه می‌شود (بدون هیچ اطلاعی از بعدش). قانون‌هایی که در همان پنجره به حداقل میانگین سود می‌رسند (با دست‌کم
        GA_WF_MIN_TRADES معامله و حداکثر GA_MAX_FILTERS شرط) روی GA_WF_TEST_MONTHS ماهِ بعد معامله می‌شوند؛ گام بعد یک
        ماه جلوتر است.
      • نتیجه‌ی ماه‌های تست پشت‌سرهم = عملکرد واقعی «هر ماه دوباره بهینه کن» روی داده‌ی دیده‌نشده.
      • قانون‌های گام ماه جاری (بهینه‌شده روی GA_WF_TRAIN_MONTHS ماهِ منتهی به این ماه) فقط وقتی به مدل اضافه می‌شوند که
        سابقه‌ی Walk-Forward در GA_WF_STEPS گام اخیر به حداقل میانگین سود رسیده باشد.
      • دانه‌ی هر گام فقط الگوهای نام‌دار و قانون‌های گام قبل است (نه کتابخانه‌ی آموزش‌های کل تاریخچه که از آینده خبر دارد).
      • نتیجه‌ی هر گام کنار مدل ذخیره می‌شود (ga_walkforward.json)؛ بار اول همه‌ی گام‌ها حساب می‌شوند و بعد از آن هر ماه
        فقط یک بهینه‌سازی تازه (و سنجش ماهی که تمام شده) لازم است.
    """

    def __init__(self, prim, tradable, mkt, ts, P, min_avg, cache_path=None, cancelled=None, progress=None):
        self.prim, self.tradable, self.mkt = prim, tradable, mkt
        self.ts = np.asarray(ts, dtype=np.int64)
        self.P = int(P)
        self.n = len(self.ts)
        self.min_avg = float(min_avg)
        # حداقل سابقه‌ی Walk-Forward (به همان واحد min_avg): پیش‌فرض همان حداقل الگوها
        self.gate = (self.min_avg if GA_WF_MIN_PROFIT_USD is None
                     else self.min_avg * float(GA_WF_MIN_PROFIT_USD) / max(float(PATTERN_MIN_PROFIT_USD), 1e-9))
        self.cache_path = cache_path
        self.cancelled = cancelled or (lambda: False)
        self.progress = progress
        self.month = self.ts.astype('datetime64[s]').astype('datetime64[M]').astype(np.int64)
        self.m0, self.m1 = (int(self.month[0]), int(self.month[-1])) if self.n else (0, 0)
        self.tail = int(GA_WF_TAIL_DAYS * 86400 // max(1, self.P))
        self.config = {'ga_version': GA_VERSION, 'decision_rule': DECISION_RULE, 'tf_seconds': self.P,
                       'train_months': GA_WF_TRAIN_MONTHS, 'test_months': GA_WF_TEST_MONTHS,
                       'max_filters': GA_MAX_FILTERS, 'max_num': GA_MAX_NUM, 'islands': GA_WF_ISLANDS,
                       'pop': GA_WF_POP, 'generations': GA_WF_GENERATIONS, 'top': GA_WF_TOP,
                       'min_trades': GA_WF_MIN_TRADES, 'sl_units': float(mkt.sl_units),
                       'sl_choices': list(pattern_sl_choices()), 'fitness_flip': bool(GA_FITNESS_FLIP), 'search_sl': 'default',
                       'min_profit_usd': float(PATTERN_MIN_PROFIT_USD), 'lot': float(LOT),
                       'exit_mode': PATTERN_EXIT_MODE, 'tp_units': pattern_tp_units() if pattern_fixed_exits() else None,
                       'min_win_rate': float(PATTERN_MIN_WIN_RATE), 'select_by_profit': bool(PATTERN_SELECT_BY_PROFIT),
                       'patterns': sorted(GA_BULL)}
        self.evals = 0

    @staticmethod
    def month_text(m):
        return f"{1970 + int(m) // 12}-{int(m) % 12 + 1:02d}"

    def _idx(self, m):
        """اولین کندل ماه m (یا n)"""
        return int(np.searchsorted(self.month, int(m), side='left'))

    def _load(self):
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('config') == self.config:
                return dict(data.get('folds') or {})
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return {}

    def _save(self, folds):
        if self.cache_path:
            _atomic_write_json({'config': self.config, 'updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'folds': folds}, self.cache_path)

    def _engine(self, k):
        """موتور ژنتیک روی پنجره‌ی آموزش گام k (ماه‌های [k − آموزش، k)) + ماه تست + دنباله برای بسته شدن معامله‌ها"""
        lo, hi = self._idx(k - GA_WF_TRAIN_MONTHS), self._idx(k)
        te_hi = self._idx(k + GA_WF_TEST_MONTHS)
        ext = min(self.n, te_hi + self.tail)
        prim = {key: v[lo:ext] for key, v in self.prim.items()}
        trad = np.array(self.tradable[lo:ext], dtype=bool)
        trad[te_hi - lo:] = False                             # بعد از ماه تست ورود نیست؛ فقط بسته شدن معامله‌ها
        mkt = self.mkt.slice(lo, ext)
        tr_end = hi - lo
        tr_rows = np.flatnonzero(trad[:tr_end])
        mv = np.concatenate([mkt.move[1][tr_rows], mkt.move[-1][tr_rows]]) if len(tr_rows) else np.zeros(0)
        scale = float(np.nanmedian(np.abs(mv))) if np.isfinite(mv).any() else 1.0
        eng = _GAEngine(prim, trad, mkt, tr_end, scale, GA_WF_MIN_TRADES, seed=GA_SEED + 104729 * int(k), best_sl=False,
                        drift_end=tr_end, grid=_ga_feature_grid(prim, tr_rows))
        return eng, tr_end, te_hi - lo

    def _test(self, eng, rules, te_lo, te_hi):
        """معامله‌ی قانون‌های یک گام روی ماه تست (آمار و حد سود از پنجره‌ی آموزش)، یک پوزیشن در هر لحظه"""
        items = [(r, eng.mask(r), r['stats']) for r in rules]
        rets, dirs = eng.portfolio(items, te_lo, te_hi)
        return {'r': [round(float(x), 5) for x in rets], 'd': [int(x) for x in dirs]}

    def _optimize(self, k, seeds):
        """یک گام: هسته‌ی ژنتیک روی پنجره‌ی آموزش، انتخاب قانون‌ها و (اگر ماه تست تمام شده) سنجش روی ماه تست"""
        if k - GA_WF_TRAIN_MONTHS < self.m0 or self._idx(k) - self._idx(k - GA_WF_TRAIN_MONTHS) < 1000:
            return None                                       # تاریخچه برای پنجره‌ی آموزش کامل کافی نیست
        t0 = time.time()
        eng, tr_end, te_end = self._engine(k)
        for island in range(GA_WF_ISLANDS):
            if not eng.run(self.cancelled, None, island, extra_seeds=seeds, pop_size=GA_WF_POP,
                           generations=GA_WF_GENERATIONS, patience=GA_WF_PATIENCE, verbose=False):
                return 'cancelled'
        self.evals += len(eng.cache)
        top, masks = [], []
        for fit, gene, _ in eng.hall_of_fame():               # قانون‌های متنوع (هم‌پوشانی سیگنال Train کم)
            if fit <= 0 or len(top) >= GA_WF_TOP:
                break
            m_tr = np.logical_or.reduce([eng.mask(r_) for r_ in _ga_pair(gene)])[:tr_end]
            dup = any(np.count_nonzero(m_tr & cm) / max(1, np.count_nonzero(m_tr | cm)) > GA_MAX_OVERLAP
                      for cm in masks)
            if not dup:
                top.append(_ga_norm_gene(gene))
                masks.append(m_tr)
        rules = []
        for gene in top:
            for r_ in _ga_pair(gene):
                if not (r_['filters'] or r_['conds']):
                    continue                                  # بدون شرط = خود الگوی نام‌دار (با آمار کل تاریخچه در مدل)
                rows_tr = np.flatnonzero(eng.mask(r_)[:tr_end])      # حد ضرر قانون: بهترینِ ۱.۵/۲/۳ ATR در پنجره‌ی آموزش
                st = (eng.mkt.best_rule_stats(rows_tr, int(r_['dir']), end=tr_end, min_n=GA_WF_MIN_TRADES)
                      if len(rows_tr) else None)
                if pattern_qualifies(st, self.min_avg, GA_WF_MIN_TRADES):
                    rules.append({'dir': int(r_['dir']), 'trig': r_['trig'],
                                  'filters': [[nm, bool(g)] for nm, g in r_['filters']],
                                  'conds': [[f_, op, float(thr)] for f_, op, thr in r_['conds']],
                                  'source': 'ga', 'stats': st})
        rules.sort(key=lambda r: (-r['stats']['avg'], -r['stats']['n']))
        fold = {'train': [self.month_text(k - GA_WF_TRAIN_MONTHS), self.month_text(k - 1)],
                'genes': [{'side': g['side'], 'trig': g['trig'], 'filters': [[nm, bool(x)] for nm, x in g['filters']],
                           'conds': [[f_, op, float(thr)] for f_, op, thr in g['conds']]} for g in top],
                'rules': rules, 'evals': len(eng.cache), 'seconds': round(time.time() - t0, 1),
                'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        if k + GA_WF_TEST_MONTHS <= self.m1:                 # ماه تست کامل است
            fold['oos'] = self._test(eng, rules, tr_end, te_end)
        return fold

    def run(self):
        """
        همه‌ی گام‌های لازم (فقط آن‌هایی که در فایل نیستند) و جمع‌بندی سابقه. خروجی None اگر لغو شد، وگرنه dict:
        accepted، rules (قانون‌های ماه جاری)، oos (سابقه‌ی ماه‌های تست)، months (جدول گام‌ها)، evals
        """
        folds = self._load()
        cur = self.m1
        keys = [cur - j * GA_WF_TEST_MONTHS for j in range(GA_WF_STEPS, -1, -1)]   # قدیمی → جدید؛ آخری = ماه جاری
        names = [self.month_text(k) for k in keys]
        folds = {k_: v for k_, v in folds.items() if k_ in names}
        todo = sum(1 for nm in names if nm not in folds)
        print(f"🧬 Walk-forward genetic optimization: train {GA_WF_TRAIN_MONTHS} months → test {GA_WF_TEST_MONTHS} month, "
              f"re-optimized every month, last {GA_WF_STEPS} months | ≤{GA_MAX_FILTERS} conditions per rule, "
              f"≥{GA_WF_MIN_TRADES} train trades | {todo} step(s) to optimize"
              + (" (first run: this takes a while; afterwards one step per month)" if todo > 2 else ""))
        seeds = []
        t0 = time.time()
        for i, (k, nm) in enumerate(zip(keys, names)):
            if self.cancelled():
                return None
            fold = folds.get(nm)
            if fold is None:
                fold = self._optimize(k, seeds)
                if fold == 'cancelled':
                    return None
                if fold is None:
                    continue
                folds[nm] = fold
                self._save(folds)
                oos = fold.get('oos')
                oos_txt = (f"test month: {len(oos['r'])} trades, avg {np.mean(oos['r']):+.3f}" if oos and oos['r']
                           else ("test month: no trades" if oos else "current month (no test yet)"))
                print(f"   {nm} ← train {fold['train'][0]}…{fold['train'][1]}: {len(fold['rules'])} rule(s) | {oos_txt} "
                      f"| {fold['evals']:,} combinations, {fold['seconds']:.0f}s")
            elif 'oos' not in fold and k + GA_WF_TEST_MONTHS <= cur:
                eng, tr_end, te_end = self._engine(k)          # ماهی که حالا تمام شده: فقط سنجش، بدون بهینه‌سازی دوباره
                fold['oos'] = self._test(eng, fold['rules'], tr_end, te_end)
                self._save(folds)
            seeds = fold.get('genes') or []
            if self.progress:
                self.progress(i + 1)
        hist = [(nm, folds[nm]) for k, nm in zip(keys, names) if nm in folds and folds[nm].get('oos') is not None]
        r = np.array([x for _, f in hist for x in f['oos']['r']], dtype=float)
        d = np.array([x for _, f in hist for x in f['oos']['d']], dtype=int)
        n_ = len(r)
        avg = float(r.mean()) if n_ else 0.0
        t = float(avg / (r.std(ddof=1) / np.sqrt(n_))) if n_ > 2 and r.std(ddof=1) > 0 else 0.0
        is_avg = [np.mean([ru['stats']['avg'] for ru in f['rules']]) for _, f in hist if f['rules']]
        wfe = round(avg / float(np.mean(is_avg)), 3) if is_avg and np.mean(is_avg) > 0 else None   # کارایی Walk-Forward
        pos_months = sum(1 for _, f in hist if f['oos']['r'] and sum(f['oos']['r']) > 0)
        traded_months = sum(1 for _, f in hist if f['oos']['r'])
        live = folds.get(names[-1])
        accepted = bool(n_ >= GA_WF_MIN_OOS_TRADES and avg >= self.gate and live is not None)
        print(f"📈 Walk-forward record ({len(hist)} test months, rules re-optimized every month, never seen before testing): "
              f"{n_} trades (B{int((d == 1).sum())}/S{int((d == -1).sum())}) | avg {avg:+.3f}/trade (t={t:+.1f}) | "
              f"total {r.sum():+.1f} | win {np.mean(r > 0) if n_ else 0:.0%} | positive months {pos_months}/{traded_months} "
              f"| walk-forward efficiency {'—' if wfe is None else f'{wfe:.2f}'} | minimum {self.gate:.3f}/trade")
        live_rules = live['rules'] if live else []
        if accepted:
            print(f"✅ Walk-forward passed → this month's {len(live_rules)} genetic rule(s) (optimized on "
                  f"{live['train'][0]}…{live['train'][1]}) join the model")
        else:
            why = (f"only {n_} test trades (< {GA_WF_MIN_OOS_TRADES})" if n_ < GA_WF_MIN_OOS_TRADES
                   else f"test trades averaged {avg:+.3f} < minimum {self.gate:.3f}")
            print(f"🛡️ Walk-forward not passed ({why}) → genetic rules are not traded this month; the named patterns "
                  f"still trade")
        months = [{'month': nm, 'rules': len(f['rules']), 'trades': len(f['oos']['r']),
                   'total': round(float(sum(f['oos']['r'])), 3)} for nm, f in hist]
        return {'accepted': accepted, 'rules': live_rules if accepted else [], 'live_rules': live_rules,
                'oos': {'trades': int(n_), 'avg': avg, 't': t, 'total': float(r.sum()), 'win': float(np.mean(r > 0)) if n_ else 0.0,
                        'wfe': wfe, 'positive_months': int(pos_months), 'months': int(len(hist))},
                'months': months, 'evals': int(self.evals), 'minutes': (time.time() - t0) / 60}


def ga_library_path(model_path=None):
    """کتابخانه‌ی الگوهای یادگرفته‌شده (کنار مدل، در پوشه‌ی مدل همین نماد/تایم‌فریم)"""
    return os.path.join(os.path.dirname(model_path or MODEL_PATH) or ".", "ga_library.json")


def _ga_load_library(path):
    """
    کتابخانه‌ی الگو: الگوهای ترکیبی که در آموزش‌های قبلی موفق بوده‌اند + اسم شکل‌های کندلی کشف‌شده.
    قانون نامعتبر (مثلاً الگویی که دیگر تعریف نشده) کنار گذاشته می‌شود. فایل خراب = کتابخانه‌ی خالی (با هشدار).
    """
    empty = {'version': 1, 'next_shape_id': 1, 'shape_names': {}, 'entries': []}
    if not os.path.exists(path):
        return empty
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lib = json.load(f)
    except (OSError, ValueError) as e:
        print(f"⚠️ Pattern library unreadable ({e}) — starting a new one")
        return empty
    entries = []
    for e in lib.get('entries', []):
        gene = e.get('gene') or {}
        trig = gene.get('trig')
        filters = tuple(sorted((str(n), bool(g)) for n, g in gene.get('filters', [])))
        conds = [tuple(cd) for cd in (gene.get('conds') or [])]
        if (_ga_valid_trig(trig) and all(n in GA_FILTERS for n, _ in filters)
                and all(_ga_valid_cond(cd) for cd in conds)):
            e['gene'] = _ga_norm_gene({'side': gene.get('side', 0), 'trig': trig, 'filters': filters, 'conds': conds})
            entries.append(e)
    names = {k: v for k, v in (lib.get('shape_names') or {}).items() if _ga_valid_shape(k)}
    return {'version': 1, 'next_shape_id': int(lib.get('next_shape_id', 1) or 1), 'shape_names': names,
            'entries': entries}


def _ga_save_library(lib, path):
    out = dict(lib)
    out['updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out['entries'] = []
    for e in lib['entries']:
        d = dict(e)
        g = _ga_norm_gene(e['gene'])
        d['gene'] = {'side': g['side'], 'trig': g['trig'], 'filters': [[n, bool(x)] for n, x in g['filters']],
                     'conds': [[f, op, float(thr)] for f, op, thr in g['conds']]}
        out['entries'].append(d)
    _atomic_write_json(out, path)


def _ga_name_shapes(lib, gene):
    """اسم دائمی برای شکل کشف‌شده‌ی تازه (و آینه‌اش) — «Learned #N» / «Learned #N mirror»"""
    code = gene['trig']
    if not _ga_is_shape(code) or code in lib['shape_names']:
        return
    k = lib['next_shape_id']
    lib['next_shape_id'] = k + 1
    lib['shape_names'][code] = f"Learned #{k}"
    mirror = _ga_mirror_shape(code)
    if mirror not in lib['shape_names']:
        lib['shape_names'][mirror] = f"Learned #{k} mirror"
    GA_SHAPE_NAMES.update(lib['shape_names'])


# ─────────────────────────── v8: کشف «دست باز» روی کل تاریخچه ───────────────────────────
def _ga_is_named_only(gene):
    """ژن بدون هیچ شرطی روی یک الگوی نام‌دار = خود همان الگو (جداگانه و با آمار خودش در مدل است)"""
    g = _ga_norm_gene(gene)
    return not g['filters'] and not g['conds'] and not _ga_is_shape(g['trig'])


def _ga_rows_of(eng, gene):
    """کندل‌های سیگنال یک ژن (خرید و/یا فروش آینه‌ای) روی کل تاریخچه — برای سنجش هم‌پوشانی"""
    m = None
    for r_ in _ga_pair(gene):
        mk = eng.mask(r_)
        m = mk if m is None else (m | mk)
    return np.flatnonzero(m) if m is not None else np.zeros(0, dtype=np.int64)


def _ga_overlap(a, b):
    """هم‌پوشانی دو مجموعه‌ی سیگنال (Jaccard = اشتراک ÷ اجتماع)؛ a و b اندیس‌های مرتب و یکتا"""
    if not len(a) or not len(b):
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    if len(a) / len(b) <= GA_MAX_OVERLAP:
        return 0.0                                    # اندازه‌ها آن‌قدر فرق دارند که هم‌پوشانی زیاد ممکن نیست
    pos = np.minimum(np.searchsorted(b, a), len(b) - 1)
    inter = int(np.count_nonzero(b[pos] == a))
    return inter / float(len(a) + len(b) - inter)


def _ga_diverse(eng, cands, limit):
    """
    از (برازندگی، ژن)ها به ترتیب بهترین اول، حداکثر limit ژن با برازندگی مثبت که سیگنالشان با ژن‌های انتخاب‌شده‌ی قبلی بیشتر
    از GA_MAX_OVERLAP هم‌پوشانی ندارد (نسخه‌های تقریباً تکراری یک قانون کنار گذاشته می‌شوند، نه قانون‌های واقعاً متفاوت)
    """
    kept, kept_rows, seen = [], [], set()
    for fit, gene in cands[:GA_SCAN_MAX]:
        if len(kept) >= limit or fit <= 0:
            break
        g = _ga_norm_gene(gene)
        key = _ga_gene_key(g)
        if key in seen or _ga_is_named_only(g):
            continue
        seen.add(key)
        rows = _ga_rows_of(eng, g)
        if any(_ga_overlap(rows, kr) > GA_MAX_OVERLAP for kr in kept_rows):
            continue
        kept.append((float(fit), g))
        kept_rows.append(rows)
    return kept


def _ga_rule_dict(r_, st, fit):
    """قانون یک‌طرفه‌ی ژنتیک برای مدل (قابل ذخیره در JSON)"""
    return {'dir': int(r_['dir']), 'trig': r_['trig'],
            'filters': [[nm, bool(x)] for nm, x in r_['filters']],
            'conds': [[f_, op, float(thr)] for f_, op, thr in r_['conds']],
            'source': 'ga', 'stats': st, 'fitness': round(float(fit), 3)}


def _ga_extra_rules(full_res, have_keys, min_avg):
    """
    قانون‌های سودده‌ی دیگر ژنتیک که در مدل نیستند (بیرون از سقف تعداد یا نسخه‌های نزدیک قانون‌های انتخاب‌شده)، به ترتیب
    برازندگی — برای رسیدن به هدف معامله‌ی روزانه. فقط قانون‌هایی با میانگین سود مثبت بعد از هزینه‌ها (pattern_qualifies).
    """
    eng = full_res['engine']
    out, keys = [], set(have_keys)
    for fit, gene, _ in eng.hall_of_fame()[:GA_SCAN_MAX]:
        if fit <= 0:
            break
        if _ga_is_named_only(gene):
            continue
        for r_ in _ga_pair(gene):
            if not (r_['filters'] or r_['conds'] or _ga_is_shape(r_['trig'])):
                continue
            key = _ga_key(r_)
            if key in keys:
                continue
            st = eng.stats(r_)
            if pattern_qualifies(st, min_avg):
                keys.add(key)
                out.append(_ga_rule_dict(r_, st, fit))
    return out


def _ga_discover_full(prim, tradable, mkt, min_avg, library, cancelled):
    """
    v8 «دست باز»: الگوریتم ژنتیک روی کل تاریخچه تا آخرین کندل یاد می‌گیرد (جزیره‌ها + جست‌وجوی محلی روی بهترین‌ها، با
    دانه‌ی قانون‌های کتابخانه)، و هر قانون متنوعی که برازندگی مثبت دارد و هر سمتش (خرید/فروش) بعد از همه‌ی هزینه‌ها میانگین
    سود مثبت و دست‌کم PATTERN_MIN_TRADES معامله داشته معامله می‌شود. آمار هر قانون (حد سود/ضرر) هم از کل تاریخچه است.
    خروجی: None اگر لغو شد، وگرنه dict: rules (قانون‌های یک‌طرفه برای مدل)، genes [(برازندگی، ژن)] انتخاب‌شده، evals
    """
    t0 = time.time()
    n = len(tradable)
    rows_all = np.flatnonzero(tradable)
    mv = np.concatenate([mkt.move[1][rows_all], mkt.move[-1][rows_all]]) if len(rows_all) else np.zeros(0)
    scale = float(np.nanmedian(np.abs(mv))) if np.isfinite(mv).any() else 1.0
    eng = _GAEngine(prim, tradable, mkt, n, scale, PATTERN_MIN_TRADES, drift_end=n,
                    grid=_ga_feature_grid(prim, rows_all))
    seeds = [e['gene'] for e in library['entries']]
    print(f"🧬 Genetic discovery on all {n:,} training bars (free hand): {GA_ISLANDS} island(s) × up to "
          f"{GA_GENERATIONS} generations × {GA_POP} rules, ≤{GA_MAX_FILTERS} conditions per rule"
          + (f", seeded with {len(seeds)} rule(s) from the pattern library" if seeds else ""))

    def on_gen(island, gen, gene, fit):
        training_state.update_progress(island * GA_GENERATIONS + gen)
        if gen == 1 or gen % 10 == 0:
            print(f"   island {island + 1}, generation {gen}: best fitness {fit:+.1f} — {_ga_gene_text(gene)}")

    for island in range(GA_ISLANDS):
        if not eng.run(cancelled, on_gen, island, extra_seeds=seeds):
            return None

    top = _ga_diverse(eng, [(f, g) for f, g, _ in eng.hall_of_fame()], GA_LOCAL_TOP)
    if top:
        print(f"🔧 Local search on the {len(top)} best distinct rules (neighbouring thresholds, dropping conditions)...")
    for _, g in top:
        if cancelled():
            return None
        eng.refine(g, cancelled)

    chosen = _ga_diverse(eng, [(f, g) for f, g, _ in eng.hall_of_fame()], GA_POOL_NEW)
    rules, keys = [], set()
    for fit, g in chosen:
        for r_ in _ga_pair(g):
            if not (r_['filters'] or r_['conds'] or _ga_is_shape(r_['trig'])):
                continue                                  # خود الگوی نام‌دار؛ جداگانه در مدل است
            key = _ga_key(r_)
            if key in keys:
                continue
            st = eng.stats(r_)
            if pattern_qualifies(st, min_avg):
                keys.add(key)
                rules.append(_ga_rule_dict(r_, st, fit))
    minutes = (time.time() - t0) / 60
    print(f"✅ Genetic discovery: {len(eng.cache):,} combinations tested in {minutes:.1f} min → {len(chosen)} distinct "
          f"rule(s) with positive fitness → {len(rules)} one-sided rule(s) with a positive average after all costs "
          f"(≥{PATTERN_MIN_TRADES} past trades) join the model")
    return {'rules': rules, 'genes': chosen, 'evals': len(eng.cache), 'minutes': minutes, 'engine': eng}


def _ga_update_library(library, eng, chosen):
    """
    کتابخانه‌ی الگو بعد از کشف «دست باز»: قانون‌های این آموزش اضافه/تازه می‌شوند (رد = صفر)؛ قانونی از کتابخانه که این بار
    انتخاب نشد یک «رد» می‌گیرد و بعد از GA_LIB_MAX_FAILS رد پشت‌سرهم حذف می‌شود. شکل‌های کندلی تازه اسم دائمی می‌گیرند
    («Learned #N»). حداکثر GA_LIB_MAX قانون (کم‌ردتر و پربرازنده‌تر اول). خروجی: (تعداد تازه، تعداد حذف‌شده)
    """
    now_txt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chosen_fit = {}
    for fit, g in chosen:
        chosen_fit.setdefault(_ga_gene_key(g), (float(fit), _ga_norm_gene(g)))
    entries, seen, removed = [], set(), 0
    for e in library['entries']:
        key = _ga_gene_key(e['gene'])
        if key in seen:
            continue
        seen.add(key)
        if key in chosen_fit:
            e.update(fails=0, fitness=round(chosen_fit[key][0], 3), last_ok=now_txt)
        else:
            e['fails'] = int(e.get('fails', 0) or 0) + 1
            hit = eng.cache.get(key)
            if hit is not None:
                e['fitness'] = round(float(hit[0]), 3)
            if e['fails'] >= GA_LIB_MAX_FAILS:
                removed += 1
                continue
        entries.append(e)
    added = 0
    for key, (fit, g) in chosen_fit.items():
        if key not in seen:
            entries.append({'gene': g, 'fails': 0, 'fitness': round(fit, 3), 'added': now_txt, 'last_ok': now_txt})
            added += 1
    for e in entries:
        if int(e.get('fails', 0) or 0) == 0:
            _ga_name_shapes(library, e['gene'])
    for e in entries:
        e['text'] = _ga_gene_text(e['gene'])
    entries.sort(key=lambda e: (int(e.get('fails', 0) or 0), -float(e.get('fitness', 0.0) or 0.0)))
    removed += max(0, len(entries) - GA_LIB_MAX)
    library['entries'] = entries[:GA_LIB_MAX]
    return added, removed


def _ga_load_dataset(db_path, limit=None):
    """کندل‌های لیبل‌دار (OHLC + ATR + اسپرد + بازده واقعی خرید/فروش) به ترتیب زمان؛ limit = فقط این تعداد کندل آخر"""
    conn = _db_connect(db_path)
    try:
        q = ("SELECT timestamp_us AS ts, open_price, high_price, low_price, close_price, atr, spread_px, "
             "target, buy_ret, sell_ret, buy_win, sell_win, buy_exit, sell_exit "
             "FROM training_data_multiclass WHERE COALESCE(is_synthetic, 0) = 0 ORDER BY timestamp_us DESC")
        if limit:
            q += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(q, conn).iloc[::-1].reset_index(drop=True)
    except Exception as e:
        print(f"✗ Cannot read labeled data: {e}")
        return None
    finally:
        conn.close()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    ok = np.isfinite(df[['open_price', 'high_price', 'low_price', 'close_price', 'atr']].to_numpy(float)).all(axis=1)
    if (~ok).any():
        print(f"ℹ️ {int((~ok).sum())} labeled rows have no OHLC (older version) — skipped")
    return df[ok].reset_index(drop=True)


def _ga_feature_grid(prim, rows):
    """آستانه‌های قابل انتخاب هر ویژگی عددی: چندک‌های آن روی کندل‌های Train"""
    qs = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    grid = {}
    for f in GA_NUM_FEATURES:
        x = prim['n:' + f][rows].astype(np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 500:
            continue
        g = np.unique(np.array([_ga_round(v) for v in np.quantile(x, qs)]))
        if len(g) >= 3:
            grid[f] = g
    return grid


def _ga_perf(rets, dirs):
    return _perf_from_returns(np.asarray(rets, dtype=float), int(np.sum(dirs == 1)), int(np.sum(dirs == -1)))


def _ga_pattern_report(named, engine, full_stats, qualifies, ts, te_lo, atr_now, path):
    """
    گزارش همه‌ی الگوهای نام‌دار (pattern_report.csv کنار مدل + خلاصه در کنسول): هر الگو جدا، با حد سود/ضرر خودش، یک
    پوزیشن در هر لحظه و بعد از همه‌ی هزینه‌ها — نرخ برد، میانگین سود هر معامله (واحد مدل و دلار با LOT فعلی) و تعداد
    معامله در روز، روی کل تاریخچه، روی Train و روی Test (Test با حد سود/ضرر و آمار Train، مثل ربات زنده)
    """
    n = len(ts)
    days = lambda lo, hi: max(1.0, (float(ts[min(hi, n) - 1]) - float(ts[lo])) / 86400.0 * 5 / 7)
    d_all, d_tr, d_te = days(0, n), days(0, engine.tr_end), days(te_lo, n)
    size = float(_contract_size.get(SYMBOL) or 100.0)
    usd = lambda x: x * (1.0 if LABEL_MODE == "fixed" else float(atr_now or 0.0)) * size * float(LOT)
    rows = []
    for r_ in named:
        st, st_tr = full_stats(r_), engine.stats(r_)
        r_te = engine.trades_in(r_, te_lo, n)[0] if st_tr else np.zeros(0)
        rows.append({
            'pattern': GA_NAMES.get(r_['trig'], r_['trig']), 'code': r_['trig'], 'side': 'BUY' if r_['dir'] == 1 else 'SELL',
            'selected': bool(qualifies(st)),
            'trades': int(st['n']) if st else 0, 'trades_per_day': round((st['n'] if st else 0) / d_all, 2),
            'win_rate': round(float(st['win']), 4) if st else None,
            'avg_per_trade': round(float(st['avg']), 4) if st else None,
            'avg_usd': round(usd(float(st['avg'])), 2) if st else None,
            'train_trades': int(st_tr['n']) if st_tr else 0,
            'train_win_rate': round(float(st_tr['win']), 4) if st_tr else None,
            'train_avg': round(float(st_tr['avg']), 4) if st_tr else None,
            'test_trades': int(len(r_te)), 'test_trades_per_day': round(len(r_te) / d_te, 2),
            'test_win_rate': round(float(np.mean(r_te > 0)), 4) if len(r_te) else None,
            'test_avg': round(float(np.mean(r_te)), 4) if len(r_te) else None,
            'tp': round(float(st['tp']), 4) if st else None, 'sl': round(float(st['sl']), 4) if st else None,
            'occurrences_per_day': round((st['occ'] if st else 0) / d_all, 2),
        })
    rep = pd.DataFrame(rows).sort_values('avg_per_trade', ascending=False, na_position='last')
    try:
        rep.to_csv(path, index=False, encoding='utf-8-sig')
        where = f" → {path}"
    except OSError as e:
        where = f" (not saved: {e})"
    u = ret_unit()
    print(f"\n📑 All {len(rep)} named patterns, each traded alone ({pattern_exit_text()}, one position, all costs; "
          f"whole history {d_all:,.0f} trading days){where}")
    print(f"   {'pattern':<44}{'side':<5}{'trades/day':>11}{'win':>7}{'avg ' + u:>10}{'$/trade':>9}"
          f"{'test win':>10}{'test avg':>10}  selected")
    for _, x in rep.iterrows():
        if x['trades'] == 0:
            continue
        tw = f"{x['test_win_rate']:.0%}" if x['test_win_rate'] is not None and np.isfinite(x['test_win_rate']) else "—"
        ta = f"{x['test_avg']:+.3f}" if x['test_avg'] is not None and np.isfinite(x['test_avg']) else "—"
        print(f"   {x['pattern'][:43]:<44}{x['side']:<5}{x['trades_per_day']:>11.2f}{x['win_rate']:>7.0%}"
              f"{x['avg_per_trade']:>+10.3f}{x['avg_usd']:>+9.2f}{tw:>10}{ta:>10}  {'✓' if x['selected'] else ''}")
    return rep


def _train_ga_locked(result, primary_db, model_save_path, last_train_file, P, S):
    """
    آموزش مدل الگوها (داخل قفل آموزش). همان خروجی train_and_validate.
      ۱) آمار هر الگو: معامله با حد سود/ضرر ثابت (v7؛ یا در حالت runs حد سود = میانگین بیشترین قیمت قبل از خوردن حد ضرر)،
         یک پوزیشن در هر لحظه، بعد از اسپرد + لغزش + کمیسیون + سواپ → نرخ برد و میانگین سود واقعی هر معامله.
      ۲) الگوریتم ژنتیک الگوهای ترکیبی (الگو + شرایط بازار) کشف می‌کند — v8 «دست باز»: روی کل تاریخچه تا آخرین کندل، با
         دانه‌ی کتابخانه (_ga_discover_full)؛ با GA_WALK_FORWARD: روش ماهانه‌ی v7.
      ۳) الگوهایی در مدل‌اند که دست‌کم PATTERN_MIN_TRADES معامله‌ی گذشته و میانگین سود مثبت بعد از همه‌ی هزینه‌ها دارند
         (pattern_qualifies؛ PATTERN_MIN_PROFIT_USD = 0 یعنی هر میانگین مثبت).
      سنجش منصفانه: آمار فقط از Train، معامله روی Validation و Test؛ مدل نهایی آمار را از کل تاریخچه تا آخرین کندل می‌گیرد.
      گزارش همه‌ی الگوها (نرخ برد، سود هر معامله، تعداد در روز، Train/Test) کنار مدل: pattern_report.csv
    """
    global last_training_report
    t0 = time.time()
    try:
        print("\n" + "=" * 60)
        if not GA_DISCOVERY:
            disc_txt = ""
        elif GA_WALK_FORWARD:
            disc_txt = (f" + genetic discovery (walk-forward: {GA_WF_TRAIN_MONTHS}-month train → "
                        f"{GA_WF_TEST_MONTHS}-month test, re-optimized monthly)")
        else:
            disc_txt = " + genetic discovery on all training bars (free hand: every profitable rule trades)"
        tf_name = lambda sec: next((k for k, v in TF_SECONDS.items() if v == int(sec)), f"{int(sec)}s")
        print(f"🧬 TRAINING PATTERN MODEL v{GA_VERSION} — {pattern_exit_text()}; a pattern trades if {pattern_selection_text()}"
              f" | {len(GA_BULL) + len(GA_BEAR)} named patterns{disc_txt} | patterns on {tf_name(P)}, trend on "
              f"{tf_name(S)} | last {GA_TRAIN_BARS:,} bars | target ≥{GA_MIN_TRADES_PER_DAY:g} trade/day")
        print("=" * 60)
        if not GA_DISCOVERY:
            training_state.start(total_epochs=1)
        elif GA_WALK_FORWARD:
            training_state.start(total_epochs=GA_WF_STEPS + 1)
        else:
            training_state.start(total_epochs=GA_ISLANDS * GA_GENERATIONS + 1)

        df = _ga_load_dataset(primary_db, limit=GA_TRAIN_BARS)
        warm = ga_warmup(P, S)
        if df is None or len(df) < max(int(MIN_SAMPLES_FOR_TRAINING), warm + 3000):
            print(f"✗ Not enough labeled bars for the pattern model ({0 if df is None else len(df)})")
            save_train_time(False, "Not enough labeled data for the pattern model", last_train_file, model_save_path)
            return result

        num = lambda col: df[col].to_numpy(dtype=float)
        o, h, l, c, atr = num('open_price'), num('high_price'), num('low_price'), num('close_price'), num('atr')
        ts = df['ts'].to_numpy(dtype=np.int64)
        labeled = ((num('target') >= 0) & np.isin(num('buy_win'), (0, 1)) & np.isin(num('sell_win'), (0, 1))
                   & np.isfinite(num('buy_ret')) & np.isfinite(num('sell_ret'))
                   & (num('buy_exit') >= 1) & (num('sell_exit') >= 1))

        years = (int(ts[-1]) - int(ts[0])) / (365.25 * 86400.0)
        if len(df) < 0.9 * GA_TRAIN_BARS:
            print(f"⚠️ Only {len(df):,} bars of history ({years:.1f} years; training uses up to {GA_TRAIN_BARS:,}) — in "
                  f"MetaTrader set Tools → Options → Charts → 'Max bars in chart' higher and restart it; the bot then "
                  f"downloads the rest by itself")
        print(f"🔎 Detecting {len(GA_BULL) + len(GA_BEAR)} patterns, {len(GA_FILTERS)} context filters and "
              f"{len(GA_NUM_FEATURES)} numeric features on {len(df):,} bars ({years:.1f} years)...")
        prim, valid = ga_primitives(o, h, l, c, atr, ts, P, S)
        tradable = valid & labeled & (atr > 0) & ~prim['pre_break']

        n = len(df)
        fixed = LABEL_MODE == "fixed"
        unit = np.ones(n) if fixed else atr
        # هزینه‌ی امروز برای کل تاریخچه: الگوها برای معامله‌ی «الان» انتخاب می‌شوند، پس اسپرد کندل‌های قدیمی (که نسبت به
        # ATR چند برابر امروز بود) تا ۱.۵ برابرِ نسبت اسپرد/ATR اخیر محدود می‌شود و لغزش + کمیسیون به نسبت ATR امروز
        tail = slice(max(0, n - 5000), n)
        atr_ref = float(np.nanmedian(atr[tail]))
        px_ref = float(np.nanmedian(c[tail]))
        spread_raw = np.nan_to_num(num('spread_px'), nan=0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            sp_ratio = float(np.nanmedian(spread_raw[tail] / atr[tail]))
        ok_ratio = bool(np.isfinite(sp_ratio) and sp_ratio > 0)
        spread_m = np.minimum(spread_raw, 1.5 * sp_ratio * atr) if ok_ratio else spread_raw
        com_frac = float(_commission_frac.get(SYMBOL, MODEL_COMMISSION_FRAC))
        cost_x = float(EXTRA_COST_POINTS) * _point_size.get(SYMBOL, 0.0)
        extra_px = cost_x + (MODEL_SLIPPAGE_ATR + com_frac * px_ref / max(atr_ref, 1e-9)) * atr
        swap = _swap_info.get(SYMBOL)
        mkt = _GAMarket(o, h, l, c, unit, spread_m, ts, P, extra_px, dict(swap, ref_price=px_ref) if swap else None)
        u = ret_unit()
        atr_now = atr_ref if not fixed else None
        min_avg = pattern_min_avg_units(atr_ref)
        if fixed and atr_ref > 0 and not 0.1 <= float(FIXED_SL_PRICE) / atr_ref <= 50:
            print(f"⚠️ Fixed SL {_px_txt(FIXED_SL_PRICE)} is {float(FIXED_SL_PRICE) / atr_ref:.3g}× the typical "
                  f"{tf_name(P)} ATR of {SYMBOL} ({_px_txt(atr_ref)}) — Fixed TP/SL are price distances of this symbol; "
                  f"set them in Settings → Model (or use ATR mode)")

        def qualifies(st_):
            """الگو معامله می‌شود: معاملات گذشته‌ی کافی و (نرخ برد ≥ حداقل یا میانگین سود واقعی هر معامله ≥ حداقل)"""
            return pattern_qualifies(st_, min_avg)

        gap = mkt.k + 1                     # هیچ معامله‌ای از یک بخش به بخش بعد نمی‌رسد
        tr_end = int(n * 0.70)
        va_lo, va_hi = tr_end + gap, tr_end + gap + int(n * 0.15)
        te_lo = va_hi + gap
        if te_lo >= n - 50:
            print("✗ Not enough data for split")
            save_train_time(False, "Not enough data for train/val/test split", last_train_file, model_save_path)
            return result

        def span(lo, hi):
            f = lambda x: datetime.fromtimestamp(int(x), tz=timezone.utc).strftime('%Y-%m-%d')
            return f"{f(ts[lo])}→{f(ts[min(hi, n) - 1])}"

        print(f"📊 Split: Train={tr_end} ({span(0, tr_end)}) | Val={va_hi - va_lo} ({span(va_lo, va_hi)}) | "
              f"Test={n - te_lo} ({span(te_lo, n)})")
        if mkt.has_swap:
            print(f"🌙 Overnight swap included: long {mkt.swap_px[1]:+.4g}, short {mkt.swap_px[-1]:+.4g} per night")
        sl_txt = " / ".join((_px_txt(x) if fixed else f"{x:g}×ATR") for x in pattern_sl_choices())
        print(f"💸 Costs per trade, today's conditions applied to the whole history: spread ≈{sp_ratio:.3f}×ATR + entry "
              f"slippage {MODEL_SLIPPAGE_ATR:g}×ATR + commission ≈${com_frac * px_ref:.3f} of price | stop-loss per pattern "
              f"{sl_txt} | {pattern_exit_text()} | selection: {pattern_selection_text()}"
              + (f" (${PATTERN_MIN_PROFIT_USD:g} = {min_avg:.3f} {u}/trade)" if PATTERN_MIN_PROFIT_USD > 0 else ""))

        tr_rows = np.flatnonzero(tradable[:tr_end])
        mv_tr = np.concatenate([mkt.move[1][tr_rows], mkt.move[-1][tr_rows]])
        scale = float(np.nanmedian(np.abs(mv_tr))) if len(tr_rows) else 1.0
        # حداقل معامله روی Train طوری که روی Test (۱۵٪) هم حدود MIN_TEST_TRADES معامله بماند
        min_train = max(60, int(np.ceil(MIN_TEST_TRADES * 70 / 15)))
        min_val = max(5, int(MIN_TEST_TRADES) // 3)
        grid = _ga_feature_grid(prim, tr_rows)

        engine = _GAEngine(prim, tradable, mkt, tr_end, scale, min_train, drift_end=va_hi, grid=grid)

        # ── ۱) آمار هر الگوی نام‌دار روی Train ──
        named = [{'dir': 1 if p_ in GA_BULL else -1, 'trig': p_, 'filters': (), 'conds': (), 'source': 'pattern'}
                 for p_ in GA_BULL + GA_BEAR]
        table = [(r_, engine.stats(r_)) for r_ in named]
        table = [(r_, st_) for r_, st_ in table if st_]
        keep_tr = sorted([(r_, st_) for r_, st_ in table if qualifies(st_)], key=lambda x: -x[1]['avg'])
        print(f"📋 Pattern statistics on Train ({u} units): {len(keep_tr)} of {len(table)} patterns meet the selection "
              f"rule ({pattern_selection_text()})")
        for r_, st_ in keep_tr[:12]:
            print(f"   {'BUY ' if r_['dir'] == 1 else 'SELL'} {GA_NAMES[r_['trig']]:<40} {ga_stats_text(st_, atr_now)}")

        lib_path = ga_library_path(model_save_path)
        # قانون‌های یادگرفته‌شده‌ی آموزش‌های قبلی (دانه‌ی کشف «دست باز») + اسم شکل‌های کندلی کشف‌شده؛
        # در حالت Walk-Forward فقط اسم شکل‌ها استفاده می‌شود
        library = _ga_load_library(lib_path)
        GA_SHAPE_NAMES.update(library['shape_names'])

        def cancelled():
            return training_cancel.is_set()

        def items_for(rules, st_of):
            """[(rule, mask, stats)] الگوهایی که به حداقل میانگین سود می‌رسند (qualifies)، به ترتیب سود هر معامله"""
            out_ = []
            for r_ in rules:
                st_ = st_of(r_)
                if qualifies(st_):
                    out_.append((r_, engine.mask(r_), st_))
            return sorted(out_, key=lambda x: (-x[2]['avg'], -x[2]['n']))

        # ── ۲) کشف الگوهای ترکیبی با الگوریتم ژنتیک ──
        #   «دست باز» (پیش‌فرض v8): روی کل تاریخچه، هر قانون سودده معامله می‌شود
        #   GA_WALK_FORWARD: روش v7 (هر ماه دوباره بهینه‌سازی روی پنجره‌ی اخیر + دروازه‌ی سابقه)
        wf_res = full_res = None
        lib_added = lib_removed = 0
        if GA_DISCOVERY and GA_WALK_FORWARD:
            wf = GAWalkForward(prim, tradable, mkt, ts, P, min_avg, ga_walkforward_path(model_save_path),
                               cancelled=cancelled, progress=training_state.update_progress)
            wf_res = wf.run()
            if wf_res is None:
                print("⏹️ Training cancelled — nothing saved")
                result['cancelled'] = True
                return result
        elif GA_DISCOVERY:
            full_res = _ga_discover_full(prim, tradable, mkt, min_avg, library, cancelled)
            if full_res is None:
                print("⏹️ Training cancelled — nothing saved")
                result['cancelled'] = True
                return result
            lib_added, lib_removed = _ga_update_library(library, full_res['engine'], full_res['genes'])
        else:
            print("ℹ️ Genetic discovery of pattern combinations is off (GA_DISCOVERY): the model uses the named "
                  "patterns that meet the selection rule")
        if wf_res:
            ga_rules, n_evals = list(wf_res['rules']), int(wf_res['evals'])
        elif full_res:
            ga_rules, n_evals = list(full_res['rules']), int(full_res['evals'])
        else:
            ga_rules, n_evals = [], 0

        # سنجش منصفانه: آمار فقط از Train، معامله روی Validation و Test (مثل ربات زنده)
        items_tr = items_for(named, engine.stats)       # الگوهای ژنتیک را خود Walk-Forward سنجیده است
        rv, dv_ = engine.portfolio(items_tr, va_lo, va_hi)
        rt, dt_ = engine.portfolio(items_tr, te_lo, n)
        perf_val, perf_test = _ga_perf(rv, dv_), _ga_perf(rt, dt_)
        named_items = items_for(named, engine.stats)
        rtn, dtn = engine.portfolio(named_items, te_lo, n)
        perf_test_named = _ga_perf(rtn, dtn)

        # مدل نهایی: آمار هر الگو از کل تاریخچه («گذشته‌ی بازار» تا همین الان)
        full_st = {}

        def full_stats(r_):
            key = _ga_key(r_)
            if key not in full_st:
                rows = np.flatnonzero(engine.mask(r_))
                full_st[key] = (mkt.best_rule_stats(rows, int(r_['dir']), min_n=PATTERN_MIN_TRADES)
                                if len(rows) else None)
            return full_st[key]

        _ga_pattern_report(named, engine, full_stats, qualifies, ts, te_lo, atr_now,
                           os.path.join(os.path.dirname(ga_model_path(model_save_path)) or ".", "pattern_report.csv"))
        final_items = items_for(named, full_stats)
        final_items += [(r_, None, r_['stats']) for r_ in ga_rules]
        final_items.sort(key=lambda x: (-x[2]['avg'], -x[2]['n']))

        # ── هدف «هر روز معامله»: کل مدل روی کل داده‌ی آموزش، یک پوزیشن در هر لحظه مثل ربات زنده. اگر کمتر از
        #    GA_MIN_TRADES_PER_DAY بود، قانون‌های سودده‌ی دیگر ژنتیک (به ترتیب برازندگی) اضافه می‌شوند؛ قانون ضررده هرگز ──
        trade_days = max(1, len(np.unique(ts[tradable] // 86400)))

        def run_model(items_):
            if not items_:
                return np.zeros(0), np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int64)
            return engine.portfolio([(r_, engine.mask(r_), st_) for r_, _, st_ in items_], 0, n, with_rows=True)

        ra, da, rows_a = run_model(final_items)
        n_fill = 0
        if full_res and len(ra) / trade_days < GA_MIN_TRADES_PER_DAY:
            before = len(ra) / trade_days
            extra = _ga_extra_rules(full_res, {_ga_key(r_) for r_, _, _ in final_items}, min_avg)
            stalled = False
            while extra and n_fill < GA_DAILY_FILL_MAX and len(ra) / trade_days < GA_MIN_TRADES_PER_DAY:
                batch, extra = extra[:10], extra[10:]
                trial = sorted(final_items + [(r_, None, r_['stats']) for r_ in batch],
                               key=lambda x: (-x[2]['avg'], -x[2]['n']))
                ra2, da2, rows2 = run_model(trial)
                if len(ra2) < len(ra) + 0.01 * trade_days:
                    stalled = True                            # این دسته معامله‌ی روزانه را بیشتر نکرد: اضافه نمی‌شود
                    break
                final_items, (ra, da, rows_a) = trial, (ra2, da2, rows2)
                n_fill += len(batch)
            after = len(ra) / trade_days
            why = ("" if after >= GA_MIN_TRADES_PER_DAY else
                   " (more rules would not add trades: with one position at a time the open trade blocks new "
                   "entries)" if stalled else
                   " (no more profitable rules — rules that lost money after costs are never added)")
            print(f"📅 Daily-trading target ≥{GA_MIN_TRADES_PER_DAY:g}/day: the model gave {before:.2f}/day → added "
                  f"{n_fill} more profitable genetic rule(s) → {after:.2f}/day{why}")
        per_day_all = len(ra) / trade_days
        days_with = len(np.unique(ts[rows_a] // 86400)) / trade_days if len(rows_a) else 0.0
        n_named = sum(1 for r_, _, _ in final_items if r_.get('source') == 'pattern')
        ga_src = ("of this month's walk-forward step" if wf_res else
                  "learned on all training bars" if full_res else "")
        print(f"\n🏆 Pattern model: {len(final_items)} patterns trade = {n_named} named (of {len(named)}; whole history "
              f"up to the last bar: {pattern_selection_text()}) + "
              f"{len(final_items) - n_named} genetic rule(s) {ga_src}; when several patterns "
              f"appear on one bar, the one with the best past result per trade trades:")
        for r_, _, st_ in final_items[:20]:
            print(f"   • {_ga_rule_text(r_, with_stats=False)}\n       {ga_stats_text(st_, atr_now)}")
        if len(final_items) > 20:
            print(f"   … and {len(final_items) - 20} more")

        te_sel = np.flatnonzero(engine.ok_rows[te_lo:]) + te_lo
        baseline_buy = float(np.nanmean(mkt.move[1][te_sel] - mkt.cost[te_sel])) if len(te_sel) else 0.0
        baseline_sell = float(np.nanmean(mkt.move[-1][te_sel] - mkt.cost[te_sel])) if len(te_sel) else 0.0
        per_day = len(rt) / max(1.0, (ts[n - 1] - ts[te_lo]) / 86400.0 * 5 / 7)
        ga_note = ("walk-forward record above" if wf_res else
                   "learned on all data, so they have no unseen-data test in free-hand mode" if full_res else "off")
        print(f"\n💰 Out-of-sample simulation of the named patterns together (chosen on Train only; {pattern_exit_text()}, "
              f"one position at a time, {u} units, all costs; genetic rules: {ga_note}):")
        print(f"   Validation: {_format_perf(perf_val)}")
        print(f"   Test      : {_format_perf(perf_test)} | ≈{per_day:.1f} trades/day")
        print(f"   Baseline (1 hour, no pattern): buy exp={baseline_buy:+.3f} {u} | sell exp={baseline_sell:+.3f} {u}")
        perf_in_sample = None
        if final_items:
            # کل مدل (الگوهای نام‌دار + قانون‌های ژنتیک) روی کل داده‌ی آموزش، مثل ربات زنده — درون‌نمونه‌ای: سقف خوش‌بینانه
            perf_in_sample = _ga_perf(ra, da)
            print(f"   Whole model on all training bars (IN-SAMPLE — the statistics"
                  f"{' and the genetic search' if full_res else ''} saw this data, so read it as an optimistic upper "
                  f"bound, not a forecast): {_format_perf(perf_in_sample)} | ≈{per_day_all:.1f} trades/day, "
                  f"a trade on {days_with:.0%} of trading days")

        quality_notes = []
        if not final_items:
            quality_notes.append(f"no pattern meets the selection rule ({pattern_selection_text()}) — Model "
                                 f"mode will not open trades until a later training finds one")
        if final_items and per_day_all < GA_MIN_TRADES_PER_DAY:
            quality_notes.append(f"only {per_day_all:.2f} trades/day (target {GA_MIN_TRADES_PER_DAY:g}) — not enough "
                                 f"profitable patterns/rules")
        if perf_test['trades'] < MIN_TEST_TRADES:
            quality_notes.append(f"few test trades ({perf_test['trades']} < {MIN_TEST_TRADES})")
        if perf_test['expectancy'] < min_test_expectancy():
            quality_notes.append(f"test expectancy {perf_test['expectancy']:+.3f} < {min_test_expectancy():.3g} {u}")
        if perf_test['profit_factor'] < MIN_TEST_PROFIT_FACTOR:
            quality_notes.append(f"test PF {perf_test['profit_factor']:.2f} < {MIN_TEST_PROFIT_FACTOR}")

        final_rules = []
        for r_, _, st_ in final_items:
            final_rules.append({'dir': int(r_['dir']), 'trig': r_['trig'],
                                'filters': [[nm, bool(g)] for nm, g in r_['filters']],
                                'conds': [[f_, op, float(thr)] for f_, op, thr in r_['conds']],
                                'source': r_.get('source', 'ga'), 'stats': st_,
                                'win_rate': float(st_['win']), 'text': _ga_rule_text(r_, with_stats=False)})
        has_buy = any(r['dir'] == 1 for r in final_rules)
        has_sell = any(r['dir'] == -1 for r in final_rules)

        minutes = (time.time() - t0) / 60
        now_txt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nan_head = {'auc': float('nan'), 'brier': float('nan'), 'base_rate': float('nan')}
        metrics = {
            'kind': 'ga_patterns',
            'heads': {'buy': {'train': nan_head, 'test': nan_head}, 'sell': {'train': nan_head, 'test': nan_head}},
            'val': perf_val, 'test': perf_test, 'test_named_only': perf_test_named, 'rules': final_rules,
            'buy_threshold': 0.0 if has_buy else DISABLED_THRESHOLD,
            'sell_threshold': 0.0 if has_sell else DISABLED_THRESHOLD,
            'baseline_buy': baseline_buy, 'baseline_sell': baseline_sell,
            'label_mode': LABEL_MODE, 'ret_unit': u, 'auc_history': [],
            'quality_warnings': quality_notes, 'minutes': minutes, 'time': now_txt,
            'rules_tested': n_evals, 'in_sample': perf_in_sample,
            'trades_per_day': per_day_all if final_items else 0.0, 'days_with_trade': days_with if final_items else 0.0,
            'daily_fill_rules': n_fill,
            'ga_mode': 'walk_forward' if wf_res else ('full_history' if full_res else 'off'),
            'walk_forward': ({k_: v for k_, v in wf_res.items() if k_ not in ('rules', 'live_rules')}
                             if wf_res else None),
        }
        result['metrics'] = metrics
        last_training_report = metrics

        meta = {
            'kind': 'ga_patterns', 'ga_version': GA_VERSION,
            'label_signature': label_signature(), 'decision_rule': DECISION_RULE,
            'label_mode': LABEL_MODE, 'barriers': barrier_text(),
            'sl_units': float(mkt.sl_units), 'min_profit_usd': float(PATTERN_MIN_PROFIT_USD),
            'min_avg_units': float(min_avg), 'min_trades': int(PATTERN_MIN_TRADES), 'lot': float(LOT),
            'history_bars': int(n),
            'swap': swap if mkt.has_swap else None,
            'primary_tf_seconds': int(P), 'secondary_tf_seconds': int(S), 'pivot_k': GA_PIVOT_K,
            'timesteps': TIMESTEPS, 'timesteps_secondary': TIMESTEPS_SECONDARY,
            'buy_threshold': metrics['buy_threshold'], 'sell_threshold': metrics['sell_threshold'],
            'trained_at': now_txt, 'val': perf_val, 'test': perf_test,
            'quality_warnings': quality_notes, 'rules_tested': n_evals, 'in_sample': perf_in_sample,
            'ga_mode': metrics['ga_mode'],
            'walk_forward': ({k_: v for k_, v in wf_res.items() if k_ not in ('rules', 'live_rules', 'months')}
                             if wf_res else None),
            'data_end_ts': int(ts[-1]), 'test_start_ts': int(ts[te_lo]),
            'shape_names': dict(library['shape_names']),
            'library_shapes': sorted({c_ for e in library['entries'] for c_ in
                                      (e['gene']['trig'], _ga_mirror_trig(e['gene']['trig']))
                                      if _ga_is_shape(c_)}),
            'library_size': len(library['entries']),
        }
        model = GAPatternModel(final_rules, meta)
        _atomic_write_json(model.to_dict(), ga_model_path(model_save_path))
        if full_res:
            # کتابخانه بعد از ذخیره‌ی مدل: قانون‌های یادگرفته‌شده دانه‌ی آموزش بعدی می‌شوند
            _ga_save_library(library, lib_path)
            print(f"📚 Pattern library: {len(library['entries'])} rule(s) saved ({lib_added} new, {lib_removed} dropped "
                  f"after {GA_LIB_MAX_FAILS} trainings in a row without being chosen) → {lib_path}")

        summary = (f"Pattern model v{GA_VERSION}: {len(final_rules)} patterns | test {_format_perf(perf_test)} | "
                   f"Val exp={perf_val['expectancy']:+.3f}")
        if quality_notes:
            summary += " | Warnings: " + "; ".join(quality_notes)
            print(f"\n⚠️ Quality checks not met (saved anyway — no save filter): {'; '.join(quality_notes)}")
        print(f"\n⏱️ Training time: {minutes * 60:.0f} s | genetic combinations tested this run: {n_evals:,}")
        print(f"✅ Pattern model saved and deployed → {ga_model_path(model_save_path)}")
        save_train_time(True, summary, last_train_file, model_save_path)
        result.update(accepted=True, model=model, scaler_p=GA_NO_SCALER, scaler_s=GA_NO_SCALER, meta=meta)
        print("=" * 60 + "\n")
        return result

    except Exception as e:
        print(f"✗ Pattern model training error: {e}")
        traceback.print_exc()
        save_train_time(False, f"Pattern model training error: {e}", last_train_file, model_save_path)
        return result
    finally:
        training_state.finish()


def load_ga_model(model_path=None):
    """مدل الگوهای ذخیره‌شده یا None"""
    path = ga_model_path(model_path)
    with open(path, 'r', encoding='utf-8') as f:
        return GAPatternModel.from_dict(json.load(f))


# ========================
# Prediction
# ========================
# سازگاری با نسخه‌ی قبل: (کلاس، [p_sell, p_none, p_buy], اطمینان)
_NEUTRAL_PREDICTION = (1, [0.0, 1.0, 0.0], 0.0)
_short_history_warned = False


def model_matches_settings(model):
    """مدل باید مدل GA همین تنظیمات (تایم‌فریم‌ها و نسخه) باشد؛ مدل قدیمی شبکه‌ی عصبی دیگر استفاده نمی‌شود"""
    if not isinstance(model, GAPatternModel):
        return False
    meta = model.meta
    return (int(meta.get('ga_version', -1)) == GA_VERSION
            and int(meta.get('primary_tf_seconds', -1)) == int(PRIMARY_TF_SECONDS)
            and int(meta.get('secondary_tf_seconds', -1)) == ga_htf_seconds()
            and all(_ga_valid_trig(r['trig']) and all(nm in GA_FILTERS for nm, _ in r['filters'])
                    and all(_ga_valid_cond(cd) for cd in r['conds']) and _ga_valid_stats(r.get('stats'))
                    for r in model.rules))


def _ga_valid_stats(st):
    """آمار الگو برای معامله کافی است؟ (حد سود و ضرر مثبت و فرصت معامله)"""
    try:
        return (float(st['tp']) > 0 and float(st['sl']) > 0 and int(st.get('hold') or 0) >= 0
                and bool(np.isfinite(float(st['tp'])) and np.isfinite(float(st['sl']))))
    except (TypeError, ValueError, KeyError):
        return False


def get_model_probabilities(model, scaler_primary, scaler_secondary,
                            features_primary_df, features_secondary_df, df_primary=None):
    """
    مدل GA: (نرخ برد Validation قانون خرید برقرار، همان برای فروش) روی آخرین کندل df_primary؛ 0 = قانونی برقرار نیست.
    الگوها از OHLC حساب می‌شوند، پس df_primary (کندل‌ها با Open/High/Low/Close/ATR_14) لازم است؛ بدون آن None.
    """
    if isinstance(model, GAPatternModel):
        try:
            res = model.decide(df_primary) if df_primary is not None else None
        except Exception as e:
            print(f"✗ GA prediction error: {e}")
            return None
        return None if res is None else (float(res['p_buy']), float(res['p_sell']))
    return _nn_model_probabilities(model, scaler_primary, scaler_secondary,
                                   features_primary_df, features_secondary_df)


def _nn_model_probabilities(model, scaler_primary, scaler_secondary,
                            features_primary_df, features_secondary_df):
    """
    (قدیمی — برای مدل شبکه‌ی عصبی) احتمال برد خرید و فروش برای آخرین کندل: (p_buy_win, p_sell_win) یا None.
    پنجره‌ها باید بعد از دوره‌ی warm-up باشند تا فیچرها دقیقاً مثل داده‌ی آموزش باشند.
    """
    global _short_history_warned
    try:
        if model is None or scaler_primary is None or scaler_secondary is None:
            return None

        need_p = WARMUP_BARS + TIMESTEPS
        need_s = WARMUP_BARS + TIMESTEPS_SECONDARY
        if len(features_primary_df) < need_p or len(features_secondary_df) < need_s:
            if not _short_history_warned:
                print(f"⚠️ Not enough bars for prediction (primary {len(features_primary_df)}/{need_p}, "
                      f"secondary {len(features_secondary_df)}/{need_s}) — use bot.live_history_bars()")
                _short_history_warned = True
            return None

        n_features = features_primary_df.shape[1]
        if features_secondary_df.shape[1] != n_features:
            print(f"✗ Feature mismatch: Primary={n_features} vs Secondary={features_secondary_df.shape[1]}")
            return None

        primary_recent = features_primary_df.iloc[-TIMESTEPS:].to_numpy(dtype=np.float64)
        secondary_recent = features_secondary_df.iloc[-TIMESTEPS_SECONDARY:].to_numpy(dtype=np.float64)
        if not (np.isfinite(primary_recent).all() and np.isfinite(secondary_recent).all()):
            print("⚠️ Input features contain NaN/Inf")
            return None

        Xp = scaler_primary.transform(primary_recent).reshape(1, TIMESTEPS, n_features).astype(np.float32)
        Xs = scaler_secondary.transform(secondary_recent).reshape(1, TIMESTEPS_SECONDARY, n_features).astype(np.float32)

        out = np.asarray(model([Xp, Xs], training=False)).reshape(-1)
        if len(out) == 2:
            return float(out[0]), float(out[1])
        if len(out) == 3:            # مدل سه‌کلاسه‌ی قدیمی
            return float(out[2]), float(out[0])
        print(f"✗ Unexpected model output size {len(out)}")
        return None

    except Exception as e:
        print(f"✗ Prediction error: {e}")
        traceback.print_exc()
        return None


def get_model_prediction_multiclass(model, scaler_primary, scaler_secondary,
                                    features_primary_df, features_secondary_df, df_primary=None):
    """
    سازگاری با نسخه‌ی قبل (main_app این اسم را import می‌کند).
    Returns: (predicted_class, [p_sell_win, p_none, p_buy_win], confidence)
      predicted_class: 2 اگر خرید قوی‌تر است، 0 اگر فروش، 1 اگر مدل در دسترس نیست
      مدل GA: p = نرخ برد Validation قانون برقرار (0 = قانونی برقرار نیست)؛ بدون df_primary خنثی.
    """
    probs = get_model_probabilities(model, scaler_primary, scaler_secondary,
                                    features_primary_df, features_secondary_df, df_primary=df_primary)
    if probs is None:
        return _NEUTRAL_PREDICTION
    p_buy, p_sell = probs
    if p_buy == 0.0 and p_sell == 0.0:
        return _NEUTRAL_PREDICTION
    cls = 2 if p_buy >= p_sell else 0
    p_none = (1.0 - p_buy) * (1.0 - p_sell)
    return cls, [p_sell, p_none, p_buy], max(p_buy, p_sell)


def get_combined_signal_multiclass(model, scaler_primary, scaler_secondary,
                                   df_primary, features_primary_df,
                                   features_secondary_df, publish=True, verbose=True,
                                   mode=None, return_reason=False, return_exit=False, df_long=None):
    """
    سیگنال نهایی بر اساس منبع انتخاب‌شده (SIGNAL_MODE) — دو حالت کاملاً مستقل:
      'model'      : استراتژی الگوریتم ژنتیک. الگوهای کندلی/کلاسیک، فیلترهای زمینه و شرط‌های عددی روی آخرین کندل
                     df_primary (باید کندل بسته باشد) حساب می‌شوند؛ اگر قانونی از قانون‌های اعتبارسنجی‌شده برقرار باشد
                     معامله در همان جهت با خروجِ همان قانون (اگر خرید و فروش هر دو برقرار باشند، معامله‌ای نیست).
      'indicators' : فقط اندیکاتورها. امتیاز Confluence با MIN_CONFLUENCE_SCORE مقایسه می‌شود
                     (decide_indicator_signals). مدل اصلاً اجرا نمی‌شود.

    امتیاز Confluence در حالت مدل فقط برای نمایش در GUI محاسبه می‌شود و روی تصمیم اثری ندارد.

    Returns:
        (final_signal, confidence) — final_signal: 1 Buy / -1 Sell / 0 No trade
        (با return_reason=True متن دلیل تصمیم، و با return_exit=True پروفایل خروج قانون GA (یا None) هم برمی‌گردد)
    """
    mode = (mode or SIGNAL_MODE)
    confluence_signals = get_confluence_signals(df_primary)
    confluence_score, _ = calculate_confluence_score(confluence_signals)

    final_signal = 0
    confidence = 0.0
    p_buy = p_sell = 0.0
    model_ok = False
    predicted_class = 1
    model_conf = 0.0
    ga = None
    exit_params = None

    if mode == "indicators":
        final_signal = int(decide_indicator_signals([confluence_score], MIN_CONFLUENCE_SCORE)[0])
        confidence = float(min(abs(confluence_score) / 5.0, 1.0)) if final_signal != 0 else 0.0
        if final_signal != 0:
            decision_reason = f"Indicators: score {confluence_score:+.1f} (min ±{MIN_CONFLUENCE_SCORE:g})"
        else:
            decision_reason = f"Indicators: score {confluence_score:+.1f} below ±{MIN_CONFLUENCE_SCORE:g}"
    elif isinstance(model, GAPatternModel):
        try:
            # df_long: کندل‌های بسته‌ی خام با پنجره‌ی بلند الگوها (live_history_bars)؛ df_primary پنجره‌ی کوتاه اندیکاتورهاست
            ga = model.decide(df_long if df_long is not None else df_primary)
        except Exception as e:
            print(f"✗ GA decision error: {e}")
            traceback.print_exc()
            ga = None
        if ga is None:
            decision_reason = "GA: not enough closed bars for pattern detection"
        else:
            model_ok = True
            p_buy, p_sell = float(ga['p_buy']), float(ga['p_sell'])
            final_signal = int(ga['signal'])
            if final_signal != 0:
                rules = ga['buy_rules'] if final_signal == 1 else ga['sell_rules']
                confidence = model_conf = max(p_buy, p_sell)
                predicted_class = 2 if final_signal == 1 else 0
                exit_params = ga.get('exit')
                # اولین قانون = پراولویت‌ترین؛ خروج معامله مال همین قانون است
                decision_reason = "GA: " + " | ".join(r.get('text') or _ga_rule_text(r) for r in rules)
                if ga.get('blocked_rules'):
                    decision_reason += f" | {_ga_breakout_text(ga['breakout'])}"
            elif ga['buy_rules'] and ga['sell_rules']:
                decision_reason = "GA: buy and sell rules both matched — no trade"
            elif ga.get('pre_break'):
                decision_reason = (f"GA: last {PRE_BREAK_MINUTES} minutes before the daily market break — "
                                   f"no new entries")
            elif len(model) == 0:
                decision_reason = (f"GA: no pattern meets the selection rule ({pattern_selection_text()}) — "
                                   f"Model mode does not trade until a retrain finds one")
            elif ga.get('blocked_rules'):
                decision_reason = f"GA: no trade — {_ga_breakout_text(ga['breakout'])}"
            else:
                seen = ", ".join(_ga_trig_name(p) for p in ga['patterns']) or "none"
                decision_reason = f"GA: no rule matched (patterns on this bar: {seen})"
    elif model is not None:
        decision_reason = "Saved model is not a GA pattern model — retraining needed"
    else:
        decision_reason = "No trained model (Model mode)"

    if publish:
        atr = float(df_primary['ATR_14'].iloc[-1]) if 'ATR_14' in df_primary.columns else 0.0
        shared_signal.update(
            signal=final_signal,
            confidence=float(confidence),
            model_conf=float(model_conf),
            confluence_score=float(confluence_score),
            model_prob=float(model_conf),
            atr=atr,
            reason=decision_reason,
            indicators={
                'signal_mode': mode,
                'model_class': predicted_class,
                'model_probs': [p_sell, (1.0 - p_buy) * (1.0 - p_sell), p_buy] if model_ok else [0.0, 1.0, 0.0],
                'p_buy_win': p_buy,
                'p_sell_win': p_sell,
                'ga_patterns': [_ga_trig_name(p) for p in ga['patterns']] if ga else [],
                'ga_pattern_dirs': [_ga_trig_dir(p) for p in ga['patterns']] if ga else [],
                'ga_rules': [r.get('text') or _ga_rule_text(r) for r in (ga['buy_rules'] + ga['sell_rules'])]
                if ga else [],
                'ga_exit': exit_params,
                **confluence_signals
            }
        )

    if verbose:
        print("\n" + "=" * 60)
        print(f"📊 SIGNAL ANALYSIS — source: {'GA PATTERN MODEL' if mode != 'indicators' else 'INDICATORS only'}")
        print("=" * 60)
        if mode == "indicators":
            parts = " ".join(f"{k}={v:+.1f}" for k, v in confluence_signals.items())
            print(f"📈 CONFLUENCE: {confluence_score:+.1f}  ({parts})")
        elif model_ok:
            seen = ", ".join(_ga_trig_name(p) for p in ga['patterns']) or "none"
            print(f"🧬 GA ({len(model)} rules): patterns on this bar: {seen}")
        else:
            print("🧬 GA MODEL: not available")
        if final_signal != 0:
            print(f"🎯 DECISION: {'🟢 BUY' if final_signal == 1 else '🔴 SELL'} | "
                  f"val win rate={confidence:.1%} | {decision_reason}")
        else:
            print(f"🎯 DECISION: ⏸️ NO TRADE | {decision_reason}")
        print("=" * 60 + "\n")

    out = (final_signal, confidence)
    if return_reason:
        out = out + (decision_reason,)
    if return_exit:
        out = out + (exit_params,)
    return out


# ========================
# Trading Functions
# ========================
_OK_RETCODES = {mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED}
_INVALID_FILL = getattr(mt5, 'TRADE_RETCODE_INVALID_FILL', 10030)


def is_bot_position(pos):
    return int(getattr(pos, 'magic', 0)) == MAGIC


def get_positions(symbol):
    """فقط پوزیشن‌های خود ربات (بر اساس magic) — به معاملات دستی دست نمی‌زنیم"""
    try:
        return [p for p in (mt5.positions_get(symbol=symbol) or []) if is_bot_position(p)]
    except Exception:
        return []


def _filling_candidates(symbol_info):
    """ترتیب حالت‌های Filling بر اساس چیزی که بروکر پشتیبانی می‌کند"""
    modes = []
    flags = int(getattr(symbol_info, 'filling_mode', 0) or 0)
    if flags & 1:
        modes.append(mt5.ORDER_FILLING_FOK)
    if flags & 2:
        modes.append(mt5.ORDER_FILLING_IOC)
    modes.append(mt5.ORDER_FILLING_RETURN)
    for m in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC):
        if m not in modes:
            modes.append(m)
    return modes


def _send_order(request, symbol_info, label):
    """ارسال سفارش با امتحان حالت‌های Filling مجاز + لاگ دقیق خطا"""
    result = None
    for filling in _filling_candidates(symbol_info):
        req = dict(request, type_filling=filling)
        result = mt5.order_send(req)
        if result is None:
            print(f"❌ {label}: order_send returned None | {mt5.last_error()}")
            return None
        if result.retcode in _OK_RETCODES:
            return result
        if result.retcode != _INVALID_FILL:
            break
    if result is not None:
        print(f"❌ {label} failed | retcode={result.retcode} | {getattr(result, 'comment', '')}")
    return result


def normalize_volume(symbol_info, lot):
    step = float(symbol_info.volume_step or 0.01)
    vmin = float(symbol_info.volume_min or step)
    vmax = float(symbol_info.volume_max or lot)
    lot = min(float(lot), vmax)
    lot = np.floor(lot / step + 1e-9) * step
    decimals = max(0, -int(np.floor(np.log10(step)))) if step < 1 else 0
    lot = round(lot, decimals)
    return lot if lot >= vmin - 1e-12 else 0.0


def calculate_lot(symbol, direction, entry_price, sl_price):
    """
    حجم معامله = LOT تنظیم‌شده، ولی اگر ریسک تا SL بیشتر از MAX_POSITION_PERCENT موجودی باشد کم می‌شود.
    اگر حتی حداقل لات بروکر ریسک مجاز را رد کند، 0 برمی‌گرداند (معامله انجام نشود).
    """
    info = mt5.symbol_info(symbol)
    account = mt5.account_info()
    if info is None or account is None:
        return 0.0

    lot = float(LOT)
    if MAX_POSITION_PERCENT > 0:
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        loss_one_lot = mt5.order_calc_profit(order_type, symbol, 1.0, entry_price, sl_price)
        if loss_one_lot is not None and loss_one_lot < 0:
            risk_money = float(account.equity) * MAX_POSITION_PERCENT / 100.0
            max_lot = risk_money / abs(loss_one_lot)
            if max_lot < lot:
                print(f"⚖️ Lot reduced {lot} → {max_lot:.3f} (risk cap {MAX_POSITION_PERCENT}% of equity)")
                lot = max_lot
    return normalize_volume(info, lot)


def open_market_order(symbol, direction, atr, comment=BOT_COMMENT, use_tp=True, sl_distance=None,
                      tp_distance=None, hold=None):
    """
    باز کردن معامله‌ی بازار با SL/TP (فاصله‌ی ثابت قیمت یا ضریب ATR، دقیقاً مثل لیبل) نسبت به قیمت واقعی ورود،
    رعایت حداقل فاصله‌ی بروکر، گرد کردن به digits و کنترل ریسک.
    use_tp=False: فقط حد ضرر روی بروکر (حالت یادگیری تقویتی: خروج را خود عامل تصمیم می‌گیرد).
    sl_distance / tp_distance: فاصله‌ی حد ضرر / حد سود به واحد قیمت (خروج قانون GA یا عامل RL)؛ None = مثل لیبل مدل؛
    tp_distance=0 یعنی بدون حد سود. hold: حداکثر نگه‌داری (کندل) برای خروج زمانی همین پوزیشن.
    """
    if bot_stop_event.is_set():
        return None
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        return None

    digits = int(info.digits)
    point = float(info.point)
    min_dist = max(int(info.trade_stops_level or 0), int(info.trade_freeze_level or 0)) * point

    tp_raw, sl_raw = barrier_distances(atr)
    if sl_distance is not None and np.isfinite(sl_distance) and sl_distance > 0:
        sl_raw = float(sl_distance)
    if tp_distance is not None:
        if np.isfinite(tp_distance) and tp_distance > 0:
            tp_raw = float(tp_distance)
        else:
            use_tp = False
    sl_dist = max(sl_raw, min_dist + point)
    tp_dist = max(tp_raw, min_dist + point)

    if direction == 1:
        order_type, price = mt5.ORDER_TYPE_BUY, float(tick.ask)
        sl, tp = price - sl_dist, price + tp_dist
    else:
        order_type, price = mt5.ORDER_TYPE_SELL, float(tick.bid)
        sl, tp = price + sl_dist, price - tp_dist
    sl, tp = round(sl, digits), round(tp, digits)
    if not use_tp:
        tp = 0.0

    lot = calculate_lot(symbol, direction, price, sl)
    if lot <= 0:
        print(f"⛔ Trade skipped: minimum lot would exceed the {MAX_POSITION_PERCENT}% risk cap")
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": str(comment)[:31],          # MT5 توضیح بلندتر از ۳۱ کاراکتر را رد می‌کند
        "type_time": mt5.ORDER_TIME_GTC,
    }
    result = _send_order(request, info, f"OPEN {'BUY' if direction == 1 else 'SELL'}")
    if result is not None and result.retcode in _OK_RETCODES:
        fill = float(getattr(result, 'price', 0) or price)
        action = "BUY" if direction == 1 else "SELL"
        tp_txt = f"{tp:.{digits}f}" if tp else "none"
        hold_txt = f" | max hold {int(hold)} bars" if hold else (" | no time limit" if hold == 0 else "")
        print(f"🟢 {action} {lot} @ {fill:.{digits}f} | SL={sl:.{digits}f} | TP={tp_txt}{hold_txt}")
        log_trade(f"OPEN_{action}", fill, lot, comment)
        order_ticket = int(getattr(result, 'order', 0) or 0)
        if hold is not None and order_ticket:
            _POSITION_HOLD[order_ticket] = int(hold)
        _ensure_stop_loss(symbol, info, order_ticket, sl_dist, tp_dist if use_tp else 0.0)
    return result


# حداکثر نگه‌داری هر پوزیشن (کندل) طبق قانون GA که آن را باز کرده؛ بعد از ری‌استارت از توضیح سفارش («h96») خوانده می‌شود
_POSITION_HOLD = {}
_HOLD_RE = re.compile(r'(?<![A-Za-z0-9])h(\d{1,4})(?![0-9])')


def position_hold_bars(pos, default=None):
    """حداکثر نگه‌داری یک پوزیشن (کندل): قانون بازکننده (۰ = بدون محدودیت زمان)، یا FUTURE_STEPS برای پوزیشن‌های بدون اطلاع"""
    for key in (int(getattr(pos, 'ticket', 0) or 0), int(getattr(pos, 'identifier', 0) or 0)):
        if key in _POSITION_HOLD:
            return _POSITION_HOLD[key]
    m = _HOLD_RE.search(str(getattr(pos, 'comment', '') or ''))
    if m:
        return max(0, int(m.group(1)))
    return FUTURE_STEPS if default is None else int(default)


def _ensure_stop_loss(symbol, info, order_ticket, sl_dist, tp_dist):
    """
    اگر بروکر SL (یا TP) را روی سفارش بازار نگذاشته باشد، بلافاصله روی همان پوزیشن تازه گذاشته می‌شود
    (نسبت به قیمت واقعی ورود). پوزیشن بدون حد ضرر در حالت RL (که TP ندارد) ریسک نامحدود است.
    """
    if not order_ticket:
        return
    try:
        digits = int(info.digits)
        for pos in get_positions(symbol):
            if order_ticket not in (int(pos.ticket), int(getattr(pos, 'identifier', 0) or 0)):
                continue
            has_sl = float(pos.sl or 0.0) > 0
            has_tp = float(pos.tp or 0.0) > 0 or tp_dist <= 0
            if has_sl and has_tp:
                continue
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            entry = float(pos.price_open)
            sl = float(pos.sl) if has_sl else round(entry - sl_dist if is_buy else entry + sl_dist, digits)
            tp = float(pos.tp or 0.0)
            if tp <= 0 and tp_dist > 0:
                tp = round(entry + tp_dist if is_buy else entry - tp_dist, digits)
            res = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": int(pos.ticket),
                                  "symbol": symbol, "sl": sl, "tp": tp, "magic": MAGIC})
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"🛡️ #{pos.ticket}: missing stops set | SL={sl:.{digits}f} | TP={tp:.{digits}f}")
            else:
                print(f"❌ #{pos.ticket}: could not set missing SL/TP | "
                      f"retcode={res.retcode if res else mt5.last_error()} — check the position manually!")
    except Exception as e:
        print(f"⚠️ Stop-loss check error: {e}")


def open_order(symbol, order_type, lot, sl, tp):
    """سازگاری با نسخه‌ی قبل: سفارش بازار با SL/TP مشخص"""
    if bot_stop_event.is_set():
        return None
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        return None
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
        "volume": normalize_volume(info, lot), "type": order_type,
        "price": float(price), "sl": round(float(sl), info.digits), "tp": round(float(tp), info.digits),
        "deviation": DEVIATION, "magic": MAGIC, "comment": BOT_COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
    }
    return _send_order(request, info, "OPEN")


def close_position(pos, reason="Signal"):
    try:
        info = mt5.symbol_info(pos.symbol)
        tick = mt5.symbol_info_tick(pos.symbol)
        if info is None or tick is None:
            return None
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": float(pos.volume),
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": int(pos.ticket),
            "price": float(tick.bid if is_buy else tick.ask),
            "deviation": DEVIATION,
            "magic": MAGIC,
            "comment": f"{BOT_COMMENT}:{reason}"[:31],
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = _send_order(request, info, f"CLOSE #{pos.ticket}")
        if result is not None and result.retcode in _OK_RETCODES:
            print(f"🔴 Closed #{pos.ticket} ({'BUY' if is_buy else 'SELL'}) | {reason} | "
                  f"P&L≈${float(pos.profit):+.2f}")
        return result
    except Exception as e:
        print(f"✗ close_position error: {e}")
        return None


_say_once_last = {}


def _say_once(tag, key, msg):
    """
    پیام تکراری (اسپرد زیاد، قفل روزانه، ...) در حالت تصمیم لحظه‌ای هر چند ثانیه تکرار می‌شد؛
    فقط وقتی key عوض شود (مثلاً کندل جدید) چاپ شود.
    """
    if _say_once_last.get(tag) != key:
        _say_once_last[tag] = key
        print(msg)


_ENTRY_RETRY_SECONDS = 30
_last_failed_entry = {'t': 0.0}


def _open_entry(direction, atr, comment, use_tp=True, sl_distance=None, tp_distance=None, hold=None):
    """
    باز کردن معامله؛ اگر بروکر سفارش را رد کرد (requote، پول ناکافی، ...) تا _ENTRY_RETRY_SECONDS ثانیه
    دوباره امتحان نکن (در تصمیم لحظه‌ای وگرنه هر چند ثانیه سفارش ردشده تکرار می‌شد).
    خروجی: True اگر معامله باز شد.
    """
    if time.time() - _last_failed_entry['t'] < _ENTRY_RETRY_SECONDS:
        return False
    res = open_market_order(SYMBOL, direction, atr, comment=comment, use_tp=use_tp, sl_distance=sl_distance,
                            tp_distance=tp_distance, hold=hold)
    ok = res is not None and res.retcode in _OK_RETCODES
    if not ok:
        _last_failed_entry['t'] = time.time()
    return ok


def entry_max_delay(tf_seconds):
    """
    حداکثر تأخیر (ثانیه) بین بسته شدن کندل تصمیم و اجرای ورود. بیشتر از این یعنی بازار بین بسته شدن کندل و
    الان بسته بوده (آخر هفته، توقف روزانه، قطعی): قیمت ورود دیگر Close آن کندل نیست (فرض لیبل‌ها/شبیه‌ساز).
    """
    return max(90, int(tf_seconds) // 2)


def _stale_entry_note(bar, delay):
    _say_once('entry', ('stale', bar),
              f"⏸️ Signal bar closed {int(delay)}s ago (market was closed in between) — "
              f"exits allowed, no new entry on this bar")


def execute_signal(signal, confidence, atr, daily_locked, spread_now, bar=None, allow_entry=True, delay=0,
                   exit_params=None):
    """
    اجرای سیگنال حالت مدل/اندیکاتور: بستن پوزیشن مخالف (Reverse)، سپس ورود.
    bar: کندلی که تصمیم در آن گرفته شده (پیام‌های تکراری در هر کندل فقط یک بار چاپ می‌شوند)
    allow_entry=False: سیگنال کهنه است (entry_max_delay)؛ فقط بستن پوزیشن مخالف، بدون ورود.
    exit_params: حد سود/ضرر خود الگو (tp_units/sl_units به واحد ATR یا دلار، hold کندل؛ 0 = بدون محدودیت زمان) و
                 reverse=False یعنی پوزیشن باز با سیگنال مخالف بسته نمی‌شود؛ None = TP/SL و افق تنظیمات.
    خروجی: (acted, opened) — acted یعنی پوزیشنی بسته یا باز شد، opened یعنی پوزیشن جدید باز شد.
    """
    acted = opened = False
    if signal == 0:
        return acted, opened
    if exit_params is None or exit_params.get('reverse', True):
        for pos in get_positions(SYMBOL):
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            if (is_buy and signal == -1) or (not is_buy and signal == 1):
                res = close_position(pos, reason="Reverse")
                acted = acted or (res is not None and res.retcode in _OK_RETCODES)
    if not allow_entry:
        _stale_entry_note(bar, delay)
        return acted, opened
    if len(get_positions(SYMBOL)) >= MAX_POSITIONS:
        return acted, opened
    if daily_locked:
        _say_once('entry', ('locked', bar), "🛑 Signal ignored — daily loss limit reached")
        return acted, opened
    if MAX_SPREAD_ATR > 0 and atr > 0 and spread_now > MAX_SPREAD_ATR * atr:
        _say_once('entry', ('spread', bar),
                  f"⏸️ Spread too wide ({spread_now:.5g} > {MAX_SPREAD_ATR}×ATR) — entry skipped")
        return acted, opened
    if not np.isfinite(atr) or atr <= 0:
        _say_once('entry', ('atr', bar), "⏸️ Invalid ATR — entry skipped")
        return acted, opened
    if exit_params:
        # حد سود/ضرر آماری الگو: در حالت ATR ضریب ATR همین کندل، در حالت fixed مستقیماً دلار قیمت
        mult = float(atr) if exit_params.get('unit', 'atr') == 'atr' else 1.0
        hold = exit_params.get('hold')
        hold = int(FUTURE_STEPS if hold is None else hold)       # 0 = بدون محدودیت زمان
        ok = _open_entry(signal, atr, f"{BOT_COMMENT} h{hold} c={confidence:.2f}",
                         sl_distance=float(exit_params['sl_units']) * mult,
                         tp_distance=float(exit_params['tp_units']) * mult, hold=hold)
    else:
        ok = _open_entry(signal, atr, f"{BOT_COMMENT} c={confidence:.2f}")
    if ok:
        acted = opened = True
    return acted, opened


_DEAL_REASON_NAMES = {
    getattr(mt5, 'DEAL_REASON_SL', 4): "Stop Loss",
    getattr(mt5, 'DEAL_REASON_TP', 5): "Take Profit",
    getattr(mt5, 'DEAL_REASON_SO', 6): "Stop Out",
    getattr(mt5, 'DEAL_REASON_EXPERT', 3): "Bot",
    getattr(mt5, 'DEAL_REASON_CLIENT', 0): "Manual",
}


_close_log_retries = {}      # تیکت → تعداد دفعاتی که معامله‌ی بستنش هنوز در تاریخچه نبوده
_CLOSE_LOG_MAX_RETRIES = 60


def check_closed_positions(previous_tickets):
    """پوزیشن‌های ربات که از دور قبل بسته شده‌اند را با سود خالص (با کمیسیون و سواپ) لاگ کن"""
    try:
        raw = mt5.positions_get(symbol=SYMBOL)
        if raw is None:
            # خطای موقت MT5 (نه «هیچ پوزیشنی»): اگر اینجا لیست خالی فرض شود همه‌ی تیکت‌ها از ردیابی
            # حذف می‌شوند و وقتی واقعاً بسته شوند هیچ‌وقت لاگ نمی‌شوند
            return previous_tickets
        current_tickets = {p.ticket for p in raw if is_bot_position(p)}
        pending = set()
        for ticket in previous_tickets - current_tickets:
            deals = mt5.history_deals_get(position=ticket) or []
            out_deals = [d for d in deals if d.entry != mt5.DEAL_ENTRY_IN]
            if not out_deals:
                # معامله‌ی بستن هنوز به تاریخچه‌ی ترمینال نرسیده: دورهای بعد دوباره امتحان کن
                tries = _close_log_retries.get(ticket, 0) + 1
                if tries < _CLOSE_LOG_MAX_RETRIES:
                    _close_log_retries[ticket] = tries
                    pending.add(ticket)
                else:
                    _close_log_retries.pop(ticket, None)
                    print(f"⚠️ Closed position #{ticket} not found in trade history — not logged")
                continue
            _close_log_retries.pop(ticket, None)
            pnl = sum(float(d.profit) + float(d.commission) + float(d.swap) + float(getattr(d, 'fee', 0.0))
                      for d in deals)
            last = out_deals[-1]
            action = "CLOSE_BUY" if last.type == mt5.DEAL_TYPE_SELL else "CLOSE_SELL"
            reason = _DEAL_REASON_NAMES.get(int(getattr(last, 'reason', -1)), "Other")
            log_trade(action, float(last.price), float(sum(d.volume for d in out_deals)), reason, pnl)
            print(f"📝 Logged: {action} #{ticket} | {reason} | Net P&L: ${pnl:+.2f}")
        return current_tickets | pending
    except Exception as e:
        print(f"Check closed error: {e}")
        return previous_tickets


_log_lock = threading.Lock()


def log_trade(action, price, volume, reason="", pnl=0.0):
    try:
        entry = pd.DataFrame([{
            'Date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'Action': action,
            'Price': price,
            'Volume': volume,
            'Reason': reason,
            'PnL': pnl
        }])
        with _log_lock:
            os.makedirs(os.path.dirname(LOG_CSV) or ".", exist_ok=True)
            exists = os.path.exists(LOG_CSV)
            entry.to_csv(LOG_CSV, mode='a' if exists else 'w', header=not exists, index=False)
    except Exception as e:
        print(f"⚠️ Trade log error: {e}")


# ═══════════════════════════════════════════════════════════════
# RISK: حد ضرر روزانه + خروج زمانی
# ═══════════════════════════════════════════════════════════════
def _server_now(symbol):
    tick = mt5.symbol_info_tick(symbol)
    return int(tick.time) if tick and tick.time else None


def get_daily_pnl(symbol):
    """
    سود/زیان امروز ربات (بسته‌شده + شناور) و موجودی ابتدای روز.
    «امروز» بر اساس ساعت سرور بروکر است.
    """
    server_now = _server_now(symbol)
    account = mt5.account_info()
    if server_now is None or account is None:
        return None

    day_start = server_now - server_now % 86400
    date_from = datetime.fromtimestamp(day_start, tz=timezone.utc)
    date_to = datetime.fromtimestamp(server_now + 86400, tz=timezone.utc)

    deals = mt5.history_deals_get(date_from, date_to) or []
    realized = sum(
        float(d.profit) + float(d.commission) + float(d.swap) + float(getattr(d, 'fee', 0.0))
        for d in deals
        if int(getattr(d, 'magic', 0)) == MAGIC and d.symbol == symbol and int(d.time) >= day_start
    )
    floating = sum(float(p.profit) + float(p.swap) for p in get_positions(symbol))

    start_balance = float(account.balance) - realized
    return {'realized': realized, 'floating': floating, 'total': realized + floating,
            'start_balance': start_balance, 'day': day_start}


class DailyRiskGuard:
    """وقتی زیان امروز به MAX_DAILY_LOSS_PERCENT برسد: پوزیشن‌های ربات بسته و ورود تا فردا متوقف می‌شود"""

    CLOSE_RETRY_SECONDS = 30

    def __init__(self):
        self.locked_day = None
        self._last_close_try = 0.0

    def check(self, symbol):
        if MAX_DAILY_LOSS_PERCENT <= 0:
            return False
        stats = get_daily_pnl(symbol)
        if stats is None or stats['start_balance'] <= 0:
            return self.locked_day is not None

        if self.locked_day is not None and self.locked_day != stats['day']:
            print("🔓 New trading day — daily loss lock released")
            self.locked_day = None

        loss_pct = -stats['total'] / stats['start_balance'] * 100.0
        if self.locked_day is None and loss_pct >= MAX_DAILY_LOSS_PERCENT:
            self.locked_day = stats['day']
            self._last_close_try = 0.0
            print(f"🛑 DAILY LOSS LIMIT HIT: {loss_pct:.2f}% ≥ {MAX_DAILY_LOSS_PERCENT}% "
                  f"(realized ${stats['realized']:+.2f}, floating ${stats['floating']:+.2f}). "
                  f"Closing bot positions, no new trades until tomorrow.")
        if self.locked_day is not None and time.time() - self._last_close_try >= self.CLOSE_RETRY_SECONDS:
            # تا پایان روز هیچ پوزیشن رباتی نباید باز بماند؛ اگر بستن قبلی رد شده بود
            # (requote، بازار بسته، قطعی) دوباره امتحان کن — نه فقط یک بار در لحظه‌ی قفل شدن
            positions = get_positions(symbol)
            if positions:
                self._last_close_try = time.time()
                for pos in positions:
                    close_position(pos, reason="DailyLoss")
        return self.locked_day is not None


def bars_held_since(pos, bar_times, now=None, tf_seconds=None):
    """
    چند کندل از ورود پوزیشن گذشته.
    now = زمان فعلی سرور (tick.time): «زمان بازار» به واحد کندل و کسری — باقی‌مانده‌ی کندل ورود +
      کندل‌های کامل بعد از آن + بخش گذشته‌ی کندل جاری. فقط کندل‌هایی که واقعاً وجود دارند شمرده می‌شوند،
      پس تعطیلی آخر هفته حساب نمی‌شود. با ورود وسط کندل (تصمیم لحظه‌ای) هم دقیق است.
    now = None: رفتار قبلی — تعداد کندل‌های bar_times از کندل ورود به بعد (خود کندل ورود = ۱).
    tf_seconds: طول کندل‌های bar_times (پیش‌فرض تایم‌فریم مدل؛ عامل RL تایم‌فریم خودش را می‌دهد).
    """
    tf = int(tf_seconds or PRIMARY_TF_SECONDS)
    bar_times = np.asarray(bar_times, dtype=np.int64)
    t_entry = int(pos.time)
    entry_bar = t_entry - t_entry % tf
    if now is None:
        return int(np.sum(bar_times >= entry_bar))
    now = int(now)
    cur_bar = now - now % tf
    if cur_bar <= entry_bar:
        return max(0.0, (now - t_entry) / tf)
    held = (entry_bar + tf - t_entry) / tf                                     # باقی‌مانده‌ی کندل ورود
    held += int(np.sum((bar_times > entry_bar) & (bar_times < cur_bar)))       # کندل‌های کامل بعدی
    held += (now - cur_bar) / tf                                               # بخش گذشته‌ی کندل جاری
    return held


def apply_time_exit(symbol, bar_times, max_bars=None, force=False, now=None, tf_seconds=None):
    """
    بستن پوزیشن‌هایی که max_bars کندل از ورودشان گذشته است.
    max_bars=None: حداکثر نگه‌داری خود هر پوزیشن (قانون GA که آن را باز کرده؛ position_hold_bars) — همان افقی که
    قانون رویش انتخاب و سنجیده شده. شمارش بر اساس کندل‌های موجود است نه زمان ساعت، تا تعطیلی آخر هفته پوزیشن را
    زودتر نبندد (با now: زمان بازارِ کسری، مناسب ورود وسط کندل؛ بدون now: تعداد کندل‌های بسته).
    force=True: حتی اگر USE_TIME_EXIT خاموش باشد (حالت یادگیری تقویتی: حداکثر نگه‌داری جزء قانون عامل است).
    """
    if (not USE_TIME_EXIT and not force) or bar_times is None or len(bar_times) == 0:
        return
    for pos in get_positions(symbol):
        limit = position_hold_bars(pos) if max_bars is None else int(max_bars)
        if limit <= 0:
            continue                                  # الگوی بدون محدودیت زمان: فقط حد سود/ضرر
        if bars_held_since(pos, bar_times, now, tf_seconds) >= limit:
            close_position(pos, reason="TimeExit")
            _POSITION_HOLD.pop(int(pos.ticket), None)


# ========================
# Helper Functions برای فایل آموزش
# ========================
def _read_train_file(last_train_file):
    data = {}
    if not os.path.exists(last_train_file):
        return data
    try:
        with open(last_train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip()] = value.strip()
    except Exception as e:
        print(f"⚠️ Error reading train file: {e}")
    return data


def _parse_dt(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def active_last_train_file():
    """فایل وضعیت آموزشِ منبع سیگنال فعلی (مدل یا عامل یادگیری تقویتی)"""
    return rl_last_train_file() if SIGNAL_MODE == "rl" else LAST_TRAIN_FILE


def get_last_train_time(last_train_file=None):
    """آخرین آموزش موفق (مدل ذخیره‌شده)"""
    data = _read_train_file(last_train_file or active_last_train_file())
    return _parse_dt(data.get("Last Successful Fine-tune"))


def get_last_train_attempt(last_train_file=None):
    """آخرین تلاش آموزش (موفق یا ناموفق)"""
    data = _read_train_file(last_train_file or active_last_train_file())
    return _parse_dt(data.get("Last Attempt")) or _parse_dt(data.get("Last Successful Fine-tune"))


def save_train_time(success=True, message="", last_train_file=None, model_path=None, signature=None):
    """ذخیره وضعیت آموزش؛ زمان آخرین موفقیت در شکست‌های بعدی حفظ می‌شود"""
    last_train_file = last_train_file or LAST_TRAIN_FILE
    model_path = model_path or MODEL_PATH
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_success = _read_train_file(last_train_file).get("Last Successful Fine-tune")

    lines = []
    if success:
        lines.append(f"Last Successful Fine-tune: {now_str}")
    elif previous_success:
        lines.append(f"Last Successful Fine-tune: {previous_success}")
    lines.append(f"Last Attempt: {now_str}")
    lines.append(f"Status: {'Success' if success else 'Failed'}")
    lines.append(f"Model Path: {model_path}")
    lines.append(f"Signature: {signature or training_signature()}")
    if message:
        lines.append(f"{'Details' if success else 'Error'}: {message}")

    try:
        tmp = last_train_file + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, last_train_file)
        print(f"📝 Train status saved to {os.path.basename(last_train_file)}")
        return True
    except Exception as e:
        print(f"❌ Failed to save train time: {e}")
        return False


def should_train_now(last_train_file=None, signature=None, interval_hours=None):
    """
    آیا وقت آموزش رسیده؟ فاصله از آخرین «تلاش» حساب می‌شود،
    تا یک آموزش ناموفق باعث آموزش‌های پشت‌سرهم (هر ۵ دقیقه) نشود.
    اگر آخرین تلاش با تنظیمات/نسخه‌ی دیگری بوده (مثلاً فیچرها یا TP/SL عوض شده)، همین الان.
    """
    data = _read_train_file(last_train_file or LAST_TRAIN_FILE)
    last_attempt = _parse_dt(data.get("Last Attempt")) or _parse_dt(data.get("Last Successful Fine-tune"))
    if last_attempt is None:
        print("📊 No previous training found. Need initial training.")
        return True

    if data.get("Signature") != (signature or training_signature()):
        print("📊 Model setup changed since the last training attempt — training needed now.")
        return True

    hours_passed = (datetime.now() - last_attempt).total_seconds() / 3600
    interval = FINETUNE_INTERVAL_HOURS if interval_hours is None else interval_hours
    if hours_passed >= interval:
        print(f"✅ Time for {'RL retraining' if interval_hours is not None else 'fine-tuning'}! "
              f"({hours_passed:.1f} >= {interval:g} hours)")
        return True
    return False


# ═══════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING — عامل اسکلپ (Dueling Double DQN با n-step و چند عامل مستقل)
# ═══════════════════════════════════════════════════════════════
# زمان‌بندی دقیقاً مثل ربات زنده:
#   تصمیم فقط روی کندل بسته‌ی تایم‌فریم اصلی (مثلاً M1) و اجرا با اولین قیمت کندل بعد (Open).
#   خرید با Ask (= Bid + اسپرد) باز و با Bid بسته می‌شود؛ فروش با Bid باز و با Ask بسته می‌شود.
# v7 (2026-10-16، به خواست کاربر): حد ضرر و حد سود از پیش داده نمی‌شوند — عامل هنگام ورود خودش انتخاب می‌کند:
#   اکشن‌ها: FLAT (بیرون ماندن / بستن)، HOLD (نگه داشتن پوزیشن)، و ورود = جهت × حد ضرر × حد سود
#   (RL_SL_CHOICES / RL_TP_CHOICES ضریب ATR تایم‌فریم اصلی؛ حد سود 0 = بدون حد سود، بستن با خود عامل).
#   هر کندل هم می‌تواند خودش ببندد یا برعکس کند. حد ضرر و حد سود انتخاب‌شده روی بروکر ثبت می‌شوند (حد ضرر همیشه هست:
#   محافظ قطعی اینترنت؛ گزینه‌ی خیلی دور یعنی «خودم مدیریت می‌کنم»).
#   شبیه‌ساز: لمس حد ضرر/سود داخل کندل، گپ اول کندل با قیمت Open، و اگر هر دو در یک کندل لمس شوند حد ضرر (بدبینانه).
#   پاداش هر کندل = تغییر سود/زیان (با اسپرد، لغزش و هزینه‌ی اضافه) ÷ میانه‌ی ATR (۵۰۰ کندل) هنگام ورود — متناسب با سود
#   دلاری با لات ثابت، و مستقل از حد ضرری که خود عامل انتخاب می‌کند.
#   حداکثر نگه‌داری RL_MAX_HOLD کندل؛ ورود وقتی اسپرد > MAX_SPREAD_ATR × ATR ممنوع است (مثل ربات زنده).
# ورودی عامل: کندل‌های خام اصلی و دوم نسبت به کلوز فعلی ÷ ATR، بازده چندمقیاسی، ساختار رنج، رژیم نوسان،
# اسپرد، حجم تیکی، ساعت روز و روز هفته، شکاف زمانی و زمینه‌ی مدل الگو؛ وضعیت پوزیشن: جهت، سود شناور، کندل نگه‌داری،
# حد ضرر و حد سود همین معامله.
# یک تابع ورودی (_rl_features) هم برای آموزش و هم برای تصمیم زنده استفاده می‌شود تا هر دو دقیقاً یکی باشند.
# شبکه: MLP با سر Q «فاکتورشده» (NumPy، بدون کتابخانه‌ی جدید): Q(ورود) = V + مزیت جهت + مزیت حد ضرر + مزیت حد سود —
# هر چیزی که درباره‌ی یک حد ضرر یاد می‌گیرد به همه‌ی حد سودها هم می‌رسد (نمونه‌ی کمتر لازم، بیش‌برازش کمتر).
# Double DQN با ماسک اکشن‌های مجاز، بازده n-step، چند عامل مستقل (میانگین Q)؛ حداقل برتری ورود و ضریب احتیاط روی
# Validation تنظیم می‌شوند. بعد از سنجش روی Test، عامل‌ها با همان تنظیمات روی کل داده (تا آخرین کندل) دوباره آموزش
# می‌بینند (RL_REFIT_ALL_DATA). عامل جدید همیشه جایگزین می‌شود، مگر Test آن ضررده باشد (آن وقت عامل فعلی حداکثر
# RL_MAX_KEEPS بار پشت‌سرهم نگه داشته می‌شود).

# v4: ورودی «کندل‌های نگه‌داری» در شبیه‌ساز بدون پوزیشن صفر است (مثل ربات زنده) — عامل قبلی دوباره آموزش می‌بیند
# v5: ورودی زمینه‌ی مدل الگو، ارزیابی کامپایل‌شده (numba) و ضریب احتیاط گروه عامل‌ها (kappa)
# v6: فهرست الگوهای مدل عوض شد (شمارش الگوهای صعودی/نزولی در ورودی زمینه)
# v7: حد ضرر/سود انتخاب خود عامل، سر Q فاکتورشده، پاداش به واحد ATR، لغزش، آموزش دوباره روی کل داده، قانون جایگزینی تازه
# v8: ورودی الگوهای مدل روی M5 (کدام الگو، نه فقط تعدادشان) + ویژگی‌های M5 با روند M30
RL_VERSION = 8
RL_WINDOW_P = 16          # چند کندل آخر تایم‌فریم اصلی (خام)
RL_WINDOW_S = 12          # چند کندل آخر تایم‌فریم دوم (خام)
RL_RET_LAGS = (1, 3, 5, 10, 20, 60)
RL_RAW_WARMUP = 600       # کندل‌های اول که ATR و رژیم نوسانشان هنوز کامل نیست (استفاده نمی‌شوند)
RL_VOL_WINDOW = 500       # پنجره‌ی میانه‌ی ATR برای رژیم نوسان (و واحد پاداش)
RL_OBS_CLIP = 15.0
# ورودی زمینه: همان ۳۰ ویژگی عددی مدل الگو (RSI، ADX، روند و رنج، سقف/کف روز قبل، تایم‌فریم بالاتر، ...)
# + تعداد الگوهای صعودی و نزولی روی همان کندل — از ga_primitives، یکی در آموزش و تصمیم زنده.
# (خاموش کردنش امضای عامل را عوض می‌کند و آموزش دوباره لازم می‌شود)
RL_CONTEXT_FEATURES = True
RL_CONTEXT_SCALE = {'rsi14': 50.0, 'rsi2': 50.0, 'adx': 50.0}     # تقسیم بر این مقدار تا در بازه‌ی حدود ±۱ باشند
# فهرست ثابت ویژگی‌های زمینه‌ی عامل (همان ۳۰ ویژگی نسخه‌ی قبل، به همین ترتیب): ویژگی‌هایی که بعداً به مدل الگو اضافه
# می‌شوند (v9: روند M30) ورودی عامل را عوض نمی‌کنند
RL_CONTEXT_NUM = (
    'rsi14', 'rsi2', 'ret5', 'ret20', 'ret60', 'sma20_dist', 'sma50_dist', 'sma200_dist', 'sma20_slope', 'sma50_slope',
    'pos20', 'pos100', 'pos_1d', 'pos_5d', 'eff20', 'body', 'clv', 'streak', 'pdh_dist', 'pdl_dist', 'dopen_dist',
    'htf_sma_dist', 'htf_pos', 'htf_slope', 'atr_ratio', 'range', 'contraction', 'bb_width', 'adx', 'day_range',
)
RL_N_CONTEXT = (len(RL_CONTEXT_NUM) + 2) if RL_CONTEXT_FEATURES else 0
# v8 (2026-10-16، به خواست کاربر): ورودی الگوهای مدل — همان الگوهای مدل الگو (۷۳ جفت صعودی/نزولی) روی کندل‌های بسته‌ی
# تایم‌فریم دوم عامل (M5) و ویژگی‌های عددی M5 با روند M30 (همان چیزی که حالت Model می‌بیند). هر جفت یک ورودی: +1 الگوی
# صعودی، −1 الگوی نزولی روی آخرین کندل بسته‌ی M5، با اثر کم‌شونده‌ی دو کندل M5 قبل‌تر (×0.5، ×0.25). عامل خودش یاد
# می‌گیرد بعد از کدام الگو وارد شود و با ظاهر شدن کدام الگو (مثلاً الگوی مخالف) معامله را ببندد.
RL_PATTERN_INPUTS = True
RL_PATTERN_AGE = 3
RL_PATTERN_DECAY = 0.5
RL_PATTERN_NUM = tuple(GA_NUM_FEATURES)
RL_N_PATTERN_FLAGS = len(GA_PATTERN_PAIRS) if RL_PATTERN_INPUTS else 0
RL_N_PATTERN = (RL_N_PATTERN_FLAGS + len(RL_PATTERN_NUM)) if RL_PATTERN_INPUTS else 0
RL_N_MARKET_BASE = RL_WINDOW_P * 4 + RL_WINDOW_S * 4 + len(RL_RET_LAGS) + 12 + RL_N_CONTEXT
RL_N_MARKET = RL_N_MARKET_BASE + RL_N_PATTERN
RL_RAW_COLS = slice(RL_N_MARKET_BASE, RL_N_MARKET_BASE + RL_N_PATTERN_FLAGS)   # ±1 الگوها بدون استانداردسازی

# ── اکشن‌ها (v7): عامل حد ضرر و حد سود هر معامله را خودش انتخاب می‌کند ──
RL_SL_CHOICES = (0.5, 1.0, 2.0, 4.0, 8.0)   # حد ضرر = ضریب ATR کندل تصمیم (هیچ‌وقت کمتر از RL_MIN_SL_SPREADS × اسپرد)
RL_TP_CHOICES = (0.0, 0.5, 1.0, 2.0, 4.0)   # حد سود = ضریب ATR کندل تصمیم؛ 0 = بدون حد سود (خروج با خود عامل)
RL_MIN_SL_SPREADS = 3.0   # حد ضرر هیچ‌وقت کمتر از این چند برابر اسپرد نمی‌شود (وگرنه نویز اسپرد معامله را می‌بندد)
# لغزش ورود سفارش بازار (دلار قیمت؛ میانگین اندازه‌گیری‌شده از معاملات واقعی همین حساب ≈0.14) — هزینه‌ی هر معامله در
# شبیه‌ساز، کنار اسپرد و EXTRA_COST_POINTS
RL_SLIPPAGE_PRICE = 0.14
_RL_KSL, _RL_KTP = len(RL_SL_CHOICES), len(RL_TP_CHOICES)
RL_N_ACTIONS = 2 + 2 * _RL_KSL * _RL_KTP
RL_A_FLAT, RL_A_HOLD = 0, 1
# جدول اکشن‌ها: جهت (0 = بی‌پوزیشن/نگه داشتن)، ضریب حد ضرر، ضریب حد سود، اندیس حد ضرر/سود
RL_ACT_SIDE = np.zeros(RL_N_ACTIONS, dtype=np.int64)
RL_ACT_SL = np.zeros(RL_N_ACTIONS, dtype=np.float64)
RL_ACT_TP = np.zeros(RL_N_ACTIONS, dtype=np.float64)
RL_ACT_SLI = np.full(RL_N_ACTIONS, -1, dtype=np.int64)
RL_ACT_TPI = np.full(RL_N_ACTIONS, -1, dtype=np.int64)
for _si, _side in enumerate((1, -1)):
    for _i, _ks in enumerate(RL_SL_CHOICES):
        for _j, _kt in enumerate(RL_TP_CHOICES):
            _a = 2 + _si * _RL_KSL * _RL_KTP + _i * _RL_KTP + _j
            RL_ACT_SIDE[_a], RL_ACT_SL[_a], RL_ACT_TP[_a] = _side, _ks, _kt
            RL_ACT_SLI[_a], RL_ACT_TPI[_a] = _i, _j
RL_LONG_ACTIONS = np.flatnonzero(RL_ACT_SIDE == 1)
RL_SHORT_ACTIONS = np.flatnonzero(RL_ACT_SIDE == -1)


def _rl_head_matrix():
    """
    سر Q فاکتورشده: خروجی شبکه z = [V, A_flat, A_hold, A_جهت(۲), A_حدضرر(۲×K), A_حدسود(۲×K)] و Q = z @ T.
    مزیت ورود = مزیت جهت + مزیت حد ضرر + مزیت حد سود (همان جهت)؛ Q = V + مزیت − میانگین مزیت‌ها (دوئلینگ).
    """
    K = 5 + 2 * _RL_KSL + 2 * _RL_KTP
    M = np.zeros((K - 1, RL_N_ACTIONS))
    M[0, RL_A_FLAT] = 1.0
    M[1, RL_A_HOLD] = 1.0
    for a in range(2, RL_N_ACTIONS):
        s = 0 if RL_ACT_SIDE[a] == 1 else 1
        M[2 + s, a] = 1.0
        M[4 + s * _RL_KSL + RL_ACT_SLI[a], a] = 1.0
        M[4 + 2 * _RL_KSL + s * _RL_KTP + RL_ACT_TPI[a], a] = 1.0
    M -= M.mean(axis=1, keepdims=True)
    return np.vstack([np.ones((1, RL_N_ACTIONS)), M]).astype(np.float32)


RL_HEAD_T = _rl_head_matrix()
RL_N_HEAD = RL_HEAD_T.shape[0]
# وضعیت پوزیشن: [خرید؟، فروش؟، سود شناور ÷ فاصله‌ی حد ضرر، کندل نگه‌داری ÷ حداکثر، فاصله‌ی حد ضرر ÷ ATR فعلی،
#               فاصله‌ی حد سود ÷ فاصله‌ی حد ضرر، حد سود دارد؟] — همه از پوزیشن واقعی بروکر هم قابل محاسبه‌اند
RL_POS_FEATURES = 7
RL_HIDDEN = (128, 64)
RL_GAMMA = 0.98
RL_NSTEP = 3
RL_N_ENVS = 64
RL_EPISODE_BARS = 2048
RL_BATCH = 256
RL_BUFFER = 300_000
RL_WARMUP_TRANSITIONS = 20_000
RL_UPDATES_PER_ITER = 2
RL_LR = 2e-4
RL_WEIGHT_DECAY = 0.1     # کاهش وزن جداشده (AdamW)؛ همراه با نویز ورودی و شبکه‌ی کوچک ضد بیش‌برازش
RL_TARGET_SYNC = 1000
RL_EVAL_POINTS = 12
RL_EPS_START, RL_EPS_END, RL_EPS_DECAY_FRAC = 1.0, 0.05, 0.5
RL_OBS_NOISE = 0.2       # نویز گاوسی روی ورودی‌های نرمال‌شده در هر به‌روزرسانی (ضد بیش‌برازش)
RL_PATIENCE = 5           # توقف زودهنگام: این تعداد چک پشت‌سرهم بدون بهتر شدن Validation (بعد از پایان اکتشاف)
RL_ENTRY_MARGINS = (0.0, 0.03, 0.06, 0.12, 0.25, 0.5)   # حداقل برتری Q ورود نسبت به «بیرون ماندن» (واحد ATR)
RL_ENTRY_KAPPAS = (0.0, 0.5, 1.0, 2.0)                  # ضریب احتیاط: ورود با Q منهای kappa × اختلاف نظر عامل‌ها
# دست ربات باز: margin/kappa فقط بین حالت‌هایی انتخاب می‌شوند که روی Validation روزی دست‌کم این‌قدر معامله دارند،
# و بین آن‌ها حالتی که بیشترین مجموع سود Validation را دارد
RL_MIN_TRADES_PER_DAY = 1.0
RL_MIN_BARS = RL_RAW_WARMUP + 5000                # کمترین کندل خام برای آموزش
# بعد از انتخاب تنظیمات و سنجش روی Test، هر عامل با همان seed و همان تعداد گام روی کل داده (Train + Validation + Test)
# دوباره آموزش می‌بیند، تا عامل مستقر ماه‌های آخر بازار را هم دیده باشد (عدد Test برآورد همین روش است)
RL_REFIT_ALL_DATA = True
RL_MAX_KEEPS = 3          # عامل فعلی حداکثر این تعداد آموزش پشت‌سرهم (وقتی Test عامل جدید ضررده است) نگه داشته می‌شود
RL_EXIT_NAMES = ('agent', 'stop_loss', 'end', 'take_profit', 'max_hold')


def rl_history_bars():
    """چند کندل خام برای آموزش عامل دانلود شود (کمی بیشتر از RL_DATA_BARS، ولی نه بیشتر از سقف دیتابیس)"""
    return min(DB_MAX_BARS, max(int(RL_DATA_BARS) + RL_RAW_WARMUP + 1000, HISTORY_BARS.get(RL_TF_STR, 90000)))


def rl_agent_path():
    return RL_AGENT_PATH


def rl_meta_path():
    return rl_agent_path() + ".meta.json"


def rl_last_train_file():
    return RL_LAST_TRAIN_FILE


def rl_sl_text():
    """«agent-chosen SL 0.5/1/2/4/8×ATR(M1), TP none/0.5/1/2/4×ATR»"""
    sl = "/".join(f"{x:g}" for x in RL_SL_CHOICES)
    tp = "/".join("none" if x <= 0 else f"{x:g}" for x in RL_TP_CHOICES)
    return f"agent-chosen SL {sl}×ATR({RL_TF_STR}) (≥{RL_MIN_SL_SPREADS:g}×spread), TP {tp}×ATR"


def rl_sl_distance(atr, spread, mult=None):
    """فاصله‌ی حد ضرر (واحد قیمت) با ضریب ATR انتخاب‌شده — یک فرمول برای شبیه‌ساز و ربات زنده"""
    mult = RL_SL_CHOICES[len(RL_SL_CHOICES) // 2] if mult is None else mult
    return np.maximum(float(mult) * np.asarray(atr, dtype=np.float64),
                      RL_MIN_SL_SPREADS * np.asarray(spread, dtype=np.float64))


def rl_action_exits(a, atr, spread):
    """(جهت، فاصله‌ی حد ضرر، فاصله‌ی حد سود یا 0) اکشن a با ATR و اسپرد کندل تصمیم"""
    a = int(a)
    side = int(RL_ACT_SIDE[a])
    if side == 0:
        return 0, 0.0, 0.0
    sl = float(rl_sl_distance(atr, spread, RL_ACT_SL[a]))
    tp = float(RL_ACT_TP[a]) * float(atr)
    return side, sl, tp


def rl_action_text(a, atr=None, spread=None):
    """«LONG SL 2×ATR ($3.10) TP none»"""
    a = int(a)
    if a == RL_A_FLAT:
        return "FLAT"
    if a == RL_A_HOLD:
        return "HOLD"
    side, sl, tp = rl_action_exits(a, atr or 0.0, spread or 0.0)
    sl_txt = f"SL {RL_ACT_SL[a]:g}×ATR" + (f" ({_px_txt(sl)})" if atr else "")
    tp_txt = "TP none" if RL_ACT_TP[a] <= 0 else (f"TP {RL_ACT_TP[a]:g}×ATR" + (f" ({_px_txt(tp)})" if atr else ""))
    return f"{'LONG' if side == 1 else 'SHORT'} {sl_txt} {tp_txt}"


def rl_signature():
    """هر چیزی که معنی محیط/ورودی عامل را عوض کند؛ عامل فقط با همین امضا معتبر است"""
    sl = ",".join(f"{x:g}" for x in RL_SL_CHOICES)
    tp = ",".join(f"{x:g}" for x in RL_TP_CHOICES)
    return (f"rl{RL_VERSION}|scalp|{RL_TF_STR}+{RL_SECONDARY_TF_STR}|sl={sl}|tp={tp}|minsl={RL_MIN_SL_SPREADS:g}sp"
            f"|hold={RL_MAX_HOLD}|obs={RL_N_MARKET}:{RL_WINDOW_P}/{RL_WINDOW_S}|pos={RL_POS_FEATURES}"
            f"|xc={EXTRA_COST_POINTS}|slip={RL_SLIPPAGE_PRICE:g}|pat={RL_N_PATTERN}:{_rl_pattern_tag()}")


def _rl_pattern_tag():
    """نشان کوتاه فهرست الگوها و ویژگی‌های ورودی عامل (عوض شدن فهرست = آموزش دوباره)"""
    if not RL_PATTERN_INPUTS:
        return "off"
    names = "|".join(a + "/" + b for a, b in GA_PATTERN_PAIRS) + "#" + "|".join(RL_PATTERN_NUM)
    h = 0
    for ch in names.encode('utf-8'):
        h = (h * 131 + ch) % 1_000_000_007
    return f"{RL_PATTERN_AGE}x{RL_PATTERN_DECAY:g}-{h}-g{GA_VERSION}"     # تعریف الگوها عوض شد = آموزش دوباره


def _rl_valid_mask(pos, can_open, expired=None):
    """
    اکشن‌های مجاز هر حالت (n, RL_N_ACTIONS): FLAT همیشه؛ HOLD فقط با پوزیشن؛ ورود فقط در جهت دیگر (بدون پوزیشن: هر دو
    جهت؛ با پوزیشن: برعکس کردن) و فقط وقتی اسپرد اجازه می‌دهد؛ بعد از حداکثر نگه‌داری (expired) فقط FLAT.
    """
    pos = np.asarray(pos).reshape(-1)
    n = len(pos)
    can_open = np.broadcast_to(np.asarray(can_open, dtype=bool), (n,))
    expired = np.zeros(n, dtype=bool) if expired is None else np.broadcast_to(np.asarray(expired, dtype=bool), (n,))
    m = np.zeros((n, RL_N_ACTIONS), dtype=bool)
    m[:, RL_A_FLAT] = True
    m[:, RL_A_HOLD] = (pos != 0) & ~expired
    m[:, RL_LONG_ACTIONS] = ((pos != 1) & can_open & ~expired)[:, None]
    m[:, RL_SHORT_ACTIONS] = ((pos != -1) & can_open & ~expired)[:, None]
    return m


def _rl_mask_from_pf(pf, can_open):
    """ماسک اکشن‌های مجاز از روی ورودی وضعیت پوزیشن (برای هدف Double DQN)"""
    pos = np.where(pf[:, 0] > 0.5, 1, np.where(pf[:, 1] > 0.5, -1, 0))
    expired = (pos != 0) & (pf[:, 3] >= 1.0 - 1e-6)
    return _rl_valid_mask(pos, can_open, expired)


# ─────────────────────────── شبکه‌ی Q (NumPy) ───────────────────────────
class NumpyQNet:
    """
    MLP با تنه‌ی مشترک ReLU و یک سر خطی z که با ماتریس ثابت T به Q همه‌ی اکشن‌ها تبدیل می‌شود (Q = z @ T):
    v7: سر فاکتورشده‌ی RL_HEAD_T (V + مزیت جهت + مزیت حد ضرر + مزیت حد سود، دوئلینگ).
    آموزش با Huber loss، Adam، محدودیت نرم گرادیان و کاهش وزن جداشده (AdamW).
    """

    def __init__(self, in_dim, hidden=None, head=None, seed=0):
        rng = np.random.default_rng(seed)
        hidden = RL_HIDDEN if hidden is None else hidden
        self.T = np.asarray(RL_HEAD_T if head is None else head, dtype=np.float32)
        dims = [int(in_dim), *[int(h) for h in hidden]]
        self.W = [(rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i])).astype(np.float32)
                  for i in range(len(dims) - 1)]
        last = dims[-1]
        self.W.append((rng.standard_normal((last, self.T.shape[0])) * np.sqrt(1.0 / last) * 0.1).astype(np.float32))
        self.b = [np.zeros(w.shape[1], dtype=np.float32) for w in self.W]
        self._reset_opt()

    def _reset_opt(self):
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0

    @property
    def n_trunk(self):
        return len(self.W) - 1

    @property
    def in_dim(self):
        return self.W[0].shape[0]

    @property
    def n_actions(self):
        return self.T.shape[1]

    def q_head(self):
        """(W, b) لایه‌ی آخر به شکل مستقیم Q: h @ Wq + bq"""
        return self.W[-1] @ self.T, self.b[-1] @ self.T

    def forward(self, x, cache=False):
        h = np.asarray(x, dtype=np.float32)
        acts = [h]
        for k in range(self.n_trunk):
            h = np.maximum(h @ self.W[k] + self.b[k], 0.0)
            acts.append(h)
        q = (h @ self.W[-1] + self.b[-1]) @ self.T
        return (q, acts) if cache else q

    def grads(self, x, actions, targets, delta=1.0):
        """Huber loss و گرادیان همه‌ی پارامترها (W..., b...) برای Q(x, action) → target"""
        q, acts = self.forward(x, cache=True)
        n = q.shape[0]
        idx = np.arange(n)
        err = q[idx, actions] - targets
        abs_err = np.abs(err)
        loss = float(np.mean(np.where(abs_err <= delta, 0.5 * err ** 2, delta * (abs_err - 0.5 * delta))))
        g = np.zeros_like(q)
        g[idx, actions] = np.clip(err, -delta, delta) / n
        g_z = g @ self.T.T                                     # Q = z @ T
        h = acts[-1]
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        grads_W[-1], grads_b[-1] = h.T @ g_z, g_z.sum(axis=0)
        gh = g_z @ self.W[-1].T
        for k in reversed(range(self.n_trunk)):
            gh = gh * (acts[k + 1] > 0)
            grads_W[k] = acts[k].T @ gh
            grads_b[k] = gh.sum(axis=0)
            if k > 0:
                gh = gh @ self.W[k].T
        return loss, grads_W + grads_b

    def train_step(self, x, actions, targets, lr=None, delta=1.0, clip=10.0, weight_decay=None):
        lr = RL_LR if lr is None else lr
        weight_decay = RL_WEIGHT_DECAY if weight_decay is None else weight_decay
        loss, grads = self.grads(x, actions, targets, delta)
        norm = float(np.sqrt(sum(float((gr.astype(np.float64) ** 2).sum()) for gr in grads)))
        scale = min(1.0, clip / (norm + 1e-12))
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p, gr, m, v in zip(self.W + self.b, grads, self._m, self._v):
            gr = gr * scale
            m *= b1
            m += (1 - b1) * gr
            v *= b2
            v += (1 - b2) * gr * gr
            p -= (lr * (m / (1 - b1 ** self._t)) / (np.sqrt(v / (1 - b2 ** self._t)) + eps)).astype(np.float32)
        if weight_decay > 0:
            for W in self.W:
                W -= (lr * weight_decay) * W
        return loss

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]
        self.T = other.T.copy()

    def weights(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set_weights(self, weights):
        W, b = weights
        self.W = [w.copy() for w in W]
        self.b = [x.copy() for x in b]

    def to_arrays(self, prefix=""):
        out = {f"{prefix}W{k}": w for k, w in enumerate(self.W)}
        out.update({f"{prefix}b{k}": b for k, b in enumerate(self.b)})
        out[f"{prefix}T"] = self.T
        return out

    @classmethod
    def from_arrays(cls, arrays, prefix=""):
        n_layers = 0
        while f"{prefix}W{n_layers}" in arrays:
            n_layers += 1
        if n_layers < 2 or f"{prefix}T" not in arrays:
            raise ValueError("not a v7 agent network")
        net = cls.__new__(cls)
        net.W = [np.asarray(arrays[f"{prefix}W{k}"], dtype=np.float32) for k in range(n_layers)]
        net.b = [np.asarray(arrays[f"{prefix}b{k}"], dtype=np.float32) for k in range(n_layers)]
        net.T = np.asarray(arrays[f"{prefix}T"], dtype=np.float32)
        if net.W[-1].shape[1] != net.T.shape[0]:
            raise ValueError("head size mismatch")
        net._reset_opt()
        return net


def _q_mean(nets, x):
    """میانگین Q چند عامل (ensemble)"""
    q = nets[0].forward(x)
    for net in nets[1:]:
        q = q + net.forward(x)
    return q / len(nets)


def _q_stats(nets, x):
    """میانگین و انحراف معیار Q عامل‌ها (اختلاف نظر عامل‌ها = عدم اطمینان)"""
    qs = np.stack([net.forward(x).astype(np.float64) for net in nets])
    return qs.mean(axis=0), qs.std(axis=0)


def _rl_pick_action(q, valid, current, margin, q_std=None, kappa=0.0):
    """
    بهترین اکشن مجاز، با این شرط که باز کردن پوزیشن تازه (یا برعکس کردن) فقط وقتی انجام شود که Q آن (منهای
    kappa × اختلاف نظر عامل‌ها) دست‌کم `margin` (واحد ATR) از «بیرون ماندن» بهتر باشد؛ وگرنه بهترینِ «بیرون ماندن» و
    «نگه داشتن پوزیشن فعلی». یکی با هسته‌ی کامپایل‌شده‌ی ارزیابی (_rl_eval_kernel).
    """
    qv = np.where(valid, np.asarray(q, dtype=np.float64), -np.inf)
    a = int(np.argmax(qv))
    if RL_ACT_SIDE[a] != 0:
        doubt = float(kappa) * float(q_std[a]) if (q_std is not None and kappa) else 0.0
        if float(q[a]) - doubt - float(q[RL_A_FLAT]) < margin:
            a = RL_A_HOLD if (current != 0 and valid[RL_A_HOLD] and q[RL_A_HOLD] > q[RL_A_FLAT]) else RL_A_FLAT
    return a


def _rl_q_summary(q):
    """[Q بیرون، بهترین Q خرید، بهترین Q فروش] — برای نمایش (سازگار با داشبورد قبلی)"""
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[RL_A_FLAT], q[RL_LONG_ACTIONS].max(), q[RL_SHORT_ACTIONS].max()])


class RLAgent:
    """چند شبکه‌ی آموزش‌دیده (میانگین Q) + نرمال‌سازی ورودی + حداقل برتری ورود و ضریب احتیاط (kappa)"""

    def __init__(self, qnet, mu, sd, meta=None):
        self.nets = list(qnet) if isinstance(qnet, (list, tuple)) else [qnet]
        self.qnet = self.nets[0]
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sd = np.asarray(sd, dtype=np.float32)
        self.meta = dict(meta or {})

    @property
    def entry_margin(self):
        return float(self.meta.get('entry_margin', 0.0) or 0.0)

    @property
    def kappa(self):
        return float(self.meta.get('kappa', 0.0) or 0.0)

    def _x(self, market_obs, pos_feats):
        m = (np.asarray(market_obs, dtype=np.float32).reshape(1, -1) - self.mu) / self.sd
        return np.concatenate([m, np.asarray(pos_feats, dtype=np.float32).reshape(1, -1)], axis=1)

    def q_values(self, market_obs, pos_feats):
        return _q_mean(self.nets, self._x(market_obs, pos_feats))[0]

    def q_stats(self, market_obs, pos_feats):
        q_mean, q_std = _q_stats(self.nets, self._x(market_obs, pos_feats))
        return q_mean[0], q_std[0]


# ─────────────────────────── ورودی عامل (مشترک آموزش و زنده) ───────────────────────────
def _wilder_atr(h, l, c, n=14):
    """ATR وایلدر (مثل ta.volatility.average_true_range): اولین مقدار در اندیس n-1، قبل از آن صفر"""
    h, l, c = (np.asarray(x, dtype=np.float64) for x in (h, l, c))
    cp = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - cp), np.abs(l - cp)]), axis=0)
    atr = np.zeros(len(c))
    if len(c) >= n:
        atr[n - 1] = tr[:n].mean()
        k = (n - 1) / n
        for i in range(n, len(c)):
            atr[i] = atr[i - 1] * k + tr[i] / n
    return atr


def _rl_conservative_spread(spread_raw):
    """بیشترِ (اسپرد کندل، میانه‌ی اسپرد ۱۰۰ کندل اخیر) — ستون اسپرد کندل جهش‌ها را نشان نمی‌دهد"""
    s = pd.Series(np.nan_to_num(np.asarray(spread_raw, dtype=np.float64), nan=0.0).clip(min=0.0))
    return np.maximum(s.to_numpy(), s.rolling(SPREAD_MEDIAN_WINDOW, min_periods=1).median().to_numpy())


def _fill_rel_window(out, col, ohlc, end_idx, c_ref, a_ref, W, chunk=20000):
    """پنجره‌ی W کندلی OHLC که به end_idx ختم می‌شود، نسبت به کلوز فعلی ÷ ATR فعلی، در ستون‌های col.."""
    n = len(end_idx)
    ok = end_idx >= W - 1
    offs = np.arange(-W + 1, 1)
    for s0 in range(0, n, chunk):
        rows = np.flatnonzero(ok[s0:s0 + chunk]) + s0
        if len(rows) == 0:
            continue
        blk = ohlc[end_idx[rows][:, None] + offs[None, :]]                       # (k, W, 4)
        blk = (blk - c_ref[rows, None, None]) / a_ref[rows, None, None]
        out[rows, col:col + W * 4] = np.clip(blk.reshape(len(rows), -1), -RL_OBS_CLIP, RL_OBS_CLIP)


_RL_PAT_CACHE = {'key': None, 'val': None, 'now': None, 'ts': None}


def rl_secondary_warmup(S=None):
    """کندل‌های تایم‌فریم دوم که الگوها و ویژگی‌های مدل روی آن (با روند M30) لازم دارند"""
    if not RL_PATTERN_INPUTS:
        return 0
    S = int(S or RL_SECONDARY_TF_SECONDS)
    return ga_warmup(S, ga_htf_seconds(S)) + RL_PATTERN_AGE + 10


def rl_live_bars_s():
    """تعداد کندل تایم‌فریم دوم RL برای تصمیم زنده (پنجره‌ی خام + warm-up الگوهای M5)"""
    return max(rl_live_bars(), rl_secondary_warmup() + RL_WINDOW_S + 100)


def _rl_pattern_inputs(bs, S, idx_s):
    """
    ورودی الگوهای مدل برای هر کندل اصلی (n, RL_N_PATTERN): ga_primitives روی کندل‌های تایم‌فریم دوم (bs) با روند
    ga_htf_seconds(S)، از آخرین کندل بسته‌ی دوم (idx_s) — همان تابعی که حالت Model استفاده می‌کند، فقط گذشته.
    جفت الگو: +1 صعودی / −1 نزولی روی آخرین کندل، + RL_PATTERN_DECAY × کندل قبل و … ؛ سپس ویژگی‌های عددی.
    NaN = هنوز نامعلوم (warm-up). یک cache: در تصمیم زنده تا کندل دوم تازه‌ای بسته نشود دوباره حساب نمی‌شود.
    """
    ts_s = bs['time'].to_numpy(dtype=np.int64)
    m = len(ts_s)
    key = (m, int(ts_s[0]) if m else 0, int(ts_s[-1]) if m else 0, int(S))
    if _RL_PAT_CACHE['key'] != key:
        o, h, l, c = (bs[k].to_numpy(dtype=np.float64) for k in ('open', 'high', 'low', 'close'))
        atr_s = _wilder_atr(h, l, c)
        prim, valid = ga_primitives(o, h, l, c, atr_s, ts_s, int(S), ga_htf_seconds(int(S)))
        per = np.full((m, RL_N_PATTERN), np.nan, dtype=np.float32)
        now = np.zeros((m, RL_N_PATTERN_FLAGS), dtype=np.int8)
        for k, (b, s_) in enumerate(GA_PATTERN_PAIRS):
            x = prim[b].astype(np.float64) - prim[s_].astype(np.float64)
            now[:, k] = prim[b].astype(np.int8) - prim[s_].astype(np.int8)
            acc = x.copy()
            for a_ in range(1, RL_PATTERN_AGE):
                acc[a_:] += (RL_PATTERN_DECAY ** a_) * x[:-a_]
            per[:, k] = acc
        for j, f in enumerate(RL_PATTERN_NUM):
            v = prim['n:' + f].astype(np.float64) / RL_CONTEXT_SCALE.get(f, 1.0)
            per[:, RL_N_PATTERN_FLAGS + j] = np.clip(v, -RL_OBS_CLIP, RL_OBS_CLIP)
        ok_s = valid & (np.arange(m) >= RL_PATTERN_AGE - 1)
        per[~ok_s] = np.nan
        _RL_PAT_CACHE.update(key=key, val=per, now=now, ts=ts_s.copy())
    per = _RL_PAT_CACHE['val']
    out = np.full((len(idx_s), RL_N_PATTERN), np.nan, dtype=np.float32)
    ok = (idx_s >= 0) & (idx_s < m)
    out[ok] = per[idx_s[ok]]
    return out


def rl_pattern_names_now(t_last, P=None, S=None):
    """
    اسم الگوهای مدل روی آخرین کندل بسته‌ی تایم‌فریم دوم تا بسته شدن کندل اصلی t_last (از همان محاسبه‌ی ورودی عامل،
    بعد از rl_live_context) — برای دلیل تصمیم
    """
    ts_s, now = _RL_PAT_CACHE.get('ts'), _RL_PAT_CACHE.get('now')
    if not RL_PATTERN_INPUTS or ts_s is None or now is None or not len(ts_s):
        return []
    P = int(P or RL_TF_SECONDS)
    S = int(S or RL_SECONDARY_TF_SECONDS)
    k = int(np.searchsorted(ts_s, int(t_last) + P - S, side='right')) - 1
    if k < 0:
        return []
    out = []
    for j, (b, s_) in enumerate(GA_PATTERN_PAIRS):
        if now[k, j] > 0:
            out.append(GA_NAMES.get(b, b))
        elif now[k, j] < 0:
            out.append(GA_NAMES.get(s_, s_))
    return out


def _rl_features(bp, bs, P, S):
    """
    ورودی بازار عامل برای همه‌ی کندل‌های bp (تایم‌فریم اصلی) + زمینه‌ی bs (تایم‌فریم دوم).
    bp/bs: DataFrame با ستون‌های time, open, high, low, close, spread_px, volume (قیمت Bid، اسپرد به واحد قیمت).
    از تایم‌فریم دوم فقط کندل‌هایی استفاده می‌شوند که تا بسته شدن کندل اصلی بسته شده‌اند.
    خروجی dict آرایه‌ها هم‌طول bp؛ 'valid' کندل‌هایی را نشان می‌دهد که همه‌ی ورودی‌هایشان کامل است.
    """
    t = bp['time'].to_numpy(dtype=np.int64)
    o, h, l, c = (bp[k].to_numpy(dtype=np.float64) for k in ('open', 'high', 'low', 'close'))
    n = len(c)
    atr = _wilder_atr(h, l, c)
    spread = _rl_conservative_spread(bp['spread_px'].to_numpy(dtype=np.float64))
    vol = (pd.to_numeric(bp['volume'], errors='coerce').to_numpy(dtype=np.float64)
           if 'volume' in bp.columns else np.full(n, np.nan))
    obs = np.full((n, RL_N_MARKET), np.nan, dtype=np.float32)
    idx = np.arange(n)

    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(atr > 0, atr, np.nan)
        col = 0
        # ۱) کندل‌های خام تایم‌فریم اصلی
        _fill_rel_window(obs, col, np.stack([o, h, l, c], axis=1), idx, c, a, RL_WINDOW_P)
        col += RL_WINDOW_P * 4
        # ۲) کندل‌های خام تایم‌فریم دوم (هم‌تراز، بدون نگاه به آینده)
        ts_s = bs['time'].to_numpy(dtype=np.int64)
        ohlc_s = np.stack([bs[k].to_numpy(dtype=np.float64) for k in ('open', 'high', 'low', 'close')], axis=1)
        idx_s = np.searchsorted(ts_s, t + P - S, side='right') - 1
        _fill_rel_window(obs, col, ohlc_s, idx_s, c, a, RL_WINDOW_S)
        col += RL_WINDOW_S * 4

        def put(x, lo=-RL_OBS_CLIP, hi=RL_OBS_CLIP):
            nonlocal col
            obs[:, col] = np.clip(x, lo, hi)
            col += 1

        # ۳) بازده چندمقیاسی
        for k in RL_RET_LAGS:
            r = np.full(n, np.nan)
            if n > k:
                r[k:] = (c[k:] - c[:-k]) / a[k:]
            put(r, -10.0, 10.0)
        # ۴) ساختار رنج: جای کلوز در رنج ۶۰ کندل، فاصله تا سقف/کف ۲۴۰ کندل
        hh60 = pd.Series(h).rolling(60).max().to_numpy()
        ll60 = pd.Series(l).rolling(60).min().to_numpy()
        r60 = hh60 - ll60
        put(np.where(r60 > 0, 2 * (c - ll60) / r60 - 1, 0.0), -1.0, 1.0)
        put((pd.Series(h).rolling(240).max().to_numpy() - c) / a, 0.0, 20.0)
        put((c - pd.Series(l).rolling(240).min().to_numpy()) / a, 0.0, 20.0)
        # ۵) رژیم نوسان: کوتاه/بلند، ATR نسبت به میانه‌ی ۵۰۰ کندل، اسپرد/ATR
        rng_ = pd.Series(h - l)
        put(np.log(rng_.rolling(5).mean().to_numpy() / rng_.rolling(60).mean().to_numpy()), -2.0, 2.0)
        med_atr = pd.Series(a).rolling(RL_VOL_WINDOW, min_periods=100).median().to_numpy()
        put(np.log(a / med_atr), -2.0, 2.0)
        put(spread / a, 0.0, 2.0)
        # ۶) حجم تیکی نسبت به میانه‌ی ۶۰ کندل (داده‌ی قدیمی بدون حجم → صفر = خنثی)
        vmed = pd.Series(vol).rolling(60, min_periods=10).median().to_numpy()
        put(np.nan_to_num(np.log((vol + 1.0) / (vmed + 1.0)), nan=0.0), -3.0, 3.0)
        # ۷) زمان (ساعت سرور بروکر): ساعت روز و روز هفته
        tod = (t % 86400) / 86400.0
        put(np.sin(2 * np.pi * tod), -1.0, 1.0)
        put(np.cos(2 * np.pi * tod), -1.0, 1.0)
        put(((t // 86400 + 3) % 7) / 6.0, 0.0, 1.0)          # دوشنبه = 0
        # ۸) شکاف زمانی: کندل‌های جاافتاده قبل از این کندل، و چند کندل از آخرین شکاف (شروع جلسه) گذشته
        gap = np.zeros(n)
        if n > 1:
            gap[1:] = np.maximum((t[1:] - t[:-1]) / float(P) - 1.0, 0.0)
        put(np.minimum(gap, 120.0) / 120.0, 0.0, 1.0)
        last_gap = np.maximum.accumulate(np.where(gap > 3, idx, 0))
        put(np.minimum(idx - last_gap, 300) / 300.0, 0.0, 1.0)
        # ۹) زمینه (v5): ویژگی‌های عددی مدل الگو + تعداد الگوهای صعودی/نزولی (فقط گذشته؛ همان تابع تصمیم زنده)
        if RL_CONTEXT_FEATURES:
            prim, _ = ga_primitives(o, h, l, c, atr, t, P, S)
            for f in RL_CONTEXT_NUM:
                put(prim['n:' + f].astype(np.float64) / RL_CONTEXT_SCALE.get(f, 1.0))     # NaN = هنوز نامعلوم
            put(np.sum([prim[p] for p in GA_BULL], axis=0) / 5.0, 0.0, RL_OBS_CLIP)
            put(np.sum([prim[p] for p in GA_BEAR], axis=0) / 5.0, 0.0, RL_OBS_CLIP)
        # ۱۰) الگوهای مدل روی کندل‌های بسته‌ی تایم‌فریم دوم + ویژگی‌های آن با روند M30 (v8)
        if RL_PATTERN_INPUTS:
            obs[:, col:col + RL_N_PATTERN] = _rl_pattern_inputs(bs, S, idx_s)
            col += RL_N_PATTERN
    assert col == RL_N_MARKET, (col, RL_N_MARKET)

    valid = (idx >= rl_raw_warmup(P, S)) & (atr > 0) & np.isfinite(obs).all(axis=1)
    # واحد پاداش و گزارش (v7): میانه‌ی ATR ۵۰۰ کندل — مستقل از حد ضرری که عامل انتخاب می‌کند
    unit = np.where(np.isfinite(med_atr) & (med_atr > 0), med_atr, atr)
    can_open = np.ones(n, dtype=bool) if RL_MAX_SPREAD_ATR <= 0 else (spread <= RL_MAX_SPREAD_ATR * atr)
    return {'ts': t, 'open': o, 'high': h, 'low': l, 'close': c, 'spread': spread, 'atr': atr,
            'unit': unit, 'can_open': can_open, 'obs': obs, 'valid': valid}


def load_rl_dataset(primary_db_path=None, secondary_db_path=None, limit=None,
                    primary_tf_seconds=None, secondary_tf_seconds=None):
    """
    داده‌ی عامل از جدول raw_bars (مستقل از لیبل‌ها): جدیدترین `limit` کندل اصلی + کندل‌های تایم‌فریم دوم.
    خروجی: dict آرایه‌ها (فقط کندل‌های معتبر، به ترتیب زمانی) یا None.
    """
    primary_db_path = primary_db_path or RL_PRIMARY_DB_PATH
    secondary_db_path = secondary_db_path or RL_SECONDARY_DB_PATH
    P = primary_tf_seconds or RL_TF_SECONDS
    S = secondary_tf_seconds or RL_SECONDARY_TF_SECONDS
    try:
        bp = load_raw_bars(primary_db_path, limit=int(limit or RL_DATA_BARS))
        if len(bp) < RL_MIN_BARS:
            print(f"✗ Not enough raw {RL_TF_STR} bars for RL ({len(bp)} < {RL_MIN_BARS})")
            return None
        bs = load_raw_bars(secondary_db_path,
                           since=int(bp['time'].iloc[0]) - (RL_WINDOW_S + 50 + rl_secondary_warmup(S)) * S)
    except Exception as e:
        print(f"✗ Cannot read raw bars for RL: {e}")
        return None
    if len(bs) < RL_WINDOW_S + 10:
        print(f"✗ Not enough raw {RL_SECONDARY_TF_STR} bars for RL ({len(bs)})")
        return None

    f = _rl_features(bp, bs, P, S)
    sel = np.flatnonzero(f['valid'])
    if len(sel) < 5000:
        print(f"✗ Not enough valid raw bars for RL ({len(sel)})")
        return None
    take = slice(int(sel[0]), int(sel[-1]) + 1) if sel[-1] - sel[0] + 1 == len(sel) else sel
    data = {k: f[k][take] for k in ('ts', 'open', 'high', 'low', 'close', 'spread', 'atr', 'unit', 'can_open')}
    data['obs'] = f['obs'][take]
    # هزینه‌ی هر معامله جدا از اسپرد: لغزش ورود + هزینه‌ی اضافه‌ی تنظیمات (دلار قیمت)
    data['extra_cost'] = float(EXTRA_COST_POINTS) * _point_size.get(SYMBOL, 0.0) + float(RL_SLIPPAGE_PRICE)
    no_vol = float(np.mean(np.isnan(pd.to_numeric(bp['volume'], errors='coerce').to_numpy(dtype=float))))
    if no_vol > 0.05:
        print(f"ℹ️ {no_vol:.0%} of raw bars have no tick volume (saved by an older version) — "
              f"they get a neutral volume input")
    return data


def _slice_data(data, lo, hi, obs=None):
    out = {k: (v[lo:hi] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    if obs is not None:
        out['obs'] = obs
    return out


# ─────────────────────────── محیط معامله (برداری) ───────────────────────────
class VecTradingEnv:
    """
    چند شبیه‌ساز موازی روی تاریخچه. حالت i = لحظه‌ی بسته شدن کندل i.
      اجرا: بستن/باز کردن با Open کندل i+1 (خرید: ورود با Ask، خروج با Bid؛ فروش برعکس).
      ورود با حد ضرر/سود انتخاب‌شده‌ی عامل (ضریب ATR کندل i؛ حد ضرر ≥ RL_MIN_SL_SPREADS × اسپرد).
      کندل i+1: اگر Open از حد ضرر/سود گذشته باشد (گپ) با Open بسته می‌شود؛ وگرنه لمس حد ضرر با قیمت حد ضرر، لمس حد
      سود با قیمت حد سود، و اگر هر دو لمس شوند حد ضرر (بدبینانه). ارزش‌گذاری پوزیشن باز در Close (خرید با Bid، فروش با Ask).
      بعد از RL_MAX_HOLD کندل، در تصمیم بعدی اجباری بسته می‌شود؛ ورود وقتی can_open=False باشد انجام نمی‌شود.
    پاداش = تغییر سود/زیان (به قیمت) ÷ میانه‌ی ATR هنگام ورود (منهای لغزش و هزینه‌ی اضافه در ورود).
    """

    def __init__(self, data, n_envs, lo, hi, max_hold, episode_bars=RL_EPISODE_BARS, rng=None,
                 sequential=False):
        self.d = data
        self.n = int(n_envs)
        self.lo, self.hi = int(lo), int(hi)
        self.max_hold = int(max_hold)
        self.rng = rng or np.random.default_rng(0)
        self.sequential = sequential
        self.episode_bars = (self.hi - self.lo) if sequential else int(max(1, min(episode_bars, self.hi - self.lo - 2)))
        self.pos = np.zeros(self.n, dtype=np.int8)
        self.entry = np.zeros(self.n)
        self.sl_level = np.zeros(self.n)
        self.tp_level = np.zeros(self.n)
        self.has_tp = np.zeros(self.n, dtype=bool)
        self.sl_dist = np.ones(self.n)
        self.tp_dist = np.zeros(self.n)
        self.unit_e = np.ones(self.n)
        self.act_e = np.zeros(self.n, dtype=np.int64)
        self.held = np.zeros(self.n, dtype=np.int32)
        self.steps = np.zeros(self.n, dtype=np.int32)
        self.i = np.full(self.n, self.lo, dtype=np.int64)
        if not sequential:
            self.reset(np.ones(self.n, dtype=bool))

    def reset(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        top = max(self.lo + 1, self.hi - self.episode_bars - 1)
        self.i[mask] = self.rng.integers(self.lo, top, size=k)
        self.pos[mask] = 0
        self.held[mask] = 0
        self.steps[mask] = 0

    def _unreal(self, c, s):
        """سود/زیان شناور به قیمت (خرید با Bid، فروش با Ask)"""
        return np.where(self.pos == 1, c - self.entry,
                        np.where(self.pos == -1, self.entry - (c + s), 0.0))

    def pos_features(self):
        i = self.i
        in_pos = self.pos != 0
        return _rl_pos_features(self.pos, self._unreal(self.d['close'][i], self.d['spread'][i]),
                                np.where(in_pos, self.held, 0), self.sl_dist, self.tp_dist, self.d['atr'][i],
                                self.max_hold)

    def valid_mask(self):
        """اکشن‌های مجاز حالت فعلی هر شبیه‌ساز"""
        expired = (self.pos != 0) & (self.held >= self.max_hold)
        return _rl_valid_mask(self.pos, self.d['can_open'][self.i], expired)

    def _record(self, trades, mask, pnl, reason):
        for e in np.flatnonzero(mask):
            trades.append({'dir': int(self.pos[e]), 'pnl': float(pnl[e]), 'ret': float(pnl[e] / self.unit_e[e]),
                           'held': int(self.held[e]), 'reason': reason, 'action': int(self.act_e[e]),
                           'env': int(e)})

    def step(self, actions, record=False):
        """خروجی: (پاداش، پایان اپیزود، معاملات بسته‌شده، اکشن واقعاً اجراشده)"""
        d = self.d
        i = self.i
        j = i + 1
        c, s = d['close'][i], d['spread'][i]
        o1, h1, l1, c1, s1 = d['open'][j], d['high'][j], d['low'][j], d['close'][j], d['spread'][j]
        xc = d['extra_cost']
        a = np.asarray(actions, dtype=np.int64).copy()
        # قانون‌های ربات زنده: خروج اجباری بعد از حداکثر نگه‌داری، HOLD بدون پوزیشن = بیرون ماندن،
        # ورود هم‌جهت ممنوع (نگه داشتن)، و ممنوعیت ورود/برعکس کردن با اسپرد زیاد (فقط بستن)
        expired = (self.pos != 0) & (self.held >= self.max_hold)
        side = RL_ACT_SIDE[a]
        a = np.where((a == RL_A_HOLD) & (self.pos == 0), RL_A_FLAT, a)
        a = np.where((side != 0) & (side == self.pos), RL_A_HOLD, a)
        a = np.where((RL_ACT_SIDE[a] != 0) & ~d['can_open'][i], RL_A_FLAT, a)
        a = np.where(expired, RL_A_FLAT, a)
        side = RL_ACT_SIDE[a]
        target = np.where(a == RL_A_HOLD, self.pos, side).astype(np.int8)
        reward = np.zeros(self.n)
        trades = [] if record else None

        u0 = self._unreal(c, s)
        # ۱) بستن پوزیشنی که با هدف فرق دارد، با Open کندل بعد
        closing = (self.pos != 0) & (target != self.pos)
        if closing.any():
            exit_pnl = np.where(self.pos == 1, o1 - self.entry, self.entry - (o1 + s1))
            reward += np.where(closing, (exit_pnl - u0) / self.unit_e, 0.0)
            if record:
                self._record(trades, closing & ~expired, exit_pnl - xc, 'agent')
                self._record(trades, closing & expired, exit_pnl - xc, 'max_hold')
            self.pos[closing] = 0
        held_over = self.pos != 0

        # ۲) باز کردن پوزیشن جدید با Open کندل بعد، با حد ضرر/سود انتخابی عامل
        opening = (self.pos == 0) & (target != 0)
        if opening.any():
            atr_i = d['atr'][i]
            sl_d = np.maximum(RL_ACT_SL[a] * atr_i, RL_MIN_SL_SPREADS * s)
            tp_d = RL_ACT_TP[a] * atr_i
            long_o = opening & (target == 1)
            short_o = opening & (target == -1)
            self.pos[opening] = target[opening]
            self.entry = np.where(long_o, o1 + s1, np.where(short_o, o1, self.entry))
            self.sl_dist = np.where(opening, sl_d, self.sl_dist)
            self.tp_dist = np.where(opening, tp_d, self.tp_dist)
            self.has_tp = np.where(opening, tp_d > 0, self.has_tp)
            self.sl_level = np.where(long_o, self.entry - sl_d, np.where(short_o, self.entry + sl_d, self.sl_level))
            self.tp_level = np.where(long_o, self.entry + tp_d, np.where(short_o, self.entry - tp_d, self.tp_level))
            self.unit_e = np.where(opening, d['unit'][i], self.unit_e)
            self.act_e = np.where(opening, a, self.act_e)
            self.held[opening] = 0
            reward[opening] -= xc / self.unit_e[opening]

        # ۳) کندل بعد: گپ Open، حد ضرر، حد سود، سپس ارزش‌گذاری در Close
        is_long, is_short = self.pos == 1, self.pos == -1
        tp_on = self.has_tp
        # گپ فقط برای پوزیشنی که از قبل باز بوده (پوزیشن تازه با همین Open باز شده)
        gap_sl = held_over & np.where(is_long, o1 <= self.sl_level, o1 + s1 >= self.sl_level)
        gap_tp = held_over & tp_on & ~gap_sl & np.where(is_long, o1 >= self.tp_level, o1 + s1 <= self.tp_level)
        hit_sl = gap_sl | (~gap_tp & np.where(is_long, l1 <= self.sl_level, h1 + s1 >= self.sl_level))
        hit_tp = ~hit_sl & tp_on & (gap_tp | np.where(is_long, h1 >= self.tp_level, l1 + s1 <= self.tp_level))
        hit_sl &= is_long | is_short
        hit_tp &= is_long | is_short
        fill_sl = np.where(is_long, np.minimum(self.sl_level, o1), np.maximum(self.sl_level, o1 + s1))
        fill_tp = np.where(is_long, np.maximum(self.tp_level, o1), np.minimum(self.tp_level, o1 + s1))
        mark = np.where(is_long, c1 - self.entry, self.entry - (c1 + s1))
        px_sl = np.where(is_long, fill_sl - self.entry, self.entry - fill_sl)
        px_tp = np.where(is_long, fill_tp - self.entry, self.entry - fill_tp)
        u1 = np.where(hit_sl, px_sl, np.where(hit_tp, px_tp, mark))
        in_pos = is_long | is_short
        u1 = np.where(in_pos, u1, 0.0)
        u_start = np.where(held_over, u0, 0.0)
        reward += np.where(in_pos, (u1 - u_start) / self.unit_e, 0.0)
        self.held[in_pos] += 1
        if record:
            if hit_sl.any():
                self._record(trades, hit_sl, u1 - xc, 'stop_loss')
            if hit_tp.any():
                self._record(trades, hit_tp, u1 - xc, 'take_profit')
        self.pos[hit_sl | hit_tp] = 0

        self.i = j
        self.steps += 1
        done = (self.steps >= self.episode_bars) | (self.i + 1 >= self.hi)
        return reward.astype(np.float32), done, trades, a

    def close_all(self, reason='end'):
        """بستن پوزیشن‌های باز در پایان بازه‌ی ارزیابی (به ارزش Close فعلی)"""
        trades = []
        open_ = self.pos != 0
        if open_.any():
            pnl = self._unreal(self.d['close'][self.i], self.d['spread'][self.i]) - self.d['extra_cost']
            self._record(trades, open_, pnl, reason)
        self.pos[:] = 0
        return trades


def _rl_pos_features(pos, unreal, held, sl_dist, tp_dist, atr_now, max_hold):
    """
    ورودی وضعیت پوزیشن (n, RL_POS_FEATURES) — یک فرمول برای شبیه‌ساز، هسته‌ی ارزیابی و ربات زنده:
    [خرید؟، فروش؟، سود شناور ÷ فاصله‌ی حد ضرر، کندل نگه‌داری ÷ حداکثر، فاصله‌ی حد ضرر ÷ ATR فعلی،
     فاصله‌ی حد سود ÷ فاصله‌ی حد ضرر، حد سود دارد؟]؛ بدون پوزیشن همه صفر.
    """
    pos = np.asarray(pos).reshape(-1)
    in_pos = pos != 0
    sl_dist = np.maximum(np.asarray(sl_dist, dtype=np.float64).reshape(-1), 1e-9)
    tp_dist = np.asarray(tp_dist, dtype=np.float64).reshape(-1)
    atr_now = np.maximum(np.asarray(atr_now, dtype=np.float64).reshape(-1), 1e-9)
    u = np.clip(np.asarray(unreal, dtype=np.float64).reshape(-1) / sl_dist, -2.0, 6.0) / 2.0
    hd = np.minimum(np.asarray(held, dtype=np.float64).reshape(-1), max_hold) / max_hold
    slr = np.clip(sl_dist / atr_now, 0.0, 16.0) / 8.0
    has_tp = tp_dist > 0
    tpr = np.where(has_tp, np.clip(tp_dist / sl_dist, 0.0, 16.0) / 4.0, 0.0)
    out = np.stack([pos == 1, pos == -1, u, hd, slr, tpr, has_tp], axis=1).astype(np.float32)
    out[~in_pos] = 0.0
    return out


# ─────────────────────────── ارزیابی ───────────────────────────
def _rl_score(rets, min_n):
    """
    امتیاز انتخاب: سود کل با کسر عدم‌قطعیت = n × (میانگین − انحراف‌معیار/√n) به واحد ATR.
    بدون معامله = 0؛ با معامله‌ی خیلی کم، مثبت حساب نمی‌شود (شانسی است).
    """
    n = len(rets)
    if n == 0:
        return 0.0
    if n < max(2, min_n):
        return float(min(0.0, np.sum(rets)))
    return float(np.sum(rets) - rets.std(ddof=1) * np.sqrt(n))


def _rl_policy_trades_py(nets, obs_std, data, lo, hi, margin=0.0, kappa=0.0):
    """اجرای حریصانه‌ی عامل با شبیه‌ساز NumPy (کند؛ وقتی numba نصب نیست): لیست معاملات"""
    env = VecTradingEnv(data, 1, lo, hi, RL_MAX_HOLD, sequential=True)
    trades = []
    while env.i[0] + 1 < hi:
        x = np.concatenate([obs_std[env.i], env.pos_features()], axis=1)
        q_mean, q_std = _q_stats(nets, x)
        a = _rl_pick_action(q_mean[0], env.valid_mask()[0], int(env.pos[0]), margin, q_std[0], kappa)
        _, _, tr, _ = env.step(np.array([a]), record=True)
        trades.extend(tr)
    trades.extend(env.close_all())
    return trades


# ارزیابی سریع با numba (اختیاری): همان شبیه‌ساز و همان قانون انتخاب اکشن، کامپایل‌شده؛ ضرب ورودی بازار در لایه‌ی اول
# شبکه برای همه‌ی کندل‌ها یک‌جا (BLAS) حساب می‌شود و فقط بخش وابسته به پوزیشن گام‌به‌گام.
_RL_EVAL_JIT = {'fn': None, 'tried': False}


def _rl_eval_kernel(pre, Wp, W1, b1, Wq, bq, act_side, act_sl, act_tp, o, h, l, c, s, atr, unit, can_open,
                    lo, hi, max_hold, xc, min_sl_sp, margin, kappa,
                    out_ret, out_pnl, out_dir, out_held, out_reason, out_act):
    M = pre.shape[0]
    H1 = pre.shape[2]
    H2 = W1.shape[2]
    A = Wq.shape[2]
    h1v = np.empty(H1, dtype=np.float64)
    h2v = np.empty(H2, dtype=np.float64)
    qm = np.empty(A, dtype=np.float64)
    q2 = np.empty(A, dtype=np.float64)
    pf = np.empty(7, dtype=np.float64)
    pos = 0
    entry = 0.0
    sl_level = 0.0
    tp_level = 0.0
    has_tp = False
    sl_dist = 1.0
    tp_dist = 0.0
    unit_e = 1.0
    act_e = 0
    held = 0
    n_tr = 0
    i = lo
    while i + 1 < hi:
        # ── ورودی پوزیشن (مثل _rl_pos_features) ──
        for k in range(7):
            pf[k] = 0.0
        if pos != 0:
            if pos == 1:
                unreal = c[i] - entry
            else:
                unreal = entry - (c[i] + s[i])
            sd_ = sl_dist if sl_dist > 1e-9 else 1e-9
            u = unreal / sd_
            if u < -2.0:
                u = -2.0
            elif u > 6.0:
                u = 6.0
            pf[0] = 1.0 if pos == 1 else 0.0
            pf[1] = 1.0 if pos == -1 else 0.0
            pf[2] = u / 2.0
            hh = held if held < max_hold else max_hold
            pf[3] = hh / max_hold
            an = atr[i] if atr[i] > 1e-9 else 1e-9
            r_ = sd_ / an
            if r_ > 16.0:
                r_ = 16.0
            pf[4] = r_ / 8.0
            if has_tp:
                r_ = tp_dist / sd_
                if r_ > 16.0:
                    r_ = 16.0
                pf[5] = r_ / 4.0
                pf[6] = 1.0
        # ── Q همه‌ی عامل‌ها: میانگین و انحراف معیار ──
        for a in range(A):
            qm[a] = 0.0
            q2[a] = 0.0
        for m in range(M):
            for k in range(H1):
                v = pre[m, i - lo, k]
                for p in range(7):
                    v += pf[p] * Wp[m, p, k]
                h1v[k] = v if v > 0.0 else 0.0
            for k in range(H2):
                v = b1[m, k]
                for q in range(H1):
                    v += h1v[q] * W1[m, q, k]
                h2v[k] = v if v > 0.0 else 0.0
            for a in range(A):
                v = bq[m, a]
                for q in range(H2):
                    v += h2v[q] * Wq[m, q, a]
                qm[a] += v
                q2[a] += v * v
        for a in range(A):
            qm[a] /= M
            var = q2[a] / M - qm[a] * qm[a]
            q2[a] = np.sqrt(var) if var > 0.0 else 0.0
        # ── اکشن‌های مجاز و انتخاب (مثل _rl_valid_mask و _rl_pick_action) ──
        expired = pos != 0 and held >= max_hold
        best = 0
        for a in range(1, A):
            if a == 1:
                ok = pos != 0 and not expired
            else:
                sd = act_side[a]
                ok = (not expired) and can_open[i] and sd != pos
            if ok and qm[a] > qm[best]:
                best = a
        if act_side[best] != 0:
            if (qm[best] - kappa * q2[best]) - qm[0] < margin:
                if pos != 0 and (not expired) and qm[1] > qm[0]:
                    best = 1
                else:
                    best = 0
        # ── یک گام شبیه‌ساز (مثل VecTradingEnv.step) ──
        j = i + 1
        if best == 1:
            tgt = pos
        else:
            tgt = act_side[best]
        if pos != 0 and tgt != pos:
            if pos == 1:
                pnl = o[j] - entry
            else:
                pnl = entry - (o[j] + s[j])
            pnl -= xc
            out_ret[n_tr] = pnl / unit_e
            out_pnl[n_tr] = pnl
            out_dir[n_tr] = pos
            out_held[n_tr] = held
            out_reason[n_tr] = 4 if expired else 0
            out_act[n_tr] = act_e
            n_tr += 1
            pos = 0
        held_over = pos != 0
        if pos == 0 and tgt != 0:
            pos = tgt
            sl_d = act_sl[best] * atr[i]
            if sl_d < min_sl_sp * s[i]:
                sl_d = min_sl_sp * s[i]
            tp_d = act_tp[best] * atr[i]
            if tgt == 1:
                entry = o[j] + s[j]
                sl_level = entry - sl_d
                tp_level = entry + tp_d
            else:
                entry = o[j]
                sl_level = entry + sl_d
                tp_level = entry - tp_d
            sl_dist = sl_d
            tp_dist = tp_d
            has_tp = tp_d > 0.0
            unit_e = unit[i]
            act_e = best
            held = 0
        if pos != 0:
            if pos == 1:
                gap_sl = held_over and o[j] <= sl_level
                gap_tp = held_over and has_tp and (not gap_sl) and o[j] >= tp_level
                hit_sl = gap_sl or ((not gap_tp) and l[j] <= sl_level)
                hit_tp = (not hit_sl) and has_tp and (gap_tp or h[j] >= tp_level)
                if hit_sl:
                    fill = sl_level if sl_level < o[j] else o[j]
                    u1 = fill - entry
                elif hit_tp:
                    fill = tp_level if tp_level > o[j] else o[j]
                    u1 = fill - entry
                else:
                    u1 = c[j] - entry
            else:
                ask_o = o[j] + s[j]
                gap_sl = held_over and ask_o >= sl_level
                gap_tp = held_over and has_tp and (not gap_sl) and ask_o <= tp_level
                hit_sl = gap_sl or ((not gap_tp) and h[j] + s[j] >= sl_level)
                hit_tp = (not hit_sl) and has_tp and (gap_tp or l[j] + s[j] <= tp_level)
                if hit_sl:
                    fill = sl_level if sl_level > ask_o else ask_o
                    u1 = entry - fill
                elif hit_tp:
                    fill = tp_level if tp_level < ask_o else ask_o
                    u1 = entry - fill
                else:
                    u1 = entry - (c[j] + s[j])
            held += 1
            if hit_sl or hit_tp:
                pnl = u1 - xc
                out_ret[n_tr] = pnl / unit_e
                out_pnl[n_tr] = pnl
                out_dir[n_tr] = pos
                out_held[n_tr] = held
                out_reason[n_tr] = 1 if hit_sl else 3
                out_act[n_tr] = act_e
                n_tr += 1
                pos = 0
        i = j
    if pos != 0:
        if pos == 1:
            pnl = c[i] - entry
        else:
            pnl = entry - (c[i] + s[i])
        pnl -= xc
        out_ret[n_tr] = pnl / unit_e
        out_pnl[n_tr] = pnl
        out_dir[n_tr] = pos
        out_held[n_tr] = held
        out_reason[n_tr] = 2
        out_act[n_tr] = act_e
        n_tr += 1
    return n_tr


def _rl_eval_jit():
    """
    تابع کامپایل‌شده‌ی ارزیابی (اولین بار در هر اجرای برنامه، چند ثانیه)؛ None اگر numba نصب نیست.
    کش دیسکی numba عمداً استفاده نمی‌شود: اگر این فایل با اسم ماژول دیگری بارگذاری شده باشد کش خراب می‌شود.
    """
    if not _RL_EVAL_JIT['tried']:
        _RL_EVAL_JIT['tried'] = True
        try:
            import numba
            _RL_EVAL_JIT['fn'] = numba.njit(cache=False, fastmath=False)(_rl_eval_kernel)
        except Exception:
            _RL_EVAL_JIT['fn'] = None
    return _RL_EVAL_JIT['fn']


def _rl_kernel_args(nets, obs_std, data, lo, hi):
    """ورودی‌های هسته‌ی ارزیابی (ضرب ورودی بازار در لایه‌ی اول برای همه‌ی کندل‌ها یک‌جا)"""
    n_mkt = obs_std.shape[1]
    seg = np.ascontiguousarray(obs_std[lo:hi], dtype=np.float32)
    pre = np.stack([(seg @ net.W[0][:n_mkt] + net.b[0]).astype(np.float64) for net in nets])
    stack = lambda xs: np.ascontiguousarray(np.stack(xs).astype(np.float64))
    heads = [net.q_head() for net in nets]
    return (pre, stack([net.W[0][n_mkt:] for net in nets]),
            stack([net.W[1] for net in nets]), stack([net.b[1] for net in nets]),
            stack([w for w, _ in heads]), stack([b for _, b in heads]),
            RL_ACT_SIDE, RL_ACT_SL, RL_ACT_TP,
            *(np.ascontiguousarray(data[key], dtype=np.float64)
              for key in ('open', 'high', 'low', 'close', 'spread', 'atr', 'unit')),
            np.ascontiguousarray(data['can_open'], dtype=np.bool_))


def _rl_policy_arrays(nets, obs_std, data, lo, hi, margin=0.0, kappa=0.0, use_kernel=None):
    """
    اجرای حریصانه‌ی عامل روی [lo, hi)، کندل به کندل و با یک پوزیشن — دقیقاً مثل معامله‌ی زنده.
    خروجی: dict آرایه‌های معاملات (ret به واحد ATR، pnl، dir، held، action، reason: 0 = خود عامل، 1 = حد ضرر،
    2 = پایان بازه، 3 = حد سود، 4 = حداکثر نگه‌داری)
    use_kernel: None = numba اگر هست؛ 'py' = همان هسته بدون کامپایل (کند، برای آزمون)؛ False = شبیه‌ساز NumPy
    """
    fn = _rl_eval_jit() if use_kernel is None else (_rl_eval_kernel if use_kernel == 'py' else None)
    if fn is not None and all(net.n_trunk == 2 for net in nets) and hi - lo > 2:
        L = hi - lo + 2
        out_ret, out_pnl = np.zeros(L), np.zeros(L)
        out_dir, out_held, out_reason, out_act = (np.zeros(L, dtype=np.int64) for _ in range(4))
        try:
            k = fn(*_rl_kernel_args(nets, obs_std, data, lo, hi), int(lo), int(hi), int(RL_MAX_HOLD),
                   float(data['extra_cost']), float(RL_MIN_SL_SPREADS), float(margin), float(kappa),
                   out_ret, out_pnl, out_dir, out_held, out_reason, out_act)
            return {'ret': out_ret[:k], 'pnl': out_pnl[:k], 'dir': out_dir[:k], 'held': out_held[:k],
                    'reason': out_reason[:k], 'action': out_act[:k]}
        except Exception as e:
            # numba در این سیستم کار نکرد: از این به بعد ارزیابی با NumPy (کندتر ولی همان نتیجه)
            print(f"⚠️ Fast RL evaluation unavailable ({type(e).__name__}: {e}) — using the NumPy simulator")
            _RL_EVAL_JIT['fn'] = None
    trades = _rl_policy_trades_py(nets, obs_std, data, lo, hi, margin, kappa)
    codes = {nm: k for k, nm in enumerate(RL_EXIT_NAMES)}
    col = lambda key, dt: np.array([t[key] for t in trades], dtype=dt)
    return {'ret': col('ret', float), 'pnl': col('pnl', float), 'dir': col('dir', np.int64),
            'held': col('held', np.int64), 'action': col('action', np.int64),
            'reason': np.array([codes[t['reason']] for t in trades], dtype=np.int64)}


def _rl_perf_arrays(tr, data, lo, hi, min_n):
    """کارنامه‌ی معاملات (expectancy / total_atr به واحد ATR) + سهم هر حد ضرر/سود انتخابی و هر نوع خروج"""
    r, pnl = tr['ret'], tr['pnl']
    n_long = int(np.sum(tr['dir'] == 1))
    perf = _perf_from_returns(r, n_long, len(r) - n_long)
    days = max(1, len(np.unique(np.asarray(data['ts'][lo:hi]) // 86400)))
    eq = np.cumsum(r) if len(r) else np.zeros(1)
    exits = {RL_EXIT_NAMES[k]: int(np.sum(tr['reason'] == k)) for k in range(len(RL_EXIT_NAMES))
             if np.any(tr['reason'] == k)}
    acts = tr['action']
    sl_use = {f"{RL_SL_CHOICES[k]:g}": int(np.sum(RL_ACT_SLI[acts] == k)) for k in range(_RL_KSL)} if len(r) else {}
    tp_use = ({("none" if RL_TP_CHOICES[k] <= 0 else f"{RL_TP_CHOICES[k]:g}"): int(np.sum(RL_ACT_TPI[acts] == k))
               for k in range(_RL_KTP)} if len(r) else {})
    exp_price = float(pnl.mean()) if len(pnl) else 0.0
    perf.update({
        'unit': 'ATR',
        'exp_price': exp_price,
        'exp_money': money_per_trade(exp_price),
        'total_money': money_per_trade(float(pnl.sum())) if len(pnl) else 0.0,
        'trades_per_day': len(r) / days,
        'max_dd_R': float(np.max(np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq])),
        'avg_bars': float(np.mean(tr['held'])) if len(r) else 0.0,
        'exits': exits, 'sl_choices': sl_use, 'tp_choices': tp_use,
        'score': _rl_score(r, min_n),
    })
    return perf


def evaluate_rl_policy(qnet, obs_std, data, lo, hi, margin=0.0, min_n=5, kappa=0.0):
    """اجرای حریصانه‌ی عامل روی یک بازه، کندل به کندل و با یک پوزیشن — دقیقاً مثل معامله‌ی زنده"""
    nets = qnet.nets if isinstance(qnet, RLAgent) else (list(qnet) if isinstance(qnet, (list, tuple)) else [qnet])
    tr = _rl_policy_arrays(nets, obs_std, data, lo, hi, margin, kappa)
    return _rl_perf_arrays(tr, data, lo, hi, min_n), tr['ret']


def _format_rl_perf(p, choices=False):
    money = p.get('exp_money')
    money_txt = f", ≈${money:+.2f} at lot {LOT}" if money is not None else ""
    txt = (f"trades={p['trades']} (L{p['buys']}/S{p['sells']}, {p.get('trades_per_day', 0):.1f}/day) | "
           f"win={p['win_rate']:.1%} | exp={p['expectancy']:+.3f} ATR (${p.get('exp_price', 0):+.2f} price{money_txt}) | "
           f"PF={p['profit_factor']:.2f} | total={p['total_atr']:+.1f} ATR | maxDD={p.get('max_dd_R', 0):.1f} ATR")
    if p.get('trades'):
        ex = ", ".join(f"{k}={v}" for k, v in sorted(p.get('exits', {}).items()))
        txt += f" | avg hold={p.get('avg_bars', 0):.1f} bars | exits: {ex}"
        if choices:
            pct = lambda dct: ", ".join(f"{k}:{v / max(1, p['trades']):.0%}" for k, v in dct.items() if v)
            txt += (f"\n               agent's choices — SL×ATR {pct(p.get('sl_choices', {}))} | "
                    f"TP×ATR {pct(p.get('tp_choices', {}))}")
    return txt


# ─────────────────────────── آموزش ───────────────────────────
def _rl_explore_actions(rng, pos, valid):
    """
    اکشن تصادفی مجاز با سهم برابر برای «نوع» تصمیم (نه برای هر کدام از ۵۰ ورود): بدون پوزیشن ⅓ بیرون، ⅓ خرید، ⅓ فروش؛
    با پوزیشن ⅓ بستن، ⅓ نگه داشتن، ⅓ برعکس — حد ضرر و حد سود ورود تصادفی
    """
    n = len(pos)
    kind = rng.integers(0, 3, n)
    k_ent = _RL_KSL * _RL_KTP
    ent = rng.integers(0, k_ent, n)
    long_a = 2 + ent
    short_a = 2 + k_ent + ent
    flat_choice = np.where(kind == 0, RL_A_FLAT, np.where(kind == 1, long_a, short_a))
    rev = np.where(pos == 1, short_a, long_a)
    pos_choice = np.where(kind == 0, RL_A_FLAT, np.where(kind == 1, RL_A_HOLD, rev))
    a = np.where(pos == 0, flat_choice, pos_choice)
    ok = valid[np.arange(n), a]
    return np.where(ok, a, RL_A_FLAT)


def _rl_masked_argmax(q, valid):
    return np.argmax(np.where(valid, q, -np.inf), axis=1)


def _train_one_agent(data, obs_std, tr, va, seed, steps_total, min_val_trades, cancelled, on_check, tag,
                     stop_iter=None):
    """
    آموزش یک عامل Double DQN (سر فاکتورشده، ماسک اکشن‌های مجاز، بازده n-step) روی بازه‌ی tr؛ بهترین نسخه روی va
    نگه داشته می‌شود. va=None (آموزش دوباره روی کل داده): بدون سنجش، تا گام stop_iter با همان برنامه‌ی اکتشاف.
    خروجی: dict(weights, perf, score, point, iters) یا None اگر لغو شد.
    """
    tr_lo, tr_hi = tr
    N = RL_N_ENVS
    nst = RL_NSTEP
    in_dim = obs_std.shape[1] + RL_POS_FEATURES
    rng = np.random.default_rng(seed)
    q = NumpyQNet(in_dim, seed=seed)
    tq = NumpyQNet(in_dim, seed=seed)
    tq.copy_from(q)
    can_open = data['can_open']

    size = RL_BUFFER
    b_i = np.zeros(size, dtype=np.int64)
    b_j = np.zeros(size, dtype=np.int64)
    b_pf = np.zeros((size, RL_POS_FEATURES), dtype=np.float32)
    b_pf2 = np.zeros((size, RL_POS_FEATURES), dtype=np.float32)
    b_a = np.zeros(size, dtype=np.int64)
    b_r = np.zeros(size, dtype=np.float32)
    b_disc = np.zeros(size, dtype=np.float32)
    buf = {'ptr': 0, 'filled': 0}

    def emit(mask, i0, pf0, a0, ret, j1, pf1, disc):
        k = int(mask.sum())
        if k == 0:
            return
        slots = (buf['ptr'] + np.arange(k)) % size
        b_i[slots], b_pf[slots], b_a[slots], b_r[slots] = i0[mask], pf0[mask], a0[mask], ret[mask]
        b_j[slots], b_pf2[slots], b_disc[slots] = j1[mask], pf1[mask], disc
        buf['ptr'] = (buf['ptr'] + k) % size
        buf['filled'] = min(size, buf['filled'] + k)

    # تاریخچه‌ی n گام اخیر هر شبیه‌ساز (برای بازده n-step)
    H_i = np.zeros((nst, N), dtype=np.int64)
    H_pf = np.zeros((nst, N, RL_POS_FEATURES), dtype=np.float32)
    H_a = np.zeros((nst, N), dtype=np.int64)
    H_r = np.zeros((nst, N), dtype=np.float64)
    age = np.zeros(N, dtype=np.int64)
    gpow = RL_GAMMA ** np.arange(nst + 1)

    env = VecTradingEnv(data, N, tr_lo, tr_hi, RL_MAX_HOLD, rng=rng)
    n_iters = max(RL_EVAL_POINTS, int(steps_total) // N)
    eval_every = max(1, n_iters // RL_EVAL_POINTS)
    decay_iters = max(1, int(n_iters * RL_EPS_DECAY_FRAC))
    last_iter = n_iters if stop_iter is None else max(1, min(n_iters, int(stop_iter)))
    updates, losses = 0, []
    best = {'score': -np.inf, 'weights': None, 'perf': None, 'point': 0, 'iters': 0}
    eval_idx = 0
    n_mkt = obs_std.shape[1]
    ar = np.arange(RL_BATCH)

    for it in range(last_iter):
        if cancelled():
            return None
        eps = max(RL_EPS_END, RL_EPS_START - (RL_EPS_START - RL_EPS_END) * it / decay_iters)
        i0 = env.i.copy()
        pf0 = env.pos_features()
        valid0 = env.valid_mask()
        greedy = _rl_masked_argmax(q.forward(np.concatenate([obs_std[i0], pf0], axis=1)), valid0)
        explore = rng.random(N) < eps
        actions = np.where(explore, _rl_explore_actions(rng, env.pos.astype(np.int64), valid0), greedy)
        rewards, done, _, executed = env.step(actions)
        pf1 = env.pos_features()
        j1 = env.i.copy()

        slot = it % nst
        H_i[slot], H_pf[slot], H_a[slot], H_r[slot] = i0, pf0, executed, rewards
        age += 1
        full = age >= nst
        if full.any():                                   # گذار کامل n گامی برای قدیمی‌ترین گام پنجره
            s0 = (it - nst + 1) % nst
            ret = sum(gpow[m] * H_r[(it - nst + 1 + m) % nst] for m in range(nst))
            emit(full, H_i[s0], H_pf[s0], H_a[s0], ret, j1, pf1, gpow[nst])
        if done.any():                                   # پایان اپیزود: گام‌های باقی‌مانده با بازده کوتاه‌تر
            pend = np.where(age >= nst, nst - 1, age)
            for k in range(nst - 1):
                m_k = done & (pend > k)
                if not m_k.any():
                    continue
                sk = (it - k) % nst
                ret = sum(gpow[m] * H_r[(it - k + m) % nst] for m in range(k + 1))
                emit(m_k, H_i[sk], H_pf[sk], H_a[sk], ret, j1, pf1, gpow[k + 1])
            age[done] = 0
            env.reset(done)

        if buf['filled'] >= RL_WARMUP_TRANSITIONS:
            for _ in range(RL_UPDATES_PER_ITER):
                bi = rng.integers(0, buf['filled'], RL_BATCH)
                x = np.concatenate([obs_std[b_i[bi]], b_pf[bi]], axis=1)
                if RL_OBS_NOISE > 0:
                    x[:, :n_mkt] += rng.normal(0, RL_OBS_NOISE, (RL_BATCH, n_mkt)).astype(np.float32)
                x2 = np.concatenate([obs_std[b_j[bi]], b_pf2[bi]], axis=1)
                valid2 = _rl_mask_from_pf(b_pf2[bi], can_open[b_j[bi]])
                a2 = _rl_masked_argmax(q.forward(x2), valid2)                 # Double DQN، فقط اکشن‌های مجاز
                q2 = tq.forward(x2)[ar, a2]
                y = b_r[bi] + b_disc[bi] * q2
                losses.append(q.train_step(x, b_a[bi], y))
                updates += 1
                if updates % RL_TARGET_SYNC == 0:
                    tq.copy_from(q)

        if va is not None and (it + 1) % eval_every == 0 and eval_idx < RL_EVAL_POINTS:
            eval_idx += 1
            perf_v, _ = evaluate_rl_policy(q, obs_std, data, va[0], va[1], min_n=min_val_trades)
            score = perf_v['score']
            improved = score > best['score']
            if improved:
                best.update(score=score, weights=q.weights(), perf=perf_v, point=eval_idx, iters=it + 1)
            loss_txt = f"{np.mean(losses[-200:]):.4f}" if losses else "—"
            print(f"   [{tag}] check {eval_idx}/{RL_EVAL_POINTS} | step {(it + 1) * N:,} | eps={eps:.2f} | "
                  f"loss={loss_txt} | Val: trades={perf_v['trades']} ({perf_v['trades_per_day']:.1f}/day) "
                  f"exp={perf_v['expectancy']:+.3f} ATR PF={perf_v['profit_factor']:.2f} "
                  f"score={score:+.1f}{' ⭐' if improved else ''}")
            on_check(eval_idx, perf_v)
            if eps <= RL_EPS_END and eval_idx - best['point'] >= RL_PATIENCE:
                print(f"   [{tag}] ⏹️ early stop: no validation improvement in {RL_PATIENCE} checks")
                break
        elif va is None and (it + 1) % eval_every == 0:
            on_check(min(RL_EVAL_POINTS, (it + 1) // eval_every), None)
    if va is None:
        best.update(weights=q.weights(), point=0, iters=last_iter)
    elif best['weights'] is None:
        best.update(weights=q.weights(), perf=evaluate_rl_policy(q, obs_std, data, va[0], va[1],
                                                                 min_n=min_val_trades)[0],
                    point=eval_idx, iters=last_iter)
    return best


def train_rl_agent(db_path=None, secondary_db_path=None, progress_callback=None, total_steps=None, seed=42,
                   wait=False, stop_event=None):
    """
    آموزش عامل اسکلپ: RL_ENSEMBLE عامل (seed جدا) روی ۷۰٪ اول تاریخچه، انتخاب بهترین نسخه‌ی هر کدام و
    حداقل برتری ورود روی Validation (۱۵٪ بعد)، و سنجش روی Test (۱۵٪ آخر).
    عامل جدید فقط وقتی جایگزین عامل فعلی می‌شود که روی همان Test از آن بدتر نباشد (اولین عامل همیشه ذخیره می‌شود).
    wait=True: اگر آموزش دیگری در حال اجراست صبر کن (برای Setup برنامه).
    stop_event: رویدادی که آموزش را متوقف می‌کند (ترد آموزش ربات: bot_stop_event؛ Setup برنامه: None)
    خروجی: dict با accepted, agent, meta, metrics (+ skipped / cancelled / kept)
    """
    result = {'accepted': False, 'agent': None, 'meta': None, 'metrics': {}, 'skipped': False,
              'cancelled': False, 'kept': False}

    def waiting():
        if progress_callback:
            progress_callback(0.0, "Waiting for the running training to stop...")

    if not _acquire_training_lock('rl', wait=wait, on_wait=waiting, stop_event=stop_event):
        print("⏳ Another training is already running — this request is skipped")
        result['skipped'] = True
        return result
    try:
        return _train_rl_locked(result, db_path, secondary_db_path, progress_callback, total_steps, seed,
                                stop_event)
    except Exception as e:
        print(f"✗ RL training error: {e}")
        traceback.print_exc()
        save_train_time(False, f"RL training error: {e}", rl_last_train_file(), rl_agent_path(), rl_signature())
        return result
    finally:
        training_state.finish()
        _release_training_lock()


def _train_rl_locked(result, db_path, secondary_db_path, progress_callback, total_steps, seed, stop_event=None):
    global last_training_report

    def progress(frac, msg):
        if progress_callback:
            try:
                progress_callback(frac, msg)
            except Exception:
                pass

    def cancelled():
        return training_cancel.is_set() or (stop_event is not None and stop_event.is_set())

    print("\n" + "=" * 60)
    print(f"🎮 TRAINING RL SCALPER v{RL_VERSION} — {RL_TF_STR}+{RL_SECONDARY_TF_STR} | {rl_sl_text()} | "
          f"max hold {RL_MAX_HOLD} bars | agent opens, sizes its stop/target and closes by itself | "
          f"factored Double DQN, {RL_NSTEP}-step, ensemble of {RL_ENSEMBLE}")
    print("=" * 60)
    passes = 2 if RL_REFIT_ALL_DATA else 1
    training_state.start(total_epochs=RL_EVAL_POINTS * RL_ENSEMBLE * passes)
    status_file, agent_file, sig = rl_last_train_file(), rl_agent_path(), rl_signature()

    data = load_rl_dataset(db_path, secondary_db_path)
    if data is None:
        save_train_time(False, "Not enough data for RL", status_file, agent_file, sig)
        return result

    n = len(data['close'])
    gap = RL_WINDOW_P + RL_MAX_HOLD + max(RL_RET_LAGS)
    tr_lo, tr_hi = 0, int(n * 0.70)
    va_lo = tr_hi + gap
    va_hi = va_lo + int(n * 0.15)
    te_lo = va_hi + gap
    te_hi = n
    if te_hi - te_lo < 1000 or tr_hi - tr_lo < 3000:
        print("✗ Not enough data for RL train/val/test split")
        save_train_time(False, "Not enough data for RL split", status_file, agent_file, sig)
        return result

    def span(lo, hi):
        f = lambda x: datetime.fromtimestamp(int(x), tz=timezone.utc).strftime('%Y-%m-%d')
        return f"{f(data['ts'][lo])}→{f(data['ts'][hi - 1])}"

    print(f"📊 Bars: train={tr_hi - tr_lo} ({span(tr_lo, tr_hi)}) | val={va_hi - va_lo} ({span(va_lo, va_hi)}) | "
          f"test={te_hi - te_lo} ({span(te_lo, te_hi)})")
    atr_med = float(np.median(data['atr']))
    cost_u = (data['spread'] + data['extra_cost']) / data['unit']
    sl_txt = ", ".join(_px_txt(k * atr_med) for k in RL_SL_CHOICES)
    print(f"💸 Spread avg={_px_txt(np.mean(data['spread']))} + slippage/extra {_px_txt(data['extra_cost'])} per trade → "
          f"round-trip cost ≈ {np.median(cost_u):.2f} ATR per trade | ATR({RL_TF_STR}) median {_px_txt(atr_med)} → "
          f"stop-loss choices ≈ {sl_txt} | entries blocked by spread filter: {1 - np.mean(data['can_open']):.1%} of bars")

    # نرمال‌سازی با آمار Train (در جا، تا حافظه دو برابر نشود)
    obs = data['obs']
    mu = obs[tr_lo:tr_hi].mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = np.maximum(obs[tr_lo:tr_hi].std(axis=0, dtype=np.float64), 1e-3).astype(np.float32)
    if RL_PATTERN_INPUTS:
        # ±1 الگوها همان‌طور می‌مانند: استانداردسازی یک الگوی کمیاب را به عددی بزرگ (مثلاً ۱۵) تبدیل می‌کند
        mu[RL_RAW_COLS], sd[RL_RAW_COLS] = 0.0, 1.0
    obs_std = np.ascontiguousarray(obs, dtype=np.float32)
    obs_std -= mu
    obs_std /= sd
    data['obs'] = obs_std

    steps_total = int(total_steps or RL_TRAIN_STEPS)
    min_val_trades = max(5, int(MIN_TEST_TRADES) // 3)
    t0 = time.time()
    print(f"🏋️ {RL_ENSEMBLE} agent(s) × {steps_total:,} simulated steps ({RL_N_ENVS} parallel histories, "
          f"{RL_N_ACTIONS} actions: flat / hold / buy or sell × {_RL_KSL} stops × {_RL_KTP} targets), validation check "
          f"{RL_EVAL_POINTS}× each — best version of each agent on validation is kept")

    members, best_iters = [], []
    for m in range(RL_ENSEMBLE):
        def on_check(k, perf_v, m=m):
            training_state.update_progress(m * RL_EVAL_POINTS + k)
            progress((m + k / RL_EVAL_POINTS) / (RL_ENSEMBLE * passes + 0.5),
                     f"RL agent {m + 1}/{RL_ENSEMBLE}, check {k}/{RL_EVAL_POINTS}: val trades={perf_v['trades']} "
                     f"exp={perf_v['expectancy']:+.3f} ATR")
        best = _train_one_agent(data, obs_std, (tr_lo, tr_hi), (va_lo, va_hi), seed + 1000 * m, steps_total,
                                min_val_trades, cancelled, on_check, f"agent {m + 1}/{RL_ENSEMBLE}")
        if best is None:
            print("⏹️ RL training cancelled — nothing saved")
            result['cancelled'] = True
            return result
        print(f"   ✔ agent {m + 1}: kept check #{best['point']} after {best['iters'] * RL_N_ENVS:,} steps "
              f"(val score {best['score']:+.1f}, {best['perf']['trades']} trades)")
        net = NumpyQNet(obs_std.shape[1] + RL_POS_FEATURES, seed=seed)
        net.set_weights(best['weights'])
        members.append(net)
        best_iters.append(int(best['iters']))

    # حداقل برتری ورود و ضریب احتیاط (اختلاف نظر عامل‌ها) برای کل گروه، روی Validation
    progress(RL_ENSEMBLE / (RL_ENSEMBLE * passes + 0.5), "Tuning the entry margin and caution on validation...")
    kappas = RL_ENTRY_KAPPAS if len(members) > 1 else (0.0,)
    tuned = []
    for kp in kappas:
        for mg in RL_ENTRY_MARGINS:
            if cancelled():
                result['cancelled'] = True
                return result
            p, _ = evaluate_rl_policy(members, obs_std, data, va_lo, va_hi, margin=mg, min_n=min_val_trades,
                                      kappa=kp)
            tuned.append((p['score'], mg, kp, p))
            print(f"   🎚️ entry margin {mg:.2f} ATR, caution κ={kp:g} → val trades={p['trades']} "
                  f"({p['trades_per_day']:.1f}/day) exp={p['expectancy']:+.3f} ATR PF={p['profit_factor']:.2f} "
                  f"score={p['score']:+.1f}")
    # فقط حالت‌هایی که ربات واقعاً معامله می‌کند (نه «تقریباً هیچ معامله»)، و بین آن‌ها بیشترین مجموع سود Validation
    val_days = max(1, len(np.unique(np.asarray(data['ts'][va_lo:va_hi]) // 86400)))
    min_active = max(min_val_trades, int(np.ceil(RL_MIN_TRADES_PER_DAY * val_days)))
    active = [t for t in tuned if t[3]['trades'] >= min_active]
    if active:
        best_score, margin, kappa, perf_val = max(active, key=lambda x: (x[3]['total_atr'], -x[1], -x[2]))
        print(f"   ✔ chosen: entry margin {margin:.2f} ATR, caution κ={kappa:g} — highest validation total "
              f"({perf_val['total_atr']:+.1f} ATR) among settings with ≥{min_active} trades "
              f"(≈{RL_MIN_TRADES_PER_DAY:g}/day)")
    else:
        best_score, margin, kappa, perf_val = max(tuned, key=lambda x: (x[0], -x[1], -x[2]))
    perf_test, _ = evaluate_rl_policy(members, obs_std, data, te_lo, te_hi, margin=margin, min_n=MIN_TEST_TRADES,
                                      kappa=kappa)
    print(f"💰 Out-of-sample — RL scalper (ensemble of {len(members)}, entry margin {margin:.2f} ATR, caution "
          f"κ={kappa:g}, {rl_sl_text()}, max hold {RL_MAX_HOLD}, 1 ATR = 500-bar median ATR, spread + slippage, "
          f"fills at next bar open, stop first when stop and target touch in one bar):")
    print(f"   Validation: {_format_rl_perf(perf_val)}")
    print(f"   Test      : {_format_rl_perf(perf_test, choices=True)}")
    if perf_test.get('total_money') is not None:
        print(f"   Test total ≈ ${perf_test['total_money']:+.2f} at lot {LOT} over {span(te_lo, te_hi)}")

    quality_notes = []
    if perf_val['trades'] < min_val_trades:
        quality_notes.append("the agent mostly stays flat — it found no pattern that pays for the spread")
    if perf_test['trades'] < MIN_TEST_TRADES:
        quality_notes.append(f"few test trades ({perf_test['trades']} < {MIN_TEST_TRADES})")
    if perf_test['expectancy'] < MIN_TEST_EXPECTANCY_ATR:
        quality_notes.append(f"test expectancy {perf_test['expectancy']:+.3f} < {MIN_TEST_EXPECTANCY_ATR:g} ATR")
    if perf_test['profit_factor'] < MIN_TEST_PROFIT_FACTOR:
        quality_notes.append(f"test PF {perf_test['profit_factor']:.2f} < {MIN_TEST_PROFIT_FACTOR}")

    # ── آموزش دوباره روی کل داده تا آخرین کندل (همان seed، همان تعداد گام و برنامه‌ی اکتشاف) ──
    final = members
    refit = False
    if RL_REFIT_ALL_DATA:
        print(f"🔁 Re-training each agent on ALL data up to the last bar ({span(0, n)}) with the chosen step count, so "
              f"the deployed agent has seen the latest months (the test result above estimates this procedure)")
        final = []
        for m in range(RL_ENSEMBLE):
            def on_check2(k, _perf, m=m):
                training_state.update_progress((RL_ENSEMBLE + m) * RL_EVAL_POINTS + k)
                progress((RL_ENSEMBLE + m + k / RL_EVAL_POINTS) / (RL_ENSEMBLE * passes + 0.5),
                         f"RL agent {m + 1}/{RL_ENSEMBLE}: re-training on all data ({k}/{RL_EVAL_POINTS})")
            res_m = _train_one_agent(data, obs_std, (0, n), None, seed + 1000 * m, steps_total, min_val_trades,
                                     cancelled, on_check2, f"refit {m + 1}/{RL_ENSEMBLE}", stop_iter=best_iters[m])
            if res_m is None:
                print("⏹️ RL training cancelled — nothing saved")
                result['cancelled'] = True
                return result
            net = NumpyQNet(obs_std.shape[1] + RL_POS_FEATURES, seed=seed)
            net.set_weights(res_m['weights'])
            final.append(net)
            print(f"   ✔ agent {m + 1} re-trained on all data ({best_iters[m] * RL_N_ENVS:,} steps)")
        refit = True
    minutes = (time.time() - t0) / 60
    print(f"⏱️ RL training time: {minutes:.1f} min")

    # ── جایگزینی: عامل تازه همیشه، مگر Test آن ضررده باشد (بدون مقایسه با عامل فعلی روی بازه‌ای که آن عامل قبلاً
    #    دیده یا بارها برنده‌اش شده — آن مقایسه به نفع عامل قدیمی سوگیری دارد) ──
    old = load_rl_agent(verbose=False)
    old_keeps = int(old.meta.get('kept', 0) or 0) if old is not None else 0
    old_live = None
    if old is not None and old.nets[0].in_dim == obs_std.shape[1] + RL_POS_FEATURES:
        k0 = int(np.searchsorted(data['ts'], int(old.meta.get('data_end_ts', data['ts'][-1])), side='right'))
        if n - k0 >= 1000:
            raw = obs_std[k0:] * sd + mu
            old_obs = ((raw - old.mu) / old.sd).astype(np.float32)
            old_live, _ = evaluate_rl_policy(old, old_obs, _slice_data(data, k0, n, old_obs), 0, n - k0,
                                             margin=old.entry_margin, min_n=MIN_TEST_TRADES, kappa=old.kappa)
            print(f"   Current agent on the bars it has never seen ({span(k0, n)}): {_format_rl_perf(old_live)}")
    new_ok = perf_test['trades'] > 0 and perf_test['total_atr'] > 0
    deploy = old is None or new_ok or old_keeps >= RL_MAX_KEEPS

    metrics = {'kind': 'rl', 'val': perf_val, 'test': perf_test, 'entry_margin': margin, 'kappa': kappa,
               'members': len(final), 'current_agent_unseen': old_live, 'deployed': bool(deploy), 'refit': refit,
               'quality_warnings': quality_notes, 'minutes': minutes,
               'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    result['metrics'] = metrics
    last_training_report = metrics

    if not deploy:
        print(f"\n🛡️ New agent NOT deployed: its test result is a loss ({perf_test['total_atr']:+.1f} ATR) — keeping "
              f"the current agent ({old_keeps + 1}/{RL_MAX_KEEPS} trainings in a row; after that the newest agent is "
              f"deployed anyway so the agent never goes stale)")
        try:
            meta_old = dict(old.meta)
            meta_old['kept'] = old_keeps + 1
            _atomic_write_json(meta_old, rl_meta_path())
        except Exception as e:
            print(f"⚠️ Could not update the current agent's metadata: {e}")
        save_train_time(True, f"New RL agent lost money on test ({perf_test['total_atr']:+.1f} ATR) — current agent "
                              f"kept ({old_keeps + 1}/{RL_MAX_KEEPS})", status_file, agent_file, sig)
        result['kept'] = True
        print("=" * 60 + "\n")
        return result
    if old is not None and not new_ok:
        print(f"\n⚠️ The new agent lost money on test, but the current one was already kept {old_keeps} times in a "
              f"row — deploying the newest agent so it stays up to date")

    meta = {
        'signature': sig, 'trained_at': metrics['time'], 'in_dim': int(obs_std.shape[1] + RL_POS_FEATURES),
        'members': len(final), 'hidden': list(RL_HIDDEN), 'entry_margin': float(margin), 'kappa': float(kappa),
        'gamma': RL_GAMMA, 'nstep': RL_NSTEP, 'version': RL_VERSION,
        'window_p': RL_WINDOW_P, 'window_s': RL_WINDOW_S, 'max_hold': RL_MAX_HOLD,
        'sl': rl_sl_text(), 'actions': {'sl_atr': list(RL_SL_CHOICES), 'tp_atr': list(RL_TP_CHOICES)},
        'slippage': float(RL_SLIPPAGE_PRICE), 'label_mode': 'rl', 'ret_unit': 'ATR',
        'refit_all_data': refit, 'steps': [int(x * RL_N_ENVS) for x in best_iters], 'kept': 0,
        'val': perf_val, 'test': perf_test, 'quality_warnings': quality_notes,
        'data_end_ts': int(data['ts'][-1]), 'test_start_ts': int(data['ts'][te_lo]),
    }
    arrays = {}
    for m, net in enumerate(final):
        arrays.update(net.to_arrays(prefix=f"m{m}_"))
    arrays.update(mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    tmp = agent_file + ".tmp"
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, agent_file)
    _atomic_write_json(meta, rl_meta_path())
    if quality_notes:
        print(f"\n⚠️ Quality notes: {'; '.join(quality_notes)}")
    print(f"\n✅ RL scalper saved and deployed → {agent_file}")
    details = f"RL test {_format_rl_perf(perf_test)}"
    if quality_notes:
        details += " | Warnings: " + "; ".join(quality_notes)
    save_train_time(True, details, status_file, agent_file, sig)
    result.update(accepted=True, agent=RLAgent(final, mu, sd, meta), meta=meta)
    print("=" * 60 + "\n")
    return result


# ─────────────────────────── ذخیره / بارگذاری ───────────────────────────
def rl_agent_status():
    """(ok, reason) بدون بارگذاری وزن‌ها"""
    if not os.path.exists(rl_agent_path()):
        return False, "RL agent file missing"
    try:
        with open(rl_meta_path(), 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False, "RL agent has no metadata"
    if meta.get('signature') != rl_signature():
        return False, f"RL agent was trained for {meta.get('signature')} but current setup is {rl_signature()}"
    return True, "ok"


def load_rl_agent(verbose=True):
    ok, reason = rl_agent_status()
    if not ok:
        if verbose:
            print(f"⚠️ No usable RL agent: {reason} — training needed")
        return None
    try:
        with np.load(rl_agent_path()) as z:
            arrays = {k: z[k] for k in z.files}
        with open(rl_meta_path(), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        nets = [NumpyQNet.from_arrays(arrays, prefix=f"m{m}_") for m in range(int(meta.get('members', 1)))]
        agent = RLAgent(nets, arrays['mu'], arrays['sd'], meta)
        if any(net.in_dim != int(meta.get('in_dim', -1)) for net in nets):
            raise ValueError("input size mismatch")
    except Exception as e:
        if verbose:
            print(f"✗ RL agent load failed: {e}")
        return None
    if verbose:
        t = meta.get('test', {})
        print(f"✓ RL scalper loaded (trained {meta.get('trained_at')}, {len(agent.nets)} agents, entry margin "
              f"{agent.entry_margin:.2f} ATR, caution κ={agent.kappa:g}) | {meta.get('sl')} | max hold "
              f"{meta.get('max_hold')} | test exp={t.get('expectancy', 0):+.3f} ATR, PF={t.get('profit_factor', 0):.2f}, "
              f"{t.get('trades_per_day', 0):.1f} trades/day")
    return agent


# ─────────────────────────── تصمیم زنده ───────────────────────────
def _rl_frame(df):
    """کندل‌های fetch_ohlcv → قالب raw_bars (همان واحدهایی که در دیتابیس ذخیره می‌شود)"""
    point = _point_size.get(SYMBOL) or _symbol_point(SYMBOL)
    sp = (df['spread'].to_numpy(dtype=np.float64) * point) if ('spread' in df.columns and point > 0) \
        else np.zeros(len(df))
    return pd.DataFrame({
        'time': _bar_epochs(df), 'open': df['Open'].to_numpy(dtype=np.float64),
        'high': df['High'].to_numpy(dtype=np.float64), 'low': df['Low'].to_numpy(dtype=np.float64),
        'close': df['Close'].to_numpy(dtype=np.float64), 'spread_px': np.maximum(sp, 0.0),
        'volume': df['Volume'].to_numpy(dtype=np.float64) if 'Volume' in df.columns else np.nan,
    })


def _live_raw_frames(df_p, df_s):
    """
    کندل‌های هم‌تراز برای تصمیم/نمایش عامل: فقط کندل‌های بسته (عامل فقط روی کندل بسته تصمیم می‌گیرد)،
    و از تایم‌فریم دوم فقط کندل‌هایی که تا پایان آخرین کندل اصلی بسته می‌شوند (مثل آموزش).
    """
    df_p, _ = _drop_forming_bar(df_p, df_p, RL_TF_SECONDS, SYMBOL)
    if df_p is None or len(df_p) == 0 or df_s is None:
        return df_p, df_s
    df_s, _ = _drop_forming_bar(df_s, df_s, RL_SECONDARY_TF_SECONDS, SYMBOL)
    cutoff = int(_bar_epochs(df_p)[-1]) + RL_TF_SECONDS - RL_SECONDARY_TF_SECONDS
    return df_p, df_s[_bar_epochs(df_s) <= cutoff]


def rl_live_context(df_p, df_s):
    """
    ورودی و قیمت‌های لازم برای تصمیم روی آخرین کندل بسته‌ی df_p، با همان تابع آموزش (_rl_features).
    خروجی dict (obs, close, spread, atr, can_open, time) یا None اگر کندل کافی نیست.
    (حد ضرر/سود را خود عامل در get_rl_decision انتخاب می‌کند: ctx['sl'] / ctx['tp'])
    """
    if df_p is None or df_s is None or len(df_p) < rl_raw_warmup() + 1 or len(df_s) < RL_WINDOW_S:
        return None
    f = _rl_features(_rl_frame(df_p), _rl_frame(df_s), RL_TF_SECONDS, RL_SECONDARY_TF_SECONDS)
    if not f['valid'][-1]:
        return None
    return {'obs': f['obs'][-1], 'close': float(f['close'][-1]), 'spread': float(f['spread'][-1]),
            'atr': float(f['atr'][-1]), 'can_open': bool(f['can_open'][-1]), 'time': int(f['ts'][-1])}


def rl_live_market_obs(df_p, df_s):
    """همان ورودی شبیه‌ساز برای آخرین کندل df_p؛ None اگر کندل کافی نیست"""
    ctx = rl_live_context(df_p, df_s)
    return None if ctx is None else ctx['obs']


def rl_live_position_state(df_p, positions, spread_now=0.0, now=None, ctx=None):
    """
    وضعیت پوزیشن مثل شبیه‌ساز: (جهت فعلی، ورودی وضعیت پوزیشن _rl_pos_features).
    ارزش‌گذاری با کلوز آخرین کندل بسته (و اسپرد همان کندل)، نگه‌داری = تعداد کندل‌های بسته از کندل ورود،
    حد ضرر و حد سود از خود پوزیشن روی بروکر (همان‌هایی که عامل هنگام ورود انتخاب کرد).
    """
    if not positions:
        return 0, np.zeros(RL_POS_FEATURES, dtype=np.float32)
    pos = positions[0]
    direction = 1 if pos.type == mt5.POSITION_TYPE_BUY else -1
    entry = float(pos.price_open)
    atr_now = float(ctx['atr']) if ctx else (float(df_p['ATR_14'].iloc[-1]) if 'ATR_14' in df_p.columns else 0.0)
    c = float(ctx['close']) if ctx else float(df_p['Close'].iloc[-1])
    s = float(ctx['spread']) if ctx else float(spread_now or 0.0)
    sl_dist = abs(entry - float(pos.sl)) if float(pos.sl or 0) > 0 else 0.0
    if sl_dist <= 0:
        # پوزیشن بدون حد ضرر (دستی یا خطای بروکر): مثل دورترین حد ضرر عامل
        sl_dist = float(rl_sl_distance(atr_now, s, max(RL_SL_CHOICES))) or 1.0
    tp_dist = abs(float(pos.tp) - entry) if float(pos.tp or 0) > 0 else 0.0
    unreal = (c - entry) if direction == 1 else (entry - (c + s))
    held = bars_held_since(pos, df_p['time'].to_numpy(), tf_seconds=RL_TF_SECONDS) if 'time' in df_p.columns else 0
    pf = _rl_pos_features([direction], [unreal], [held], [sl_dist], [tp_dist], [atr_now], RL_MAX_HOLD)[0]
    return direction, pf


def get_rl_decision(agent, df_p, df_s, positions=None, spread_now=None,
                    publish=True, verbose=True, now=None, return_reason=False, return_context=False):
    """
    تصمیم عامل روی آخرین کندل بسته‌ی df_p: موقعیت هدف 0 / 1 / -1 (یا None اگر ممکن نیست).
    df_p/df_s باید فقط کندل‌های بسته‌ی هم‌تراز باشند (_live_raw_frames یا کندل‌های بسته‌ی حلقه‌ی ربات).
    خروجی: (target, current, q) — q = [Q بیرون، بهترین Q خرید، بهترین Q فروش]؛ با return_reason متن دلیل، با
    return_context ورودی‌های زنده هم (ctx['sl'] / ctx['tp'] = فاصله‌ی حد ضرر/سود انتخابی عامل برای ورود؛ tp=0 بدون حد سود).
    """
    confluence_signals = get_confluence_signals(df_p) if 'RSI_14' in df_p.columns else {}
    confluence_score, _ = calculate_confluence_score(confluence_signals)
    ctx = rl_live_context(df_p, df_s)
    if positions is None:
        positions = get_positions(SYMBOL)
    if spread_now is None:
        try:
            tick = mt5.symbol_info_tick(SYMBOL)
            spread_now = float(tick.ask) - float(tick.bid) if tick else 0.0
        except Exception:
            spread_now = 0.0
    current, pf = rl_live_position_state(df_p, positions, spread_now, now, ctx)

    target, q, q_sum = None, None, None
    if agent is None:
        reason = "No trained RL agent"
    elif ctx is None:
        reason = "Not enough bars for the RL agent"
    else:
        q, q_sd = agent.q_stats(ctx['obs'], pf)
        expired = current != 0 and pf[3] >= 1.0 - 1e-9
        valid = _rl_valid_mask([current], [ctx['can_open']], [expired])[0]
        a = _rl_pick_action(q, valid, current, agent.entry_margin, q_sd, agent.kappa)
        side, sl_d, tp_d = rl_action_exits(a, ctx['atr'], ctx['spread'])
        target = current if a == RL_A_HOLD else side
        ctx = dict(ctx, action=int(a), sl=(sl_d if side else None), tp=(tp_d if side else 0.0))
        if expired:
            action_txt = "EXIT (max hold)"
        elif a == RL_A_HOLD:
            action_txt = "HOLD"
        elif a == RL_A_FLAT:
            action_txt = "CLOSE" if current != 0 else "WAIT"
        else:
            action_txt = ("REVERSE → " if current != 0 else "") + rl_action_text(a, ctx['atr'], ctx['spread'])
        best_l = int(RL_LONG_ACTIONS[np.argmax(q[RL_LONG_ACTIONS])])
        best_s = int(RL_SHORT_ACTIONS[np.argmax(q[RL_SHORT_ACTIONS])])
        q_sum = _rl_q_summary(q)
        pats = rl_pattern_names_now(ctx['time'])
        pat_txt = f" | {RL_SECONDARY_TF_STR} patterns: {', '.join(pats[:4])}" if pats else ""
        reason = (f"RL: {action_txt}{pat_txt} | Q flat={q[RL_A_FLAT]:+.3f}"
                  + (f" hold={q[RL_A_HOLD]:+.3f}" if current != 0 else "")
                  + f" | best buy {q[best_l]:+.3f} ({rl_action_text(best_l)[5:]}) | best sell {q[best_s]:+.3f} "
                    f"({rl_action_text(best_s)[6:]}) (margin {agent.entry_margin:.2f} ATR, caution κ={agent.kappa:g})")

    if publish:
        gap = float(np.sort(q_sum)[-1] - np.sort(q_sum)[-2]) if q_sum is not None else 0.0
        atr = float(ctx['atr']) if ctx else (float(df_p['ATR_14'].iloc[-1]) if 'ATR_14' in df_p.columns else 0.0)
        shared_signal.update(
            signal=target or 0, confidence=gap, model_conf=gap, confluence_score=float(confluence_score),
            model_prob=gap, atr=atr, reason=reason,
            indicators={'signal_mode': 'rl', 'rl_q': [float(v) for v in q_sum] if q_sum is not None else None,
                        'rl_current': current, **confluence_signals})

    if verbose:
        print("\n" + "=" * 60)
        print("📊 SIGNAL ANALYSIS — source: RL scalper")
        print("=" * 60)
        pos_txt = {0: "flat", 1: "LONG", -1: "SHORT"}[current]
        print(f"📍 Position: {pos_txt} | unrealized={pf[2] * 2:+.2f} × its stop distance | "
              f"held={pf[3] * RL_MAX_HOLD:.0f}/{RL_MAX_HOLD} bars")
        print(f"🎯 {reason}")
        print("=" * 60 + "\n")
    out = (target, current, q_sum)
    if return_reason:
        out = out + (reason,)
    if return_context:
        out = out + (ctx,)
    return out


_display_agent_cache = {'stamp': None, 'agent': None}


def _frame_tf_seconds(df):
    """طول کندل‌های یک DataFrame (کوچک‌ترین فاصله‌ی زمانی بین کندل‌های آخر)"""
    try:
        d = np.diff(_bar_epochs(df)[-60:])
        d = d[d > 0]
        return int(d.min()) if len(d) else None
    except Exception:
        return None


def get_rl_signal(df_p=None, df_s=None):
    """
    برای داشبورد: عامل ذخیره‌شده را (با cache) بارگذاری و تصمیم روی آخرین کندل بسته را بدون انتشار برمی‌گرداند.
    اگر کندل‌ها داده نشوند یا تایم‌فریمشان با تایم‌فریم RL فرق کند، کندل‌های تایم‌فریم RL خودش گرفته می‌شود.
    """
    if (df_p is None or df_s is None or _frame_tf_seconds(df_p) != RL_TF_SECONDS
            or _frame_tf_seconds(df_s) != RL_SECONDARY_TF_SECONDS):
        df_p = fetch_ohlcv(SYMBOL, RL_TIMEFRAME, rl_live_bars() + 1)
        df_s = fetch_ohlcv(SYMBOL, RL_SECONDARY_TIMEFRAME, rl_live_bars_s() + 1)
        if df_p is None or df_s is None:
            return None, 0, None
    df_p, df_s = _live_raw_frames(df_p, df_s)
    try:
        stamp = (os.path.getmtime(rl_agent_path()), os.path.getmtime(rl_meta_path()))
    except OSError:
        stamp = None
    if stamp != _display_agent_cache['stamp']:
        _display_agent_cache['agent'] = load_rl_agent(verbose=False) if stamp else None
        _display_agent_cache['stamp'] = stamp
    return get_rl_decision(_display_agent_cache['agent'], df_p, df_s, publish=False, verbose=False)


def execute_rl_target(target, atr, daily_locked, spread_now, bar=None, sl_distance=None, can_open=True,
                      allow_entry=True, delay=0, tp_distance=0.0):
    """
    اجرای موقعیت هدف عامل: بستن پوزیشن ناهمسو، سپس باز کردن پوزیشن جدید با حد ضرر و حد سودی که خود عامل انتخاب کرده.
    atr: ATR تایم‌فریم اصلی (همان ورودی عامل)؛ sl_distance / tp_distance: فاصله‌ی حد ضرر / حد سود انتخابی عامل
    (ctx['sl'] / ctx['tp'] از get_rl_decision؛ tp_distance=0 یعنی بدون حد سود)؛
    can_open: فیلتر اسپرد کندل تصمیم (مثل شبیه‌ساز). bar: کندلی که تصمیم در آن اجرا می‌شود.
    allow_entry=False: کندل تصمیم کهنه است (entry_max_delay)؛ فقط خروج، بدون ورود.
    خروجی: (acted, opened) — acted یعنی پوزیشنی بسته یا باز شد، opened یعنی پوزیشن جدید باز شد.
    """
    acted = opened = False
    if target is None:
        return acted, opened
    positions = get_positions(SYMBOL)
    dirs = {1 if p.type == mt5.POSITION_TYPE_BUY else -1 for p in positions}
    current = dirs.pop() if len(dirs) == 1 else 0
    if positions and target == current:
        return acted, opened                           # نگه داشتن

    for pos in positions:
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        if (1 if is_buy else -1) != target:
            res = close_position(pos, reason="RL-exit" if target == 0 else "RL-reverse")
            acted = acted or (res is not None and res.retcode in _OK_RETCODES)
    if target == 0 or get_positions(SYMBOL):
        return acted, opened
    if not allow_entry:
        _stale_entry_note(bar, delay)
        return acted, opened
    if daily_locked:
        _say_once('entry', ('locked', bar), "🛑 RL entry ignored — daily loss limit reached")
        return acted, opened
    if not can_open or (RL_MAX_SPREAD_ATR > 0 and atr > 0 and spread_now > RL_MAX_SPREAD_ATR * atr):
        # در ساعت‌های کم‌نوسان این هر کندل تکرار می‌شود؛ ساعتی یک بار کافی است
        _say_once('entry', ('spread', int(bar or 0) // 3600),
                  f"⏸️ Spread too wide for scalping (> {RL_MAX_SPREAD_ATR}×ATR {atr:.5g}) — RL entries skipped "
                  f"while it lasts")
        return acted, opened
    if not np.isfinite(atr) or atr <= 0:
        _say_once('entry', ('atr', bar), "⏸️ Invalid ATR — RL entry skipped")
        return acted, opened
    if sl_distance is None:
        sl_distance = float(rl_sl_distance(atr, spread_now, max(RL_SL_CHOICES)))
    tp = float(tp_distance or 0.0)
    if _open_entry(target, atr, f"{BOT_COMMENT} RL", use_tp=tp > 0, sl_distance=sl_distance,
                   tp_distance=tp if tp > 0 else 0.0):
        acted = opened = True
    return acted, opened


def prepare_rl_agent(progress_callback=None):
    """
    برای برنامه: اتصال، دانلود تاریخچه‌ی عمیق کندل خام و آموزش عامل.
    اگر آموزش مدل معمولی در حال اجرا باشد، لغو می‌شود و RL منتظر می‌ماند تا متوقف شود.
    خروجی: 'ready' | 'failed' | 'cancelled' | 'skipped' | 'no_data' | 'mt5_error'
    """
    def progress(p, msg):
        print(f"[{int(p * 100):3d}%] {msg}")
        if progress_callback:
            try:
                progress_callback(p, msg)
            except Exception:
                pass

    progress(0.05, "Connecting to MT5...")
    if not ensure_mt5_connected():
        progress(1.0, "❌ MT5 connection failed")
        return "mt5_error"
    progress(0.15, f"Downloading raw {RL_TF_STR} + {RL_SECONDARY_TF_STR} candles (deep history)...")
    rl_sync_raw_bars()                     # فقط کندل خام تایم‌فریم‌های RL؛ به داده‌ی مدل دست نمی‌زند
    n = count_raw_bars(RL_PRIMARY_DB_PATH)
    if n < RL_MIN_BARS:
        progress(1.0, f"⚠️ Not enough raw bars ({n}/{RL_MIN_BARS})")
        return "no_data"
    if SIGNAL_MODE != "rl":
        progress(1.0, "Signal source changed — RL training skipped")
        return "cancelled"
    # آموزش مدل معمولی که هنوز در حال اجراست دیگر لازم نیست (منبع سیگنال RL است): متوقفش کن
    if current_training_kind() == 'model':
        request_training_cancel('model')
    progress(0.30, "Training RL scalper (simulation on your history)...")
    res = train_rl_agent(progress_callback=lambda f, m: progress(0.30 + 0.70 * f, m), wait=True)
    if res['accepted']:
        status, msg = "ready", "✅ RL scalper ready"
    elif res.get('kept'):
        status, msg = "ready", "✅ Current RL agent kept (the new one was not better on test)"
    elif res.get('cancelled'):
        status, msg = "cancelled", "⏹️ RL training cancelled"
    elif res.get('skipped'):
        status, msg = "skipped", "⏳ Another training is still running — RL training will run later"
    else:
        status, msg = "failed", "⚠️ RL training failed — nothing saved (see log)"
    progress(1.0, msg)
    return status


class _ServerClock:
    """
    ساعت سرور بروکر از روی تیک‌ها: اختلاف (زمان تیک − ساعت این کامپیوتر)؛ بیشترین مقدار چند دقیقه‌ی اخیر
    (تازه‌ترین تیک) بهترین تخمین است. برای بیدار شدن درست سر بسته شدن کندل.
    """
    WINDOW = 300

    def __init__(self):
        self.samples = deque()

    def update(self, tick):
        local = time.time()
        stamp = (int(getattr(tick, 'time_msc', 0) or 0) / 1000.0) or float(tick.time)
        self.samples.append((local, stamp - local))
        while self.samples and local - self.samples[0][0] > self.WINDOW:
            self.samples.popleft()

    def now(self):
        if not self.samples:
            return None
        return time.time() + max(off for _, off in self.samples)


def bar_close_wait_seconds(live_bar, clock, market_live, tf_seconds):
    """چقدر صبر کنیم تا کندل جاری (با طول tf_seconds) بسته شود — تصمیم درست بعد از بسته شدن کندل"""
    srv = clock.now()
    if srv is None or not market_live:
        return float(CHECK_INTERVAL)
    delay = (live_bar + tf_seconds) - srv + 0.25
    if delay <= 0:
        # کندل باید بسته شده باشد ولی هنوز تیکی در کندل جدید نیامده: کوتاه صبر کن (نه بی‌نهایت)
        return 0.5 if -delay < 15 else float(CHECK_INTERVAL)
    return float(min(max(delay, 0.25), CHECK_INTERVAL))


def rl_wait_seconds(live_bar, clock, market_live):
    """چقدر صبر کنیم تا کندل جاری بسته شود (تصمیم عامل درست بعد از بسته شدن کندل)"""
    return bar_close_wait_seconds(live_bar, clock, market_live, RL_TF_SECONDS)


# ═══════════════════════════════════════════════════════════════#
#                تریلینگ استاپ                                  #
# ═══════════════════════════════════════════════════════════════#
def apply_advanced_trailing_and_risk():
    """
    تریلینگ استاپ فقط برای پوزیشن‌های ربات.
    Fixed  : وقتی سود به Trigger رسید، SL به نقطه‌ی قفل سود (Lock) می‌رود.
    Dynamic: وقتی سود به Trigger رسید، SL با فاصله‌ی دلاری ثابت دنبال قیمت می‌آید.
    """
    ts_mode = TRAILING_STOP_MODE
    if ts_mode not in ("Fixed", "Dynamic"):
        return
    if SIGNAL_MODE == "rl":
        # در حالت یادگیری تقویتی خروج را خود عامل تصمیم می‌گیرد (شبیه‌سازش تریلینگ ندارد)
        return

    positions = get_positions(SYMBOL)
    if not positions:
        return
    symbol_info = mt5.symbol_info(SYMBOL)
    tick_info = mt5.symbol_info_tick(SYMBOL)
    if symbol_info is None or tick_info is None:
        return

    digits = symbol_info.digits
    point = symbol_info.point
    stops_level = float(symbol_info.trade_stops_level or 0) * point
    freeze_level = float(symbol_info.trade_freeze_level or 0) * point
    broker_min_distance = max(stops_level, freeze_level, point)

    for pos in positions:
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        current_profit = float(pos.profit)
        open_price = float(pos.price_open)
        current_sl = float(pos.sl or 0.0)
        current_price = float(tick_info.bid if is_buy else tick_info.ask)
        favorable_move = (current_price - open_price) if is_buy else (open_price - current_price)

        if favorable_move <= 0 or current_profit <= 0:
            continue

        # تبدیل دلار به فاصله‌ی قیمت بر اساس نسبت فعلی سود/حرکت همین پوزیشن
        price_per_dollar = favorable_move / current_profit

        if ts_mode == "Fixed":
            if current_profit < FIXED_TRIGGER_PROFIT:
                continue
            lock = FIXED_LOCK_PROFIT * price_per_dollar
            target_sl = round(open_price + lock if is_buy else open_price - lock, digits)
            distance = (current_price - target_sl) if is_buy else (target_sl - current_price)
            if distance < broker_min_distance:
                continue
        else:
            if current_profit < DYNAMIC_TRIGGER_PROFIT:
                continue
            dist = max(DYNAMIC_TRAIL_DISTANCE * price_per_dollar, broker_min_distance)
            target_sl = round(current_price - dist if is_buy else current_price + dist, digits)

        # SL فقط در جهت قفل سود بیشتر جابه‌جا می‌شود
        if is_buy and current_sl >= target_sl - point:
            continue
        if not is_buy and current_sl != 0.0 and current_sl <= target_sl + point:
            continue

        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": SYMBOL,
            "sl": target_sl,
            "tp": float(pos.tp or 0.0),
            "magic": MAGIC,
        })
        side = 'BUY' if is_buy else 'SELL'
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"🔄 [{ts_mode} TS] {side} #{pos.ticket} | profit=${current_profit:.2f} | "
                  f"SL -> {target_sl:.{digits}f}")
        else:
            print(f"❌ [{ts_mode} TS] {side} #{pos.ticket} FAILED | "
                  f"retcode={result.retcode if result else mt5.last_error()} | target_sl={target_sl:.{digits}f}")


def wait_with_live_trailing(seconds):
    """منتظر میمونه ولی هر ثانیه trailing رو چک میکنه (همه‌ی کارهای MT5 در یک ترد)"""
    end_time = time.time() + max(0.0, float(seconds))
    while not bot_stop_event.is_set():
        try:
            apply_advanced_trailing_and_risk()
        except Exception as e:
            print(f"Trailing wait error: {e}")
        remaining = end_time - time.time()
        if remaining <= 0:
            break
        bot_stop_event.wait(min(1.0, remaining))


def trailing_worker():
    """(سازگاری با نسخه‌ی قبل) تریلینگ مستقل؛ حلقه‌ی اصلی دیگر از آن استفاده نمی‌کند"""
    while not bot_stop_event.is_set():
        try:
            apply_advanced_trailing_and_risk()
        except Exception as e:
            print(f"Trailing worker error: {e}")
        bot_stop_event.wait(1.0)


# ========================
# Training Thread Class
# ========================
class AutoTrainingThread:
    def __init__(self):
        self.is_training = False
        self.training_queue = queue.Queue()
        self.thread = None

    def check_and_train(self):
        """چک کن و در صورت نیاز آموزش بده (مدل یا عامل یادگیری تقویتی، بسته به منبع سیگنال)"""
        kind = "rl" if SIGNAL_MODE == "rl" else "model"
        if self.is_training:
            return False
        if kind == "rl":
            if not should_train_now(rl_last_train_file(), rl_signature(), RL_RETRAIN_HOURS):
                return False
        elif not should_train_now():
            return False

        if kind == "rl":
            db_count = count_raw_bars(RL_PRIMARY_DB_PATH)
            need = RL_MIN_BARS
            if db_count < need:
                print(f"⚠️ Not enough raw bars for RL training ({db_count} < {need})")
                return False
        else:
            db_count = count_labeled_samples(PRIMARY_DB_PATH)
            if db_count < MIN_SAMPLES_FOR_TRAINING:
                print(f"⚠️ Not enough labeled data for training ({db_count} < {MIN_SAMPLES_FOR_TRAINING})")
                return False

        print(f"📚 Starting {'RL agent training' if kind == 'rl' else 'fine-tune'} with {db_count} "
              f"{'raw bars' if kind == 'rl' else 'labeled samples'}...")
        self.start_training(kind)
        return True

    def start_training(self, kind="model"):
        """شروع آموزش در Background — kind: 'model' یا 'rl'"""
        if self.is_training:
            print("⏳ Training already in progress...")
            return
        self.is_training = True
        self.thread = threading.Thread(target=self._train_in_background, args=(kind,), daemon=True)
        self.thread.start()
        print(f"🔄 {'RL agent training' if kind == 'rl' else 'Fine-tuning'} started in background...")

    def _train_in_background(self, kind="model"):
        start_time = datetime.now()
        try:
            print(f"\n🎯 AUTOMATIC {'RL TRAINING' if kind == 'rl' else 'FINE-TUNING'} STARTED "
                  f"({start_time:%Y-%m-%d %H:%M:%S})")
            if kind == "rl":
                # آموزش خود ربات: با Stop ربات متوقف می‌شود
                result = train_rl_agent(stop_event=bot_stop_event)
                if result['accepted']:
                    self.training_queue.put(("rl", result['agent'], result['meta']))
            else:
                result = train_and_validate()
                if result['accepted']:
                    # مدل و اسکیلرهای خودش با هم تحویل داده می‌شوند (نه از دیسک)
                    self.training_queue.put(("model", result['model'], result['scaler_p'],
                                             result['scaler_s'], result['meta']))
            minutes = (datetime.now() - start_time).total_seconds() / 60
            if result['accepted']:
                print(f"✅ Training complete in {minutes:.1f} min — new {kind} ready.")
            else:
                print(f"ℹ️ Training finished in {minutes:.1f} min — current {kind} kept.")
        except Exception as e:
            if kind == "rl":
                save_train_time(False, str(e), rl_last_train_file(), rl_agent_path(), rl_signature())
            else:
                save_train_time(success=False, message=str(e))
            print(f"❌ Training error: {e}")
        finally:
            self.is_training = False

    def get_new_model(self):
        """اگه مدل/عامل جدید آماده باشه: ('model', model, scaler_p, scaler_s, meta) یا ('rl', agent, meta)"""
        try:
            return self.training_queue.get_nowait()
        except queue.Empty:
            return None


def deployed_model_status():
    """
    بررسی می‌کند مدل GA ذخیره‌شده با نسخه و تنظیمات فعلی سازگار است یا نه (بدون ساختن مدل).
    خروجی: (ok, reason) — main_app برای تصمیم به ساخت مدل از این استفاده می‌کند.
    """
    path = ga_model_path()
    if not os.path.exists(path):
        return False, "GA pattern model file missing (Model mode now uses the genetic pattern strategy)"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            meta = json.load(f).get('meta', {})
    except (OSError, ValueError, AttributeError):
        return False, "GA pattern model file is unreadable"
    if meta.get('label_signature') != label_signature():
        return False, (f"model was trained for {meta.get('label_signature')} "
                       f"but current setup is {label_signature()}")
    if meta.get('decision_rule') != DECISION_RULE:
        return False, (f"model was built for decision rule {meta.get('decision_rule')!r}, "
                       f"current rule is {DECISION_RULE!r}")
    if int(meta.get('ga_version', -1)) != GA_VERSION:
        return False, f"GA model version {meta.get('ga_version')} differs from {GA_VERSION}"
    if (int(meta.get('primary_tf_seconds', -1)) != int(PRIMARY_TF_SECONDS)
            or int(meta.get('secondary_tf_seconds', -1)) != ga_htf_seconds()):
        return False, "GA model was built for other timeframes"
    return True, "ok"


def load_deployed_model(verbose=True):
    """
    بارگذاری مدل GA ذخیره‌شده؛ اگر ناسازگار یا خراب بود (None, None, None).
    خروجی سه‌تایی برای سازگاری: (مدل، جای‌نگهدار اسکیلر، جای‌نگهدار اسکیلر) — مدل GA اسکیلر ندارد.
    """
    ok, reason = deployed_model_status()
    if not ok:
        if verbose:
            print(f"⚠️ No usable GA pattern model: {reason} — (re)training needed")
        return None, None, None
    try:
        model = load_ga_model()
    except Exception as e:
        if verbose:
            print(f"✗ GA model load failed: {e}")
        return None, None, None
    if not model_matches_settings(model):
        if verbose:
            print("⚠️ Saved GA model does not match current settings — retraining needed")
        return None, None, None

    set_active_model_meta(model.meta)
    if verbose:
        meta = model.meta
        test = meta.get('test', {})
        unit = "$" if meta.get('label_mode') == "fixed" else "ATR"
        print(f"✓ GA pattern model loaded (trained {meta.get('trained_at')}) | {len(model)} rule(s) | "
              f"test exp={test.get('expectancy', 0):+.3f} {unit}, PF={test.get('profit_factor', 0):.2f}, "
              f"trades={test.get('trades', 0)}")
        for r in model.rules:
            print(f"   • {r.get('text') or _ga_rule_text(r)}")
    return model, GA_NO_SCALER, GA_NO_SCALER


# ========================
# MAIN LOOP
# ========================
class _TickWatch:
    """
    بازار زنده است؟ تیک باید در چند دقیقه‌ی اخیر (به ساعت همین کامپیوتر) عوض شده باشد.
    بعد از شروع ربات تا اولین تیک جدید، و وقتی تیک‌ها چند دقیقه متوقف شده‌اند (آخر هفته، تعطیلی روزانه،
    قطعی) معامله‌ی جدید انجام نمی‌شود — در تصمیم لحظه‌ای کندل «بسته‌ی قدیمی» دیگر نشانه‌ی بسته بودن بازار نیست.
    """
    STALE_AFTER = 180

    def __init__(self):
        self.last = None
        self.changed_at = None

    def update(self, tick):
        stamp = int(getattr(tick, 'time_msc', 0) or 0) or int(tick.time) * 1000
        now = time.time()
        if self.last is not None and stamp != self.last:
            self.changed_at = now
        self.last = stamp
        return self.changed_at is not None and now - self.changed_at <= self.STALE_AFTER


def _prepare_frames(df_p, df_s):
    """
    اندیکاتور + فیچر هر دو تایم‌فریم، و هم‌ترازی تایم‌فریم دوم با آخرین کندل اصلی (همان قاعده‌ی آموزش:
    فقط کندل‌هایی از تایم‌فریم دوم که تا پایان آخرین کندل اصلی بسته می‌شوند).
    اگر آخرین کندل اصلی در حال شکل‌گیری باشد، همه‌چیز طوری حساب می‌شود که انگار همین الان بسته شده.
    خروجی: (df_p, features_p, df_s, features_s)
    """
    df_p = compute_advanced_indicators(df_p)
    features_p = create_features_improved(df_p)
    df_s = compute_advanced_indicators(df_s)
    features_s = create_features_improved(df_s)
    cutoff = int(df_p['time'].iloc[-1]) + PRIMARY_TF_SECONDS - SECONDARY_TF_SECONDS
    mask = df_s['time'].to_numpy() <= cutoff
    return df_p, features_p, df_s[mask], features_s[mask]


def _bot_acted_in_bar(bar_open, tf_seconds=None):
    """
    آیا ربات در کندل bar_open خودش معامله‌ای باز/بسته کرده (مثلاً قبل از ری‌استارت)؟
    SL/TP/Stop-out بروکر حساب نمی‌شوند. برای رعایت «حداکثر یک اقدام در هر کندل» بعد از ری‌استارت وسط کندل.
    """
    try:
        tf = int(tf_seconds or PRIMARY_TF_SECONDS)
        deals = mt5.history_deals_get(datetime.fromtimestamp(bar_open, tz=timezone.utc),
                                      datetime.fromtimestamp(bar_open + 2 * tf, tz=timezone.utc))
    except Exception:
        return False
    broker = {getattr(mt5, 'DEAL_REASON_SL', 4), getattr(mt5, 'DEAL_REASON_TP', 5), getattr(mt5, 'DEAL_REASON_SO', 6)}
    return any(int(getattr(d, 'magic', 0)) == MAGIC and d.symbol == SYMBOL and int(d.time) >= bar_open
               and int(getattr(d, 'reason', -1)) not in broker for d in (deals or []))


_last_decision_log = {'key': None}


def _log_decision(bar, decision, reason, now):
    """تصمیم لحظه‌ای: یک خط فقط وقتی تصمیم عوض شد یا کندل جدید شروع شد (نه هر چند ثانیه)"""
    if _last_decision_log['key'] == (bar, decision):
        return
    _last_decision_log['key'] = (bar, decision)
    ts = datetime.fromtimestamp(int(now), tz=timezone.utc).strftime('%H:%M:%S')
    print(f"⏱️ {ts} (server) | {reason}")


def _save_closed_frames(closed_frames):
    """ذخیره‌ی داده‌ی آموزش مدل برای یک کندل بسته‌ی جدید (فیچر/لیبل؛ کندل خام RL جدا ذخیره می‌شود)"""
    if closed_frames is None:
        return
    dfp_c, fp_c, dfs_c, fs_c = closed_frames
    if len(dfs_c) >= WARMUP_BARS + TIMESTEPS_SECONDARY:
        save_training_data_multiclass(dfp_c, fp_c, db_path=PRIMARY_DB_PATH)
        save_secondary_training_data(dfs_c, fs_c, db_path=SECONDARY_DB_PATH)
    else:
        print("⚠️ Not enough secondary bars — this bar's data was not saved")


def _latest_closed_bar_time(timeframe):
    df = fetch_ohlcv(SYMBOL, timeframe, 3, closed_only=True)
    return int(df['time'].iloc[-1]) if df is not None and len(df) else None


def main_loop_improved(shutdown_mt5_on_exit=False):
    """
    حلقه اصلی بات.
    - مدل و عامل RL هر کدام تایم‌فریم‌های خودشان را دارند (مثلاً مدل M5+M15، عامل اسکلپ M1+M5).
    - عامل اسکلپ (RL): فقط روی کندل بسته‌ی تایم‌فریم خودش، درست بعد از بسته شدن هر کندل
      (ربات سر بسته شدن کندل بیدار می‌شود)؛ اجرا با اولین قیمت کندل بعد — همان چیزی که شبیه‌ساز فرض می‌کند.
    - مدل (الگوریتم ژنتیک روی الگوهای کندلی/کلاسیک): فقط روی کندل بسته، درست بعد از بسته شدن هر کندل
      (الگوی کندلی تا بسته شدن کندل قطعی نیست؛ لیبل‌ها هم ورود در Close کندل را فرض می‌کنند).
    - اندیکاتور با INTRABAR_DECISIONS (پیش‌فرض): هر CHECK_INTERVAL ثانیه روی کندل در حال شکل‌گیری
      حساب می‌شوند (انگار همین الان بسته می‌شود) و به‌محض خوب بودن شرایط معامله باز می‌شود؛
      حداکثر یک اقدام (باز/بستن با سیگنال) در هر کندل تا وسط کندل باز و بسته‌ی پشت‌سرهم نشود.
      وگرنه: تصمیم فقط یک بار بعد از بسته شدن هر کندل.
    - داده‌ی آموزش مدل (لیبل، فیچر) در هر حالتی، و کندل خام RL در حالت RL، فقط از کندل‌های بسته ذخیره می‌شود.
    - هر دور: تریلینگ، خروج زمانی (به زمان بازار)، حد ضرر روزانه
    - همه‌ی فراخوانی‌های MT5 در همین ترد (ترد آموزش به MT5 دست نمی‌زند)
    """
    print("=" * 60)
    print("🤖 ADVANCED AI TRADING BOT")
    print("=" * 60)

    trainer = AutoTrainingThread()
    reload_runtime_settings()

    try:
        init_mt5()
        init_database(db_path=PRIMARY_DB_PATH)
        init_database(db_path=SECONDARY_DB_PATH)

        print(f"📌 {SYMBOL} | Model {PRIMARY_TF_STR}+{SECONDARY_TF_STR} ({barrier_text()}, horizon={FUTURE_STEPS} bars) | "
              f"RL scalper {RL_TF_STR}+{RL_SECONDARY_TF_STR} ({rl_sl_text()}, max hold {RL_MAX_HOLD} bars) | "
              f"magic={MAGIC} | lot={LOT}")
        print(f"📁 Model data → {PROFILE_DATA_DIR} | RL data → {os.path.dirname(RL_PRIMARY_DB_PATH)}")
        print(f"🔀 Signal source: {SIGNAL_MODE_TEXT[SIGNAL_MODE]} "
              f"(changeable from the app; the modes are independent)")
        if N_HISTORY < live_history_bars():
            print(f"ℹ️ History Length {N_HISTORY} is short for warm-up + model window — "
                  f"using {live_history_bars()} bars for live decisions")

        last_train = get_last_train_time()
        if last_train:
            hours_ago = (datetime.now() - last_train).total_seconds() / 3600
            print(f"📅 Last successful fine-tune: {last_train:%Y-%m-%d %H:%M:%S} ({hours_ago:.1f}h ago)")

        # در حالت «فقط اندیکاتور» مدل لازم نیست؛ اگر بعداً حالت مدل انتخاب شود، داخل حلقه بارگذاری می‌شود
        model = scaler_p = scaler_s = None
        rl_agent = None
        if SIGNAL_MODE == "model":
            model, scaler_p, scaler_s = load_deployed_model()
        elif SIGNAL_MODE == "rl":
            rl_agent = load_rl_agent()
        last_model_try = time.time()
        last_mode = SIGNAL_MODE

        # جمع‌آوری اولیه: اگر داده کم است کل تاریخچه، وگرنه از آخرین کندل ذخیره‌شده تا الان (بدون شکاف)
        print("\n📥 Collecting initial data...")
        have = count_labeled_samples(PRIMARY_DB_PATH)
        if have < MIN_SAMPLES_FOR_TRAINING * 5 or history_backfill_needed():
            collect_training_data()
        else:
            n_p, n_s = catch_up_bars()
            if n_p > live_history_bars() or n_s > live_history_bars():
                print(f"ℹ️ Bot was offline for a while — downloading {n_p} {PRIMARY_TF_STR} / "
                      f"{n_s} {SECONDARY_TF_STR} bars to fill the gap")
            collect_training_data(primary_bars=n_p, secondary_bars=n_s)
        db_count = count_labeled_samples(PRIMARY_DB_PATH)
        print(f"📊 Primary DB (labeled): {db_count} samples")
        if SIGNAL_MODE == "rl":
            rl_sync_raw_bars()                   # کندل خام تایم‌فریم‌های RL تا همین الان

        # آموزش اولیه در پس‌زمینه تا مدیریت پوزیشن‌های باز متوقف نشود
        if (SIGNAL_MODE == "model" and model is None and db_count >= MIN_SAMPLES_FOR_TRAINING
                and should_train_now()):
            print("\n🎯 Initial training required (background)...")
            trainer.start_training("model")
        elif (SIGNAL_MODE == "rl" and rl_agent is None and count_raw_bars(RL_PRIMARY_DB_PATH) >= RL_MIN_BARS
                and should_train_now(rl_last_train_file(), rl_signature(), RL_RETRAIN_HOURS)):
            print("\n🎮 Initial RL agent training required (background)...")
            trainer.start_training("rl")

        tracked_tickets = {p.ticket for p in get_positions(SYMBOL)}
        risk_guard = DailyRiskGuard()
        tick_watch = _TickWatch()
        server_clock = _ServerClock()
        last_train_check = datetime.now()

        # آخرین کندل بسته‌ی فعلی را «پردازش‌شده» علامت بزن (داده‌اش را جمع‌آوری اولیه همین الان ذخیره کرد؛
        # در حالت کندل بسته: اولین تصمیم روی کندل بعدی است، تا روی کندلی که شاید معامله‌اش شده دوباره وارد نشود)
        last_bar_time = _latest_closed_bar_time(TIMEFRAME)
        last_rl_bar_time = _latest_closed_bar_time(RL_TIMEFRAME)
        # حداکثر یک اقدام با سیگنال در هر کندل؛ اگر ربات قبل از ری‌استارت در همین کندل معامله کرده، تا کندل بعد صبر کن
        act_tf = RL_TF_SECONDS if SIGNAL_MODE == "rl" else PRIMARY_TF_SECONDS
        now0 = _server_now(SYMBOL)
        cur_bar0 = (now0 - now0 % act_tf) if now0 else None
        last_action_bar = cur_bar0 if (cur_bar0 is not None and _bot_acted_in_bar(cur_bar0, act_tf)) else None

        print(f"\n🚀 Starting main trading loop (Model = GA patterns, decides right after each closed "
              f"{PRIMARY_TF_STR} bar; Indicators decide on {PRIMARY_TF_STR} "
              f"{'in real time on the forming bar' if INTRABAR_DECISIONS else 'on each closed bar'}; "
              f"RL scalper decides right after each closed {RL_TF_STR} bar)...\n")

        while not bot_stop_event.is_set():
            try:
                reload_runtime_settings()

                if not ensure_mt5_connected():
                    wait_with_live_trailing(CHECK_INTERVAL)
                    continue

                # ── مدل جدید از ترد آموزش ──
                new = trainer.get_new_model()
                if new is not None:
                    if new[0] == "rl":
                        rl_agent = new[1]
                        print("\n✨ New RL agent deployed!")
                    else:
                        _, model, scaler_p, scaler_s, meta = new
                        set_active_model_meta(meta)
                        print(f"\n✨ New GA pattern model deployed! ({len(model)} rule(s))")
                    gc.collect()

                # ── تغییر منبع سیگنال از برنامه ──
                if SIGNAL_MODE != last_mode:
                    if SIGNAL_MODE == "rl":
                        print(f"\n🔀 Signal source switched to {SIGNAL_MODE_TEXT['rl']} ({RL_TF_STR}+"
                              f"{RL_SECONDARY_TF_STR}) — first decision on the next closed {RL_TF_STR} bar")
                        rl_sync_raw_bars()           # داده‌ی خام RL ممکن است از آخرین بار کهنه شده باشد
                        last_rl_bar_time = _latest_closed_bar_time(RL_TIMEFRAME)
                    else:
                        live_next = SIGNAL_MODE == "indicators" and INTRABAR_DECISIONS
                        print(f"\n🔀 Signal source switched to {SIGNAL_MODE_TEXT[SIGNAL_MODE]} — applies "
                              f"{'from the next check' if live_next else 'from the next closed bar'}")
                    last_mode = SIGNAL_MODE
                    last_model_try = 0.0

                # ── حالت مدل بدون مدل: هر دقیقه دنبال مدل معتبر روی دیسک بگرد (مثلاً بعد از Setup برنامه) ──
                if SIGNAL_MODE == "model" and model is None and time.time() - last_model_try >= 60:
                    last_model_try = time.time()
                    if deployed_model_status()[0]:
                        model, scaler_p, scaler_s = load_deployed_model()
                        if model is not None:
                            print("✨ Model found on disk and loaded")
                if SIGNAL_MODE == "rl" and rl_agent is None and time.time() - last_model_try >= 60:
                    last_model_try = time.time()
                    if rl_agent_status()[0]:
                        rl_agent = load_rl_agent()
                        if rl_agent is not None:
                            print("✨ RL agent found on disk and loaded")

                # ── مدیریت ریسک (هر دور) ──
                daily_locked = risk_guard.check(SYMBOL)
                tracked_tickets = check_closed_positions(tracked_tickets)

                # ── آموزش دوره‌ای (در ترد جدا) — در حالت مدل یا یادگیری تقویتی ──
                if (datetime.now() - last_train_check).total_seconds() >= 300:
                    if SIGNAL_MODE in ("model", "rl"):
                        trainer.check_and_train()
                    last_train_check = datetime.now()

                # منبع سیگنال یک بار برای کل این دور (برنامه ممکن است از ترد دیگری وسط دور عوضش کند)
                mode = SIGNAL_MODE

                # ── کندل‌های تایم‌فریم مدل (داده‌ی آموزش مدل در هر حالت + تصمیم مدل/اندیکاتور) ──
                n_bars = live_history_bars()            # پنجره‌ی بلند: فقط برای تشخیص الگوهای مدل
                n_feat = feature_history_bars()         # اندیکاتور/فیچر/ذخیره‌ی داده: همان پنجره‌ی کوتاه
                df_p_all = fetch_ohlcv(SYMBOL, TIMEFRAME, n_bars + 1)
                df_s_all = fetch_ohlcv(SYMBOL, SECONDARY_TIMEFRAME, n_feat + 1)
                tick = mt5.symbol_info_tick(SYMBOL)
                if (df_p_all is None or df_s_all is None or tick is None
                        or len(df_p_all) < max(200, WARMUP_BARS + TIMESTEPS + 10)):
                    wait_with_live_trailing(CHECK_INTERVAL)
                    continue
                now = int(tick.time)
                market_live = tick_watch.update(tick)
                server_clock.update(tick)
                spread_now = float(tick.ask) - float(tick.bid)
                live_bar = int(df_p_all['time'].iloc[-1])     # کندلی که آخرین تیک در آن است
                df_p_closed, _ = _drop_forming_bar(df_p_all, df_p_all, PRIMARY_TF_SECONDS, SYMBOL)

                # یک بار برای هر کندل بسته‌ی جدید مدل: کندل‌های بسته (ذخیره‌ی داده‌ی مدل آخر همین دور)
                closed_frames = None
                closed_bar_time = int(df_p_closed['time'].iloc[-1])
                new_closed_bar = closed_bar_time != last_bar_time
                if new_closed_bar:
                    last_bar_time = closed_bar_time
                    df_s_closed, _ = _drop_forming_bar(df_s_all, df_s_all, SECONDARY_TF_SECONDS, SYMBOL)
                    closed_frames = _prepare_frames(df_p_closed.iloc[-n_feat:], df_s_closed)

                wait_s = CHECK_INTERVAL
                if mode == "rl":
                    # ── عامل اسکلپ: کندل‌های تایم‌فریم خودش؛ فقط روی کندل بسته، درست بعد از بسته شدن آن ──
                    df_rp = fetch_ohlcv(SYMBOL, RL_TIMEFRAME, rl_live_bars() + 1)
                    df_rs = fetch_ohlcv(SYMBOL, RL_SECONDARY_TIMEFRAME, rl_live_bars_s() + 1)
                    if df_rp is not None and df_rs is not None and len(df_rp) > RL_WINDOW_P:
                        rl_live_bar = int(df_rp['time'].iloc[-1])
                        if market_live:
                            # عامل خودش بعد از RL_MAX_HOLD کندل می‌بندد؛ این فقط پشتیبان است اگر کندلی از دست برود
                            apply_time_exit(SYMBOL, df_rp['time'].to_numpy(), max_bars=RL_MAX_HOLD + 1, force=True,
                                            now=now, tf_seconds=RL_TF_SECONDS)
                        dfp_c, dfs_c = _live_raw_frames(df_rp, df_rs)
                        rl_closed = int(dfp_c['time'].iloc[-1]) if dfp_c is not None and len(dfp_c) else None
                        if rl_closed is not None and rl_closed != last_rl_bar_time:
                            last_rl_bar_time = rl_closed
                            target, _, _, reason, ctx = get_rl_decision(
                                rl_agent, dfp_c, dfs_c, spread_now=spread_now, verbose=False,
                                return_reason=True, return_context=True)
                            _log_decision(rl_closed, target, reason, now)
                            if not market_live:
                                if target:
                                    _say_once('market', rl_live_bar, "⏸️ Market closed or no fresh ticks — no trading")
                            elif rl_live_bar != last_action_bar and ctx is not None:
                                rl_delay = now - (rl_closed + RL_TF_SECONDS)
                                acted, opened = execute_rl_target(target, ctx['atr'], daily_locked, spread_now,
                                                                  bar=rl_live_bar, sl_distance=ctx.get('sl'),
                                                                  tp_distance=ctx.get('tp', 0.0),
                                                                  can_open=ctx['can_open'],
                                                                  allow_entry=rl_delay <= entry_max_delay(RL_TF_SECONDS),
                                                                  delay=rl_delay)
                                if acted:
                                    last_action_bar = rl_live_bar
                                if opened:
                                    tracked_tickets |= {p.ticket for p in get_positions(SYMBOL)}
                            # کندل خام RL بعد از تصمیم ذخیره می‌شود، تا اجرای معامله معطل نشود
                            save_raw_bars(dfp_c, RL_PRIMARY_DB_PATH, SYMBOL, RL_TF_SECONDS)
                            save_raw_bars(dfs_c, RL_SECONDARY_DB_PATH, SYMBOL, RL_SECONDARY_TF_SECONDS)
                        wait_s = rl_wait_seconds(rl_live_bar, server_clock, market_live)
                else:
                    # ── خروج زمانی مدل (هر دور، به زمان بازار؛ در حالت «فقط اندیکاتور» اعمال نمی‌شود) ──
                    if market_live and mode == "model":
                        apply_time_exit(SYMBOL, df_p_all['time'].to_numpy(), now=now)

                    # ── مدل/اندیکاتور ──
                    # مدل GA: فقط روی کندل بسته (الگوی کندلی تا بسته شدن کندل قطعی نیست و لیبل‌ها هم ورود در
                    # Close کندل را فرض می‌کنند)، درست بعد از بسته شدن کندل. اندیکاتورها: طبق INTRABAR_DECISIONS.
                    intrabar = INTRABAR_DECISIONS and mode == "indicators"
                    frames = None
                    if intrabar:
                        frames = _prepare_frames(df_p_all.iloc[-(n_feat + 1):], df_s_all)
                    elif new_closed_bar:
                        frames = closed_frames
                    if frames is not None:
                        dfp, fp, dfs, fs = frames
                        if mode == "indicators" and len(dfs) < WARMUP_BARS + TIMESTEPS_SECONDARY:
                            _say_once('frames', live_bar, "⚠️ Not enough secondary bars — no decision on this bar")
                        else:
                            atr = float(dfp['ATR_14'].iloc[-1])
                            signal, confidence, reason, exit_params = get_combined_signal_multiclass(
                                model, scaler_p, scaler_s, dfp, fp, fs,
                                verbose=not intrabar, mode=mode, return_reason=True, return_exit=True,
                                df_long=df_p_closed if mode == "model" else None)
                            if intrabar:
                                _log_decision(live_bar, signal, reason, now)

                            # حداکثر یک اقدام با سیگنال (باز/بستن/برعکس) در هر کندل؛
                            # خروج با SL/TP/خروج زمانی/حد ضرر روزانه این محدودیت را ندارد
                            if not market_live:
                                if signal:
                                    _say_once('market', live_bar, "⏸️ Market closed or no fresh ticks — no trading")
                            elif live_bar != last_action_bar:
                                # تصمیم روی کندل بسته: اگر بازار بعد از بسته شدن آن کندل تعطیل بوده، ورود نکن
                                delay = 0 if intrabar else now - (closed_bar_time + PRIMARY_TF_SECONDS)
                                acted, opened = execute_signal(signal, confidence, atr, daily_locked, spread_now,
                                                               bar=live_bar,
                                                               allow_entry=delay <= entry_max_delay(PRIMARY_TF_SECONDS),
                                                               delay=delay, exit_params=exit_params)
                                if acted:
                                    last_action_bar = live_bar
                                if opened:
                                    # اجتماع، نه جایگزینی: پوزیشن‌هایی که همین الان بسته شدند باید دور بعد لاگ شوند
                                    tracked_tickets |= {p.ticket for p in get_positions(SYMBOL)}
                    if not intrabar:
                        # بیدار شدن درست سر بسته شدن کندل بعدی (تصمیم روی کندل بسته بدون تأخیر CHECK_INTERVAL)
                        wait_s = bar_close_wait_seconds(live_bar, server_clock, market_live, PRIMARY_TF_SECONDS)

                # داده‌ی آموزش مدل (در هر حالت) بعد از تصمیم‌ها، تا اجرای معامله معطل نشود
                _save_closed_frames(closed_frames)
                wait_with_live_trailing(wait_s)

            except Exception as e:
                print(f"✗ Loop error: {e}")
                traceback.print_exc()
                bot_stop_event.wait(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n⚠️ Stopped by user")
        if trainer.is_training and trainer.thread is not None:
            print("⏳ Waiting for training to complete...")
            trainer.thread.join(timeout=10)

    finally:
        # وقتی از GUI اجرا می‌شود اتصال MT5 مشترک است؛ نباید قطعش کنیم
        if shutdown_mt5_on_exit:
            mt5.shutdown()
        print("✓ Bot stopped")


# ========================
# RUN
# ========================
if __name__ == "__main__":
    main_loop_improved(shutdown_mt5_on_exit=True)
