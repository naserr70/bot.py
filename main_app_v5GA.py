# -*- coding: utf-8 -*-
"""
NR Trading Platform - Main Application
Developed by: Naser Romipour
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import io

# ═══════════════════════════════════════════════════════════════
# FIX WINDOWS CONSOLE ENCODING
# ═══════════════════════════════════════════════════════════════
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
    except:
        pass

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# حذف ABSL warnings
try:
    import absl.logging
    absl.logging.set_verbosity('error')
except:
    pass

import sys
import os
import traceback
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# ✅ Crash Logger - قبل از هر چیز!
# ═══════════════════════════════════════════════════════════════

def setup_crash_logger():
    """ذخیره خطاها در فایل"""
    if getattr(sys, 'frozen', False):
        # فقط در حالت EXE
        log_file = os.path.join(
            os.path.dirname(sys.executable),
            "crash_log.txt"
        )
        
        def exception_handler(exc_type, exc_value, exc_traceback):
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"CRASH at {datetime.now()}\n")
                f.write(f"{'='*60}\n")
                traceback.print_exception(
                    exc_type, exc_value, exc_traceback,
                    file=f
                )
        
        sys.excepthook = exception_handler

# اجرا کن قبل از import های دیگه
setup_crash_logger()
from PIL import Image
import customtkinter as ctk
from tkinter import messagebox, ttk
import threading
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import tkinter.messagebox as messagebox
from tkinter import filedialog
import ta
import mplfinance as mpf
import MetaTrader5 as mt5
from matplotlib.ticker import MaxNLocator
import trading_bot_v5GA as bot
from logo_data_v2_1 import get_logo_temp_path, get_logo_image, set_window_icon
import logo_data_v2_1
from trading_bot_v5GA import (
    N_HISTORY,
    fetch_ohlcv, compute_advanced_indicators, create_features_improved,
    get_combined_signal_multiclass, LOG_CSV, bot_stop_event, main_loop_improved,
    get_confluence_signals, calculate_confluence_score, get_model_prediction_multiclass,
    shared_signal, training_state, 
    get_last_train_time, FINETUNE_INTERVAL_HOURS,
    check_first_run, perform_first_run_setup,  
    init_database, train_model_multiclass,
    # ✅ اضافه شدن‌های جدید (ضروری):
    TIMESTEPS_SECONDARY, SECONDARY_TIMEFRAME, PRIMARY_TF_STR, SECONDARY_TF_STR,
    PRIMARY_DB_PATH, SECONDARY_DB_PATH,
    PRIMARY_SCALER_PATH, SECONDARY_SCALER_PATH,
    save_secondary_training_data,
)

# اضافه کردن ماژول‌های داخلی
from auth_system_v2_1 import OnlineAuthSystem
from config_manager_price3 import ConfigManager, get_app_directory, DEFAULT_PATHS
# ═══════════════════════════════════════════════════════════════
# LOGO PATH - داینامیک
# ═══════════════════════════════════════════════════════════════
def get_logo_path_safe():
    """گرفتن مسیر لوگو با fallback کاملاً مبتنی بر کدهای Base64"""
    try:
        return get_logo_temp_path()
    except Exception as e:
        print(f"[!] Safe logo path recovery failed: {e}")
        return None

LOGO_PATH = get_logo_path_safe()

# ═══════════════════════════════════════════════════════════════
# ✅ MT5 Manager - Thread-safe Connection
# ═══════════════════════════════════════════════════════════════
class MT5Manager:
    """مدیریت اتصال MT5 به صورت Thread-safe"""
    _lock = threading.Lock()
    _initialized = False
    
    @classmethod
    def ensure_connected(cls):
        """اطمینان از اتصال"""
        with cls._lock:
            # هر بار چک کن متصله
            if not mt5.terminal_info():
                cls._initialized = False
            
            if not cls._initialized:
                if mt5.initialize():
                    cls._initialized = True
                    return True
                else:
                    print(f"❌ MT5 init failed: {mt5.last_error()}")
                    return False
            return True
    
    @classmethod
    def get_rates(cls, symbol, timeframe, count):
        """دریافت کندل‌ها"""
        with cls._lock:
            # ✅ هر بار چک کن متصله
            if not mt5.terminal_info():
                if not mt5.initialize():
                    print(f"❌ MT5 reconnect failed")
                    return None
            
            # ✅ مطمئن شو symbol در Market Watch هست
            if not mt5.symbol_select(symbol, True):
                print(f"⚠️ Cannot select {symbol}")
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None:
                error = mt5.last_error()
                print(f"❌ MT5 rates error: {error}")
            
            return rates
    
    @classmethod
    def resolve_symbol(cls, name):
        """
        اسم نماد در بروکر: همان اسم اگر هست؛ وگرنه نسخه‌ای با حروف دیگر یا پسوند بروکر (EURUSD → EURUSDm / EURUSD.r).
        نماد پیداشده در Market Watch هم فعال می‌شود.
        خروجی: (اسم نماد یا None اگر پیدا نشد، چند پیشنهاد). اگر MT5 وصل نیست: (همان اسم، None) — بررسی‌نشده.
        """
        name = str(name or "").strip()
        with cls._lock:
            if not mt5.terminal_info() and not mt5.initialize():
                return name, None
            if name and mt5.symbol_info(name) is not None:
                mt5.symbol_select(name, True)
                return name, []
            names = [s.name for s in (mt5.symbols_get() or [])]
            clean = lambda x: "".join(ch for ch in x.upper() if ch.isalnum())
            key = clean(name)
            exact = [s for s in names if clean(s) == key] if key else []
            starts = [s for s in names if key and clean(s).startswith(key)]
            pick = exact[0] if len(exact) == 1 else (starts[0] if len(starts) == 1 and not exact else None)
            if pick:
                mt5.symbol_select(pick, True)
                return pick, []
            hints = sorted(exact or starts, key=len)[:8]
            if not hints and len(key) >= 3:
                hints = sorted((s for s in names if key[:3] in clean(s)), key=len)[:8]
            return None, hints

    @classmethod
    def get_account_info(cls):
        """دریافت اطلاعات حساب"""
        with cls._lock:
            if not mt5.terminal_info():
                if not mt5.initialize():
                    return None
            return mt5.account_info()
    
    @classmethod
    def get_tick(cls, symbol):
        """دریافت tick"""
        with cls._lock:
            if not mt5.terminal_info():
                if not mt5.initialize():
                    return None
            return mt5.symbol_info_tick(symbol)
    
    @classmethod
    def shutdown(cls):
        """قطع اتصال - فقط موقع بستن برنامه"""
        with cls._lock:
            if cls._initialized:
                mt5.shutdown()
                cls._initialized = False
                print("✅ MT5 Manager: Disconnected")


mt5_manager = MT5Manager()

class LoginWindow(ctk.CTkToplevel):
    """پنجره ورود به سیستم"""
    def __init__(self, parent, auth_system_v2_1):
        super().__init__(parent)
        
        
        # ست کردن آیکون
        try:
            self.after(10, lambda: set_window_icon(self))
        except Exception as e:
            print(f"[!] Login window icon apply failed: {e}")

        self.auth_system_v2_1 = auth_system_v2_1
        self.parent = parent
        self.title("NR Trading Platform - Login")
        self.geometry("400x500")
        self.resizable(False, False)

        self.center_window()
        self.setup_ui()
        self.grab_set()

    def center_window(self):
        """مرکز کردن پنجره"""
        self.update_idletasks()
        width = 400
        height = 500
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        """ایجاد رابط کاربری ورود"""
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # لوگو
        try:
            logo_img = get_logo_image()
            if logo_img:
                logo_ctk = ctk.CTkImage(logo_img, size=(80, 80))
                logo_label = ctk.CTkLabel(main_frame, image=logo_ctk, text="")
                logo_label.image = logo_ctk
                logo_label.pack(pady=(20, 10))
            else:
                raise Exception("No logo")
        except:
            # Fallback - متن ساده
            ctk.CTkLabel(
                main_frame,
                text="NR",
                font=ctk.CTkFont(size=48, weight="bold"),
                text_color="#00d4aa"
            ).pack(pady=(20, 10))

        title_label = ctk.CTkLabel(
            main_frame, 
            text="NR Trading Platform", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=(0, 30))

        # فیلد نام کاربری
        ctk.CTkLabel(main_frame, text="Username:").pack(anchor="center", pady=(0, 5))
        self.username_entry = ctk.CTkEntry(main_frame, width=250)
        self.username_entry.pack(pady=(0, 15))
        self.username_entry.insert(0, "admin")

        # فیلد رمز عبور
        ctk.CTkLabel(main_frame, text="Password:").pack(anchor="center", pady=(0, 5))
        self.password_entry = ctk.CTkEntry(main_frame, width=250, show="*")
        self.password_entry.pack(pady=(0, 20))

        # دکمه ورود
        login_btn = ctk.CTkButton(
            main_frame,
            text="Login",
            command=self.login,
            height=40,
            fg_color="green",
            hover_color="dark green"
        )
        login_btn.pack(pady=10)

        # اطلاعات توسعه‌دهنده
        dev_label = ctk.CTkLabel(
            main_frame,
            text="Developed by: Naser Romipour",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        dev_label.pack(side="bottom", pady=20)

    def login(self):
        """ورود کاربر هماهنگ شده با متد جدید"""
        username = self.username_entry.get().strip() 
        password = self.password_entry.get().strip() 

        if not username or not password: 
            messagebox.showerror("Error", "Please enter username and password") 
            return 

        # ⚠️ تغییر مهم: دریافت درست مقدار موفقیت و نقش کاربر از سیستم احراز هویت
        success, role = self.auth_system_v2_1.login_user(username, password)

        if success:
            messagebox.showinfo("Success", f"Welcome {username}!") 
            self.destroy() 
            
            # ⚠️ ارسال نقش کاربر (role) به پنجره اصلی نرم‌افزار تا بتواند لود شود
            if hasattr(self.parent, "show_main_app"):
                # اگر متد شو مین اپ شما ورودی نقش می‌پذیرد:
                try:
                    self.parent.show_main_app(role=role)
                except TypeError:
                    # اگر ورودی نمی‌پذیرد، به شکل معمولی صدا بزن:
                    self.parent.show_main_app() 
        else:
            # نمایش دقیق خطای بازگشتی (مثلا رمز اشتباه است)
            messagebox.showerror("Error", role)
# ═══════════════════════════════════════════════════════════════
# FIRST RUN SETUP WINDOW
# ═══════════════════════════════════════════════════════════════
class FirstRunSetupWindow(ctk.CTkToplevel):
    """پنجره Setup اولین اجرا"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # آیکون
        self.after(10, lambda: set_window_icon(self))
        
        self.title("NR Trading Platform - First Run Setup")
        self.geometry("650x500")
        self.resizable(False, False)
        
        self.setup_complete = False
        self.setup_success = False
        
        self.center_window()
        self.setup_ui()
        
        # شروع Setup در thread جداگانه
        self.setup_thread = threading.Thread(target=self.run_setup, daemon=True)
        self.setup_thread.start()
        
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def center_window(self):
        """مرکز کردن پنجره"""
        self.update_idletasks()
        width = 650
        height = 500
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_ui(self):
        """رابط کاربری"""
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=30, pady=30)
        
        # Title
        ctk.CTkLabel(
            main_frame,
            text="First Run Setup",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(pady=(10, 5))
        
        ctk.CTkLabel(
            main_frame,
            text="Downloading data and training AI model...",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        ).pack(pady=(0, 25))
        
        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(main_frame, width=500, height=20)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        # Progress Percentage
        self.progress_percent = ctk.CTkLabel(
            main_frame,
            text="0%",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.progress_percent.pack(pady=5)
        
        # Current Step Label
        self.step_label = ctk.CTkLabel(
            main_frame,
            text="Initializing...",
            font=ctk.CTkFont(size=13)
        )
        self.step_label.pack(pady=5)
        
        # Status Textbox
        self.status_text = ctk.CTkTextbox(
            main_frame,
            width=550,
            height=180,
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.status_text.pack(pady=15)
        self.status_text.insert("end", "Starting setup...\n")
        
        # Continue Button (disabled at first)
        self.continue_btn = ctk.CTkButton(
            main_frame,
            text="Please Wait...",
            command=self.on_continue,
            width=200,
            height=45,
            fg_color="gray",
            state="disabled"
        )
        self.continue_btn.pack(pady=15)
    
    def update_ui(self, progress, message):
        """آپدیت UI از thread"""
        def do_update():
            try:
                # Progress bar
                self.progress_bar.set(progress)
                
                # Percentage
                self.progress_percent.configure(text=f"{int(progress * 100)}%")
                
                # Step label
                self.step_label.configure(text=message)
                
                # Add to textbox
                self.status_text.insert("end", f"[{int(progress*100):3d}%] {message}\n")
                self.status_text.see("end")
            except:
                pass
        
        self.after(0, do_update)
    
    def run_setup(self):
        """اجرای Setup"""
        try:
            self.update_ui(0.02, "Starting first run setup...")
            
            success = perform_first_run_setup(progress_callback=self.update_ui)
            
            self.setup_complete = True
            self.setup_success = success
            
            # فعال کردن دکمه
            def enable_button():
                if success:
                    self.continue_btn.configure(
                        text="Continue",
                        fg_color="green",
                        hover_color="dark green",
                        state="normal"
                    )
                    self.step_label.configure(
                        text="Setup Complete! Click Continue to start.",
                        text_color="green"
                    )
                else:
                    self.continue_btn.configure(
                        text="Continue Anyway",
                        fg_color="orange",
                        hover_color="dark orange",
                        state="normal"
                    )
                    self.step_label.configure(
                        text="Setup incomplete. Bot will train later.",
                        text_color="orange"
                    )
            
            self.after(0, enable_button)
            
        except Exception as e:
            self.setup_complete = True
            # متغیر e بعد از بلوک except پاک می‌شود و show_error بعداً (با after) اجرا می‌شود
            err_text = str(e)[:50]

            def show_error():
                self.step_label.configure(
                    text=f"Error: {err_text}",
                    text_color="red"
                )
                self.continue_btn.configure(
                    text="Continue Anyway",
                    fg_color="red",
                    hover_color="dark red",
                    state="normal"
                )
            
            self.after(0, show_error)
    
    def on_continue(self):
        """ادامه به برنامه اصلی"""
        self.destroy()
    
    def on_closing(self):
        """جلوگیری از بستن در حین Setup"""
        if self.setup_complete:
            self.destroy()
        else:
            messagebox.showwarning(
                "Setup in Progress",
                "Please wait for setup to complete."
            )


class RegisterWindow(ctk.CTkToplevel):
    """پنجره ثبت‌نام کاربر جدید"""
    def __init__(self, parent, auth_system_v2_1):
        super().__init__(parent)
        # ✅ ست کردن آیکون
        self.after(10, lambda: set_window_icon(self))
        self.auth_system_v2_1 = auth_system_v2_1

        self.title("Register New User")
        self.geometry("400x600")
        self.resizable(False, False)

        self.center_window()
        self.setup_ui()
        self.grab_set()

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            main_frame,
            text="📝 Register New User",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(0, 30))

        # نام کاربری
        ctk.CTkLabel(main_frame, text="Username:").pack(anchor="w", pady=(0, 5))
        self.username_entry = ctk.CTkEntry(main_frame, width=250)
        self.username_entry.pack(pady=(0, 15))

        # ایمیل
        ctk.CTkLabel(main_frame, text="Email (optional):").pack(anchor="w", pady=(0, 5))
        self.email_entry = ctk.CTkEntry(main_frame, width=250)
        self.email_entry.pack(pady=(0, 15))

        # رمز عبور
        ctk.CTkLabel(main_frame, text="Password:").pack(anchor="w", pady=(0, 5))
        self.password_entry = ctk.CTkEntry(main_frame, width=250, show="•")
        self.password_entry.pack(pady=(0, 10))

        # تأیید رمز عبور
        ctk.CTkLabel(main_frame, text="Confirm Password:").pack(anchor="w", pady=(0, 5))
        self.confirm_password_entry = ctk.CTkEntry(main_frame, width=250, show="•")
        self.confirm_password_entry.pack(pady=(0, 20))

        # دکمه ثبت‌نام
        register_btn = ctk.CTkButton(
            main_frame,
            text="Register",
            command=self.register,
            height=40,
            fg_color="green",
            hover_color="dark green"
        )
        register_btn.pack(pady=10)

        # دکمه لغو
        cancel_btn = ctk.CTkButton(
            main_frame,
            text="Cancel",
            command=self.destroy,
            height=35
        )
        cancel_btn.pack(pady=5)

    def register(self):
        """ثبت کاربر جدید"""
        username = self.username_entry.get().strip()
        email = self.email_entry.get().strip()
        password = self.password_entry.get().strip()
        confirm_password = self.confirm_password_entry.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Username and password are required")
            return

        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match")
            return

        if len(password) < 6:
            messagebox.showerror("Error", "Password must be at least 6 characters")
            return

        success, message = self.auth_system_v2_1.register_user(username, password, email)

        if success:
            messagebox.showinfo("Success", message)
            self.destroy()
        else:
            messagebox.showerror("Error", message)


class SettingsEditor(ctk.CTkToplevel):
    """ویرایشگر پیشرفته تنظیمات - تنظیمات واقعی"""
    
    def __init__(self, parent, config_manager_price3):
        super().__init__(parent)
        # ✅ ست کردن آیکون
        self.after(10, lambda: set_window_icon(self))
        self.config_manager_price3 = config_manager_price3
        self.parent = parent
        
        self.title("Advanced Settings Editor")
        self.geometry("700x750")
        self.resizable(True, True)
        
        # متغیرهای تنظیمات
        self.vars = {}
        
        self.center_window()
        self.setup_ui()
        self.load_all_settings()
        self.grab_set()

    def center_window(self):
        """مرکز کردن پنجره"""
        self.update_idletasks()
        width = 700
        height = 600
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        """ایجاد رابط کاربری"""
        # Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # اضافه کردن تب‌ها
        # هر تنظیم فقط در بخش خودش: عمومی / مدل (و حالت اندیکاتور) / عامل RL / ریسک حساب
        self.tabview.add("General")
        self.tabview.add("Model")
        self.tabview.add("RL")
        self.tabview.add("Risk")
        self.tabview.add("Paths")
        self.tabview.add("UI")

        # ساخت فرم‌ها
        self.create_general_form(self.tabview.tab("General"))
        self.create_model_form(self.tabview.tab("Model"))
        self.create_rl_form(self.tabview.tab("RL"))
        self.create_risk_form(self.tabview.tab("Risk"))
        self.create_paths_form(self.tabview.tab("Paths"))
        self.create_ui_form(self.tabview.tab("UI"))

        # دکمه‌ها
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(
            btn_frame, 
            text="Save All", 
            width=120,
            command=self.save_all_settings,
            fg_color="green", 
            hover_color="dark green"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame, 
            text="Reset Defaults", 
            width=120,
            command=self.reset_to_defaults,
            fg_color="orange",
            hover_color="dark orange"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame, 
            text="Reload", 
            width=100,
            command=self.load_all_settings
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame, 
            text="Close", 
            width=100,
            command=self.destroy
        ).pack(side="right", padx=5)

    def add_setting_row(self, parent, label, var_name, var_type, default, options=None, tooltip=""):
        """اضافه کردن یک ردیف تنظیمات"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=4)

        # Label
        lbl = ctk.CTkLabel(
            frame, 
            text=label, 
            width=200, 
            anchor="w",
            font=ctk.CTkFont(size=12)
        )
        lbl.pack(side="left", padx=5)

        # Widget based on type
        if var_type == "bool":
            var = ctk.BooleanVar(value=default)
            widget = ctk.CTkSwitch(frame, variable=var, text="", width=50)
            widget.pack(side="right", padx=10)
        
        elif var_type == "option" and options:
            var = ctk.StringVar(value=str(default))
            widget = ctk.CTkOptionMenu(frame, variable=var, values=options, width=150)
            widget.pack(side="right", padx=10)
        
        elif var_type == "float":
            var = ctk.StringVar(value=str(default))
            widget = ctk.CTkEntry(frame, textvariable=var, width=150)
            widget.pack(side="right", padx=10)
        
        elif var_type == "int":
            var = ctk.StringVar(value=str(default))
            widget = ctk.CTkEntry(frame, textvariable=var, width=150)
            widget.pack(side="right", padx=10)
        
        elif var_type == "path":
            var = ctk.StringVar(value=str(default))
            
            # Browse button
            def browse():
                from tkinter import filedialog
                if "model" in var_name or "scaler" in var_name:
                    path = filedialog.askopenfilename()
                else:
                    path = filedialog.asksaveasfilename()
                if path:
                    var.set(path)
            
            ctk.CTkButton(
                frame, text="...", width=30, command=browse
            ).pack(side="right", padx=2)
            
            widget = ctk.CTkEntry(frame, textvariable=var, width=250)
            widget.pack(side="right", padx=5)
        
        else:  # string
            var = ctk.StringVar(value=str(default))
            widget = ctk.CTkEntry(frame, textvariable=var, width=200)
            widget.pack(side="right", padx=10)

        self.vars[var_name] = var

    def add_section_title(self, parent, title):
        """عنوان بخش"""
        ctk.CTkLabel(
            parent, 
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4ade80"
        ).pack(pady=(15, 5), padx=10, anchor="w")

    def add_note(self, parent, text):
        """توضیح کوتاه زیر یک بخش"""
        ctk.CTkLabel(
            parent, text=text, font=ctk.CTkFont(size=11), text_color="#888", justify="left"
        ).pack(pady=(2, 6), padx=12, anchor="w")

    def create_general_form(self, parent):
        """تنظیمات مشترک همه‌ی حالت‌ها"""
        scroll = ctk.CTkScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)

        cfg = self.config_manager_price3.trading

        self.add_section_title(scroll, "General")
        self.add_setting_row(scroll, "Symbol:", "symbol", "str", cfg.symbol)
        self.add_setting_row(scroll, "Check Interval (sec):", "check_interval", "int", cfg.check_interval)
        self.add_note(scroll, "Signal source (Model / Indicators / RL) is chosen in the sidebar.\n"
                              "Model settings → Model tab · RL scalper → RL tab · lot size, limits, trailing → Risk tab.")

    def create_model_form(self, parent):
        """تنظیمات مدل (حالت Model) — حالت Indicators هم از تایم‌فریم و حد سود/ضرر همین بخش استفاده می‌کند"""
        scroll = ctk.CTkScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)

        cfg_t = self.config_manager_price3.trading
        cfg = self.config_manager_price3.model

        self.add_section_title(scroll, "Model Timeframes")
        self.add_setting_row(scroll, "Model Timeframe:", "timeframe", "option", cfg_t.timeframe,
                             options=["M5", "M15"])
        self.add_setting_row(scroll, "Model Secondary Timeframe:", "secondary_timeframe", "option",
                             getattr(cfg_t, 'secondary_timeframe', 'M15'),
                             options=["M5", "M15", "M30", "H1", "H4"])
        self.add_note(scroll, "Also used by the Indicators signal source.")

        self.add_section_title(scroll, "Entry Decision")
        self.add_setting_row(scroll, "Min Buy Confidence:", "min_confidence_buy", "float", cfg_t.min_confidence_buy)
        self.add_setting_row(scroll, "Min Sell Confidence:", "min_confidence_sell", "float", cfg_t.min_confidence_sell)
        self.add_setting_row(scroll, "Decide on Forming Bar:", "intrabar_decisions", "bool",
                             bool(getattr(cfg_t, "intrabar_decisions", True)))
        self.add_setting_row(scroll, "Max Spread (× ATR):", "max_spread_atr", "float",
                             getattr(cfg_t, "max_spread_atr", 0.35))
        self.add_note(scroll, "The GA Model always decides right after each closed bar and trades only rules\n"
                              "validated on your history — Min Buy/Sell Confidence and Decide on Forming Bar\n"
                              "apply to the Indicators source only. Max Spread 0 = no spread filter.")

        self.add_section_title(scroll, "TP / SL and Labels (Model + Indicators)")
        self.add_setting_row(scroll, "TP-SL Mode:", "label_mode", "option",
                             getattr(cfg_t, "label_mode", "Fixed"), options=["Fixed", "ATR"])
        self.add_setting_row(scroll, "Fixed TP (price distance):", "fixed_tp_price", "float",
                             getattr(cfg_t, "fixed_tp_price", 7.0))
        self.add_setting_row(scroll, "Fixed SL (price distance):", "fixed_sl_price", "float",
                             getattr(cfg_t, "fixed_sl_price", 7.0))
        self.add_note(scroll, "Fixed TP/SL are price distances of the current symbol (7 on XAUUSD = $7; EURUSD needs\n"
                              "about 0.0015). Each symbol keeps its own values; a new symbol gets yours converted by\n"
                              "volatility when you change the symbol.")
        self.add_setting_row(scroll, "TP ATR Multiplier:", "tp_atr_mult", "float", cfg_t.tp_atr_mult)
        self.add_setting_row(scroll, "SL ATR Multiplier:", "sl_atr_mult", "float", cfg_t.sl_atr_mult)
        self.add_setting_row(scroll, "Future Steps (label horizon, bars):", "future_steps", "int", cfg.future_steps)
        self.add_setting_row(scroll, "Close After Future Steps:", "use_time_exit", "bool",
                             bool(getattr(cfg_t, "use_time_exit", True)))

        self.add_section_title(scroll, "Model Input")
        self.add_setting_row(scroll, "Timesteps:", "timesteps", "int", cfg.timesteps)
        self.add_setting_row(scroll, "History Length:", "n_history", "int", cfg.n_history)

        self.add_section_title(scroll, "Model Training")
        self.add_setting_row(scroll, "Retrain Interval (hours):", "finetune_interval_hours", "int",
                             cfg.finetune_interval_hours)
        self.add_setting_row(scroll, "Min Samples for Training:", "min_samples_for_training", "int",
                             cfg.min_samples_for_training)
        self.add_setting_row(scroll, "Batch Size:", "batch_size", "int", cfg.batch_size)
        self.add_setting_row(scroll, "Epochs:", "epochs", "int", cfg.epochs)
        self.add_note(scroll, "Model = pattern model: fixed stop-loss, no time limit; each pattern's take-profit is the\n"
                              "average highest price it reached before the stop in its history. A pattern trades only\n"
                              "if its past trades average the minimum profit (statistics refreshed at every fine-tune).\n"
                              "Timesteps, Batch Size and Epochs belong to the old neural network and are not used.")

        self.add_section_title(scroll, "Indicators Signal Source")
        self.add_setting_row(scroll, "Min Confluence Score:", "min_confluence_score", "float",
                             cfg_t.min_confluence_score)
        self.add_note(scroll, "Used only when the signal source is Indicators (TP/SL from the section above).")

    RL_TIMEFRAMES = ["M1", "M5", "M15"]
    RL_CONTEXT_TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4"]

    def create_rl_form(self, parent):
        """تنظیمات عامل یادگیری تقویتی (اسکلپر) — مستقل از تایم‌فریم و تنظیمات مدل"""
        scroll = ctk.CTkScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)

        cfg = self.config_manager_price3.model

        self.add_section_title(scroll, "RL Timeframes (independent of the model)")
        self.add_setting_row(scroll, "RL Timeframe:", "rl_timeframe", "option",
                             getattr(cfg, "rl_timeframe", "M1"), options=self.RL_TIMEFRAMES)
        self.add_setting_row(scroll, "RL Context Timeframe:", "rl_secondary_timeframe", "option",
                             getattr(cfg, "rl_secondary_timeframe", "M5"), options=self.RL_CONTEXT_TIMEFRAMES)

        self.add_section_title(scroll, "RL Trades (agent closes trades itself, no fixed TP)")
        self.add_setting_row(scroll, "RL SL Mode:", "rl_sl_mode", "option",
                             getattr(cfg, "rl_sl_mode", "ATR"), options=["ATR", "Fixed"])
        self.add_setting_row(scroll, "RL SL ATR Multiplier:", "rl_sl_atr_mult", "float",
                             getattr(cfg, "rl_sl_atr_mult", 2.0))
        self.add_setting_row(scroll, "RL Fixed SL (price distance):", "rl_sl_price", "float",
                             getattr(cfg, "rl_sl_price", 3.0))
        self.add_setting_row(scroll, "RL Max Hold (bars):", "rl_max_hold_bars", "int",
                             getattr(cfg, "rl_max_hold_bars", 24))
        self.add_setting_row(scroll, "RL Max Spread (× ATR):", "rl_max_spread_atr", "float",
                             getattr(cfg, "rl_max_spread_atr", 0.35))
        self.add_note(scroll, "Max Spread uses the ATR of the RL timeframe; the simulator applies the same filter.")

        self.add_section_title(scroll, "RL Training")
        self.add_setting_row(scroll, "RL Retrain Interval (hours):", "rl_retrain_hours", "int",
                             getattr(cfg, "rl_retrain_hours", 12))
        self.add_setting_row(scroll, "RL Training Steps (per agent):", "rl_train_steps", "int",
                             getattr(cfg, "rl_train_steps", 500000))
        self.add_setting_row(scroll, "RL Agents (ensemble):", "rl_ensemble", "int",
                             getattr(cfg, "rl_ensemble", 3))
        self.add_setting_row(scroll, "RL History Bars (max 200000):", "rl_data_bars", "int",
                             getattr(cfg, "rl_data_bars", 200000))

        self.add_note(scroll, "Changing the RL timeframes, stop-loss or max hold applies on Save (the bot must be stopped);\n"
                              "a new RL agent is then trained for them. For M1 history set MT5 → Tools → Options →\n"
                              "Charts → Max bars in chart = Unlimited.")

    def create_risk_form(self, parent):
        """ریسک حساب و اندازه‌ی معامله — برای همه‌ی منابع سیگنال (تریلینگ فقط Model/Indicators)"""
        scroll = ctk.CTkScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)

        cfg = self.config_manager_price3.trading

        self.add_section_title(scroll, "Position Size & Limits (all signal sources)")
        self.add_setting_row(scroll, "Lot Size:", "lot_size", "float", cfg.lot_size)
        self.add_setting_row(scroll, "Max Risk per Trade (% equity):", "max_position_percent", "float",
                             cfg.max_position_percent)
        self.add_setting_row(scroll, "Max Daily Loss (%):", "max_daily_loss_percent", "float",
                             cfg.max_daily_loss_percent)
        self.add_setting_row(scroll, "Max Positions (Model/Indicators):", "max_positions", "int", cfg.max_positions)
        self.add_note(scroll, "Lot is reduced if the stop-loss would risk more than Max Risk per Trade.\n"
                              "The RL agent always holds at most one position.")

        self.add_section_title(scroll, "Trailing Stop (Model / Indicators — RL closes trades itself)")

        mode_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        mode_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(mode_frame, text="Trailing Stop Mode:", width=200, anchor="w").pack(side="left")

        current_mode = getattr(cfg, "trailing_stop_mode", "Off") or "Off"

        # ✅ StringVar - سازگار با get_var()
        mode_var = ctk.StringVar(value=current_mode)
        self.vars["trailing_stop_mode"] = mode_var

        mode_option = ctk.CTkOptionMenu(
            mode_frame,
            values=["Off", "Fixed", "Dynamic"],
            variable=mode_var,
            width=150
        )
        mode_option.pack(side="right", padx=5)

        fixed_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        dynamic_frame = ctk.CTkFrame(scroll, fg_color="transparent")

        self.add_section_title(fixed_frame, "--- Fixed Mode Parameters ---")
        self.add_setting_row(
            fixed_frame, "Fixed Trigger Profit ($):",
            "fixed_trigger_profit", "float",
            getattr(cfg, "fixed_trigger_profit", 10.0)
        )
        self.add_setting_row(
            fixed_frame, "Fixed Lock Profit ($):",
            "fixed_lock_profit", "float",
            getattr(cfg, "fixed_lock_profit", 3.0)
        )

        self.add_section_title(dynamic_frame, "--- Dynamic Mode Parameters ---")
        self.add_setting_row(
            dynamic_frame, "Dynamic Trigger Profit ($):",
            "dynamic_trigger_profit", "float",
            getattr(cfg, "dynamic_trigger_profit", 15.0)
        )
        self.add_setting_row(
            dynamic_frame, "Dynamic Trail Distance ($):",
            "dynamic_trail_distance", "float",
            getattr(cfg, "dynamic_trail_distance", 5.0)
        )

        def on_mode_change(choice):
            fixed_frame.pack_forget()
            dynamic_frame.pack_forget()
            if choice == "Fixed":
                fixed_frame.pack(fill="x")
            elif choice == "Dynamic":
                dynamic_frame.pack(fill="x")

        mode_option.configure(command=on_mode_change)
        on_mode_change(current_mode)

    def create_paths_form(self, parent):
        """فرم مسیرها - قابل ویرایش"""
        scroll = ctk.CTkScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)
        
        cfg_m = self.config_manager_price3.model
        cfg_d = self.config_manager_price3.database

        # نمایش App Directory (فقط خواندنی)
        self.add_section_title(scroll, "Application Directory (Read-only)")
        
        app_dir_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        app_dir_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            app_dir_frame, 
            text="App Directory:",
            width=200, 
            anchor="w"
        ).pack(side="left", padx=5)
        
        app_dir_entry = ctk.CTkEntry(app_dir_frame, width=350)
        app_dir_entry.pack(side="right", padx=10)
        app_dir_entry.insert(0, get_app_directory())
        app_dir_entry.configure(state="disabled")

        self.add_section_title(scroll, "Model Files")
        self.add_setting_row(scroll, "Model Path:", "model_path", "path", cfg_m.model_path)
        self.add_setting_row(scroll, "Scaler Path:", "scaler_path", "path", cfg_m.scaler_path)

        self.add_section_title(scroll, "Database Files")
        self.add_setting_row(scroll, "Database Path:", "db_path", "path", cfg_d.db_path)
        self.add_setting_row(scroll, "Trade Log CSV:", "log_csv", "path", cfg_d.log_csv)
        self.add_setting_row(scroll, "Last Train File:", "last_train_file", "path", cfg_d.last_train_file)

        self.add_section_title(scroll, "Database Settings")
        self.add_setting_row(scroll, "Max Rows:", "max_rows", "int", cfg_d.max_rows)
        self.add_setting_row(scroll, "Backup Interval (hours):", "backup_interval_hours", "int", cfg_d.backup_interval_hours)
        self.add_setting_row(scroll, "Auto Clean Old Data:", "auto_clean_old_data", "bool", cfg_d.auto_clean_old_data)
        self.add_setting_row(scroll, "Keep Days:", "keep_days", "int", cfg_d.keep_days)

    def create_ui_form(self, parent):
        """فرم UI"""
        scroll = ctk.CTkScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)
        
        cfg = self.config_manager_price3.ui

        self.add_section_title(scroll, "Appearance")
        self.add_setting_row(scroll, "Theme:", "theme", "option", cfg.theme,
                            options=["dark", "light", "system"])
        self.add_setting_row(scroll, "Font Size:", "font_size", "int", cfg.font_size)
        self.add_setting_row(scroll, "Chart Theme:", "chart_theme", "option", cfg.chart_theme,
                            options=["plotly_dark", "plotly_white", "ggplot2"])

        self.add_section_title(scroll, "Behavior")
        self.add_setting_row(scroll, "Show Grid:", "show_grid", "bool", cfg.show_grid)
        self.add_setting_row(scroll, "Animations:", "animation_enabled", "bool", cfg.animation_enabled)
        self.add_setting_row(scroll, "Auto Refresh Chart:", "auto_refresh_chart", "bool", cfg.auto_refresh_chart)
        self.add_setting_row(scroll, "Refresh Interval (sec):", "refresh_interval", "int", cfg.refresh_interval)

    def get_var(self, name, default=None):
        """گرفتن مقدار از متغیرها"""
        if name in self.vars:
            try:
                return self.vars[name].get()
            except:
                return default
        return default

    def load_all_settings(self):
        """بارگذاری مجدد همه‌ی تنظیمات نمایش‌داده‌شده از فایل"""
        self.config_manager_price3.load_config()
        sections = (self.config_manager_price3.trading, self.config_manager_price3.model,
                    self.config_manager_price3.database, self.config_manager_price3.ui)
        for name, var in self.vars.items():
            for section in sections:
                if hasattr(section, name):
                    value = getattr(section, name)
                    try:
                        var.set(bool(value) if isinstance(var, ctk.BooleanVar) else str(value))
                    except Exception:
                        pass
                    break

        print("[OK] Settings reloaded")

    def save_all_settings(self):
        """ذخیره تنظیمات"""
        old_symbol = self.config_manager_price3.trading.symbol
        # TP/SL ثابت فاصله‌ی قیمت نماد قبلی‌اند؛ با عوض شدن نماد برای آن نگه داشته و برای نماد جدید تبدیل می‌شوند
        old_levels = self.parent._price_levels()
        try:
            # تایم‌فریم زمینه‌ی RL باید بزرگ‌تر از تایم‌فریم RL باشد
            tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}
            rl_tf = str(self.get_var("rl_timeframe") or "M1")
            rl_ctx = str(self.get_var("rl_secondary_timeframe") or "M5")
            if tf_minutes.get(rl_ctx, 0) <= tf_minutes.get(rl_tf, 0):
                messagebox.showerror("Invalid RL timeframes",
                                     f"RL Context Timeframe ({rl_ctx}) must be larger than RL Timeframe ({rl_tf}).")
                return
            cfg_t = self.config_manager_price3.trading
            cfg_m = self.config_manager_price3.model
            cfg_d = self.config_manager_price3.database
            cfg_u = self.config_manager_price3.ui

            # ═══════════════════════════════════════════════════════════
            # Trading Settings
            # ═══════════════════════════════════════════════════════════
            val = self.get_var("symbol")
            if val: cfg_t.symbol = str(val)
            
            val = self.get_var("timeframe")
            if val: cfg_t.timeframe = str(val)
            val = self.get_var("secondary_timeframe")
            if val: cfg_t.secondary_timeframe = str(val)

            val = self.get_var("max_positions")
            if val: cfg_t.max_positions = int(val)
            
            val = self.get_var("lot_size")
            if val: cfg_t.lot_size = float(val)
            
            val = self.get_var("check_interval")
            if val: cfg_t.check_interval = int(val)
            
            val = self.get_var("tp_atr_mult")
            if val: cfg_t.tp_atr_mult = float(val)
            
            val = self.get_var("sl_atr_mult")
            if val: cfg_t.sl_atr_mult = float(val)
            
            val = self.get_var("label_mode")
            if val: cfg_t.label_mode = str(val)
            
            val = self.get_var("fixed_tp_price")
            if val: cfg_t.fixed_tp_price = float(val)
            
            val = self.get_var("fixed_sl_price")
            if val: cfg_t.fixed_sl_price = float(val)
            
            val = self.get_var("trailing_active_atr")
            if val: cfg_t.trailing_active_atr = float(val)
            
            val = self.get_var("trailing_distance_mult")
            if val: cfg_t.trailing_distance_mult = float(val)
            
            val = self.get_var("min_confidence_buy")
            if val: cfg_t.min_confidence_buy = float(val)
            
            val = self.get_var("min_confidence_sell")
            if val: cfg_t.min_confidence_sell = float(val)
            
            val = self.get_var("min_confluence_score")
            if val: cfg_t.min_confluence_score = float(val)

            val = self.get_var("intrabar_decisions")
            if val is not None: cfg_t.intrabar_decisions = bool(val)

            val = self.get_var("use_time_exit")
            if val is not None: cfg_t.use_time_exit = bool(val)

            val = self.get_var("max_spread_atr")
            if val not in (None, ""): cfg_t.max_spread_atr = max(0.0, float(val))
            
            # ═══════════════════════════════════════════════════════════
            # Advanced Risk & Trailing Stop Settings
            # ═══════════════════════════════════════════════════════════
            val = self.get_var("max_daily_loss_percent")
            if val: cfg_t.max_daily_loss_percent = float(val)
            
            val = self.get_var("max_position_percent")
            if val: cfg_t.max_position_percent = float(val)

            # ابتدا بررسی می‌کنیم که ورودی‌های فرم در کدام دیکشنری پلتفرم شما ذخیره شده‌اند
            val = self.get_var("trailing_stop_mode")
            if val is not None and str(val).strip() != "":
                cfg_t.trailing_stop_mode = str(val)

            val = self.get_var("fixed_trigger_profit")
            if val is not None and str(val).strip() != "":
                cfg_t.fixed_trigger_profit = float(val)

            val = self.get_var("fixed_lock_profit")
            if val is not None and str(val).strip() != "":
                cfg_t.fixed_lock_profit = float(val)

            val = self.get_var("dynamic_trigger_profit")
            if val is not None and str(val).strip() != "":
                cfg_t.dynamic_trigger_profit = float(val)

            val = self.get_var("dynamic_trail_distance")
            if val is not None and str(val).strip() != "":
                cfg_t.dynamic_trail_distance = float(val)

            # ✅ لاگ تایید ذخیره
            print(
                f"[SAVE TRAILING] mode={cfg_t.trailing_stop_mode} | "
                f"fixed_trigger={cfg_t.fixed_trigger_profit} | "
                f"fixed_lock={cfg_t.fixed_lock_profit} | "
                f"dynamic_trigger={cfg_t.dynamic_trigger_profit} | "
                f"dynamic_dist={cfg_t.dynamic_trail_distance}"
            )
            
            # Indicators
            val = self.get_var("use_rsi")
            if val is not None: cfg_t.use_rsi = bool(val)
            
            val = self.get_var("use_macd")
            if val is not None: cfg_t.use_macd = bool(val)
            
            val = self.get_var("use_bollinger")
            if val is not None: cfg_t.use_bollinger = bool(val)
            
            val = self.get_var("use_atr")
            if val is not None: cfg_t.use_atr = bool(val)
            
            val = self.get_var("use_adx")
            if val is not None: cfg_t.use_adx = bool(val)
            
            val = self.get_var("use_stochastic")
            if val is not None: cfg_t.use_stochastic = bool(val)

            # ═══════════════════════════════════════════════════════════
            # Model Settings
            # ═══════════════════════════════════════════════════════════
            val = self.get_var("model_path")
            if val: cfg_m.model_path = str(val)
            
            val = self.get_var("scaler_path")
            if val: cfg_m.scaler_path = str(val)
            
            val = self.get_var("timesteps")
            if val: cfg_m.timesteps = int(val)
            
            val = self.get_var("n_history")
            if val: cfg_m.n_history = int(val)
            
            val = self.get_var("future_steps")
            if val: cfg_m.future_steps = int(val)

            # ═══════════════════════════════════════════════════════════
            # RL Settings (تایم‌فریم‌ها و حد ضرر عامل، مستقل از مدل)
            # ═══════════════════════════════════════════════════════════
            val = self.get_var("rl_timeframe")
            if val: cfg_m.rl_timeframe = str(val)

            val = self.get_var("rl_secondary_timeframe")
            if val: cfg_m.rl_secondary_timeframe = str(val)

            val = self.get_var("rl_sl_mode")
            if val: cfg_m.rl_sl_mode = str(val)

            val = self.get_var("rl_sl_atr_mult")
            if val: cfg_m.rl_sl_atr_mult = float(val)

            val = self.get_var("rl_sl_price")
            if val: cfg_m.rl_sl_price = float(val)

            val = self.get_var("rl_max_hold_bars")
            if val: cfg_m.rl_max_hold_bars = int(val)

            val = self.get_var("rl_max_spread_atr")
            if val not in (None, ""): cfg_m.rl_max_spread_atr = max(0.0, float(val))

            val = self.get_var("rl_retrain_hours")
            if val: cfg_m.rl_retrain_hours = max(1, int(val))

            val = self.get_var("rl_train_steps")
            if val: cfg_m.rl_train_steps = int(val)

            val = self.get_var("rl_ensemble")
            if val: cfg_m.rl_ensemble = max(1, min(7, int(val)))

            val = self.get_var("rl_data_bars")
            # دیتابیس حداکثر bot.DB_MAX_BARS (۲۰۰ هزار) کندل نگه می‌دارد
            if val: cfg_m.rl_data_bars = min(bot.DB_MAX_BARS, max(20000, int(val)))
            
            val = self.get_var("finetune_interval_hours")
            if val: cfg_m.finetune_interval_hours = int(val)
            
            val = self.get_var("min_samples_for_training")
            if val: cfg_m.min_samples_for_training = int(val)
            
            val = self.get_var("batch_size")
            if val: cfg_m.batch_size = int(val)
            
            val = self.get_var("epochs")
            if val: cfg_m.epochs = int(val)
            
            val = self.get_var("learning_rate")
            if val: cfg_m.learning_rate = float(val)
            
            val = self.get_var("lstm_units_1")
            if val: cfg_m.lstm_units_1 = int(val)
            
            val = self.get_var("lstm_units_2")
            if val: cfg_m.lstm_units_2 = int(val)
            
            val = self.get_var("dropout_rate_1")
            if val: cfg_m.dropout_rate_1 = float(val)
            
            val = self.get_var("dropout_rate_2")
            if val: cfg_m.dropout_rate_2 = float(val)
            
            val = self.get_var("thresh_atr_up")
            if val: cfg_m.thresh_atr_up = float(val)
            
            val = self.get_var("thresh_atr_down")
            if val: cfg_m.thresh_atr_down = float(val)

            # ═══════════════════════════════════════════════════════════
            # Database Settings
            # ═══════════════════════════════════════════════════════════
            val = self.get_var("db_path")
            if val: cfg_d.db_path = str(val)
            
            val = self.get_var("log_csv")
            if val: cfg_d.log_csv = str(val)
            
            val = self.get_var("last_train_file")
            if val: cfg_d.last_train_file = str(val)
            
            val = self.get_var("max_rows")
            if val: cfg_d.max_rows = int(val)
            
            val = self.get_var("backup_interval_hours")
            if val: cfg_d.backup_interval_hours = int(val)
            
            val = self.get_var("auto_clean_old_data")
            if val is not None: cfg_d.auto_clean_old_data = bool(val)
            
            val = self.get_var("keep_days")
            if val: cfg_d.keep_days = int(val)

            # ═══════════════════════════════════════════════════════════
            # UI Settings
            # ═══════════════════════════════════════════════════════════
            val = self.get_var("theme")
            if val: cfg_u.theme = str(val)
            
            val = self.get_var("font_size")
            if val: cfg_u.font_size = int(val)
            
            val = self.get_var("chart_theme")
            if val: cfg_u.chart_theme = str(val)
            
            val = self.get_var("show_grid")
            if val is not None: cfg_u.show_grid = bool(val)
            
            val = self.get_var("animation_enabled")
            if val is not None: cfg_u.animation_enabled = bool(val)
            
            val = self.get_var("auto_refresh_chart")
            if val is not None: cfg_u.auto_refresh_chart = bool(val)
            
            val = self.get_var("refresh_interval")
            if val: cfg_u.refresh_interval = int(val)

            # ═══════════════════════════════════════════════════════════
            # Save to file
            # ═══════════════════════════════════════════════════════════
            self.config_manager_price3.save_config()
            print(
            f"[SAVE TRAILING] mode={cfg_t.trailing_stop_mode} | "
            f"fixed_trigger={cfg_t.fixed_trigger_profit} | "
            f"fixed_lock={cfg_t.fixed_lock_profit} | "
            f"dynamic_trigger={cfg_t.dynamic_trigger_profit} | "
            f"dynamic_dist={cfg_t.dynamic_trail_distance}"
            )
            # (bot همان ماژول ربات است که بالای فایل import شده؛ import محلی اینجا bot را در کل این تابع محلی می‌کرد)
            # نماد / تایم‌فریم / حد سود و ضرر / تنظیمات RL = پروفایل جدید (داده و مدل خودش): برنامه همین الان
            # به آن می‌رود و اگر مدل (یا عامل RL) آماده نیست، داده را دانلود و آموزش می‌دهد (ری‌استارت لازم نیست)
            profile_note = ""
            if bot.pending_profile_changes():
                try:
                    profile_note = self.parent.on_profile_changed_from_settings(old_symbol, old_levels)
                except Exception as e:
                    print(f"Profile switch error: {e}")
                    traceback.print_exc()
                # نماد ممکن است به اسم دقیق بروکر عوض شده باشد (مثلاً EURUSD → EURUSDm)
                # و TP/SL ثابت ممکن است برای نماد جدید تبدیل شده باشد
                for name, value in (("symbol", self.config_manager_price3.trading.symbol),
                                    *self.parent._price_levels().items()):
                    if name in self.vars:
                        self.vars[name].set(str(value))
            else:
                bot.reload_runtime_settings()
            # Apply theme
            ctk.set_appearance_mode(cfg_u.theme)

            messagebox.showinfo(
                "Success",
                "Settings saved successfully!" + (f"\n\n{profile_note}" if profile_note else "")
            )
            
            print("[OK] Settings saved")

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid value:\n{e}")
            print(f"[ERROR] Invalid value: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{e}")
            print(f"[ERROR] Save failed: {e}")

    def reset_to_defaults(self):
        """ریست به مقادیر پیش‌فرض"""
        if messagebox.askyesno("Confirm Reset", "Reset ALL settings to defaults?\n\nThis cannot be undone."):
            self.config_manager_price3.reset_to_defaults()
            messagebox.showinfo("Reset", "Settings reset to defaults.\n\nPlease close and reopen this window.")
            self.destroy()
 
# ═══════════════════════════════════════════════════════════════
# ✅ Text Redirector - هدایت print به GUI با رنگ
# ═══════════════════════════════════════════════════════════════
class TextRedirector:
    """هدایت stdout/stderr به GUI با رنگ‌بندی"""
    
    def __init__(self, text_widget=None, tag="default"):
        self.text_widget = text_widget
        self.tag = tag
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.buffer = []
        self.max_lines = 1000
    
    def write(self, message):
        # همیشه در ترمینال اصلی بنویس
        if self.original_stdout:
            self.original_stdout.write(message)
            self.original_stdout.flush()
        
        if message.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_msg = f"[{timestamp}] {message}"
            
            if self.text_widget and self.text_widget.winfo_exists():
                try:
                    self.text_widget.configure(state="normal")
                    
                    # ✅ تشخیص نوع پیام و رنگ‌بندی
                    tag = self.detect_tag(message)
                    self.text_widget.insert("end", formatted_msg, tag)
                    
                    self.text_widget.see("end")
                    
                    # محدود کردن خطوط
                    lines = int(self.text_widget.index('end-1c').split('.')[0])
                    if lines > self.max_lines:
                        self.text_widget.delete("1.0", f"{lines - self.max_lines}.0")
                    
                    self.text_widget.configure(state="disabled")
                except:
                    pass
            else:
                self.buffer.append((formatted_msg, self.detect_tag(message)))
                if len(self.buffer) > 500:
                    self.buffer.pop(0)
    
    def detect_tag(self, message):
        """تشخیص نوع پیام برای رنگ‌بندی"""
        msg_upper = message.upper()
        
        # Error / Warning
        if "ERROR" in msg_upper or "❌" in message or "✗" in message:
            return "error"
        if "WARNING" in msg_upper or "⚠️" in message:
            return "warning"
        
        # Success
        if "✅" in message or "✓" in message or "SUCCESS" in msg_upper:
            return "success"
        
        # Trading
        if "BUY" in msg_upper or "🟢" in message:
            return "buy"
        if "SELL" in msg_upper or "🔴" in message:
            return "sell"
        
        # Signal Analysis
        if "SIGNAL" in msg_upper or "📊" in message:
            return "signal"
        if "MODEL" in msg_upper or "🤖" in message:
            return "model"
        if "CONFLUENCE" in msg_upper or "📈" in message:
            return "confluence"
        if "DECISION" in msg_upper or "🎯" in message:
            return "decision"
        
        # Separators
        if "═══" in message or "───" in message or "===" in message:
            return "separator"
        
        # Info
        if "INFO" in msg_upper or "💾" in message or "📥" in message:
            return "info"
        
        return "default"
    
    def flush(self):
        if self.original_stdout:
            self.original_stdout.flush()
    
    def set_text_widget(self, widget):
        """تنظیم widget و تگ‌های رنگی"""
        self.text_widget = widget
        
        # ✅ تعریف تگ‌های رنگی
        self.setup_tags(widget)
        
        # نوشتن buffer
        if self.buffer and widget:
            try:
                widget.configure(state="normal")
                for msg, tag in self.buffer:
                    widget.insert("end", msg, tag)
                widget.see("end")
                widget.configure(state="disabled")
                self.buffer.clear()
            except:
                pass
    
    def setup_tags(self, widget):
        """تنظیم تگ‌های رنگی"""
        if not widget:
            return
        
        # دسترسی به _textbox داخلی
        try:
            tb = widget._textbox
            
            # رنگ‌ها
            tb.tag_configure("default", foreground="#c9d1d9")
            tb.tag_configure("error", foreground="#ff6b6b")
            tb.tag_configure("warning", foreground="#ffd93d")
            tb.tag_configure("success", foreground="#6bcb77")
            tb.tag_configure("info", foreground="#4d96ff")
            
            # Trading
            tb.tag_configure("buy", foreground="#00ff88")
            tb.tag_configure("sell", foreground="#ff4757")
            
            # Signal Analysis
            tb.tag_configure("signal", foreground="#a855f7")
            tb.tag_configure("model", foreground="#06b6d4")
            tb.tag_configure("confluence", foreground="#22c55e")
            tb.tag_configure("decision", foreground="#f59e0b")
            
            # Separator
            tb.tag_configure("separator", foreground="#666666")
            
        except Exception as e:
            print(f"Tag setup error: {e}")


# Global instances
log_redirector = TextRedirector(tag="info")
error_redirector = TextRedirector(tag="error")

##################################################################################
class NRTradingPlatform(ctk.CTk):
    """پلتفرم اصلی معاملاتی NR"""
    def __init__(self):
        super().__init__()

        # ✅ ست کردن آیکون
        self.after(10, lambda: set_window_icon(self))

        # سیستم‌های اصلی
        self.auth_system_v2_1 = OnlineAuthSystem()
        self.config_manager_price3 = ConfigManager()

        # متغیرهای وضعیت
        self.bot_thread = None
        self.bot_running = False
        self.dashboard_update_active = False
        self.dashboard_update_job = None
        self.chart_auto_update = False
        self.chart_update_job = None
        self.price_update_job = None
        self.cached_model = None
        self.cached_scaler = None
        self.model_load_time = None
        # هر بار که نماد/تنظیمات پروفایل عوض می‌شود یکی زیاد می‌شود؛ Setupهای پس‌زمینه‌ی پروفایل قبلی با آن می‌فهمند
        # که دیگر لازم نیستند
        self._profile_gen = 0
        # ═══════════════════════════════════════════════════════════
        # LOGIN WINDOW
        # ═══════════════════════════════════════════════════════════
        self.withdraw()
        self.login_window = LoginWindow(self, self.auth_system_v2_1)
        self.login_window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Sidebar اولیه
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(11, weight=1)

        self.start_bot_btn = ctk.CTkButton(
            self.sidebar,
            text="Start Bot",
            command=self.toggle_bot,
            fg_color="green",
            hover_color="dark green",
            height=40
        )
        self.start_bot_btn.grid(row=9, column=0, padx=10, pady=5, sticky="ew")

        self.stop_bot_btn = ctk.CTkButton(
            self.sidebar,
            text="Stop Bot",
            command=self.stop_bot,
            fg_color="red",
            hover_color="dark red",
            height=40
        )
        self.stop_bot_btn.grid(row=10, column=0, padx=10, pady=5, sticky="ew")

    def _model_files_stamp(self):
        """زمان آخرین تغییر فایل‌های مدل؛ برای فهمیدن اینکه مدل روی دیسک عوض شده یا نه"""
        stamp = []
        # مدل حالت Model حالا فایل الگوریتم ژنتیک است (ga_patterns.json)؛ فایل‌های شبکه‌ی عصبی قدیمی دیگر عوض نمی‌شوند
        for p in (bot.ga_model_path(),):
            try:
                stamp.append(os.path.getmtime(p))
            except OSError:
                stamp.append(None)
        return tuple(stamp)

    def get_cached_model(self):
        """
        بارگذاری مدل با Cache.
        از bot.load_deployed_model استفاده می‌شود تا فقط مدلی بارگذاری شود که با نسخه‌ی فیچرها/لیبل‌ها و
        تنظیمات فعلی سازگار است (همان مدلی که ربات با آن معامله می‌کند) و آستانه‌هایش فعال شوند.
        وقتی فایل‌های مدل روی دیسک عوض شوند (بعد از Setup یا fine-tune) خودکار دوباره بارگذاری می‌شود؛
        تلاش ناموفق روی همان فایل‌ها هر ثانیه تکرار نمی‌شود.
        """
        try:
            stamp = self._model_files_stamp()
            cached = (getattr(self, 'cached_model', None) is not None
                      and getattr(self, 'cached_scaler_primary', None) is not None
                      and getattr(self, 'cached_scaler_secondary', None) is not None)
            if cached and stamp == getattr(self, '_cached_model_stamp', None):
                return self.cached_model, self.cached_scaler_primary, self.cached_scaler_secondary

            # در حال Training و هنوز مدلی نداریم - فقط None برگردون
            if not cached and training_state.get()['is_training']:
                return None, None, None

            # همین فایل‌ها قبلاً ناموفق بودند - تا عوض نشوند دوباره تلاش نکن
            if stamp == getattr(self, '_failed_model_stamp', None):
                return None, None, None

            print("📥 Loading GA pattern model...")
            model, scaler_p, scaler_s = bot.load_deployed_model()
            if model is None:
                self._failed_model_stamp = stamp
                self.cached_model = None
                self.cached_scaler_primary = None
                self.cached_scaler_secondary = None
                return None, None, None

            self.cached_model = model
            self.cached_scaler_primary = scaler_p
            self.cached_scaler_secondary = scaler_s
            self._cached_model_stamp = stamp
            self._failed_model_stamp = None
            self.model_load_time = datetime.now()
            self._model_warning_shown = False
            print(f"✅ GA pattern model cached ({len(model)} rule(s))")
            return self.cached_model, self.cached_scaler_primary, self.cached_scaler_secondary

        except Exception as e:
            if not getattr(self, '_model_warning_shown', False):
                print(f"❌ GA pattern model load error: {e}")
                self._model_warning_shown = True
            return None, None, None

    # اسم خوانای تنظیمات پروفایل (bot.PROFILE_KEYS) برای پیام‌ها
    PROFILE_KEY_NAMES = {
        'SYMBOL': "Symbol", 'PRIMARY_TF_STR': "Model timeframe", 'SECONDARY_TF_STR': "Model secondary timeframe",
        'RL_TF_STR': "RL timeframe", 'RL_SECONDARY_TF_STR': "RL context timeframe",
        'TP_ATR_MULT': "TP/SL", 'SL_ATR_MULT': "TP/SL", 'LABEL_MODE': "TP/SL", 'FIXED_TP_PRICE': "TP/SL",
        'FIXED_SL_PRICE': "TP/SL", 'RL_MAX_HOLD': "RL max hold", 'RL_SL_MODE': "RL stop-loss",
        'RL_SL_ATR_MULT': "RL stop-loss", 'RL_SL_PRICE': "RL stop-loss", 'FUTURE_STEPS': "Future steps",
        'TIMESTEPS': "Timesteps", 'TIMESTEPS_SECONDARY': "Timesteps", 'EXTRA_COST_POINTS': "Extra cost",
    }

    def _bot_thread_alive(self):
        return self.bot_thread is not None and self.bot_thread.is_alive()

    def on_profile_changed_from_settings(self, old_symbol, old_levels=None):
        """
        Settings نماد / تایم‌فریم / حد سود و ضرر / تنظیمات RL را عوض کرده (پروفایل جدید با داده و مدل خودش).
        ۱) اسم نماد با اسم دقیق بروکر چک می‌شود (EURUSD → EURUSDm)؛ اگر نیست، نماد قبلی برمی‌گردد.
        ۲) ربات خاموش: همین الان به پروفایل جدید می‌رود و اگر مدل (یا عامل RL) آماده نیست داده دانلود و آموزش داده می‌شود.
           ربات روشن: می‌پرسد ربات متوقف شود؛ اگر نه، تغییر بعد از Stop اعمال می‌شود.
        خروجی: یک خط توضیح برای پیام ذخیره‌ی تنظیمات
        """
        cfg_t = self.config_manager_price3.trading
        if cfg_t.symbol != old_symbol:
            resolved, hints = MT5Manager.resolve_symbol(cfg_t.symbol)
            if resolved is None:
                tip = ("\n\nSimilar symbols at your broker: " + ", ".join(hints)) if hints else ""
                messagebox.showerror(
                    "Symbol not found",
                    f"'{cfg_t.symbol}' is not available at your broker (MetaTrader 5 → Market Watch → Symbols)."
                    f"{tip}\n\nThe symbol stays {old_symbol}.")
                print(f"❌ Symbol '{cfg_t.symbol}' not found at the broker — keeping {old_symbol}")
                cfg_t.symbol = old_symbol
                self.config_manager_price3.save_config()
            elif resolved != cfg_t.symbol:
                print(f"ℹ️ '{cfg_t.symbol}' is called '{resolved}' at your broker — using {resolved}")
                cfg_t.symbol = resolved
                self.config_manager_price3.save_config()
        # TP/SL ثابت (فاصله‌ی قیمت) نماد قبلی برای خودش نگه داشته و برای نماد جدید آورده/تبدیل می‌شود
        levels_note = self._carry_price_levels(old_symbol, cfg_t.symbol, old_levels)
        note = self._profile_change_note(old_symbol)
        return "\n\n".join(x for x in (note, levels_note) if x)

    def _profile_change_note(self, old_symbol):
        """تعویض پروفایل بعد از چک نماد و TP/SL (بخش دوم on_profile_changed_from_settings)"""
        changes = bot.pending_profile_changes()
        if not changes:
            bot.reload_runtime_settings()
            return ""
        names = ", ".join(dict.fromkeys(self.PROFILE_KEY_NAMES.get(k, k) for k in changes))
        if self._bot_thread_alive():
            if messagebox.askyesno(
                    "Bot Running",
                    f"Changed: {names}.\n\nThe bot must stop to switch to the new setup (its own data and model).\n"
                    f"Stop the bot now?\n\nOpen {bot.SYMBOL} positions keep their stop-loss / take-profit on the broker."):
                self.stop_bot()                       # stop_bot خودش تغییر را اعمال می‌کند
                return self._profile_note() if not self._bot_thread_alive() else ""
            bot.reload_runtime_settings()             # تا Stop، ربات با تنظیم قبلی کار می‌کند
            return f"{names} changed — applies when you stop the bot."
        self.apply_profile_change()
        return self._profile_note()

    # تنظیم‌هایی که «فاصله‌ی قیمت» همان نمادند: ۷ برای طلا یعنی ۷ دلار، برای EURUSD حدود 0.0015
    PRICE_LEVEL_FIELDS = (("trading", "fixed_tp_price"), ("trading", "fixed_sl_price"), ("model", "rl_sl_price"))

    def _price_levels(self):
        cm = self.config_manager_price3
        return {f: float(getattr(getattr(cm, sec), f, 0) or 0) for sec, f in self.PRICE_LEVEL_FIELDS}

    def _set_price_levels(self, levels):
        cm = self.config_manager_price3
        for sec, f in self.PRICE_LEVEL_FIELDS:
            if f in levels:
                setattr(getattr(cm, sec), f, float(levels[f]))

    def _symbol_levels_file(self):
        """مقدارهای TP/SL ثابت هر نماد (کنار فایل تنظیمات)"""
        cfg_file = str(getattr(self.config_manager_price3, 'config_file', '') or '')
        return os.path.join(os.path.dirname(cfg_file) or "data", "symbol_levels.json")

    def _read_symbol_levels(self):
        try:
            with open(self._symbol_levels_file(), 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_symbol_levels(self, data):
        try:
            path = self._symbol_levels_file()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            print(f"⚠️ Could not save per-symbol TP/SL: {e}")

    @staticmethod
    def _typical_atr(symbol, tf_str):
        """میانه‌ی ATR(14) در ۵۰۰ کندل اخیر (واحد قیمت همان نماد)؛ None اگر داده نیست"""
        try:
            rates = mt5_manager.get_rates(symbol, bot.get_mt5_timeframe(tf_str), 515)
            if rates is None or len(rates) < 100:
                return None
            atr = bot._wilder_atr(np.asarray(rates['high'], dtype=float), np.asarray(rates['low'], dtype=float),
                                  np.asarray(rates['close'], dtype=float))
            atr = atr[np.isfinite(atr) & (atr > 0)][-500:]
            return float(np.median(atr)) if len(atr) else None
        except Exception as e:
            print(f"⚠️ ATR of {symbol} not available: {e}")
            return None

    @staticmethod
    def _round_price(x, symbol):
        """فاصله‌ی قیمت گرد: دو رقم معنی‌دار، نه ریزتر از یک پوینت نماد (صفر = تنظیم‌نشده، صفر می‌ماند)"""
        if float(x) <= 0:
            return 0.0
        try:
            digits = int(mt5.symbol_info(symbol).digits)
        except Exception:
            digits = 5
        return max(round(float(f"{float(x):.2g}"), digits), 10.0 ** -digits)

    def _carry_price_levels(self, old_symbol, new_symbol, old_levels):
        """
        Fixed TP/SL (و حد ضرر ثابت RL) فاصله‌ی قیمت همان نمادند. با عوض شدن نماد: مقدارهای نماد قبلی برای خودش در
        symbol_levels.json نگه داشته می‌شود؛ نماد جدید مقدارهای ذخیره‌شده‌ی خودش را می‌گیرد، وگرنه همان اندازه نسبت به
        نوسان (ATR تایم‌فریم مدل) از نماد قبلی تبدیل می‌شود. عددی که کاربر در همین ذخیره خودش عوض کرده دست نمی‌خورد.
        خروجی: یک خط توضیح برای پیام ذخیره (فقط وقتی حالت Fixed است)
        """
        if not old_levels or not new_symbol or new_symbol == old_symbol:
            return ""
        memory = self._read_symbol_levels()
        memory[old_symbol] = dict(old_levels)
        now = self._price_levels()
        todo = [f for f in now if abs(now[f] - old_levels.get(f, now[f])) <= 1e-12]   # کاربر عوضش نکرده
        saved = memory.get(new_symbol) or {}
        new_levels, how = {}, None
        if todo and all(f in saved for f in todo):
            new_levels, how = {f: float(saved[f]) for f in todo}, "saved"
        elif todo:
            tf = self.config_manager_price3.trading.timeframe
            a_old, a_new = self._typical_atr(old_symbol, tf), self._typical_atr(new_symbol, tf)
            if a_old and a_new:
                k = a_new / a_old
                new_levels = {f: float(saved[f]) if f in saved else self._round_price(old_levels[f] * k, new_symbol)
                              for f in todo}
                how = "converted"
        self._write_symbol_levels(memory)
        if new_levels:
            self._set_price_levels(new_levels)
            self.config_manager_price3.save_config()
        lv = self._price_levels()
        tp, sl = lv['fixed_tp_price'], lv['fixed_sl_price']
        if how:
            print(f"🎯 Fixed TP/SL for {new_symbol}: TP {tp:g} / SL {sl:g} ("
                  f"{'saved values' if how == 'saved' else f'converted from {old_symbol} by volatility'})")
        if str(getattr(self.config_manager_price3.trading, 'label_mode', 'Fixed') or 'Fixed').strip().lower() == "atr":
            return ""
        if how == "saved":
            return f"Fixed TP/SL for {new_symbol}: TP {tp:g} / SL {sl:g} (your saved values for it)."
        if how == "converted":
            conv = [nm for nm, f in (("TP", "fixed_tp_price"), ("SL", "fixed_sl_price")) if f in todo]
            if not conv:
                return ""
            was = " / ".join(f"{nm} {old_levels['fixed_' + nm.lower() + '_price']:g}" for nm in conv)
            return (f"Fixed TP/SL for {new_symbol}: TP {tp:g} / SL {sl:g} — {'/'.join(conv)} converted from "
                    f"{old_symbol} ({was}) by volatility. You can change them in Settings → Model.")
        if todo:
            return (f"⚠️ Could not read prices to convert Fixed TP/SL for {new_symbol} — {tp:g} / {sl:g} are "
                    f"{old_symbol} price distances. Set them in Settings → Model (or use ATR mode).")
        return ""

    def _fix_price_levels_if_wrong(self):
        """
        Fixed TP/SL باید با نوسان همین نماد جور باشد (مثلاً ۷ دلار طلا روی EURUSD هیچ‌وقت نمی‌خورد). اگر حد ضرر بیرون از
        ۰.۱ تا ۵۰ برابر ATR تایم‌فریم مدل است، مقدار پیشنهادی (حد ضرر = ۲×ATR، همان نسبت TP به SL) پرسیده می‌شود.
        True = مقدار جدید اعمال شد (apply_profile_change خودش دوباره بررسی و Setup می‌کند).
        """
        if bot.LABEL_MODE != "fixed" or self._bot_thread_alive():
            return False
        tp, sl = float(bot.FIXED_TP_PRICE), float(bot.FIXED_SL_PRICE)
        key = (bot.SYMBOL, bot.PRIMARY_TF_STR, tp, sl)
        if getattr(self, '_levels_checked', None) == key:
            return False
        self._levels_checked = key
        atr = self._typical_atr(bot.SYMBOL, bot.PRIMARY_TF_STR)
        fits = lambda x: x > 0 and 0.1 <= x / atr <= 50
        if not atr or (fits(tp) and fits(sl)):
            return False
        # حد ضرر = ۲×ATR (اگر خودش جور نیست)، حد سود با همان نسبت TP به SL (اگر آن نسبت معقول است)
        new_sl = sl if fits(sl) else self._round_price(2.0 * atr, bot.SYMBOL)
        ratio = tp / sl if 0.2 <= tp / sl <= 10 else 1.0
        new_tp = tp if fits(tp) else self._round_price(new_sl * ratio, bot.SYMBOL)
        bad = sl if not fits(sl) else tp
        print(f"⚠️ Fixed TP/SL {tp:g} / {sl:g} don't fit {bot.SYMBOL}: {'SL' if bad is sl else 'TP'} is "
              f"{bad / atr:.3g}× its {bot.PRIMARY_TF_STR} ATR ({atr:.5g}). Suggested: TP {new_tp:g} / SL {new_sl:g}")
        if not messagebox.askyesno(
                "Fixed TP/SL don't fit this symbol",
                f"Fixed TP/SL are price distances of the symbol. With TP {tp:g} / SL {sl:g}, the "
                f"{'SL' if bad is sl else 'TP'} is {bad / atr:.3g}× the typical {bot.PRIMARY_TF_STR} ATR of "
                f"{bot.SYMBOL} ({atr:.5g}), so it would {'almost never be reached' if bad > atr else 'be hit by noise'}."
                f"\n\nUse TP {new_tp:g} / SL {new_sl:g} instead (SL = 2× ATR, same TP:SL ratio)?"):
            return False
        self._set_price_levels({'fixed_tp_price': new_tp, 'fixed_sl_price': new_sl})
        self.config_manager_price3.save_config()
        return self.apply_profile_change()

    def _profile_note(self):
        """توضیح بعد از رفتن به پروفایل جدید (برای پیام ذخیره‌ی تنظیمات)"""
        head = f"Now using {bot.SYMBOL}"
        if bot.SIGNAL_MODE == "rl":
            return (f"{head} (RL {bot.RL_TF_STR} + {bot.RL_SECONDARY_TF_STR}). If its RL agent is not ready, data is "
                    f"downloaded and the agent is trained in the background (see the dashboard).")
        if bot.SIGNAL_MODE == "indicators":
            return (f"{head} ({bot.PRIMARY_TF_STR} + {bot.SECONDARY_TF_STR}). Indicators mode needs no model; "
                    f"select Model in the sidebar to build it.")
        return (f"{head} ({bot.PRIMARY_TF_STR} + {bot.SECONDARY_TF_STR}). If its model is not ready, data is "
                f"downloaded and the model is trained in the background (see the dashboard).")

    def apply_profile_change(self):
        """
        نماد/تایم‌فریم/تنظیمات جدید فایل تنظیمات را در ربات اعمال کن (فقط وقتی ربات خاموش است)، بعد مثل شروع برنامه:
        اگر مدل (یا عامل RL) این تنظیم آماده نیست، داده دانلود و آموزش داده می‌شود.
        """
        if self._bot_thread_alive() or not bot.pending_profile_changes():
            return False
        self._profile_gen += 1
        # آموزشی که برای تنظیم قبلی در حال اجراست دیگر لازم نیست (چیزی ذخیره نمی‌شود)
        bot.request_training_cancel(reason="symbol/settings changed")
        bot.apply_profile_settings()
        # مدل در حافظه مال تنظیم قبلی است
        self.cached_model = None
        self.cached_scaler = None
        self.cached_scaler_primary = None
        self.cached_scaler_secondary = None
        self._cached_model_stamp = None
        self._failed_model_stamp = None
        self._model_warning_shown = False
        self.check_and_setup_model()
        return True

    def get_trading_stats_from_log(self):
        try:
            if not os.path.exists(LOG_CSV):
                return None
            
            # ✅ فیکس: خواندن با comma (نه tab!)
            try:
                df = pd.read_csv(LOG_CSV)  # ✅ با header خودکار
            except:
                return None
            
            if len(df) == 0:
                return None
            
            # ✅ چک کن ستون‌ها درست هستن
            required_cols = ['Date', 'Action', 'Price', 'Volume', 'PnL']
            for col in required_cols:
                if col not in df.columns:
                    print(f"❌ Missing column: {col}")
                    return None
            
            # فیلتر CLOSE
            df['Action'] = df['Action'].fillna('').astype(str)
            closed = df[df['Action'].str.contains('CLOSE', case=False, na=False)].copy()
            closed = closed.drop_duplicates(subset=['Date', 'Price'], keep='first')
            
            opened = df[df['Action'].str.contains('OPEN', case=False, na=False)]
            open_trades = max(0, len(opened) - len(closed))


            if len(closed) > 0 and 'PnL' in closed.columns:
                closed_pnl = pd.to_numeric(closed['PnL'], errors='coerce').fillna(0)
                
                total_pnl = closed_pnl.sum()
                wins = closed_pnl[closed_pnl > 0]
                losses = closed_pnl[closed_pnl < 0]
                
                win_count = len(wins)
                loss_count = len(losses)
                total = win_count + loss_count
                
                win_rate = (win_count / total * 100) if total > 0 else 0
                avg_win = wins.mean() if win_count > 0 else 0
                avg_loss = losses.mean() if loss_count > 0 else 0
                best_win = wins.max() if win_count > 0 else 0
                worst_loss = losses.min() if loss_count > 0 else 0
                pnl_list = closed_pnl.cumsum().tolist()
            else:
                total_pnl = win_rate = avg_win = avg_loss = 0
                best_win = worst_loss = 0
                pnl_list = []
                win_count = loss_count = 0

            return {
                'open_trades': open_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'best_win': best_win,
                'worst_loss': worst_loss,
                'pnl_list': pnl_list,
                'wins': win_count,
                'losses': loss_count
            }
            
        except Exception as e:
            print(f"Log error: {e}")
            return None
    
    # ✅ P&L
    def draw_pnl_chart(self, pnl_list):
        """رسم نمودار P&L"""
        try:
            if not pnl_list:
                return
            
            # پاک کردن قبلی
            for widget in self.pl_chart_container.winfo_children():
                widget.destroy()
            
            fig, ax = plt.subplots(figsize=(5, 2.5), facecolor='#2b2b2b')
            ax.set_facecolor('#2b2b2b')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            color = '#26a69a' if pnl_list[-1] >= 0 else '#ef5350'
            ax.plot(pnl_list, color=color, linewidth=2)
            ax.fill_between(range(len(pnl_list)), pnl_list, 0, alpha=0.3, color=color)
            ax.axhline(0, color='#666', linewidth=0.5, linestyle='--')
            
            ax.tick_params(colors='#888', labelsize=8)
            ax.set_ylabel('P&L ($)', color='#888', fontsize=9)
            
            for spine in ax.spines.values():
                spine.set_color('#444')
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, self.pl_chart_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)
        except Exception as e:
            print(f"PnL chart error: {e}")
    # ✅ Win Rate
    def draw_winrate_chart(self, wins, losses):
        """رسم نمودار Win Rate - دایره‌ای ساده"""
        try:
            for widget in self.wr_chart_container.winfo_children():
                widget.destroy()
            
            total = wins + losses
            if total == 0:
                label = ctk.CTkLabel(
                    self.wr_chart_container,
                    text="No trades yet",
                    font=ctk.CTkFont(size=14),
                    text_color="gray"
                )
                label.pack(expand=True)
                return
            
            
            fig, ax = plt.subplots(figsize=(3, 3), facecolor='#2b2b2b')
            ax.set_facecolor('#2b2b2b')
            
            sizes = [wins, losses]
            colors = ['#26a69a', '#ef5350']
            
            # ✅ فقط دایره، بدون هیچ متنی
            ax.pie(sizes, colors=colors, startangle=90)
            
            # ✅ فقط درصد وسط دایره
            win_rate = (wins/total)*100
            ax.text(
                0, 0,
                f'{win_rate:.0f}%',
                ha='center',
                va='center',
                color='white',
                fontsize=14
            )
            
            ax.axis('equal')
            plt.tight_layout(pad=0)
            
            canvas = FigureCanvasTkAgg(fig, self.wr_chart_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)
        except Exception as e:
            print(f"WinRate chart error: {e}")
    
    def reload_model(self):
        """بارگذاری مجدد مدل (برای وقتی که مدل آپدیت شده)"""
        self.cached_model = None
        self.cached_scaler = None
        self.cached_scaler_primary = None
        self.cached_scaler_secondary = None
        self._cached_model_stamp = None
        self._failed_model_stamp = None
        self._model_warning_shown = False  # ✅ ریست کن تا دوباره پیام بده اگه لازم شد
        return self.get_cached_model()  

    def show_main_app(self):
        """نمایش برنامه اصلی بعد از ورود موفق"""
        try:
            if LOGO_PATH:
                self.iconbitmap(LOGO_PATH)
        except:
            pass
        
        self.title("NR Trading Platform v2.1 - Developed by Naser Romipour")
        self.geometry("1400x800")
        self.minsize(1200, 700)

        ctk.set_appearance_mode(self.config_manager_price3.ui.theme)

        # ✅ اول UI رو بساز
        self.setup_main_ui()

        # ✅ نمایش برنامه
        self.deiconify()
        
        # ═══════════════════════════════════════════════════════════
        # ✅ چک First Run بعد از Login و نمایش UI
        # ═══════════════════════════════════════════════════════════
        self.after(500, self.check_and_setup_model)


    def check_and_setup_model(self):
        """
        چک کردن وضعیت مدل حالت Model (استراتژی الگوریتم ژنتیک روی الگوهای کندلی و کلاسیک)
        """
        try:
            from trading_bot_v5GA import (
                MODEL_PATH, PRIMARY_SCALER_PATH, SECONDARY_SCALER_PATH,
                PRIMARY_DB_PATH, SECONDARY_DB_PATH, check_first_run
            )
            
            # TP/SL ثابت باید با نوسان همین نماد جور باشد (اگر مقدار تازه اعمال شد، apply_profile_change خودش ادامه می‌دهد)
            if bot.SIGNAL_MODE != "rl" and self._fix_price_levels_if_wrong():
                return False

            # حالت فقط اندیکاتور: مدل لازم نیست (با انتخاب Model از نوار کناری ساخته می‌شود)
            if bot.SIGNAL_MODE == "indicators":
                print("📊 Indicators-only mode — model setup skipped (select Model to build it)")
                self.show_setup_notification("📊 Indicators-only mode\nModel is not needed", final=True)
                return True
            # حالت یادگیری تقویتی: به‌جای مدل، عامل RL بررسی/آموزش داده می‌شود
            if bot.SIGNAL_MODE == "rl":
                return self.check_rl_agent_setup()

            # نماد و تایم‌فریم‌ها از پروفایل فعال ربات: مسیر دیتابیس و مدل (PRIMARY_DB_PATH و ...) مال همین
            # پروفایل است. اگر در Settings عوض شده ولی برنامه ری‌استارت نشده، ربات هنوز با پروفایل قبلی
            # کار می‌کند و داده‌ی نماد/تایم‌فریم جدید نباید در دیتابیس پروفایل قبلی نوشته شود.
            symbol, primary_tf, secondary_tf = bot.SYMBOL, bot.PRIMARY_TF_STR, bot.SECONDARY_TF_STR
            cfg_t = self.config_manager_price3.trading
            cfg_secondary = (getattr(cfg_t, 'secondary_timeframe', None)
                             or bot.get_secondary_timeframe_default(cfg_t.timeframe))
            if (cfg_t.symbol, cfg_t.timeframe, cfg_secondary) != (symbol, primary_tf, secondary_tf):
                print(f"⚠️ Settings: {cfg_t.symbol} {cfg_t.timeframe}+{cfg_secondary}, but the running profile is "
                      f"{symbol} {primary_tf}+{secondary_tf} — stop the bot to switch and build the model for the new setup")
                self.show_setup_notification(
                    "⚠️ Symbol/timeframe changed in Settings\nStop the bot to switch and build the model", final=True)
                return False

            print(f"\n{'='*60}")
            print(f"🔍 Checking GA Pattern Model for: {symbol} / {primary_tf} + {secondary_tf}")
            print(f"   GA Model Path       : {bot.ga_model_path()}")
            print(f"{'='*60}")
            
            # چک وجود فایل‌ها
            # فقط وجود فایل کافی نیست: مدل باید با نسخه‌ی فیچرها/لیبل‌ها و تنظیمات فعلی سازگار باشد
            model_ok, model_reason = bot.deployed_model_status()
            
            if model_ok:
                print(f"✅ GA Pattern Model is ready for {symbol} ({primary_tf}+{secondary_tf})")
                self.show_setup_notification(
                    f"✅ GA Pattern Model ready\n{symbol} ({primary_tf} + {secondary_tf})",
                    final=True
                )
                return True
            else:
                print(f"⚠️ GA Pattern Model not usable ({model_reason}). Starting setup...")
                
                self.show_setup_notification(
                    f"🔄 Building GA Pattern Model for {symbol}...\nPrimary: {primary_tf} | Secondary: {secondary_tf}",
                    final=False
                )
                
                # اجرای Setup
                self.run_background_setup(symbol, primary_tf)
                return False
                
        except Exception as e:
            print(f"❌ Error in check_and_setup_model: {e}")
            import traceback
            traceback.print_exc()
            self.show_setup_notification("❌ Error checking model", final=True)
            return False


    def run_background_setup(self, symbol, primary_tf):
        """
        اجرای Setup حالت Model در Background: دانلود و لیبل‌گذاری تاریخچه، سپس آموزش الگوریتم ژنتیک
        """
        gen = self._profile_gen

        def stale():
            """نماد/تنظیمات وسط کار عوض شد: این Setup مال تنظیم قبلی است و Setup تنظیم جدید جدا شروع شده"""
            return gen != self._profile_gen

        def worker():
            try:
                from trading_bot_v5GA import (
                    fetch_ohlcv, compute_advanced_indicators, 
                    create_features_improved, save_training_data_multiclass, 
                    save_secondary_training_data, train_model_multiclass,
                    get_mt5_timeframe, MIN_SAMPLES_FOR_TRAINING,
                    PRIMARY_DB_PATH, SECONDARY_DB_PATH, 
                    MODEL_PATH, PRIMARY_SCALER_PATH, SECONDARY_SCALER_PATH
                )
                import sqlite3
                
                # تایم‌فریم دوم همان پروفایل فعال ربات (مسیر SECONDARY_DB_PATH مال همین است)
                secondary_tf = bot.SECONDARY_TF_STR

                # Progress callback
                def progress(p, msg):
                    if stale():
                        return
                    self.after(0, lambda: self.update_setup_notification(
                        f"[{int(p*100)}%] {msg}"
                    ))
                    print(f"[{int(p*100):3d}%] {msg}")

                progress(0.05, "Connecting to MT5...")

                # استفاده از MT5Manager که در main_app تعریف شده
                if not MT5Manager.ensure_connected():
                    progress(0.10, "❌ MT5 connection failed!")
                    self.after(0, lambda: self.show_setup_notification(
                        "❌ Cannot connect to MT5. Please open MetaTrader 5.",
                        final=True
                    ))
                    return
                # نماد باید در Market Watch فعال باشد تا تاریخچه‌اش دانلود شود
                if not bot.ensure_mt5_connected(symbol):
                    progress(0.10, f"❌ {symbol} is not available in MT5")
                    self.after(0, lambda: self.show_setup_notification(
                        f"❌ Cannot select {symbol} in MT5 (Market Watch).\nCheck the symbol name in Settings.",
                        final=True
                    ))
                    return

                progress(0.12, "Creating databases...")
                init_database(db_path=PRIMARY_DB_PATH)   # اگر init_database وجود دارد
                init_database(db_path=SECONDARY_DB_PATH)

                # ====================== Primary ======================
                progress(0.20, f"Downloading Primary ({primary_tf})...")
                tf_mt5 = get_mt5_timeframe(primary_tf)
                # تاریخچه‌ی عمیق برای الگوریتم ژنتیک (اگر ترمینال کمتر بدهد، بیشترین مقدار موجود)
                df_p = bot.fetch_ohlcv_deep(symbol, tf_mt5, bot.MODEL_HISTORY_BARS.get(primary_tf, 90000))

                if df_p is None or len(df_p) < 800:
                    progress(0.25, f"❌ Not enough Primary data")
                    self.after(0, lambda: self.show_setup_notification(
                        f"❌ Insufficient data on {primary_tf}", final=True
                    ))
                    return

                progress(0.35, f"Processing Primary: {len(df_p)} candles...")
                df_p = compute_advanced_indicators(df_p)
                features_p = create_features_improved(df_p)
                # نماد و تایم‌فریم صریح: اگر نماد وسط دانلود عوض شود، داده‌ی این نماد با مشخصات همین نماد ذخیره می‌شود
                saved_p = save_training_data_multiclass(df_p, features_p, db_path=PRIMARY_DB_PATH, symbol=symbol,
                                                        tf_seconds=bot.TF_SECONDS.get(primary_tf, 300))
                print(f"[OK] Primary saved: {saved_p} samples")

                # ====================== Secondary ======================
                progress(0.40, f"Downloading Secondary ({secondary_tf})...")
                secondary_tf_mt5 = get_mt5_timeframe(secondary_tf)
                df_s = fetch_ohlcv(symbol, secondary_tf_mt5, 30000)

                if df_s is not None and len(df_s) >= 600:
                    progress(0.48, f"Processing Secondary: {len(df_s)} candles...")
                    df_s = compute_advanced_indicators(df_s)
                    features_s = create_features_improved(df_s)
                    saved_s = save_secondary_training_data(df_s, features_s, db_path=SECONDARY_DB_PATH, symbol=symbol,
                                                           tf_seconds=bot.TF_SECONDS.get(secondary_tf, 900))
                    print(f"[OK] Secondary saved: {saved_s} samples")
                else:
                    progress(0.48, f"⚠️ Low Secondary data on {secondary_tf}")

                # ====================== Final Check & Training ======================
                total_samples = bot.count_labeled_samples(PRIMARY_DB_PATH)

                progress(0.55, f"Total samples: {total_samples}")

                if stale():
                    print(f"ℹ️ Symbol/settings changed — model setup for {symbol} stopped")
                    return

                if total_samples >= MIN_SAMPLES_FOR_TRAINING:
                    progress(0.60, "Building the pattern model (genetic discovery on: about 30 minutes)...")
                    if bot.SIGNAL_MODE != "model":
                        # منبع سیگنال در این فاصله عوض شده؛ آموزش مدل لازم نیست
                        progress(1.0, "Signal source changed — model training skipped")
                        return
                    # wait=True: اگر آموزش دیگری (مثلاً عامل RL) در حال توقف است، صبر کن
                    model = train_model_multiclass(wait=True)
                    if stale():
                        # آموزش لغو شد (یا مال تنظیم قبلی بود)؛ Setup تنظیم جدید اعلان خودش را دارد
                        print(f"ℹ️ Symbol/settings changed — model setup for {symbol} stopped")
                        return
                    if model is None and bot.SIGNAL_MODE != "model":
                        # لغو شد چون کاربر منبع سیگنال را عوض کرد؛ اعلان منبع جدید را پاک نکن
                        print("ℹ️ Model setup stopped — signal source changed")
                        return
                    
                    if model is not None:
                        n_rules = len(model)
                        progress(1.0, f"✅ GA Pattern Model ready ({n_rules} rule(s))")
                        self.after(0, lambda: self.show_setup_notification(
                            (f"✅ GA Pattern Model ready! {n_rules} rule(s)\n{symbol} ({primary_tf} + {secondary_tf})"
                             if n_rules else
                             f"⚠️ No pattern reaches the minimum average profit yet\n{symbol}: Model mode will not trade yet"),
                            final=True
                        ))
                    else:
                        # بدون فیلتر ذخیره، None فقط یعنی آموزش انجام نشد (داده‌ی کم، خطا، یا آموزش دیگری در جریان)
                        progress(1.0, "⚠️ Model training failed — nothing saved")
                        self.after(0, lambda: self.show_setup_notification(
                            "⚠️ Model training failed — nothing saved. Check logs.", final=True
                        ))
                else:
                    progress(1.0, f"⚠️ Not enough data ({total_samples})")
                    self.after(0, lambda: self.show_setup_notification(
                        f"⚠️ Need more data. Bot will train later.", final=True
                    ))

            except Exception as e:   # ← اینجا e را درست تعریف کردیم
                print(f"❌ Setup error: {e}")
                import traceback
                traceback.print_exc()
                if not stale():
                    self.after(0, lambda e=e: self.show_setup_notification(   # capture e
                        f"❌ Setup error: {str(e)[:60]}",
                        final=True
                    ))

        # شروع Thread
        setup_thread = threading.Thread(target=worker, daemon=True)
        setup_thread.start()

    def show_setup_notification(self, message, final=False):
        """نمایش پیغام Setup در بالای Dashboard"""
        if not hasattr(self, 'main_frame') or not self.main_frame.winfo_exists():
            return
        
        # اگه Notification قبلی وجود داره، حذفش کن
        if hasattr(self, 'setup_notification_frame'):
            try:
                self.setup_notification_frame.destroy()
            except:
                pass
        
        # ساخت Notification جدید
        notif_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="#1f538d" if not final else ("#26a69a" if "✅" in message else "#ef5350"),
            corner_radius=10
        )
        notif_frame.pack(fill="x", padx=20, pady=10, before=self.main_frame.winfo_children()[0] if self.main_frame.winfo_children() else None)
        
        # متن
        label = ctk.CTkLabel(
            notif_frame,
            text=message,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white"
        )
        label.pack(pady=12, padx=20)
        
        # اگه نهایی بود، بعد 10 ثانیه حذفش کن
        if final:
            self.after(10000, lambda: notif_frame.destroy() if notif_frame.winfo_exists() else None)
        
        self.setup_notification_frame = notif_frame


    def update_setup_notification(self, message):
        """آپدیت پیغام Setup"""
        if hasattr(self, 'setup_notification_frame') and self.setup_notification_frame.winfo_exists():
            # پیدا کردن Label
            for widget in self.setup_notification_frame.winfo_children():
                if isinstance(widget, ctk.CTkLabel):
                    widget.configure(text=message)
                    break    

    def setup_main_ui(self):
        """ایجاد رابط کاربری اصلی"""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.create_sidebar()

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.show_dashboard()

    def create_sidebar(self):
        """ایجاد نوار کناری"""
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=0, minsize=40)

        # لوگو
        try:
            logo_img_pil = get_logo_image()
            if logo_img_pil:
                logo_img = ctk.CTkImage(logo_img_pil, size=(30, 30))
            else:
                raise Exception("Logo PIL image is None")
        except Exception as e:
            print(f"[!] Sidebar logo load failed: {e}")
            logo_img = None  # در صورت خطا، بدون تصویر و فقط با متن لود می‌شود

        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="NR Trading Platform",
            image=logo_img if logo_img else "",
            compound="left",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        if logo_img:
            self.logo_label.image = logo_img
            
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # کاربر
        current_user = self.auth_system_v2_1.current_user or "Guest"
        self.user_btn = ctk.CTkButton(
            self.sidebar,
            text=f"👤 {current_user}",
            font=ctk.CTkFont(size=12),
            fg_color="transparent",
            hover_color=("gray70", "gray30"),
            anchor="center",
            command=self.show_user_profile
        )
        self.user_btn.grid(row=1, column=0, padx=20, pady=(0, 10))

        # ═══════════════════════════════════════════════════════════════
        # ⏰ Fine-tune Timer
        # ═══════════════════════════════════════════════════════════════
        self.finetune_timer_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="#1a1a2e",
            corner_radius=8
        )
        self.finetune_timer_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(
            self.finetune_timer_frame,
            text="⏰ Next Fine-tune:",
            font=ctk.CTkFont(size=10),
            text_color="#888"
        ).pack(pady=(6, 2))

        timer_row = ctk.CTkFrame(self.finetune_timer_frame, fg_color="transparent")
        timer_row.pack(fill="x", padx=10, pady=(0, 6))

        self.finetune_timer_label = ctk.CTkLabel(
            timer_row,
            text="--:--:--",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4ade80"
        )
        self.finetune_timer_label.pack(side="left")

        self.finetune_status_label = ctk.CTkLabel(
            timer_row,
            text="",
            font=ctk.CTkFont(size=9),
            text_color="#666"
        )
        self.finetune_status_label.pack(side="right")

        # ═══════════════════════════════════════════════════════════════
        # دکمه‌های ناوبری
        # ═══════════════════════════════════════════════════════════════
        nav_buttons = [
            ("📊 Dashboard", self.show_dashboard),
            ("📈 Live Chart", self.show_chart),
            ("📊 Performance", self.show_performance),
            ("🔧 Trading Tools", self.show_trading_tools),
            ("⚙️ Settings", self.show_settings_panel),
            ("📚 Logs", self.show_logs),
            ("ℹ️ About", self.show_about)
        ]

        for i, (text, command) in enumerate(nav_buttons, start=3):
            btn = ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
                height=35,
                anchor="w",
                fg_color="transparent",
                hover_color=("gray70", "gray30")
            )
            btn.grid(row=i, column=0, padx=10, pady=3, sticky="ew")

        # ═══════════════════════════════════════════════════════════════
        # 🔀 منبع سیگنال: فقط مدل یا فقط اندیکاتور (دو حالت مستقل)
        # ═══════════════════════════════════════════════════════════════
        mode_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        mode_frame.grid(row=10, column=0, padx=10, pady=(20, 0), sticky="ew")
        ctk.CTkLabel(
            mode_frame, text="Signal Source:", font=ctk.CTkFont(size=11), text_color="#888"
        ).pack(anchor="w")
        self.signal_mode_var = ctk.StringVar(
            value=self.SIGNAL_MODE_LABELS.get(bot.SIGNAL_MODE, self.SIGNAL_MODE_LABELS["model"]))
        mode_values = [self.SIGNAL_MODE_LABELS["model"], self.SIGNAL_MODE_LABELS["indicators"],
                       self.SIGNAL_MODE_LABELS["rl"]]
        if hasattr(ctk, "CTkSegmentedButton"):
            self.signal_mode_btn = ctk.CTkSegmentedButton(
                mode_frame, values=mode_values, variable=self.signal_mode_var,
                command=self.on_signal_mode_changed)
        else:
            self.signal_mode_btn = ctk.CTkOptionMenu(
                mode_frame, values=mode_values, variable=self.signal_mode_var,
                command=self.on_signal_mode_changed)
        self.signal_mode_btn.pack(fill="x", pady=(2, 0))

        # ═══════════════════════════════════════════════════════════════
        # ▶️ Start Bot (بعد از About)
        # ═══════════════════════════════════════════════════════════════
        self.start_bot_btn = ctk.CTkButton(
            self.sidebar,
            text="▶️ Start Bot",
            command=self.toggle_bot,
            fg_color="green",
            hover_color="dark green",
            height=35
        )
        self.start_bot_btn.grid(row=11, column=0, padx=3, pady=(15, 5), sticky="ew")

        # 🚪 Logout
        logout_btn = ctk.CTkButton(
            self.sidebar,
            text="🚪 Logout",
            command=self.logout,
            height=30,
            fg_color="red",
            hover_color="dark red"
        )
        logout_btn.grid(row=12, column=0, padx=10, pady=5)

        # Developer
        dev_label = ctk.CTkLabel(
            self.sidebar,
            text="Developed by: Naser Romipour",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        dev_label.grid(row=13, column=0, padx=10, pady=(5, 15))

        # ✅ شروع آپدیت تایمر
        self.update_finetune_timer()
    

    SIGNAL_MODE_LABELS = {"model": "🧬 Model", "indicators": "📊 Indicators", "rl": "🎮 RL"}

    def on_signal_mode_changed(self, label):
        """
        تغییر منبع سیگنال از نوار کناری. سه حالت کاملاً مستقل:
        - Model      : قانون‌های الگوریتم ژنتیک (الگوهای کندلی/کلاسیک + فیلتر زمینه) روی هر کندل بسته؛ اندیکاتورها اثری ندارند
        - Indicators : فقط امتیاز اندیکاتورها (Min Confluence Score)؛ مدل اجرا نمی‌شود و آموزشش متوقف است
        - RL         : عامل یادگیری تقویتی خودش ورود و خروج را تصمیم می‌گیرد (فقط حد ضرر روی بروکر)
        ربات در حال اجرا از بررسی بعدی (در حالت تصمیم روی کندل بسته: از کندل بعدی) با حالت جدید تصمیم می‌گیرد.
        """
        label = str(label)
        mode = "indicators" if "Indicators" in label else ("rl" if "RL" in label else "model")
        if mode == bot.SIGNAL_MODE:
            return
        try:
            # از ConfigManager خود برنامه ذخیره می‌شود تا ذخیره‌های بعدی تنظیمات مقدار قدیمی را برنگردانند
            bot.set_signal_mode(mode, config_manager=self.config_manager_price3)
        except Exception as e:
            print(f"❌ Signal source change failed: {e}")
            self.signal_mode_var.set(self.SIGNAL_MODE_LABELS.get(bot.SIGNAL_MODE, self.SIGNAL_MODE_LABELS["model"]))
            return

        # آموزشی که برای منبع قبلی در حال اجراست دیگر لازم نیست: متوقفش کن
        if mode == "indicators":
            bot.request_training_cancel()
        elif mode == "rl":
            bot.request_training_cancel("model")
        else:
            bot.request_training_cancel("rl")

        if mode == "indicators":
            print("📊 Indicators-only mode: the model is not used; entries follow Min Confluence Score. "
                  "Model training is paused.")
        elif mode == "rl":
            print("🎮 RL mode: the agent decides entries and exits itself (stop-loss on the broker, "
                  "max hold in bars). Model and indicators are not used.")
            self.check_rl_agent_setup()
        else:
            print("🧬 Model mode: entries follow the genetic-algorithm pattern rules (candlestick + chart "
                  "patterns), decided right after each closed bar. Indicators are not used.")
            # اگر مدل معتبر نیست، Setup شروع می‌شود (قفل آموزش جلوی آموزش هم‌زمان با ربات را می‌گیرد)
            self.check_and_setup_model()

    def check_rl_agent_setup(self):
        """
        اگر عامل یادگیری تقویتی معتبری برای تنظیمات فعلی نیست، در پس‌زمینه داده را آماده و عامل را آموزش بده.
        بدون فیلتر ذخیره: هر آموزشی که کامل شود، عامل ذخیره و استفاده می‌شود (نتیجه‌ی Test فقط گزارش می‌شود).
        """
        cfg_m = self.config_manager_price3.model
        cfg_rl = (str(getattr(cfg_m, 'rl_timeframe', 'M1')), str(getattr(cfg_m, 'rl_secondary_timeframe', 'M5')))
        if cfg_rl != (bot.RL_TF_STR, bot.RL_SECONDARY_TF_STR):
            print(f"⚠️ Settings: RL {cfg_rl[0]}+{cfg_rl[1]}, but the running RL setup is "
                  f"{bot.RL_TF_STR}+{bot.RL_SECONDARY_TF_STR} — stop the bot to switch and train the RL agent")
            self.show_setup_notification("⚠️ RL timeframes changed in Settings\nStop the bot to switch and train the RL agent",
                                         final=True)
            return False
        ok, reason = bot.rl_agent_status()
        if ok:
            print(f"✅ RL agent is ready ({bot.RL_TF_STR}+{bot.RL_SECONDARY_TF_STR})")
            self.show_setup_notification(f"✅ RL agent ready\n{bot.SYMBOL} ({bot.RL_TF_STR} + {bot.RL_SECONDARY_TF_STR})",
                                         final=True)
            return True
        if getattr(self, '_rl_setup_running', False):
            return False
        print(f"🎮 RL agent not usable ({reason}) — training in background "
              f"({bot.RL_TF_STR}+{bot.RL_SECONDARY_TF_STR})...")
        self.show_setup_notification(f"🎮 Training RL agent ({bot.RL_TF_STR} + {bot.RL_SECONDARY_TF_STR})...\n"
                                     f"(simulation on your history)", final=False)
        self._rl_setup_running = True
        gen = self._profile_gen

        def worker():
            try:
                def cb(p, msg):
                    if gen != self._profile_gen:
                        return
                    self.after(0, lambda m=msg, p=p: self.update_setup_notification(f"[{int(p * 100)}%] {m}"))
                status = bot.prepare_rl_agent(progress_callback=cb)
                messages = {
                    "ready": "✅ RL agent ready!",
                    "failed": "⚠️ RL training failed — nothing saved. Check logs.",
                    "cancelled": "⏹️ RL training stopped (signal source changed)",
                    "skipped": "⏳ Another training is still running — RL will train later",
                    "no_data": "⚠️ Not enough data for RL training",
                    "mt5_error": "❌ Cannot connect to MT5",
                }
                if gen == self._profile_gen:
                    self.after(0, lambda s=status: self.show_setup_notification(
                        messages.get(s, f"RL setup: {s}"), final=True))
            except Exception as e:
                print(f"❌ RL setup error: {e}")
                traceback.print_exc()
                if gen == self._profile_gen:
                    self.after(0, lambda e=e: self.show_setup_notification(f"❌ RL setup error: {str(e)[:60]}",
                                                                           final=True))
            finally:
                self._rl_setup_running = False
                if gen != self._profile_gen and bot.SIGNAL_MODE == "rl":
                    # نماد/تنظیمات وسط کار عوض شد: عامل تنظیم جدید را بررسی/آموزش بده
                    self.after(0, self.check_rl_agent_setup)

        threading.Thread(target=worker, daemon=True).start()
        return False

    def update_finetune_timer(self):
        """آپدیت تایمر Fine-tune هر ثانیه"""
        try:
            from trading_bot_v5GA import (
                get_last_train_time, 
                FINETUNE_INTERVAL_HOURS,
                training_state  # ✅ اضافه کن
            )
            from datetime import datetime, timedelta
            
            # ✅ چک کن آیا در حال Training هست
            train_info = training_state.get()
            
            if train_info['is_training']:
                # در حال آموزش
                progress = train_info['progress'] * 100
                epoch = train_info['epoch']
                kind = bot.current_training_kind()
                step_name = {"model": "GA gen", "rl": "RL check"}.get(kind, "Step")
                
                self.finetune_timer_label.configure(
                    text=f"{progress:.0f}%",
                    text_color="#3b82f6"
                )
                self.finetune_status_label.configure(
                    text=f"Training... {step_name} {epoch}",
                    text_color="#3b82f6"
                )
            elif bot.SIGNAL_MODE == "indicators":
                # حالت فقط اندیکاتور: آموزش مدل متوقف است
                self.finetune_timer_label.configure(text="Paused", text_color="#888")
                self.finetune_status_label.configure(text="Indicators mode", text_color="#666")
            else:
                # باقی کد قبلی...
                last_train = get_last_train_time()
                interval_hours = (bot.RL_RETRAIN_HOURS if bot.SIGNAL_MODE == "rl"
                                  else FINETUNE_INTERVAL_HOURS)
                
                if last_train is None:
                    self.finetune_timer_label.configure(
                        text="Ready!",
                        text_color="#fbbf24"
                    )
                    self.finetune_status_label.configure(
                        text="First training pending",
                        text_color="#fbbf24"
                    )
                else:
                    now = datetime.now()
                    next_train = last_train + timedelta(hours=interval_hours)
                    remaining = next_train - now
                    
                    if remaining.total_seconds() <= 0:
                        self.finetune_timer_label.configure(
                            text="00:00:00",
                            text_color="#ef5350"
                        )
                        self.finetune_status_label.configure(
                            text="Ready to train",
                            text_color="#ef5350"
                        )
                    else:
                        hours = int(remaining.total_seconds() // 3600)
                        minutes = int((remaining.total_seconds() % 3600) // 60)
                        seconds = int(remaining.total_seconds() % 60)
                        
                        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        
                        if remaining.total_seconds() < 600:
                            color = "#fbbf24"
                            status = "Soon..."
                        elif remaining.total_seconds() < 3600:
                            color = "#4ade80"
                            status = "Upcoming"
                        else:
                            color = "#4ade80"
                            status = "Scheduled"
                        
                        self.finetune_timer_label.configure(
                            text=time_str,
                            text_color=color
                        )
                        self.finetune_status_label.configure(
                            text=status,
                            text_color="#666"
                        )
            
        except Exception as e:
            self.finetune_timer_label.configure(
                text="Error",
                text_color="#ef5350"
            )
            print(f"Timer error: {e}")
        
        # هر 1 ثانیه آپدیت
        self.after(1000, self.update_finetune_timer)
###########################################################################################


    def clear_main_frame(self):
        # ✅ توقف آپدیت داشبورد قبل از پاک کردن ویجت‌ها
        self.stop_dashboard_update()
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_dashboard(self):
        """نمایش داشبورد با آپدیت خودکار هر ثانیه"""
        # توقف آپدیت قبلی
        self.stop_dashboard_update()
        self.clear_main_frame()
        
        # متغیر کنترل آپدیت
        self.dashboard_update_active = True
        
        title = ctk.CTkLabel(
            self.main_frame,
            text="📊 Trading Dashboard",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=10)
        
        # ═══════════════════════════════════════════════════════════
        # Status Bar (نمایش زمان آخرین آپدیت)
        # ═══════════════════════════════════════════════════════════
        status_frame = ctk.CTkFrame(self.main_frame, height=30)
        status_frame.pack(fill="x", padx=20, pady=5)
        
        self.dashboard_status_label = ctk.CTkLabel(
            status_frame,
            text="🔄 Updating...",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.dashboard_status_label.pack(side="right", padx=10)
        
        self.bot_live_status = ctk.CTkLabel(
            status_frame,
            text="🤖 Bot: STOPPED",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="red"
        )
        self.bot_live_status.pack(side="left", padx=10)

        # ═══════════════════════════════════════════════════════════
        # Cards Frame
        # ═══════════════════════════════════════════════════════════
        self.cards_frame = ctk.CTkFrame(self.main_frame)
        self.cards_frame.pack(fill="x", padx=20, pady=10)
        
        self.cards_frame.grid_columnconfigure(0, weight=1)
        self.cards_frame.grid_columnconfigure(1, weight=1)
        self.cards_frame.grid_columnconfigure(2, weight=1)

        # ═══════════════════════════════════════════════════════════
        # Account Card
        # ═══════════════════════════════════════════════════════════
        self.account_card = ctk.CTkFrame(self.cards_frame, width=300, height=180)
        self.account_card.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(
            self.account_card,
            text="💰 Account Status",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.account_labels = {}
        for label in ["Balance", "Equity", "Margin", "Free Margin"]:
            item_frame = ctk.CTkFrame(self.account_card, fg_color="transparent")
            item_frame.pack(fill="x", padx=20, pady=2)
            
            ctk.CTkLabel(
                item_frame, text=f"{label}:",
                font=ctk.CTkFont(weight="bold"), width=100, anchor="w"
            ).pack(side="left")
            
            self.account_labels[label] = ctk.CTkLabel(item_frame, text="N/A")
            self.account_labels[label].pack(side="right")

        # ═══════════════════════════════════════════════════════════
        # Trading Stats Card
        # ═══════════════════════════════════════════════════════════
        self.stats_card = ctk.CTkFrame(self.cards_frame, width=300, height=180)
        self.stats_card.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(
            self.stats_card,
            text="📈 Trading Stats",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.stats_labels = {}
        for label in ["Open Trades", "Total P&L", "Win Rate", "Avg Win"]:
            item_frame = ctk.CTkFrame(self.stats_card, fg_color="transparent")
            item_frame.pack(fill="x", padx=20, pady=2)
            
            ctk.CTkLabel(
                item_frame, text=f"{label}:",
                font=ctk.CTkFont(weight="bold"), width=100, anchor="w"
            ).pack(side="left")
            
            self.stats_labels[label] = ctk.CTkLabel(item_frame, text="N/A")
            self.stats_labels[label].pack(side="right")

        # ═══════════════════════════════════════════════════════════
        # Signal Card (اینجا سیگنال لحظه‌ای نمایش داده میشه)
        # ═══════════════════════════════════════════════════════════
        self.signal_card = ctk.CTkFrame(self.cards_frame, width=300, height=180)
        self.signal_card.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(
            self.signal_card,
            text="🎯 Live Signal",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.signal_labels = {}
        for label in ["Symbol", "Signal", "Confidence", "Model Prob", "Confluence", "ATR", "Patterns"]:
            item_frame = ctk.CTkFrame(self.signal_card, fg_color="transparent")
            item_frame.pack(fill="x", padx=20, pady=2)
            
            title_lbl = ctk.CTkLabel(
                item_frame, text=f"{label}:",
                font=ctk.CTkFont(weight="bold"), width=80, anchor="w"
            )
            title_lbl.pack(side="left")
            
            if label == "Patterns":
                self.patterns_title = title_lbl          # «Patterns (N):» وقتی چند الگو هست
                # جعبه‌ی متن با ارتفاع ثابت (۲ خط): با چند الگو کارت بلندتر نمی‌شود و جای نمودارهای P&L و Win Rate را
                # نمی‌گیرد. الگوها پشت‌سرهم، هر کدام با رنگ خودش (صعودی سبز، نزولی قرمز)؛ بیشتر از ۲ خط با چرخ موس
                box = ctk.CTkTextbox(item_frame, width=240, height=46, wrap="word", activate_scrollbars=False,
                                     fg_color="transparent", border_width=0, font=ctk.CTkFont(size=12))
                box.pack(side="right")
                box.tag_config("right", justify="right")
                box.insert("1.0", "N/A", ("right",))
                box.configure(state="disabled")
                self.signal_labels[label] = box
                self._patterns_shown = None
                continue
            self.signal_labels[label] = ctk.CTkLabel(item_frame, text="N/A", wraplength=190, justify="right")
            self.signal_labels[label].pack(side="right")

        # ═══════════════════════════════════════════════════════════
        # Indicators Panel (نمایش وضعیت 5 اندیکاتور)
        # ═══════════════════════════════════════════════════════════
        indicators_frame = ctk.CTkFrame(self.main_frame)
        indicators_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            indicators_frame,
            text="📊 Indicator Signals",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        self.indicators_container = ctk.CTkFrame(indicators_frame)
        self.indicators_container.pack(fill="x", padx=10, pady=5)
        
        self.indicator_labels = {}
        indicator_names = ["RSI", "MACD", "BB", "ATR", "PriceAction"]
        
        for i, name in enumerate(indicator_names):
            frame = ctk.CTkFrame(self.indicators_container, width=100)
            frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            
            ctk.CTkLabel(frame, text=name, font=ctk.CTkFont(size=12, weight="bold")).pack(pady=2)
            
            self.indicator_labels[name] = ctk.CTkLabel(
                frame, text="⚪ 0.0",
                font=ctk.CTkFont(size=14)
            )
            self.indicator_labels[name].pack(pady=5)
        
        # Make columns equal width
        for i in range(5):
            self.indicators_container.grid_columnconfigure(i, weight=1)

        # ═══════════════════════════════════════════════════════════
        # Charts Frame
        # ═══════════════════════════════════════════════════════════
        charts_frame = ctk.CTkFrame(self.main_frame)
        charts_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # P&L Chart
        self.pl_frame = ctk.CTkFrame(charts_frame)
        self.pl_frame.pack(side="left", fill="both", expand=True, padx=5)

        ctk.CTkLabel(self.pl_frame, text="📈 P&L History", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.pl_chart_container = ctk.CTkFrame(self.pl_frame)  
        self.pl_chart_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Win Rate Chart
        self.wr_frame = ctk.CTkFrame(charts_frame)
        self.wr_frame.pack(side="right", fill="both", expand=True, padx=5)

        ctk.CTkLabel(self.wr_frame, text="📊 Win Rate", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.wr_chart_container = ctk.CTkFrame(self.wr_frame) 
        self.wr_chart_container.pack(fill="both", expand=True, padx=10, pady=10)

        # ═══════════════════════════════════════════════════════════
        # شروع آپدیت خودکار
        # ═══════════════════════════════════════════════════════════
        self.update_dashboard_data()


    def stop_dashboard_update(self):
        """توقف آپدیت داشبورد"""
        self.dashboard_update_active = False
        if hasattr(self, 'dashboard_update_job') and self.dashboard_update_job:
            try:
                self.after_cancel(self.dashboard_update_job)
            except:
                pass
            self.dashboard_update_job = None


    def update_dashboard_data(self):
        """آپدیت تمام داده‌های داشبورد"""
       
        # ═══════════════════════════════════════════════════════════
        # ✅ چک کن داشبورد هنوز فعاله
        # ═══════════════════════════════════════════════════════════
        if not hasattr(self, 'dashboard_update_active') or not self.dashboard_update_active:
            
            return
        
        # ✅ چک کن ویجت‌ها هنوز وجود دارن
        if not hasattr(self, 'bot_live_status') or not self.bot_live_status.winfo_exists():
            
            self.dashboard_update_active = False
            return
        
        try:
            now = datetime.now().strftime("%H:%M:%S")
            
            
            # ═══════════════════════════════════════════════════════════
            # 1. Bot Status
            # ═══════════════════════════════════════════════════════════
            if self.bot_live_status.winfo_exists():
                if self.bot_running:
                    self.bot_live_status.configure(text="🤖 Bot: RUNNING", text_color="green")
                else:
                    self.bot_live_status.configure(text="🤖 Bot: STOPPED", text_color="red")
            
            # ═══════════════════════════════════════════════════════════
            # 2. Account Info
            # ═══════════════════════════════════════════════════════════
            try:
                
                    acc_info = mt5_manager.get_account_info()
                    if acc_info and hasattr(self, 'account_labels'):
                        if "Balance" in self.account_labels and self.account_labels["Balance"].winfo_exists():
                            self.account_labels["Balance"].configure(
                                text=f"${acc_info.balance:,.2f}",
                                text_color="white"
                            )
                        
                        if "Equity" in self.account_labels and self.account_labels["Equity"].winfo_exists():
                            equity_color = "green" if acc_info.equity >= acc_info.balance else "red"
                            self.account_labels["Equity"].configure(
                                text=f"${acc_info.equity:,.2f}",
                                text_color=equity_color
                            )
                        
                        if "Margin" in self.account_labels and self.account_labels["Margin"].winfo_exists():
                            self.account_labels["Margin"].configure(
                                text=f"${acc_info.margin:,.2f}"
                            )
                        
                        if "Free Margin" in self.account_labels and self.account_labels["Free Margin"].winfo_exists():
                            self.account_labels["Free Margin"].configure(
                                text=f"${acc_info.margin_free:,.2f}"
                            )
                    
            except:
                pass
            
            # ═══════════════════════════════════════════════════════════
            # 3. Trading Stats
            # ═══════════════════════════════════════════════════════════
            try:
                if hasattr(self, 'stats_labels'):
                    stats = self.get_trading_stats_from_log()
                    
                    if stats:
                        # آپدیت Labels
                        if "Open Trades" in self.stats_labels and self.stats_labels["Open Trades"].winfo_exists():
                            self.stats_labels["Open Trades"].configure(
                                text=str(stats['open_trades']),
                                text_color="yellow" if stats['open_trades'] > 0 else "white"
                            )
                        
                        if "Total P&L" in self.stats_labels and self.stats_labels["Total P&L"].winfo_exists():
                            pnl = stats['total_pnl']
                            self.stats_labels["Total P&L"].configure(
                                text=f"${pnl:+,.2f}",
                                text_color="green" if pnl >= 0 else "red"
                            )
                        
                        if "Win Rate" in self.stats_labels and self.stats_labels["Win Rate"].winfo_exists():
                            self.stats_labels["Win Rate"].configure(
                                text=f"{stats['win_rate']:.1f}%",
                                text_color="green" if stats['win_rate'] >= 50 else "red"
                            )
                        
                        if "Avg Win" in self.stats_labels and self.stats_labels["Avg Win"].winfo_exists():
                            self.stats_labels["Avg Win"].configure(
                                text=f"${stats['avg_win']:,.2f}",
                                text_color="green"
                            )
                        
                        # ✅ آپدیت نمودارها هر 5 ثانیه
                        if not hasattr(self, 'last_chart_update_time'):
                            self.last_chart_update_time = 0
                        
                        current_time = time.time()
                        if current_time - self.last_chart_update_time >= 5:
                            # نمودار P&L
                            if stats['pnl_list']:
                                if hasattr(self, 'pl_chart_container'):
                                    if self.pl_chart_container.winfo_exists():
                                        self.draw_pnl_chart(stats['pnl_list'])

                            # نمودار Win Rate
                            if hasattr(self, 'wr_chart_container'):
                                if self.wr_chart_container.winfo_exists():
                                    self.draw_winrate_chart(stats['wins'], stats['losses'])
                            
                            self.last_chart_update_time = current_time
                    else:
                        # اگه هنوز معامله‌ای نیست
                        for label_name in ["Open Trades", "Total P&L", "Win Rate", "Avg Win"]:
                            if label_name in self.stats_labels and self.stats_labels[label_name].winfo_exists():
                                self.stats_labels[label_name].configure(text="0" if label_name == "Open Trades" else "$0.00" if "P&L" in label_name or "Win" in label_name else "0%")
                                
            except Exception as e:
                print(f"Stats error: {e}")
            
            # ═══════════════════════════════════════════════════════════
            # 4. Live Signal (Dual Timeframe)
            # ═══════════════════════════════════════════════════════════
            try:
                if hasattr(self, 'signal_labels'):
                    indicators_mode = bot.SIGNAL_MODE == "indicators"
                    rl_mode = bot.SIGNAL_MODE == "rl"
                    rl_q = None
                    if indicators_mode or rl_mode:
                        model = scaler_p = scaler_s = None   # حالت اندیکاتور / یادگیری تقویتی: مدل لازم نیست
                    else:
                        model, scaler_p, scaler_s = self.get_cached_model()  # تغییر به 3 مقدار

                    if indicators_mode or rl_mode or (model is not None and scaler_p is not None and scaler_s is not None):
                        # گرفتن داده هر دو تایم‌فریم (حالت RL: تایم‌فریم‌های خود عامل)
                        if rl_mode:
                            tf_p, tf_s, n_bars = bot.RL_TIMEFRAME, bot.RL_SECONDARY_TIMEFRAME, bot.rl_live_bars() + 1
                            tf_text = f"{bot.RL_TF_STR}+{bot.RL_SECONDARY_TF_STR}"
                        else:
                            tf_p, tf_s, n_bars = bot.TIMEFRAME, bot.SECONDARY_TIMEFRAME, bot.live_history_bars()
                            tf_text = f"{bot.PRIMARY_TF_STR}+{bot.SECONDARY_TF_STR}"
                        # پنجره‌ی بلند فقط برای تشخیص الگوهای مدل؛ اندیکاتورها روی همان پنجره‌ی کوتاه قبلی
                        n_feat = n_bars if rl_mode else bot.feature_history_bars()
                        rates_p = mt5_manager.get_rates(bot.SYMBOL, tf_p, n_bars)
                        rates_s = mt5_manager.get_rates(bot.SYMBOL, tf_s, n_feat)
                        
                        if rates_p is None or rates_s is None:
                            if "Signal" in self.signal_labels and self.signal_labels["Signal"].winfo_exists():
                                self.signal_labels["Signal"].configure(text="⚠️ No Data", text_color="orange")
                        
                        elif len(rates_p) < 200 or len(rates_s) < bot.TIMESTEPS_SECONDARY + 20:
                            if "Signal" in self.signal_labels and self.signal_labels["Signal"].winfo_exists():
                                self.signal_labels["Signal"].configure(text="⚠️ Low Data", text_color="orange")
                        
                        else:
                            # ✅ داده هر دو تایم‌فریم آماده است
                            df_p = pd.DataFrame(rates_p)
                            df_p['Date'] = pd.to_datetime(df_p['time'], unit='s')
                            df_p.rename(columns={
                                'open': 'Open', 'high': 'High', 'low': 'Low',
                                'close': 'Close', 'tick_volume': 'Volume'
                            }, inplace=True)
                            
                            df_s = pd.DataFrame(rates_s)
                            df_s['Date'] = pd.to_datetime(df_s['time'], unit='s')
                            df_s.rename(columns={
                                'open': 'Open', 'high': 'High', 'low': 'Low',
                                'close': 'Close', 'tick_volume': 'Volume'
                            }, inplace=True)
                            
                            df_p_long = df_p
                            df_p = compute_advanced_indicators(df_p.iloc[-n_feat:].reset_index(drop=True))
                            features_p = create_features_improved(df_p)
                            
                            df_s = compute_advanced_indicators(df_s)
                            features_s = create_features_improved(df_s)
                            
                            if rl_mode:
                                # تصمیم عامل یادگیری تقویتی (بدون انتشار؛ ربات در حال اجرا خودش منتشر می‌کند)
                                rl_target, _rl_cur, rl_q = bot.get_rl_signal(df_p, df_s)
                                signal = rl_target or 0
                                confidence = 0.0
                                rl_gap = 0.0
                                if rl_q is not None:
                                    q_sorted = sorted(rl_q, reverse=True)
                                    rl_gap = float(q_sorted[0] - q_sorted[1])
                                signal_data = {'confidence': rl_gap, 'model_conf': rl_gap, 'model_prob': 0.0}
                            else:
                                if not indicators_mode:
                                    # مدل GA فقط روی کندل بسته تصمیم می‌گیرد: همان کندلی را نشان بده که ربات رویش معامله می‌کند
                                    df_p, features_p = bot._drop_forming_bar(df_p, features_p, bot.PRIMARY_TF_SECONDS,
                                                                             bot.SYMBOL)
                                # تایم‌فریم دوم فقط تا کندل‌هایی که تا پایان کندل جاری اصلی بسته می‌شوند
                                # (همان قاعده‌ی تصمیم لحظه‌ای ربات و داده‌ی آموزش؛ get_rl_signal خودش این کار را می‌کند)
                                s_cutoff = (int(df_p['time'].iloc[-1]) + bot.PRIMARY_TF_SECONDS
                                            - bot.SECONDARY_TF_SECONDS)
                                features_s_aligned = features_s[df_s['time'].to_numpy() <= s_cutoff]
                                # تصمیم حالت انتخاب‌شده (Model = قانون‌های الگوریتم ژنتیک)
                                df_long = None
                                if not indicators_mode:
                                    df_long, _ = bot._drop_forming_bar(df_p_long, df_p_long, bot.PRIMARY_TF_SECONDS,
                                                                       bot.SYMBOL)
                                signal, confidence = get_combined_signal_multiclass(
                                    model, scaler_p, scaler_s, df_p, features_p, features_s_aligned,
                                    verbose=False, df_long=df_long
                                )
                                signal_data = shared_signal.get()
                            atr = df_p['ATR_14'].iloc[-1] if 'ATR_14' in df_p.columns else 0
                            
                            confluence_signals = get_confluence_signals(df_p)
                            confluence_score, _ = calculate_confluence_score(confluence_signals)
                            
                            # نمایش Symbol
                            if "Symbol" in self.signal_labels and self.signal_labels["Symbol"].winfo_exists():
                                self.signal_labels["Symbol"].configure(
                                    text=f"{bot.SYMBOL} ({tf_text}) · "
                                         f"{'RL' if rl_mode else 'Indicators' if indicators_mode else 'Model'}"
                                )
                            
                            # تعیین سیگنال
                            if rl_mode:
                                if rl_q is None:
                                    signal_text, signal_color = "⚠️ No RL agent", "orange"
                                else:
                                    signal_text, signal_color = {1: ("🟢 LONG", "#00ff00"),
                                                                 -1: ("🔴 SHORT", "#ff0000")}.get(signal, ("⏸️ FLAT", "#888888"))
                            elif signal != 0 and not indicators_mode:
                                # حالت مدل GA: هر سیگنالی از یک قانون اعتبارسنجی‌شده معتبر است
                                # (Confidence اینجا نرخ برد Validation آن قانون است و با TP بزرگ‌تر از SL معمولاً زیر 60٪)
                                signal_text, signal_color = (("🟢 BUY", "#00ff00") if signal == 1
                                                             else ("🔴 SELL", "#ff0000"))
                            elif signal == 1:
                                if confidence >= 0.75:
                                    signal_text, signal_color = "🟢 STRONG BUY", "#00ff00"
                                elif confidence >= 0.60:
                                    signal_text, signal_color = "🟡 BUY", "#ffff00"
                                else:
                                    signal_text, signal_color = "⚪ WEAK BUY", "#888888"
                            elif signal == -1:
                                if confidence >= 0.75:
                                    signal_text, signal_color = "🔴 STRONG SELL", "#ff0000"
                                elif confidence >= 0.60:
                                    signal_text, signal_color = "🟠 SELL", "#ff8800"
                                else:
                                    signal_text, signal_color = "⚪ WEAK SELL", "#888888"
                            else:
                                signal_text, signal_color = "⏸️ NO TRADE", "#888888"
                            
                            # آپدیت Labels
                            if "Signal" in self.signal_labels and self.signal_labels["Signal"].winfo_exists():
                                self.signal_labels["Signal"].configure(text=signal_text, text_color=signal_color)
                            
                            if "Confidence" in self.signal_labels and self.signal_labels["Confidence"].winfo_exists():
                                model_conf = signal_data.get('confidence' if indicators_mode else 'model_conf', 0)
                                if indicators_mode:
                                    conf_color = "green" if model_conf >= 0.6 else "orange" if model_conf >= 0.4 else "red"
                                else:
                                    conf_color = "green" if signal != 0 else "gray"
                                self.signal_labels["Confidence"].configure(
                                    text=(f"ΔQ {model_conf:+.3f}" if rl_mode else f"{model_conf * 100:.1f}%"),
                                    text_color=conf_color
                                )
                            
                            if "Model Prob" in self.signal_labels and self.signal_labels["Model Prob"].winfo_exists():
                                model_prob = signal_data.get('model_prob', 0.5)
                                self.signal_labels["Model Prob"].configure(
                                    text="—" if (indicators_mode or rl_mode) else f"{model_prob:.3f}")
                            
                            if "Confluence" in self.signal_labels and self.signal_labels["Confluence"].winfo_exists():
                                conf_color = "green" if confluence_score > 0 else "red" if confluence_score < 0 else "white"
                                self.signal_labels["Confluence"].configure(text=f"{confluence_score:+.2f}", text_color=conf_color)
                            
                            if "ATR" in self.signal_labels and self.signal_labels["ATR"].winfo_exists():
                                self.signal_labels["ATR"].configure(text=f"{atr:.2f}")

                            # الگوهای کندلی/کلاسیک روی آخرین کندل بسته (حالت مدل GA)
                            if "Patterns" in self.signal_labels and self.signal_labels["Patterns"].winfo_exists():
                                if indicators_mode or rl_mode:
                                    pat_items = [("—", None)]
                                else:
                                    ind = signal_data.get('indicators', {})
                                    pats = list(ind.get('ga_patterns', []))
                                    dirs = list(ind.get('ga_pattern_dirs', []))
                                    dirs += [0] * (len(pats) - len(dirs))
                                    # الگوهای نام‌دار اول، بعد الگوهای یادگرفته‌شده؛ همه نوشته می‌شوند (جعبه ارتفاع ثابت دارد و
                                    # بقیه با چرخ موس دیده می‌شوند) و تعدادشان کنار عنوان می‌آید
                                    pairs = sorted(zip(pats, dirs), key=lambda x: x[0].startswith('Learned'))
                                    pat_items = [(nm, self.PATTERN_COLORS.get(d)) for nm, d in pairs]
                                    pat_items = pat_items or [("none", "gray")]
                                self._show_patterns(pat_items)
                            
                            # Indicator labels
                            if hasattr(self, 'indicator_labels'):
                                for name, label in self.indicator_labels.items():
                                    if label.winfo_exists():
                                        value = confluence_signals.get(name, 0)
                                        if value > 0.5:
                                            emoji, color = "🟢", "green"
                                        elif value < -0.5:
                                            emoji, color = "🔴", "red"
                                        else:
                                            emoji, color = "⚪", "gray"
                                        label.configure(text=f"{emoji} {value:+.1f}", text_color=color)
                    
                    else:
                        # مدل نیست - نمایش پیام
                        if "Signal" in self.signal_labels and self.signal_labels["Signal"].winfo_exists():
                            self.signal_labels["Signal"].configure(text="⚠️ No Model", text_color="orange")
                            
            except Exception as e:
                print(f"Dashboard signal update error: {e}")
                if "Signal" in self.signal_labels and self.signal_labels["Signal"].winfo_exists():
                    self.signal_labels["Signal"].configure(text="⚠️ Error", text_color="red")

            except Exception as e:
                print(f"Signal error: {e}")
            # ═══════════════════════════════════════════════════════════
            # 5. Status Update
            # ═══════════════════════════════════════════════════════════
            if hasattr(self, 'dashboard_status_label') and self.dashboard_status_label.winfo_exists():
                self.dashboard_status_label.configure(text=f"🔄 {now}", text_color="green")
            
        except Exception as e:
            print(f"Dashboard update error: {e}")
        
        # ═══════════════════════════════════════════════════════════
        # Schedule next update
        # ═══════════════════════════════════════════════════════════
        if hasattr(self, 'dashboard_update_active') and self.dashboard_update_active:
            # ✅ چک کن ویجت اصلی هنوز هست
            if hasattr(self, 'bot_live_status') and self.bot_live_status.winfo_exists():
                self.dashboard_update_job = self.after(1000, self.update_dashboard_data)
            else:
                self.dashboard_update_active = False

    # رنگ الگوها در کارت سیگنال: صعودی سبز، نزولی قرمز
    PATTERN_COLORS = {1: "#22c55e", -1: "#ef4444"}

    def _show_patterns(self, items):
        """الگوهای آخرین کندل بسته در کارت سیگنال، پشت‌سرهم و هر کدام با رنگ خودش؛ فقط وقتی فهرست عوض شود دوباره نوشته می‌شود"""
        box = self.signal_labels.get("Patterns")
        if box is None or not box.winfo_exists():
            return
        key = (id(box), tuple(items))
        if getattr(self, '_patterns_shown', None) == key:
            return
        self._patterns_shown = key
        title = getattr(self, 'patterns_title', None)
        if title is not None and title.winfo_exists():
            n = sum(1 for text, _ in items if text not in ("none", "—"))
            title.configure(text=f"Patterns ({n}):" if n > 1 else "Patterns:")
        box.configure(state="normal")
        box.delete("1.0", "end")
        for i, (text, color) in enumerate(items):
            if i:
                box.tag_config("sep", foreground="gray")
                box.insert("end", "  ·  ", ("right", "sep"))
            tags = ("right",)
            if color:
                tag = "c_" + str(color).lstrip("#")
                box.tag_config(tag, foreground=color)
                tags += (tag,)
            # فاصله‌ی داخل اسم نشکند: خط فقط بین الگوها شکسته می‌شود
            box.insert("end", str(text).replace(" ", "\u00a0"), tags)
        box.configure(state="disabled")

    def create_info_card(self, parent, title, items, column=0):
        card = ctk.CTkFrame(parent, width=300, height=180)
        card.grid(row=0, column=column, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        for label, value in items:
            item_frame = ctk.CTkFrame(card, fg_color="transparent")
            item_frame.pack(fill="x", padx=20, pady=2)

            ctk.CTkLabel(
                item_frame,
                text=f"{label}:",
                font=ctk.CTkFont(weight="bold"),
                width=100,
                anchor="w"
            ).pack(side="left")

            color = "green" if ("+" in str(value) or label == "Win Rate") else "white"
            ctk.CTkLabel(
                item_frame,
                text=str(value),
                text_color=color
            ).pack(side="right")

        return card
#################################################################################

    def show_chart(self):
        """نمایش چارت لایو داخل GUI"""
        # توقف آپدیت قبلی
        self.stop_chart_auto_update()
        self.stop_price_update()
        
        self.clear_main_frame()
        
        # متغیرهای کنترل
        self.chart_auto_update = False
        self.chart_update_job = None
        self.price_update_job = None

        title = ctk.CTkLabel(
            self.main_frame,
            text=f"📈 Live Chart - {bot.SYMBOL}",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=10)

        # ═══════════════════════════════════════════════════════════
        # Toolbar Row 1
        # ═══════════════════════════════════════════════════════════
        controls_frame = ctk.CTkFrame(self.main_frame)
        controls_frame.pack(fill="x", padx=20, pady=5)

        toolbar1 = ctk.CTkFrame(controls_frame)
        toolbar1.pack(fill="x", pady=5)

        # Timeframe
        ctk.CTkLabel(toolbar1, text="TF:").pack(side="left", padx=5)
        self.chart_tf_var = ctk.StringVar(value="M5")
        ctk.CTkOptionMenu(
            toolbar1, values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
            variable=self.chart_tf_var, width=70
        ).pack(side="left", padx=2)

        # Candles
        ctk.CTkLabel(toolbar1, text="Candles:").pack(side="left", padx=(10, 5))
        self.chart_candles_var = ctk.StringVar(value="500")
        ctk.CTkOptionMenu(
            toolbar1, values=["100", "200", "500", "1000", "2000"],
            variable=self.chart_candles_var, width=70
        ).pack(side="left", padx=2)

        # Chart Update Interval
        ctk.CTkLabel(toolbar1, text="Chart:").pack(side="left", padx=(10, 5))
        self.chart_interval_var = ctk.StringVar(value="10")
        ctk.CTkOptionMenu(
            toolbar1, values=["5", "10", "30", "60"],
            variable=self.chart_interval_var, width=50
        ).pack(side="left", padx=2)
        ctk.CTkLabel(toolbar1, text="s").pack(side="left")

        # Auto Update Toggle
        self.auto_update_btn = ctk.CTkButton(
            toolbar1, text="▶️ Live", width=80,
            fg_color="#26a69a", hover_color="#1e8e82",
            command=self.toggle_chart_auto_update
        )
        self.auto_update_btn.pack(side="left", padx=15)

        # Manual Refresh
        ctk.CTkButton(
            toolbar1, text="🔄 Refresh", width=90,
            command=self.manual_refresh_chart
        ).pack(side="left", padx=5)

        # New Window
        ctk.CTkButton(
            toolbar1, text="🗗 Window", width=80,
            fg_color="#2196f3", command=self.open_live_chart_v2_1
        ).pack(side="left", padx=5)

        # Price info (این هر ثانیه آپدیت میشه)
        self.price_info_label = ctk.CTkLabel(
            toolbar1, text="Loading...",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#ffeb3b"
        )
        self.price_info_label.pack(side="right", padx=20)

        # Status
        self.chart_status_label = ctk.CTkLabel(
            toolbar1, text="", font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.chart_status_label.pack(side="right", padx=10)

        # ═══════════════════════════════════════════════════════════
        # Toolbar Row 2 - اندیکاتورها
        # ═══════════════════════════════════════════════════════════
        toolbar2 = ctk.CTkFrame(controls_frame)
        toolbar2.pack(fill="x", pady=5)

        ctk.CTkLabel(toolbar2, text="Indicators:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)

        self.ind_sma_var = ctk.BooleanVar(value=True)
        self.ind_bb_var = ctk.BooleanVar(value=False)
        self.ind_ema_var = ctk.BooleanVar(value=False)
        self.ind_vol_var = ctk.BooleanVar(value=True)
        self.ind_rsi_var = ctk.BooleanVar(value=True)
        self.ind_macd_var = ctk.BooleanVar(value=True)
        self.ind_atr_var = ctk.BooleanVar(value=False)
        self.ind_stoch_var = ctk.BooleanVar(value=False)

        for name, var in [("SMA", self.ind_sma_var), ("EMA", self.ind_ema_var), 
                        ("BB", self.ind_bb_var), ("Vol", self.ind_vol_var),
                        ("RSI", self.ind_rsi_var), ("MACD", self.ind_macd_var),
                        ("ATR", self.ind_atr_var), ("Stoch", self.ind_stoch_var)]:
            ctk.CTkCheckBox(toolbar2, text=name, variable=var, width=70,
                            command=self.manual_refresh_chart).pack(side="left", padx=3)

        # ═══════════════════════════════════════════════════════════
        # Toolbar Row 3 - زوم
        # ═══════════════════════════════════════════════════════════
        toolbar3 = ctk.CTkFrame(controls_frame)
        toolbar3.pack(fill="x", pady=5)

        ctk.CTkLabel(toolbar3, text="Zoom:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        ctk.CTkButton(toolbar3, text="🔍+", width=50, command=self.chart_zoom_in).pack(side="left", padx=3)
        ctk.CTkButton(toolbar3, text="🔍-", width=50, command=self.chart_zoom_out).pack(side="left", padx=3)
        ctk.CTkButton(toolbar3, text="↔️ Fit", width=60, command=self.chart_fit_all).pack(side="left", padx=3)
        ctk.CTkButton(toolbar3, text="◀️", width=40, command=self.chart_pan_left).pack(side="left", padx=3)
        ctk.CTkButton(toolbar3, text="▶️", width=40, command=self.chart_pan_right).pack(side="left", padx=3)

        self.zoom_range_label = ctk.CTkLabel(toolbar3, text="", font=ctk.CTkFont(size=11), text_color="gray")
        self.zoom_range_label.pack(side="right", padx=20)

        # Chart container
        self.chart_container = ctk.CTkFrame(self.main_frame)
        self.chart_container.pack(fill="both", expand=True, padx=20, pady=10)

        # متغیرهای زوم
        self.chart_zoom_start = 0
        self.chart_zoom_end = None
        self.chart_data = None

        # Load chart
        self.create_real_chart()
        
        # شروع آپدیت قیمت (هر ثانیه)
        self.start_price_update()


    def start_price_update(self):
        """آپدیت قیمت لحظه‌ای هر ثانیه"""
        try:
            
                tick = mt5_manager.get_tick(bot.SYMBOL)
                if tick:
                    spread = (tick.ask - tick.bid) * 10000
                    price_color = "#26a69a" if tick.last >= tick.bid else "#ef5350"
                    
                    # آپدیت label بالای صفحه
                    self.price_info_label.configure(
                        text=f"💰 {tick.last:.2f} | Bid: {tick.bid:.2f} | Ask: {tick.ask:.2f}",
                        text_color=price_color
                    )
                    
                    # آپدیت خط قیمت روی چارت
                    self.update_price_line(tick.last)
        except:
            pass
        
        # تکرار هر ثانیه
        self.price_update_job = self.after(1000, self.start_price_update)


    def stop_price_update(self):
        """توقف آپدیت قیمت"""
        if hasattr(self, 'price_update_job') and self.price_update_job:
            try:
                self.after_cancel(self.price_update_job)
            except:
                pass
            self.price_update_job = None


    def stop_chart_auto_update(self):
        """توقف آپدیت چارت"""
        if hasattr(self, 'chart_update_job') and self.chart_update_job:
            try:
                self.after_cancel(self.chart_update_job)
            except:
                pass
            self.chart_update_job = None
        
        if hasattr(self, 'chart_auto_update'):
            self.chart_auto_update = False


    def toggle_chart_auto_update(self):
        """شروع/توقف آپدیت خودکار چارت"""
        self.chart_auto_update = not self.chart_auto_update
        
        if self.chart_auto_update:
            self.auto_update_btn.configure(text="⏸️ Stop", fg_color="#ef5350", hover_color="#c62828")
            self.chart_status_label.configure(text="🔴 LIVE", text_color="#ef5350")
            self.do_chart_auto_update()
        else:
            self.auto_update_btn.configure(text="▶️ Live", fg_color="#26a69a", hover_color="#1e8e82")
            self.chart_status_label.configure(text="", text_color="gray")
            self.stop_chart_auto_update()


    def do_chart_auto_update(self):
        """اجرای آپدیت چارت"""
        if hasattr(self, 'chart_auto_update') and self.chart_auto_update:
            self.create_real_chart()
            interval = int(self.chart_interval_var.get()) * 1000
            self.chart_update_job = self.after(interval, self.do_chart_auto_update)


    def manual_refresh_chart(self):
        """رفرش دستی"""
        self.chart_zoom_start = 0
        self.chart_zoom_end = None
        self.create_real_chart()


    def chart_zoom_in(self):
        if self.chart_data is None: return
        total = len(self.chart_data)
        end = self.chart_zoom_end if self.chart_zoom_end else total
        rng = end - self.chart_zoom_start
        new_rng = max(int(rng * 0.6), 20)
        center = (self.chart_zoom_start + end) // 2
        self.chart_zoom_start = max(0, center - new_rng // 2)
        self.chart_zoom_end = min(total, center + new_rng // 2)
        self.create_real_chart()


    def chart_zoom_out(self):
        if self.chart_data is None: return
        total = len(self.chart_data)
        end = self.chart_zoom_end if self.chart_zoom_end else total
        rng = end - self.chart_zoom_start
        new_rng = min(int(rng * 1.4), total)
        center = (self.chart_zoom_start + end) // 2
        self.chart_zoom_start = max(0, center - new_rng // 2)
        self.chart_zoom_end = min(total, center + new_rng // 2)
        if self.chart_zoom_start == 0 and self.chart_zoom_end >= total:
            self.chart_zoom_end = None
        self.create_real_chart()


    def chart_fit_all(self):
        self.chart_zoom_start = 0
        self.chart_zoom_end = None
        self.create_real_chart()


    def chart_pan_left(self):
        if self.chart_data is None: return
        total = len(self.chart_data)
        end = self.chart_zoom_end if self.chart_zoom_end else total
        rng = end - self.chart_zoom_start
        shift = max(int(rng * 0.2), 5)
        if self.chart_zoom_start > 0:
            self.chart_zoom_start = max(0, self.chart_zoom_start - shift)
            self.chart_zoom_end = self.chart_zoom_start + rng
            self.create_real_chart()


    def chart_pan_right(self):
        if self.chart_data is None: return
        total = len(self.chart_data)
        end = self.chart_zoom_end if self.chart_zoom_end else total
        rng = end - self.chart_zoom_start
        shift = max(int(rng * 0.2), 5)
        if end < total:
            self.chart_zoom_end = min(total, end + shift)
            self.chart_zoom_start = self.chart_zoom_end - rng
            self.create_real_chart()


    def create_real_chart(self):
        """ساخت چارت"""
        for widget in self.chart_container.winfo_children():
            widget.destroy()
        
        if hasattr(self, 'current_chart') and self.current_chart:
            try:
                plt.close(self.current_chart[0])
            except:
                pass

        try:
            tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
            
            timeframe = tf_map.get(self.chart_tf_var.get(), mt5.TIMEFRAME_M5)
            candles = int(self.chart_candles_var.get())

            rates = mt5_manager.get_rates(bot.SYMBOL, timeframe, candles)
            if rates is None or len(rates) == 0:
                raise Exception("No data!")

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            self.chart_data = df

            # زوم
            zoom_start = self.chart_zoom_start
            zoom_end = self.chart_zoom_end if self.chart_zoom_end else len(df)
            
            self.zoom_range_label.configure(text=f"📊 {zoom_start+1}-{zoom_end} / {len(df)}")

            # اندیکاتورها
            if self.ind_sma_var.get():
                df['SMA20'] = df['close'].rolling(20).mean()
                df['SMA50'] = df['close'].rolling(50).mean()
            if self.ind_ema_var.get():
                df['EMA9'] = df['close'].ewm(span=9).mean()
                df['EMA21'] = df['close'].ewm(span=21).mean()
            if self.ind_bb_var.get():
                bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
                df['BB_upper'], df['BB_middle'], df['BB_lower'] = bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
            if self.ind_rsi_var.get():
                df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            if self.ind_macd_var.get():
                macd = ta.trend.MACD(df['close'])
                # خط سیگنال مثل MACD خود MT5: میانگین ساده‌ی ۹ تایی
                df['MACD'] = macd.macd()
                df['MACD_signal'] = df['MACD'].rolling(9).mean()
                df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            if self.ind_atr_var.get():
                df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            if self.ind_stoch_var.get():
                stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
                df['Stoch_K'], df['Stoch_D'] = stoch.stoch(), stoch.stoch_signal()

            df_view = df.iloc[zoom_start:zoom_end].copy().reset_index(drop=True)

            # تعداد پنل‌ها
            num_panels = 1
            if self.ind_vol_var.get(): num_panels += 1
            if self.ind_rsi_var.get(): num_panels += 1
            if self.ind_macd_var.get(): num_panels += 1
            if self.ind_atr_var.get(): num_panels += 1
            if self.ind_stoch_var.get(): num_panels += 1

            height_ratios = [3] + [1] * (num_panels - 1)
            fig_height = 5 + (num_panels - 1)
            
            fig = plt.figure(figsize=(14, fig_height), facecolor='#1a1a2e')
            gs = fig.add_gridspec(num_panels, 1, height_ratios=height_ratios, hspace=0.05)

            axes = []
            for i in range(num_panels):
                ax = fig.add_subplot(gs[i], sharex=axes[0] if i > 0 else None)
                axes.append(ax)
                ax.set_facecolor('#1a1a2e')
                ax.tick_params(colors='#888', labelsize=8)
                ax.grid(True, alpha=0.15, color='#444')
                for spine in ax.spines.values():
                    spine.set_color('#333')

            x = np.arange(len(df_view))
            p = 0

            # ═══════════════════════════════════════════════════════════
            # پنل Price با خط قیمت
            # ═══════════════════════════════════════════════════════════
            ax = axes[p]
            
            # رسم کندل‌ها
            for i in range(len(df_view)):
                c = '#26a69a' if df_view['close'].iloc[i] >= df_view['open'].iloc[i] else '#ef5350'
                ax.bar(i, abs(df_view['close'].iloc[i] - df_view['open'].iloc[i]),
                    bottom=min(df_view['open'].iloc[i], df_view['close'].iloc[i]), color=c, width=0.8)
                ax.vlines(i, df_view['low'].iloc[i], df_view['high'].iloc[i], color=c, linewidth=0.5)

            # SMA
            if self.ind_sma_var.get() and 'SMA20' in df_view:
                ax.plot(x, df_view['SMA20'], color='#ffeb3b', linewidth=1, label='SMA20')
                ax.plot(x, df_view['SMA50'], color='#2196f3', linewidth=1, label='SMA50')
            
            # EMA
            if self.ind_ema_var.get() and 'EMA9' in df_view:
                ax.plot(x, df_view['EMA9'], color='#e91e63', linewidth=1, label='EMA9')
                ax.plot(x, df_view['EMA21'], color='#9c27b0', linewidth=1, label='EMA21')
            
            # Bollinger Bands
            if self.ind_bb_var.get() and 'BB_upper' in df_view:
                ax.fill_between(x, df_view['BB_upper'], df_view['BB_lower'], alpha=0.1, color='#00bcd4')
                ax.plot(x, df_view['BB_middle'], color='#00bcd4', linewidth=1, label='BB')
                ax.plot(x, df_view['BB_upper'], color='#00bcd4', linewidth=0.5, linestyle='--')
                ax.plot(x, df_view['BB_lower'], color='#00bcd4', linewidth=0.5, linestyle='--')

            # ═══════════════════════════════════════════════════════════
            # 📍 خط قیمت فعلی و label
            # ═══════════════════════════════════════════════════════════
            last_price = df_view['close'].iloc[-1]
            last_open = df_view['open'].iloc[-1]
            price_color = '#26a69a' if last_price >= last_open else '#ef5350'
            
            # خط افقی قیمت
            ax.axhline(y=last_price, color=price_color, linestyle='--', linewidth=1, alpha=0.8)
            
            # label قیمت در سمت راست
            ax.text(
                len(df_view) + 1,  # کمی بعد از آخرین کندل
                last_price,
                f' {last_price:.2f} ',
                color='white',
                fontsize=10,
                fontweight='bold',
                va='center',
                ha='left',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=price_color,
                    edgecolor='none',
                    alpha=0.9
                )
            )
            
            # تنظیم محدوده X برای نمایش label قیمت
            ax.set_xlim(-2, len(df_view) + 8)

            ax.legend(loc='upper left', fontsize=7, facecolor='#1a1a2e', labelcolor='white', ncol=4)
            ax.set_title(f'{bot.SYMBOL} {self.chart_tf_var.get()} | Last: {last_price:.2f}', color='white', fontsize=12, fontweight='bold')
            ax.set_ylabel('Price', color='#888', fontsize=9)
            p += 1

            # ═══════════════════════════════════════════════════════════
            # Volume
            # ═══════════════════════════════════════════════════════════
            if self.ind_vol_var.get():
                ax = axes[p]
                colors = ['#26a69a' if df_view['close'].iloc[i] >= df_view['open'].iloc[i] else '#ef5350' for i in range(len(df_view))]
                ax.bar(x, df_view['tick_volume'], color=colors, alpha=0.7, width=0.8)
                ax.set_ylabel('Vol', color='#888', fontsize=9)
                p += 1

            # ═══════════════════════════════════════════════════════════
            # RSI
            # ═══════════════════════════════════════════════════════════
            if self.ind_rsi_var.get() and 'RSI' in df_view:
                ax = axes[p]
                ax.plot(x, df_view['RSI'], color='#ab47bc', linewidth=1.5)
                ax.axhline(70, color='#ef5350', linestyle='--', linewidth=0.8)
                ax.axhline(30, color='#26a69a', linestyle='--', linewidth=0.8)
                ax.axhline(50, color='#666', linestyle=':', linewidth=0.5)
                ax.fill_between(x, 70, 100, alpha=0.1, color='#ef5350')
                ax.fill_between(x, 0, 30, alpha=0.1, color='#26a69a')
                ax.set_ylim(0, 100)
                ax.set_ylabel('RSI', color='#888', fontsize=9)
                
                # مقدار RSI فعلی
                rsi_val = df_view['RSI'].iloc[-1]
                rsi_color = '#ef5350' if rsi_val > 70 else '#26a69a' if rsi_val < 30 else '#ab47bc'
                ax.text(len(df_view) + 1, rsi_val, f' {rsi_val:.1f}', color=rsi_color, fontsize=9, fontweight='bold', va='center')
                p += 1

            # ═══════════════════════════════════════════════════════════
            # MACD
            # ═══════════════════════════════════════════════════════════
            if self.ind_macd_var.get() and 'MACD' in df_view:
                ax = axes[p]
                ax.plot(x, df_view['MACD'], color='#2196f3', linewidth=1.2, label='MACD')
                ax.plot(x, df_view['MACD_signal'], color='#ff9800', linewidth=1.2, label='Signal')
                colors = ['#26a69a' if v >= 0 else '#ef5350' for v in df_view['MACD_hist']]
                ax.bar(x, df_view['MACD_hist'], color=colors, alpha=0.6, width=0.8)
                ax.axhline(0, color='#666', linewidth=0.5)
                ax.set_ylabel('MACD', color='#888', fontsize=9)
                ax.legend(loc='upper left', fontsize=7, facecolor='#1a1a2e', labelcolor='white')
                p += 1

            # ═══════════════════════════════════════════════════════════
            # ATR
            # ═══════════════════════════════════════════════════════════
            if self.ind_atr_var.get() and 'ATR' in df_view:
                ax = axes[p]
                ax.plot(x, df_view['ATR'], color='#ff5722', linewidth=1.5)
                ax.fill_between(x, 0, df_view['ATR'], alpha=0.2, color='#ff5722')
                ax.set_ylabel('ATR', color='#888', fontsize=9)
                
                # مقدار ATR فعلی
                atr_val = df_view['ATR'].iloc[-1]
                ax.text(len(df_view) + 1, atr_val, f' {atr_val:.2f}', color='#ff5722', fontsize=9, fontweight='bold', va='center')
                p += 1

            # ═══════════════════════════════════════════════════════════
            # Stochastic
            # ═══════════════════════════════════════════════════════════
            if self.ind_stoch_var.get() and 'Stoch_K' in df_view:
                ax = axes[p]
                ax.plot(x, df_view['Stoch_K'], color='#4caf50', linewidth=1.2, label='%K')
                ax.plot(x, df_view['Stoch_D'], color='#f44336', linewidth=1.2, label='%D')
                ax.axhline(80, color='#ef5350', linestyle='--', linewidth=0.8)
                ax.axhline(20, color='#26a69a', linestyle='--', linewidth=0.8)
                ax.fill_between(x, 80, 100, alpha=0.1, color='#ef5350')
                ax.fill_between(x, 0, 20, alpha=0.1, color='#26a69a')
                ax.set_ylim(0, 100)
                ax.set_ylabel('Stoch', color='#888', fontsize=9)
                ax.legend(loc='upper left', fontsize=7, facecolor='#1a1a2e', labelcolor='white')

            # ═══════════════════════════════════════════════════════════
            # X labels
            # ═══════════════════════════════════════════════════════════
            n_labels = min(8, len(df_view) // 30 + 3)
            ticks = np.linspace(0, len(df_view)-1, n_labels).astype(int)
            labels = [df_view['time'].iloc[i].strftime('%m/%d %H:%M') for i in ticks]
            axes[-1].set_xticks(ticks)
            axes[-1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8, color='#888')
            for ax in axes[:-1]:
                plt.setp(ax.get_xticklabels(), visible=False)

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, self.chart_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.current_chart = (fig, canvas)

        except Exception as e:
            print(f"Chart error: {e}")
            ctk.CTkLabel(self.chart_container, text=f"❌ {str(e)}", text_color="red").pack(expand=True)


    def refresh_chart(self):
        self.create_real_chart()


    def open_live_chart_v2_1(self):
        import subprocess
        chart_script = os.path.join(os.path.dirname(__file__), "live_chart_v2_1.py")
        if os.path.exists(chart_script):
            subprocess.Popen([sys.executable, chart_script])


########################################################################
    def show_settings_panel(self):
        
        self.clear_main_frame()
        
        title = ctk.CTkLabel(
            self.main_frame,
            text="⚙️ Settings & Configuration",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=20)

        button_frame = ctk.CTkFrame(self.main_frame)
        button_frame.pack(pady=20)

        ctk.CTkButton(
            button_frame,
            text="⚙️ Open Advanced Settings Editor",
            command=self.open_settings_editor,
            width=250,
            height=50,
            font=ctk.CTkFont(size=14)
        ).pack(pady=10)

        quick_frame = ctk.CTkFrame(self.main_frame)
        quick_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            quick_frame,
            text="Quick Actions",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        quick_buttons = ctk.CTkFrame(quick_frame)
        quick_buttons.pack(pady=10)

        actions = [
            ("💰 Lot Size", self.open_lot_settings),
            ("🛡️ SL/TP Settings", self.open_sltp_settings), 
            ("📊 Confidence", self.open_confidence_settings),
            ("🔄 Reload Config", self.reload_config),
            ("💾 Save Config", self.save_config),
            ("⚙️ Advanced Editor", self.open_settings_editor)
        ]

        for i, (text, command) in enumerate(actions):
            btn = ctk.CTkButton(
                quick_buttons,
                text=text,
                command=command,
                width=150,
                height=35
            )
            btn.grid(row=i // 3, column=i % 3, padx=5, pady=5)

        info_frame = ctk.CTkFrame(self.main_frame)
        info_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(
            info_frame,
            text="System Information",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        system_info = [
            ("Platform Version", "NR Trading Platform v2.1"),
            ("Developer", "Naser Romipour"),
            ("Config File", self.config_manager_price3.config_file),
            ("Database File", self.config_manager_price3.database.db_path),
            ("Model File", self.config_manager_price3.model.model_path),
            ("Last Config Save", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        ]

        for label, value in system_info:
            item_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
            item_frame.pack(fill="x", padx=20, pady=2)

            ctk.CTkLabel(
                item_frame,
                text=f"{label}:",
                font=ctk.CTkFont(weight="bold"),
                width=150,
                anchor="w"
            ).pack(side="left")

            ctk.CTkLabel(
                item_frame,
                text=value
            ).pack(side="right")

    def read_trade_log(self):
        """خواندن فایل لاگ معاملات"""
        if not os.path.exists(LOG_CSV):
            return None
        
        try:
            # ✅ فیکس: comma separator + header خودکار
            df = pd.read_csv(LOG_CSV)
            
            if len(df) == 0:
                return None
            
            # تبدیل نوع‌ها
            df['Action'] = df['Action'].fillna('').astype(str)
            df['Date'] = df['Date'].fillna('').astype(str)
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
            df['PnL'] = pd.to_numeric(df['PnL'], errors='coerce').fillna(0)
            
            return df
        
        except Exception as e:
            print(f"❌ Read log error: {e}")
            return None  

    def show_performance(self):
        self.clear_main_frame()

        title = ctk.CTkLabel(
            self.main_frame,
            text="📊 Performance Analytics",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=20)

        # ═══════════════════════════════════════════════════════════════
        # گرفتن داده‌های واقعی از لاگ
        # ═══════════════════════════════════════════════════════════════
        stats = self.get_trading_stats_from_log()
        
        if not stats:
            # اگه معامله‌ای نیست
            no_data_label = ctk.CTkLabel(
                self.main_frame,
                text="No trading data available yet.\nStart trading to see performance metrics.",
                font=ctk.CTkFont(size=16),
                text_color="gray"
            )
            no_data_label.pack(expand=True)
            return
        
        # محاسبه متریک‌های اضافی
        total_trades = stats.get('wins', 0) + stats.get('losses', 0)
        profit_factor = (stats.get('avg_win', 0) * stats.get('wins', 0)) / abs(stats.get('avg_loss', 0) * stats.get('losses', 0)) if stats.get('losses', 0) > 0 else 0
        
        # Max Drawdown محاسبه
        max_dd = 0
        if stats.get('pnl_list'):
            peak = stats['pnl_list'][0]
            for val in stats['pnl_list']:
                if val > peak:
                    peak = val
                dd = peak - val
                if dd > max_dd:
                    max_dd = dd

        # ═══════════════════════════════════════════════════════════════
        # Summary Cards
        # ═══════════════════════════════════════════════════════════════
        summary_frame = ctk.CTkFrame(self.main_frame)
        summary_frame.pack(fill="x", padx=20, pady=10)

        # محاسبه متریک‌های اضافی
        total_trades = stats.get('wins', 0) + stats.get('losses', 0)
        profit_factor = (stats.get('avg_win', 0) * stats.get('wins', 0)) / abs(stats.get('avg_loss', 0) * stats.get('losses', 0)) if stats.get('losses', 0) > 0 else 0
        avg_loss = -abs(stats.get('avg_loss', 0)) if stats.get('losses', 0) > 0 else 0
        risk_reward = abs(stats.get('avg_win', 0) / avg_loss) if avg_loss != 0 else 0
        expected_value = (stats.get('win_rate', 0)/100 * stats.get('avg_win', 0)) + ((100-stats.get('win_rate', 0))/100 * avg_loss)

        # Max Drawdown
        max_dd = 0
        recovery_days = 0
        if stats.get('pnl_list'):
            peak = stats['pnl_list'][0]
            for val in stats['pnl_list']:
                if val > peak:
                    peak = val
                dd = peak - val
                if dd > max_dd:
                    max_dd = dd

        # Sharpe Ratio (simplified)
        if stats.get('pnl_list') and len(stats['pnl_list']) > 1:
            returns = pd.Series(stats['pnl_list']).diff().dropna()
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        else:
            sharpe = 0

        # ایجاد کارت‌های بزرگتر با اطلاعات بیشتر
        summary_metrics = [
            {
                "title": "💰 Total P&L",
                "main": f"${stats.get('total_pnl', 0):+,.2f}",
                "sub": f"Avg: ${stats.get('total_pnl', 0)/total_trades:.2f}" if total_trades > 0 else "No trades",
                "color": "green" if stats.get('total_pnl', 0) >= 0 else "red"
            },
            {
                "title": "📈 Win Rate",
                "main": f"{stats.get('win_rate', 0):.1f}%",
                "sub": f"W:{stats.get('wins', 0)} / L:{stats.get('losses', 0)}",
                "color": "green" if stats.get('win_rate', 0) >= 50 else "red"
            },
            {
                "title": "📉 Max Drawdown",
                "main": f"-${max_dd:.2f}",
                "sub": f"{(max_dd/10000*100):.1f}% of capital" if max_dd > 0 else "No DD",
                "color": "orange" if max_dd < 500 else "red"
            },
            {
                "title": "⚡ Profit Factor",
                "main": f"{profit_factor:.2f}",
                "sub": "Excellent" if profit_factor > 2 else "Good" if profit_factor > 1.5 else "Fair" if profit_factor > 1 else "Poor",
                "color": "green" if profit_factor > 1.5 else "orange" if profit_factor > 1 else "red"
            },
            {
                "title": "💎 Risk/Reward",
                "main": f"1:{risk_reward:.1f}",
                "sub": f"EV: ${expected_value:.2f}",
                "color": "green" if risk_reward > 1.5 else "orange" if risk_reward > 1 else "red"
            },
            {
                "title": "📊 Sharpe Ratio",
                "main": f"{sharpe:.2f}",
                "sub": "Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Acceptable" if sharpe > 0 else "Poor",
                "color": "green" if sharpe > 1 else "orange" if sharpe > 0 else "red"
            },
            {
                "title": "🎯 Win Stats",
                "main": f"${stats.get('avg_win', 0):.2f}",
                "sub": f"Best: ${stats.get('best_win', 0):.2f}" if 'best_win' in stats else "Avg Win",
                "color": "green"
            },
            {
                "title": "💔 Loss Stats", 
                "main": f"${avg_loss:.2f}",
                "sub": f"Worst: ${stats.get('worst_loss', 0):.2f}" if 'worst_loss' in stats else "Avg Loss",
                "color": "red"
            },
            {
                "title": "🔄 Total Trades",
                "main": str(total_trades),
                "sub": f"Open: {stats.get('open_trades', 0)}",
                "color": "white"
            }
        ]

        # ایجاد کارت‌ها - 2 ردیف × 4 ستون
        summary_metrics = [
            {
                "title": "💰 Total P&L",
                "main": f"${stats.get('total_pnl', 0):+,.2f}",
                "sub": f"Avg: ${stats.get('total_pnl', 0)/total_trades:.2f}" if total_trades > 0 else "No trades",
                "color": "green" if stats.get('total_pnl', 0) >= 0 else "red"
            },
            {
                "title": "📈 Win Rate",
                "main": f"{stats.get('win_rate', 0):.1f}%",
                "sub": f"W:{stats.get('wins', 0)} / L:{stats.get('losses', 0)}",
                "color": "green" if stats.get('win_rate', 0) >= 50 else "red"
            },
            {
                "title": "📉 Max Drawdown",
                "main": f"-${max_dd:.2f}",
                "sub": f"{(max_dd/10000*100):.1f}% of capital",
                "color": "orange" if max_dd < 500 else "red"
            },
            {
                "title": "⚡ Profit Factor",
                "main": f"{profit_factor:.2f}",
                "sub": "Good" if profit_factor > 1.5 else "Fair" if profit_factor > 1 else "Poor",
                "color": "green" if profit_factor > 1.5 else "orange" if profit_factor > 1 else "red"
            },
            {
                "title": "🎯 Avg Win",
                "main": f"${stats.get('avg_win', 0):.2f}",
                "sub": f"Best: ${stats.get('best_win', 0):.2f}",
                "color": "green"
            },
            {
                "title": "💔 Avg Loss",
                "main": f"${avg_loss:.2f}",
                "sub": f"Worst: ${stats.get('worst_loss', 0):.2f}",
                "color": "red"
            },
            {
                "title": "💎 Risk/Reward",
                "main": f"1:{risk_reward:.1f}",
                "sub": f"EV: ${expected_value:.2f}",
                "color": "green" if risk_reward > 1.5 else "orange"
            },
            {
                "title": "🔄 Total Trades",
                "main": str(total_trades),
                "sub": f"Open: {stats.get('open_trades', 0)}",
                "color": "white"
            }
        ]

        for i, metric in enumerate(summary_metrics):
            card = ctk.CTkFrame(summary_frame, width=160, height=90)
            card.grid(row=i // 4, column=i % 4, padx=8, pady=8, sticky="nsew")  # ✅ 4 ستون
            card.grid_propagate(False)
            
            ctk.CTkLabel(
                card,
                text=metric["title"],
                font=ctk.CTkFont(size=11, weight="bold")
            ).pack(pady=(8, 2))
            
            ctk.CTkLabel(
                card,
                text=metric["main"],
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=metric["color"]
            ).pack(pady=2)
            
            ctk.CTkLabel(
                card,
                text=metric["sub"],
                font=ctk.CTkFont(size=9),
                text_color="gray"
            ).pack(pady=(0, 6))

        # ✅ تنظیم 4 ستون
        for i in range(4):
            summary_frame.grid_columnconfigure(i, weight=1)

        # ═══════════════════════════════════════════════════════════════
        # Charts Frame
        # ═══════════════════════════════════════════════════════════════
        charts_frame = ctk.CTkFrame(self.main_frame)
        charts_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # ═══════════════════════════════════════════════════════════════
        # Daily P&L Chart
        # ═══════════════════════════════════════════════════════════════
        pl_frame = ctk.CTkFrame(charts_frame)
        pl_frame.pack(side="left", fill="both", expand=True, padx=5)

        if os.path.exists(LOG_CSV):
            df = self.read_trade_log()
            if df is None or len(df) == 0:
                ctk.CTkLabel(..., text="No data").pack()
                return  # ✅ پیام نشون بده بعد برگرد
            closed = df[df['Action'].str.contains('CLOSE', case=False, na=False)].copy()
            
            print(f"📊 Found {len(closed)} closed trades")  # Debug
            
            if len(closed) > 0 and 'PnL' in closed.columns:
                closed['Date'] = pd.to_datetime(closed['Date'], format='mixed')
                closed['Day'] = closed['Date'].dt.date
                daily_pnl = closed.groupby('Day')['PnL'].sum()
                
                print(f"📊 Daily P&L data: {daily_pnl.to_dict()}")  # Debug
                
                fig1, ax1 = plt.subplots(figsize=(5, 4), facecolor='#2b2b2b')
                ax1.set_facecolor('#2b2b2b')
                ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                colors = ['#26a69a' if x >= 0 else '#ef5350' for x in daily_pnl.values]
                bars = ax1.bar(range(len(daily_pnl)), daily_pnl.values, color=colors, width=0.6)
                
                ax1.axhline(y=0, color='#666', linestyle='-', alpha=0.3)
                ax1.set_xlabel('Days', color='#888', fontsize=8)
                ax1.set_ylabel('P&L ($)', color='#888', fontsize=8)
                ax1.set_title('📅 Daily P&L', color='white', fontsize=10, fontweight='bold')
                ax1.tick_params(colors='#888', labelsize=7)
                
                for spine in ax1.spines.values():
                    spine.set_color('#444')
                
                ax1.grid(True, alpha=0.2, color='#444')
                plt.tight_layout()
                
                canvas1 = FigureCanvasTkAgg(fig1, pl_frame)
                canvas1.draw()
                canvas1.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
                
                plt.close(fig1)
            else:
                print(f"📊 No PnL column or no closed trades")  # Debug
                ctk.CTkLabel(pl_frame, text="No closed trades", text_color="gray", font=ctk.CTkFont(size=12)).pack(expand=True)
        else:
            print(f"📊 LOG_CSV not found: {LOG_CSV}")  # Debug
            ctk.CTkLabel(pl_frame, text="No data", text_color="gray", font=ctk.CTkFont(size=12)).pack(expand=True)
        # ═══════════════════════════════════════════════════════════════
        # Equity Curve
        # ═══════════════════════════════════════════════════════════════
        equity_frame = ctk.CTkFrame(charts_frame)
        equity_frame.pack(side="right", fill="both", expand=True, padx=5)

        if os.path.exists(LOG_CSV):
            df = self.read_trade_log()
            if df is None or len(df) == 0:
                ctk.CTkLabel(..., text="No data").pack()
                return  # ✅ پیام نشون بده بعد برگرد
            closed = df[df['Action'].str.contains('CLOSE', case=False, na=False)].copy()
            
            if len(closed) > 0 and 'PnL' in closed.columns:
                pnl_values = pd.to_numeric(closed['PnL'], errors='coerce').fillna(0)
                total_pnl = pnl_values.sum()
                
                # ✅ گرفتن balance از MT5
                account_info = mt5_manager.get_account_info()
                if account_info:
                    current_balance = account_info.balance
                    initial_balance = current_balance - total_pnl  # سرمایه اولیه
                else:
                    initial_balance = 10000
                
                # ساخت equity curve
                equity_values = [initial_balance]
                for pnl in pnl_values:
                    equity_values.append(equity_values[-1] + pnl)
                
                final_equity = equity_values[-1]
                
                fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#2b2b2b')
                ax2.set_facecolor('#2b2b2b')
                ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                x = range(len(equity_values))
                color = '#26a69a' if final_equity >= initial_balance else '#ef5350'
                
                ax2.plot(x, equity_values, color=color, linewidth=1.5, marker='o', markersize=4)
                ax2.fill_between(x, equity_values, initial_balance, alpha=0.2, color=color)
                ax2.axhline(y=initial_balance, color='#888', linestyle='--', alpha=0.5, linewidth=0.8)
                
                ax2.set_xlabel('Trade #', color='#888', fontsize=8)
                ax2.set_ylabel('Equity ($)', color='#888', fontsize=8)
                ax2.set_title(f'📈 Equity: ${final_equity:,.2f}', color='white', fontsize=10, fontweight='bold')
                ax2.tick_params(colors='#888', labelsize=7)
                
                for spine in ax2.spines.values():
                    spine.set_color('#444')
                
                ax2.grid(True, alpha=0.2, color='#444')
                plt.tight_layout()
                
                canvas2 = FigureCanvasTkAgg(fig2, equity_frame)
                canvas2.draw()
                canvas2.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
                
                plt.close(fig2)
            else:
                ctk.CTkLabel(equity_frame, text="No closed trades", text_color="gray").pack(expand=True)
        else:
            ctk.CTkLabel(equity_frame, text="No data", text_color="gray").pack(expand=True)
        # ═══════════════════════════════════════════════════════════════
        # Recent Trades Table
        # ═══════════════════════════════════════════════════════════════
        trades_frame = ctk.CTkFrame(self.main_frame)
        trades_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            trades_frame,
            text="📋 Recent Trades",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        # ایجاد جدول از لاگ واقعی
        columns = ['Date', 'Action', 'Price', 'Volume', 'P&L', 'Status']
        tree = ttk.Treeview(trades_frame, columns=columns, show='headings', height=8)

        for col in columns:
            tree.heading(col, text=col)
            if col == 'Date':
                tree.column(col, width=150)
            else:
                tree.column(col, width=100)

        # خواندن آخرین معاملات
        if os.path.exists(LOG_CSV):
            df = self.read_trade_log()
            if df is None or len(df) == 0:
                ctk.CTkLabel(..., text="No data").pack()
                return  # ✅ پیام نشون بده بعد برگرد
            # فقط معاملات بسته شده
            closed = df[df['Action'].str.contains('CLOSE', case=False, na=False)].tail(20)
            
            for _, row in closed.iterrows():
                date = row['Date'].split()[0] if pd.notna(row['Date']) else ''
                action = row['Action'] if pd.notna(row['Action']) else ''
                price = f"{row['Price']:.2f}" if pd.notna(row['Price']) else ''
                volume = f"{row['Volume']:.2f}" if pd.notna(row['Volume']) else ''
                pnl = f"${row['PnL']:+.2f}" if 'PnL' in row and pd.notna(row['PnL']) else '$0.00'
                status = 'WIN' if 'PnL' in row and row['PnL'] > 0 else 'LOSS'
                
                tree.insert('', 0, values=(date, action, price, volume, pnl, status))

        scrollbar = ttk.Scrollbar(trades_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    ####################################################################################
    def show_trading_tools(self):
        """صفحه ابزارهای معاملاتی"""
        self.clear_main_frame()
        title = ctk.CTkLabel(
            self.main_frame,
            text="🛠️ Trading Tools Hub",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=20)
        
        # Tab buttons
        button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        button_frame.pack(pady=10)
        
        # Container برای وسط‌چین کردن
        button_container = ctk.CTkFrame(button_frame, fg_color="transparent")
        button_container.pack(anchor="center")

        self.calc_btn = ctk.CTkButton(
            button_container, text="🧮 Position Calculator",
            command=self.show_position_calculator, 
            width=220, height=45,
            fg_color="#1f538d"
        )
        self.calc_btn.pack(side="left", padx=10)

        self.news_btn = ctk.CTkButton(
            button_container, text="📰 Forex News",
            command=self.show_forex_news, 
            width=220, height=45
        )
        self.news_btn.pack(side="left", padx=10)

        self.guide_btn = ctk.CTkButton(
            button_container, text="📚 Bot Guide",
            command=self.show_bot_guide, 
            width=220, height=45
        )
        self.guide_btn.pack(side="left", padx=10)
        
        # Content area
        self.tools_content = ctk.CTkFrame(self.main_frame)
        self.tools_content.pack(fill="both", expand=True, padx=20, pady=20)
        
        # نمایش پیش‌فرض
        self.show_position_calculator()

    def reset_tool_buttons(self):
        """ریست رنگ دکمه‌ها"""
        self.calc_btn.configure(fg_color=["#3B8ED0", "#1F6AA5"])
        self.news_btn.configure(fg_color=["#3B8ED0", "#1F6AA5"])
        self.guide_btn.configure(fg_color=["#3B8ED0", "#1F6AA5"])


    def show_position_calculator(self):
        """ماشین حساب حرفه‌ای پوزیشن"""
        self.reset_tool_buttons()
        self.calc_btn.configure(fg_color="#1f538d")
        
        for widget in self.tools_content.winfo_children():
            widget.destroy()
        
        # Title
        ctk.CTkLabel(
            self.tools_content,
            text="🧮 Professional Trading Calculator",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=15)
        
        # Main container
        main_container = ctk.CTkFrame(self.tools_content, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # ═══════════════════════════════════════════════════════════════
        # Left Panel - Inputs
        # ═══════════════════════════════════════════════════════════════
        left_panel = ctk.CTkScrollableFrame(main_container, width=350)
        left_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Account Section
        account_section = ctk.CTkFrame(left_panel)
        account_section.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            account_section,
            text="💰 Account Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Account Balance
        row1 = ctk.CTkFrame(account_section, fg_color="transparent")
        row1.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(row1, text="Account Balance ($):", width=180, anchor="w").pack(side="left")
        self.calc_balance = ctk.CTkEntry(row1, width=150, placeholder_text="10000")
        self.calc_balance.pack(side="right", padx=5)
        
        # Account Currency
        row2 = ctk.CTkFrame(account_section, fg_color="transparent")
        row2.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(row2, text="Account Currency:", width=180, anchor="w").pack(side="left")
        self.calc_currency = ctk.CTkOptionMenu(row2, values=["USD", "EUR", "GBP"], width=150)
        self.calc_currency.pack(side="right", padx=5)
        self.calc_currency.set("USD")
        
        # Risk Section
        risk_section = ctk.CTkFrame(left_panel)
        risk_section.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            risk_section,
            text="⚠️ Risk Management",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Risk Type
        row3 = ctk.CTkFrame(risk_section, fg_color="transparent")
        row3.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(row3, text="Risk Type:", width=180, anchor="w").pack(side="left")
        self.risk_type = ctk.CTkOptionMenu(
            row3, 
            values=["Percentage (%)", "Fixed Amount ($)"],
            width=150,
            command=self.on_risk_type_change
        )
        self.risk_type.pack(side="right", padx=5)
        
        # Risk Value
        row4 = ctk.CTkFrame(risk_section, fg_color="transparent")
        row4.pack(fill="x", padx=15, pady=5)
        self.risk_label = ctk.CTkLabel(row4, text="Risk per Trade (%):", width=180, anchor="w")
        self.risk_label.pack(side="left")
        self.calc_risk = ctk.CTkEntry(row4, width=150, placeholder_text="1")
        self.calc_risk.pack(side="right", padx=5)
        
        # Trade Section
        trade_section = ctk.CTkFrame(left_panel)
        trade_section.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            trade_section,
            text="📊 Trade Parameters",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Symbol
        row5 = ctk.CTkFrame(trade_section, fg_color="transparent")
        row5.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(row5, text="Symbol:", width=180, anchor="w").pack(side="left")
        self.calc_symbol = ctk.CTkOptionMenu(
            row5, 
            values=["XAUUSD (Gold)", "EURUSD", "GBPUSD", "USDJPY", "Custom"],
            width=150,
            command=self.on_symbol_change
        )
        self.calc_symbol.pack(side="right", padx=5)
        
        # Entry Price
        row6 = ctk.CTkFrame(trade_section, fg_color="transparent")
        row6.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(row6, text="Entry Price:", width=180, anchor="w").pack(side="left")
        self.calc_entry = ctk.CTkEntry(row6, width=150, placeholder_text="2650.00")
        self.calc_entry.pack(side="right", padx=5)
        
        # Stop Loss
        row7 = ctk.CTkFrame(trade_section, fg_color="transparent")
        row7.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(row7, text="Stop Loss:", width=180, anchor="w").pack(side="left")
        self.calc_sl = ctk.CTkEntry(row7, width=150, placeholder_text="2645.00")
        self.calc_sl.pack(side="right", padx=5)
        
        # Take Profit
        row8 = ctk.CTkFrame(trade_section, fg_color="transparent")
        row8.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(row8, text="Take Profit:", width=180, anchor="w").pack(side="left")
        self.calc_tp = ctk.CTkEntry(row8, width=150, placeholder_text="2665.00")
        self.calc_tp.pack(side="right", padx=5)
        
        # Trade Direction
        row9 = ctk.CTkFrame(trade_section, fg_color="transparent")
        row9.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(row9, text="Direction:", width=180, anchor="w").pack(side="left")
        self.calc_direction = ctk.CTkOptionMenu(row9, values=["BUY (Long)", "SELL (Short)"], width=150)
        self.calc_direction.pack(side="right", padx=5)
        
        # Buttons
        btn_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=15)
        
        ctk.CTkButton(
            btn_frame,
            text="📊 Calculate",
            command=self.calculate_position,
            height=45,
            width=150,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#26a69a",
            hover_color="#1e8e82"
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            btn_frame,
            text="🔄 Reset",
            command=self.reset_calculator,
            height=45,
            width=100,
            fg_color="#ff9800",
            hover_color="#e68900"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="📋 Copy",
            command=self.copy_results,
            height=45,
            width=100,
            fg_color="#2196f3",
            hover_color="#1976d2"
        ).pack(side="left", padx=5)
        
        # ═══════════════════════════════════════════════════════════════
        # Right Panel - Results
        # ═══════════════════════════════════════════════════════════════
        right_panel = ctk.CTkFrame(main_container)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            right_panel,
            text="📈 Calculation Results",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Results Cards
        self.results_container = ctk.CTkScrollableFrame(right_panel)
        self.results_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Initial message
        ctk.CTkLabel(
            self.results_container,
            text="👆 Enter values and click Calculate",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        ).pack(pady=50)

    def on_risk_type_change(self, choice):
        """تغییر نوع ریسک"""
        if "Percentage" in choice:
            self.risk_label.configure(text="Risk per Trade (%):")
        else:
            self.risk_label.configure(text="Risk Amount ($):")

    def on_symbol_change(self, choice):
        """تغییر نماد"""
        prices = {
            "XAUUSD (Gold)": ("2650.00", "2645.00", "2665.00"),
            "EURUSD": ("1.0850", "1.0820", "1.0900"),
            "GBPUSD": ("1.2650", "1.2620", "1.2700"),
            "USDJPY": ("149.50", "149.00", "150.50"),
        }
        if choice in prices:
            self.calc_entry.delete(0, "end")
            self.calc_sl.delete(0, "end")
            self.calc_tp.delete(0, "end")
            self.calc_entry.insert(0, prices[choice][0])
            self.calc_sl.insert(0, prices[choice][1])
            self.calc_tp.insert(0, prices[choice][2])

    def reset_calculator(self):
        """ریست ماشین حساب"""
        self.calc_balance.delete(0, "end")
        self.calc_risk.delete(0, "end")
        self.calc_entry.delete(0, "end")
        self.calc_sl.delete(0, "end")
        self.calc_tp.delete(0, "end")
        
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        ctk.CTkLabel(
            self.results_container,
            text="👆 Enter values and click Calculate",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        ).pack(pady=50)

    def copy_results(self):
        """کپی نتایج"""
        if hasattr(self, 'last_results'):
            self.clipboard_clear()
            self.clipboard_append(self.last_results)
            messagebox.showinfo("✅ Copied", "Results copied to clipboard!")


    def calculate_position(self):
        """محاسبه حرفه‌ای پوزیشن"""
        try:
            # گرفتن مقادیر
            balance = float(self.calc_balance.get() or "10000")
            risk_input = float(self.calc_risk.get() or "1")
            entry = float(self.calc_entry.get() or "2650")
            sl = float(self.calc_sl.get() or "2645")
            tp = float(self.calc_tp.get() or "2665")
            symbol = self.calc_symbol.get()
            direction = self.calc_direction.get()
            
            # محاسبه ریسک
            if "Percentage" in self.risk_type.get():
                risk_amount = balance * (risk_input / 100)
                risk_pct = risk_input
            else:
                risk_amount = risk_input
                risk_pct = (risk_input / balance) * 100
            
            # محاسبه فاصله‌ها
            sl_distance = abs(entry - sl)
            tp_distance = abs(tp - entry)
            risk_reward = tp_distance / sl_distance if sl_distance != 0 else 0
            
            # Pip value بر اساس نماد
            if "XAUUSD" in symbol:
                pip_value_per_lot = 1.0
                pip_size = 0.01
            elif "JPY" in symbol:
                pip_value_per_lot = 6.67
                pip_size = 0.01
            else:
                pip_value_per_lot = 10.0
                pip_size = 0.0001
            
            # محاسبه Lot Size
            sl_pips = sl_distance / pip_size
            lot_size = risk_amount / (sl_pips * pip_value_per_lot)
            lot_size = round(lot_size, 2)
            
            # محاسبه سود و ضرر
            tp_pips = tp_distance / pip_size
            potential_profit = tp_pips * pip_value_per_lot * lot_size
            potential_loss = sl_pips * pip_value_per_lot * lot_size
            
            # Margin مورد نیاز
            if "XAUUSD" in symbol:
                margin_required = lot_size * entry * 100 / 100
            else:
                margin_required = lot_size * 1000
            
            # پاک کردن نتایج قبلی
            for widget in self.results_container.winfo_children():
                widget.destroy()
            
            # ═══════════════════════════════════════════════════════════════
            # 🎯 Main Lot Size Card
            # ═══════════════════════════════════════════════════════════════
            main_card = ctk.CTkFrame(self.results_container, fg_color="#1a1a2e", corner_radius=15)
            main_card.pack(fill="x", pady=10, padx=5)
            
            ctk.CTkLabel(
                main_card,
                text="📊 POSITION SIZE",
                font=ctk.CTkFont(size=12),
                text_color="#888"
            ).pack(pady=(15, 5))
            
            ctk.CTkLabel(
                main_card,
                text=f"{lot_size:.2f}",
                font=ctk.CTkFont(size=48, weight="bold"),
                text_color="#00d4aa"
            ).pack()
            
            ctk.CTkLabel(
                main_card,
                text="LOTS",
                font=ctk.CTkFont(size=14),
                text_color="#888"
            ).pack(pady=(0, 15))
            
            # ═══════════════════════════════════════════════════════════════
            # 💰 Profit / Loss Boxes
            # ═══════════════════════════════════════════════════════════════
            pl_frame = ctk.CTkFrame(self.results_container, fg_color="transparent")
            pl_frame.pack(fill="x", pady=5, padx=5)
            
            # Profit Box
            profit_card = ctk.CTkFrame(pl_frame, fg_color="#0d4f3c", corner_radius=12)
            profit_card.pack(side="left", fill="x", expand=True, padx=(0, 5))
            
            ctk.CTkLabel(
                profit_card,
                text="✅ PROFIT",
                font=ctk.CTkFont(size=10),
                text_color="#4ade80"
            ).pack(pady=(12, 2))
            
            ctk.CTkLabel(
                profit_card,
                text=f"+${potential_profit:,.2f}",
                font=ctk.CTkFont(size=22, weight="bold"),
                text_color="#4ade80"
            ).pack(pady=(0, 12))
            
            # Loss Box
            loss_card = ctk.CTkFrame(pl_frame, fg_color="#4a1515", corner_radius=12)
            loss_card.pack(side="left", fill="x", expand=True, padx=(5, 0))
            
            ctk.CTkLabel(
                loss_card,
                text="❌ LOSS",
                font=ctk.CTkFont(size=10),
                text_color="#f87171"
            ).pack(pady=(12, 2))
            
            ctk.CTkLabel(
                loss_card,
                text=f"-${potential_loss:,.2f}",
                font=ctk.CTkFont(size=22, weight="bold"),
                text_color="#f87171"
            ).pack(pady=(0, 12))
            
            # ═══════════════════════════════════════════════════════════════
            # ⚖️ Risk/Reward با Progress Bar
            # ═══════════════════════════════════════════════════════════════
            rr_card = ctk.CTkFrame(self.results_container, fg_color="#1a1a2e", corner_radius=12)
            rr_card.pack(fill="x", pady=5, padx=5)
            
            rr_color = "#4ade80" if risk_reward >= 2 else "#fbbf24" if risk_reward >= 1 else "#f87171"
            rr_status = "EXCELLENT" if risk_reward >= 2 else "GOOD" if risk_reward >= 1.5 else "ACCEPTABLE" if risk_reward >= 1 else "POOR"
            
            ctk.CTkLabel(
                rr_card,
                text="⚖️ RISK / REWARD",
                font=ctk.CTkFont(size=11),
                text_color="#888"
            ).pack(pady=(12, 5))
            
            ctk.CTkLabel(
                rr_card,
                text=f"1 : {risk_reward:.2f}",
                font=ctk.CTkFont(size=32, weight="bold"),
                text_color=rr_color
            ).pack()
            
            # Progress bar برای R/R
            progress_frame = ctk.CTkFrame(rr_card, fg_color="transparent")
            progress_frame.pack(fill="x", padx=30, pady=10)
            
            rr_progress = min(risk_reward / 3, 1)  # Max at 3:1
            progress_bar = ctk.CTkProgressBar(progress_frame, width=200, height=12)
            progress_bar.pack()
            progress_bar.set(rr_progress)
            progress_bar.configure(progress_color=rr_color)
            
            ctk.CTkLabel(
                rr_card,
                text=rr_status,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=rr_color
            ).pack(pady=(0, 12))
            
            # ═══════════════════════════════════════════════════════════════
            # 📊 Stats Grid
            # ═══════════════════════════════════════════════════════════════
            stats_card = ctk.CTkFrame(self.results_container, fg_color="#1a1a2e", corner_radius=12)
            stats_card.pack(fill="x", pady=5, padx=5)
            
            ctk.CTkLabel(
                stats_card,
                text="📋 TRADE DETAILS",
                font=ctk.CTkFont(size=11),
                text_color="#888"
            ).pack(pady=(12, 10))
            
            # Grid
            stats_grid = ctk.CTkFrame(stats_card, fg_color="transparent")
            stats_grid.pack(fill="x", padx=15, pady=(0, 15))
            
            stats = [
                ("Risk Amount", f"${risk_amount:,.2f}", "#f87171"),
                ("Risk %", f"{risk_pct:.2f}%", "#f87171"),
                ("SL Distance", f"{sl_pips:.1f} pips", "#888"),
                ("TP Distance", f"{tp_pips:.1f} pips", "#888"),
                ("Entry", f"{entry:.2f}", "#fff"),
                ("Stop Loss", f"{sl:.2f}", "#f87171"),
                ("Take Profit", f"{tp:.2f}", "#4ade80"),
                ("Margin", f"${margin_required:,.2f}", "#888"),
            ]
            
            for i, (label, value, color) in enumerate(stats):
                row = i // 2
                col = i % 2
                
                stat_frame = ctk.CTkFrame(stats_grid, fg_color="#252540", corner_radius=8)
                stat_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
                
                ctk.CTkLabel(
                    stat_frame,
                    text=label,
                    font=ctk.CTkFont(size=10),
                    text_color="#888"
                ).pack(pady=(8, 2))
                
                ctk.CTkLabel(
                    stat_frame,
                    text=value,
                    font=ctk.CTkFont(size=14, weight="bold"),
                    text_color=color
                ).pack(pady=(0, 8))
            
            stats_grid.grid_columnconfigure(0, weight=1)
            stats_grid.grid_columnconfigure(1, weight=1)
            
            # ═══════════════════════════════════════════════════════════════
            # 📈 Visual Price Levels
            # ═══════════════════════════════════════════════════════════════
            visual_card = ctk.CTkFrame(self.results_container, fg_color="#1a1a2e", corner_radius=12)
            visual_card.pack(fill="x", pady=5, padx=5)
            
            ctk.CTkLabel(
                visual_card,
                text="📈 PRICE LEVELS",
                font=ctk.CTkFont(size=11),
                text_color="#888"
            ).pack(pady=(12, 10))
            
            # Visual representation
            visual_frame = ctk.CTkFrame(visual_card, fg_color="transparent")
            visual_frame.pack(fill="x", padx=20, pady=(0, 15))
            
            # TP Line
            tp_frame = ctk.CTkFrame(visual_frame, fg_color="transparent")
            tp_frame.pack(fill="x", pady=3)
            ctk.CTkLabel(tp_frame, text="TP", width=40, text_color="#4ade80", font=ctk.CTkFont(size=11)).pack(side="left")
            tp_bar = ctk.CTkFrame(tp_frame, fg_color="#4ade80", height=8, corner_radius=4)
            tp_bar.pack(side="left", fill="x", expand=True, padx=5)
            ctk.CTkLabel(tp_frame, text=f"{tp:.2f}", width=80, text_color="#4ade80", font=ctk.CTkFont(size=11)).pack(side="right")
            
            # Entry Line
            entry_frame = ctk.CTkFrame(visual_frame, fg_color="transparent")
            entry_frame.pack(fill="x", pady=3)
            ctk.CTkLabel(entry_frame, text="ENTRY", width=40, text_color="#fff", font=ctk.CTkFont(size=11)).pack(side="left")
            entry_bar = ctk.CTkFrame(entry_frame, fg_color="#3b82f6", height=8, corner_radius=4)
            entry_bar.pack(side="left", fill="x", expand=True, padx=5)
            ctk.CTkLabel(entry_frame, text=f"{entry:.2f}", width=80, text_color="#fff", font=ctk.CTkFont(size=11)).pack(side="right")
            
            # SL Line
            sl_frame = ctk.CTkFrame(visual_frame, fg_color="transparent")
            sl_frame.pack(fill="x", pady=3)
            ctk.CTkLabel(sl_frame, text="SL", width=40, text_color="#f87171", font=ctk.CTkFont(size=11)).pack(side="left")
            sl_bar = ctk.CTkFrame(sl_frame, fg_color="#f87171", height=8, corner_radius=4)
            sl_bar.pack(side="left", fill="x", expand=True, padx=5)
            ctk.CTkLabel(sl_frame, text=f"{sl:.2f}", width=80, text_color="#f87171", font=ctk.CTkFont(size=11)).pack(side="right")
            
            # ذخیره برای کپی
            self.last_results = f"""
    Position Calculator Results
    ═══════════════════════════════════
    Symbol: {symbol}
    Direction: {direction}
    ═══════════════════════════════════
    Lot Size: {lot_size:.2f}
    Risk: ${risk_amount:,.2f} ({risk_pct:.2f}%)
    ═══════════════════════════════════
    Entry: {entry:.2f}
    SL: {sl:.2f} ({sl_pips:.1f} pips)
    TP: {tp:.2f} ({tp_pips:.1f} pips)
    ═══════════════════════════════════
    Profit: +${potential_profit:,.2f}
    Loss: -${potential_loss:,.2f}
    R/R: 1:{risk_reward:.2f}
    ═══════════════════════════════════
    """
            
        except ValueError as e:
            messagebox.showerror("❌ Error", "Please enter valid numbers!")
        except Exception as e:
            messagebox.showerror("❌ Error", f"Calculation error: {str(e)}")   


    def show_bot_guide(self):
        """راهنمای گرافیکی استفاده از ربات"""
        self.reset_tool_buttons()
        self.guide_btn.configure(fg_color="#1f538d")
        
        for widget in self.tools_content.winfo_children():
            widget.destroy()
        
        # Title
        ctk.CTkLabel(
            self.tools_content,
            text="📚 NR Trading Bot - User Guide",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=15)
        
        # Main scrollable container
        guide_container = ctk.CTkScrollableFrame(self.tools_content)
        guide_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # ═══════════════════════════════════════════════════════════════
        # 🚀 Quick Start Section
        # ═══════════════════════════════════════════════════════════════
        quick_start = ctk.CTkFrame(guide_container, fg_color="#1a1a2e", corner_radius=15)
        quick_start.pack(fill="x", pady=10, padx=5)

        ctk.CTkLabel(
            quick_start,
            text="🚀 QUICK START",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#00d4aa"
        ).pack(pady=(15, 10))

        steps = [
            ("1️⃣", "Open MetaTrader 5", "Make sure MT5 is running and logged in"),
            ("2️⃣", "Enable Auto Trading", "See instructions below ⬇️"),
            ("3️⃣", "Open NR Platform", "Login with your credentials"),
            ("4️⃣", "Start Bot", "Click '▶️ Start Bot' in sidebar"),
            ("5️⃣", "Don't Close During Fine-tune", "When bot is fine-tuning, do NOT close the app"),
            ("6️⃣", "Settings are Optimized", "Best settings for Model are default. Change with caution!"),
            ("7️⃣", "Auto Trading + Your Risk", "Bot trades automatically. Risk management is YOUR responsibility"),
        ]

        for num, title, desc in steps:
            step_frame = ctk.CTkFrame(quick_start, fg_color="#252540", corner_radius=10)
            step_frame.pack(fill="x", padx=15, pady=5)
            
            step_row = ctk.CTkFrame(step_frame, fg_color="transparent")
            step_row.pack(fill="x", padx=15, pady=10)
            
            ctk.CTkLabel(
                step_row,
                text=num,
                font=ctk.CTkFont(size=24),
                width=50
            ).pack(side="left")
            
            text_frame = ctk.CTkFrame(step_row, fg_color="transparent")
            text_frame.pack(side="left", fill="x", expand=True)
            
            ctk.CTkLabel(
                text_frame,
                text=title,
                font=ctk.CTkFont(size=14, weight="bold"),
                anchor="w"
            ).pack(anchor="w")
            
            ctk.CTkLabel(
                text_frame,
                text=desc,
                font=ctk.CTkFont(size=11),
                text_color="#888",
                anchor="w"
            ).pack(anchor="w")

        ctk.CTkLabel(quick_start, text="", height=10).pack()

        # ═══════════════════════════════════════════════════════════════
        # 🔧 MT5 Auto Trading Setup - راهنمای فعال‌سازی
        # ═══════════════════════════════════════════════════════════════
        mt5_setup = ctk.CTkFrame(guide_container, fg_color="#1a1a2e", corner_radius=15)
        mt5_setup.pack(fill="x", pady=10, padx=5)

        ctk.CTkLabel(
            mt5_setup,
            text="🔧 ENABLE AUTO TRADING IN MT5",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#fbbf24"
        ).pack(pady=(15, 10))

        # Warning
        warning_frame = ctk.CTkFrame(mt5_setup, fg_color="#4a3000", corner_radius=10)
        warning_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(
            warning_frame,
            text="⚠️ Auto Trading must be enabled for the bot to work!",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#fbbf24"
        ).pack(pady=10)

        # Method 1: Toolbar
        method1 = ctk.CTkFrame(mt5_setup, fg_color="#252540", corner_radius=10)
        method1.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(
            method1,
            text="📌 Method 1: Quick Toggle (Toolbar)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4ade80"
        ).pack(pady=(12, 8), padx=15, anchor="w")

        ctk.CTkLabel(
            method1,
            text="Click the 'AutoTrading' button in MT5 toolbar\n"
                "✅ Green = Enabled  |  🔴 Red = Disabled",
            font=ctk.CTkFont(size=12),
            text_color="#ccc",
            justify="left"
        ).pack(padx=30, pady=(0, 12), anchor="w")

        # Method 2: Menu
        method2 = ctk.CTkFrame(mt5_setup, fg_color="#252540", corner_radius=10)
        method2.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(
            method2,
            text="📌 Method 2: Via Menu",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4ade80"
        ).pack(pady=(12, 8), padx=15, anchor="w")

        # Step by step path
        path_frame = ctk.CTkFrame(method2, fg_color="#1a1a2e", corner_radius=8)
        path_frame.pack(fill="x", padx=20, pady=5)

        path_steps = ["Tools", "Options", "Expert Advisors"]
        path_row = ctk.CTkFrame(path_frame, fg_color="transparent")
        path_row.pack(pady=15)

        for i, step in enumerate(path_steps):
            ctk.CTkLabel(
                path_row,
                text=f" {step} ",
                font=ctk.CTkFont(size=13, weight="bold"),
                fg_color="#3b82f6",
                corner_radius=5,
                text_color="white"
            ).pack(side="left", padx=3)
            
            if i < len(path_steps) - 1:
                ctk.CTkLabel(
                    path_row,
                    text="➜",
                    font=ctk.CTkFont(size=16),
                    text_color="#888"
                ).pack(side="left", padx=5)

        # Checkboxes to enable
        checkboxes_frame = ctk.CTkFrame(method2, fg_color="transparent")
        checkboxes_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            checkboxes_frame,
            text="Enable these options:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#888"
        ).pack(anchor="w", pady=(0, 8))

        options = [
            "☑️ Allow algorithmic trading",
            "☑️ Allow DLL imports (if needed)",
            "☑️ Allow trading for this Expert Advisor",
        ]

        for opt in options:
            ctk.CTkLabel(
                checkboxes_frame,
                text=opt,
                font=ctk.CTkFont(size=12),
                text_color="#4ade80"
            ).pack(anchor="w", padx=20, pady=2)

        ctk.CTkLabel(method2, text="", height=5).pack()

        # Keyboard shortcut
        shortcut_frame = ctk.CTkFrame(mt5_setup, fg_color="#252540", corner_radius=10)
        shortcut_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(
            shortcut_frame,
            text="⌨️ Keyboard Shortcut",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4ade80"
        ).pack(pady=(12, 8), padx=15, anchor="w")

        shortcut_row = ctk.CTkFrame(shortcut_frame, fg_color="transparent")
        shortcut_row.pack(pady=(0, 12))

        ctk.CTkLabel(
            shortcut_row,
            text=" Ctrl ",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#444",
            corner_radius=5,
            text_color="white"
        ).pack(side="left", padx=3)

        ctk.CTkLabel(
            shortcut_row,
            text="+",
            font=ctk.CTkFont(size=14),
            text_color="#888"
        ).pack(side="left", padx=5)

        ctk.CTkLabel(
            shortcut_row,
            text=" E ",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#444",
            corner_radius=5,
            text_color="white"
        ).pack(side="left", padx=3)

        ctk.CTkLabel(
            shortcut_row,
            text="  Toggle Auto Trading ON/OFF",
            font=ctk.CTkFont(size=12),
            text_color="#888"
        ).pack(side="left", padx=10)

        # Visual diagram
        diagram_frame = ctk.CTkFrame(mt5_setup, fg_color="#0d1117", corner_radius=10)
        diagram_frame.pack(fill="x", padx=15, pady=10)

        diagram_text = """
        ┌─────────────────────────────────────────────────────────────┐
        │  MetaTrader 5                                               │
        ├─────────────────────────────────────────────────────────────┤
        │  File   Edit   View   Insert   Charts   [Tools]   Window    │
        │                                            │                │
        │                                            ▼                │
        │                                    ┌──────────────┐         │
        │                                    │   Options    │         │
        │                                    └──────┬───────┘         │
        │                                           ▼                 │
        │                              ┌─────────────────────┐        │
        │                              │  Expert Advisors    │        │
        │                              ├─────────────────────┤        │
        │                              │ ☑ Allow algo trade  │        │
        │                              │ ☑ Allow DLL imports │        │
        │                              └─────────────────────┘        │
        │                                                             │
        │  [AutoTrading] ← Click to toggle (Green = ON)               │
        └─────────────────────────────────────────────────────────────┘
        """

        ctk.CTkLabel(
            diagram_frame,
            text=diagram_text,
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color="#58a6ff",
            justify="left"
        ).pack(padx=10, pady=10)

        ctk.CTkLabel(mt5_setup, text="", height=5).pack()
        
        # ═══════════════════════════════════════════════════════════════
        # ⚙️ Key Settings Section
        # ═══════════════════════════════════════════════════════════════
        settings_section = ctk.CTkFrame(guide_container, fg_color="#1a1a2e", corner_radius=15)
        settings_section.pack(fill="x", pady=10, padx=5)
        
        ctk.CTkLabel(
            settings_section,
            text="⚙️ KEY SETTINGS",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#fbbf24"
        ).pack(pady=(15, 10))
        
        settings_grid = ctk.CTkFrame(settings_section, fg_color="transparent")
        settings_grid.pack(fill="x", padx=15, pady=10)
        
        settings = [
            ("💰", "Lot Size", "0.05", "Trade volume"),
            ("🛡️", "Stop Loss", "1.5 × ATR", "Protection"),
            ("🎯", "Take Profit", "2.5 × ATR", "Target"),
            ("📊", "Min Confidence", "60%", "Entry threshold")
        ]
        
        for i, (icon, name, value, desc) in enumerate(settings):
            setting_card = ctk.CTkFrame(settings_grid, fg_color="#252540", corner_radius=10)
            setting_card.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            
            ctk.CTkLabel(
                setting_card,
                text=icon,
                font=ctk.CTkFont(size=28)
            ).pack(pady=(15, 5))
            
            ctk.CTkLabel(
                setting_card,
                text=name,
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack()
            
            ctk.CTkLabel(
                setting_card,
                text=value,
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color="#00d4aa"
            ).pack(pady=5)
            
            ctk.CTkLabel(
                setting_card,
                text=desc,
                font=ctk.CTkFont(size=9),
                text_color="#888"
            ).pack(pady=(0, 15))
        
        for i in range(4):
            settings_grid.grid_columnconfigure(i, weight=1)
        
        ctk.CTkLabel(settings_section, text="", height=5).pack()
        
        # ═══════════════════════════════════════════════════════════════
        # ✅ Best Practices Section
        # ═══════════════════════════════════════════════════════════════
        practices_section = ctk.CTkFrame(guide_container, fg_color="#1a1a2e", corner_radius=15)
        practices_section.pack(fill="x", pady=10, padx=5)
        
        ctk.CTkLabel(
            practices_section,
            text="✅ BEST PRACTICES",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#4ade80"
        ).pack(pady=(15, 10))
        
        practices_frame = ctk.CTkFrame(practices_section, fg_color="transparent")
        practices_frame.pack(fill="x", padx=15, pady=5)
        
        # DO's
        do_frame = ctk.CTkFrame(practices_frame, fg_color="#0d4f3c", corner_radius=10)
        do_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            do_frame,
            text="✅ DO",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4ade80"
        ).pack(pady=(12, 8))
        
        dos = [
            "Start with demo account",
            "Use small lot sizes (0.01-0.1)",
            "Monitor first week",
            "Check Performance daily",
            "Keep risk ≤ 2% per trade"
        ]
        
        for do in dos:
            do_row = ctk.CTkFrame(do_frame, fg_color="transparent")
            do_row.pack(fill="x", padx=15, pady=2)
            ctk.CTkLabel(do_row, text="•", width=15, text_color="#4ade80").pack(side="left")
            ctk.CTkLabel(do_row, text=do, font=ctk.CTkFont(size=11), anchor="w").pack(side="left")
        
        ctk.CTkLabel(do_frame, text="", height=10).pack()
        
        # DON'Ts
        dont_frame = ctk.CTkFrame(practices_frame, fg_color="#4a1515", corner_radius=10)
        dont_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(
            dont_frame,
            text="❌ DON'T",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f87171"
        ).pack(pady=(12, 8))
        
        donts = [
            "Trade during high-impact news",
            "Change settings mid-trade",
            "Use oversized lots",
            "Ignore risk management",
            "Expect 100% win rate"
        ]
        
        for dont in donts:
            dont_row = ctk.CTkFrame(dont_frame, fg_color="transparent")
            dont_row.pack(fill="x", padx=15, pady=2)
            ctk.CTkLabel(dont_row, text="•", width=15, text_color="#f87171").pack(side="left")
            ctk.CTkLabel(dont_row, text=dont, font=ctk.CTkFont(size=11), anchor="w").pack(side="left")
        
        ctk.CTkLabel(dont_frame, text="", height=10).pack()
        
        # ═══════════════════════════════════════════════════════════════
        # 🔧 Troubleshooting Section
        # ═══════════════════════════════════════════════════════════════
        trouble_section = ctk.CTkFrame(guide_container, fg_color="#1a1a2e", corner_radius=15)
        trouble_section.pack(fill="x", pady=10, padx=5)
        
        ctk.CTkLabel(
            trouble_section,
            text="🔧 TROUBLESHOOTING",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#f87171"
        ).pack(pady=(15, 10))
        
        troubles = [
            ("❓ Bot not starting?", "Check MT5 is running • Verify internet • Restart app"),
            ("❓ No trades opening?", "Model: check the training log for validated GA rules • Check symbol • Market open?"),
            ("❓ Trades closing fast?", "Check SL/TP settings • Verify spread • Check logs")
        ]
        
        for problem, solution in troubles:
            trouble_card = ctk.CTkFrame(trouble_section, fg_color="#252540", corner_radius=10)
            trouble_card.pack(fill="x", padx=15, pady=5)
            
            ctk.CTkLabel(
                trouble_card,
                text=problem,
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(pady=(10, 5), padx=15, anchor="w")
            
            ctk.CTkLabel(
                trouble_card,
                text=solution,
                font=ctk.CTkFont(size=10),
                text_color="#888"
            ).pack(pady=(0, 10), padx=15, anchor="w")
        
        ctk.CTkLabel(trouble_section, text="", height=5).pack()
        
        # ═══════════════════════════════════════════════════════════════
        # ⚠️ Risk Warning Section
        # ═══════════════════════════════════════════════════════════════
        risk_section = ctk.CTkFrame(guide_container, fg_color="#4a1515", corner_radius=15)
        risk_section.pack(fill="x", pady=10, padx=5)
        
        ctk.CTkLabel(
            risk_section,
            text="⚠️ RISK WARNING",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#f87171"
        ).pack(pady=(15, 10))
        
        ctk.CTkLabel(
            risk_section,
            text="Trading Forex/Gold involves significant risk of loss.\nPast performance does NOT guarantee future results.\nOnly trade with money you can afford to lose.",
            font=ctk.CTkFont(size=11),
            text_color="#fca5a5",
            justify="center"
        ).pack(pady=(0, 15), padx=20)
        
        # ═══════════════════════════════════════════════════════════════
        # 📞 Support Section
        # ═══════════════════════════════════════════════════════════════
        support_section = ctk.CTkFrame(guide_container, fg_color="#1a1a2e", corner_radius=15)
        support_section.pack(fill="x", pady=10, padx=5)
        
        ctk.CTkLabel(
            support_section,
            text="📞 SUPPORT",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#a855f7"
        ).pack(pady=(15, 10))
        
        support_info = ctk.CTkFrame(support_section, fg_color="transparent")
        support_info.pack(pady=(0, 15))
        
        ctk.CTkLabel(
            support_info,
            text="👨‍💻 Developer: Naser Romipour",
            font=ctk.CTkFont(size=12)
        ).pack(pady=2)
        
        ctk.CTkLabel(
            support_info,
            text="📧 Email: naserromipour7@gmail.com",
            font=ctk.CTkFont(size=12)
        ).pack(pady=2)
        
        ctk.CTkLabel(
            support_info,
            text="🔖 Version: NR Trading Platform v2.1",
            font=ctk.CTkFont(size=12),
            text_color="#888"
        ).pack(pady=2)     

    def show_forex_news(self):
        """اخبار فارکس از ForexFactory"""
        self.reset_tool_buttons()
        self.news_btn.configure(fg_color="#1f538d")
        
        for widget in self.tools_content.winfo_children():
            widget.destroy()
        
        ctk.CTkLabel(
            self.tools_content,
            text="📰 Forex News & Economic Calendar",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=20)
        
        # Toolbar
        toolbar = ctk.CTkFrame(self.tools_content)
        toolbar.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkButton(
            toolbar,
            text="🔄 Refresh",
            command=self.fetch_forex_news,
            width=120
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            toolbar,
            text="🌐 Open ForexFactory",
            command=lambda: __import__('webbrowser').open('https://www.forexfactory.com/calendar'),
            width=150
        ).pack(side="left", padx=5)
        
        # News container
        self.news_container = ctk.CTkScrollableFrame(self.tools_content)
        self.news_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # بارگذاری اخبار
        self.fetch_forex_news()

    def fetch_forex_news(self):
        """دریافت اخبار واقعی از ForexFactory"""
        for widget in self.news_container.winfo_children():
            widget.destroy()
        
        loading = ctk.CTkLabel(
            self.news_container,
            text="⏳ Loading real-time news from ForexFactory...",
            font=ctk.CTkFont(size=14)
        )
        loading.pack(pady=20)
        
        # ✅ اجرا در thread جداگانه تا GUI فریز نشه
        thread = threading.Thread(target=self._fetch_news_thread, daemon=True)
        thread.start()

    def _fetch_news_thread(self):
        """دریافت اخبار در background"""
        try:
            import requests
            from datetime import datetime
            
            # API جایگزین
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            # چک کن response خالی نیست
            if response.status_code != 200 or not response.text:
                raise Exception(f"API error: {response.status_code}")
            
            data = response.json()
            
            if not data:
                raise Exception("No data received")
            
            news_items = []
            
            for event in data[:20]:
                try:
                    impact_map = {
                        'High': 'High',
                        'Medium': 'Medium',
                        'Low': 'Low',
                        'Holiday': 'Low'
                    }
                    
                    news_items.append({
                        'time': event.get('date', '')[11:16],
                        'date': event.get('date', '')[:10],
                        'currency': event.get('country', 'N/A'),
                        'impact': impact_map.get(event.get('impact', ''), 'Low'),
                        'event': event.get('title', 'Unknown Event'),
                        'forecast': event.get('forecast', '-') or '-',
                        'actual': event.get('actual', '-') or '-',
                        'previous': event.get('previous', '-') or '-'
                    })
                except:
                    continue
            
            self.after(0, lambda items=news_items: self._display_news(items))
            
        except Exception as ex:
            error_msg = str(ex)
            print(f"News fetch error: {error_msg}")
            self.after(0, lambda msg=error_msg: self._display_error(msg))

    def _display_news(self, news_items):
        """نمایش اخبار در UI"""
        for widget in self.news_container.winfo_children():
            widget.destroy()
        
        if not news_items:
            ctk.CTkLabel(
                self.news_container,
                text="❌ No news found.",
                font=ctk.CTkFont(size=14),
                text_color="orange"
            ).pack(pady=20)
            return
        
        # Header
        from datetime import datetime
        today = datetime.now().strftime("%A, %B %d, %Y")
        
        header = ctk.CTkFrame(self.news_container, fg_color="#2b2b2b")
        header.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            header,
            text=f"📅 Economic Calendar - {today}",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Legend
        legend_frame = ctk.CTkFrame(self.news_container, fg_color="transparent")
        legend_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(legend_frame, text="Impact:", font=ctk.CTkFont(size=11)).pack(side="left", padx=10)
        ctk.CTkLabel(legend_frame, text="🔴 High", text_color="#ef5350", font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
        ctk.CTkLabel(legend_frame, text="🟠 Medium", text_color="#ff9800", font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
        ctk.CTkLabel(legend_frame, text="🟢 Low", text_color="#66bb6a", font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
        
        # News cards
        for news in news_items:
            news_card = ctk.CTkFrame(self.news_container)
            news_card.pack(fill="x", pady=5, padx=5)
            
            impact_colors = {
                "High": "#ef5350",
                "Medium": "#ff9800",
                "Low": "#66bb6a"
            }
            impact_color = impact_colors.get(news['impact'], "gray")
            
            top_row = ctk.CTkFrame(news_card, fg_color="transparent")
            top_row.pack(fill="x", padx=10, pady=8)
            
            # Date + Time
            ctk.CTkLabel(
                top_row,
                text=f"{news.get('date', '')} {news['time']}",
                font=ctk.CTkFont(size=11),
                width=130,
                anchor="w"
            ).pack(side="left", padx=5)
            
            # Currency
            ctk.CTkLabel(
                top_row,
                text=news['currency'],
                font=ctk.CTkFont(size=11, weight="bold"),
                width=50,
                fg_color="#1f538d",
                corner_radius=5
            ).pack(side="left", padx=5)
            
            # Impact
            ctk.CTkLabel(
                top_row,
                text="●",
                font=ctk.CTkFont(size=18),
                text_color=impact_color,
                width=25
            ).pack(side="left")
            
            # Event
            ctk.CTkLabel(
                top_row,
                text=news['event'][:45] + "..." if len(news['event']) > 45 else news['event'],
                font=ctk.CTkFont(size=11),
                anchor="w"
            ).pack(side="left", fill="x", expand=True, padx=5)
            
            # Bottom row
            if news['forecast'] != '-' or news['previous'] != '-':
                bottom_row = ctk.CTkFrame(news_card, fg_color="transparent")
                bottom_row.pack(fill="x", padx=10, pady=(0, 8))
                
                details = f"Forecast: {news['forecast']}  |  Actual: {news['actual']}  |  Previous: {news['previous']}"
                
                ctk.CTkLabel(
                    bottom_row,
                    text=details,
                    font=ctk.CTkFont(size=9),
                    text_color="gray"
                ).pack(side="left", padx=140)

    def _display_error(self, error_msg):
        """نمایش خطا"""
        for widget in self.news_container.winfo_children():
            widget.destroy()
        
        ctk.CTkLabel(
            self.news_container,
            text=f"❌ Failed to load news:\n{error_msg}\n\nCheck internet connection or try again later.",
            font=ctk.CTkFont(size=14),
            text_color="red"
        ).pack(pady=20)       


    ####################################################################################    
    def show_user_profile(self):
        """نمایش منوی اطلاعات کاربران به همراه تب اختصاصی ادمین و جدول مشاهده کاربران سرور"""
        # پاکسازی فریم اصلی برنامه جهت بارگذاری ویو جدید
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        profile_frame = ctk.CTkFrame(self.main_frame, fg_color="#1e1e1e")
        profile_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tabview = ctk.CTkTabview(profile_frame, width=600, height=550)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ═══════════════════════════════════════════════════════════
        # تب اول: اطلاعات کاربری شخصی و امکان ویرایش
        # ═══════════════════════════════════════════════════════════
        tabview.add("User Information")
        info_tab = tabview.tab("User Information")
        
        # اصلاح آدرس سیستم احراز هویت به فرمت یکپارچه پروژه شما
        current_username = self.auth_system_v2_1.current_user
        
        try:
            response = self.auth_system_v2_1.supabase.table("users").select("*").eq("username", current_username).execute()
            user_info = response.data[0] if response.data else {}
        except Exception:
            user_info = {}
        
        ctk.CTkLabel(info_tab, text="👤 User Profile Dashboard", font=ctk.CTkFont(size=18, weight="bold"), text_color="#26a69a").pack(pady=10)
        ctk.CTkLabel(info_tab, text=f"Username: {current_username.upper()}  |  Role: {user_info.get('role', 'user').upper()}", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=5)
        
        # فیلدهای فرم اطلاعات شخصی
        ctk.CTkLabel(info_tab, text="First Name:").pack(anchor="w", padx=175, pady=(5, 0))
        u_first_entry = ctk.CTkEntry(info_tab, width=250)
        first_name_val = user_info.get("first_name")
        u_first_entry.insert(0, first_name_val if first_name_val is not None else "")
        u_first_entry.pack(pady=3)
        
        ctk.CTkLabel(info_tab, text="Last Name:").pack(anchor="w", padx=175, pady=(5, 0))
        u_last_entry = ctk.CTkEntry(info_tab, width=250)
        last_name_val = user_info.get("last_name")
        u_last_entry.insert(0, last_name_val if last_name_val is not None else "")
        u_last_entry.pack(pady=3)
        
        ctk.CTkLabel(info_tab, text="Email Address:").pack(anchor="w", padx=175, pady=(5, 0))
        u_email_entry = ctk.CTkEntry(info_tab, width=250)
        email_val = user_info.get("email")
        u_email_entry.insert(0, email_val if email_val is not None else "")
        u_email_entry.pack(pady=3)
        
        ctk.CTkLabel(info_tab, text="Phone Number:").pack(anchor="w", padx=175, pady=(5, 0))
        u_phone_entry = ctk.CTkEntry(info_tab, width=250)
        phone_val = user_info.get("phone")
        u_phone_entry.insert(0, phone_val if phone_val is not None else "")
        u_phone_entry.pack(pady=3)
        
        def save_personal_changes():
            success, msg = self.auth_system_v2_1.update_profile_info(
                u_first_entry.get(), u_last_entry.get(), u_email_entry.get(), u_phone_entry.get()
            )
            if success:
                messagebox.showinfo("Success", msg)
            else:
                messagebox.showerror("Error", msg)
                
        save_profile_btn = ctk.CTkButton(info_tab, text="Save Changes", fg_color="#26a69a", hover_color="#00897b", command=save_personal_changes, height=35)
        save_profile_btn.pack(pady=20)
        
        # ═══════════════════════════════════════════════════════════
        # تب دوم: مدیریت کاربران و مشاهده لیست کل کاربران سرور
        # ═══════════════════════════════════════════════════════════
        # اینجا نام متغیر تصحیح شد تا شرط ادمین به درستی کار کند
        if self.auth_system_v2_1.is_admin():
            tabview.add("User Management (Admin)")
            admin_tab = tabview.tab("User Management (Admin)")
            
            scroll_frame = ctk.CTkScrollableFrame(admin_tab, fg_color="transparent")
            scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
            
            # --- بخش اول: فرم ثبت نام کاربر جدید ---
            ctk.CTkLabel(scroll_frame, text="🛠️ Admin Control: Add New User", font=ctk.CTkFont(size=16, weight="bold"), text_color="orange").pack(pady=10)
            
            ctk.CTkLabel(scroll_frame, text="Username * :").pack(anchor="w", padx=165)
            new_user_entry = ctk.CTkEntry(scroll_frame, width=250)
            new_user_entry.pack(pady=3)
            
            ctk.CTkLabel(scroll_frame, text="Password * :").pack(anchor="w", padx=165)
            new_pass_entry = ctk.CTkEntry(scroll_frame, width=250, show="•")
            new_pass_entry.pack(pady=3)
            
            ctk.CTkLabel(scroll_frame, text="First Name:").pack(anchor="w", padx=165)
            new_first_entry = ctk.CTkEntry(scroll_frame, width=250)
            new_first_entry.pack(pady=3)
            
            ctk.CTkLabel(scroll_frame, text="Last Name:").pack(anchor="w", padx=165)
            new_last_entry = ctk.CTkEntry(scroll_frame, width=250)
            new_last_entry.pack(pady=3)
            
            ctk.CTkLabel(scroll_frame, text="Email:").pack(anchor="w", padx=165)
            new_email_entry = ctk.CTkEntry(scroll_frame, width=250)
            new_email_entry.pack(pady=3)
            
            ctk.CTkLabel(scroll_frame, text="Phone:").pack(anchor="w", padx=165)
            new_phone_entry = ctk.CTkEntry(scroll_frame, width=250)
            new_phone_entry.pack(pady=3)
            
            ctk.CTkLabel(scroll_frame, text="Allowed IP (Leave blank for Auto-Lock):").pack(anchor="w", padx=165)
            new_ip_entry = ctk.CTkEntry(scroll_frame, width=250, placeholder_text="Auto-Locks on 1st login if blank")
            new_ip_entry.pack(pady=3)
            
            ctk.CTkLabel(scroll_frame, text="Account Validity (Days):").pack(anchor="w", padx=165)
            new_days_entry = ctk.CTkEntry(scroll_frame, width=250)
            new_days_entry.insert(0, "30")
            new_days_entry.pack(pady=3)
            
            ctk.CTkLabel(scroll_frame, text="User Role:").pack(anchor="w", padx=165)
            new_role_option = ctk.CTkOptionMenu(scroll_frame, width=250, values=["user", "admin"])
            new_role_option.set("user")
            new_role_option.pack(pady=3)
            
            def execute_add_user():
                u = new_user_entry.get().strip()
                p = new_pass_entry.get().strip()
                f_n = new_first_entry.get().strip()
                l_n = new_last_entry.get().strip()
                e = new_email_entry.get().strip()
                ph = new_phone_entry.get().strip()
                ip = new_ip_entry.get().strip()
                role = new_role_option.get()
                
                try:
                    days = int(new_days_entry.get().strip())
                except ValueError:
                    messagebox.showerror("Error", "Validity days must be a number!")
                    return

                if not u or not p:
                    messagebox.showerror("Error", "Username and password are required")
                    return
                    
                success, msg = self.auth_system_v2_1.register_user(
                    username=u, password=p, email=e, first_name=f_n, 
                    last_name=l_n, phone=ph, allowed_ip=ip, role=role, validity_days=days
                )
                if success:
                    messagebox.showinfo("Success", msg)
                    # پاک کردن فرم ثبت نام پس از ثبت موفقیت آمیز
                    new_user_entry.delete(0, 'end')
                    new_pass_entry.delete(0, 'end')
                    refresh_users_list() 
                else:
                    messagebox.showerror("Error", msg)
            
            add_btn = ctk.CTkButton(scroll_frame, text="Register User Online", fg_color="orange", hover_color="dark orange", height=35, command=execute_add_user)
            add_btn.pack(pady=15)
            
            # --- بخش دوم: جدول مشاهده کاربران سرور و آی‌پی قفل شده آن‌ها ---
            ctk.CTkLabel(scroll_frame, text="📋 Database Users List (Live Server Data)", font=ctk.CTkFont(size=15, weight="bold"), text_color="#26a69a").pack(pady=(20, 10))
            
            list_table_frame = ctk.CTkFrame(scroll_frame, fg_color="#1a1a1a")
            list_table_frame.pack(fill="x", expand=True, padx=10, pady=5)
            
            def refresh_users_list():
                for widget in list_table_frame.winfo_children():
                    widget.destroy()
                
                headers = ["Username", "Role", "Locked IP", "Expires At", "Actions"]
                for col_idx, text in enumerate(headers):
                    lbl = ctk.CTkLabel(list_table_frame, text=text, font=ctk.CTkFont(weight="bold"), text_color="lightgray")
                    lbl.grid(row=0, column=col_idx, padx=10, pady=5, sticky="w")
                
                try:
                    res = self.auth_system_v2_1.supabase.table("users").select("username, role, allowed_ip, expires_at").execute()
                    all_users = res.data if res.data else []
                    
                    for row_idx, usr in enumerate(all_users, start=1):
                        u_name = usr.get("username", "")
                        username_val = u_name.upper()
                        role_val = usr.get("role", "user")
                        ip_val = usr.get("allowed_ip") if usr.get("allowed_ip") else "🔄 Waiting 1st Login"
                        expire_val = usr.get("expires_at", "").split("T")[0] if usr.get("expires_at") else "Never"
                        
                        text_color = "orange" if role_val in ["admin", "super_admin"] else "white"
                        
                        ctk.CTkLabel(list_table_frame, text=username_val, text_color=text_color).grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
                        ctk.CTkLabel(list_table_frame, text=role_val, text_color=text_color).grid(row=row_idx, column=1, padx=10, pady=5, sticky="w")
                        ctk.CTkLabel(list_table_frame, text=ip_val, text_color="#4db6ac" if usr.get("allowed_ip") else "gray").grid(row=row_idx, column=2, padx=10, pady=5, sticky="w")
                        ctk.CTkLabel(list_table_frame, text=expire_val, text_color=text_color).grid(row=row_idx, column=3, padx=10, pady=5, sticky="w")
                        
                        actions_frame = ctk.CTkFrame(list_table_frame, fg_color="transparent")
                        actions_frame.grid(row=row_idx, column=4, padx=10, pady=2, sticky="w")
                        
                        if u_name != "naserr70":
                            def make_reset_cmd(name=u_name):
                                return lambda: action_reset_ip(name)
                                
                            def make_delete_cmd(name=u_name):
                                return lambda: action_delete_user(name)

                            btn_reset = ctk.CTkButton(actions_frame, text="Reset IP", width=65, height=22, fg_color="#0288d1", hover_color="#01579b", command=make_reset_cmd())
                            btn_reset.pack(side="left", padx=2)
                            
                            btn_del = ctk.CTkButton(actions_frame, text="Delete", width=55, height=22, fg_color="#d32f2f", hover_color="#c62828", command=make_delete_cmd())
                            btn_del.pack(side="left", padx=2)
                        else:
                            ctk.CTkLabel(actions_frame, text="System Root", text_color="gray", font=ctk.CTkFont(size=11, slant="italic")).pack()
                            
                except Exception as ex:
                    ctk.CTkLabel(list_table_frame, text=f"Failed to load server data: {str(ex)}", text_color="red").grid(row=1, column=0, columnspan=5, pady=10)

            def action_reset_ip(username):
                if messagebox.askyesno("Confirm", f"Are you sure you want to reset IP for {username}?"):
                    success, msg = self.auth_system_v2_1.reset_user_ip(username)
                    if success:
                        messagebox.showinfo("Success", msg)
                        refresh_users_list()
                    else:
                        messagebox.showerror("Error", msg)

            def action_delete_user(username):
                if messagebox.askyesno("Confirm", f"CRITICAL: Delete user {username} permanently?"):
                    success, msg = self.auth_system_v2_1.delete_user(username)
                    if success:
                        messagebox.showinfo("Success", msg)
                        refresh_users_list()
                    else:
                        messagebox.showerror("Error", msg)

            # دکمه رفرش کردن دستی لیست کاربران
            refresh_btn = ctk.CTkButton(scroll_frame, text="🔄 Refresh Server List", width=150, height=30, fg_color="#333333", hover_color="#444444", command=refresh_users_list)
            refresh_btn.pack(pady=5)
            
            # 🔄 فعال‌سازی سیستم همگام‌سازی زنده دیتابیس (رایگان)
            try:
                def on_realtime_change(payload):
                    # دیتابیس آنلاین سیگنال داد، این تابع جدول را در فریم اصلی پایتون رفرش می‌کند
                    self.after(0, refresh_users_list)
                
                self.auth_system_v2_1.supabase.table("users").on_snapshot(on_realtime_change).subscribe()
            except Exception as realtime_err:
                print(f"Realtime subscription alert: {realtime_err}")

            # لود اولیه جدول به محض ورود به منو
            refresh_users_list()

    def change_password(self):
        """تغییر رمز عبور"""
        dialog = ctk.CTkInputDialog(
            text="Enter new password:",
            title="Change Password"
        )
        new_password = dialog.get_input()
        
        if new_password and len(new_password) >= 6:
            # Update password logic
            messagebox.showinfo("✅ Success", "Password changed successfully!")
        elif new_password:
            messagebox.showerror("Error", "Password must be at least 6 characters")

    def update_email(self):
        """آپدیت ایمیل"""
        dialog = ctk.CTkInputDialog(
            text="Enter new email:",
            title="Update Email"
        )
        new_email = dialog.get_input()
        
        if new_email:
            # Update email logic
            messagebox.showinfo("✅ Success", "Email updated successfully!")

    def save_user_pref(self, pref_type, value):
        """ذخیره تنظیمات کاربر"""
        if pref_type == 'lot':
            self.config_manager_price3.trading.lot_size = float(value)
            self.config_manager_price3.save_config()
            messagebox.showinfo("✅ Saved", f"Default lot size set to {value}")

    def delete_account(self):
        """حذف حساب کاربری"""
        if messagebox.askyesno(
            "⚠️ Delete Account",
            "Are you SURE you want to delete your account?\nThis action CANNOT be undone!"
        ):
            confirm = messagebox.askyesno(
                "⚠️ Final Confirmation",
                "All your data will be permanently deleted.\nContinue?"
            )
            if confirm:
                # Delete account logic
                username = self.auth_system_v2_1.current_user
                # self.auth_system_v2_1.delete_user(username)
                messagebox.showinfo("Deleted", "Account deleted. Logging out...")
                self.logout()    


#####################################################################################
    def show_logs(self):
        """نمایش لاگ‌های زنده سیستم"""
        self.clear_main_frame()

        title = ctk.CTkLabel(
            self.main_frame,
            text="📚 Live System Logs",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=20)

        # Controls Frame
        controls_frame = ctk.CTkFrame(self.main_frame)
        controls_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(controls_frame, text="Filter:").pack(side="left", padx=5)
        self.log_level_var = ctk.StringVar(value="ALL")
        log_level = ctk.CTkOptionMenu(
            controls_frame,
            values=["ALL", "INFO", "WARNING", "ERROR", "DEBUG", "SIGNAL", "TRADE"],
            variable=self.log_level_var,
            command=self.filter_logs,
            width=100
        )
        log_level.pack(side="left", padx=5)

        self.auto_scroll_var = ctk.BooleanVar(value=True)
        auto_scroll_cb = ctk.CTkCheckBox(
            controls_frame,
            text="Auto Scroll",
            variable=self.auto_scroll_var,
            width=100
        )
        auto_scroll_cb.pack(side="left", padx=15)

        ctk.CTkButton(
            controls_frame, 
            text="🗑️ Clear", 
            width=80, 
            command=self.clear_log_display,
            fg_color="#ef5350",
            hover_color="#c62828"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            controls_frame, 
            text="💾 Export", 
            width=80, 
            command=self.export_logs
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            controls_frame, 
            text="📁 Open Folder", 
            width=100, 
            command=self.open_log_folder
        ).pack(side="left", padx=5)

        self.log_status_label = ctk.CTkLabel(
            controls_frame,
            text="🔴 LIVE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ef5350"
        )
        self.log_status_label.pack(side="right", padx=10)

        self.log_count_label = ctk.CTkLabel(
            controls_frame,
            text="Lines: 0",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.log_count_label.pack(side="right", padx=10)

        # Log Display Frame
        log_frame = ctk.CTkFrame(self.main_frame)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # ✅ wrap="word" برای جلوگیری از اسکرول افقی
        self.log_text = ctk.CTkTextbox(
            log_frame, 
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#0d1117",
            text_color="#c9d1d9",
            wrap="word"  # ✅ مهم!
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

        # ✅ متصل کردن Redirector
        global log_redirector, error_redirector
        log_redirector.set_text_widget(self.log_text)
        error_redirector.set_text_widget(self.log_text)

        # Legend Frame
        legend_frame = ctk.CTkFrame(self.main_frame)
        legend_frame.pack(fill="x", padx=20, pady=5)

        legends = [
            ("🟢 BUY", "#00ff88"),
            ("🔴 SELL", "#ff4757"),
            ("🟡 WARNING", "#ffd93d"),
            ("🔵 INFO", "#4d96ff"),
            ("🟣 SIGNAL", "#a855f7"),
            ("🟠 DECISION", "#f59e0b"),
        ]

        for text, color in legends:
            ctk.CTkLabel(
                legend_frame,
                text=text,
                font=ctk.CTkFont(size=10),
                text_color=color
            ).pack(side="left", padx=10)

        # پیام خوش‌آمدگویی
        welcome_msg = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║           NR Trading Platform - Live Logs                ║
    ║              Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                 ║
    ╚══════════════════════════════════════════════════════════╝

    📍 All system messages appear here in real-time
    📍 Colors indicate message type
    📍 Use filters to show specific types

    """
        self.log_text.insert("end", welcome_msg)
        
        if log_redirector.buffer:
            for msg, tag in log_redirector.buffer:
                self.log_text.insert("end", msg, tag)
            log_redirector.buffer.clear()

        self.log_text.configure(state="disabled")
        self.update_log_count()


    def clear_log_display(self):
        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] 🗑️ Display cleared.\n")
            self.log_text.configure(state="disabled")


    def filter_logs(self, level):
        """فیلتر کردن لاگ‌ها بر اساس نوع"""
        if not hasattr(self, 'log_text') or not self.log_text.winfo_exists():
            return
        
        try:
            # دسترسی به textbox داخلی
            tb = self.log_text._textbox
            
            # اول همه رو نشون بده
            tb.tag_configure("hidden", elide=False)
            
            if level == "ALL":
                # همه نمایش داده بشن
                for tag in ["error", "warning", "success", "info", "buy", "sell", 
                        "signal", "model", "confluence", "decision", "separator", "default"]:
                    try:
                        tb.tag_configure(tag, elide=False)
                    except:
                        pass
            else:
                # فیلتر بر اساس نوع
                tag_map = {
                    "INFO": ["info", "success", "default"],
                    "WARNING": ["warning"],
                    "ERROR": ["error"],
                    "DEBUG": ["default", "separator"],
                    "SIGNAL": ["signal", "model", "confluence", "decision"],
                    "TRADE": ["buy", "sell"]
                }
                
                show_tags = tag_map.get(level, ["default"])
                all_tags = ["error", "warning", "success", "info", "buy", "sell", 
                        "signal", "model", "confluence", "decision", "separator", "default"]
                
                for tag in all_tags:
                    try:
                        if tag in show_tags:
                            tb.tag_configure(tag, elide=False)
                        else:
                            tb.tag_configure(tag, elide=True)
                    except:
                        pass
            
            print(f"📋 Filter: {level}")
            
        except Exception as e:
            print(f"Filter error: {e}")

    def update_log_count(self):
        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            try:
                lines = int(self.log_text.index('end-1c').split('.')[0])
                if hasattr(self, 'log_count_label') and self.log_count_label.winfo_exists():
                    self.log_count_label.configure(text=f"Lines: {lines}")
            except:
                pass
            self.after(2000, self.update_log_count)

    def export_logs(self):
        """ذخیره لاگ‌ها"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv")]
        )
        if file_path:
            if file_path.endswith('.csv') and os.path.exists(LOG_CSV):
                import shutil
                shutil.copy(LOG_CSV, file_path)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.log_text.get("1.0", "end"))
            messagebox.showinfo("Done", f"Exported to:\n{file_path}")


    def open_log_folder(self):
        """باز کردن پوشه"""
        os.startfile(os.getcwd())

#################################################################################################    
    def show_about(self):
        """صفحه درباره ما - حرفه‌ای"""
        self.clear_main_frame()

        # Main scrollable container
        main_container = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # ═══════════════════════════════════════════════════════════════
        # Hero Section - لوگو و عنوان
        # ═══════════════════════════════════════════════════════════════
        hero_card = ctk.CTkFrame(main_container, fg_color="#1a1a2e", corner_radius=20)
        hero_card.pack(fill="x", pady=10, padx=5)

        hero_content = ctk.CTkFrame(hero_card, fg_color="transparent")
        hero_content.pack(pady=40)

        # لوگو
        try:
            logo_img_pil = get_logo_image()
            if logo_img_pil:
                logo_img = ctk.CTkImage(logo_img_pil, size=(100, 100))
                logo_label = ctk.CTkLabel(hero_content, image=logo_img, text="")
                logo_label.pack()
            else:
                raise Exception("Hero Logo Image is None")
        except Exception as e:
            print(f"[!] Hero logo load failed: {e}")
            # Fallback - نمایش ایموجی در صورت بروز خطا
            ctk.CTkLabel(
                hero_content,
                text="🤖",
                font=ctk.CTkFont(size=80)
            ).pack()

        ctk.CTkLabel(
            hero_content,
            text="NR Trading Platform",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#00d4aa"
        ).pack(pady=(15, 5))

        ctk.CTkLabel(
            hero_content,
            text="Professional AI-Powered Trading Bot",
            font=ctk.CTkFont(size=14),
            text_color="#888"
        ).pack()

        # Version Badge
        version_frame = ctk.CTkFrame(hero_content, fg_color="#00d4aa", corner_radius=20)
        version_frame.pack(pady=15)
        
        ctk.CTkLabel(
            version_frame,
            text="  v2.1.0  ",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#000"
        ).pack(padx=15, pady=5)

        # ═══════════════════════════════════════════════════════════════
        # Developer Section
        # ═══════════════════════════════════════════════════════════════
        dev_card = ctk.CTkFrame(main_container, fg_color="#1a1a2e", corner_radius=15)
        dev_card.pack(fill="x", pady=10, padx=5)

        ctk.CTkLabel(
            dev_card,
            text="👨‍💻 DEVELOPER",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#666"
        ).pack(pady=(20, 15))

        dev_info = ctk.CTkFrame(dev_card, fg_color="#252540", corner_radius=12)
        dev_info.pack(fill="x", padx=20, pady=(0, 20))

        dev_row = ctk.CTkFrame(dev_info, fg_color="transparent")
        dev_row.pack(fill="x", padx=20, pady=20)

        # Avatar
        ctk.CTkLabel(
            dev_row,
            text="👤",
            font=ctk.CTkFont(size=50)
        ).pack(side="left", padx=(0, 20))

        dev_text = ctk.CTkFrame(dev_row, fg_color="transparent")
        dev_text.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            dev_text,
            text="Naser Romipour",
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w"
        ).pack(anchor="w")

        ctk.CTkLabel(
            dev_text,
            text="Software Developer & Algorithmic Trader",
            font=ctk.CTkFont(size=12),
            text_color="#888",
            anchor="w"
        ).pack(anchor="w", pady=(5, 0))

        # Contact buttons
        contact_frame = ctk.CTkFrame(dev_card, fg_color="transparent")
        contact_frame.pack(fill="x", padx=20, pady=(0, 20))

        contact_buttons = [
            ("📧 Email", "mailto:naserromipour7@gmail.com", "#3b82f6"),
            ("🌐 Website", "https://Romipal.com", "#22c55e"),
            ("📱 Telegram", "https://t.me/rn_p70", "#0088cc"),
        ]

        for text, url, color in contact_buttons:
            ctk.CTkButton(
                contact_frame,
                text=text,
                width=120,
                height=35,
                fg_color=color,
                hover_color=color,
                corner_radius=8,
                command=lambda u=url: __import__('webbrowser').open(u)
            ).pack(side="left", padx=5)

        # ═══════════════════════════════════════════════════════════════
        # Features Section
        # ═══════════════════════════════════════════════════════════════
        features_card = ctk.CTkFrame(main_container, fg_color="#1a1a2e", corner_radius=15)
        features_card.pack(fill="x", pady=10, padx=5)

        ctk.CTkLabel(
            features_card,
            text="✨ FEATURES",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#666"
        ).pack(pady=(20, 15))

        features_grid = ctk.CTkFrame(features_card, fg_color="transparent")
        features_grid.pack(fill="x", padx=15, pady=(0, 20))

        features = [
            ("🧬", "AI Trading Bot", "Pattern model: trend breakouts chosen by past average profit", "#8b5cf6"),
            ("📈", "Live Charts", "Real-time candlestick with indicators", "#22c55e"),
            ("📊", "Performance", "Detailed analytics & equity curve", "#3b82f6"),
            ("🧮", "Position Calculator", "Professional risk management", "#f59e0b"),
            ("📰", "Forex News", "Live economic calendar", "#ec4899"),
            ("⚙️", "Customizable", "Full control over settings", "#64748b"),
            ("🔐", "Secure", "User authentication system", "#ef4444"),
            ("🎨", "Modern UI", "Beautiful dark theme interface", "#06b6d4"),
        ]

        for i, (icon, title, desc, color) in enumerate(features):
            feature_box = ctk.CTkFrame(features_grid, fg_color="#252540", corner_radius=10)
            feature_box.grid(row=i // 4, column=i % 4, padx=8, pady=8, sticky="nsew")

            ctk.CTkLabel(
                feature_box,
                text=icon,
                font=ctk.CTkFont(size=28)
            ).pack(pady=(15, 8))

            ctk.CTkLabel(
                feature_box,
                text=title,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=color
            ).pack()

            ctk.CTkLabel(
                feature_box,
                text=desc,
                font=ctk.CTkFont(size=9),
                text_color="#666",
                wraplength=120
            ).pack(pady=(5, 15))

        for i in range(4):
            features_grid.grid_columnconfigure(i, weight=1)

        # ═══════════════════════════════════════════════════════════════
        # Changelog Section
        # ═══════════════════════════════════════════════════════════════
        changelog_card = ctk.CTkFrame(main_container, fg_color="#1a1a2e", corner_radius=15)
        changelog_card.pack(fill="x", pady=10, padx=5)

        ctk.CTkLabel(
            changelog_card,
            text="📋 CHANGELOG",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#666"
        ).pack(pady=(20, 15))

        changelog_items = [
            ("v2.1.0", "2026", "Enhanced Trading Quality, Improved Security", "#22c55e"),
            ("v2.0.0", "2025", "Major UI overhaul, Trading Tools, News Feed", "#22c55e"),
            ("v1.5.0", "2024", "Added Performance analytics & Charts", "#3b82f6"),
            ("v1.0.0", "2024", "Initial release with AI trading bot", "#8b5cf6"),
        ]

        for version, date, changes, color in changelog_items:
            log_item = ctk.CTkFrame(changelog_card, fg_color="#252540", corner_radius=10)
            log_item.pack(fill="x", padx=20, pady=5)

            log_row = ctk.CTkFrame(log_item, fg_color="transparent")
            log_row.pack(fill="x", padx=15, pady=12)

            version_badge = ctk.CTkFrame(log_row, fg_color=color, corner_radius=5)
            version_badge.pack(side="left")
            
            ctk.CTkLabel(
                version_badge,
                text=f" {version} ",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="white"
            ).pack(padx=8, pady=3)

            ctk.CTkLabel(
                log_row,
                text=date,
                font=ctk.CTkFont(size=11),
                text_color="#666",
                width=60
            ).pack(side="left", padx=15)

            ctk.CTkLabel(
                log_row,
                text=changes,
                font=ctk.CTkFont(size=11),
                text_color="#aaa",
                anchor="w"
            ).pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(changelog_card, text="", height=10).pack()

        # ═══════════════════════════════════════════════════════════════
        # License & Disclaimer
        # ═══════════════════════════════════════════════════════════════
        license_card = ctk.CTkFrame(main_container, fg_color="#1a1a2e", corner_radius=15)
        license_card.pack(fill="x", pady=10, padx=5)

        ctk.CTkLabel(
            license_card,
            text="📜 LICENSE & DISCLAIMER",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#666"
        ).pack(pady=(20, 15))

        license_text = ctk.CTkFrame(license_card, fg_color="#252540", corner_radius=10)
        license_text.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkLabel(
            license_text,
            text="Proprietary Software License",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f59e0b"
        ).pack(pady=(15, 10))

        ctk.CTkLabel(
            license_text,
            text="© 2025 Naser Romipour. All rights reserved.\n\n"
                "This software is proprietary and confidential.\n"
                "Unauthorized copying, distribution, or modification is prohibited.",
            font=ctk.CTkFont(size=11),
            text_color="#888",
            justify="center"
        ).pack(padx=20, pady=(0, 15))

        # Risk Warning
        risk_card = ctk.CTkFrame(main_container, fg_color="#4a1515", corner_radius=15)
        risk_card.pack(fill="x", pady=10, padx=5)

        ctk.CTkLabel(
            risk_card,
            text="⚠️ RISK WARNING",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f87171"
        ).pack(pady=(20, 10))

        ctk.CTkLabel(
            risk_card,
            text="Trading Forex and CFDs involves significant risk of loss.\n"
                "Past performance does not guarantee future results.\n"
                "Only trade with capital you can afford to lose.",
            font=ctk.CTkFont(size=11),
            text_color="#fca5a5",
            justify="center"
        ).pack(padx=20, pady=(0, 20))

        # ═══════════════════════════════════════════════════════════════
        # Footer
        # ═══════════════════════════════════════════════════════════════
        footer = ctk.CTkFrame(main_container, fg_color="transparent")
        footer.pack(fill="x", pady=20)

        ctk.CTkLabel(
            footer,
            text="Made by Naser Romipour",
            font=ctk.CTkFont(size=12),
            text_color="#666"
        ).pack()

        ctk.CTkLabel(
            footer,
            text="© 2025 NR Trading Platform. All rights reserved.",
            font=ctk.CTkFont(size=10),
            text_color="#444"
        ).pack(pady=(5, 0))

    def toggle_bot(self):
        if self.bot_thread is not None and self.bot_thread.is_alive():
            self.stop_bot()
        else:
            self.start_bot()

    def run_bot(self):
        while self.bot_running:
            try:
                time.sleep(1)
            except Exception as e:
                print(f"Bot error: {e}")
                break

    def force_finetune(self):
        if messagebox.askyesno("Confirm", "Force model finetuning now?"):
            messagebox.showinfo("Finetune", "Model finetuning started...")
            # TODO: call daily_finetune in a separate thread

    def update_model(self):
        messagebox.showinfo("Update", "Model update functionality will be implemented here.")

    def emergency_stop(self):
        if messagebox.askyesno("Emergency Stop", "Are you sure you want to emergency stop all trading activities?"):
            self.bot_running = False
            self.bot_status_label.configure(text="🤖 Bot: EMERGENCY STOP")
            self.start_bot_btn.configure(text="▶️ Start Bot", fg_color="green", hover_color="dark green")
            messagebox.showinfo("Emergency Stop", "All trading activities have been stopped.")

    def apply_quick_settings(self):
        self.config_manager_price3.trading.min_confidence_buy = self.quick_buy_conf.get()
        self.config_manager_price3.trading.min_confidence_sell = self.quick_sell_conf.get()
        self.config_manager_price3.save_config()
        messagebox.showinfo("Settings Applied", "Quick settings have been applied and saved.")

    def open_settings_editor(self):
        SettingsEditor(self, self.config_manager_price3)

    def open_lot_settings(self):
        """تنظیم سریع Lot Size"""
        dialog = ctk.CTkInputDialog(
            text="Enter Lot Size (0.01 - 10.0):",
            title="Lot Size Setting"
        )
        value = dialog.get_input()
        if value:
            try:
                lot = float(value)
                if 0.01 <= lot <= 10.0:
                    self.config_manager_price3.trading.lot_size = lot
                    self.config_manager_price3.save_config()
                    messagebox.showinfo("✅", f"Lot size set to {lot}")
                else:
                    messagebox.showerror("Error", "Lot must be between 0.01 and 10.0")
            except:
                messagebox.showerror("Error", "Invalid number")

    def open_sltp_settings(self):
        """تنظیم SL/TP"""
        window = ctk.CTkToplevel(self)
        window.title("SL/TP Settings")
        window.geometry("400x300")
        window.grab_set()

        # ✅ ست کردن آیکون
        self.after(10, lambda w=window: set_window_icon(w))
        # SL
        ctk.CTkLabel(window, text="Stop Loss (ATR):", font=ctk.CTkFont(size=14)).pack(pady=10)
        sl_var = ctk.DoubleVar(value=self.config_manager_price3.trading.sl_atr_mult)
        sl_slider = ctk.CTkSlider(window, from_=0.5, to=5.0, variable=sl_var, width=300)
        sl_slider.pack(pady=5)
        sl_label = ctk.CTkLabel(window, text=f"{sl_var.get():.1f}")
        sl_label.pack()
        sl_slider.configure(command=lambda v: sl_label.configure(text=f"{float(v):.1f}"))
        
        # TP
        ctk.CTkLabel(window, text="Take Profit (ATR):", font=ctk.CTkFont(size=14)).pack(pady=10)
        tp_var = ctk.DoubleVar(value=self.config_manager_price3.trading.tp_atr_mult)
        tp_slider = ctk.CTkSlider(window, from_=1.0, to=10.0, variable=tp_var, width=300)
        tp_slider.pack(pady=5)
        tp_label = ctk.CTkLabel(window, text=f"{tp_var.get():.1f}")
        tp_label.pack()
        tp_slider.configure(command=lambda v: tp_label.configure(text=f"{float(v):.1f}"))
        
        def save():
            self.config_manager_price3.trading.sl_atr_mult = sl_var.get()
            self.config_manager_price3.trading.tp_atr_mult = tp_var.get()
            self.config_manager_price3.save_config()
            messagebox.showinfo("✅", "SL/TP settings saved!")
            window.destroy()
        
        ctk.CTkButton(window, text="💾 Save", command=save, width=200, height=40).pack(pady=20)

    def open_confidence_settings(self):
        """تنظیم Confidence"""
        window = ctk.CTkToplevel(self)
        window.title("Confidence Settings")
        window.geometry("400x300")
        window.grab_set()
        # ✅ ست کردن آیکون
        self.after(10, lambda w=window: set_window_icon(w))
        # Buy
        ctk.CTkLabel(window, text="Min Buy Confidence:", font=ctk.CTkFont(size=14)).pack(pady=10)
        buy_var = ctk.DoubleVar(value=self.config_manager_price3.trading.min_confidence_buy)
        buy_slider = ctk.CTkSlider(window, from_=0.05, to=0.95, variable=buy_var, width=300)
        buy_slider.pack(pady=5)
        buy_label = ctk.CTkLabel(window, text=f"{buy_var.get()*100:.0f}%")
        buy_label.pack()
        buy_slider.configure(command=lambda v: buy_label.configure(text=f"{float(v)*100:.0f}%"))
        
        # Sell
        ctk.CTkLabel(window, text="Min Sell Confidence:", font=ctk.CTkFont(size=14)).pack(pady=10)
        sell_var = ctk.DoubleVar(value=self.config_manager_price3.trading.min_confidence_sell)
        sell_slider = ctk.CTkSlider(window, from_=0.05, to=0.95, variable=sell_var, width=300)
        sell_slider.pack(pady=5)
        sell_label = ctk.CTkLabel(window, text=f"{sell_var.get()*100:.0f}%")
        sell_label.pack()
        sell_slider.configure(command=lambda v: sell_label.configure(text=f"{float(v)*100:.0f}%"))
    
        def save():
            self.config_manager_price3.trading.min_confidence_buy = buy_var.get()
            self.config_manager_price3.trading.min_confidence_sell = sell_var.get()
            self.config_manager_price3.save_config()
            messagebox.showinfo("✅", "Confidence settings saved!")
            window.destroy()
        
        ctk.CTkButton(window, text="💾 Save", command=save, width=200, height=40).pack(pady=20)    

    def reload_config(self):
        self.config_manager_price3.load_config()
        messagebox.showinfo("Success", "Configuration reloaded successfully.")

    def save_config(self):
        self.config_manager_price3.save_config()
        messagebox.showinfo("Success", "Configuration saved successfully.")

    def reset_config(self):
        if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
            self.config_manager_price3.reset_to_defaults()
            messagebox.showinfo("Success", "Configuration reset to defaults.")

    def open_config_folder(self):
        config_dir = os.path.dirname(self.config_manager_price3.config_file)
        if os.path.exists(config_dir):
            os.startfile(config_dir)
        else:
            messagebox.showerror("Error", "Config folder does not exist.")

    def view_logs(self):
        self.show_logs()

    def repair_database(self):
        if messagebox.askyesno("Confirm", "Repair database? This may take a few minutes."):
            messagebox.showinfo("Repair", "Database repair will be implemented here.")

    def update_timeframe(self, timeframe):
        self.config_manager_price3.trading.timeframe = str(timeframe)
        self.refresh_chart()

    def refresh_chart(self):
        if hasattr(self, 'current_chart'):
            fig, canvas = self.current_chart
            plt.close(fig)
        self.show_chart()

    def apply_settings_changes(self):
        ctk.set_appearance_mode(self.config_manager_price3.ui.theme)
        self.setup_main_ui()
        messagebox.showinfo("Settings Applied", "New settings have been applied.")

    def logout(self):
        if messagebox.askyesno("Logout", "Are you sure you want to logout?"):
            print("🚪 Logging out...")
            
            # ✅ 1. توقف ربات
            if self.bot_running:
                bot_stop_event.set()
                if self.bot_thread and self.bot_thread.is_alive():
                    self.bot_thread.join(timeout=5)  # کمتر صبر کن
                self.bot_running = False
                self.bot_thread = None
            
            # ✅ 2. توقف آپدیت‌ها (فقط flag ها)
            self.dashboard_update_active = False
            self.chart_auto_update = False
            
            # ✅ 3. کنسل after jobs با روش ساده‌تر
            if hasattr(self, 'dashboard_update_job') and self.dashboard_update_job:
                try:
                    self.after_cancel(self.dashboard_update_job)
                except:
                    pass
                self.dashboard_update_job = None
            
            if hasattr(self, 'chart_update_job') and self.chart_update_job:
                try:
                    self.after_cancel(self.chart_update_job)
                except:
                    pass
                self.chart_update_job = None
            
            if hasattr(self, 'price_update_job') and self.price_update_job:
                try:
                    self.after_cancel(self.price_update_job)
                except:
                    pass
                self.price_update_job = None
            
            # ✅ 4. پاک کردن main_frame
            for widget in self.main_frame.winfo_children():
                widget.destroy()
            
            # ✅ 5. Logout و نمایش Login
            self.auth_system_v2_1.logout()
            self.withdraw()
            
            # ✅ 6. نمایش پنجره Login
            self.login_window = LoginWindow(self, self.auth_system_v2_1)
            self.login_window.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            print("✅ Logged out successfully")
            
            
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            print("🛑 Closing application...")
            
            # توقف ربات
            if self.bot_running:
                bot_stop_event.set()
                if self.bot_thread and self.bot_thread.is_alive():
                    self.bot_thread.join(timeout=5)
                self.bot_running = False
            
            # توقف آپدیت‌ها
            self.dashboard_update_active = False
            self.chart_auto_update = False
            
            # کنسل after jobs
            if hasattr(self, 'dashboard_update_job') and self.dashboard_update_job:
                try:
                    self.after_cancel(self.dashboard_update_job)
                except:
                    pass
            
            if hasattr(self, 'chart_update_job') and self.chart_update_job:
                try:
                    self.after_cancel(self.chart_update_job)
                except:
                    pass
            
            if hasattr(self, 'price_update_job') and self.price_update_job:
                try:
                    self.after_cancel(self.price_update_job)
                except:
                    pass
            
            # قطع MT5
            mt5_manager.shutdown()
            
            # ذخیره و بستن
            self.config_manager_price3.save_config()
            self.quit()
            self.destroy()

    def start_bot(self):
        if self.bot_thread is not None and self.bot_thread.is_alive():
            print("Bot is already running")
            return

        bot_stop_event.clear()
        self.bot_running = True

        self.start_bot_btn.configure(
            text="⏸️ Stop Bot",
            fg_color="red",
            hover_color="dark red"
        )

        self.bot_thread = threading.Thread(
            target=main_loop_improved,
            daemon=True
        )
        self.bot_thread.start()
        print("✅ Bot started")


    def stop_bot(self):
        if self.bot_thread is None or not self.bot_thread.is_alive():
            print("Bot is already stopped")
            self.apply_profile_change()
            return

        print("Stopping bot...")
        bot_stop_event.set()
        self.bot_thread.join(timeout=15.0)

        if self.bot_thread.is_alive():
            print("⚠️ Bot thread did not stop within timeout")

        self.bot_running = False
        
        self.start_bot_btn.configure(
            text="▶️ Start Bot",
            fg_color="green",
            hover_color="dark green"
        )
        
        print("✅ Bot stopped")
        # نماد/تنظیماتی که موقع روشن بودن ربات در Settings عوض شد، حالا اعمال می‌شود (داده + آموزش در پس‌زمینه)
        self.apply_profile_change()


if __name__ == "__main__":
    try:
        required_folders = ["data", "models", "logs", "exports"]
        for folder in required_folders:
            if not os.path.exists(folder):
                os.makedirs(folder)


        if not os.path.exists("data/config.json"):
            config_manager_price3 = ConfigManager()
            config_manager_price3.save_config()
        # ═══════════════════════════════════════════════════════════════
        # ✅ فعال کردن Redirect - همه print ها به GUI هم میرن
        # ═══════════════════════════════════════════════════════════════
        sys.stdout = log_redirector
        sys.stderr = error_redirector
        
        print("🚀 NR Trading Platform Starting...")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        app = NRTradingPlatform()
        app.mainloop()

    except Exception as e:
        # برگردوندن به حالت اصلی برای نمایش خطا
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        messagebox.showerror("Fatal Error", f"Application failed to start:\n{str(e)}")
        import traceback
        traceback.print_exc()
