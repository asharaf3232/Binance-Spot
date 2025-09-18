# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 Binance Mastermind Trader v31.1 (Phoenix Edition - Refined) 🚀 ---
# =======================================================================================
#
# هذا الإصدار ينهض من رماد المحاولات السابقة كبوت موحد وخارق.
# تم تعديله جراحياً لاستبدال جسد التداول الخاص بـ OKX بآخر متوافق بالكام
# مع Binance، مع الحفاظ على العقل التحليلي فائق الذكاء.
#
# --- Phoenix Edition Changelog v31.1 (Binance Integration Refined) ---
#   ✅ [التبديل الأساسي] تم استبدال جميع مكونات التداول الحقيقي من OKX إلى Binance.
#   ✅ [الاتصال] استخدام `ccxt` للاتصال بـ Binance Spot مع تحديد `defaultType='spot'` لضمان التنفيذ الصحيح.
#   ✅ [السرعة] استبدال `PublicWebSocketManager` بـ `BinancePublicWS` باستخدام `binance-connector`
#      للاشتراك في ستريم `!bookTicker@arr` للحصول على أسعار لحظية بزمن وصول منخفض (استخدام سعر المنتصف).
#   ✅ [السرعة] استبدال `PrivateWebSocketManager` بـ `BinancePrivateWS` باستخدام `binance-connector`
#      للاشتراك في `User Data Stream` وتفعيل الصفقات فوراً عند التعبئة الكاملة (FILLED executionReport).
#   ✅ [التوافق] تعديل وظائف إدارة الصفقات (`_close_trade`, `TradeGuardian`) لتكون متوافقة مع Binance API.
#   ✅ [التحسين] توحيد صيغة الرموز (e.g., 'BTC/USDT') في جميع أنحاء التطبيق باستخدام دالة مساعدة.
#   ✅ [التحسين] تحديث أسماء المتغيرات والتعليقات لتعكس الانتقال الكامل إلى Binance.
#
# --- ⚠️ متطلبات جديدة ⚠️ ---
#   pip install scipy feedparser nltk python-binance binance-connector
#
# =======================================================================================


# --- Core Libraries ---
import asyncio
import os
import logging
import json
import re
import time
import random
from datetime import datetime, timedelta, timezone, time as dt_time
from zoneinfo import ZoneInfo
import hmac
import base64
from collections import defaultdict, Counter
import copy

# --- Database & Networking ---
import aiosqlite
import websockets
import websockets.exceptions
import httpx
import feedparser

# --- Data Analysis & CCXT ---
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt

# --- [ترقية] مكتبات جديدة للعقل المطور و Binance ---
try:
    from binance.spot import Spot as BinanceSpotClient
    from binance.websocket.spot.websocket_client import SpotWebsocketClient as BinanceWebsocketClient
    BINANCE_CONNECTOR_AVAILABLE = True
except ImportError:
    BINANCE_CONNECTOR_AVAILABLE = False
    logging.critical("Binance Connector library not found. Please run 'pip install binance-connector'.")

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not found. News sentiment analysis will be disabled.")

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Library 'scipy' not found. RSI Divergence strategy will be disabled.")


# --- Telegram & Environment ---
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import BadRequest, TimedOut, Forbidden
from dotenv import load_dotenv

# =======================================================================================
# --- ⚙️ Core Configuration ⚙️ ---
# =======================================================================================
load_dotenv()

# --- NEW: Binance API Keys ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_AV_KEY_HERE')

TIMEFRAME = '15m'
SCAN_INTERVAL_SECONDS = 900
SUPERVISOR_INTERVAL_SECONDS = 120
TIME_SYNC_INTERVAL_SECONDS = 3600

APP_ROOT = '.'
DB_FILE = os.path.join(APP_ROOT, 'mastermind_trader_v31_binance.db')
SETTINGS_FILE = os.path.join(APP_ROOT, 'mastermind_trader_settings_v31.json')

EGYPT_TZ = ZoneInfo("Africa/Cairo")

class SafeFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'trade_id'): record.trade_id = 'N/A'
        return super().format(record)

log_formatter = SafeFormatter('%(asctime)s - %(levelname)s - [TradeID:%(trade_id)s] - %(message)s')
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)
root_logger = logging.getLogger(); root_logger.handlers = [log_handler]; root_logger.setLevel(logging.INFO)

class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'extra' not in kwargs: kwargs['extra'] = {}
        if 'trade_id' not in kwargs['extra']: kwargs['extra']['trade_id'] = 'N/A'
        return msg, kwargs
logger = ContextAdapter(logging.getLogger("Binance_Phoenix_Trader"), {})

# =======================================================================================
# --- 🔬 Global Bot State & Locks 🔬 ---
# =======================================================================================
class BotState:
    def __init__(self):
        self.settings = {}
        self.trading_enabled = True
        self.active_preset_name = "مخصص"
        self.last_signal_time = {}
        self.application = None
        self.exchange = None
        self.market_mood = {"mood": "UNKNOWN", "reason": "تحليل لم يتم بعد"}
        self.private_ws = None
        self.public_ws = None
        self.trade_guardian = None
        self.last_scan_info = {}
        self.all_markets = []
        self.last_markets_fetch = 0

bot_data = BotState()
scan_lock = asyncio.Lock()
trade_management_lock = asyncio.Lock()

# =======================================================================================
# --- 💡 Default Settings, Filters & UI Constants 💡 ---
# =======================================================================================
DEFAULT_SETTINGS = {
    "real_trade_size_usdt": 15.0,
    "max_concurrent_trades": 5,
    "top_n_symbols_by_volume": 300,
    "worker_threads": 10,
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 2.0,
    "trailing_sl_enabled": True,
    "trailing_sl_activation_percent": 1.5,
    "trailing_sl_callback_percent": 1.0,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "sniper_pro", "whale_radar", "rsi_divergence", "supertrend_pullback"],
    "market_mood_filter_enabled": True,
    "fear_and_greed_threshold": 30,
    "adx_filter_enabled": True,
    "adx_filter_level": 25,
    "btc_trend_filter_enabled": True,
    "news_filter_enabled": True,
    "asset_blacklist": ["USDC", "DAI", "TUSD", "FDUSD", "USDD", "PYUSD", "USDT", "BNB", "BTC", "ETH"], # Removed exchange tokens
    "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "min_rvol": 1.5},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.8},
    "trend_filters": {"ema_period": 200, "htf_period": 50, "enabled": True},
    "spread_filter": {"max_spread_percent": 0.5},
    "rsi_divergence": {"rsi_period": 14, "lookback_period": 35, "peak_trough_lookback": 5, "confirm_with_rsi_exit": True},
    "supertrend_pullback": {"atr_period": 10, "atr_multiplier": 3.0, "swing_high_lookback": 10},
}
STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي", "breakout_squeeze_pro": "اختراق انضغاطي",
    "support_rebound": "ارتداد الدعم", "sniper_pro": "القناص المحترف", "whale_radar": "رادار الحيتان",
    "rsi_divergence": "دايفرجنس RSI", "supertrend_pullback": "انعكاس سوبرترند"
}
PRESET_NAMES_AR = {"professional": "احترافي", "strict": "متشدد", "lenient": "متساهل", "very_lenient": "فائق التساهل"}
SETTINGS_PRESETS = {
    "professional": copy.deepcopy(DEFAULT_SETTINGS),
    "strict": {**copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 3, "risk_reward_ratio": 2.5, "fear_and_greed_threshold": 40, "adx_filter_level": 28, "liquidity_filters": {"min_quote_volume_24h_usd": 2000000, "min_rvol": 2.0}},
    "lenient": {**copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 8, "risk_reward_ratio": 1.8, "fear_and_greed_threshold": 25, "adx_filter_level": 20, "liquidity_filters": {"min_quote_volume_24h_usd": 500000, "min_rvol": 1.2}},
    "very_lenient": {**copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 12, "adx_filter_enabled": False, "market_mood_filter_enabled": False, "trend_filters": {"ema_period": 200, "htf_period": 50, "enabled": False}, "liquidity_filters": {"min_quote_volume_24h_usd": 250000, "min_rvol": 1.0}}
}

# =======================================================================================
# --- Helper, Settings & DB Management ---
# =======================================================================================
def format_binance_symbol_to_ccxt(binance_symbol: str) -> str:
    """Converts a Binance symbol like 'BTCUSDT' to ccxt format 'BTC/USDT'."""
    known_quotes = ['USDT', 'BUSD', 'USDC', 'TUSD', 'BTC', 'ETH', 'BNB']
    for quote in known_quotes:
        if binance_symbol.endswith(quote):
            base = binance_symbol[:-len(quote)]
            return f"{base}/{quote}"
    # Fallback for less common quotes, assuming 3 or 4 letter quote assets
    if len(binance_symbol) > 4:
         return f"{binance_symbol[:-4]}/{binance_symbol[-4:]}"
    elif len(binance_symbol) > 3:
        return f"{binance_symbol[:-3]}/{binance_symbol[-3:]}"
    return binance_symbol

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f: bot_data.settings = json.load(f)
        else: bot_data.settings = copy.deepcopy(DEFAULT_SETTINGS)
    except Exception: bot_data.settings = copy.deepcopy(DEFAULT_SETTINGS)
    default_copy = copy.deepcopy(DEFAULT_SETTINGS)
    for key, value in default_copy.items():
        if isinstance(value, dict):
            if key not in bot_data.settings or not isinstance(bot_data.settings[key], dict): bot_data.settings[key] = {}
            for sub_key, sub_value in value.items(): bot_data.settings[key].setdefault(sub_key, sub_value)
        else: bot_data.settings.setdefault(key, value)
    determine_active_preset(); save_settings()
    logger.info(f"Settings loaded. Active preset: {bot_data.active_preset_name}")

def determine_active_preset():
    current_settings_for_compare = copy.deepcopy(bot_data.settings)
    for name, preset_settings in SETTINGS_PRESETS.items():
        if current_settings_for_compare == preset_settings:
            bot_data.active_preset_name = PRESET_NAMES_AR.get(name, "مخصص"); return
    bot_data.active_preset_name = "مخصص"

def save_settings():
    with open(SETTINGS_FILE, 'w') as f: json.dump(bot_data.settings, f, indent=4)

async def safe_send_message(bot, text, **kwargs):
    try: await bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except Exception as e: logger.error(f"Telegram Send Error: {e}")
async def safe_edit_message(query, text, **kwargs):
    try: await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except BadRequest as e:
        if "Message is not modified" not in str(e): logger.warning(f"Edit Message Error: {e}")
    except Exception as e: logger.error(f"Edit Message Error: {e}")

async def init_database():
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT, entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, status TEXT, reason TEXT, order_id TEXT, highest_price REAL DEFAULT 0, trailing_sl_active BOOLEANE DEFAULT 0, close_price REAL, pnl_usdt REAL, signal_strength INTEGER DEFAULT 1)')
            cursor = await conn.execute("PRAGMA table_info(trades)")
            columns = [row[1] for row in await cursor.fetchall()]
            if 'signal_strength' not in columns:
                await conn.execute("ALTER TABLE trades ADD COLUMN signal_strength INTEGER DEFAULT 1")
            await conn.commit()
        logger.info("Phoenix database initialized successfully.")
    except Exception as e: logger.critical(f"Database initialization failed: {e}")

async def log_pending_trade_to_db(signal, buy_order):
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("INSERT INTO trades (timestamp, symbol, reason, order_id, status, entry_price, take_profit, stop_loss, signal_strength) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (datetime.now(EGYPT_TZ).isoformat(), signal['symbol'], signal['reason'], buy_order['id'], 'pending', signal['entry_price'], signal['take_profit'], signal['stop_loss'], signal.get('strength', 1)))
            await conn.commit()
            logger.info(f"Logged pending trade for {signal['symbol']} with order ID {buy_order['id']}.")
            return True
    except Exception as e: logger.error(f"DB Log Pending Error: {e}"); return False

# =======================================================================================
# --- 🧠 Mastermind Brain (Analysis & Mood) 🧠 ---
# =======================================================================================
# NEW FUNCTION - Added as part of the integration
async def translate_text_gemini(text_list):
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found in .env file. Skipping translation.")
        return text_list, False
    if not text_list:
        return [], True

    prompt = "Translate the following English headlines to Arabic. Return only the translated text, with each headline on a new line:\n\n" + "\n".join(text_list)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            translated_text = result['candidates'][0]['content']['parts'][0]['text']
            return translated_text.strip().split('\n'), True
    except Exception as e:
        logger.error(f"Gemini translation failed: {e}")
        return text_list, False


def get_alpha_vantage_economic_events():
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == 'YOUR_AV_KEY_HERE': return []
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    params = {'function': 'ECONOMIC_CALENDAR', 'horizon': '3month', 'apikey': ALPHA_VANTAGE_API_KEY}
    try:
        response = httpx.get('https://www.alphavantage.co/query', params=params, timeout=20)
        response.raise_for_status(); data_str = response.text
        if "premium" in data_str.lower(): return []
        lines = data_str.strip().split('\r\n')
        if len(lines) < 2: return []
        header = [h.strip() for h in lines[0].split(',')]
        events = [dict(zip(header, [v.strip() for v in line.split(',')])) for line in lines[1:]]
        high_impact_events = [e.get('event', 'Unknown Event') for e in events if e.get('releaseDate', '') == today_str and e.get('impact', '').lower() == 'high' and e.get('country', '') in ['USD', 'EUR']]
        if high_impact_events: logger.warning(f"High-impact events today: {high_impact_events}")
        return high_impact_events
    except httpx.RequestError as e: logger.error(f"Failed to fetch economic calendar: {e}"); return None

def get_latest_crypto_news(limit=15):
    urls = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]
    headlines = [entry.title for url in urls for entry in feedparser.parse(url).entries[:7]]
    return list(set(headlines))[:limit]

def analyze_sentiment_of_headlines(headlines):
    if not headlines or not NLTK_AVAILABLE: return "N/A", 0.0
    sia = SentimentIntensityAnalyzer()
    score = sum(sia.polarity_scores(h)['compound'] for h in headlines) / len(headlines)
    if score > 0.15: mood = "إيجابية"
    elif score < -0.15: mood = "سلبية"
    else: mood = "محايدة"
    return mood, score

async def get_fundamental_market_mood():
    s = bot_data.settings
    if not s.get('news_filter_enabled', True): return {"mood": "POSITIVE", "reason": "فلتر الأخبار معطل"}
    high_impact_events = await asyncio.to_thread(get_alpha_vantage_economic_events)
    if high_impact_events is None: return {"mood": "DANGEROUS", "reason": "فشل جلب البيانات الاقتصادية"}
    if high_impact_events: return {"mood": "DANGEROUS", "reason": f"أحداث هامة اليوم: {', '.join(high_impact_events)}"}
    latest_headlines = await asyncio.to_thread(get_latest_crypto_news)
    sentiment, score = analyze_sentiment_of_headlines(latest_headlines)
    logger.info(f"Market sentiment score: {score:.2f} ({sentiment})")
    if score > 0.25: return {"mood": "POSITIVE", "reason": f"مشاعر إيجابية (الدرجة: {score:.2f})"}
    elif score < -0.25: return {"mood": "NEGATIVE", "reason": f"مشاعر سلبية (الدرجة: {score:.2f})"}
    else: return {"mood": "NEUTRAL", "reason": f"مشاعر محايدة (الدرجة: {score:.2f})"}

def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

async def get_fear_and_greed_index():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            return int(r.json()['data'][0]['value'])
    except Exception: return None

async def get_market_mood():
    s = bot_data.settings
    if s.get('btc_trend_filter_enabled', True):
        try:
            htf_period = s['trend_filters']['htf_period']
            ohlcv = await bot_data.exchange.fetch_ohlcv('BTC/USDT', '4h', limit=htf_period + 5)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['sma'] = ta.sma(df['close'], length=htf_period)
            is_btc_bullish = df['close'].iloc[-1] > df['sma'].iloc[-1]
            btc_mood_text = "صاعد ✅" if is_btc_bullish else "هابط ❌"
            if not is_btc_bullish: return {"mood": "NEGATIVE", "reason": "اتجاه BTC هابط", "btc_mood": btc_mood_text}
        except Exception as e: return {"mood": "DANGEROUS", "reason": f"فشل جلب بيانات BTC: {e}", "btc_mood": "UNKNOWN"}
    else: btc_mood_text = "الفلتر معطل"
    if s.get('market_mood_filter_enabled', True):
        fng = await get_fear_and_greed_index()
        if fng is not None and fng < s['fear_and_greed_threshold']:
            return {"mood": "NEGATIVE", "reason": f"مشاعر خوف شديد (F&G: {fng})", "btc_mood": btc_mood_text}
    return {"mood": "POSITIVE", "reason": "وضع السوق مناسب", "btc_mood": btc_mood_text}

def analyze_momentum_breakout(df, params, rvol, adx_value):
    df.ta.vwap(append=True); df.ta.bbands(length=20, append=True); df.ta.macd(append=True); df.ta.rsi(append=True)
    last, prev = df.iloc[-2], df.iloc[-3]
    macd_col, macds_col, bbu_col, rsi_col = find_col(df.columns, "MACD_"), find_col(df.columns, "MACDs_"), find_col(df.columns, "BBU_"), find_col(df.columns, "RSI_")
    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and last[rsi_col] < 68):
        return {"reason": "momentum_breakout"}
    return None

def analyze_breakout_squeeze_pro(df, params, rvol, adx_value):
    df.ta.bbands(length=20, append=True); df.ta.kc(length=20, scalar=1.5, append=True); df.ta.obv(append=True)
    bbu_col, bbl_col, kcu_col, kcl_col = find_col(df.columns, "BBU_"), find_col(df.columns, "BBL_"), find_col(df.columns, "KCUe_"), find_col(df.columns, "KCLEe_")
    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]
    if is_in_squeeze and (last['close'] > last[bbu_col]) and (last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5) and (df['OBV'].iloc[-2] > df['OBV'].iloc[-3]):
        return {"reason": "breakout_squeeze_pro"}
    return None

async def analyze_support_rebound(df, params, rvol, adx_value, exchange, symbol):
    try:
        ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
        if len(ohlcv_1h) < 50: return None
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        current_price = df_1h['close'].iloc[-1]
        recent_lows = df_1h['low'].rolling(window=10, center=True).min()
        supports = recent_lows[recent_lows.notna()]
        closest_support = max([s for s in supports if s < current_price], default=None)
        if not closest_support or ((current_price - closest_support) / closest_support * 100 > 1.0): return None
        last_candle_15m = df.iloc[-2]
        if last_candle_15m['close'] > last_candle_15m['open'] and last_candle_15m['volume'] > df['volume'].rolling(window=20).mean().iloc[-2] * 1.5:
            return {"reason": "support_rebound"}
    except Exception: return None
    return None

def analyze_sniper_pro(df, params, rvol, adx_value):
    try:
        compression_candles = 24
        if len(df) < compression_candles + 2: return None
        compression_df = df.iloc[-compression_candles-1:-1]
        highest_high, lowest_low = compression_df['high'].max(), compression_df['low'].min()
        if lowest_low <= 0: return None
        volatility = (highest_high - lowest_low) / lowest_low * 100
        if volatility < 12.0:
            last_candle = df.iloc[-2]
            if last_candle['close'] > highest_high and last_candle['volume'] > compression_df['volume'].mean() * 2:
                return {"reason": "sniper_pro"}
    except Exception: return None
    return None

async def analyze_whale_radar(df, params, rvol, adx_value, exchange, symbol):
    try:
        ob = await exchange.fetch_order_book(symbol, limit=20)
        if not ob or not ob.get('bids'): return None
        if sum(float(price) * float(qty) for price, qty in ob['bids'][:10]) > 30000:
            return {"reason": "whale_radar"}
    except Exception: return None
    return None

def analyze_rsi_divergence(df, params, rvol, adx_value):
    if not SCIPY_AVAILABLE: return None
    df.ta.rsi(length=params.get('rsi_period', 14), append=True)
    rsi_col = find_col(df.columns, f"RSI_{params.get('rsi_period', 14)}")
    if not rsi_col or df[rsi_col].isnull().all(): return None
    subset = df.iloc[-params.get('lookback_period', 35):].copy()
    price_troughs_idx, _ = find_peaks(-subset['low'], distance=params.get('peak_trough_lookback', 5))
    rsi_troughs_idx, _ = find_peaks(-subset[rsi_col], distance=params.get('peak_trough_lookback', 5))
    if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
        p_low1_idx, p_low2_idx = price_troughs_idx[-2], price_troughs_idx[-1]
        r_low1_idx, r_low2_idx = rsi_troughs_idx[-2], rsi_troughs_idx[-1]
        is_divergence = (subset.iloc[p_low2_idx]['low'] < subset.iloc[p_low1_idx]['low'] and subset.iloc[r_low2_idx][rsi_col] > subset.iloc[r_low1_idx][rsi_col])
        if is_divergence:
            rsi_exits_oversold = (subset.iloc[r_low1_idx][rsi_col] < 35 and subset.iloc[-2][rsi_col] > 40)
            confirmation_price = subset.iloc[p_low2_idx:]['high'].max()
            price_confirmed = df.iloc[-2]['close'] > confirmation_price
            if (not params.get('confirm_with_rsi_exit', True) or rsi_exits_oversold) and price_confirmed:
                return {"reason": "rsi_divergence"}
    return None

def analyze_supertrend_pullback(df, params, rvol, adx_value):
    df.ta.supertrend(length=params.get('atr_period', 10), multiplier=params.get('atr_multiplier', 3.0), append=True)
    st_dir_col = find_col(df.columns, f"SUPERTd_{params.get('atr_period', 10)}_")
    if not st_dir_col: return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if prev[st_dir_col] == -1 and last[st_dir_col] == 1:
        recent_swing_high = df['high'].iloc[-params.get('swing_high_lookback', 10):-2].max()
        if last['close'] > recent_swing_high:
            return {"reason": "supertrend_pullback"}
    return None

SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout, "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "support_rebound": analyze_support_rebound, "sniper_pro": analyze_sniper_pro, "whale_radar": analyze_whale_radar,
    "rsi_divergence": analyze_rsi_divergence, "supertrend_pullback": analyze_supertrend_pullback
}

# =======================================================================================
# --- 🚀 Hybrid Core Protocol (Execution & Management) 🚀 ---
# =======================================================================================
async def activate_trade(order_id, symbol):
    bot = bot_data.application.bot; log_ctx = {'trade_id': 'N/A'}
    try:
        order_details = await bot_data.exchange.fetch_order(order_id, symbol)
        filled_price, gross_filled_quantity = order_details.get('average', 0.0), order_details.get('filled', 0.0)
        if gross_filled_quantity <= 0 or filled_price <= 0:
            logger.error(f"Order {order_id} invalid fill data. Price: {filled_price}, Qty: {gross_filled_quantity}."); return
        net_filled_quantity = gross_filled_quantity
        base_currency = symbol.split('/')[0]
        if 'fee' in order_details and order_details['fee'] and 'cost' in order_details['fee']:
            fee_cost, fee_currency = order_details['fee']['cost'], order_details['fee']['currency']
            if fee_currency == base_currency:
                net_filled_quantity -= fee_cost
                logger.info(f"Fee of {fee_cost} {fee_currency} deducted. Net quantity for {symbol} is {net_filled_quantity}.")
        if net_filled_quantity <= 0: logger.error(f"Net quantity for {order_id} is zero or less. Aborting."); return
        balance_after = await bot_data.exchange.fetch_balance()
        usdt_remaining = balance_after.get('USDT', {}).get('free', 0)
    except Exception as e:
        logger.error(f"Could not fetch data for trade activation: {e}", exc_info=True)
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("UPDATE trades SET status = 'failed', reason = 'Activation Fetch Error' WHERE order_id = ?", (order_id,)); await conn.commit()
        return
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        trade = await (await conn.execute("SELECT * FROM trades WHERE order_id = ? AND status = 'pending'", (order_id,))).fetchone()
        if not trade: logger.info(f"Activation ignored for {order_id}: Trade not pending."); return
        trade = dict(trade); log_ctx['trade_id'] = trade['id']
        logger.info(f"Activating trade #{trade['id']} for {symbol}...", extra=log_ctx)
        risk = filled_price - trade['stop_loss']
        new_take_profit = filled_price + (risk * bot_data.settings['risk_reward_ratio'])
        await conn.execute("UPDATE trades SET status = 'active', entry_price = ?, quantity = ?, take_profit = ? WHERE id = ?", (filled_price, net_filled_quantity, new_take_profit, trade['id']))
        active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]
        await conn.commit()

    await bot_data.public_ws.subscribe([symbol])
    trade_cost, tp_percent, sl_percent = filled_price * net_filled_quantity, (new_take_profit / filled_price - 1) * 100, (1 - trade['stop_loss'] / filled_price) * 100
    
    reasons_en = trade['reason'].split(' + ')
    reasons_ar = [STRATEGY_NAMES_AR.get(r.strip(), r.strip()) for r in reasons_en]
    reason_display_str = ' + '.join(reasons_ar)
    strength_stars = '⭐' * trade.get('signal_strength', 1)

    success_msg = (f"**✅ تم تأكيد الشراء | {symbol}**\n**الاستراتيجية:** {reason_display_str}\n**قوة الإشارة:** {strength_stars}\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n"
                   f"🔸 **الصفقة رقم:** `#{trade['id']}`\n"
                   f"🔸 **سعر التنفيذ:** `${filled_price:,.4f}`\n"
                   f"🔸 **الكمية (صافي):** `{net_filled_quantity:,.4f}` {symbol.split('/')[0]}\n"
                   f"🔸 **التكلفة:** `${trade_cost:,.2f}`\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n"
                   f"🎯 **الهدف (TP):** `${new_take_profit:,.4f}` `(ربح متوقع: {tp_percent:+.2f}%)`\n"
                   f"🛡️ **الوقف (SL):** `${trade['stop_loss']:,.4f}` `(خسارة مقبولة: {sl_percent:.2f}%)`\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n"
                   f"💰 **السيولة المتبقية (USDT):** `${usdt_remaining:,.2f}`\n"
                   f"🔄 **إجمالي الصفقات النشطة:** `{active_trades_count}`\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n"
                   f"الحارس الأمين يراقب الصفقة الآن.")
    await safe_send_message(bot, success_msg)

async def exponential_backoff_with_jitter(run_coro, *args, **kwargs):
    retries = 0; base_delay, max_delay = 2, 120
    while True:
        try: await run_coro(*args, **kwargs)
        except Exception as e:
            retries += 1; backoff_delay = min(max_delay, base_delay * (2 ** retries)); jitter = random.uniform(0, backoff_delay * 0.5); total_delay = backoff_delay + jitter
            logger.error(f"Coroutine {run_coro.__name__} failed: {e}. Retrying in {total_delay:.2f} seconds...")
            await asyncio.sleep(total_delay)

# --- NEW: Binance Private WebSocket Manager ---
class BinancePrivateWS:
    def __init__(self):
        self.ws_client = None
        self.listen_key = None
        self.rest_client = BinanceSpotClient(BINANCE_API_KEY, BINANCE_API_SECRET)

    async def _refresh_listen_key_loop(self):
        while True:
            await asyncio.sleep(30 * 60)  # Sleep for 30 minutes
            try:
                if self.listen_key:
                    self.rest_client.renew_listen_key(self.listen_key)
                    logger.info("Successfully renewed Binance listen key.")
            except Exception as e:
                logger.error(f"Failed to renew Binance listen key: {e}. A new key will be generated on next connection.")
                # The main loop will handle reconnection and getting a new key
                break 

    def _message_handler(self, message):
        try:
            if not isinstance(message, dict): return
            event_type = message.get('e')
            if event_type == 'executionReport':
                # Only trigger on full fill to avoid multiple activation calls for a single order.
                # The supervisor job will catch any trades that get stuck.
                if (message.get('X') == 'FILLED' and message.get('S') == 'BUY'):
                    
                    symbol = format_binance_symbol_to_ccxt(message['s'])
                    order_id = str(message['i'])
                    logger.info(f"Received FULL FILL execution for Buy Order {order_id} on {symbol}")
                    asyncio.create_task(activate_trade(order_id, symbol))
        except Exception as e:
            logger.error(f"Error in private WebSocket message handler: {e}")

    async def _run_loop(self):
        logger.info("Connecting to Binance User Data Stream...")
        try:
            self.listen_key = self.rest_client.new_listen_key()["listenKey"]
            logger.info("✅ [Fast Reporter] Binance listen key obtained.")
            
            # Start the renewal task
            asyncio.create_task(self._refresh_listen_key_loop())

            self.ws_client = BinanceWebsocketClient()
            self.ws_client.user_data(
                listen_key=self.listen_key,
                id=1,
                callback=self._message_handler,
            )
            logger.info("✅ [Fast Reporter] Connected to Binance User Data Stream.")
            # The binance-connector library handles its own event loop, so we just sleep to keep the task alive
            while True:
                await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Binance private WebSocket connection failed: {e}")
            if self.ws_client:
                self.ws_client.stop()
            raise e # Raise exception to trigger the exponential backoff

    async def run(self):
        await exponential_backoff_with_jitter(self._run_loop)

async def the_supervisor_job(context: ContextTypes.DEFAULT_TYPE):
    logger.info("🕵️ Supervisor: Auditing pending trades...")
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        two_mins_ago = (datetime.now(EGYPT_TZ) - timedelta(minutes=2)).isoformat()
        stuck_trades = await (await conn.execute("SELECT * FROM trades WHERE status = 'pending' AND timestamp <= ?", (two_mins_ago,))).fetchall()
        if not stuck_trades: logger.info("🕵️ Supervisor: Audit complete. No abandoned trades found."); return
        for trade_data in stuck_trades:
            trade = dict(trade_data); order_id, symbol = trade['order_id'], trade['symbol']
            logger.warning(f"🕵️ Supervisor: Found abandoned trade #{trade['id']}. Investigating.", extra={'trade_id': trade['id']})
            try:
                order_status = await bot_data.exchange.fetch_order(order_id, symbol)
                if order_status['status'] == 'closed' and order_status.get('filled', 0) > 0:
                    logger.info(f"🕵️ Supervisor: API confirms {order_id} was filled. Activating.", extra={'trade_id': trade['id']})
                    await activate_trade(order_id, symbol)
                elif order_status['status'] == 'canceled': await conn.execute("UPDATE trades SET status = 'failed' WHERE id = ?", (trade['id'],))
                else: await bot_data.exchange.cancel_order(order_id, symbol); await conn.execute("UPDATE trades SET status = 'failed' WHERE id = ?", (trade['id'],))
                await conn.commit()
            except Exception as e: logger.error(f"🕵️ Supervisor: Failed to rectify trade #{trade['id']}: {e}", extra={'trade_id': trade['id']})

class TradeGuardian:
    def __init__(self, application): self.application = application
    async def handle_ticker_update(self, ticker_data):
        async with trade_management_lock:
            # MODIFIED: Use new unified keys 'symbol' and 'price'
            symbol = ticker_data['symbol']; current_price = float(ticker_data['price'])
            try:
                async with aiosqlite.connect(DB_FILE) as conn:
                    conn.row_factory = aiosqlite.Row
                    trade = await (await conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = 'active'", (symbol,))).fetchone()
                    if not trade: return
                    trade = dict(trade); settings = bot_data.settings
                    if settings['trailing_sl_enabled']:
                        new_highest_price = max(trade.get('highest_price', 0), current_price)
                        if new_highest_price > trade.get('highest_price', 0): await conn.execute("UPDATE trades SET highest_price = ? WHERE id = ?", (new_highest_price, trade['id']))
                        if not trade['trailing_sl_active'] and current_price >= trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100):
                            trade['trailing_sl_active'] = True; trade['stop_loss'] = trade['entry_price']
                            await conn.execute("UPDATE trades SET trailing_sl_active = 1, stop_loss = ? WHERE id = ?", (trade['entry_price'], trade['id']))
                            await safe_send_message(self.application.bot, f"**🚀 تأمين الأرباح! | #{trade['id']} {symbol}**\nتم رفع وقف الخسارة إلى نقطة الدخول: `${trade['entry_price']}`")
                        if trade['trailing_sl_active']:
                            new_sl = new_highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                            if new_sl > trade['stop_loss']:
                                trade['stop_loss'] = new_sl; await conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_sl, trade['id']))
                    await conn.commit()
                if current_price >= trade['take_profit']: await self._close_trade(trade, "ناجحة (TP)", current_price)
                elif current_price <= trade['stop_loss']: await self._close_trade(trade, "فاشلة (SL)", current_price)
            except Exception as e: logger.error(f"Guardian Ticker Error for {symbol}: {e}", exc_info=True)

    async def _close_trade(self, trade, reason, close_price):
        symbol, trade_id = trade['symbol'], trade['id']
        bot, log_ctx = self.application.bot, {'trade_id': trade_id}
        logger.info(f"Guardian: Closing {symbol}. Reason: {reason}", extra=log_ctx)
        try:
            asset_to_sell = symbol.split('/')[0]
            balance = await bot_data.exchange.fetch_balance()
            available_quantity = balance.get(asset_to_sell, {}).get('free', 0.0)
            if available_quantity <= 0:
                logger.critical(f"Attempted to close #{trade_id} but no balance for {asset_to_sell}.", extra=log_ctx)
                async with aiosqlite.connect(DB_FILE) as conn:
                    await conn.execute("UPDATE trades SET status = 'closure_failed', reason = 'Zero balance' WHERE id = ?", (trade_id,)); await conn.commit()
                await safe_send_message(bot, f"🚨 **فشل إغلاق** 🚨\nلا يوجد رصيد متاح من `{asset_to_sell}` لإغلاق الصفقة `#{trade_id}`."); return
            
            # MODIFIED: Use CCXT's unified method without exchange-specific params. This is correct for Binance Spot.
            formatted_quantity = bot_data.exchange.amount_to_precision(symbol, available_quantity)
            await bot_data.exchange.create_market_sell_order(symbol, formatted_quantity)

            pnl = (close_price - trade['entry_price']) * trade['quantity']
            pnl_percent = (close_price / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
            emoji = "✅" if pnl > 0 else "🛑"
            async with aiosqlite.connect(DB_FILE) as conn:
                await conn.execute("UPDATE trades SET status = ?, close_price = ?, pnl_usdt = ? WHERE id = ?", (reason, close_price, pnl, trade['id'])); await conn.commit()
            
            # Unsubscribe from this symbol
            await bot_data.public_ws.unsubscribe([symbol])
            
            start_dt = datetime.fromisoformat(trade['timestamp']); end_dt = datetime.now(EGYPT_TZ)
            duration = end_dt - start_dt
            days, rem = divmod(duration.total_seconds(), 86400); hours, rem = divmod(rem, 3600); minutes, _ = divmod(rem, 60)
            duration_str = f"{int(days)}d {int(hours)}h {int(minutes)}m" if days > 0 else f"{int(hours)}h {int(minutes)}m"
            highest_price_val = max(trade.get('highest_price', 0), close_price)
            highest_pnl_percent = ((highest_price_val - trade['entry_price']) / trade['entry_price'] * 100) if trade['entry_price'] > 0 else 0

            msg = (f"**{emoji} تم إغلاق الصفقة | #{trade_id} {symbol}**\n"
                   f"**السبب:** {reason}\n"
                   f"**الربح/الخسارة:** `${pnl:,.2f}` ({pnl_percent:+.2f}%)\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n"
                   f"🔹 **مدة الصفقة:** {duration_str}\n"
                   f"🔹 **أعلى قمة وصلت لها:** `${highest_price_val:,.4f}` ({highest_pnl_percent:+.2f}%)")
            await safe_send_message(bot, msg)
        except Exception as e:
            logger.critical(f"Unexpected CRITICAL error closing trade: {e}", exc_info=True, extra=log_ctx)
            await safe_send_message(bot, f"🚨 **فشل حرج** 🚨\nخطأ غير متوقع عند إغلاق `#{trade_id}`.")

    async def sync_subscriptions(self):
        try:
            async with aiosqlite.connect(DB_FILE) as conn:
                active_symbols = [row[0] for row in await (await conn.execute("SELECT DISTINCT symbol FROM trades WHERE status = 'active'")).fetchall()]
            if active_symbols: 
                logger.info(f"Guardian: Syncing subs: {active_symbols}")
                await bot_data.public_ws.subscribe(active_symbols)
        except Exception as e: logger.error(f"Guardian Sync Error: {e}")

# --- NEW: Binance Public WebSocket Manager ---
class BinancePublicWS:
    def __init__(self, handler_coro):
        self.ws_client = None
        self.handler = handler_coro
        self.active_subscriptions = set()

    def _message_handler(self, message):
        try:
            # message is a list of bookTicker updates for all symbols from the !bookTicker@arr stream
            if isinstance(message, list):
                for tick in message:
                    symbol_raw = tick.get('s')
                    if not symbol_raw: continue
                    
                    # Convert to standard format and check if we are watching this symbol
                    symbol_standard = format_binance_symbol_to_ccxt(symbol_raw)
                    if symbol_standard in self.active_subscriptions:
                        best_bid = float(tick.get('b', 0))
                        best_ask = float(tick.get('a', 0))
                        
                        # Avoid division by zero and invalid prices
                        if best_bid > 0 and best_ask > 0:
                            # Use mid-price for a balanced trigger point, as requested.
                            mid_price = (best_bid + best_ask) / 2.0
                            
                            # MODIFIED: Use the new unified format
                            ticker_data = {'symbol': symbol_standard, 'price': mid_price}
                            asyncio.create_task(self.handler(ticker_data))
        except Exception as e:
            logger.error(f"Error in public WebSocket message handler: {e}")

    async def _run_loop(self):
        logger.info("Connecting to Binance Public Market Stream...")
        try:
            self.ws_client = BinanceWebsocketClient()
            # Subscribe to all book tickers (!bookTicker@arr). This is more efficient than individual streams.
            self.ws_client.book_ticker(id=1, callback=self._message_handler)
            logger.info("✅ [Guardian's Eyes] Connected to Binance Book Ticker stream.")
            # Keep the task alive
            while True:
                await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Binance public WebSocket connection failed: {e}")
            if self.ws_client:
                self.ws_client.stop()
            raise e # Trigger exponential backoff

    async def run(self):
        await exponential_backoff_with_jitter(self._run_loop)

    async def subscribe(self, symbols):
        new_symbols = set(symbols) - self.active_subscriptions
        if new_symbols:
            self.active_subscriptions.update(new_symbols)
            logger.info(f"👁️ [Guardian] Now watching: {list(new_symbols)}")

    async def unsubscribe(self, symbols):
        old_symbols = self.active_subscriptions.intersection(set(symbols))
        if old_symbols:
            self.active_subscriptions.difference_update(old_symbols)
            logger.info(f"👁️ [Guardian] Stopped watching: {list(old_symbols)}")

# =======================================================================================
# --- ⚡ Core Scanner & Trade Initiation Logic ⚡ ---
# =======================================================================================
async def get_binance_markets(): # Renamed from get_okx_markets
    settings = bot_data.settings
    if time.time() - bot_data.last_markets_fetch > 300:
        try:
            logger.info("Fetching and caching all Binance markets..."); 
            all_tickers = await bot_data.exchange.fetch_tickers()
            bot_data.all_markets = list(all_tickers.values()); 
            bot_data.last_markets_fetch = time.time()
        except Exception as e: logger.error(f"Failed to fetch all markets: {e}"); return []
    
    blacklist = settings.get('asset_blacklist', [])
    valid_markets = [
        t for t in bot_data.all_markets if 
        t.get('symbol') and 
        t['symbol'].endswith('/USDT') and 
        t['symbol'].split('/')[0] not in blacklist and 
        t.get('quoteVolume', 0) > settings['liquidity_filters']['min_quote_volume_24h_usd'] and
        t.get('info', {}).get('status') == 'TRADING' and # Binance specific check
        not any(k in t['symbol'] for k in ['UP', 'DOWN', 'BULL', 'BEAR']) # Binance leveraged tokens
    ]
    valid_markets.sort(key=lambda m: m.get('quoteVolume', 0), reverse=True)
    return valid_markets[:settings['top_n_symbols_by_volume']]


async def worker(queue, signals_list, errors_list):
    settings, exchange = bot_data.settings, bot_data.exchange
    while not queue.empty():
        market = await queue.get(); symbol = market['symbol']
        try:
            try:
                orderbook = await exchange.fetch_order_book(symbol, limit=1)
                best_bid, best_ask = orderbook['bids'][0][0], orderbook['asks'][0][0]
                if best_bid <= 0: continue
                spread_percent = ((best_ask - best_bid) / best_bid) * 100
                if spread_percent > settings.get('spread_filter', {}).get('max_spread_percent', 0.5): continue
            except Exception: continue
            
            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            if settings.get('trend_filters', {}).get('enabled', True):
                ema_period = settings.get('trend_filters', {}).get('ema_period', 200)
                if len(ohlcv) < ema_period + 1: continue
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp').sort_index()
                df.ta.ema(length=ema_period, append=True)
                ema_col_name = find_col(df.columns, f"EMA_{ema_period}")
                if not ema_col_name or pd.isna(df[ema_col_name].iloc[-2]): continue
                if df['close'].iloc[-2] < df[ema_col_name].iloc[-2]: continue
            else:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp').sort_index()
            
            vol_filters = settings.get('volatility_filters', {})
            atr_period, min_atr_percent = vol_filters.get('atr_period_for_filter', 14), vol_filters.get('min_atr_percent', 0.8)
            df.ta.atr(length=atr_period, append=True)
            atr_col_name = find_col(df.columns, f"ATRr_{atr_period}")
            if not atr_col_name or pd.isna(df[atr_col_name].iloc[-2]): continue
            last_close = df['close'].iloc[-2]
            atr_percent = (df[atr_col_name].iloc[-2] / last_close) * 100 if last_close > 0 else 0
            if atr_percent < min_atr_percent: continue

            df['volume_sma'] = ta.sma(df['volume'], length=20)
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] == 0: continue
            rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
            if rvol < settings['liquidity_filters']['min_rvol']: continue

            adx_value = 0
            if settings.get('adx_filter_enabled', False):
                df.ta.adx(append=True); adx_col = find_col(df.columns, "ADX_")
                adx_value = df[adx_col].iloc[-2] if adx_col and pd.notna(df[adx_col].iloc[-2]) else 0
                if adx_value < settings.get('adx_filter_level', 25): continue
            
            confirmed_reasons = []
            for name in settings['active_scanners']:
                if not (strategy_func := SCANNERS.get(name)): continue
                params = settings.get(name, {})
                func_args = {'df': df.copy(), 'params': params, 'rvol': rvol, 'adx_value': adx_value}
                if name in ['support_rebound', 'whale_radar']:
                    func_args.update({'exchange': exchange, 'symbol': symbol})
                result = await strategy_func(**func_args) if asyncio.iscoroutinefunction(strategy_func) else strategy_func(**{k: v for k, v in func_args.items() if k not in ['exchange', 'symbol']})
                if result: confirmed_reasons.append(result['reason'])

            if confirmed_reasons:
                reason_str, strength = ' + '.join(set(confirmed_reasons)), len(set(confirmed_reasons))
                entry_price = df.iloc[-2]['close']
                df.ta.atr(length=14, append=True)
                atr = df.iloc[-2].get(find_col(df.columns, "ATRr_14"), 0)
                risk = atr * settings['atr_sl_multiplier']
                stop_loss, take_profit = entry_price - risk, entry_price + (risk * settings['risk_reward_ratio'])
                signals_list.append({"symbol": symbol, "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "reason": reason_str, "strength": strength})
        except Exception as e: logger.debug(f"Worker error for {symbol}: {e}"); errors_list.append(symbol)
        finally: queue.task_done()

async def initiate_real_trade(signal):
    if not bot_data.trading_enabled:
        logger.warning(f"Trade for {signal['symbol']} blocked: Kill Switch active."); return False
    try:
        settings, exchange = bot_data.settings, bot_data.exchange; await exchange.load_markets()
        trade_size = settings['real_trade_size_usdt']
        balance = await exchange.fetch_balance(); usdt_balance = balance.get('USDT', {}).get('free', 0.0)
        if usdt_balance < trade_size:
            logger.error(f"Insufficient USDT for {signal['symbol']}. Have: {usdt_balance}, Need: {trade_size}")
            await safe_send_message(bot_data.application.bot, f"⚠️ **رصيد USDT غير كافٍ!**"); return False
        
        # For Binance MARKET buy orders, we specify the quoteOrderQty (how much USDT to spend)
        params = {'quoteOrderQty': trade_size}
        # The amount parameter for create_market_buy_order is not used by Binance when quoteOrderQty is set, so we pass None or 0.
        buy_order = await exchange.create_market_buy_order(signal['symbol'], None, params)
        
        if await log_pending_trade_to_db(signal, buy_order):
            await safe_send_message(bot_data.application.bot, f"🚀 تم إرسال أمر شراء لـ `{signal['symbol']}`."); return True
        else:
            await exchange.cancel_order(buy_order['id'], signal['symbol']); return False
    except ccxt.InsufficientFunds as e: logger.error(f"REAL TRADE FAILED {signal['symbol']}: {e}"); await safe_send_message(bot_data.application.bot, f"⚠️ **رصيد غير كافٍ!**"); return False
    except Exception as e: logger.error(f"REAL TRADE FAILED {signal['symbol']}: {e}", exc_info=True); return False

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        if not bot_data.trading_enabled: logger.info("Scan skipped: Kill Switch is active."); return
        scan_start_time = time.time(); logger.info("--- Starting new Phoenix Engine scan (Platform: Binance)... ---")
        settings, bot = bot_data.settings, context.bot
        
        if settings.get('news_filter_enabled', True):
            mood_result_fundamental = await get_fundamental_market_mood()
            if mood_result_fundamental['mood'] in ["NEGATIVE", "DANGEROUS"]:
                bot_data.market_mood = mood_result_fundamental
                logger.warning(f"SCAN SKIPPED: Fundamental mood is {mood_result_fundamental['mood']}. Reason: {mood_result_fundamental['reason']}")
                await safe_send_message(bot, f"🔬 *ملخص الفحص*\n- **الحالة:** تم التخطي\n- **السبب:** مزاج السوق سلبي/خطر.\n- **التفاصيل:** {mood_result_fundamental['reason']}"); return
        
        mood_result = await get_market_mood()
        bot_data.market_mood = mood_result
        if mood_result['mood'] in ["NEGATIVE", "DANGEROUS"]:
            logger.warning(f"SCAN SKIPPED: {mood_result['reason']}")
            await safe_send_message(bot, f"🔬 *ملخص الفحص*\n- **الحالة:** تم التخطي\n- **السبب:** {mood_result['reason']}"); return
        
        async with aiosqlite.connect(DB_FILE) as conn:
            active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active' OR status = 'pending'")).fetchone())[0]
        if active_trades_count >= settings['max_concurrent_trades']:
            logger.info(f"Scan skipped: Max trades ({active_trades_count}) reached."); return

        top_markets = await get_binance_markets();
        if not top_markets: logger.info("Scan complete: No markets passed filters."); return
        
        queue, signals_found, analysis_errors = asyncio.Queue(), [], []
        for market in top_markets: await queue.put(market)
        worker_tasks = [asyncio.create_task(worker(queue, signals_found, analysis_errors)) for _ in range(settings.get("worker_threads", 10))]
        await queue.join(); [task.cancel() for task in worker_tasks]
        
        trades_opened_count = 0
        signals_found.sort(key=lambda s: s.get('strength', 0), reverse=True)

        for signal in signals_found:
            if active_trades_count >= settings['max_concurrent_trades']: break
            if time.time() - bot_data.last_signal_time.get(signal['symbol'], 0) > (SCAN_INTERVAL_SECONDS * 0.9):
                bot_data.last_signal_time[signal['symbol']] = time.time()
                if await initiate_real_trade(signal):
                    active_trades_count += 1; trades_opened_count += 1
                await asyncio.sleep(2)
        
        scan_duration = time.time() - scan_start_time
        bot_data.last_scan_info = {"start_time": datetime.fromtimestamp(scan_start_time, EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), "duration_seconds": int(scan_duration), "checked_symbols": len(top_markets), "analysis_errors": len(analysis_errors)}
        summary_message = (f"🔬 *ملخص الفحص الأخير*\n\n"
                           f"- **الحالة:** اكتمل بنجاح\n"
                           f"- **المدة:** {int(scan_duration)} ثانية | **العملات:** {len(top_markets)}\n"
                           f"----------------------------------\n"
                           f"- **إجمالي الإشارات:** {len(signals_found)}\n"
                           f"- **✅ صفقات جديدة:** {trades_opened_count}\n"
                           f"- **⚠️ أخطاء:** {len(analysis_errors)}")
        await safe_send_message(bot, summary_message)

async def check_time_sync(context: ContextTypes.DEFAULT_TYPE):
    try:
        server_time = await bot_data.exchange.fetch_time(); local_time = int(time.time() * 1000); diff = abs(server_time - local_time)
        if diff > 2000: await safe_send_message(context.bot, f"⚠️ **تحذير مزامنة الوقت** ⚠️\nفارق `{diff}` ميلي ثانية.")
        else: logger.info(f"Time sync OK. Diff: {diff}ms.")
    except Exception as e: logger.error(f"Time sync check failed: {e}")

# =======================================================================================
# --- 🤖 Telegram UI & Bot Startup 🤖 ---
# =======================================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Dashboard 🖥️"], ["الإعدادات ⚙️"]]
    await update.message.reply_text("أهلاً بك في **Binance Mastermind Trader v31.1 (Phoenix Edition)**", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True), parse_mode=ParseMode.MARKDOWN)

async def manual_scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not bot_data.trading_enabled: await (update.message or update.callback_query.message).reply_text("🔬 الفحص محظور. مفتاح الإيقاف مفعل."); return
    await (update.message or update.callback_query.message).reply_text("🔬 أمر فحص يدوي... قد يستغرق بعض الوقت.")
    context.job_queue.run_once(perform_scan, 1)

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ks_status_emoji = "🚨" if not bot_data.trading_enabled else "✅"
    ks_status_text = "مفتاح الإيقاف (مفعل)" if not bot_data.trading_enabled else "الحالة (طبيعية)"
    keyboard = [
        [InlineKeyboardButton("💼 نظرة عامة على المحفظة", callback_data="db_portfolio"), InlineKeyboardButton("📈 الصفقات النشطة", callback_data="db_trades")],
        [InlineKeyboardButton("📜 سجل الصفقات المغلقة", callback_data="db_history"), InlineKeyboardButton("📊 الإحصائيات والأداء", callback_data="db_stats")],
        [InlineKeyboardButton("🌡️ تحليل مزاج السوق", callback_data="db_mood"), InlineKeyboardButton("🔬 فحص فوري", callback_data="db_manual_scan")],
        [InlineKeyboardButton("🗓️ التقرير اليومي", callback_data="db_daily_report")],
        [InlineKeyboardButton(f"{ks_status_emoji} {ks_status_text}", callback_data="kill_switch_toggle"), InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="db_diagnostics")]
    ]
    message_text = "🖥️ *لوحة التحكم الرئيسية*\n\nاختر التقرير أو البيانات التي تريد عرضها:"
    if not bot_data.trading_enabled: message_text += "\n\n**تحذير: تم تفعيل مفتاح الإيقاف.**"
    target_message = update.message or update.callback_query.message
    if update.callback_query: await safe_edit_message(update.callback_query, message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else: await target_message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

async def send_daily_report(context: ContextTypes.DEFAULT_TYPE):
    today_str = datetime.now(EGYPT_TZ).strftime('%Y-%m-%d')
    logger.info(f"Generating daily report for {today_str}...")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            closed_today = await (await conn.execute("SELECT * FROM trades WHERE status LIKE '%(%' AND date(timestamp) = ?", (today_str,))).fetchall()
        if not closed_today:
            report_message = f"**🗓️ التقرير اليومي | {today_str}**\n\nلم يتم إغلاق أي صفقات اليوم."
        else:
            wins = [t for t in closed_today if t['status'].startswith('ناجحة')]
            total_pnl = sum(t['pnl_usdt'] for t in closed_today if t['pnl_usdt'] is not None)
            win_rate = (len(wins) / len(closed_today) * 100) if closed_today else 0
            best_trade = max(closed_today, key=lambda t: t.get('pnl_usdt', -float('inf')), default=None)
            strategy_counter = Counter(r for t in closed_today for r in t['reason'].split(' + '))
            most_active_strategy_en = strategy_counter.most_common(1)[0][0] if strategy_counter else "N/A"
            most_active_strategy_ar = STRATEGY_NAMES_AR.get(most_active_strategy_en, most_active_strategy_en)
            parts = [f"**🗓️ التقرير اليومي المفصل | {today_str}**\n"]
            parts.append("💰 **الأداء المالي:**")
            parts.append(f"  - الربح/الخسارة الصافي: `${total_pnl:+.2f}`")
            parts.append("\n📊 **إحصائيات الصفقات:**")
            parts.append(f"  - الإجمالي: {len(closed_today)} | ✅ الرابحة: {len(wins)} | ❌ الخاسرة: {len(closed_today) - len(wins)}")
            parts.append(f"  - معدل النجاح: {win_rate:.1f}%")
            if best_trade and best_trade['pnl_usdt'] > 0:
                parts.append(f"\n🏆 **أفضل صفقة:** `{best_trade['symbol']}` | `${best_trade['pnl_usdt']:+.2f}`")
            parts.append(f"\n💡 **الاستراتيجية الأنشط:** *{most_active_strategy_ar}*")
            report_message = "\n".join(parts)
        await safe_send_message(context.bot, report_message)
    except Exception as e: logger.error(f"Failed to generate daily report: {e}", exc_info=True)

async def daily_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await (update.message or update.callback_query.message).reply_text("⏳ جاري إرسال التقرير اليومي...")
    await send_daily_report(context)

async def toggle_kill_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; bot_data.trading_enabled = not bot_data.trading_enabled
    if bot_data.trading_enabled: await query.answer("✅ تم استئناف التداول الطبيعي."); await safe_send_message(context.bot, "✅ **تم استئناف التداول الطبيعي.**")
    else: await query.answer("🚨 تم تفعيل مفتاح الإيقاف!", show_alert=True); await safe_send_message(context.bot, "🚨 **تحذير: تم تفعيل مفتاح الإيقاف!**")
    await show_dashboard_command(update, context)

async def show_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row; trades = await (await conn.execute("SELECT id, symbol, status FROM trades WHERE status = 'active' OR status = 'pending' ORDER BY id DESC")).fetchall()
    if not trades: text = "لا توجد صفقات حالية."; keyboard = [[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]; await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard)); return
    text = "📈 *الصفقات النشطة*\nاختر صفقة لعرض تفاصيلها:\n"; keyboard = []
    for trade in trades: status_emoji = "✅" if trade['status'] == 'active' else "⏳"; button_text = f"#{trade['id']} {status_emoji} | {trade['symbol']}"; keyboard.append([InlineKeyboardButton(button_text, callback_data=f"check_{trade['id']}")])
    keyboard.append([InlineKeyboardButton("🔄 تحديث", callback_data="db_trades")]); keyboard.append([InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]); await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))

async def check_trade_details(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    trade_id = int(query.data.split('_')[1])
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        trade = await cursor.fetchone()
    if not trade:
        await query.answer("لم يتم العثور على الصفقة."); return
    trade = dict(trade)
    if trade['status'] == 'pending':
        message = f"**⏳ حالة الصفقة #{trade_id}**\n- **العملة:** `{trade['symbol']}`\n- **الحالة:** في انتظار تأكيد التنفيذ..."
    else:
        try:
            ticker = await bot_data.exchange.fetch_ticker(trade['symbol'])
            current_price = ticker['last']
            pnl = (current_price - trade['entry_price']) * trade['quantity']
            pnl_percent = (current_price / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
            pnl_text = f"💰 **الربح/الخسارة الحالية:** `${pnl:+.2f}` ({pnl_percent:+.2f}%)"
            current_price_text = f"- **السعر الحالي:** `${current_price}`"
        except Exception:
            pnl_text = "💰 تعذر جلب الربح/الخسارة الحالية."
            current_price_text = "- **السعر الحالي:** `تعذر الجلب`"

        message = (
            f"**✅ حالة الصفقة #{trade_id}**\n\n"
            f"- **العملة:** `{trade['symbol']}`\n"
            f"- **سعر الدخول:** `${trade['entry_price']}`\n"
            f"{current_price_text}\n"
            f"- **الكمية:** `{trade['quantity']}`\n"
            f"----------------------------------\n"
            f"- **الهدف (TP):** `${trade['take_profit']}`\n"
            f"- **الوقف (SL):** `${trade['stop_loss']}`\n"
            f"----------------------------------\n"
            f"{pnl_text}"
        )
    await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للصفقات", callback_data="db_trades")]]))

async def show_mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("جاري تحليل مزاج السوق...")
    fng_task = asyncio.create_task(get_fear_and_greed_index())
    headlines_task = asyncio.create_task(asyncio.to_thread(get_latest_crypto_news))
    mood_task = asyncio.create_task(get_market_mood())
    markets_task = asyncio.create_task(get_binance_markets())
    fng_index = await fng_task
    original_headlines = await headlines_task
    mood = await mood_task
    all_markets = await markets_task
    translated_headlines, translation_success = await translate_text_gemini(original_headlines)
    news_sentiment, _ = analyze_sentiment_of_headlines(original_headlines)
    top_gainers, top_losers = [], []
    if all_markets:
        sorted_by_change = sorted([m for m in all_markets if m.get('percentage') is not None], key=lambda m: m['percentage'], reverse=True)
        top_gainers = sorted_by_change[:3]
        top_losers = sorted_by_change[-3:]
    verdict = "الحالة العامة للسوق تتطلب الحذر."
    if mood['mood'] == 'POSITIVE': verdict = "المؤشرات الفنية إيجابية، مما قد يدعم فرص الشراء."
    if fng_index and fng_index > 65: verdict = "المؤشرات الفنية إيجابية ولكن مع وجود طمع في السوق، يرجى الحذر من التقلبات."
    elif fng_index and fng_index < 30: verdict = "يسود الخوف على السوق، قد تكون هناك فرص للمدى الطويل ولكن المخاطرة عالية حالياً."
    gainers_str = "\n".join([f"  `{g['symbol']}` `({g.get('percentage', 0):+.2f}%)`" for g in top_gainers]) or "  لا توجد بيانات."
    losers_str = "\n".join([f"  `{l['symbol']}` `({l.get('percentage', 0):+.2f}%)`" for l in reversed(top_losers)]) or "  لا توجد بيانات."
    news_header = "📰 آخر الأخبار (مترجمة آلياً):" if translation_success else "📰 آخر الأخبار (الترجمة غير متاحة):"
    news_str = "\n".join([f"  - _{h}_" for h in translated_headlines]) or "  لا توجد أخبار."
    message = (
        f"**🌡️ تحليل مزاج السوق الشامل**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"**⚫️ الخلاصة:** *{verdict}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"**📊 المؤشرات الرئيسية:**\n"
        f"  - **اتجاه BTC العام:** {mood.get('btc_mood', 'N/A')}\n"
        f"  - **الخوف والطمع:** {fng_index or 'N/A'}\n"
        f"  - **مشاعر الأخبار:** {news_sentiment}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"**🚀 أبرز الرابحين:**\n{gainers_str}\n\n"
        f"**📉 أبرز الخاسرين:**\n{losers_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{news_header}\n{news_str}\n"
    )
    keyboard = [[InlineKeyboardButton("🔄 تحديث", callback_data="db_mood")], [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
    await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_strategy_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        cursor = await conn.execute("SELECT reason, status FROM trades WHERE status LIKE 'ناجحة%' OR status LIKE 'فاشلة%'")
        trades = await cursor.fetchall()
    if not trades:
        await safe_edit_message(update.callback_query, "لا توجد صفقات مغلقة لتحليلها.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))
        return
    stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
    for reason, status in trades:
        if not reason: continue
        reasons = reason.split(' + ')
        for r in reasons:
            if status.startswith('ناجحة'): stats[r]['wins'] += 1
            else: stats[r]['losses'] += 1
    report = ["**📜 تقرير أداء الاستراتيجيات**"]
    for r, s in sorted(stats.items(), key=lambda item: item[1]['wins'] + item[1]['losses'], reverse=True):
        total = s['wins'] + s['losses']
        wr = (s['wins'] / total * 100) if total > 0 else 0
        report.append(f"\n--- *{STRATEGY_NAMES_AR.get(r, r)}* ---\n  - الصفقات: {total} ({s['wins']}✅ / {s['losses']}❌)\n  - النجاح: {wr:.2f}%")
    await safe_edit_message(update.callback_query, "\n".join(report), reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))

async def show_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT pnl_usdt, status FROM trades WHERE status LIKE 'ناجحة%' OR status LIKE 'فاشلة%'")
        trades_data = await cursor.fetchall()
    if not trades_data:
        await safe_edit_message(update.callback_query, "لا توجد صفقات مغلقة لعرض الإحصائيات.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))
        return
    total_trades = len(trades_data)
    total_pnl = sum(t['pnl_usdt'] for t in trades_data if t['pnl_usdt'] is not None)
    wins_data = [t['pnl_usdt'] for t in trades_data if t['status'].startswith('ناجحة') and t['pnl_usdt'] is not None]
    losses_data = [t['pnl_usdt'] for t in trades_data if t['status'].startswith('فاشلة') and t['pnl_usdt'] is not None]
    win_count = len(wins_data)
    loss_count = len(losses_data)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    avg_win = sum(wins_data) / win_count if win_count > 0 else 0
    avg_loss = sum(losses_data) / loss_count if loss_count > 0 else 0
    profit_factor = sum(wins_data) / abs(sum(losses_data)) if sum(losses_data) != 0 else float('inf')
    message = (
        f"**📊 إحصائيات الأداء التفصيلية**\n\n"
        f"**💰 إجمالي الربح/الخسارة الصافي:** `...`"
    )
    await safe_edit_message(update.callback_query, message, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))

async def show_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer("جاري جلب بيانات المحفظة...")
    try:
        balance = await bot_data.exchange.fetch_balance() # Default is 'spot' for binance
        owned_assets = {asset: data['total'] for asset, data in balance.items() if isinstance(data, dict) and data.get('total', 0) > 0}
        usdt_balance = balance.get('USDT', {}); total_usdt_equity = usdt_balance.get('total', 0); free_usdt = usdt_balance.get('free', 0)
        assets_to_fetch = [f"{asset}/USDT" for asset in owned_assets if asset != 'USDT']
        tickers = {}
        if assets_to_fetch:
            try: tickers = await bot_data.exchange.fetch_tickers(assets_to_fetch)
            except Exception as e: logger.warning(f"Could not fetch all tickers for portfolio: {e}")
        asset_details = []; total_assets_value_usdt = 0
        for asset, total in owned_assets.items():
            if asset == 'USDT': continue
            symbol = f"{asset}/USDT"; value_usdt = 0
            if symbol in tickers and tickers[symbol] is not None: value_usdt = tickers[symbol].get('last', 0) * total
            total_assets_value_usdt += value_usdt
            if value_usdt >= 1.0: asset_details.append(f"  - `{asset}`: `{total:,.6f}` `(≈ ${value_usdt:,.2f})`")
        total_equity = total_usdt_equity + total_assets_value_usdt
        async with aiosqlite.connect(DB_FILE) as conn:
            cursor_pnl = await conn.execute("SELECT SUM(pnl_usdt) FROM trades WHERE status LIKE 'ناجحة%' OR status LIKE 'فاشلة%'")
            total_realized_pnl = (await cursor_pnl.fetchone())[0] or 0.0
            cursor_trades = await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")
            active_trades_count = (await cursor_trades.fetchone())[0]
        assets_str = "\n".join(asset_details) if asset_details else "  لا توجد أصول أخرى بقيمة تزيد عن 1 دولار."
        message = (
            f"**💼 نظرة عامة على المحفظة**\n"
            f"🗓️ {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"**💰 إجمالي قيمة المحفظة:** `≈ ${total_equity:,.2f}`\n"
            f"  - **السيولة المتاحة (USDT):** `${free_usdt:,.2f}`\n"
            f"  - **قيمة الأصول الأخرى:** `≈ ${total_assets_value_usdt:,.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"**📊 تفاصيل الأصول (أكثر من 1$):**\n"
            f"{assets_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"**📈 أداء التداول:**\n"
            f"  - **الربح/الخسارة المحقق:** `${total_realized_pnl:,.2f}`\n"
            f"  - **عدد الصفقات النشطة:** {active_trades_count}\n"
        )
        keyboard = [[InlineKeyboardButton("🔄 تحديث", callback_data="db_portfolio")], [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
        await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.error(f"Portfolio fetch error: {e}", exc_info=True)
        await safe_edit_message(query, f"حدث خطأ أثناء جلب رصيد المحفظة: {e}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة", callback_data="back_to_dashboard")]]))

async def show_trade_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT symbol, pnl_usdt, status FROM trades WHERE status LIKE 'ناجحة%' OR status LIKE 'فاشلة%' ORDER BY id DESC LIMIT 10")
        closed_trades = await cursor.fetchall()
    if not closed_trades:
        text = "لا يوجد سجل للصفقات المغلقة."
        keyboard = [[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
        await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))
        return
    history_list = ["📜 *آخر 10 صفقات مغلقة*"]
    for trade in closed_trades:
        emoji = "✅" if trade['status'].startswith('ناجحة') else "🛑"
        pnl = trade['pnl_usdt'] or 0.0
        history_list.append(f"{emoji} `{trade['symbol']}` | الربح/الخسارة: `${pnl:,.2f}`")
    text = "\n".join(history_list)
    keyboard = [[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
    await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_diagnostics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; s = bot_data.settings
    scan_info = bot_data.last_scan_info
    determine_active_preset()
    nltk_status = "متاحة ✅" if NLTK_AVAILABLE else "غير متاحة ❌"
    scan_time = scan_info.get("start_time", "لم يتم بعد")
    scan_duration = f'{scan_info.get("duration_seconds", "N/A")} ثانية'
    scan_checked = scan_info.get("checked_symbols", "N/A")
    scan_errors = scan_info.get("analysis_errors", "N/A")
    scanners_list = "\n".join([f"  - {STRATEGY_NAMES_AR.get(key, key)}" for key in s['active_scanners']])
    scan_job = context.job_queue.get_jobs_by_name("perform_scan")
    next_scan_time = scan_job[0].next_t.astimezone(EGYPT_TZ).strftime('%H:%M:%S') if scan_job and scan_job[0].next_t else "N/A"
    db_size = f"{os.path.getsize(DB_FILE) / 1024:.2f} KB" if os.path.exists(DB_FILE) else "N/A"
    async with aiosqlite.connect(DB_FILE) as conn:
        total_trades = (await (await conn.execute("SELECT COUNT(*) FROM trades")).fetchone())[0]
        active_trades = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]
    report = (
        f"🕵️‍♂️ *تقرير التشخيص الشامل*\n\n"
        f"تم إنشاؤه في: {datetime.now(EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"----------------------------------\n"
        f"⚙️ **حالة النظام والبيئة**\n"
        f"- NLTK (تحليل الأخبار): {nltk_status}\n\n"
        f"🔬 **أداء آخر فحص**\n"
        f"- وقت البدء: {scan_time}\n"
        f"- المدة: {scan_duration}\n"
        f"- العملات المفحوصة: {scan_checked}\n"
        f"- فشل في التحليل: {scan_errors} عملات\n\n"
        f"🔧 **الإعدادات النشطة**\n"
        f"- **النمط الحالي: {bot_data.active_preset_name}**\n"
        f"- الماسحات المفعلة:\n{scanners_list}\n"
        f"----------------------------------\n"
        f"🔩 **حالة العمليات الداخلية**\n"
        f"- فحص العملات: يعمل, التالي في: {next_scan_time}\n"
        f"- الاتصال بـ Binance: متصل ✅\n"
        f"- قاعدة البيانات:\n"
        f"  - الاتصال: ناجح ✅\n"
        f"  - حجم الملف: {db_size}\n"
        f"  - إجمالي الصفقات: {total_trades} ({active_trades} نشطة)\n"
        f"----------------------------------"
    )
    await safe_edit_message(query, report, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 تحديث", callback_data="db_diagnostics")], [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))

async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🎛️ تعديل المعايير المتقدمة", callback_data="settings_params")],
        [InlineKeyboardButton("🔭 تفعيل/تعطيل الماسحات", callback_data="settings_scanners")],
        [InlineKeyboardButton("🗂️ أنماط جاهزة", callback_data="settings_presets")],
        [InlineKeyboardButton("🚫 القائمة السوداء", callback_data="settings_blacklist"), InlineKeyboardButton("🗑️ إدارة البيانات", callback_data="settings_data")]
    ]
    message_text = "⚙️ *الإعدادات الرئيسية*\n\nاختر فئة الإعدادات التي تريد تعديلها."
    target_message = update.message or update.callback_query.message
    if update.callback_query: await safe_edit_message(update.callback_query, message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else: await target_message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = bot_data.settings
    def bool_format(key, text):
        val = s.get(key, False)
        emoji = "✅" if val else "❌"
        return f"{text}: {emoji} مفعل"
    def get_nested_value(d, keys):
        current_level = d
        for key in keys:
            if isinstance(current_level, dict) and key in current_level: current_level = current_level[key]
            else: return None
        return current_level
    keyboard = [
        [InlineKeyboardButton("--- إعدادات عامة ---", callback_data="noop")],
        [InlineKeyboardButton(f"عدد العملات للفحص: {s['top_n_symbols_by_volume']}", callback_data="param_set_top_n_symbols_by_volume"),
         InlineKeyboardButton(f"أقصى عدد للصفقات: {s['max_concurrent_trades']}", callback_data="param_set_max_concurrent_trades")],
        [InlineKeyboardButton(f"عمال الفحص المتزامنين: {s['worker_threads']}", callback_data="param_set_worker_threads")],
        [InlineKeyboardButton("--- إعدادات المخاطر ---", callback_data="noop")],
        [InlineKeyboardButton(f"حجم الصفقة ($): {s['real_trade_size_usdt']}", callback_data="param_set_real_trade_size_usdt"),
         InlineKeyboardButton(f"مضاعف وقف الخسارة (ATR): {s['atr_sl_multiplier']}", callback_data="param_set_atr_sl_multiplier")],
        [InlineKeyboardButton(f"نسبة المخاطرة/العائد: {s['risk_reward_ratio']}", callback_data="param_set_risk_reward_ratio")],
        [InlineKeyboardButton(bool_format('trailing_sl_enabled', 'تفعيل الوقف المتحرك'), callback_data="param_toggle_trailing_sl_enabled")],
        [InlineKeyboardButton(f"تفعيل الوقف المتحرك (%): {s['trailing_sl_activation_percent']}", callback_data="param_set_trailing_sl_activation_percent"),
         InlineKeyboardButton(f"مسافة الوقف المتحرك (%): {s['trailing_sl_callback_percent']}", callback_data="param_set_trailing_sl_callback_percent")],
        [InlineKeyboardButton("--- الفلاتر والاتجاه ---", callback_data="noop")],
        [InlineKeyboardButton(bool_format('btc_trend_filter_enabled', 'فلتر الاتجاه العام (BTC)'), callback_data="param_toggle_btc_trend_filter_enabled")],
        [InlineKeyboardButton(f"فترة EMA للاتجاه: {get_nested_value(s, ['trend_filters', 'ema_period'])}", callback_data="param_set_trend_filters_ema_period")],
        [InlineKeyboardButton(f"أقصى سبريد مسموح (%): {get_nested_value(s, ['spread_filter', 'max_spread_percent'])}", callback_data="param_set_spread_filter_max_spread_percent")],
        [InlineKeyboardButton(f"أدنى ATR مسموح (%): {get_nested_value(s, ['volatility_filters', 'min_atr_percent'])}", callback_data="param_set_volatility_filters_min_atr_percent")],
        [InlineKeyboardButton(bool_format('market_mood_filter_enabled', 'فلتر الخوف والطمع'), callback_data="param_toggle_market_mood_filter_enabled"),
         InlineKeyboardButton(f"حد مؤشر الخوف: {s['fear_and_greed_threshold']}", callback_data="param_set_fear_and_greed_threshold")],
        [InlineKeyboardButton(bool_format('adx_filter_enabled', 'فلتر ADX'), callback_data="param_toggle_adx_filter_enabled"),
         InlineKeyboardButton(f"مستوى فلتر ADX: {s['adx_filter_level']}", callback_data="param_set_adx_filter_level")],
        [InlineKeyboardButton(bool_format('news_filter_enabled', 'فلتر الأخبار والبيانات'), callback_data="param_toggle_news_filter_enabled")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, "🎛️ *المعايير المتقدمة*\n\nاضغط على أي معيار لتغيير قيمته:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    active_scanners = bot_data.settings['active_scanners']
    for key, name in STRATEGY_NAMES_AR.items():
        status_emoji = "✅" if key in active_scanners else "❌"
        keyboard.append([InlineKeyboardButton(f"{status_emoji} {name}", callback_data=f"scanner_toggle_{key}")])
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")])
    await safe_edit_message(update.callback_query, "اختر الماسحات لتفعيلها أو تعطيلها:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🚦 احترافي", callback_data="preset_set_professional")],
        [InlineKeyboardButton("🎯 متشدد", callback_data="preset_set_strict")],
        [InlineKeyboardButton("🌙 متساهل", callback_data="preset_set_lenient")],
        [InlineKeyboardButton("⚠️ فائق التساهل", callback_data="preset_set_very_lenient")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, "اختر نمط إعدادات جاهز:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_blacklist_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    blacklist = bot_data.settings.get('asset_blacklist', [])
    blacklist_str = ", ".join(f"`{item}`" for item in blacklist) if blacklist else "لا توجد عملات في القائمة."
    text = f"🚫 *القائمة السوداء*\n\nالعملات التالية لن يتم فحصها أو التداول عليها:\n\n{blacklist_str}"
    keyboard = [
        [InlineKeyboardButton("➕ إضافة عملة", callback_data="blacklist_add"), InlineKeyboardButton("➖ إزالة عملة", callback_data="blacklist_remove")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_data_management_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("‼️ مسح كل الصفقات ‼️", callback_data="data_clear_confirm")], [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]]
    await safe_edit_message(update.callback_query, "🗑️ *إدارة البيانات*\n\n**تحذير:** هذا الإجراء سيحذف سجل جميع الصفقات بشكل نهائي.", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_clear_data_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("نعم، متأكد. احذف كل شيء.", callback_data="data_clear_execute")], [InlineKeyboardButton("لا، تراجع.", callback_data="settings_data")]]
    await safe_edit_message(update.callback_query, "🛑 **تأكيد نهائي** 🛑\n\nهل أنت متأكد أنك تريد حذف جميع بيانات الصفقات؟", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_clear_data_execute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await safe_edit_message(query, "جاري حذف البيانات...", reply_markup=None)
    try:
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            logger.info("Database file has been deleted by user.")
        await init_database()
        await safe_edit_message(query, "✅ تم حذف جميع بيانات الصفقات بنجاح.")
    except Exception as e:
        logger.error(f"Failed to clear data: {e}")
        await safe_edit_message(query, f"❌ حدث خطأ أثناء حذف البيانات: {e}")
    await asyncio.sleep(2)
    await show_settings_menu(update, context)

async def handle_scanner_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    scanner_key = query.data.replace("scanner_toggle_", "")
    active_scanners = bot_data.settings['active_scanners']
    if scanner_key not in STRATEGY_NAMES_AR:
        logger.error(f"Invalid scanner key: '{scanner_key}'"); await query.answer("خطأ: مفتاح الماسح غير صالح.", show_alert=True); return
    if scanner_key in active_scanners:
        if len(active_scanners) > 1: active_scanners.remove(scanner_key)
        else: await query.answer("يجب تفعيل ماسح واحد على الأقل.", show_alert=True); return
    else: active_scanners.append(scanner_key)
    save_settings(); determine_active_preset()
    await query.answer(f"{STRATEGY_NAMES_AR[scanner_key]} {'تم تفعيله' if scanner_key in active_scanners else 'تم تعطيله'}")
    await show_scanners_menu(update, context)

async def handle_preset_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; preset_key = query.data.split('_')[-1]
    if preset_settings := SETTINGS_PRESETS.get(preset_key):
        current_scanners = bot_data.settings.get('active_scanners', [])
        bot_data.settings = copy.deepcopy(preset_settings)
        bot_data.settings['active_scanners'] = current_scanners
        determine_active_preset(); save_settings()
        await query.answer(f"نمط '{PRESET_NAMES_AR.get(preset_key)}' تم تطبيقه. الحالة الحالية: '{bot_data.active_preset_name}'")
        await show_settings_menu(update, context)
    else: await query.answer("لم يتم العثور على النمط.")

async def handle_parameter_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; param_key = query.data.replace("param_set_", "")
    context.user_data['setting_to_change'] = param_key
    if '_' in param_key: await query.message.reply_text(f"أرسل القيمة الرقمية الجديدة لـ `{param_key}`:\n\n*ملاحظة: هذا إعداد متقدم (متشعب)، سيتم تحديثه مباشرة.*", parse_mode=ParseMode.MARKDOWN)
    else: await query.message.reply_text(f"أرسل القيمة الرقمية الجديدة لـ `{param_key}`:", parse_mode=ParseMode.MARKDOWN)

async def handle_toggle_parameter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; param_key = query.data.replace("param_toggle_", "")
    bot_data.settings[param_key] = not bot_data.settings.get(param_key, False)
    save_settings(); determine_active_preset()
    await show_parameters_menu(update, context)

async def handle_blacklist_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; action = query.data.replace("blacklist_", "")
    context.user_data['blacklist_action'] = action
    await query.message.reply_text(f"أرسل رمز العملة التي تريد **{ 'إضافتها' if action == 'add' else 'إزالتها'}** (مثال: `BTC` أو `DOGE`)")

async def handle_setting_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    if 'blacklist_action' in context.user_data:
        action = context.user_data.pop('blacklist_action'); blacklist = bot_data.settings.get('asset_blacklist', [])
        symbol = user_input.upper().replace("/USDT", "")
        if action == 'add':
            if symbol not in blacklist: blacklist.append(symbol); await update.message.reply_text(f"✅ تم إضافة `{symbol}` إلى القائمة السوداء.")
            else: await update.message.reply_text(f"⚠️ العملة `{symbol}` موجودة بالفعل.")
        elif action == 'remove':
            if symbol in blacklist: blacklist.remove(symbol); await update.message.reply_text(f"✅ تم إزالة `{symbol}` من القائمة السوداء.")
            else: await update.message.reply_text(f"⚠️ العملة `{symbol}` غير موجودة في القائمة.")
        bot_data.settings['asset_blacklist'] = blacklist; save_settings(); determine_active_preset()
        await show_blacklist_menu(Update(update.update_id, callback_query=type('Query', (), {'message': update.message, 'data': 'settings_blacklist', 'edit_message_text': (lambda *args, **kwargs: None), 'answer': (lambda *args, **kwargs: None)})()), context); return

    if not (setting_key := context.user_data.get('setting_to_change')): return

    try:
        if setting_key in bot_data.settings:
            original_value = bot_data.settings[setting_key]
            if isinstance(original_value, int):
                new_value = int(user_input)
            else:
                new_value = float(user_input)
            bot_data.settings[setting_key] = new_value
        else:
            keys = setting_key.split('_'); current_dict = bot_data.settings
            for key in keys[:-1]:
                current_dict = current_dict[key]
            last_key = keys[-1]
            original_value = current_dict[last_key]
            if isinstance(original_value, int):
                new_value = int(user_input)
            else:
                new_value = float(user_input)
            current_dict[last_key] = new_value

        save_settings(); determine_active_preset()
        await update.message.reply_text(f"✅ تم تحديث `{setting_key}` إلى `{new_value}`.")
    except (ValueError, KeyError):
        await update.message.reply_text("❌ قيمة غير صالحة. الرجاء إرسال رقم.")
    finally:
        if 'setting_to_change' in context.user_data:
            del context.user_data['setting_to_change']
        await show_parameters_menu(Update(update.update_id, callback_query=type('Query', (), {'message': update.message, 'data': 'settings_params', 'edit_message_text': (lambda *args, **kwargs: None), 'answer': (lambda *args, **kwargs: None)})()), context)
        
async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'setting_to_change' in context.user_data or 'blacklist_action' in context.user_data:
        await handle_setting_value(update, context); return
    text = update.message.text
    if text == "Dashboard 🖥️": await show_dashboard_command(update, context)
    elif text == "الإعدادات ⚙️": await show_settings_menu(update, context)

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data
    route_map = {
        "db_stats": show_stats_command, "db_trades": show_trades_command, "db_history": show_trade_history_command,
        "db_mood": show_mood_command, "db_diagnostics": show_diagnostics_command, "back_to_dashboard": show_dashboard_command,
        "db_portfolio": show_portfolio_command, "db_manual_scan": lambda u,c: manual_scan_command(u, c),
        "kill_switch_toggle": toggle_kill_switch, "db_daily_report": daily_report_command,
        "settings_main": show_settings_menu, "settings_params": show_parameters_menu, "settings_scanners": show_scanners_menu,
        "settings_presets": show_presets_menu, "settings_blacklist": show_blacklist_menu, "settings_data": show_data_management_menu,
        "blacklist_add": handle_blacklist_action, "blacklist_remove": handle_blacklist_action,
        "data_clear_confirm": handle_clear_data_confirmation, "data_clear_execute": handle_clear_data_execute,
        "noop": (lambda u,c: None)
    }
    try:
        if data in route_map: await route_map[data](update, context)
        elif data.startswith("check_"): await check_trade_details(update, context)
        elif data.startswith("scanner_toggle_"): await handle_scanner_toggle(update, context)
        elif data.startswith("preset_set_"): await handle_preset_set(update, context)
        elif data.startswith("param_set_"): await handle_parameter_selection(update, context)
        elif data.startswith("param_toggle_"): await handle_toggle_parameter(update, context)
    except Exception as e: logger.error(f"Error in button callback handler for data '{data}': {e}", exc_info=True)

async def post_init(application: Application):
    bot_data.application = application
    if not all([BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN]):
        logger.critical("FATAL: Missing critical API keys for Binance or Telegram."); return
    if not BINANCE_CONNECTOR_AVAILABLE:
        logger.critical("FATAL: Binance Connector is not installed."); return
    if NLTK_AVAILABLE:
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError: logger.info("Downloading NLTK data..."); nltk.download('vader_lexicon', quiet=True)
    try:
        # MODIFIED: Added options to default to 'spot' trading
        config = {
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        }
        bot_data.exchange = ccxt.binance(config)
        await bot_data.exchange.load_markets()
        await bot_data.exchange.fetch_balance()
        logger.info("✅ Successfully connected to Binance.")
    except Exception as e:
        logger.critical(f"🔥 FATAL: Could not connect to Binance: {e}"); return
    await check_time_sync(ContextTypes.DEFAULT_TYPE(application=application))

    # --- MODIFIED: Initialize Binance WebSocket Managers ---
    bot_data.trade_guardian = TradeGuardian(application)
    bot_data.public_ws = BinancePublicWS(bot_data.trade_guardian.handle_ticker_update)
    bot_data.private_ws = BinancePrivateWS()
    asyncio.create_task(bot_data.public_ws.run())
    asyncio.create_task(bot_data.private_ws.run())
    logger.info("Waiting 5s for WebSocket connections to establish..."); await asyncio.sleep(5)
    await bot_data.trade_guardian.sync_subscriptions()
    
    jq = application.job_queue
    jq.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name="perform_scan")
    jq.run_repeating(the_supervisor_job, interval=SUPERVISOR_INTERVAL_SECONDS, first=30, name="the_supervisor_job")
    jq.run_repeating(check_time_sync, interval=TIME_SYNC_INTERVAL_SECONDS, first=TIME_SYNC_INTERVAL_SECONDS, name="time_sync_job")
    jq.run_daily(send_daily_report, time=dt_time(hour=23, minute=55, tzinfo=EGYPT_TZ), name='daily_report')
    logger.info(f"Jobs scheduled. Daily report at 23:55.")
    try: await application.bot.send_message(TELEGRAM_CHAT_ID, "*🚀 Binance Mastermind Trader v31.1 (Phoenix Edition) بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)
    except Forbidden: logger.critical(f"FATAL: Bot not authorized for chat ID {TELEGRAM_CHAT_ID}."); return
    logger.info("--- Phoenix Engine (Binance Edition) is now fully operational ---")

async def post_shutdown(application: Application):
    if bot_data.public_ws and bot_data.public_ws.ws_client:
        bot_data.public_ws.ws_client.stop()
    if bot_data.private_ws and bot_data.private_ws.ws_client:
        bot_data.private_ws.ws_client.stop()
    if bot_data.exchange: await bot_data.exchange.close()
    logger.info("Bot has shut down.")

def main():
    logger.info("--- Starting Binance Mastermind Trader v31.1 (Phoenix Edition) ---")
    load_settings(); asyncio.run(init_database())
    app_builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    app_builder.post_init(post_init).post_shutdown(post_shutdown)
    application = app_builder.build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("scan", manual_scan_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    application.run_polling()

if __name__ == '__main__':
    main()
