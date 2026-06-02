import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import ta
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# إعدادات الصفحة
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartStocks | تحليل الأسهم",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CSS — تصميم واضح ومريح للجميع
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');

* { font-family: 'Tajawal', sans-serif !important; }

.block-container { padding: 2rem 3rem !important; max-width: 1200px; margin: auto; }

/* Header */
.header-wrap {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    border-bottom: 2px solid #e8f4fd;
    margin-bottom: 2rem;
}
.header-title {
    font-size: 3rem;
    font-weight: 900;
    color: #0a2463;
    margin: 0;
    letter-spacing: -1px;
}
.header-sub {
    font-size: 1.2rem;
    color: #5a7fa8;
    margin: 0.5rem 0 0;
}

/* بطاقة البحث */
.search-card {
    background: #f0f6ff;
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    border: 2px solid #d0e4f7;
}

/* بطاقة السهم الحالي */
.stock-header {
    background: #0a2463;
    color: white;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
}
.stock-name { font-size: 1.1rem; opacity: 0.8; margin: 0; }
.stock-price { font-size: 3rem; font-weight: 900; margin: 0.3rem 0; }
.stock-change-pos { color: #4ade80; font-size: 1.3rem; font-weight: 700; }
.stock-change-neg { color: #f87171; font-size: 1.3rem; font-weight: 700; }

/* بطاقات المعلومات */
.info-card {
    background: white;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    border: 1.5px solid #e2eaf5;
    text-align: center;
}
.info-label { font-size: 0.9rem; color: #6b7fa8; margin: 0; }
.info-value { font-size: 1.5rem; font-weight: 700; color: #0a2463; margin: 0.2rem 0 0; }

/* بطاقات التوصية */
.rec-card {
    border-radius: 20px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 0.5rem;
    border: 2px solid transparent;
}
.rec-buy    { background: #f0fdf4; border-color: #86efac; }
.rec-sell   { background: #fff1f1; border-color: #fca5a5; }
.rec-hold   { background: #fffbeb; border-color: #fcd34d; }

.rec-horizon { font-size: 1rem; font-weight: 700; color: #374151; margin: 0; }
.rec-label   { font-size: 2.2rem; font-weight: 900; margin: 0.5rem 0; }
.rec-buy  .rec-label  { color: #16a34a; }
.rec-sell .rec-label  { color: #dc2626; }
.rec-hold .rec-label  { color: #d97706; }

.rec-conf { font-size: 1rem; color: #6b7280; margin: 0; }

/* شريط الثقة */
.conf-bar-wrap { margin: 0.8rem 0 0; }
.conf-bar-bg {
    background: #e5e7eb;
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
}
.conf-bar-fill-buy  { background: #16a34a; height: 10px; border-radius: 99px; transition: width 1s; }
.conf-bar-fill-sell { background: #dc2626; height: 10px; border-radius: 99px; }
.conf-bar-fill-hold { background: #d97706; height: 10px; border-radius: 99px; }

/* احتماليات */
.probs-row {
    display: flex;
    justify-content: space-between;
    margin-top: 1rem;
    gap: 6px;
}
.prob-item {
    flex: 1;
    text-align: center;
    background: white;
    border-radius: 10px;
    padding: 0.5rem;
    border: 1px solid #e5e7eb;
}
.prob-item-label { font-size: 0.75rem; color: #9ca3af; margin: 0; }
.prob-item-val   { font-size: 1rem; font-weight: 700; margin: 0; }

/* التنبيه القانوني */
.legal-box {
    background: #fffbeb;
    border: 2px solid #fbbf24;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.legal-icon { font-size: 1.8rem; flex-shrink: 0; }
.legal-text { font-size: 1rem; color: #92400e; line-height: 1.6; margin: 0; }

/* المؤشرات الفنية */
.indicator-card {
    background: white;
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    border: 1.5px solid #e2eaf5;
    margin-bottom: 1rem;
}
.ind-title { font-size: 1rem; font-weight: 700; color: #374151; margin: 0 0 0.8rem; }
.ind-row { display: flex; justify-content: space-between; align-items: center; }
.ind-val  { font-size: 1.5rem; font-weight: 900; color: #0a2463; }
.ind-desc { font-size: 0.9rem; padding: 0.3rem 0.8rem; border-radius: 99px; font-weight: 500; }
.ind-pos  { background: #f0fdf4; color: #16a34a; }
.ind-neg  { background: #fff1f1; color: #dc2626; }
.ind-neu  { background: #f3f4f6; color: #6b7280; }

/* أسهم سريعة */
.quick-btn {
    display: inline-block;
    background: white;
    border: 2px solid #d0e4f7;
    border-radius: 12px;
    padding: 0.5rem 1.2rem;
    font-size: 1rem;
    font-weight: 700;
    color: #0a2463;
    cursor: pointer;
    margin: 0.3rem;
    transition: all 0.2s;
}
.quick-btn:hover { background: #0a2463; color: white; }

/* قسم */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0a2463;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #d0e4f7;
}

/* الخطأ */
.error-box {
    background: #fff1f1;
    border: 2px solid #fca5a5;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    color: #dc2626;
    font-size: 1.1rem;
    font-weight: 500;
}

/* Responsive */
@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .header-title { font-size: 2rem; }
    .stock-price { font-size: 2.2rem; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# تحميل النماذج
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    try:
        models['week']    = joblib.load('smartstocks_v5_models/week_model.pkl')
        models['month']   = joblib.load('smartstocks_v5_models/month_model.pkl')
        models['3months'] = joblib.load('smartstocks_v5_models/3months_model.pkl')
        models['features'] = joblib.load('smartstocks_v5_models/features.pkl')
        # label encoder اختياري
        enc_path = 'smartstocks_v5_models/label_encoder.pkl'
        if os.path.exists(enc_path):
            models['encoder'] = joblib.load(enc_path)
        return models, True
    except Exception as e:
        return {}, False

models, models_loaded = load_models()


# ─────────────────────────────────────────────
# بناء الـ Features (نفس V3)
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_market_data():
    market = {}
    for ticker in ['SPY', 'QQQ', '^VIX']:
        try:
            d = yf.download(ticker, period='3y', auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d.reset_index(inplace=True)
            market[ticker] = d[['Date','Close']].copy()
        except:
            pass
    return market


def build_features_for_prediction(data, market):
    """V5 Features — نفس الـ features اللي تدرب عليها المودل"""
    df = data.copy().sort_values('Date').reset_index(drop=True)

    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    # A) العوائد
    df['return_1d']  = close.pct_change(1)
    df['return_5d']  = close.pct_change(5)
    df['return_21d'] = close.pct_change(21)
    df['return_63d'] = close.pct_change(63)

    # B) المتوسطات
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    df['price_to_sma20']  = close / sma20 - 1
    df['price_to_sma50']  = close / sma50 - 1
    df['price_to_sma200'] = close / sma200 - 1
    df['sma20_slope']     = sma20.pct_change(5)
    df['golden_cross']    = (sma50 > sma200).astype(int)

    # C) المؤشرات الفنية
    df['RSI']       = ta.momentum.RSIIndicator(close, window=14).rsi()
    df['RSI_slope'] = df['RSI'].diff(5)

    macd = ta.trend.MACD(close)
    df['MACD_hist'] = macd.macd_diff()

    df['ATR_pct'] = ta.volatility.AverageTrueRange(
        high, low, close, window=14).average_true_range() / close

    df['bb_pct'] = ta.volatility.BollingerBands(
        close, window=20).bollinger_pband()

    df['volume_ratio'] = vol / vol.rolling(20).mean()

    # D) موقع السعر في السنة
    df['price_pct_52w'] = close.rolling(252).rank(pct=True)

    # E) بيانات السوق
    for ticker, mdf in market.items():
        safe = ticker.replace('^', '')
        tmp  = mdf.rename(columns={'Close': f'{safe}_c'})
        df   = df.merge(tmp[['Date', f'{safe}_c']], on='Date', how='left')
        df[f'{safe}_ret5']  = df[f'{safe}_c'].pct_change(5)
        df[f'{safe}_ret21'] = df[f'{safe}_c'].pct_change(21)
        df.drop(columns=[f'{safe}_c'], inplace=True)

    if '^VIX' in market:
        vix = market['^VIX'].rename(columns={'Close': 'VIX_c'})
        df  = df.merge(vix[['Date', 'VIX_c']], on='Date', how='left')
        df['VIX_level']   = df['VIX_c']
        df['VIX_above20'] = (df['VIX_c'] > 20).astype(int)
        df.drop(columns=['VIX_c'], inplace=True)

    if 'SPY' in market:
        spy = market['SPY'].copy()
        spy['sma50']  = spy['Close'].rolling(50).mean()
        spy['sma200'] = spy['Close'].rolling(200).mean()
        spy['market_regime'] = (spy['Close'] > spy['sma50']).astype(int)
        spy['bull_market']   = (spy['sma50'] > spy['sma200']).astype(int)
        df = df.merge(spy[['Date', 'market_regime', 'bull_market']],
                      on='Date', how='left')

    if 'SPY_ret5' in df.columns:
        df['rel_strength'] = df['return_5d'] - df['SPY_ret5']

    drop_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.dropna(inplace=True)

    return df


# ─────────────────────────────────────────────
# التنبؤ
# ─────────────────────────────────────────────
def predict(ticker_symbol, models, market, conf_threshold=0.50):
    data = yf.download(ticker_symbol, period='3y', auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    data['Ticker'] = ticker_symbol

    if len(data) < 100:
        return None

    df_feat = build_features_for_prediction(data, market)
    if len(df_feat) == 0:
        return None

    feature_cols = models['features']
    available = [f for f in feature_cols if f in df_feat.columns]
    latest = df_feat[available].iloc[[-1]]

    # إذا فيه أعمدة ناقصة نملأها بصفر
    for col in feature_cols:
        if col not in latest.columns:
            latest[col] = 0
    latest = latest[feature_cols]

    horizons = {
        'week':    ('أسبوع', '5 أيام'),
        'month':   ('شهر',   '21 يوم'),
        '3months': ('3 أشهر','63 يوم'),
    }

    label_map = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
    if 'encoder' in models:
        enc = models['encoder']
        label_map = dict(zip(enc.transform(enc.classes_), enc.classes_))

    results = {}
    for key, (ar_name, days_str) in horizons.items():
        if key not in models:
            continue
        model  = models[key]
        proba  = model.predict_proba(latest)[0]
        pred   = model.predict(latest)[0]
        label  = label_map.get(int(pred), 'HOLD')
        conf   = float(proba.max())

        if conf < conf_threshold:
            label = 'HOLD'

        classes = model.classes_
        prob_dict = {}
        for i, c in enumerate(classes):
            prob_dict[label_map.get(int(c), str(c))] = round(float(proba[i]) * 100, 1)

        results[key] = {
            'horizon_ar': ar_name,
            'days':       days_str,
            'label':      label,
            'confidence': round(conf * 100, 1),
            'probs':      prob_dict,
        }

    return results


# ─────────────────────────────────────────────
# جلب بيانات السهم للعرض
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_display_data(symbol):
    try:
        # نحمّل سنة للرسم البياني وحساب أعلى/أقل سعر
        hist1y = yf.download(symbol, period='1y', auto_adjust=True, progress=False)
        if isinstance(hist1y.columns, pd.MultiIndex):
            hist1y.columns = hist1y.columns.get_level_values(0)
        hist1y = hist1y.dropna()

        if hist1y.empty:
            return None, None

        # نحسب أعلى/أقل من البيانات التاريخية
        week52_high = float(hist1y['High'].max())
        week52_low  = float(hist1y['Low'].min())

        # نجيب info بطريقة سريعة
        info = {
            'fiftyTwoWeekHigh': week52_high,
            'fiftyTwoWeekLow':  week52_low,
        }
        try:
            ticker_obj = yf.Ticker(symbol)
            fast_info  = ticker_obj.fast_info
            info['marketCap']  = getattr(fast_info, 'market_cap', 0) or 0
            info['sector']     = getattr(ticker_obj, 'info', {}).get('sector', '—')
            info['industry']   = getattr(ticker_obj, 'info', {}).get('industry', '—')
            info['country']    = getattr(ticker_obj, 'info', {}).get('country', '—')
            info['longName']   = getattr(fast_info, 'quote_type', symbol)
            try:
                full_info = ticker_obj.info
                info.update(full_info)
            except:
                pass
        except:
            pass

        # نرجع آخر 6 أشهر للرسم البياني
        hist6mo = hist1y.tail(126)
        return hist6mo, info

    except:
        return None, None


# ─────────────────────────────────────────────
# الرسم البياني
# ─────────────────────────────────────────────
def make_chart(hist):
    recent = hist.tail(90)

    sma20 = recent['Close'].rolling(20).mean()
    sma50 = recent['Close'].rolling(50).mean()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.05
    )

    fig.add_trace(go.Candlestick(
        x=recent.index,
        open=recent['Open'],
        high=recent['High'],
        low=recent['Low'],
        close=recent['Close'],
        name='السعر',
        increasing_line_color='#16a34a',
        decreasing_line_color='#dc2626',
        increasing_fillcolor='#16a34a',
        decreasing_fillcolor='#dc2626',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=recent.index, y=sma20,
        name='متوسط 20 يوم',
        line=dict(color='#f59e0b', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=recent.index, y=sma50,
        name='متوسط 50 يوم',
        line=dict(color='#3b82f6', width=2)
    ), row=1, col=1)

    colors = ['#16a34a' if c >= o else '#dc2626'
              for c, o in zip(recent['Close'], recent['Open'])]

    fig.add_trace(go.Bar(
        x=recent.index,
        y=recent['Volume'],
        name='حجم التداول',
        marker_color=colors,
        opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        height=480,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            font=dict(family='Tajawal', size=13)
        ),
        font=dict(family='Tajawal'),
        yaxis=dict(title='السعر ($)', gridcolor='#f3f4f6'),
        yaxis2=dict(title='الحجم', gridcolor='#f3f4f6'),
        xaxis2=dict(gridcolor='#f3f4f6'),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f3f4f6')

    return fig


# ─────────────────────────────────────────────
# دوال مساعدة للعرض
# ─────────────────────────────────────────────
def conf_to_text(conf):
    """تحويل نسبة الثقة لكلمة واضحة"""
    if conf >= 75:
        return 'عالية جداً'
    elif conf >= 60:
        return 'عالية'
    elif conf >= 50:
        return 'متوسطة'
    else:
        return 'منخفضة'

def top_tendency(probs):
    """إيجاد أعلى احتمال وتحويله لجملة واضحة"""
    max_label = max(probs, key=probs.get)
    ar = {'BUY': 'الشراء', 'SELL': 'البيع', 'HOLD': 'الانتظار'}
    return f"النموذج يميل نحو {ar.get(max_label, '')}"

def rec_card_html(data):
    label   = data['label']
    conf    = data['confidence']
    horizon = data['horizon_ar']
    probs   = data['probs']

    css_class  = {'BUY': 'rec-buy', 'SELL': 'rec-sell', 'HOLD': 'rec-hold'}.get(label, 'rec-hold')
    bar_class  = {'BUY': 'conf-bar-fill-buy', 'SELL': 'conf-bar-fill-sell', 'HOLD': 'conf-bar-fill-hold'}.get(label, 'conf-bar-fill-hold')
    conf_word  = conf_to_text(conf)
    tendency   = top_tendency(probs)

    conf_color = {'عالية جداً': '#16a34a', 'عالية': '#16a34a',
                  'متوسطة': '#d97706', 'منخفضة': '#dc2626'}.get(conf_word, '#6b7280')

    # أيقونات SVG بدل إيموجي
    if label == 'BUY':
        icon_svg = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.5"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>'
        ar_label = 'شراء'
        label_color = '#16a34a'
    elif label == 'SELL':
        icon_svg = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.5"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>'
        ar_label = 'بيع'
        label_color = '#dc2626'
    else:
        icon_svg = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>'
        ar_label = 'انتظار'
        label_color = '#d97706'

    cal_icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2" style="vertical-align:-2px;margin-left:4px;"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>'

    return f"""
    <div class="rec-card {css_class}">
        <p style="font-size:0.9rem; color:#6b7280; margin:0 0 10px;">
            {cal_icon} {horizon}
        </p>
        <div style="display:flex; align-items:center; justify-content:center; gap:8px; margin:0.3rem 0;">
            {icon_svg}
            <span style="font-size:2rem; font-weight:900; color:{label_color};">{ar_label}</span>
        </div>
        <p style="font-size:0.92rem; color:#4b5563; margin:0.8rem 0 0.3rem;">{tendency}</p>
        <div style="margin:0.8rem 0 0.3rem;">
            <span style="font-size:0.88rem; color:#6b7280;">مستوى الثقة:</span>
            <span style="font-size:0.95rem; font-weight:700; color:{conf_color}; margin-right:6px;">{conf_word}</span>
        </div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-bg">
                <div class="{bar_class}" style="width:{conf}%;"></div>
            </div>
        </div>
    </div>
    """


def indicator_html(title, value, desc, status):
    css = {'pos': 'ind-pos', 'neg': 'ind-neg', 'neu': 'ind-neu'}.get(status, 'ind-neu')
    return f"""
    <div class="indicator-card">
        <p class="ind-title">{title}</p>
        <div class="ind-row">
            <span class="ind-val">{value}</span>
            <span class="ind-desc {css}">{desc}</span>
        </div>
    </div>
    """


# ─────────────────────────────────────────────
# الواجهة الرئيسية
# ─────────────────────────────────────────────

# الهيدر
st.markdown("""
<div class="header-wrap">
    <div style="display:flex; align-items:center; justify-content:center; gap:14px; margin-bottom:8px;">
        <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="48" height="48" rx="12" fill="#0a2463"/>
            <polyline points="8,34 18,22 26,28 40,14" stroke="#4ade80" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
            <circle cx="40" cy="14" r="3" fill="#4ade80"/>
            <circle cx="8" cy="34" r="3" fill="#4ade80"/>
        </svg>
        <h1 class="header-title">SmartStocks</h1>
    </div>
    <p class="header-sub">منصة تحليل الأسهم بالذكاء الاصطناعي</p>
</div>
""", unsafe_allow_html=True)

# تحذير إذا النماذج غير محملة
if not models_loaded:
    st.markdown("""
    <div class="error-box">
        ⚠️ ملفات النماذج غير موجودة.<br>
        تأكد من وجود مجلد <strong>models/</strong> يحتوي على ملفات النماذج.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── بحث السهم ───
st.markdown("""<p style="font-size:1.2rem; font-weight:700; color:#0a2463;
    margin:0 0 0.8rem; padding-bottom:0.5rem; border-bottom:2px solid #e2eaf5; display:flex; align-items:center; gap:8px;">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0a2463" stroke-width="2">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
    ابحث عن سهم</p>""", unsafe_allow_html=True)
col_input, col_btn = st.columns([3, 1])
with col_input:
    symbol_raw = st.text_input(
        label='رمز السهم',
        value='AAPL',
        placeholder='مثال: AAPL أو TSLA أو MSFT',
        label_visibility='collapsed',
    )
with col_btn:
    analyze = st.button('تحليل', type='primary', use_container_width=True)

st.markdown('<p style="font-size:0.95rem;color:#6b7280;margin:0.5rem 0 0.3rem;">أسهم شائعة:</p>', unsafe_allow_html=True)
quick_cols = st.columns(8)
quick_stocks = ['AAPL','MSFT','NVDA','TSLA','GOOGL','AMZN','META','AMD']
selected_quick = None
for i, s in enumerate(quick_stocks):
    with quick_cols[i]:
        if st.button(s, key=f'q_{s}', use_container_width=True):
            selected_quick = s

symbol = (selected_quick or symbol_raw).upper().strip()
if selected_quick:
    analyze = True

# ─── حفظ النتائج في session_state ───
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'hist' not in st.session_state:
    st.session_state.hist = None
if 'info_data' not in st.session_state:
    st.session_state.info_data = None
if 'last_symbol' not in st.session_state:
    st.session_state.last_symbol = None

# ─── التحليل ───
if analyze and symbol:
    market_data = get_market_data()
    with st.spinner(f'جاري تحليل سهم {symbol}...'):
        hist, info = fetch_display_data(symbol)
        if hist is None or hist.empty:
            st.markdown(f"""
            <div class="error-box">
                لم يتم العثور على بيانات للسهم <strong>{symbol}</strong><br>
                تأكد من صحة الرمز وحاول مرة أخرى.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        # حفظ في session_state
        st.session_state.hist        = hist
        st.session_state.info_data   = info
        st.session_state.predictions = predict(symbol, models, market_data)
        st.session_state.last_symbol = symbol

# عرض النتائج إذا موجودة
if st.session_state.get('hist') is not None:
    hist   = st.session_state.hist
    info   = st.session_state.info_data
    symbol = st.session_state.last_symbol
    if True:

        # معلومات السهم
        curr_price = float(hist['Close'].iloc[-1])
        prev_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else curr_price
        change_val = curr_price - prev_price
        change_pct = (change_val / prev_price) * 100
        change_sign  = '+' if change_pct >= 0 else ''
        change_arrow = '▲' if change_pct >= 0 else '▼'
        change_color = '#4ade80' if change_pct >= 0 else '#f87171'
        company_name = info.get('longName', info.get('shortName', symbol)) if info else symbol

        # ─── معلومات السهم ───
        market_cap = info.get('marketCap', 0) if info else 0
        week_high  = info.get('fiftyTwoWeekHigh', 0) if info else 0
        week_low   = info.get('fiftyTwoWeekLow', 0) if info else 0
        volume     = float(hist['Volume'].iloc[-1])

        if market_cap > 1e12:
            cap_str  = f"${market_cap/1e9:.0f}B"
            cap_desc = "شركة ضخمة جداً"
        elif market_cap > 1e9:
            cap_str  = f"${market_cap/1e9:.1f}B"
            cap_desc = "شركة كبيرة"
        elif market_cap > 1e6:
            cap_str  = f"${market_cap/1e6:.0f}M"
            cap_desc = "شركة متوسطة"
        else:
            cap_str  = "غير متاح"
            cap_desc = "البيانات غير متوفرة حالياً"

        if week_high > 0:
            pct_from_high = ((curr_price - week_high) / week_high) * 100
            high_desc  = f"أقل من قمته بـ {abs(pct_from_high):.0f}%" if pct_from_high < 0 else "عند قمته"
            high_color = '#dc2626' if pct_from_high < 0 else '#16a34a'
        else:
            high_desc = "—"; high_color = '#6b7280'

        if week_low > 0:
            pct_from_low = ((curr_price - week_low) / week_low) * 100
            low_desc  = f"ارتفع {pct_from_low:.0f}% من أدنى نقطة"
            low_color = '#16a34a'
        else:
            low_desc = "—"; low_color = '#6b7280'

        vol_str = f"{volume/1e6:.1f}M سهم"
        avg_vol = info.get('averageVolume', 0) if info else 0
        if avg_vol > 0:
            vol_ratio = volume / avg_vol
            if vol_ratio > 1.5:
                vol_desc = "تداول مرتفع — اهتمام كبير"
                vol_color = '#16a34a'
            elif vol_ratio < 0.5:
                vol_desc = "تداول منخفض — اهتمام قليل"
                vol_color = '#d97706'
            else:
                vol_desc = "تداول طبيعي"
                vol_color = '#6b7280'
        else:
            vol_desc = "—"; vol_color = '#6b7280'

        # SVG icons
        icon_price = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>'
        icon_cap   = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>'
        icon_high  = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>'
        icon_low   = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>'
        icon_vol   = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2"><rect x="18" y="3" width="4" height="18"/><rect x="10" y="8" width="4" height="13"/><rect x="2" y="13" width="4" height="8"/></svg>'

        def info_card_html(icon, label, value, desc, desc_color):
            return f"""
            <div style="background:white; border:1.5px solid #e2eaf5; border-radius:16px;
                        padding:1.2rem 1.5rem; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:6px; margin-bottom:8px;">
                    {icon}
                    <span style="font-size:0.85rem; color:#6b7280;">{label}</span>
                </div>
                <p style="font-size:1.5rem; font-weight:700; color:#0a2463; margin:0;">{value}</p>
                <p style="font-size:0.82rem; color:{desc_color}; margin:4px 0 0; font-weight:500;">{desc}</p>
            </div>"""

        st.markdown("""<p style="font-size:1.2rem; font-weight:700; color:#0a2463;
            margin:1.5rem 0 0.8rem; padding-bottom:0.5rem; border-bottom:2px solid #e2eaf5; display:flex; align-items:center; gap:8px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0a2463" stroke-width="2">
                <rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
            </svg>
            معلومات السهم</p>""", unsafe_allow_html=True)

        # البوكس الأزرق
        change_sign2  = '+' if change_pct >= 0 else ''
        change_color2 = '#4ade80' if change_pct >= 0 else '#f87171'
        change_arrow2 = '▲' if change_pct >= 0 else '▼'
        st.markdown(f"""
        <div style="background:#0a2463; border-radius:16px; padding:1.5rem 2rem; margin-bottom:16px;">
            <p style="color:rgba(255,255,255,0.6); font-size:0.9rem; margin:0;">{company_name} · {symbol}</p>
            <p style="color:white; font-size:2.2rem; font-weight:700; margin:4px 0;">${curr_price:,.2f}</p>
            <p style="color:{change_color2}; font-size:1rem; font-weight:600; margin:0;">
                {change_arrow2} {change_sign2}{change_val:.2f}$ ({change_sign2}{change_pct:.2f}%) اليوم
            </p>
        </div>
        """, unsafe_allow_html=True)

        # كاردز المعلومات — عمودين للجوال والكمبيوتر
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(info_card_html(
                icon_price, 'سعر السهم الآن',
                f'${curr_price:,.2f}',
                f'{change_arrow} {change_sign}{change_val:.2f}$ ({change_sign}{change_pct:.2f}%) اليوم',
                '#16a34a' if change_pct >= 0 else '#dc2626'
            ), unsafe_allow_html=True)
            st.markdown(info_card_html(
                icon_high, 'أعلى سعر في السنة',
                f'${week_high:.2f}' if week_high else '$—',
                high_desc, high_color
            ), unsafe_allow_html=True)
            st.markdown(info_card_html(
                icon_vol, 'حجم التداول اليوم',
                vol_str, vol_desc, vol_color
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(info_card_html(
                icon_cap, 'حجم الشركة',
                cap_str, cap_desc, '#6b7280'
            ), unsafe_allow_html=True)
            st.markdown(info_card_html(
                icon_low, 'أقل سعر في السنة',
                f'${week_low:.2f}' if week_low else '$—',
                low_desc, low_color
            ), unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # ─── التنبيه القانوني ───
        st.markdown("""
        <div class="legal-box">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#92400e"
                stroke-width="2" style="flex-shrink:0; margin-top:2px;">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <p class="legal-text">
                <strong>تنبيه مهم:</strong> التوصيات التالية صادرة من نموذج ذكاء اصطناعي للأغراض المعلوماتية فقط،
                وليست نصيحة مالية أو استثمارية. قرارات الاستثمار مسؤوليتك الشخصية.
                يُنصح بمراجعة مستشار مالي مرخص قبل اتخاذ أي قرار.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ─── التوصيات ───
        st.markdown("""
        <p class="section-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="2" style="vertical-align:-3px; margin-left:6px;">
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
            </svg>
            توصية الذكاء الاصطناعي
        </p>""", unsafe_allow_html=True)

        horizon_choice = st.radio(
            'اختر الفترة الزمنية:',
            options=['أسبوع', 'شهر', '3 أشهر'],
            horizontal=True,
        )
        horizon_map  = {'أسبوع': 'week', 'شهر': 'month', '3 أشهر': '3months'}
        selected_key = horizon_map[horizon_choice]

        predictions = st.session_state.predictions

        if predictions is None:
            st.markdown('<div class="error-box">❌ تعذر توليد التوصيات لهذا السهم.</div>', unsafe_allow_html=True)
        else:
            if selected_key in predictions:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(rec_card_html(predictions[selected_key]), unsafe_allow_html=True)

        # ─── الرسم البياني ───
        st.markdown("""
        <p class="section-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="2" style="vertical-align:-3px; margin-left:6px;">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
            حركة السعر — آخر 90 يوم
        </p>""", unsafe_allow_html=True)
        fig = make_chart(hist)
        st.plotly_chart(fig, use_container_width=True)

        # ─── المؤشرات الفنية ───
        st.markdown("""
        <p class="section-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="2" style="vertical-align:-3px; margin-left:6px;">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            المؤشرات الفنية
        </p>""", unsafe_allow_html=True)

        close_s  = hist['Close']
        rsi_val  = float(ta.momentum.RSIIndicator(close_s, window=14).rsi().iloc[-1])
        macd_i   = ta.trend.MACD(close_s)
        macd_v   = float(macd_i.macd().iloc[-1])
        macd_sig = float(macd_i.macd_signal().iloc[-1])
        sma20_v  = float(close_s.rolling(20).mean().iloc[-1])
        sma50_v  = float(close_s.rolling(50).mean().iloc[-1])

        # RSI
        if rsi_val > 70:
            rsi_label = 'تشبع شرائي — قد يكون السعر مرتفعاً جداً'
            rsi_color = '#dc2626'
            rsi_bg    = '#fff1f1'
            rsi_border= '#fca5a5'
        elif rsi_val < 30:
            rsi_label = 'تشبع بيعي — قد يكون السعر منخفضاً جداً'
            rsi_color = '#16a34a'
            rsi_bg    = '#f0fdf4'
            rsi_border= '#86efac'
        else:
            rsi_label = 'منطقة متوازنة — لا يوجد تشبع'
            rsi_color = '#d97706'
            rsi_bg    = '#fffbeb'
            rsi_border= '#fcd34d'

        # MACD
        if macd_v > macd_sig:
            macd_label  = 'إشارة إيجابية — الزخم صاعد'
            macd_color  = '#16a34a'
            macd_bg     = '#f0fdf4'
            macd_border = '#86efac'
        else:
            macd_label  = 'إشارة سلبية — الزخم هابط'
            macd_color  = '#dc2626'
            macd_bg     = '#fff1f1'
            macd_border = '#fca5a5'

        # SMA20
        if curr_price > sma20_v:
            sma20_label  = 'السعر فوق المتوسط — اتجاه إيجابي'
            sma20_color  = '#16a34a'
            sma20_bg     = '#f0fdf4'
            sma20_border = '#86efac'
        else:
            sma20_label  = 'السعر تحت المتوسط — اتجاه سلبي'
            sma20_color  = '#dc2626'
            sma20_bg     = '#fff1f1'
            sma20_border = '#fca5a5'

        # SMA50
        if curr_price > sma50_v:
            sma50_label  = 'السعر فوق المتوسط — اتجاه قوي'
            sma50_color  = '#16a34a'
            sma50_bg     = '#f0fdf4'
            sma50_border = '#86efac'
        else:
            sma50_label  = 'السعر تحت المتوسط — اتجاه ضعيف'
            sma50_color  = '#dc2626'
            sma50_bg     = '#fff1f1'
            sma50_border = '#fca5a5'

        def ind_card(title, subtitle, value, label, color, bg, border):
            return f"""
            <div style="background:{bg}; border:1.5px solid {border};
                        border-radius:16px; padding:1.2rem 1.5rem; height:100%;">
                <p style="font-size:0.85rem; color:#6b7280; margin:0 0 2px;">{title}</p>
                <p style="font-size:0.8rem; color:#9ca3af; margin:0 0 10px;">{subtitle}</p>
                <p style="font-size:1.8rem; font-weight:700; color:#0a2463; margin:0;">{value}</p>
                <p style="font-size:0.85rem; color:{color}; margin:6px 0 0; font-weight:500;">{label}</p>
            </div>"""

        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            st.markdown(ind_card(
                'مؤشر RSI', 'قياس قوة الاتجاه (0-100)',
                f'{rsi_val:.1f}', rsi_label, rsi_color, rsi_bg, rsi_border
            ), unsafe_allow_html=True)
        with ic2:
            st.markdown(ind_card(
                'مؤشر MACD', 'اتجاه وزخم السهم',
                f'{macd_v:.2f}', macd_label, macd_color, macd_bg, macd_border
            ), unsafe_allow_html=True)
        with ic3:
            st.markdown(ind_card(
                'متوسط 20 يوم', 'الاتجاه قصير المدى',
                f'${sma20_v:.2f}', sma20_label, sma20_color, sma20_bg, sma20_border
            ), unsafe_allow_html=True)
        with ic4:
            st.markdown(ind_card(
                'متوسط 50 يوم', 'الاتجاه متوسط المدى',
                f'${sma50_v:.2f}', sma50_label, sma50_color, sma50_bg, sma50_border
            ), unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        # ─── عن الشركة ───
        sector   = '—'
        industry = '—'
        country  = '—'
        pe       = None
        div      = None
        beta     = None

        if info:
            sector   = info.get('sector',   '—') or '—'
            industry = info.get('industry', '—') or '—'
            country  = info.get('country',  '—') or '—'
            pe       = info.get('trailingPE')
            div      = info.get('dividendYield')
            beta     = info.get('beta')

        if beta:
            if beta > 1.5:
                beta_desc  = 'مخاطر عالية جداً'
                beta_color = '#dc2626'
            elif beta > 1:
                beta_desc  = 'مخاطر أعلى من السوق'
                beta_color = '#d97706'
            else:
                beta_desc  = 'مخاطر أقل من السوق'
                beta_color = '#16a34a'
        else:
            beta_desc  = 'غير متاح'
            beta_color = '#6b7280'

        pe_str   = f'{pe:.1f}x'        if pe  else 'غير متاح'
        div_str  = f'{div*100:.2f}%'   if div else 'لا يوجد توزيعات'
        beta_str = f'{beta:.2f} — {beta_desc}' if beta else 'غير متاح'

        st.markdown("""<p style="font-size:1.2rem; font-weight:700; color:#0a2463;
            margin:1.5rem 0 0.8rem; padding-bottom:0.5rem; border-bottom:2px solid #e2eaf5;">
            عن الشركة</p>""", unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)

        with cc1:
            st.markdown(f"""
            <div style="background:white; border:1.5px solid #e2eaf5; border-radius:16px;
                        padding:1.3rem 1.5rem;">
                <p style="font-size:0.82rem; color:#9ca3af; font-weight:600;
                           margin:0 0 14px;">معلومات الشركة</p>
                <div style="display:flex; flex-direction:column; gap:12px;">
                    <div>
                        <p style="font-size:0.8rem; color:#9ca3af; margin:0;">القطاع</p>
                        <p style="font-size:1rem; color:#0a2463; font-weight:600; margin:3px 0 0;">{sector}</p>
                    </div>
                    <div>
                        <p style="font-size:0.8rem; color:#9ca3af; margin:0;">الصناعة</p>
                        <p style="font-size:1rem; color:#0a2463; font-weight:600; margin:3px 0 0;">{industry}</p>
                    </div>
                    <div>
                        <p style="font-size:0.8rem; color:#9ca3af; margin:0;">الدولة</p>
                        <p style="font-size:1rem; color:#0a2463; font-weight:600; margin:3px 0 0;">{country}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with cc2:
            st.markdown(f"""
            <div style="background:white; border:1.5px solid #e2eaf5; border-radius:16px;
                        padding:1.3rem 1.5rem;">
                <p style="font-size:0.82rem; color:#9ca3af; font-weight:600;
                           margin:0 0 14px;">مؤشرات مالية</p>
                <div style="display:flex; flex-direction:column; gap:12px;">
                    <div>
                        <p style="font-size:0.8rem; color:#9ca3af; margin:0;">نسبة P/E</p>
                        <p style="font-size:1rem; color:#0a2463; font-weight:600; margin:3px 0 0;">{pe_str}</p>
                    </div>
                    <div>
                        <p style="font-size:0.8rem; color:#9ca3af; margin:0;">التوزيعات</p>
                        <p style="font-size:1rem; color:#16a34a; font-weight:600; margin:3px 0 0;">{div_str}</p>
                    </div>
                    <div>
                        <p style="font-size:0.8rem; color:#9ca3af; margin:0;">بيتا — مستوى المخاطر</p>
                        <p style="font-size:1rem; font-weight:600; margin:3px 0 0; color:{beta_color};">{beta_str}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown('---')
st.markdown("""
<div style="text-align:center; color:#9ca3af; padding:1rem; font-size:0.9rem;">
    SmartStocks &nbsp;|&nbsp; مدعوم بنماذج XGBoost &nbsp;|&nbsp;
    البيانات من Yahoo Finance &nbsp;|&nbsp;
    للأغراض التعليمية فقط
</div>
""", unsafe_allow_html=True)