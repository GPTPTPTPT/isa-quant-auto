import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 초기 UI 세팅
# ==========================================
st.set_page_config(page_title="V8 ISA 오토파일럿", page_icon="🦅", layout="wide")
st.title("🦅 V8 ISA 자산배분 오토파일럿")
st.caption(f"최종 업데이트: {datetime.now().strftime('%Y-%m-%d')} | V8 Clinical Triage Engine")

# ==========================================
# 2. 데이터 직수입 함수 (안정성 극대화 버전)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_fred_data():
    def get_series(series_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')
        df.columns = [series_id]
        df.index.name = 'DATE'
        return df
    try:
        hy = get_series('BAMLH0A0HYM2')
        unrate = get_series('UNRATE')
        fred_df = hy.join(unrate, how='outer').ffill().last('400D')
    except:
        st.warning("⚠️ FRED 서버 지연으로 임시 매크로 지표를 대입합니다.")
        data = {'BAMLH0A0HYM2': [3.5], 'UNRATE': [4.0]}
        fred_df = pd.DataFrame(data, index=[datetime.now()])
    return fred_df

@st.cache_data(ttl=3600)
def fetch_price_data():
    tickers = ['SPY', 'QQQ', 'SOXX', 'GLD', 'TLT', 'IEF', 'SHY', 'DBC']
    start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')
    
    try:
        df = yf.download(tickers, start=start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.levels[0]: df = df['Close']
            elif 'Adj Close' in df.columns.levels[0]: df = df['Adj Close']
    except:
        dfs = []
        for t in tickers:
            try:
                url = f"https://stooq.com/q/d/l/?s={t}.us&i=d"
                tmp = pd.read_csv(url, index_col='Date', parse_dates=True)[['Close']]
                tmp.columns = [t]
                dfs.append(tmp)
            except:
                pass
        if dfs:
            df = pd.concat(dfs, axis=1).sort_index()
        else:
            df = pd.DataFrame()

    if not df.empty:
        df = df.ffill().bfill()
    return df

with st.spinner("글로벌 매크로 센서 및 자산 가격 데이터 실시간 스캔 중..."):
    fred_df = fetch_fred_data()
    price_df = fetch_price_data()

# ==========================================
# 3. 데이터 최종 유효성 검증 및 예외 처리
# ==========================================
if price_df.empty:
    st.error("⚠️ 글로벌 금융 데이터 서버(Yahoo/Stooq)가 모두 응답하지 않고 있습니다. 잠시 후 새로고침 해주세요.")
    st.stop()

# 확보된 데이터 길이에 맞춰 계산 범위 유연하게 가변화
available_len = len(price_df)
m1_win = min(21, available_len - 1)
m3_win = min(63, available_len - 1)
m6_win = min(126, available_len - 1)
m12_win = min(252, available_len - 1)

# ==========================================
# 4. 거시 지표 및 모멘텀 연산 (V8 로직)
# ==========================================
try:
    hy_spread = fred_df['BAMLH0A0HYM2'].ffill().iloc[-1] if 'BAMLH0A0HYM2' in fred_df.columns else 3.5
    
    if 'UNRATE' in fred_df.columns and len(fred_df['UNRATE'].dropna()) >= 12:
        unrate_monthly = fred_df['UNRATE'].dropna().resample('MS').first()
        sahm_rule = (unrate_monthly.rolling(3).mean() - unrate_monthly.rolling(12).min()).iloc[-1]
    else:
        sahm_rule = 0.2

    dbc_prices = price_df['DBC'] if 'DBC' in price_df.columns else price_df['SPY'] 
    dbc_win = min(200, len(dbc_prices))
    dbc_ma200 = dbc_prices.rolling(dbc_win).mean().iloc[-1]
    dbc_std200 = dbc_prices.rolling(dbc_win).std().iloc[-1]
    dbc_z = (dbc_prices.iloc[-1] - dbc_ma200) / dbc_std200 if dbc_std200 > 0 else 0

    def calc_momentum(df, tk):
        # [오타 수정 완료] tk syntax not in 삭제 
        if tk not in df.columns: return -999
        p = df[tk]
        m1 = (p.iloc[-1] / p.iloc[-m1_win]) - 1 if m1_win > 0 else 0
        m3 = (p.iloc[-1] / p.iloc[-m3_win]) - 1 if m3_win > 0 else 0
        m6 = (p.iloc[-1] / p.iloc[-m6_win]) - 1 if m6_win > 0 else 0
        m12 = (p.iloc[-1] / p.iloc[-m12_win]) - 1 if m12_win > 0 else 0
        return (m1 * 0.2) + (m3 * 0.3) + (m6 * 0.3) + (m12 * 0.2)

    active_tickers = [t for t in ['QQQ', 'SOXX', 'SPY'] if t in price_df.columns]
    off_mom = {tk: calc_momentum(price_df, tk) for tk in active_tickers}
    def_mom = {tk: calc_momentum(price_df, tk) for tk in ['GLD', 'TLT', 'IEF', 'SHY'] if tk in price_df.columns}
    
    top_off = max(off_mom, key=off_mom.get) if off_mom else 'SPY'
    
except Exception as e:
    st.error(f"시스템 지표 파싱 오류 (자동 우회 실행): {e}")
    top_off = 'SPY'
    hy_spread, sahm_rule, dbc_z = 3.5, 0.2, 0.0

# ==========================================
# 5. 트리아지(Triage) 체제 판별 및 비중 할당
# ==========================================
w_target = {tk: 0.0 for tk in ['QQQ', 'SOXX', 'SPY', 'GLD', 'TLT', 'IEF', 'SHY']}

is_crisis = hy_spread >= 5.0 or sahm_rule >= 0.5
is_inflation = dbc_z > 1.5
is_deflation = dbc_z < -1.0

regime_text = ""
regime_color = ""

if is_crisis:
    regime_text = "🚨 [CRISIS] 시스템 붕괴 감지. 현금/금 전량 대피"
    regime_color = "error"
    w_target['SHY'], w_target['GLD'] = 0.7, 0.3
elif is_inflation:
    regime_text = "🔥 [INFLATION] 원자재 발작. 방어적 혼합 자산 배분"
    regime_color = "warning"
    w_target['GLD'], w_target['SHY'], w_target['SPY'] = 0.4, 0.2, 0.4
elif is_deflation:
    regime_text = "❄️ [DEFLATION] 침체 국면. 국채 및 유동성 자산(QQQ) 헷지"
    regime_color = "info"
    w_target['TLT'] = 0.5 if ('TLT' in def_mom and def_mom['TLT'] > 0) else 0.0
    w_target['IEF'] = 0.5 if ('TLT' in def_mom and def_mom['TLT'] <= 0) else 0.0
    w_target['QQQ'] = 0.5
else:
    regime_text = "☀️ [GOLDILOCKS] 안정적 성장. 상위 공격 자산 집중"
    regime_color = "success"
    if top_off in off_mom and off_mom[top_off] > 0:
        w_target[top_off], w_target['SPY'] = 0.8, 0.2
    else:
        w_target['SHY'] = 1.0 

# ==========================================
# 6. 화면 출력 (대시보드 UI)
# ==========================================
st.subheader("1️⃣ 실시간 매크로 센서 (FRED & Market)")
c1, c2, c3 = st.columns(3)
c1.metric("HY 스프레드", f"{hy_spread:.2f}%", "5.0% 이상 위기", delta_color="inverse")
c2.metric("샴 룰 (Sahm)", f"{sahm_rule:.2f}%p", "0.5%p 이상 침체", delta_color="inverse")
c3.metric("원자재(DBC) Z-스코어", f"{dbc_z:.2f}", "1.5 이상 인플레", delta_color="inverse")

st.divider()

st.subheader("2️⃣ 이번 달 ISA 계좌 매매 지침")
if regime_color == "error": st.error(f"**{regime_text}**")
elif regime_color == "warning": st.warning(f"**{regime_text}**")
elif regime_color == "success": st.success(f"**{regime_text}**")
else: st.info(f"**{regime_text}**")

isa_mapping = {
    'QQQ': 'ACE 미국빅테크TOP7Plus',
    'SOXX': 'TIGER 미국필라델피아반도체나스닥',
    'SPY': 'TIGER 미국S&P500',
    'GLD': 'ACE KRX금현물',
    'TLT': 'ACE 미국30년국채액티브(H)',
    'IEF': 'TIGER 미국10년국채',
    'SHY': 'KODEX 미국달러SOFR금리액티브'
}

st.write("매월 10일경, HTS/MTS를 켜고 아래 비중에 맞게 리밸런싱 하십시오.")

col_a, col_b = st.columns([1, 1])
with col_a:
    st.markdown("### 🛒 목표 비중 (%)")
    for tk, weight in w_target.items():
        if weight > 0:
            st.write(f"• **{tk}**: {weight * 100:.0f}%")
            st.progress(int(weight * 100))
with col_b:
    st.markdown("### 🇰🇷 ISA 매수 추천 티커")
    for tk, weight in w_target.items():
        if weight > 0:
            st.code(f"{isa_mapping.get(tk, tk)} ({weight * 100:.0f}%)")