import streamlit as st
import pandas as pd
import requests
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 초기 UI 및 API 키 세팅
# ==========================================
st.set_page_config(page_title="V8 ISA 오토파일럿", page_icon="🦅", layout="wide")
st.title("🦅 V8 ISA 자산배분 오토파일럿")
st.caption("안정망: Tiingo 공식 금융 API 직결망 (IP 차단 원천 제거)")

with st.sidebar:
    st.header("🔑 시스템 설정")
    api_key = st.text_input("Tiingo API Token 입력", type="password", help="api.tiingo.com에서 발급받은 토큰을 입력하세요.")
    st.markdown("---")
    st.markdown("- **API 제공사:** Tiingo\n- **호출 한도:** 일 500회 (무료)")

if not api_key:
    st.warning("👈 좌측 사이드바에 Tiingo API 토큰을 입력해야 엔진이 구동됩니다.")
    st.stop()

# ==========================================
# 2. 정식 API 데이터 통신망
# ==========================================
@st.cache_data(ttl=3600)
def fetch_fred_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    def get_fred(id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
        res = requests.get(url, headers=headers)
        return pd.read_csv(io.StringIO(res.text), index_col=0, parse_dates=True)[id]
    
    try:
        hy = get_fred('BAMLH0A0HYM2')
        unrate = get_fred('UNRATE')
        return pd.concat([hy, unrate], axis=1).ffill().last('400D')
    except:
        return pd.DataFrame({'BAMLH0A0HYM2': [3.5], 'UNRATE': [4.0]}, index=[datetime.now()])

@st.cache_data(ttl=3600)
def fetch_tiingo_data(token):
    tickers = ['SPY', 'QQQ', 'SOXX', 'GLD', 'TLT', 'IEF', 'SHY', 'DBC']
    start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')
    dfs = []
    
    for tk in tickers:
        url = f"https://api.tiingo.com/tiingo/daily/{tk}/prices?startDate={start_date}&token={token}"
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                df.set_index('date', inplace=True)
                dfs.append(df[['close']].rename(columns={'close': tk}))
        else:
            st.error(f"API 통신 에러 (티커: {tk}): 유효하지 않은 API 키이거나 한도를 초과했습니다.")
            st.stop()
            
    if dfs:
        return pd.concat(dfs, axis=1).sort_index().ffill().bfill()
    else:
        st.error("금융 데이터 수집에 실패했습니다. API 키를 다시 확인해 주세요.")
        st.stop()

with st.spinner("정식 API 망을 통해 글로벌 데이터를 수신 중입니다..."):
    fred_df = fetch_fred_data()
    price_df = fetch_tiingo_data(api_key)

# ==========================================
# 3. 거시 지표 및 모멘텀 연산
# ==========================================
available_len = len(price_df)
m1_win = min(21, available_len - 1)
m3_win = min(63, available_len - 1)
m6_win = min(126, available_len - 1)
m12_win = min(252, available_len - 1)

hy_spread = fred_df['BAMLH0A0HYM2'].iloc[-1]
sahm_rule = 0.2
if 'UNRATE' in fred_df.columns and len(fred_df['UNRATE'].dropna()) >= 12:
    unrate_monthly = fred_df['UNRATE'].dropna().resample('MS').first()
    sahm_rule = (unrate_monthly.rolling(3).mean() - unrate_monthly.rolling(12).min()).iloc[-1]

dbc_win = min(200, len(price_df['DBC']))
dbc_ma200 = price_df['DBC'].rolling(dbc_win).mean().iloc[-1]
dbc_std200 = price_df['DBC'].rolling(dbc_win).std().iloc[-1]
dbc_z = (price_df['DBC'].iloc[-1] - dbc_ma200) / dbc_std200 if dbc_std200 > 0 else 0

def calc_momentum(df, tk):
    p = df[tk]
    m1 = (p.iloc[-1] / p.iloc[-m1_win]) - 1 if m1_win > 0 else 0
    m3 = (p.iloc[-1] / p.iloc[-m3_win]) - 1 if m3_win > 0 else 0
    m6 = (p.iloc[-1] / p.iloc[-m6_win]) - 1 if m6_win > 0 else 0
    m12 = (p.iloc[-1] / p.iloc[-m12_win]) - 1 if m12_win > 0 else 0
    return (m1 * 0.2) + (m3 * 0.3) + (m6 * 0.3) + (m12 * 0.2)

off_mom = {tk: calc_momentum(price_df, tk) for tk in ['QQQ', 'SOXX', 'SPY']}
def_mom = {tk: calc_momentum(price_df, tk) for tk in ['GLD', 'TLT', 'IEF', 'SHY']}
top_off = max(off_mom, key=off_mom.get) if off_mom else 'SPY'

# ==========================================
# 4. 트리아지(Triage) 체제 판별 및 비중 할당
# ==========================================
w_target = {tk: 0.0 for tk in ['QQQ', 'SOXX', 'SPY', 'GLD', 'TLT', 'IEF', 'SHY']}

is_crisis = hy_spread >= 5.0 or sahm_rule >= 0.5
is_inflation = dbc_z > 1.5
is_deflation = dbc_z < -1.0

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
    w_target['TLT'] = 0.5 if def_mom.get('TLT', 0) > 0 else 0.0
    w_target['IEF'] = 0.5 if def_mom.get('TLT', 0) <= 0 else 0.0
    w_target['QQQ'] = 0.5
else:
    regime_text = "☀️ [GOLDILOCKS] 안정적 성장. 상위 공격 자산 집중"
    regime_color = "success"
    if off_mom.get(top_off, 0) > 0:
        w_target[top_off], w_target['SPY'] = 0.8, 0.2
    else:
        w_target['SHY'] = 1.0 

# ==========================================
# 5. 대시보드 UI 출력
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