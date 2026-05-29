import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def run_v8_clinical_triage_backtest():
    print("V8 무결점 엔진 구동 중... (Cash Drag 제거, Sortino 지수 적용)")
    
    start_date, end_date = "2007-01-01", "2026-05-30"
    tickers = ['SPY', 'QQQ', 'SOXX', 'GLD', 'TLT', 'IEF', 'SHY', 'DBC']
    
    price_df = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    price_df.dropna(inplace=True) 
    
    fred_df = web.DataReader(['BAMLH0A0HYM2', 'UNRATE'], 'fred', start_date, end_date)
    unrate_monthly = fred_df['UNRATE'].resample('MS').first()
    sahm_monthly = unrate_monthly.rolling(3).mean() - unrate_monthly.rolling(12).min()
    fred_df['SAHM'] = sahm_monthly.reindex(fred_df.index).ffill() 
    fred_df.ffill(inplace=True)
    
    df = price_df.join(fred_df, how='inner').ffill()
    df['DBC_Z'] = (df['DBC'] - df['DBC'].rolling(200).mean()) / df['DBC'].rolling(200).std()
    
    trading_fee = 0.0015 
    monthly_dates = df.resample('BM').last().index
    
    cash = 1000000 
    monthly_injection = 2000000
    shares = {tk: 0 for tk in tickers}
    
    val_history, turnover_history = [], []
    target_weights = None 
    pending_rebalance = False 
    
    def get_mom(data, date_idx, tk):
        try:
            loc = data.index.get_loc(date_idx)
            if loc < 252: return 0
            m1 = (data[tk].iloc[loc] / data[tk].iloc[loc-21]) - 1
            m3 = (data[tk].iloc[loc] / data[tk].iloc[loc-63]) - 1
            m6 = (data[tk].iloc[loc] / data[tk].iloc[loc-126]) - 1
            m12 = (data[tk].iloc[loc] / data[tk].iloc[loc-252]) - 1
            return (m1 * 0.2) + (m3 * 0.3) + (m6 * 0.3) + (m12 * 0.2)
        except: return 0

    for i in range(len(df)):
        date = df.index[i]
        row = df.iloc[i]
        
        # [수정 1] T+1일 시작 시점에 적립금 즉시 투입 (Cash Drag 제거)
        is_first_day_of_month = (date.month != df.index[i-1].month if i > 0 else False)
        if is_first_day_of_month:
            cash += monthly_injection
            
        current_port_value = cash + sum(shares[k] * row[k] for k in shares)
        
        # [수정 2] T+1일 체결 (투입된 적립금을 포함하여 Diff 계산 및 매매)
        if pending_rebalance and target_weights is not None:
            trade_volume = 0
            
            # 매도 먼저 집행
            for k in shares:
                target_val = current_port_value * target_weights.get(k, 0)
                current_val = shares[k] * row[k]
                if target_val < current_val:
                    trade_amt = current_val - target_val
                    fee = trade_amt * trading_fee
                    shares[k] -= (trade_amt / row[k])
                    cash += (trade_amt - fee)
                    trade_volume += trade_amt
                    
            # 매수 집행
            for k in shares:
                target_val = current_port_value * target_weights.get(k, 0)
                current_val = shares[k] * row[k]
                if target_val > current_val:
                    trade_amt = min(target_val - current_val, cash)
                    fee = trade_amt * trading_fee
                    shares[k] += ((trade_amt - fee) / row[k])
                    cash -= trade_amt
                    trade_volume += trade_amt
            
            turnover_history.append(trade_volume / current_port_value)
            pending_rebalance = False
            
        # [T일 종가] 트리아지(Triage) 기반 체제 판별 및 시그널 생성
        if date in monthly_dates and i >= 252:
            off_mom = {tk: get_mom(df, date, tk) for tk in ['QQQ', 'SOXX', 'SPY']}
            def_mom = {tk: get_mom(df, date, tk) for tk in ['GLD', 'TLT', 'IEF', 'SHY']}
            top_off = max(off_mom, key=off_mom.get)
            w_target = {tk: 0.0 for tk in shares}
            
            is_crisis = row['BAMLH0A0HYM2'] >= 5.0 or row['SAHM'] >= 0.5
            is_inflation = row['DBC_Z'] > 1.5
            is_deflation = row['DBC_Z'] < -1.0
            
            # 엄격한 계층 구조 (확률 배분 배제)
            if is_crisis:
                w_target['SHY'], w_target['GLD'] = 0.7, 0.3
            elif is_inflation:
                w_target['GLD'], w_target['SHY'], w_target['SPY'] = 0.4, 0.2, 0.4
            elif is_deflation:
                w_target['TLT'] = 0.5 if def_mom['TLT'] > 0 else 0.0
                w_target['IEF'] = 0.5 if def_mom['TLT'] <= 0 else 0.0
                w_target['QQQ'] = 0.5
            else: 
                if off_mom[top_off] > 0:
                    w_target[top_off], w_target['SPY'] = 0.8, 0.2
                else:
                    w_target['SHY'] = 1.0
            
            target_weights = w_target
            pending_rebalance = True 
            
        val_history.append(cash + sum(shares[k] * row[k] for k in shares))

    df['Portfolio_Value'] = val_history
    
    # [수정 3] 정밀 지표 산출 (Exact CAGR & Sortino)
    total_days = (df.index[-1] - df.index[0]).days
    cagr = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) ** (365.25 / total_days) - 1
    
    daily_returns = df['Portfolio_Value'].pct_change().dropna()
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = np.sqrt(252) * (daily_returns.mean() / downside_returns.std())
    
    mdd = ((df['Portfolio_Value'] - df['Portfolio_Value'].cummax()) / df['Portfolio_Value'].cummax()).min()
    avg_annual_turnover = np.mean(turnover_history) * 12 if turnover_history else 0

    print(f"==================================================================")
    print(f" [V8 Clinical Triage Engine 결괏값]")
    print(f"==================================================================")
    print(f" 정확한 연평균 수익률(CAGR) : {cagr*100:.2f}%")
    print(f" 소르티노 지수(Sortino)     : {sortino_ratio:.2f} (하방 리스크 통제력)")
    print(f" 최대 낙폭(MDD)             : {mdd*100:.2f}%")
    print(f" 연평균 회전율(Turnover)    : {avg_annual_turnover*100:.1f}%")
    print(f"==================================================================")

run_v8_clinical_triage_backtest()