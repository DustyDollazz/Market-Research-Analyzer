import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import io
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
STATION='9410660'
LAGS=(0,1,2,3,5,10)
def constituents():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=30)
    response.raise_for_status()
    table = pd.read_html(response.text, match='Symbol')[0]
    return dict(zip(table['Symbol'].str.replace('.', '-', regex=False), table['Security']))


def tides(start, end):
    # NOAA predicted hourly levels, MLLW datum. Fetch in <= one-year blocks.
    chunks = []
    cursor = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while cursor <= end:
        stop = min(cursor + pd.DateOffset(months=11), end)
        params = {'begin_date':cursor.strftime('%Y%m%d'), 'end_date':stop.strftime('%Y%m%d'),
                  'station':STATION, 'product':'predictions', 'datum':'MLLW',
                  'time_zone':'lst_ldt', 'units':'metric', 'interval':'h', 'format':'json',
                  'application':'StockTideAnalyzer'}
        r = requests.get('https://api.tidesandcurrents.noaa.gov/api/prod/datagetter', params=params, timeout=60)
        r.raise_for_status()
        obj = r.json()
        if 'predictions' not in obj:
            raise RuntimeError('NOAA response: '+str(obj.get('error', obj))[:300])
        chunks.extend(obj['predictions'])
        cursor = stop + pd.Timedelta(days=1)
    t = pd.DataFrame(chunks)
    t['time'] = pd.to_datetime(t['t'])
    t['height'] = pd.to_numeric(t['v'], errors='coerce')
    t = t.drop_duplicates('time').set_index('time').sort_index()
    # Tidal range per calendar day, rather than time-of-day aliasing.
    return t['height'].resample('D').agg(lambda x: x.max()-x.min() if len(x)>=18 else np.nan).rename('tidal_range')


def evaluate(tickers, tide, start, end, log, min_samples=220):
    symbols = list(tickers)
    results = []
    for block in range(0, len(symbols), 40):
        group = symbols[block:block+40]
        log(f'Downloading {block+1}–{min(block+40,len(symbols))} / {len(symbols)}')
        try:
            raw = yf.download(group, start=start, end=(pd.Timestamp(end)+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                              auto_adjust=True, progress=False, threads=False)
            close = raw['Close']
            if isinstance(close, pd.Series):
                close = close.to_frame(name=group[0])
        except Exception as exc:
            log(f'Batch failed: {exc}')
            continue
        for ticker in group:
            if ticker not in close.columns:
                continue
            prices = close[ticker].dropna().sort_index()
            returns = np.log(prices).diff().replace([np.inf,-np.inf], np.nan).dropna()
            if len(returns)<min_samples+1:
                continue
            frame = pd.DataFrame({'ret':returns})
            frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
            frame = frame.join(tide, how='left')
            # Multiple lags, and multiple stocks: correct across all tested hypotheses.
            for lag in LAGS:
                paired = pd.DataFrame({'ret':frame.ret, 'tide':frame.tidal_range.shift(lag)}).dropna()
                if len(paired)<min_samples or paired.ret.std()==0 or paired.tide.std()==0:
                    continue
                r,p = pearsonr(paired.ret, paired.tide)
                # Check consistency in two disjoint chronological halves.
                halves = np.array_split(np.arange(len(paired)),2)
                rs = [pearsonr(paired.ret.iloc[ix],paired.tide.iloc[ix]).statistic for ix in halves]
                consistent = (np.sign(rs[0])==np.sign(rs[1])==np.sign(r))
                results.append(dict(ticker=ticker, company=tickers[ticker], lag=lag,
                                    r=float(r), p=float(p), n=len(paired),
                                    first_half_r=float(rs[0]), second_half_r=float(rs[1]),
                                    consistent=bool(consistent)))
    if not results:
        raise RuntimeError('No analyzable stock histories were downloaded.')
    data = pd.DataFrame(results)
    data['q'] = multipletests(data.p, method='fdr_bh')[1]
    data['evidence'] = np.where(data.consistent, -np.log10(np.maximum(data.q,1e-300)),0)
    # Best lag per stock after global correction; bars reflect evidence, not certainty.
    best = data.sort_values(['evidence','q'],ascending=[False,True]).drop_duplicates('ticker')
    return best.sort_values(['evidence','q'],ascending=[False,True]).reset_index(drop=True)



def universe(market, custom):
    if market == 'Custom':
        return {s.upper():s.upper() for s in custom.replace(',', ' ').split()}
    if market == 'S&P 500':
        return constituents()
    if market == 'Dow Jones':
        r=requests.get('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average',headers={'User-Agent':'Mozilla/5.0'},timeout=30)
        r.raise_for_status()
        for t in pd.read_html(io.StringIO(r.text)):
            if 'Symbol' in t.columns and 'Company' in t.columns:
                return dict(zip(t.Symbol,t.Company))
        raise RuntimeError('Dow constituents unavailable')
    r=requests.get('https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt',timeout=60)
    r.raise_for_status()
    t=pd.read_csv(io.StringIO(r.text),sep='|')
    t=t[t['Test Issue'].eq('N') & t.Symbol.notna()]
    return dict(zip(t.Symbol.str.replace('.', '-',regex=False),t['Security Name']))


@st.cache_resource
def workers():
    return ThreadPoolExecutor(max_workers=2), Lock()

def job(market, custom, months, limit, mode):
    names=universe(market,custom)
    if limit: names=dict(list(names.items())[:limit])
    if not names: raise ValueError("No symbols selected")
    end=dt.date.today()-dt.timedelta(days=1)
    start=(pd.Timestamp(end)-pd.DateOffset(months=months)).date()
    if mode=="Tidal":
        feature=tides(start,end)
    else:
        dates=pd.date_range(start,end)
        age=(dates-pd.Timestamp("2000-01-06 18:14")).total_seconds()/86400
        feature=pd.Series((1-np.cos(2*np.pi*age/29.530588853))/2,index=dates,name="tidal_range")
    # Serialize Yahoo downloads to avoid overlapping library calls.
    with workers()[1]:
        data=evaluate(names,feature,start,end,lambda msg: None,min_samples=12 if months==1 else min(220,months*12))
    return data,len(names)

st.set_page_config(page_title="Market Research Analyzer",layout="wide")
st.title("Market Research Analyzer")
st.caption("Cloud prototype • daily correlations")
jobs=st.session_state.setdefault("jobs",{})
results=st.session_state.setdefault("results",{})
for key,future in list(jobs.items()):
    if future.done():
        try: results[key]=future.result()
        except Exception as e: results[key]=str(e)
        del jobs[key]
dashboard,tidal,lunar,charts,report=st.tabs(["Dashboard","Tidal analysis","Lunar analysis","Stock charts","Results"])
with dashboard:
    st.write("Up to two submitted analysis jobs; market-data downloads are serialized for reliability.")
    for mode in ["Tidal","Lunar"]:
        st.write(mode+": "+("Running or waiting" if mode in jobs else "Finished" if mode in results else "Ready"))
    st.button("Refresh job status")
    st.caption("Momentum scanner and 15/30/60-minute backtesting: planned future upgrade.")
for tab,mode in [(tidal,"Tidal"),(lunar,"Lunar")]:
    with tab:
        with st.form(mode):
            market=st.selectbox("Market",["Custom","S&P 500","NASDAQ","Dow Jones"],key=mode+"market")
            custom=st.text_input("Custom symbols (used for Custom market)","SPY QQQ GLD GM",key=mode+"custom")
            months=st.selectbox("Historical months",[1,2,3,6,9,12,24,36,60,120],index=3,key=mode+"months")
            limit=st.number_input("Maximum symbols (0 = all)",0,10000,50,key=mode+"limit")
            submit=st.form_submit_button("Start "+mode+" analysis",disabled=mode in jobs)
        if submit and mode not in jobs:
            jobs[mode]=workers()[0].submit(job,market,custom,months,limit,mode)
            st.rerun()
        if mode in jobs: st.info("Job submitted. Switch tabs freely; use Refresh job status to check completion.")
        if mode in results:
            if isinstance(results[mode],str): st.error(results[mode])
            else:
                data,total=results[mode]
                st.write(f"Analyzed {len(data)} of {total} requested symbols")
                st.dataframe(data)
                st.bar_chart(data.head(25).set_index("ticker")["evidence"])
with charts:
    symbol=st.text_input("Chart ticker","SPY")
    if st.button("Load daily chart"):
        try:
            with workers()[1]:
                raw=yf.download(symbol,period="3mo",auto_adjust=True,progress=False,threads=False)
            if raw.empty: raise ValueError("No prices returned")
            close=raw["Close"].squeeze()
            st.line_chart(close)
        except Exception as e: st.error(str(e))
with report:
    for mode,value in results.items():
        if not isinstance(value,str):
            st.download_button("Export "+mode+" CSV",value[0].to_csv(index=False),mode.lower()+"_results.csv",key=mode+"export")
    if not results: st.write("No completed analyses yet.")
st.caption("Exploratory Pearson correlations with BH adjustment and split-half checks; not predictive validation. Current constituent lists introduce survivorship bias. Lunar illumination is approximate. Intraday turning points and persistent storage are not yet implemented. Free hosting may interrupt jobs.")
