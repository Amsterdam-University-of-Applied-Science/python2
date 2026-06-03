import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from IPython.display import display, Markdown

# ─────────────────────────────────────────────
# CONFIGURATIE
# ─────────────────────────────────────────────

TICKERS = [
    # Aandelen
    "MC.PA", "BRK-B", "MSFT", "JPM",
    # Obligaties (Treasury Yields)
    "^TNX", "^IRX", "^TYX",
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD",
    # Grondstoffen
    "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "ZC=F", "ZW=F", "KC=F", "ZS=F"
]

START_DATE = '2019-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Kolommen die we willen behouden na selectie
GEWENSTE_KOLOMMEN = ['Close', 'Open', 'High', 'Low', 'Volume']


# ─────────────────────────────────────────────
# STAP 1: INLEZEN DATA
# ─────────────────────────────────────────────

def stap1_inlezen(ticker_symbol: str) -> pd.DataFrame:
    """
    Haalt live data op van yfinance voor één ticker.
    Retourneert een ruwe DataFrame zoals yfinance die levert.
    """
    display(Markdown(f"## Stap 1 — Inlezen data: `{ticker_symbol}`"))

    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=START_DATE, end=END_DATE)

    if df.empty:
        print(f"⚠️ Geen data gevonden voor {ticker_symbol}")
        return pd.DataFrame()

    print(f"✅ Data opgehaald: {len(df)} rijen van {df.index.min().date()} t/m {df.index.max().date()}")
    display(df.head())
    return df


# ─────────────────────────────────────────────
# STAP 2: DATATYPES BEKIJKEN
# ─────────────────────────────────────────────

def stap2_datatypes(df: pd.DataFrame) -> None:
    """
    Toont de shape, kolomnamen, datatypes, en een statistisch overzicht
    van de ruwe DataFrame.
    """
    display(Markdown("## Stap 2 — Datatypes bekijken"))

    print(f"Shape: {df.shape[0]} rijen × {df.shape[1]} kolommen")
    print(f"Index type: {type(df.index).__name__} | Timezone: {df.index.tz}\n")

    display(Markdown("**Kolommen en datatypes:**"))
    dtype_df = pd.DataFrame({
        'Kolom': df.columns,
        'Dtype': [str(df[c].dtype) for c in df.columns],
        'Niet-null waarden': [df[c].notna().sum() for c in df.columns],
        'Null waarden': [df[c].isna().sum() for c in df.columns],
    })
    display(dtype_df.set_index('Kolom'))

    display(Markdown("**Statistisch overzicht:**"))
    display(df.describe().round(4))


# ─────────────────────────────────────────────
# STAP 3: DATA SCHONEN
# ─────────────────────────────────────────────

def stap3_schonen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Schoont de ruwe data:
    - Normaliseert de datetime index naar UTC
    - Verwijdert duplicaten
    - Reindext naar dagelijkse kalender (incl. weekenden)
    - Forward-fill en backward-fill voor ontbrekende waarden
    """
    display(Markdown("## Stap 3 — Data schonen"))

    # Normaliseer index naar UTC
    df.index = pd.to_datetime(df.index, utc=True).normalize()
    duplicaten = df.index.duplicated().sum()
    df = df[~df.index.duplicated(keep='last')]
    print(f"🗑️  Verwijderde duplicaten: {duplicaten}")

    # Reindex naar volledige dagelijkse kalender
    daily_idx = pd.date_range(
        start=max(pd.to_datetime(START_DATE).tz_localize('UTC'), df.index.min()),
        end=min(pd.to_datetime(END_DATE).tz_localize('UTC'), df.index.max()),
        freq='D'
    )
    df = df.reindex(daily_idx)
    df.index.name = "Date"

    nulls_voor = df.isna().sum().sum()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    nulls_na = df.isna().sum().sum()

    print(f"📅 Geherindexeerd naar dagelijks: {len(df)} rijen")
    print(f"🔧 Null-waarden opgevuld via ffill/bfill: {nulls_voor} → {nulls_na}")
    display(df.head())
    return df


# ─────────────────────────────────────────────
# STAP 4: DATA SELECTEREN
# ─────────────────────────────────────────────

def stap4_selecteren(df: pd.DataFrame, kolommen: list = None) -> pd.DataFrame:
    """
    Selecteert alleen de gewenste kolommen uit de DataFrame.
    Standaard: GEWENSTE_KOLOMMEN bovenin dit bestand.
    """
    display(Markdown("## Stap 4 — Data selecteren"))

    if kolommen is None:
        kolommen = GEWENSTE_KOLOMMEN

    beschikbaar = [k for k in kolommen if k in df.columns]
    weggelaten = [k for k in df.columns if k not in beschikbaar]

    print(f"✅ Geselecteerde kolommen: {beschikbaar}")
    print(f"🗂️  Weggelaten kolommen:    {weggelaten}")

    df_sel = df[beschikbaar].copy()
    display(df_sel.head())
    return df_sel


# ─────────────────────────────────────────────
# STAP 5: DATA TRANSFORMEREN
# ─────────────────────────────────────────────

def stap5_transformeren(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt berekende kolommen toe:
    - Close_diff: absolute dagelijkse verandering
    - Returns_pct: procentuele dagelijkse verandering
    Rondt alle waarden af op 6 decimalen.
    """
    display(Markdown("## Stap 5 — Data transformeren"))

    df = df.copy()

    if 'Close' in df.columns:
        df['Close_diff'] = df['Close'].diff().fillna(0)
        df['Returns_pct'] = df['Close'].pct_change().fillna(0)
        print("➕ Kolommen toegevoegd: Close_diff, Returns_pct")

    df = df.round(6)

    display(Markdown("**Voorbeeld getransformeerde data:**"))
    display(df[['Close', 'Close_diff', 'Returns_pct']].dropna().head(10))

    display(Markdown("**Statistieken nieuwe kolommen:**"))
    display(df[['Close_diff', 'Returns_pct']].describe().round(6))

    return df


# ─────────────────────────────────────────────
# STAP 6: DATA VISUALISEREN
# ─────────────────────────────────────────────

def stap6_visualiseren(df: pd.DataFrame, ticker_symbol: str) -> None:
    """
    Toont drie grafieken:
    1. Slotkoers over tijd
    2. Dagelijks rendement (%)
    3. Volume (indien beschikbaar)
    """
    display(Markdown(f"## Stap 6 — Data visualiseren: `{ticker_symbol}`"))

    # Datetime index zonder timezone voor matplotlib
    plot_index = df.index.tz_localize(None) if df.index.tz else df.index

    heeft_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
    aantal_plots = 3 if heeft_volume else 2

    fig, axes = plt.subplots(aantal_plots, 1, figsize=(14, 4 * aantal_plots), sharex=True)
    fig.suptitle(f'{ticker_symbol} — Live Data Analyse ({START_DATE} t/m {END_DATE})',
                 fontsize=14, fontweight='bold', y=1.01)

    # --- Plot 1: Slotkoers ---
    ax1 = axes[0]
    ax1.plot(plot_index, df['Close'], color='#1f77b4', linewidth=1.2, label='Slotkoers')
    ax1.set_ylabel('Prijs', fontsize=11)
    ax1.set_title('Slotkoers over tijd', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)

    # --- Plot 2: Dagelijks rendement ---
    ax2 = axes[1]
    returns = df['Returns_pct'] * 100
    kleuren = ['#2ca02c' if r >= 0 else '#d62728' for r in returns]
    ax2.bar(plot_index, returns, color=kleuren, width=1.0, alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax2.set_ylabel('Rendement (%)', fontsize=11)
    ax2.set_title('Dagelijks rendement (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 3: Volume (optioneel) ---
    if heeft_volume:
        ax3 = axes[2]
        ax3.bar(plot_index, df['Volume'], color='#7f7f7f', width=1.0, alpha=0.6)
        ax3.set_ylabel('Volume', fontsize=11)
        ax3.set_title('Handelsvolume', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.5)

    # X-as opmaak
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel('Datum', fontsize=11)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# PIPELINE: ALLE STAPPEN ACHTER ELKAAR
# ─────────────────────────────────────────────

def analyseer_ticker(ticker_symbol: str) -> pd.DataFrame:
    """
    Voert alle 6 stappen achter elkaar uit voor één ticker.
    Retourneert de volledig verwerkte DataFrame.
    """
    df_rauw      = stap1_inlezen(ticker_symbol)
    if df_rauw.empty:
        return df_rauw
    stap2_datatypes(df_rauw)
    df_schoon    = stap3_schonen(df_rauw)
    df_select    = stap4_selecteren(df_schoon)
    df_transform = stap5_transformeren(df_select)
    stap6_visualiseren(df_transform, ticker_symbol)
    return df_transform


# ─────────────────────────────────────────────
# AANROEPEN — pas de ticker aan naar wens
# ─────────────────────────────────────────────

import plotly.graph_objects as px
from plotly.subplots import make_subplots

def exporteer_naar_mooie_html(df: pd.DataFrame, ticker_symbol: str, bestandsnaam: str = "dashboard.html"):
    """
    Genereert een prachtig, responsive HTML-dashboard met interactieve grafieken
    en statistieken op basis van de verwerkte DataFrame.
    """
    # 1. Bereken een aantal key metrics voor het dashboard
    laatste_prijs = df['Close'].iloc[-1]
    vorige_prijs = df['Close'].iloc[-2] if len(df) > 1 else laatste_prijs
    absolute_verandering = laatste_prijs - vorige_prijs
    procentuele_verandering = (absolute_verandering / vorige_prijs) * 100
    
    kleur_klaar = "success" if absolute_verandering >= 0 else "danger"
    teken = "+" if absolute_verandering >= 0 else ""

    # 2. Maak de interactieve Plotly Grafieken
    heeft_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
    rijen = 3 if heeft_volume else 2
    
    fig = make_subplots(
        rows=rijen, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25] if heeft_volume else [0.6, 0.4]
    )

    # Plot 1: Slotkoers (Lijn)
    fig.add_trace(
        px.Scatter(x=df.index, y=df['Close'], name="Slotkoers", line=dict(color="#2563eb", width=2)),
        row=1, col=1
    )

    # Plot 2: Dagelijks Rendement (Bars met kleur op basis van waarde)
    ret_kleuren = ["#10b981" if r >= 0 else "#ef4444" for r in df['Returns_pct']]
    fig.add_trace(
        px.Bar(x=df.index, y=df['Returns_pct'] * 100, name="Rendement (%)", marker_color=ret_kleuren),
        row=2, col=1
    )

    # Plot 3: Volume (Optioneel)
    if heeft_volume:
        fig.add_trace(
            px.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="#64748b"),
            row=3, col=1
        )

    # Update lay-out voor een moderne 'dark/clean' look
    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        showlegend=False
    )
    
    # Update y-assen titels
    fig.update_yaxes(title_text="Prijs ($)", row=1, col=1)
    fig.update_yaxes(title_text="Rendement (%)", row=2, col=1)
    if heeft_volume:
        fig.update_yaxes(title_text="Volume", row=3, col=1)

    # Converteer de Plotly grafiek naar een kale HTML-div string
    grafiek_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # 3. De complete HTML/CSS Template (Bootstrap 5)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="nl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{ticker_symbol} Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            .card {{ border: none; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
            .metric-title {{ font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }}
            .metric-value {{ font-size: 1.75rem; font-weight: 700; color: #1e293b; }}
        </style>
    </head>
    <body>

        <div class="container py-5">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <div>
                    <h1 class="fw-bold text-dark m-0">{ticker_symbol} Live Analyse</h1>
                    <p class="text-muted m-0">Gegenereerd op {datetime.now().strftime('%d-%m-%Y %H:%M')}</p>
                </div>
                <span class="badge bg-dark px-3 py-2 fs-6">Data: yfinance</span>
            </div>

            <div class="row g-3 mb-4">
                <div class="col-md-4">
                    <div class="card p-3">
                        <div class="metric-title">Laatste Slotkoers</div>
                        <div class="metric-value">${laatste_prijs:,.2f}</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card p-3">
                        <div class="metric-title">24u Verandering</div>
                        <div class="metric-value text-{kleur_klaar}">
                            {teken}${absolute_verandering:,.2f} ({teken}{procentuele_verandering:.2f}%)
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card p-3">
                        <div class="metric-title">Gemiddeld Dagelijks Rendement</div>
                        <div class="metric-value">{(df['Returns_pct'].mean() * 100):.4f}%</div>
                    </div>
                </div>
            </div>

            <div class="card p-4 mb-4">
                <h5 class="fw-bold text-secondary mb-3">Interactieve Grafieken</h5>
                {grafiek_html}
            </div>
            
            <div class="card p-4">
                <h5 class="fw-bold text-secondary mb-3">Meest Recente Data (Laatste 5 dagen)</h5>
                <div class="table-responsive">
                    {df.tail(5).to_html(classes='table table-hover table-striped align-middle m-0', border=0, justify='left')}
                </div>
            </div>
        </div>

    </body>
    </html>
    """

    # Schrijf alles weg naar het HTML bestand
    with open(bestandsnaam, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✨ Prachtig dashboard succesvol opgeslagen als: '{bestandsnaam}'!")


