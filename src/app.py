import streamlit as st
import pandas as pd
import plotly.express as px
import h5py
import numpy as np

# --- ALAPBEÁLLÍTÁSOK ÉS ADATBETÖLTÉS ---

st.set_page_config(layout="wide", page_title="Többperiódusos portfólió optimalizálás")

ETF_META = pd.DataFrame({
    'symbol': ["SPY", "IWM", "DBEF", "DBEZ", "VNQ", "GLD", "DBA", "SHY", "IEI", "IEF", "TLT", "LQD", "HYG"],
    'name': [
        "SPDR S&P 500 ETF Trust (US Large Cap)", "iShares Russell 2000 ETF (US Small Cap)",
        "Xtrackers MSCI EAFE Hedged Equity ETF (Dev. Intl)", "Xtrackers MSCI Emerging Markets Hedged ETF (EM)",
        "Vanguard Real Estate ETF (US REIT)", "SPDR Gold Shares (Gold)",
        "Invesco DB Agriculture Fund (Agriculture)", "iShares 1-3 Year Treasury Bond ETF (Short Gov)",
        "iShares 3-7 Year Treasury Bond ETF (Short-Mid Gov)", "iShares 7-10 Year Treasury Bond ETF (Mid Gov)",
        "iShares 20+ Year Treasury Bond ETF (Long Gov)", "iShares iBoxx $ Inv. Grade Corp. Bond ETF (IG Corp)",
        "iShares iBoxx $ High Yield Corp. Bond ETF (HY Corp)"
    ]
})

LOSS_HISTORY_COLS = [
    'Epoch', 'Teljes_Veszteseg', 'Kockazat_Cel', 'Forgas_Buntetes', 
    'Horgony_Buntetes', 'Vagyon_Buntetes', 'Also_Korlat_Buntetes', 'Havi_Kockazat_Buntetes'
]

# ==============================================================================
# **** VÉGLEGES JAVÍTÁS ITT ****
# ==============================================================================
@st.cache_data
def load_data_from_hdf5(file_path):
    """Beolvassa az összes adatot a megadott HDF5 fájlból, explicit transzponálással."""
    all_data = {}
    try:
        with h5py.File(file_path, 'r') as f:
            point_keys = sorted([key for key in f.keys() if key.startswith('point_')])
            
            for key in point_keys:
                point_id = int(key.split('_')[-1])
                group = f[key]
                
                # Súlyok beolvasása és transzponálása:
                # A h5py (13, 60) alakban olvassa be, ezt transzponaljuk (60, 13)-ra.
                weights_data = np.array(group['weights']).T
                weights = pd.DataFrame(data=weights_data, columns=ETF_META['symbol'])
                
                # Loss history beolvasása es transzponálása:
                # Az R [20, 8] matrixot ment, a h5py (8, 20)-kent olvashatja.
                # A biztonsag kedveert ezt is transzponaljuk.
                loss_data = np.array(group['loss_history']).T
                loss_history = pd.DataFrame(data=loss_data, columns=LOSS_HISTORY_COLS)
                
                all_data[point_id] = {
                    'weights': weights,
                    'loss_history': loss_history,
                    'achieved_wealth': float(np.array(group['achieved_wealth'])),
                    'terminal_cCVaR': float(np.array(group['terminal_cCVaR']))
                }
        return all_data
    except FileNotFoundError:
        return None
# ==============================================================================
# **** JAVÍTÁS VÉGE ****
# ==============================================================================

# --- AZ ALKALMAZÁS TÖRZSE ---

st.title("Többperiódusos portfólió optimalizálás eredményei")

st.markdown("""
Ez az interaktív felület a többperiódusos portfólió optimalizáló modell eredményeit mutatja be. 
A modell egy 5 éves (60 hónapos) időhorizontra keres optimális befektetési stratégiákat, figyelembe véve számos
gyakorlati kényszert, mint például a tranzakciós költségek, dinamikus kényszerprofilok és a havi kockázat szigorú
kontrollja. Az alábbiakban a generált 10 pragmatikus portfólió részletes elemzése látható.
""")

HDF5_FILE = "data/optimization_results_v42_3.h5" 
all_data = load_data_from_hdf5(HDF5_FILE)

if not all_data:
    st.error(f"Hiba: A '{HDF5_FILE}' fájl nem található. Kérlek, futtasd le az R szkriptet az eredményfájl legenerálásához.")
else:
    diagnostics_df = pd.DataFrame([{
        'Portfólió ID': pid,
        'Várható végvagyon': data['achieved_wealth'],
        'Terminális cCVaR': data['terminal_cCVaR']
    } for pid, data in all_data.items()])
    
    num_months = 60
    diagnostics_df['Évesített várható hozam'] = (diagnostics_df['Várható végvagyon'])**(12 / num_months) - 1
    diagnostics_df['Évesített terminális kockázat (cCVaR)'] = diagnostics_df['Terminális cCVaR'] / np.sqrt(num_months / 12)

    st.header("1. A pragmatikus hatékony front")

    st.markdown("""
    Az ábra minden pontja egy-egy optimális portfóliót reprezentál a hozam-kockázat térben. A "kockázatot" a 
    terminális vagyon centrális CVaR-ja (cCVaR) méri, ami a legrosszabb 5%-os forgatókönyv várható veszteségét mutatja meg a 
    teljes eloszlás átlagához képest. A piros vonal a szigorúan *hatékony* portfóliókat köti össze.
    """)
    
    efficient_path_df = diagnostics_df.sort_values('Évesített terminális kockázat (cCVaR)').copy()
    efficient_path_df['max_hozam_eddig'] = efficient_path_df['Évesített várható hozam'].cummax()
    efficient_path_df = efficient_path_df[efficient_path_df['Évesített várható hozam'] >= efficient_path_df['max_hozam_eddig']]

    fig_frontier = px.scatter(
        diagnostics_df, x='Évesített terminális kockázat (cCVaR)', y='Évesített várható hozam',
        text='Portfólió ID', title='Hatékony front: kockázat vs. hozam'
    )
    fig_frontier.update_traces(textposition='top center', marker=dict(size=12, color='red', opacity=0.7))
    fig_frontier.add_traces(px.line(efficient_path_df, x='Évesített terminális kockázat (cCVaR)', y='Évesített várható hozam').data)
    fig_frontier.update_layout(xaxis_title="Évesített terminális kockázat (cCVaR)", yaxis_title="Évesített várható hozam", xaxis_tickformat=".2%", yaxis_tickformat=".2%")
    st.plotly_chart(fig_frontier, use_container_width=True)
    
    st.dataframe(diagnostics_df[['Portfólió ID', 'Évesített várható hozam', 'Évesített terminális kockázat (cCVaR)', 'Várható végvagyon']].style.format({
        'Évesített várható hozam': '{:.2%}',
        'Évesített terminális kockázat (cCVaR)': '{:.2%}',
        'Várható végvagyon': '{:.3f}'
    }))

    st.sidebar.header("Portfólió kiválasztása")
    selected_id = st.sidebar.selectbox("Válassz egy portfóliót a részletes elemzéshez:", options=diagnostics_df['Portfólió ID'], index=len(diagnostics_df) - 1)
    
    selected_data = all_data[selected_id]

    st.header(f"2. Portfólió allokáció: P{selected_id}")
    st.markdown(f"Az alábbi ábra a(z) **{selected_id}. számú portfólió** eszközallokációjának időbeli alakulását mutatja a 60 hónapos befektetési horizonton.")
    
    weights_df = selected_data['weights'].copy()
    weights_df['Hónap'] = range(1, len(weights_df) + 1)
    weights_long_df = weights_df.melt(id_vars='Hónap', var_name='symbol', value_name='Súly')
    weights_long_df = pd.merge(weights_long_df, ETF_META, on='symbol')
    
    fig_allocation = px.area(
        weights_long_df, x='Hónap', y='Súly', color='name',
        title=f"A(z) P{selected_id} portfólió havi eszközallokációja", labels={'name': 'Eszköz'}
    )
    fig_allocation.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_allocation, use_container_width=True)

    st.header(f"3. Optimalizáció diagnosztikája: P{selected_id}")
    st.markdown("Ezek az ábrák a 'motorháztető alá' engednek bepillantást, bemutatva a veszteségfüggvény és komponenseinek konvergenciáját az optimalizáció során.")

    loss_df = selected_data['loss_history'].copy()
    loss_long_df = loss_df.melt(id_vars='Epoch', var_name='Veszteség_Típus', value_name='Érték')
    
    fig_loss = px.line(
        loss_long_df, x='Epoch', y='Érték', color='Veszteség_Típus',
        facet_col='Veszteség_Típus', facet_col_wrap=4,
        title=f"A(z) P{selected_id} portfólió veszteség-komponenseinek konvergenciája"
    )
    fig_loss.update_yaxes(matches=None, title_text="Veszteség értéke")
    fig_loss.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    st.plotly_chart(fig_loss, use_container_width=True)