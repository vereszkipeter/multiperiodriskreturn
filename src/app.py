import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import h5py
import numpy as np
from streamlit_plotly_events import plotly_events

# --- ALAPBEÁLLÍTÁSOK ÉS ADATBETÖLTÉS ---

st.set_page_config(layout="wide", page_title="Portfólió Optimalizálás Dashboard")

# Az R szkriptben definiált ETF metaadatok, a konzisztencia miatt.
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
LOSS_HISTORY_COLS = ['Epoch', 'Teljes_Veszteseg', 'Kockazat_Cel', 'Forgas_Buntetes', 'Horgony_Buntetes', 'Vagyon_Buntetes', 'Also_Korlat_Buntetes', 'Havi_Kockazat_Buntetes']

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
                weights_data = np.array(group['weights']).T
                weights = pd.DataFrame(data=weights_data, columns=ETF_META['symbol'])
                loss_data = np.array(group['loss_history']).T
                loss_history = pd.DataFrame(data=loss_data, columns=LOSS_HISTORY_COLS)
                all_data[point_id] = {
                    'weights': weights, 'loss_history': loss_history,
                    'achieved_wealth': float(np.array(group['achieved_wealth'])),
                    'terminal_cCVaR': float(np.array(group['terminal_cCVaR']))
                }
        return all_data
    except FileNotFoundError:
        return None

# --- A MODELL FILOZÓFIÁJA (MAGYARÁZÓ SZEKCIÓ) ---
def display_methodology():
    st.header("A modell filozófiája és működése")
    with st.expander("Szcenárió generálás: a jövő lehetséges útjai"):
        st.markdown("""
        **Mit csinálunk?** Ahelyett, hogy egyetlen, pontszerű jövőbeli hozamot becsülnénk, **10,000 lehetséges jövőbeli forgatókönyvet** generálunk.
        
        **Miért fontos ez?** A pénzügyi piacok jövője bizonytalan. A pontbecslések (pl. "az S&P 500 jövőre 8%-ot hoz") rendkívül pontatlanok és veszélyesek. A szcenárió-alapú megközelítés elfogadja ezt a bizonytalanságot. Olyan portfóliót keresünk, amely a lehető legtöbb (mind a 10,000) forgatókönyvben jól teljesít, különös tekintettel a legrosszabb esetekre.
        
        **Hogyan csináljuk?** A historikus adatokból (2012-től) egy ún. **"block bootstrapping"** eljárással generáljuk a szcenáriókat. Ez megőrzi a piacok fontos tulajdonságait, mint például a volatilitási csomósodást (a válságok jellemzően nem egy, hanem több hónapig tartanak).
        """)

    with st.expander("Többperiódusos optimalizálás: a stratégia mint utazás"):
        st.markdown("""
        **Mit csinálunk?** A modell nem egyetlen, statikus portfóliót keres, hanem egy teljes, **60 hónapos (5 éves) befektetési pályát**, ami hónapról hónapra előírja az ideális súlyokat.
        
        **Miért fontos ez?** A befektetés egy folyamat, nem egyetlen döntés. A többperiódusos megközelítés lehetővé teszi, hogy olyan stratégiai tényezőket is figyelembe vegyünk, mint:
        - **Tranzakciós költségek:** Minden átalakításnak ára van.
        - **Újrabefektetési kockázat:** A hozamok időbeli sorrendje számít. Egy korai, nagy veszteséget sokkal nehezebb ledolgozni.
        - **Időben változó kényszerek:** A jövővel kapcsolatos bizonytalanságunkat beépíthetjük a modellbe.
        """)
        
    with st.expander("Kényszerek és büntetések (lambdák): a valóság korlátai"):
        st.markdown("""
        A modell számos, a valós életből vett kényszert kezel, amelyeket "büntetésekkel" (lambdákkal) érvényesít. A lambda egy adott szabály megszegésének "árát" jelenti.
        
        - **Havi kockázati sáv:** Ez egy kőbe vésett, szabályozói jellegű korlát. A havi kockázatnak (cCVaR) egy szűk sávon belül kell maradnia. Ez a legfontosabb kényszer.
        - **Átlagos forgási korlát:** Az átlagos havi portfólió-átalakítás mértéke korlátozott, elkerülve a túlzott kereskedést.
        - **Alsó súlykorlátok:** Bizonyos eszközökből (pl. SHY, GLD) egy minimális súlyt tartanunk kell a diverzifikáció érdekében.
        
        **Miért dinamikus és szigmoid?** A jövő bizonytalanabb, mint a jelen. Ezért a büntetések szigorúsága az idő előrehaladtával **dinamikusan enyhül**. Ezt egy **szigmoid ("S") görbével** modellezzük, ami egy stratégiailag megalapozott, sima átmenetet biztosít a kezdeti, szigorúbb szabályoktól a távoli jövő enyhébb korlátai felé.
        """)

    with st.expander('Centrális CVaR (cCVaR): a "fekete hattyúk" mérése'):
        st.markdown("""
        **Mit mérünk?** A kockázatot a hagyományos szórás helyett a **centrális feltételes várható alulmaradással (cCVaR)** mérjük.
        
        **Miért fontos ez?** A szórás a pozitív és negatív kilengéseket egyformán "bünteti", és feltételezi, hogy a hozamok eloszlása normális. A valóságban a befektetőket csak a negatív kilengések, a nagy veszteségek érdeklik, az eloszlásoknak pedig "vastag farkaik" vannak (a szélsőséges események gyakoribbak, mint gondolnánk).
        
        **Mit jelent a cCVaR?** A cCVaR azt méri, hogy a legrosszabb 5%-os forgatókönyvekben **átlagosan mennyivel marad el a vagyonunk a teljes eloszlás átlagától**. Ez egy sokkal jobb mutatója a katasztrofális, "fekete hattyú" jellegű kockázatoknak, mint a szórás. A "centrális" jelző arra utal, hogy a középértékhez viszonyítunk, így a mutató a kockázat "alakját", nem pedig a hozammal együtt mozgó "helyzetét" méri.
        """)
        
    with st.expander("Horgony portfóliók: segítség az optimalizálónak"):
        st.markdown("""
        **Mik ezek?** A horgony portfóliók előre definiált, stratégiailag értelmes portfóliók (pl. egy nagyon konzervatív vagy egy diverzifikált-agresszív), amelyeket "iránymutatásként" adunk az optimalizálónak.
        
        **Miért szükségesek?** A lehetséges portfólió-pályák végtelen univerzumában a horgonyok segítenek a modellnek megtalálni a hatékony front két kulcsfontosságú végpontját: a minimális kockázatú és a maximális hozamú stratégiákat. A mi modellünkben ezek:
        - **Min. kockázat horgony:** Kötvény- és aranytúlsúlyos (`60% SHY, 30% IEI, 10% GLD`).
        - **Max. hozam horgony:** Diverzifikált, de részvénytúlsúlyos, szabálykövető portfólió (`40% SPY, 20% IWM, stb.`).
        """)

# --- FŐ ALKALMAZÁS ---
st.title("Interaktív portfólió optimalizálási dashboard")
st.markdown("---")

display_methodology()
st.markdown("---")

HDF5_FILE = "data/optimization_results_v42_3.h5" 
all_data = load_data_from_hdf5(HDF5_FILE)

if not all_data:
    st.error(f"Hiba: A '{HDF5_FILE}' fájl nem található. Kérlek, futtasd le az R szkriptet az eredményfájl legenerálásához.")
else:
    if 'selected_portfolio_id' not in st.session_state:
        st.session_state.selected_portfolio_id = len(all_data)
    if 'selected_month' not in st.session_state:
        st.session_state.selected_month = None

    num_months = 60
    diagnostics_list = []
    for point_id, data in all_data.items():
        weights_df = data['weights']
        # Az elso havi forgast nullazzuk, összhangban az R modellel
        shifted_weights = weights_df.shift(1)
        shifted_weights.iloc[0] = weights_df.iloc[0]
        turnover = 0.5 * (weights_df - shifted_weights).abs().sum(axis=1)
        
        hhi = (weights_df**2).sum(axis=1)
        diagnostics_list.append({
            'Portfólió ID': point_id,
            'Évesített várható hozam': (data['achieved_wealth'])**(12 / num_months) - 1,
            'Évesített terminális kockázat (cCVaR)': data['terminal_cCVaR'] / np.sqrt(num_months / 12),
            'Átlagos havi forgás': turnover.mean(),
            'Átlagos koncentráció (HHI)': hhi.mean()
        })
    diagnostics_df = pd.DataFrame(diagnostics_list)

    st.header("1. A pragmatikus hatékony front elemzése")
    st.info("Kattints egy pontra az ábrán a portfólió kiválasztásához!")
    
    efficient_path_df = diagnostics_df.sort_values('Évesített terminális kockázat (cCVaR)').copy()
    efficient_path_df['max_hozam_eddig'] = efficient_path_df['Évesített várható hozam'].cummax()
    efficient_path_df = efficient_path_df[efficient_path_df['Évesített várható hozam'] >= efficient_path_df['max_hozam_eddig']]

    fig_frontier = px.scatter(
        diagnostics_df, x='Évesített terminális kockázat (cCVaR)', y='Évesített várható hozam',
        custom_data=['Portfólió ID'], title='Hatékony front: kockázat vs. hozam'
    )
    fig_frontier.update_traces(marker=dict(size=12, color='royalblue', opacity=0.8))
    fig_frontier.add_traces(go.Scatter(x=efficient_path_df['Évesített terminális kockázat (cCVaR)'], 
                                       y=efficient_path_df['Évesített várható hozam'], 
                                       mode='lines', line=dict(color='red', width=2), name='Hatékony Vonal'))
    fig_frontier.update_layout(xaxis_title="Évesített terminális kockázat (cCVaR)", yaxis_title="Évesített várható hozam", xaxis_tickformat=".2%", yaxis_tickformat=".2%")
    
    # JAVÍTÁS: Az ábrát a st.plotly_chart paranccsal kell megjeleníteni
    st.plotly_chart(fig_frontier, use_container_width=True)
    selected_points = plotly_events(fig_frontier, click_event=True)

    if selected_points:
        st.session_state.selected_portfolio_id = selected_points[0]['customdata'][0]
    
    st.sidebar.header("Portfólió kiválasztása")
    selected_id_sidebar = st.sidebar.selectbox(
        "Vagy válassz a listából:", options=diagnostics_df['Portfólió ID'],
        key='selectbox_selector', index=st.session_state.selected_portfolio_id - 1
    )
    if selected_id_sidebar != st.session_state.selected_portfolio_id:
        st.session_state.selected_portfolio_id = selected_id_sidebar

    st.markdown("---")
    
    selected_id = st.session_state.selected_portfolio_id
    st.header(f"Részletes elemzés: Portfólió P{selected_id}")

    selected_stats = diagnostics_df[diagnostics_df['Portfólió ID'] == selected_id].iloc[0]
    st.subheader("Fő stratégiai mutatók")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Évesített várható hozam", f"{selected_stats['Évesített várható hozam']:.2%}")
    col2.metric("Évesített terminális kockázat (cCVaR)", f"{selected_stats['Évesített terminális kockázat (cCVaR)']:.2%}")
    col3.metric("Átlagos havi forgás", f"{selected_stats['Átlagos havi forgás']:.2%}")
    col4.metric("Átlagos koncentráció (HHI)", f"{selected_stats['Átlagos koncentráció (HHI)']:.3f}")
    
    st.subheader("Eszközallokáció az időben")
    st.info("Kattints az allokációs ábra egy pontjára a havi részletek megtekintéséhez!")
    selected_data = all_data[selected_id]
    weights_df = selected_data['weights'].copy()
    weights_df['Hónap'] = range(1, len(weights_df) + 1)
    weights_long_df = weights_df.melt(id_vars='Hónap', var_name='symbol', value_name='Súly')
    weights_long_df = pd.merge(weights_long_df, ETF_META, on='symbol')
    
    fig_allocation = px.area(
        weights_long_df, x='Hónap', y='Súly', color='name',
        title=f"A(z) P{selected_id} portfólió havi eszközallokációja", labels={'name': 'Eszköz'}
    )
    fig_allocation.update_layout(yaxis_tickformat=".0%")
    
    # JAVÍTÁS: Az ábrát a st.plotly_chart paranccsal kell megjeleníteni
    st.plotly_chart(fig_allocation, use_container_width=True)
    selected_month_click = plotly_events(fig_allocation, click_event=True)
    
    if selected_month_click:
        st.session_state.selected_month = selected_month_click[0]['x']

    if st.session_state.selected_month:
        month = st.session_state.selected_month
        st.subheader(f"Részletek a(z) {month}. hónapra")
        monthly_weights = weights_df.loc[month - 1]
        col_pie, col_stats = st.columns([1, 1])
        with col_pie:
            pie_data = monthly_weights.drop('Hónap')
            fig_pie = px.pie(pie_data, values=pie_data.values, names=pie_data.index, title=f"Eszközallokáció a(z) {month}. hónapban", hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_stats:
            st.markdown(f"**Statisztikák a(z) {month}. hónapra:**")
            turnover_month = turnover[month-1] if month <= len(turnover) else 0
            hhi_month = (monthly_weights.drop('Hónap')**2).sum()
            st.metric("Forgás az előző hónaphoz képest", f"{turnover_month:.2%}")
            st.metric("Portfólió koncentráció (HHI)", f"{hhi_month:.3f}")
            st.warning("A havi várható hozam és cCVaR a 10,000 szcenárió alapján számított érték, melynek valós idejű kalkulációja itt nem lehetséges.")
    
    st.subheader("Optimalizáció diagnosztikája")
    loss_df = selected_data['loss_history'].copy()
    loss_long_df = loss_df.melt(id_vars='Epoch', var_name='Veszteség_Típus', value_name='Érték')
    fig_loss = px.line(
        loss_long_df, x='Epoch', y='Érték', color='Veszteség_Típus', facet_col='Veszteség_Típus', 
        facet_col_wrap=4, title=f"A(z) P{selected_id} portfólió veszteség-komponenseinek konvergenciája"
    )
    fig_loss.update_yaxes(matches=None, title_text="Veszteség értéke")
    fig_loss.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown("---")
    st.header("4. Modell értékelése és kitekintés")
    with st.expander("A modell erősségei és gyengeségei"):
        st.markdown("""**Erősségek:**\n- **Realizmus:** A modell számos, a valós életben is fontos tényezőt (költségek, korlátok, farokkockázatok) kezel.\n- **Stratégiai mélység:** A többperiódusos és dinamikus kényszer-kezelés lehetővé teszi valódi, időben konzisztens befektetési pályák tervezését.\n- **Robusztusság:** A szcenárió-alapú megközelítésnek köszönhetően a talált megoldások nem egyetlen, bizonytalan jövőképre vannak kihegyezve.\n\n**Gyengeségek:**\n- **Számítási igény:** A 10,000 szcenárió és a komplex veszteségfüggvény miatt a modell futtatása GPU-t igényel és időigényes.\n- **Paraméter-érzékenység:** Az eredmények, különösen a büntetési lambdák értékére érzékenyek.\n- **Historikus adatokon alapul:** A modell, mint minden kvantitatív stratégia, a múltbeli adatokból tanul.""")

    with st.expander("Különbségek a hagyományos modellektől"):
        st.markdown("""| Jellemző | Hagyományos (pl. Markowitz) | **Ez a modell** |\n| :--- | :--- | :--- |\n| **Időhorizont** | Egyperiódusos (statikus) | **Többperiódusos (dinamikus pálya)** |\n| **Kockázat** | Szórás (volatilitás) | **Centrális CVaR (farokkockázat)** |\n| **Bemenet** | Pontbecslések (E[R], Kovariancia) | **Teljes eloszlás (10,000 szcenárió)** |\n| **Kényszerek** | Egyszerű (pl. súlyösszeg) | **Komplex, dinamikus, nemlineáris** |\n| **Költségek** | Jellemzően figyelmen kívül hagyja | **Explicit módon kezeli** |""")
        
    with st.expander("Lehetséges fejlesztési irányok"):
        st.markdown("""- **Felső súlykorlátok:** A túlzott koncentráció elkerülése érdekében bevezethetnénk egy maximális súlyt minden eszközre.\n- **Faktor-alapú szcenáriók:** A historikus adatok helyett makrogazdasági faktorok alapján is generálhatnánk szcenáriókat.\n- **Tranzíciós pálya optimalizálása:** Egy meglévő portfólióból kiindulva is megkereshetnénk a legoptimálisabb átállási útvonalat.\n- **Gradiens-alapú lambda skálázás:** Az "okos" lambda-kalibrációs módszer implementálása a manuális hangolás kiváltására.""")