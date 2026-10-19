"""
Painel de operação da rede — protótipo em Streamlit + pandapower

Como executar:
    pip install streamlit pandapower plotly pandas numpy networkx
    streamlit run app.py

Estrutura do arquivo
    1. Rede         -> rede_base(): usa pn.case14(). Para outra rede, troque só esta função.
    2. Cálculo      -> rodar_fluxo_24h(), rodar_n1(), varredura_n1(): usam só pandapower, sem Streamlit
                       na lógica. Se o seu código de fluxo 24h/N-1 já existe, troque o corpo destas
                       funções mantendo o formato do retorno.
    3. Gráficos     -> figura_rede() e demais gráficos Plotly.
    4. Telas        -> painel principal e as duas telas de parâmetros.
"""

import copy
import json
from datetime import datetime

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Painel da rede", layout="wide")

# Versões novas do Streamlit trocaram use_container_width por width="stretch".
_NOVO = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50)
LARGURA = {"width": "stretch"} if _NOVO else {"use_container_width": True}

COR = {
    "ok": "#2E8B57",
    "atencao": "#D9A400",
    "alto": "#E8710A",
    "critico": "#C62828",
    "off": "#8A8F98",
    "barra": "#1F3A5F",
}

# Fator de carga por hora (multiplica o fator global dos parâmetros).
# Perfil médio real do subsistema Norte, calculado a partir da Curva de Carga Horária do
# ONS (dados abertos, licença CC-BY, https://dados.ons.org.br/dataset/curva-carga),
# arquivo CURVA_CARGA_2026.csv, médio de 01/01/2026 a 29/09/2026 (272 dias, 6528 horas).
# Normalizado para que a média das 10h/11h valha 1,00 (mesma convenção usada na validação
# do caso-base). Diferente de uma curva urbana típica, o Norte tem perfil mais achatado
# (vale ~0,94 às 7h, pico ~1,10 às 22h), refletindo a carga industrial de base da região.
CURVA_PADRAO = [1.071, 1.047, 1.023, 1.000, 0.983, 0.971, 0.941, 0.937, 0.963, 0.981, 0.996,
                1.004, 1.003, 1.030, 1.055, 1.058, 1.045, 1.024, 1.028, 1.057, 1.070, 1.086,
                1.097, 1.091]


# ======================================================================================
# 1. REDE  (troque rede_base() se quiser usar outra rede pandapower)
# ======================================================================================
POS14 = {0: (0.0, 3.6), 1: (2.5, 5.0), 2: (6.0, 5.0), 3: (5.0, 3.4), 4: (2.4, 3.0),
         5: (2.0, 1.7), 6: (7.0, 2.4), 7: (9.0, 2.4), 8: (5.6, 1.6), 9: (5.0, 0.3),
         10: (3.2, 0.4), 11: (0.2, 0.5), 12: (1.6, -0.7), 13: (4.4, -0.8)}

# A case14 vem com max_i_ka (linhas) e sn_mva (transformadores) enormes: o carregamento sai em ~1%
# e nenhuma sobrecarga apareceria, nem no N-1. Com True, o painel define capacidades PROVISÓRIAS:
# cada ramo fica em 55% no pico do dia (mínimo de 25 MVA). Isso não altera nenhum dado elétrico
# da rede, só a régua usada para calcular o carregamento em %.
# Se a sua rede tiver capacidades reais, coloque False e o painel usa o loading_percent do pandapower.
CALIBRAR_CAPACIDADE = True
FOLGA_PICO = 0.55
S_MIN_MVA = 25.0


def _s_ramos(net):
    """Potência aparente (MVA) de cada ramo: linhas primeiro, depois transformadores."""
    rl, rt = net.res_line, net.res_trafo
    s_l = np.maximum(np.hypot(rl.p_from_mw, rl.q_from_mvar), np.hypot(rl.p_to_mw, rl.q_to_mvar))
    s_t = np.maximum(np.hypot(rt.p_hv_mw, rt.q_hv_mvar), np.hypot(rt.p_lv_mw, rt.q_lv_mvar))
    return np.concatenate([np.asarray(s_l, dtype=float), np.asarray(s_t, dtype=float)])


def definir_capacidades(net):
    f = max(CURVA_PADRAO)
    net.load["scaling"] = f
    net.gen["scaling"] = f
    pp.runpp(net)
    cap = np.maximum(_s_ramos(net) / FOLGA_PICO, S_MIN_MVA)
    nl = len(net.line)
    net.line["cap_mva"] = cap[:nl]
    net.trafo["cap_mva"] = cap[nl:]
    net.load["scaling"] = 1.0
    net.gen["scaling"] = 1.0


@st.cache_resource
def rede_base():
    net = pn.case14()
    if CALIBRAR_CAPACIDADE:
        definir_capacidades(net)
    return net


@st.cache_resource
def info_rede():
    """Dados estáticos usados pela interface. "Ramo" = linha ou transformador (linhas vêm primeiro)."""
    net = rede_base()
    nb, nl, nt = len(net.bus), len(net.line), len(net.trafo)
    idx = net.bus.index
    de = np.concatenate([idx.get_indexer(net.line.from_bus), idx.get_indexer(net.trafo.hv_bus)])
    para = np.concatenate([idx.get_indexer(net.line.to_bus), idx.get_indexer(net.trafo.lv_bus)])
    nomes_barra = [f"B{i + 1}" for i in range(nb)]
    if nb == 14 and nl + nt == 20:
        pos = POS14
    else:  # posições automáticas para qualquer outra rede
        g = nx.Graph()
        g.add_nodes_from(range(nb))
        g.add_edges_from(zip(de.tolist(), para.tolist()))
        pos = {k: tuple(v) for k, v in nx.spring_layout(g, seed=7).items()}
    curtos = [f"L{k + 1}" for k in range(nl)] + [f"T{k + 1}" for k in range(nt)]
    nomes_ramo = [f"{curtos[k]} ({nomes_barra[de[k]]}–{nomes_barra[para[k]]})" for k in range(nl + nt)]
    barras_gen = set(idx.get_indexer(net.gen.bus).tolist()) | set(idx.get_indexer(net.ext_grid.bus).tolist())
    return {
        "pos": pos, "de": de, "para": para, "n_linhas": nl,
        "nomes_barra": nomes_barra, "nomes_linha": nomes_ramo, "curtos": curtos,
        "e_trafo": [k >= nl for k in range(nl + nt)],
        "barras_gen": barras_gen,
        "nomes_ger": ["Slack"] + [f"G{i + 1} ({nomes_barra[b]})" for i, b in enumerate(idx.get_indexer(net.gen.bus))],
        "gens0": [_limites_gerador(net, b, r) for b, r in zip(idx.get_indexer(net.gen.bus), net.gen.itertuples())],
        "n_shunts": len(net.shunt),
        "nomes_shunt": [f"Shunt {nomes_barra[b]}" for b in idx.get_indexer(net.shunt.bus)] if len(net.shunt) else [],
    }


def _limites_gerador(net, barra_idx, linha_gen):
    """Lê Pmin/Pmax reais do pandapower; se ausentes/infinitos, usa uma folga conservadora
    e marca 'limite_estimado' para a interface avisar que não é um dado oficial do caso."""
    pmin = getattr(linha_gen, "min_p_mw", 0.0)
    pmax = getattr(linha_gen, "max_p_mw", np.nan)
    pmin = 0.0 if pmin is None or not np.isfinite(pmin) else float(pmin)
    estimado = pmax is None or not np.isfinite(pmax) or pmax <= 0
    if estimado:
        pmax = max(2.0 * float(linha_gen.p_mw), 50.0)
    return {"barra": f"B{barra_idx + 1}", "p_mw": float(linha_gen.p_mw), "vm_pu": float(linha_gen.vm_pu),
            "pmin": pmin, "pmax": float(pmax), "limite_estimado": bool(estimado)}


def padrao():
    info = info_rede()
    return {
        "fator": 1.0, "curva": list(CURVA_PADRAO),
        "vmin": 0.95, "vmax": 1.10, "lim_carr": 100.0,
        "gens": [{"em_servico": True, "p_mw": g["p_mw"], "vm_pu": g["vm_pu"]} for g in info["gens0"]],
        "shunts": [True] * info["n_shunts"],
        "n1_ativa": True, "n1_linha": 0, "n1_hora": -2,  # -2 = hora do painel, -1 = hora de pico
        "n1_ranking": True, "n1_lim": 110.0,
    }


# ======================================================================================
# 2. CÁLCULO  (troque pelo seu código de fluxo 24h / N-1 mantendo o formato do retorno)
# ======================================================================================
def _ajustar_hora(net, p, h):
    f = p["fator"] * p["curva"][h]
    net.load["scaling"] = f
    net.gen["scaling"] = f


def _montar(p, hora):
    net = copy.deepcopy(rede_base())
    for i, g in enumerate(p["gens"]):
        gi = net.gen.index[i]
        net.gen.at[gi, "in_service"] = bool(g["em_servico"])
        net.gen.at[gi, "p_mw"] = float(g["p_mw"])
        net.gen.at[gi, "vm_pu"] = float(g["vm_pu"])
    for i, on in enumerate(p.get("shunts", [])):
        if i < len(net.shunt):
            net.shunt.at[net.shunt.index[i], "in_service"] = bool(on)
    _ajustar_hora(net, p, hora)
    return net


@st.cache_data(show_spinner=False)
def rodar_hora(pj, hora):
    """Fluxo de potência de uma única hora, com os parâmetros atuais (inclui ações corretivas
    de geração/capacitor). Usado pelo bloco de laboratório para o retrato 'antes' e 'depois'."""
    p = json.loads(pj)
    net = _montar(p, hora)
    ok = _rodar(net)
    return _extrair(net, ok)


def _rodar(net):
    try:
        pp.runpp(net)
        return True
    except Exception:  # LoadflowNotConverged ou rede sem referência
        return False


def _carregamento(net):
    """Carregamento (%) de cada ramo: linhas primeiro, depois transformadores."""
    if CALIBRAR_CAPACIDADE:
        cap = np.concatenate([net.line["cap_mva"].to_numpy(dtype=float), net.trafo["cap_mva"].to_numpy(dtype=float)])
        return 100.0 * _s_ramos(net) / cap
    return np.concatenate([np.asarray(net.res_line.loading_percent, dtype=float),
                           np.asarray(net.res_trafo.loading_percent, dtype=float)])


def _perdas(net):
    return float(np.nansum(net.res_line.pl_mw) + np.nansum(net.res_trafo.pl_mw))


def _extrair(net, ok):
    nb, nl = len(net.bus), len(net.line) + len(net.trafo)
    if not ok:
        return {"ok": False, "vm": np.full(nb, np.nan), "carreg": np.full(nl, np.nan),
                "p_from": np.full(nl, np.nan)}
    return {"ok": True,
            "vm": np.array(net.res_bus.vm_pu, dtype=float),
            "carreg": _carregamento(net),
            "p_from": np.concatenate([np.asarray(net.res_line.p_from_mw, dtype=float),
                                      np.asarray(net.res_trafo.p_hv_mw, dtype=float)])}


@st.cache_data(show_spinner="Calculando fluxo de 24 h...")
def rodar_fluxo_24h(pj):
    p = json.loads(pj)
    net = _montar(p, 0)
    nb, nl, ng = len(net.bus), len(net.line) + len(net.trafo), len(net.gen)
    res = {
        "conv": np.zeros(24, dtype=bool), "vm": np.full((24, nb), np.nan),
        "carreg": np.full((24, nl), np.nan), "p_from": np.full((24, nl), np.nan),
        "carga": np.zeros(24), "ger": np.full(24, np.nan), "perdas": np.full(24, np.nan),
        "ger_unid": np.full((24, ng + 1), np.nan), "viol_h": np.zeros(24, dtype=bool),
        "nomes_ger": info_rede()["nomes_ger"],
        "executado_em": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    }
    for h in range(24):
        _ajustar_hora(net, p, h)
        res["carga"][h] = float((net.load.p_mw * net.load.scaling)[net.load.in_service].sum())
        ok = _rodar(net)
        res["conv"][h] = ok
        if not ok:
            res["viol_h"][h] = True
            continue
        d = _extrair(net, True)
        res["vm"][h], res["carreg"][h], res["p_from"][h] = d["vm"], d["carreg"], d["p_from"]
        res["carga"][h] = float(net.res_load.p_mw.sum())
        unid = [float(net.res_ext_grid.p_mw.sum())] + list(np.nan_to_num(net.res_gen.p_mw.to_numpy(dtype=float)))
        res["ger_unid"][h] = unid
        res["ger"][h] = float(np.sum(unid))
        res["perdas"][h] = _perdas(net)
        viol = np.sum((_arred(d["vm"]) < p["vmin"]) | (_arred(d["vm"]) > p["vmax"])) + np.sum(d["carreg"] > p["lim_carr"])
        res["viol_h"][h] = viol > 0
    return res


@st.cache_data(show_spinner="Calculando contingência N-1...")
def rodar_n1(pj, linha, hora):
    p = json.loads(pj)
    net = _montar(p, hora)
    antes = _extrair(net, _rodar(net))
    nl = len(net.line)
    if linha < nl:
        net.line.at[net.line.index[linha], "in_service"] = False
    else:
        net.trafo.at[net.trafo.index[linha - nl], "in_service"] = False
    ok = _rodar(net)
    depois = _extrair(net, ok)
    ilhadas, perdida = [], 0.0
    if ok:
        depois["carreg"][linha] = np.nan
        depois["p_from"][linha] = 0.0
        ilhadas = np.where(np.isnan(depois["vm"]))[0].tolist()
        mask = net.load.bus.isin(net.bus.index[ilhadas])
        perdida = float((net.load.p_mw[mask] * net.load.scaling[mask]).sum())
    return {"antes": antes, "depois": depois, "ok": ok, "ilhadas": ilhadas, "carga_perdida": perdida}


def severidade(vm, carreg, ok, p):
    """Nota de severidade: 0 = sem problema. Não convergência = 100."""
    if not ok:
        return 100.0
    s = np.nansum(np.clip(carreg - p["n1_lim"], 0, None)) / 100.0
    s += np.nansum(np.clip(p["vmin"] - vm, 0, None) + np.clip(vm - p["vmax"], 0, None)) * 10.0
    s += 5.0 * np.isnan(vm).sum()
    return float(s)


@st.cache_data(show_spinner="Varrendo todos os ramos (N-1)...")
def varredura_n1(pj, hora):
    p = json.loads(pj)
    info = info_rede()
    linhas = []
    for k in range(len(info["de"])):
        r = rodar_n1(pj, k, hora)
        d = r["depois"]
        linhas.append({
            "Ramo retirado": info["nomes_linha"][k],
            "Convergiu": "sim" if r["ok"] else "não",
            "Barras ilhadas": len(r["ilhadas"]),
            "Sobrecargas": int(np.sum(d["carreg"] > p["n1_lim"])) if r["ok"] else 0,
            "Tensões fora": int(np.sum((d["vm"] < p["vmin"]) | (d["vm"] > p["vmax"]))) if r["ok"] else 0,
            "Carreg. máx. (%)": round(float(np.nanmax(d["carreg"])), 1) if r["ok"] else np.nan,
            "V mín. (pu)": round(float(np.nanmin(d["vm"])), 3) if r["ok"] else np.nan,
            "Severidade": round(severidade(d["vm"], d["carreg"], r["ok"], p), 2),
        })
    return pd.DataFrame(linhas).sort_values("Severidade", ascending=False).reset_index(drop=True)


# ======================================================================================
# 3. GRÁFICOS E COMPONENTES VISUAIS
# ======================================================================================
def hora_txt(h):
    return f"{int(h):02d}:00"


def fmt(v, nd=1, un=""):
    return "n/d" if v is None or np.isnan(v) else f"{v:.{nd}f}{un}"


def cor_carreg(c, lim):
    if c is None or np.isnan(c):
        return COR["off"]
    r = c / lim
    if r >= 1:
        return COR["critico"]
    if r >= 0.9:
        return COR["alto"]
    if r >= 0.7:
        return COR["atencao"]
    return COR["ok"]


def pill(texto, nivel):
    cor = COR[nivel]
    return (f"<span style='background:{cor}22;color:{cor};border:1px solid {cor};"
            f"padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600'>{texto}</span>")


def faixa(nivel, titulo, detalhe):
    cor = COR[nivel]
    st.markdown(
        f"<div style='border-left:8px solid {cor};background:{cor}1A;padding:10px 16px;border-radius:4px'>"
        f"<b style='color:{cor};font-size:1.1rem'>{titulo}</b>&nbsp;&nbsp;{detalhe}</div>",
        unsafe_allow_html=True)


def figura_rede(d, P, lim, linha_fora=None, altura=400):
    """Diagrama unifilar: cor/espessura da linha = carregamento, seta = sentido do fluxo,
    cor da barra = tensão dentro/fora dos limites."""
    info = info_rede()
    pos, de, para = info["pos"], info["de"], info["para"]
    vm, carreg, p_from = d["vm"], d["carreg"], d["p_from"]
    fig = go.Figure()
    mx, my, mt = [], [], []
    for k in range(len(de)):
        x0, y0 = pos[int(de[k])]
        x1, y1 = pos[int(para[k])]
        mx.append((x0 + x1) / 2)
        my.append((y0 + y1) / 2)
        if linha_fora is not None and k == linha_fora:
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip",
                                     line=dict(color=COR["off"], width=2, dash="dash")))
            mt.append(f"{info['nomes_linha'][k]}<br><b>RETIRADA (N-1)</b>")
            continue
        c, pf = carreg[k], p_from[k]
        cor = cor_carreg(c, lim)
        larg = 2 + 4 * min(max(0.0 if np.isnan(c) else c, 0.0), 100.0) / 100.0
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip",
                                 line=dict(color=cor, width=larg)))
        mt.append(f"{info['nomes_linha'][k]}<br>Carregamento: {fmt(c, 0, '%')}<br>P: {fmt(pf, 1, ' MW')}")
        if not np.isnan(pf) and abs(pf) > 1:
            comp = np.hypot(x1 - x0, y1 - y0)
            ux, uy = (x1 - x0) / comp * np.sign(pf), (y1 - y0) / comp * np.sign(pf)
            fig.add_annotation(x=mx[-1] + ux * 0.18, y=my[-1] + uy * 0.18,
                               ax=mx[-1] - ux * 0.18, ay=my[-1] - uy * 0.18,
                               xref="x", yref="y", axref="x", ayref="y", text="",
                               showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=1.6, arrowcolor=cor)
    e_t = info["e_trafo"]
    ml = [k for k in range(len(mx)) if not e_t[k]]
    mt_idx = [k for k in range(len(mx)) if e_t[k]]
    fig.add_trace(go.Scatter(x=[mx[k] for k in ml], y=[my[k] for k in ml], mode="markers",
                             marker=dict(size=14, opacity=0), hovertext=[mt[k] for k in ml], hoverinfo="text"))
    if mt_idx:  # transformadores: losango no meio do ramo
        fig.add_trace(go.Scatter(x=[mx[k] for k in mt_idx], y=[my[k] for k in mt_idx], mode="markers",
                                 marker=dict(size=11, symbol="diamond", color="#7A4EAB",
                                             line=dict(width=1.5, color="white")),
                                 hovertext=[mt[k] for k in mt_idx], hoverinfo="text"))

    ilh = set(int(i) for i in np.where(np.isnan(vm))[0])
    cores, simbolos, ht = [], [], []
    for i, v in enumerate(vm):
        if i in ilh:
            cores.append(COR["off"])
        elif v < P["vmin"] or v > P["vmax"]:
            cores.append(COR["critico"])
        else:
            cores.append(COR["barra"])
        simbolos.append("square" if i in info["barras_gen"] else "circle")
        ht.append(f"{info['nomes_barra'][i]}<br>Tensão: " + ("sem tensão (ilhada)" if i in ilh else f"{v:.3f} pu"))
    fig.add_trace(go.Scatter(
        x=[pos[i][0] for i in range(len(vm))], y=[pos[i][1] for i in range(len(vm))],
        mode="markers+text", text=[n for n in info["nomes_barra"]], textposition="middle center",
        textfont=dict(color="white", size=10), hovertext=ht, hoverinfo="text",
        marker=dict(size=32, color=cores, symbol=simbolos, line=dict(width=2, color="white"))))
    fig.update_layout(height=altura, margin=dict(l=5, r=5, t=5, b=5), showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


LEGENDA_REDE = ("Quadrado = barra com geração · círculo = barra de carga · vermelho = tensão fora do limite · "
                "cor e espessura do ramo = carregamento (verde, amarelo, laranja, vermelho) · seta = sentido do fluxo · losango roxo = transformador.")


def grafico(fig):
    st.plotly_chart(fig, **LARGURA)


def tabela(df):
    st.dataframe(df, hide_index=True, **LARGURA)


def eixo_horas(fig):
    fig.update_xaxes(tickmode="array", tickvals=list(range(0, 24, 3)),
                     ticktext=[hora_txt(h) for h in range(0, 24, 3)])
    return fig


# ======================================================================================
# 4. TELAS
# ======================================================================================
def ir(tela):
    st.session_state["tela"] = tela


PRECISAO_TELA = 3  # casas decimais mostradas para tensão (pu) — ver _arred()


def _arred(valores):
    """Arredonda na mesma precisão exibida na tela, para que o status NUNCA discorde do
    número que a pessoa está vendo. Sem isso, um valor como 1,08999... ou 1,09001... (ambos
    mostrados como "1.090") poderia cair de um lado ou do outro do limite por puro ruído
    numérico do solver, fazendo o Alerta "piscar" entre horas sem nenhuma mudança real."""
    return np.round(valores, PRECISAO_TELA)


def avaliar_hora(res, h, P):
    """Devolve (nível, título, detalhe) para a faixa de status."""
    n_h = int(res["viol_h"].sum())
    if not res["conv"][h]:
        return "critico", "Sem convergência", f"{hora_txt(h)} · o fluxo de potência não convergiu."
    vm, ld = _arred(res["vm"][h]), res["carreg"][h]
    n_v = int(np.sum((vm < P["vmin"]) | (vm > P["vmax"])))
    n_l = int(np.sum(ld > P["lim_carr"]))
    resumo = f"{hora_txt(h)} · {n_h} de 24 horas com violação"
    if n_v + n_l > 0:
        return "critico", "Crítico", f"{n_l} ramo(s) sobrecarregado(s), {n_v} barra(s) fora da faixa de tensão · {resumo}"
    if np.nanmax(ld) > 0.8 * P["lim_carr"] or np.nanmin(vm) < P["vmin"] + 0.01 or np.nanmax(vm) > P["vmax"] - 0.01:
        return "atencao", "Alerta", f"Pouca folga em algum limite · {resumo}"
    return "ok", "Normal", f"Todos os limites respeitados · {resumo}"


def tabela_violacoes(r, P):
    info = info_rede()
    a, d = r["antes"], r["depois"]
    linhas = []
    for i in r["ilhadas"]:
        linhas.append([info["nomes_barra"][i], "Ilhamento", f"{a['vm'][i]:.3f} pu", "sem tensão", "—"])
    for k in np.where(d["carreg"] > P["n1_lim"])[0]:
        linhas.append([info["nomes_linha"][k], "Sobrecarga", fmt(a["carreg"][k], 0, "%"),
                       fmt(d["carreg"][k], 0, "%"), f"{P['n1_lim']:.0f}%"])
    for i in range(len(d["vm"])):
        v = d["vm"][i]
        if np.isnan(v):
            continue
        if v < P["vmin"]:
            linhas.append([info["nomes_barra"][i], "Subtensão", f"{a['vm'][i]:.3f} pu", f"{v:.3f} pu", f"{P['vmin']:.2f} pu"])
        elif v > P["vmax"]:
            linhas.append([info["nomes_barra"][i], "Sobretensão", f"{a['vm'][i]:.3f} pu", f"{v:.3f} pu", f"{P['vmax']:.2f} pu"])
    return pd.DataFrame(linhas, columns=["Elemento", "Tipo", "Antes", "Depois", "Limite"])


def bloco_fluxo(res, hora, P, limite=23):
    info = info_rede()
    completo = limite >= 23
    n_h = int(res["viol_h"][:limite + 1].sum())
    if completo:
        badge = pill("sem violações nas 24 h", "ok") if n_h == 0 else pill(f"{n_h} h com violação", "critico")
    else:
        badge = pill(f"simulando... {limite + 1}/24 h", "atencao" if n_h else "off")
    st.markdown(f"#### Fluxo 24 h &nbsp; {badge}", unsafe_allow_html=True)
    horas = list(range(limite + 1))
    t1, t2, t3, t4 = st.tabs(["Rede", "Carga × geração", "Carregamento dos ramos", "Tensões"])

    with t1:
        d = {"vm": res["vm"][hora], "carreg": res["carreg"][hora], "p_from": res["p_from"][hora]}
        st.caption(f"Estado da rede às {hora_txt(hora)}.")
        grafico(figura_rede(d, P, P["lim_carr"]))
        st.caption(LEGENDA_REDE)
        if res["conv"][hora]:
            top = pd.DataFrame({"Ramo": info["nomes_linha"], "Carregamento (%)": d["carreg"].round(1),
                                "Fluxo (MW)": d["p_from"].round(1)}).sort_values("Carregamento (%)", ascending=False).head(5)
            st.markdown("**Ramos mais carregados nesta hora**")
            tabela(top)

    with t2:
        fig = go.Figure()
        for j, nome in enumerate(res["nomes_ger"]):
            fig.add_trace(go.Scatter(x=horas, y=res["ger_unid"][:limite + 1, j], name=nome, stackgroup="g",
                                     mode="lines", line=dict(width=0.5)))
        fig.add_trace(go.Scatter(x=horas, y=res["carga"][:limite + 1], name="Carga", mode="lines",
                                 line=dict(color=COR["critico"], width=3, dash="dot")))
        fig.add_vline(x=hora, line_dash="dash", line_color=COR["off"])
        if not completo:
            fig.update_xaxes(range=[-0.5, 23.5])  # eixo fixo em 24h, mesmo mostrando só parte da curva
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="MW",
                          legend=dict(orientation="h", y=-0.2))
        grafico(eixo_horas(fig))
        p1, p2 = st.columns(2)
        carga_vista = res["carga"][:limite + 1]
        p1.metric("Hora de pico" if completo else "Hora de pico (até agora)",
                 hora_txt(int(np.argmax(carga_vista))), f"{np.max(carga_vista):.0f} MW", delta_color="off")
        p2.metric("Perdas no dia" if completo else "Perdas acumuladas (até agora)",
                 f"{np.nansum(res['perdas'][:limite + 1]):.0f} MWh")

    with t3:
        lim = P["lim_carr"]
        zmax = lim * 1.2
        escala = [[0, COR["ok"]], [0.5, COR["ok"]], [0.62, COR["atencao"]], [0.75, COR["alto"]],
                  [0.83, COR["critico"]], [1, "#7B1010"]]
        fig = go.Figure(go.Heatmap(z=res["carreg"][:limite + 1].T, x=horas, y=info["nomes_linha"], zmin=0, zmax=zmax,
                                   colorscale=escala, colorbar=dict(title="%"),
                                   hovertemplate="%{y}<br>%{x}h: %{z:.0f}%<extra></extra>"))
        fig.add_vline(x=hora, line_dash="dash", line_color="white")
        if not completo:
            fig.update_xaxes(range=[-0.5, 23.5])
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        grafico(eixo_horas(fig))
        st.caption("Mostra qual ramo aperta e em que horário. Vermelho = acima do limite de carregamento.")

    with t4:
        vm_vista = res["vm"][:limite + 1]
        fig = go.Figure()
        vmin_h = np.nanmin(np.where(np.isnan(vm_vista), np.inf, vm_vista), axis=1)
        vmax_h = np.nanmax(np.where(np.isnan(vm_vista), -np.inf, vm_vista), axis=1)
        vmin_h[np.isinf(vmin_h)] = np.nan
        vmax_h[np.isinf(vmax_h)] = np.nan
        fig.add_trace(go.Scatter(x=horas, y=vmax_h, name="Máxima", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=horas, y=vmin_h, name="Mínima", mode="lines+markers"))
        fig.add_hline(y=P["vmax"], line_dash="dash", line_color=COR["critico"])
        fig.add_hline(y=P["vmin"], line_dash="dash", line_color=COR["critico"])
        fig.add_vline(x=hora, line_dash="dash", line_color=COR["off"])
        if not completo:
            fig.update_xaxes(range=[-0.5, 23.5])
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="pu",
                          legend=dict(orientation="h", y=-0.2))
        grafico(eixo_horas(fig))

    st.button("Parâmetros da rede", key="b_par_rede", on_click=ir, args=("params_rede",))


def bloco_n1(res, hora, pico, pj, P):
    info = info_rede()
    if not P["n1_ativa"]:
        st.markdown(f"#### Contingência N-1 &nbsp; {pill('desativada', 'off')}", unsafe_allow_html=True)
        st.info("A contingência N-1 não está em uso. Ative e escolha o ramo em Parâmetros do N-1.")
        st.button("Parâmetros do N-1", key="b_par_n1", on_click=ir, args=("params_n1",))
        return

    code = P["n1_hora"]
    h = hora if code == -2 else pico if code == -1 else int(code)
    linha = int(P["n1_linha"])
    r = rodar_n1(pj, linha, h)
    d = r["depois"]

    if not r["ok"]:
        nivel, veredito = "critico", "não convergiu"
    else:
        tab = tabela_violacoes(r, P)
        if len(tab) > 0:
            nivel, veredito = "critico", f"NÃO seguro · {len(tab)} violação(ões)"
        elif np.nanmax(d["carreg"]) > 0.9 * P["n1_lim"]:
            nivel, veredito = "atencao", "seguro, com pouca folga"
        else:
            nivel, veredito = "ok", "N-1 seguro"
    st.markdown(f"#### Contingência N-1 &nbsp; {pill(veredito, nivel)}", unsafe_allow_html=True)
    st.caption(f"Ramo retirado: {info['nomes_linha'][linha]} · hora analisada: {hora_txt(h)}")

    abas = ["Rede", "Violações", "Antes × depois"] + (["Ranking"] if P["n1_ranking"] else [])
    tabs = st.tabs(abas)

    with tabs[0]:
        if not r["ok"]:
            st.error("O fluxo de potência não convergiu com esta linha retirada.")
        else:
            grafico(figura_rede(d, P, P["n1_lim"], linha_fora=linha))
            st.caption(LEGENDA_REDE + " Ramo tracejado = retirado.")
            if r["ilhadas"]:
                st.warning(f"Ilhamento: {len(r['ilhadas'])} barra(s) sem alimentação, "
                           f"{r['carga_perdida']:.1f} MW de carga perdida.")

    with tabs[1]:
        if not r["ok"]:
            st.error("Sem convergência: o caso é considerado inseguro.")
        elif len(tab) == 0:
            st.success("Nenhuma violação de tensão ou de carregamento com esta linha retirada.")
        else:
            tabela(tab)

    with tabs[2]:
        a = r["antes"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=info["curtos"], y=a["carreg"], name="Antes", marker_color=COR["off"],
                             hovertext=info["nomes_linha"]))
        fig.add_trace(go.Bar(x=info["curtos"], y=d["carreg"], name="Depois", marker_color=COR["barra"],
                             hovertext=info["nomes_linha"]))
        fig.add_hline(y=P["n1_lim"], line_dash="dash", line_color=COR["critico"],
                      annotation_text="limite de emergência")
        fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="Carregamento (%)", legend=dict(orientation="h", y=-0.2))
        grafico(fig)
        if r["ok"]:
            c1, c2 = st.columns(2)
            c1.metric("Carreg. máx. depois", fmt(np.nanmax(d["carreg"]), 0, "%"),
                      fmt(np.nanmax(d["carreg"]) - np.nanmax(a["carreg"]), 0, " p.p."), delta_color="inverse")
            c2.metric("Tensão mín. depois", fmt(np.nanmin(d["vm"]), 3, " pu"),
                      fmt(np.nanmin(d["vm"]) - np.nanmin(a["vm"]), 3), delta_color="normal")

    if P["n1_ranking"]:
        with tabs[3]:
            rk = varredura_n1(pj, h)
            fig = go.Figure(go.Bar(
                x=rk["Severidade"], y=rk["Ramo retirado"], orientation="h",
                marker_color=[COR["ok"] if s == 0 else COR["critico"] for s in rk["Severidade"]]))
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(autorange="reversed"), xaxis_title="Severidade")
            grafico(fig)
            n_ruins = int((rk["Severidade"] > 0).sum())
            st.caption(f"{n_ruins} de {len(rk)} contingências causam algum problema em {hora_txt(h)}.")
            tabela(rk)

    st.button("Parâmetros do N-1", key="b_par_n1", on_click=ir, args=("params_n1",))


def avancar_hora():
    h = min(st.session_state.get("hora_max", 0) + 1, 23)
    st.session_state["hora_max"] = h
    st.session_state["hora"] = h


def reiniciar_simulacao():
    st.session_state["modo_horas"] = "sim"
    st.session_state["hora_max"] = 0
    st.session_state["hora"] = 0


def concluir_simulacao():
    st.session_state["modo_horas"] = "livre"


def controle_horas(pico):
    """Modo 'sim': revela as horas uma a uma, a cada clique (simula telemetria chegando
    em tempo real). Ao concluir as 24h, libera o modo 'livre', com o controle deslizante
    para revisitar qualquer hora já calculada."""
    if "modo_horas" not in st.session_state:
        st.session_state["modo_horas"] = "sim"
        st.session_state["hora_max"] = 0
        st.session_state["hora"] = 0

    c_tit, c_ctrl = st.columns([3, 2], gap="large")
    c_tit.markdown("## Painel de operação da rede")

    if st.session_state["modo_horas"] == "sim":
        hmax = st.session_state["hora_max"]
        with c_ctrl:
            st.caption(f"Modo simulação — telemetria chegando hora a hora ({hmax + 1} de 24)")
            st.progress((hmax + 1) / 24)
            b1, b2 = st.columns(2)
            b1.button("↺ Reiniciar", on_click=reiniciar_simulacao, **LARGURA)
            if hmax < 23:
                b2.button("▶ Avançar hora", on_click=avancar_hora, type="primary", **LARGURA)
            else:
                b2.button("✓ Ver as 24h livremente", on_click=concluir_simulacao, type="primary", **LARGURA)
        hora = st.session_state["hora"]
    else:
        with c_ctrl:
            hora = st.slider("Hora analisada", 0, 23, key="hora", format="%d h")
            st.button("↺ Rever simulação do início", on_click=reiniciar_simulacao)
    return hora


def restaurar_geracao():
    """Devolve geradores e shunts ao ponto de operação original (sem tocar nos demais parâmetros)."""
    info = info_rede()
    P = st.session_state["P"]
    for i, g in enumerate(info["gens0"]):
        P["gens"][i]["p_mw"] = g["p_mw"]
        P["gens"][i]["vm_pu"] = g["vm_pu"]
    P["shunts"] = [True] * info["n_shunts"]
    st.session_state.pop("lab_antes", None)


def bloco_laboratorio(P, pj, hora, res, nivel, titulo):
    """Ações corretivas clássicas de operação (IEEE/literatura de redespacho e suporte de
    reativo): ajustar a geração ativa dentro dos limites reais do gerador, e ligar/desligar
    bancos de capacitor. Mostra o estado 'antes' (guardado no clique) × 'depois' (recalculado)."""
    info = info_rede()
    aberto = nivel in ("alerta", "critico")
    with st.expander("🔧 Laboratório: ações corretivas", expanded=aberto):
        st.caption("Teste ações clássicas de operação — redespacho de geração e chaveamento de "
                  "capacitor — e compare o efeito antes/depois na hora selecionada.")
        algum_estimado = any(g["limite_estimado"] for g in info["gens0"])
        with st.form("form_lab"):
            novos_gens = []
            for i, g in enumerate(info["gens0"]):
                marca = " *" if g["limite_estimado"] else ""
                p = st.slider(f"{info['nomes_ger'][i + 1]} — potência ativa (MW){marca}",
                             float(g["pmin"]), float(g["pmax"]),
                             float(P["gens"][i]["p_mw"]), step=1.0, key=f"lab_gen_{i}")
                novos_gens.append(p)
            novos_shunts = []
            for i, nome in enumerate(info["nomes_shunt"]):
                on = st.checkbox(f"{nome} em serviço", value=bool(P["shunts"][i]), key=f"lab_shunt_{i}")
                novos_shunts.append(on)
            if algum_estimado:
                st.caption("* Limite estimado (2× o despacho original) — a rede não trouxe Pmax "
                          "oficial para este gerador; ajuste se tiver um valor melhor.")
            aplicar = st.form_submit_button("▶ Aplicar ação e comparar", type="primary")

        if aplicar:
            st.session_state["lab_antes"] = {
                "hora": hora, "vm": res["vm"][hora].copy(), "carreg": res["carreg"][hora].copy(),
                "ger": res["ger"][hora], "carga": res["carga"][hora], "perdas": res["perdas"][hora],
                "nivel": nivel, "titulo": titulo,
            }
            for i, p in enumerate(novos_gens):
                P["gens"][i]["p_mw"] = p
            P["shunts"] = novos_shunts
            st.rerun()

        antes = st.session_state.get("lab_antes")
        if antes and antes["hora"] == hora:
            st.markdown("---")
            st.markdown(f"**Comparação da ação aplicada às {hora_txt(hora)}**")
            depois_nivel, depois_titulo, _ = avaliar_hora(res, hora, P)
            c1, c2 = st.columns(2)
            c1.markdown(f"Antes: {pill(antes['titulo'], antes['nivel'])}", unsafe_allow_html=True)
            c2.markdown(f"Depois: {pill(depois_titulo, depois_nivel)}", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=info["curtos"], y=antes["carreg"], name="Antes", marker_color=COR["off"],
                                 hovertext=info["nomes_linha"]))
            fig.add_trace(go.Bar(x=info["curtos"], y=res["carreg"][hora], name="Depois", marker_color=COR["barra"],
                                 hovertext=info["nomes_linha"]))
            fig.add_hline(y=P["lim_carr"], line_dash="dash", line_color=COR["critico"],
                         annotation_text="limite de carregamento")
            fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="Carregamento (%)", legend=dict(orientation="h", y=-0.2))
            grafico(fig)

            m1, m2, m3 = st.columns(3)
            m1.metric("Tensão mín.", fmt(np.nanmin(res["vm"][hora]), 3, " pu"),
                      fmt(np.nanmin(res["vm"][hora]) - np.nanmin(antes["vm"]), 3), delta_color="normal")
            m2.metric("Carreg. máx.", fmt(np.nanmax(res["carreg"][hora]), 0, "%"),
                      fmt(np.nanmax(res["carreg"][hora]) - np.nanmax(antes["carreg"]), 0, " p.p."),
                      delta_color="inverse")
            m3.metric("Perdas", fmt(res["perdas"][hora], 1, " MW"),
                      fmt(res["perdas"][hora] - antes["perdas"], 1, " MW"), delta_color="inverse")
            st.button("Limpar comparação", on_click=lambda: st.session_state.pop("lab_antes", None))

        st.button("↺ Restaurar geração original", on_click=restaurar_geracao)


def painel():
    P = st.session_state["P"]
    info = info_rede()
    pj = json.dumps(P, sort_keys=True)
    res = rodar_fluxo_24h(pj)
    pico = int(np.argmax(res["carga"]))

    hora = controle_horas(pico)

    nivel, titulo, detalhe = avaliar_hora(res, hora, P)
    faixa(nivel, f"STATUS: {titulo}", detalhe)
    st.write("")

    m = st.columns(6)
    if res["conv"][hora]:
        vm, ld = res["vm"][hora], res["carreg"][hora]
        ib, il = int(np.nanargmin(vm)), int(np.nanargmax(ld))
        m[0].metric("Carga", f"{res['carga'][hora]:.0f} MW")
        m[1].metric("Geração", f"{res['ger'][hora]:.0f} MW")
        m[2].metric(f"Perdas ({100 * res['perdas'][hora] / res['ger'][hora]:.1f}%)", f"{res['perdas'][hora]:.1f} MW")
        m[3].metric(f"Tensão mín. ({info['nomes_barra'][ib]})", f"{np.nanmin(vm):.3f} pu")
        m[4].metric(f"Tensão máx. ({info['nomes_barra'][int(np.nanargmax(vm))]})", f"{np.nanmax(vm):.3f} pu")
        m[5].metric(f"Carreg. máx. ({info['curtos'][il]})", f"{np.nanmax(ld):.0f}%")
    else:
        for c, nome in zip(m, ["Carga", "Geração", "Perdas", "Tensão mín.", "Tensão máx.", "Carreg. máx."]):
            c.metric(nome, "—")
    st.write("")

    bloco_laboratorio(P, pj, hora, res, nivel, titulo)
    st.write("")

    limite = st.session_state.get("hora_max", 23) if st.session_state.get("modo_horas") == "sim" else 23

    col_f, col_n = st.columns(2, gap="large")
    with col_f:
        bloco_fluxo(res, hora, P, limite=limite)
    with col_n:
        bloco_n1(res, hora, pico, pj, P)

    with st.sidebar:
        st.markdown("**Rede**")
        net = rede_base()
        st.caption(f"{net.name}\n\n{len(net.bus)} barras · {len(net.line)} linhas · {len(net.trafo)} transformadores · {len(net.gen) + len(net.ext_grid)} geradores")
        st.caption(f"Última execução: {res['executado_em']}")
        st.button("Recalcular tudo", on_click=st.cache_data.clear)


def restaurar_padrao():
    st.session_state["P"] = padrao()
    for k in ("f_fator", "f_vmin", "f_vmax", "f_lim", "ed_curva", "ed_gens",
              "f_n1_ativa", "f_n1_linha", "f_n1_hora", "f_n1_rank", "f_n1_lim", "lab_antes"):
        st.session_state.pop(k, None)


def tela_params_rede():
    P, info = st.session_state["P"], info_rede()
    st.button("← Voltar ao painel", on_click=ir, args=("painel",))
    st.markdown("## Parâmetros da rede")
    st.caption("As alterações só valem ao aplicar. O painel recalcula em seguida.")
    with st.form("form_rede"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            fator = st.slider("Fator global de carga", 0.5, 1.5, float(P["fator"]), 0.05, key="f_fator")
            vmin = st.number_input("Tensão mínima (pu)", 0.80, 1.00, float(P["vmin"]), 0.01, format="%.2f", key="f_vmin")
            vmax = st.number_input("Tensão máxima (pu)", 1.00, 1.20, float(P["vmax"]), 0.01, format="%.2f", key="f_vmax")
            lim = st.slider("Limite de carregamento dos ramos (%)", 50, 150, int(P["lim_carr"]), 5, key="f_lim")
        with c2:
            st.markdown("**Curva de carga horária** (multiplica o fator global)")
            curva = st.data_editor(
                pd.DataFrame({"Hora": [hora_txt(h) for h in range(24)], "Fator": P["curva"]}),
                hide_index=True, disabled=["Hora"], num_rows="fixed", height=300, key="ed_curva",
                column_config={"Fator": st.column_config.NumberColumn(min_value=0.1, max_value=2.0, step=0.01, format="%.2f")})
        st.markdown("**Geradores**")
        gens = st.data_editor(
            pd.DataFrame({"Barra": [g["barra"] for g in info["gens0"]],
                          "Em serviço": [g["em_servico"] for g in P["gens"]],
                          "P (MW)": [g["p_mw"] for g in P["gens"]],
                          "V (pu)": [g["vm_pu"] for g in P["gens"]]}),
            hide_index=True, disabled=["Barra"], num_rows="fixed", key="ed_gens",
            column_config={"P (MW)": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.1f"),
                           "V (pu)": st.column_config.NumberColumn(min_value=0.9, max_value=1.15, step=0.005, format="%.3f")})
        b1, b2, _ = st.columns([1, 1, 3])
        aplicar = b1.form_submit_button("Aplicar e voltar", type="primary")
        b2.form_submit_button("Restaurar padrão", on_click=restaurar_padrao)

    if aplicar:
        P.update(fator=float(fator), vmin=float(vmin), vmax=float(vmax), lim_carr=float(lim))
        P["curva"] = [float(x) for x in curva["Fator"].fillna(1.0)]
        P["gens"] = [{"em_servico": bool(r["Em serviço"]), "p_mw": float(r["P (MW)"]), "vm_pu": float(r["V (pu)"])}
                     for _, r in gens.iterrows()]
        ir("painel")
        st.rerun()


def tela_params_n1():
    P, info = st.session_state["P"], info_rede()
    st.button("← Voltar ao painel", on_click=ir, args=("painel",))
    st.markdown("## Parâmetros da contingência N-1")
    st.caption("As alterações só valem ao aplicar. O painel recalcula em seguida.")
    opcoes_hora = [-2, -1] + list(range(24))
    nomes_hora = lambda v: {-2: "Mesma hora do painel", -1: "Hora de pico"}.get(v, hora_txt(v))
    with st.form("form_n1"):
        ativa = st.checkbox("Usar contingência N-1", value=bool(P["n1_ativa"]), key="f_n1_ativa")
        linha = st.selectbox("Ramo a retirar (linha ou transformador)", list(range(len(info["de"]))), index=int(P["n1_linha"]),
                             format_func=lambda k: info["nomes_linha"][k], key="f_n1_linha")
        hora = st.selectbox("Hora analisada", opcoes_hora, index=opcoes_hora.index(P["n1_hora"]),
                            format_func=nomes_hora, key="f_n1_hora")
        lim = st.slider("Limite de emergência dos ramos (%)", 100, 150, int(P["n1_lim"]), 5, key="f_n1_lim")
        rank = st.checkbox("Calcular também o ranking de todos os ramos (mais lento)",
                           value=bool(P["n1_ranking"]), key="f_n1_rank")
        b1, b2, _ = st.columns([1, 1, 3])
        aplicar = b1.form_submit_button("Aplicar e voltar", type="primary")
        b2.form_submit_button("Restaurar padrão", on_click=restaurar_padrao)

    if aplicar:
        P.update(n1_ativa=bool(ativa), n1_linha=int(linha), n1_hora=int(hora), n1_lim=float(lim),
                 n1_ranking=bool(rank))
        ir("painel")
        st.rerun()


def main():
    if "P" not in st.session_state:
        st.session_state["P"] = padrao()
    if "tela" not in st.session_state:
        st.session_state["tela"] = "painel"
    tela = st.session_state["tela"]
    if tela == "params_rede":
        tela_params_rede()
    elif tela == "params_n1":
        tela_params_n1()
    else:
        painel()


main()
