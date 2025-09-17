import streamlit as st
import pandas as pd
import io
from datetime import datetime, time, timedelta
import streamlit.components.v1 as components
import json
from uuid import uuid4

st.set_page_config(page_title='Montador de grade e detector de conflitos - IFB', layout='wide')

# -------------------------------
# Configurações/Convenções
# -------------------------------
DIA_MAP = {
    'segunda': 0, 'terca': 1, 'terça': 1, 'quarta': 2,
    'quinta': 3, 'sexta': 4, 'sabado': 5, 'sábado': 5
}
DIA_LABELS = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado']

@st.cache_data
def ler_horarios_local(path: str = 'horarios.csv') -> pd.DataFrame:
    with open(path, 'rb') as f:
        csv_bytes = f.read()
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # Normaliza colunas esperadas
    colmap = {c.lower().strip(): c for c in df.columns}
    needed = ['curso', 'disciplina', 'dia', 'inicio', 'fim']
    alias = {
        'curso': ['curso', 'course'],
        'disciplina': ['disciplina', 'componente', 'materia', 'matéria', 'subject'],
        'dia': ['dia', 'dia_semana', 'weekday', 'dia da semana'],
        'inicio': ['inicio', 'início', 'hora_inicio', 'hora início', 'start', 'start_time'],
        'fim': ['fim', 'hora_fim', 'término', 'termino', 'end', 'end_time'],
    }
    rename = {}
    for key, al in alias.items():
        for a in al:
            if a in colmap:
                rename[colmap[a]] = key
                break
    df = df.rename(columns=rename)

    missing = [k for k in needed if k not in df.columns]
    if missing:
        raise ValueError(f'Colunas ausentes no CSV: {missing}. Esperado: {needed}')

    # Normaliza dia
    df['dia'] = df['dia'].astype(str).str.lower().str.strip()
    df['dia_idx'] = df['dia'].map(DIA_MAP)
    if df['dia_idx'].isna().any():
        vals = sorted(df.loc[df['dia_idx'].isna(), 'dia'].unique())
        raise ValueError(f"Valores de 'dia' inválidos: {vals}. Use {list(DIA_MAP.keys())}")

    # Normaliza horário (HH:MM)
    def parse_hhmm(x):
        s = str(x).strip()
        for fmt in ['%H:%M', '%H:%M:%S']:
            try:
                return datetime.strptime(s, fmt).time()
            except Exception:
                pass
        # tenta 730 -> 07:30
        s_num = ''.join(ch for ch in s if ch.isdigit())
        if len(s_num) in (3, 4):
            s_num = s_num.zfill(4)
            return time(int(s_num[:2]), int(s_num[2:]))
        raise ValueError(f'Horário inválido: {x}')

    df['inicio'] = df['inicio'].map(parse_hhmm)
    df['fim'] = df['fim'].map(parse_hhmm)

    # sanity
    if (pd.Series([dt for dt in df['fim']]) <= pd.Series([dt for dt in df['inicio']])).any():
        raise ValueError('Há linhas com fim <= início.')

    # Campos opcionais
    if 'sala' not in df.columns:
        df['sala'] = ''
    if 'professor' not in df.columns and 'docente' in df.columns:
        df = df.rename(columns={'docente': 'professor'})
    if 'professor' not in df.columns:
        df['professor'] = ''

    return df

def gerar_slots(df: pd.DataFrame, passo_min=30):
    """
    Gera os slots da grade SEMPRE cobrindo 07:00–22:00,
    independentemente dos horários existentes no CSV.
    """
    inicio_padrao = time(7, 0)   # 07:00
    fim_padrao = time(22, 0)     # 22:00

    min_inicio = inicio_padrao
    max_fim = fim_padrao

    def round_down(t: time, minutes=30):
        return time(t.hour, (t.minute // minutes) * minutes)

    def round_up(t: time, minutes=30):
        add = (minutes - (t.minute % minutes)) % minutes
        hh = t.hour + (t.minute + add) // 60
        mm = (t.minute + add) % 60
        if hh > 23:
            hh, mm = 23, 59
        return time(hh, mm)

    start = round_down(min_inicio, passo_min)
    end = round_up(max_fim, passo_min)

    slots = []
    cur = datetime(2000, 1, 1, start.hour, start.minute)
    end_dt = datetime(2000, 1, 1, end.hour, end.minute)
    delta = timedelta(minutes=passo_min)
    while cur < end_dt:
        nxt = cur + delta
        slots.append((cur.time(), nxt.time()))
        cur = nxt
    return slots

def intervalo_conflita(a_ini: time, a_fim: time, b_ini: time, b_fim: time) -> bool:
    # Sobreposição estrita: início < outro_fim e fim > outro_inicio
    return (a_ini < b_fim) and (a_fim > b_ini)

def construir_quadro(selecionadas: pd.DataFrame, slots):
    grade = {d: {i: [] for i in range(len(slots))} for d in range(6)}  # segunda..sábado
    if selecionadas.empty:
        return grade, selecionadas

    selecionadas = selecionadas.copy()
    selecionadas['conflito'] = False

    # Detecta conflitos no mesmo dia
    for d in range(6):
        dd = selecionadas[selecionadas['dia_idx'] == d].reset_index(drop=True)
        for i in range(len(dd)):
            for j in range(i + 1, len(dd)):
                if intervalo_conflita(dd.loc[i, 'inicio'], dd.loc[i, 'fim'], dd.loc[j, 'inicio'], dd.loc[j, 'fim']):
                    selecionadas.loc[dd.index[i], 'conflito'] = True
                    selecionadas.loc[dd.index[j], 'conflito'] = True

    # Preenche os slots
    for _, row in selecionadas.iterrows():
        for k, (s_ini, s_fim) in enumerate(slots):
            if intervalo_conflita(row['inicio'], row['fim'], s_ini, s_fim):
                grade[row['dia_idx']][k].append({
                    'curso': row['curso'],
                    'disciplina': row['disciplina'],
                    'professor': row.get('professor', ''),
                    'sala': row.get('sala', ''),
                    'inicio': row['inicio'].strftime('%H:%M'),
                    'fim': row['fim'].strftime('%H:%M'),
                    'conflito': bool(row['conflito']),
                })
    return grade, selecionadas

def renderizar_quadro(grade, slots, render=True):
    """Gera o HTML da grade e opcionalmente renderiza no Streamlit."""
    def slot_label(t0: time, t1: time):
        return f"{t0.strftime('%H:%M')}–{t1.strftime('%H:%M')}"
    css = '''
    <style>
    .tbl {width:100%; border-collapse: collapse; table-layout: fixed;}
    .tbl th, .tbl td {border:1px solid #ddd; padding:6px; vertical-align: top; font-size: 0.9rem;}
    .tbl th {background:#f5f5f5; position: sticky; top: 0; z-index: 1;}
    .slot {min-height: 60px;}
    .chip {display:block; margin:2px 0; padding:4px 6px; border-radius:8px; border:1px solid #999; background:#fafafa;}
    .chip.conflict {background:#ffe5e5; border-color:#ff6666;}
    .chip .title {font-weight:600;}
    .chip .meta {font-size:0.75rem; opacity:0.8;}
    .timecol {width: 120px; background:#fcfcfc; font-weight:600;}
    </style>
    '''
    html = [css, "<table class='tbl'>"]
    html.append("<tr><th class='timecol'>Horário</th>" + "".join(f"<th>{lbl}</th>" for lbl in DIA_LABELS) + "</tr>")
    for i, (t0, t1) in enumerate(slots):
        html.append('<tr>')
        html.append(f"<td class='timecol'>{slot_label(t0, t1)}</td>")
        for d in range(6):
            items = grade[d][i]
            if items:
                chips = []
                for it in items:
                    conflict_cls = ' conflict' if it['conflito'] else ''
                    tt = (
                        f"<div class='chip{conflict_cls}'>"
                        f"<div class='title'>{it['disciplina']}</div>"
                        f"<div class='meta'>{it['curso']}"
                        + (f" • {it['professor']}" if it['professor'] else '')
                        + (f" • {it['sala']}" if it['sala'] else '')
                        + f" • {it['inicio']}–{it['fim']}</div>"
                        f"</div>"
                    )
                    chips.append(tt)
                cell = ''.join(chips)
            else:
                cell = ''
            html.append(f"<td class='slot'>{cell}</td>")
        html.append('</tr>')
    html.append('</table>')
    html_str = '\n'.join(html)
    if render:
        st.markdown(html_str, unsafe_allow_html=True)
    return html_str

def build_print_html(grade_html: str, resumo_df: pd.DataFrame) -> str:
    """Gera uma página HTML independente para impressão (paisagem + fit-to-page)."""
    # Tabela do resumo em HTML simples
    if resumo_df is None or resumo_df.empty:
        resumo_html = "<p><em>Nenhuma disciplina adicionada.</em></p>"
    else:
        # Renomeia a coluna de conflito para 'choque' (se necessário)
        cols = list(resumo_df.columns)
        resumo_show = resumo_df.copy()
        if 'conflito' in cols and 'choque' not in cols:
            resumo_show = resumo_show.rename(columns={'conflito': 'choque'})
        resumo_html = resumo_show.to_html(index=False, border=0, classes='resumo')

    now_str = datetime.now().strftime('%d/%m/%Y %H:%M')

    # CSS de impressão (A4 landscape, margens, ajuste de escala automático via JS)
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Grade - IFB</title>
<style>
  @page {{
    size: A4 landscape;
    margin: 10mm;
  }}
  html, body {{
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
    font-family: Arial, Helvetica, sans-serif;
  }}
  .print-wrapper {{
    width: 100%;
  }}
  h1 {{ margin: 0 0 6px 0; font-size: 20px; }}
  .meta-print {{ font-size: 12px; margin: 0 0 12px 0; color: #333; }}
  /* Reaproveita estilos da grade */
  .tbl {{ width:100%; border-collapse: collapse; table-layout: fixed; }}
  .tbl th, .tbl td {{ border:1px solid #ddd; padding:6px; vertical-align: top; font-size: 11px; }}
  .tbl th {{ background:#f5f5f5; }}
  .slot {{ min-height: 40px; }}
  .chip {{ display:block; margin:2px 0; padding:3px 5px; border-radius:6px; border:1px solid #999; background:#fafafa; }}
  .chip.conflict {{ background:#ffe5e5; border-color:#ff6666; }}
  .chip .title {{ font-weight:600; }}
  .chip .meta {{ font-size:10px; opacity:0.8; }}
  .timecol {{ width: 100px; background:#fcfcfc; font-weight:600; }}

  h2 {{ margin-top: 14px; font-size: 16px; }}
  table.resumo {{
    border-collapse: collapse; width: 100%;
    font-size: 11px;
  }}
  table.resumo th, table.resumo td {{
    border: 1px solid #ddd; padding: 6px; text-align: left;
  }}

  /* Evita quebras estranhas */
  tr, td, th {{ page-break-inside: avoid; }}
</style>
</head>
<body>
<div class="print-wrapper" id="content">
  <h1>Montador de grade e detector de conflitos - IFB</h1>
  <p class="meta-print">Emitido em: {now_str}</p>
  {grade_html}
  <h2>Resumo das disciplinas</h2>
  {resumo_html}
</div>

<script>
  // Ajuste automático para "caber em 1 página" (paisagem).
  // Mede a altura do conteúdo e, se maior que a janela de impressão, aplica escala.
  window.onload = function() {{
    try {{
      const el = document.getElementById('content');
      // margem de segurança
      const pageH = window.innerHeight - 20;
      const contentH = el.scrollHeight;
      const scale = Math.min(1, pageH / contentH);
      if (scale < 1) {{
        el.style.transform = 'scale(' + scale + ')';
        el.style.transformOrigin = 'top left';
      }}
    }} catch (e) {{}}
    setTimeout(function() {{
      window.print();
      window.close();
    }}, 300);
  }};
</script>
</body>
</html>"""

# -------------------------------
# App
# -------------------------------
st.title('Montador de grade e detector de conflitos - IFB')
st.markdown(
    "Selecione o **Curso** e depois a **Disciplina** para adicioná-la ao quadro.\n"
    "Disciplinas com **conflito de horário** aparecerão **em vermelho**.\n"
    "Use os botões para **adicionar**, **remover a última adição** ou **limpar** o quadro.\n"
    "Para atualizar a base a cada semestre, substitua o arquivo **horarios.csv** nesta pasta."
)

# Leitura obrigatória do CSV local
try:
    df = ler_horarios_local('horarios.csv')
except FileNotFoundError:
    st.error('Arquivo local `horarios.csv` não encontrado. Coloque o arquivo na mesma pasta do app.')
    st.stop()
except Exception as e:
    st.error(f'Erro ao ler `horarios.csv`: {e}')
    st.stop()

# Slots sempre cobrindo 07:00–22:00
slots = gerar_slots(df)

# Estado
if 'selecionadas' not in st.session_state:
    st.session_state['selecionadas'] = pd.DataFrame(columns=df.columns)

# Layout: seleção à esquerda, quadro à direita
left, right = st.columns([1, 2], gap='large')

with left:
    cursos = sorted(df['curso'].unique())
    curso = st.selectbox('Curso', options=cursos, key='curso_select')

    dff = df[df['curso'] == curso].copy()
    disciplinas = sorted(dff['disciplina'].unique())
    disciplina = st.selectbox('Disciplina', options=disciplinas, key='disciplina_select')

    if st.button('➕ Adicionar ao quadro', type='primary'):
        linhas = dff[dff['disciplina'] == disciplina].copy()
        st.session_state['selecionadas'] = pd.concat([st.session_state['selecionadas'], linhas], ignore_index=True)

    # Linha de botões: Remover última e Limpar
    colb1, colb2 = st.columns(2)
    with colb1:
        if st.button('🗑️ Remover última adição'):
            if len(st.session_state['selecionadas']) > 0:
                ultima = st.session_state['selecionadas'].iloc[-1]['disciplina']
                idx = st.session_state['selecionadas'][st.session_state['selecionadas']['disciplina'] == ultima].index
                st.session_state['selecionadas'].drop(idx, inplace=True)
                st.session_state['selecionadas'].reset_index(drop=True, inplace=True)
    with colb2:
        if st.button('🧹 Limpar quadro'):
            st.session_state['selecionadas'] = pd.DataFrame(columns=df.columns)

    # Avalia conflitos para controlar o botão de impressão
    _, sel_mar_local = construir_quadro(st.session_state['selecionadas'], slots)
    tem_disciplinas = not sel_mar_local.empty
    tem_conflito = tem_disciplinas and bool(sel_mar_local['conflito'].any())

    # Mensagem e botão (lateral, abaixo de Remover última adição)
    if tem_disciplinas and not tem_conflito:
        st.success('Grade montada sem conflito de disciplinas. Imprima essa página e encaminhe ao CDRA.')
    elif tem_disciplinas and tem_conflito:
        st.error('Grade com conflitos. Não será permitida a matrícula.')
    else:
        st.info('Adicione disciplinas para montar a grade.')

with right:
    st.subheader('Quadro de horários')
    grade, sel_mar = construir_quadro(st.session_state['selecionadas'], slots)
    # Renderiza e guarda o HTML da grade para impressão
    grade_html = renderizar_quadro(grade, slots, render=True)
    st.session_state['grade_html'] = grade_html

st.divider()

# Resumo das disciplinas (na página do app)
st.subheader('Resumo das disciplinas')
show = sel_mar.copy() if not st.session_state['selecionadas'].empty else pd.DataFrame(columns=df.columns.tolist() + ['conflito'])
if not show.empty:
    show['dia'] = show['dia_idx'].map({i: n for i, n in enumerate(['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado'])})
    show['inicio'] = show['inicio'].apply(lambda t: t.strftime('%H:%M'))
    show['fim'] = show['fim'].apply(lambda t: t.strftime('%H:%M'))
    show = show[['curso', 'disciplina', 'professor', 'sala', 'dia', 'inicio', 'fim', 'conflito']].rename(columns={'conflito': 'choque'})
else:
    show = pd.DataFrame(columns=['curso', 'disciplina', 'professor', 'sala', 'dia', 'inicio', 'fim', 'choque'])

st.dataframe(show, use_container_width=True, hide_index=True)
