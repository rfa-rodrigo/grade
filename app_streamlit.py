import streamlit as st
import pandas as pd
import io
from datetime import datetime, time, timedelta
from fpdf import FPDF  # pip install fpdf
import base64

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

def renderizar_quadro(grade, slots):
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
    st.markdown('\n'.join(html), unsafe_allow_html=True)

# -------------------------------
# Geração de PDF
# -------------------------------
def _latin1(s: str) -> str:
    """Converte para latin-1 com fallback, para não quebrar o fpdf."""
    if s is None:
        return ''
    return s.encode('latin-1', 'replace').decode('latin-1')

def pdf_da_selecao(sel_df: pd.DataFrame) -> bytes:
    """
    Gera PDF (A4 paisagem) com cabeçalho, data e tabela Resumo das disciplinas
    a partir da seleção (sem conflitos).
    """
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, _latin1('Montador de grade e detector de conflitos - IFB'), ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, _latin1(f'Emitido em: {datetime.now().strftime("%d/%m/%Y %H:%M")}'), ln=True)
    pdf.ln(2)

    # Cabeçalho da tabela
    pdf.set_font('Arial', 'B', 11)
    headers = ['Dia', 'Início', 'Fim', 'Curso', 'Disciplina', 'Sala', 'Professor']
    # Larguras aproximadas somando ~277mm de área útil (A4L com ~10mm margens)
    col_w = [22, 18, 18, 55, 95, 30, 35]
    for h, w in zip(headers, col_w):
        pdf.cell(w, 8, _latin1(h), border=1, align='L')
    pdf.ln(8)

    # Linhas (ordenadas por dia_idx, inicio)
    pdf.set_font('Arial', '', 10)
    dias = {i: n for i, n in enumerate(['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado'])}
    temp = sel_df.copy()
    temp['__inicio_sort__'] = pd.to_datetime(temp['inicio'], format='%H:%M', errors='coerce')
    temp = temp.sort_values(['dia_idx', '__inicio_sort__', 'disciplina', 'curso'], kind='stable')

    for _, r in temp.iterrows():
        row_vals = [
            dias.get(int(r['dia_idx']), ''),
            str(r['inicio']),
            str(r['fim']),
            str(r['curso']),
            str(r['disciplina']),
            str(r.get('sala', '')),
            str(r.get('professor', '')),
        ]
        for val, w in zip(row_vals, col_w):
            pdf.cell(w, 7, _latin1(val), border=1, align='L')
        pdf.ln(7)

    # Retorna bytes
    return pdf.output(dest='S').encode('latin-1')

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

    # Avalia conflitos para controlar o PDF
    _, sel_mar_local = construir_quadro(st.session_state['selecionadas'], slots)
    tem_disciplinas = not sel_mar_local.empty
    tem_conflito = tem_disciplinas and bool(sel_mar_local['conflito'].any())

    if tem_disciplinas and not tem_conflito:
        st.success('Grade montada sem conflito de disciplinas.')
    elif tem_disciplinas and tem_conflito:
        st.error('Grade com conflitos. Não será permitida a matrícula.')
    else:
        st.info('Adicione disciplinas para montar a grade.')

    # Botão "Gerar PDF" (habilitado apenas se não houver conflitos)
    gerar_pdf = st.button('🖨️ Gerar PDF (A4 paisagem)', disabled=not (tem_disciplinas and not tem_conflito))

    # Se clicado e sem conflitos: gera e exibe botão de download
    if gerar_pdf and tem_disciplinas and not tem_conflito:
        # Prepara DataFrame 'show' (igual ao resumo exibido no app)
        show_pdf = sel_mar_local.copy()
        show_pdf['dia'] = show_pdf['dia_idx'].map({i: n for i, n in enumerate(['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado'])})
        show_pdf['inicio'] = show_pdf['inicio'].apply(lambda t: t.strftime('%H:%M'))
        show_pdf['fim'] = show_pdf['fim'].apply(lambda t: t.strftime('%H:%M'))
        show_pdf = show_pdf[['curso', 'disciplina', 'professor', 'sala', 'dia', 'inicio', 'fim', 'conflito']]

        pdf_bytes = pdf_da_selecao(show_pdf)

        st.download_button(
            label='⬇️ Baixar PDF da grade',
            data=pdf_bytes,
            file_name='grade_ifb.pdf',
            mime='application/pdf',
            use_container_width=True
        )

with right:
    st.subheader('Quadro de horários')
    grade, sel_mar = construir_quadro(st.session_state['selecionadas'], slots)
    renderizar_quadro(grade, slots)

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
