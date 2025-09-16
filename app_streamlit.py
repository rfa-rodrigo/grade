
import streamlit as st
import pandas as pd
import io
from datetime import datetime, time, timedelta

st.set_page_config(page_title='Quadro de Horários - Detector de Choques', layout='wide')

# -------------------------------
# Configurações/Convenções
# -------------------------------
DIA_MAP = {
    'segunda': 0, 'terca': 1, 'terça': 1, 'quarta': 2,
    'quinta': 3, 'sexta': 4, 'sabado': 5, 'sábado': 5
}
DIA_LABELS = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado']

@st.cache_data
def ler_horarios(csv_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    # Normaliza colunas esperadas
    colmap = {c.lower().strip(): c for c in df.columns}
    # nomes canônicos
    needed = ['curso', 'disciplina', 'dia', 'inicio', 'fim']
    # mapeia possíveis variações
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
        raise ValueError(f'Valores de \'dia\' inválidos: {vals}. Use {list(DIA_MAP.keys())}')
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
        if len(s_num) in (3,4):
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
    # Descobre faixa de horários a partir do dataset
    min_inicio = min(df['inicio'])
    max_fim = max(df['fim'])
    # arredonda para baixo/alto em múltiplos de passo_min
    def round_down(t: time, minutes=30):
        return time(t.hour, (t.minute // minutes) * minutes)
    def round_up(t: time, minutes=30):
        add = (minutes - (t.minute % minutes)) % minutes
        hh = t.hour + (t.minute + add)//60
        mm = (t.minute + add)%60
        return time(min(hh, 23), mm)
    start = round_down(min_inicio, passo_min)
    end = round_up(max_fim, passo_min)
    # gera lista de slots
    slots = []
    cur = datetime(2000,1,1,start.hour,start.minute)
    end_dt = datetime(2000,1,1,end.hour,end.minute)
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
    # Cria estrutura dia x slot com listas de itens (podem conflitar)
    grade = {
        d: {i: [] for i in range(len(slots))}
        for d in range(6)  # segunda..sábado
    }
    # Detecta conflitos: para cada dia, marca itens que se sobrepõem
    selecionadas = selecionadas.copy()
    selecionadas['conflito'] = False
    for d in range(6):
        dd = selecionadas[selecionadas['dia_idx'] == d].reset_index(drop=True)
        for i in range(len(dd)):
            for j in range(i+1, len(dd)):
                if intervalo_conflita(dd.loc[i,'inicio'], dd.loc[i,'fim'], dd.loc[j,'inicio'], dd.loc[j,'fim']):
                    selecionadas.loc[dd.index[i], 'conflito'] = True
                    selecionadas.loc[dd.index[j], 'conflito'] = True
    # Preenche slots
    for _, row in selecionadas.iterrows():
        for k,(s_ini,s_fim) in enumerate(slots):
            if intervalo_conflita(row['inicio'], row['fim'], s_ini, s_fim):
                grade[row['dia_idx']][k].append({
                    'curso': row['curso'],
                    'disciplina': row['disciplina'],
                    'professor': row.get('professor',''),
                    'sala': row.get('sala',''),
                    'inicio': row['inicio'].strftime('%H:%M'),
                    'fim': row['fim'].strftime('%H:%M'),
                    'conflito': bool(row['conflito']),
                })
    return grade

def renderizar_quadro(grade, slots):
    # Renderiza como tabela HTML estilizada
    def slot_label(t0: time, t1: time):
        return f"{t0.strftime('%H:%M')}–{t1.strftime('%H:%M')}"
    # CSS básico
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
    # Cabeçalho
    html = [css, "<table class='tbl'>"]
    html.append("<tr><th class='timecol'>Horário</th>" + "".join(f"<th>{lbl}</th>" for lbl in DIA_LABELS) + "</tr>")
    # Linhas por slot
    for i,(t0,t1) in enumerate(slots):
        html.append('<tr>')
        html.append(f"<td class='timecol'>{slot_label(t0,t1)}</td>")
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

st.title('📅 Quadro de Horários • Choques de Disciplinas')

st.markdown('''
Este aplicativo lê um arquivo **CSV** de horários (por padrão `horarios.csv` no diretório atual)
e permite **montar o seu quadro**, escolhendo **Curso** e **Disciplina**.
Se houver **choque de horários**, as disciplinas conflitantes aparecem **em vermelho**.

**Formato esperado do CSV** (mínimo):
- `curso` — nome do curso (ex.: *Engenharia Mecânica*)
- `disciplina` — nome do componente
- `dia` — valores: `segunda`, `terca`/`terça`, `quarta`, `quinta`, `sexta`, `sabado`/`sábado`
- `inicio` — horário (ex.: `08:00`)
- `fim` — horário (ex.: `09:40`)

**Campos opcionais**: `sala`, `professor`.
''')

# Entrada do CSV
default_bytes = None
try:
    with open('horarios.csv', 'rb') as f:
        default_bytes = f.read()
except Exception:
    pass

up = st.file_uploader('Carregue um CSV (ou mantenha o padrão `horarios.csv` no diretório).', type=['csv'])
if up is not None:
    csv_bytes = up.read()
elif default_bytes is not None:
    csv_bytes = default_bytes
    st.info('Usando arquivo local `horarios.csv`.')
else:
    st.warning('Nenhum CSV encontrado. Use o botão acima para enviar um arquivo.')
    st.stop()

# Lê dataset
try:
    df = ler_horarios(csv_bytes)
except Exception as e:
    st.error(f'Erro ao ler CSV: {e}')
    st.stop()

# UI de seleção
cursos = sorted(df['curso'].unique())
curso = st.selectbox('Curso', options=cursos)

# Disciplinas por curso
dff = df[df['curso'] == curso].copy()
disciplinas = sorted(dff['disciplina'].unique())
disciplina = st.selectbox('Disciplina', options=disciplinas, key='disciplina_select')

# Estado da sessão: lista de seleções (linhas)
if 'selecionadas' not in st.session_state:
    st.session_state['selecionadas'] = pd.DataFrame(columns=df.columns)

cols = st.columns([1,1,1])
with cols[0]:
    if st.button('➕ Adicionar ao quadro', type='primary'):
        linhas = dff[dff['disciplina'] == disciplina].copy()
        st.session_state['selecionadas'] = pd.concat([st.session_state['selecionadas'], linhas], ignore_index=True)
with cols[1]:
    if st.button('🧹 Limpar quadro'):
        st.session_state['selecionadas'] = pd.DataFrame(columns=df.columns)
with cols[2]:
    if st.button('🗑️ Remover última adição'):
        if len(st.session_state['selecionadas']) > 0:
            # remove o último bloco adicionado (todas as linhas da última disciplina adicionada)
            ultima = st.session_state['selecionadas'].iloc[-1]['disciplina']
            idx = st.session_state['selecionadas'][st.session_state['selecionadas']['disciplina'] == ultima].index
            st.session_state['selecionadas'].drop(idx, inplace=True)
            st.session_state['selecionadas'].reset_index(drop=True, inplace=True)

st.divider()

# Monta grade
if len(st.session_state['selecionadas']) == 0:
    st.info('Nenhuma disciplina adicionada ainda.')
else:
    slots = gerar_slots(pd.concat([df, st.session_state['selecionadas']], ignore_index=True))
    grade = construir_quadro(st.session_state['selecionadas'], slots)

    # Sumário de conflitos
    sel = st.session_state['selecionadas'].copy()
    # marca conflitos novamente (para tabela resumo)
    sel['conflito'] = False
    for d in range(6):
        dd = sel[sel['dia_idx'] == d].reset_index(drop=True)
        for i in range(len(dd)):
            for j in range(i+1, len(dd)):
                if intervalo_conflita(dd.loc[i,'inicio'], dd.loc[i,'fim'], dd.loc[j,'inicio'], dd.loc[j,'fim']):
                    sel.loc[dd.index[i], 'conflito'] = True
                    sel.loc[dd.index[j], 'conflito'] = True

    st.subheader('Quadro de horários')
    renderizar_quadro(grade, slots)

    st.subheader('Resumo das disciplinas adicionadas')
    show = sel.copy()
    show['dia'] = show['dia_idx'].map({i:n for i,n in enumerate(['Segunda','Terça','Quarta','Quinta','Sexta','Sábado'])})
    show['inicio'] = show['inicio'].apply(lambda t: t.strftime('%H:%M'))
    show['fim'] = show['fim'].apply(lambda t: t.strftime('%H:%M'))
    show = show[['curso','disciplina','professor','sala','dia','inicio','fim','conflito']].rename(columns={
        'conflito':'choque'
    })
    st.dataframe(show, use_container_width=True, hide_index=True)

    # Exporta a seleção
    csv_out = show.to_csv(index=False).encode('utf-8')
    st.download_button('⬇️ Baixar seleção (CSV)', data=csv_out, file_name='quadro_selecionado.csv', mime='text/csv')

with st.expander('Ajuda • Dúvidas frequentes'):
    st.markdown('''
    **Como o app detecta choques?**  
    Dois blocos no **mesmo dia** entram em choque quando `início < fim_outro` **e** `fim > início_outro`.
    
    **Posso usar outros nomes de colunas?**  
    O app reconhece variações comuns (ex.: `materia`, `início`, `hora_inicio`, etc.) e renomeia para o padrão.
    
    **E se meu campus tem domingo?**  
    Atualmente a grade vai de **segunda a sábado**. É simples estender no código (`DIA_MAP`).

    **Posso mudar o tamanho do slot (30 min)?**  
    Sim, ajuste o parâmetro `passo_min` em `gerar_slots`.
    ''')
