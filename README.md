# selpop — seleção reprodutível de populações-alvo e controlos

Ferramenta em Python para automatizar a etapa mais demorada da preparação dos estudos com o biobanco
da UCI: **definir casos e controlos, escolher a amostra-índice de cada doente, emparelhar e
documentar o equilíbrio entre grupos (Tabela 1)**. Cada estudo é descrito num ficheiro YAML; o código
não precisa de ser alterado para criar um estudo novo.

## Instalação

Requer Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

## Interface gráfica

```bash
streamlit run app.py
```

Abre no browser (http://localhost:8501). Na barra lateral escolhe-se o modelo de estudo (ou carrega-se um YAML)
e carregam-se a base e os ficheiros auxiliares. Os separadores permitem:

- **Critérios** — editar os critérios de casos e controlos (ativar/desativar, acrescentar linhas) e ver o funil
  atualizado em tempo real; regra do dia-índice; lista das colunas disponíveis para as expressões.
- **Emparelhamento** — razão, suplentes, etapas (exato e calibres), semente e reequilíbrio automático.
- **Variáveis da Tabela 1** — variáveis, tipo, rótulos, quais testar; α.
- **Resultados** — correr a seleção; Tabela 1, gráfico de equilíbrio antes/depois (SMD), funil, amostras
  selecionadas; descarregar o Excel completo e o YAML da configuração usada.
- **Dados** — controlo de qualidade e ficheiros carregados (com SHA-256).
- **Configuração** — o YAML equivalente ao que está nos separadores.

Guardar sempre o YAML junto do Excel: com a mesma base, reproduz exatamente a mesma seleção, também na
linha de comandos.

> **Privacidade.** Os ficheiros carregados ficam só na memória da sessão. Correr a aplicação localmente.
> Não a publicar num servidor público (Render, Streamlit Community Cloud, etc.) com dados de doentes.
> O `.gitignore` impede que `dados/`, `resultados/` e ficheiros Excel entrem no repositório.

## Linha de comandos

```bash
cd selpop
python -m selpop estudos/prognostico_lra.yaml
python -m selpop estudos/diagnostico_drc.yaml
```

Os dados **não vêm no pacote**. Copiar para `dados/`:
- `Base_de_dados_Dem_PAT_SOFA_2026.xlsx` (base principal);
- `ISEL_BMI.xlsx` (IMC por doente; opcional — retirar o bloco `auxiliares` da configuração se não existir).

O resultado é um Excel em `resultados/`. Testes (dados sintéticos): `python -m pytest tests -q`.

## Vários estudos num só Excel

```bash
python -m selpop.consolidar estudos/prognostico_lra.yaml estudos/diagnostico_lra.yaml estudos/diagnostico_drc.yaml \
    --rotulos M1 M2 M3 --saida resultados/Selecao_amostras_biobanco_UCI_miniNIR.xlsx
```

Gera um Excel com: LEIA-ME (modelos, notas de desenho, métodos), **Resumo**, **Variaveis** (papel de cada
variável em cada modelo), **Pedido_biobanco** (lista única de amostras, por prioridade: casos, controlos
principais, suplentes; com o uso em cada modelo), Qualidade_dados e, por modelo, Funil, Tabela 1, Seleção,
Suplentes e Configuração.

### Modelos incluídos

| Rótulo | Configuração | Casos | Controlos |
|---|---|---|---|
| M1 | `prognostico_lra.yaml` | Sem DRC, LRA no internamento; amostra 1–3 dias antes do início | Sem DRC, sem LRA; dia de internamento equivalente |
| M2 | `diagnostico_lra.yaml` | Os mesmos doentes; amostra no dia de início (ou seguinte) | Re-emparelhados no dia equivalente |
| M3 | `diagnostico_drc.yaml` | DRC em dia estável (sem LRA ±2 dias) | Sem DRC, sem LRA, TFGe basal ≥60 |

## Tabela 1 para o artigo e validação em SPSS/PSPP/jamovi

Com `--publicacao` e `--sav` (linha de comandos ou consolidado) ou pelos botões do separador **Resultados**:

- **Tabela 1 em Word** (`*_Table1_en.docx`; `--lingua pt` para português): secções (demografia, comorbilidades,
  internamento, preditores, laboratório, desfechos), mediana [IIQ] ou n (%), n de cada grupo no cabeçalho, p com a
  letra do teste e notas de rodapé (testes, critérios de emparelhamento, dados em falta). Também fica no Excel
  (folha `Table1_publication` / `Mx_Table1_pub`). Nomes em inglês por omissão para as variáveis da base; a
  configuração pode sobrepor com `nome_en`, `rotulos_en` e `secao`, e os grupos com `nome_en`.
- **Dados em SPSS** (`*.sav`): amostras selecionadas (casos e controlos principais), variável `grupo` (1 = caso),
  `conjunto` (ID do caso emparelhado) e as variáveis da Tabela 1, com rótulos de variáveis e de valores.
  Abre no SPSS, PSPP, jamovi e JASP.
- **Sintaxe de validação** (`*_validacao.sps`): `NPAR TESTS /M-W`, `T-TEST` e `CROSSTABS /STATISTICS=CHISQ` para
  as mesmas variáveis. No SPSS/PSPP, comparar com a linha *Equal variances assumed* (t), *Pearson Chi-Square*
  (sem correção de Yates) ou *Fisher's Exact Test* (2×2).

Validação feita com o PSPP 2.0 nos três modelos: **71/71 testes com p idêntico** ao SelPop. As únicas diferenças
são as variáveis com mais de 2 categorias (vaga, motivo de admissão), em que o SelPop usa Fisher (Monte Carlo) e o
PSPP só oferece o qui-quadrado de Pearson.

## Papéis das variáveis na Tabela 1

Cada variável tem um `papel`:

| Papel | Significado | Conta para o reequilíbrio? |
|---|---|---|
| `confundidor` (por omissão) | Deve estar equilibrado entre grupos | Sim |
| `preditor` | Candidato a prever o desfecho (ex.: SOFA cardiovascular, troponina, lactato na LRA). Testado, **não emparelhado**: emparelhar anularia o sinal a estudar | Não |
| `interferente` | Pode alterar o espetro NIR (LDH/hemólise, bilirrubina, CK). Monitorizar | Não |
| `testar: false` | Descritiva (posterior à amostra ou informativa) | Não |

`log: true` calcula o SMD em escala log (variáveis muito assimétricas). O bloco `aliases` dá nomes curtos e
numéricos a colunas de laboratório com nomes longos (ex.: `troponina_dia`). Os modelos de LRA emparelham a
gravidade pelo `sofa_nao_renal_nao_cv_dia` (SOFA sem renal nem cardiovascular), para deixar o componente
cardiovascular livre como preditor.

## O que a ferramenta faz

1. **Carrega** a base e junta ficheiros auxiliares por ID (ex.: IMC). Regista o SHA-256 de cada ficheiro.
2. **Deriva variáveis** por dia e por doente (lista abaixo).
3. **Aplica os critérios** de cada grupo, um a um, e regista o funil (doentes e dias após cada critério).
4. **Escolhe o dia-índice** dos casos (regra configurável).
5. **Emparelha controlos** por etapas: igualdade nas variáveis "exato", diferença ≤ calibre nas restantes.
   O dia do controlo é escolhido no emparelhamento (o mais próximo do caso), ou seja, um emparelhamento
   por marco temporal. Cada doente-controlo é usado uma vez.
6. **Tabela 1** com os testes usados nos artigos do grupo (ver "Métodos") e com o SMD.
7. **Reequilíbrio opcional**: se alguma variável candidata tiver p < α, acrescenta-a ao emparelhamento
   e repete, até `max_iter`. Escolhe a iteração com menos variáveis significativas (desempate: mais casos).
8. **Exporta** o Excel com todas as decisões registadas.

### Folhas do Excel de saída

| Folha | Conteúdo |
|---|---|
| LEIA-ME | Estudo, data, semente, ficheiros de entrada (SHA-256) e métodos |
| Funil | Doentes e dias após cada critério, por grupo |
| Qualidade_dados | Incoerências encontradas (datas, idades, creatinina), com IDs |
| Tabela1 | Comparação casos vs controlos após emparelhamento; linhas com p < α a laranja |
| Equilibrio_antes_depois | p e SMD antes e depois do emparelhamento |
| Selecao | Casos e controlos principais, com nº de amostra, conjunto emparelhado e colunas para o biobanco |
| Suplentes | Controlos de reserva (por caso) |
| Casos_sem_controlo | Casos elegíveis que ficaram sem controlo |
| Iteracoes | Histórico do reequilíbrio |
| Config | Cópia integral da configuração usada |

## Configuração (YAML)

```yaml
grupos:
  casos:
    nome: "LRA (casos)"
    doente:                       # critérios ao nível do doente (cumulativos, cada um aparece no funil)
      - {nome: "Sem DRC registada", expr: "not DRC"}
    dia:                          # critérios ao nível do dia/amostra
      - {nome: "Dia com amostra", expr: "amostra"}
    indice: {regra: alvo, coluna: dias_ate_lra, alvo: 2}
  controlos: { ... }
emparelhamento:
  razao: 2
  suplentes_por_caso: 1
  etapas:
    - {exato: [SEXO, COVID], calibre: {IDADE: 10, dias_desde_admissao: 3}}
    - {exato: [SEXO],        calibre: {IDADE: 15, dias_desde_admissao: 5}}
  reequilibrar: {ativo: true, max_iter: 3, fator_calibre: 0.5, candidatas: [sofa_nao_renal_dia, DIABETES]}
equilibrio:
  alfa: 0.05                      # 0.01 no artigo das citocinas
  variaveis:
    - {col: IDADE, nome: "Idade (anos)", tipo: continua}
    - {col: SEXO, nome: "Sexo", tipo: categorica, rotulos: {0: Masculino, 1: Feminino}}
    - {col: OBITO_UCI, nome: "Óbito na UCI", tipo: categorica, testar: false}   # só descritiva
```

As expressões usam a sintaxe do `pandas` (`and`, `or`, `not`, `==`, `>=`). Podem referir qualquer
coluna da base (com acentos ou espaços, entre acentos graves: `` `SOFA total` >= 6 ``) ou derivada.

**Regras de dia-índice:** `primeiro`, `ultimo`, `minimo`/`maximo` (de uma `coluna`), `alvo`
(mais próximo de `alvo` numa `coluna`), `aleatorio`.

**Variáveis por expressão:** bloco `derivados`, por exemplo `DRC: "INSUFICIENCIA_RENAL == 1"`.

### Variáveis derivadas disponíveis

| Variável | Nível | Definição |
|---|---|---|
| `amostra` | dia | Há nº de amostra no biobanco |
| `cr_unico`, `ur_unico` | dia | Creatinina/ureia com um só valor no dia (máx = mín) |
| `cr_dia`, `ur_dia` | dia | Valor único do dia (vazio se houver mais do que um) |
| `dias_desde_admissao` | dia | Data da colheita − data de admissão na UCI |
| `em_vmi_dia`, `em_ecmo_dia` | dia | O dia está dentro do período de VMI/ECMO |
| `dias_vmi_ate_amostra` | dia | Dias em VMI à data da colheita |
| `sofa_total_dia`, `sofa_nao_renal_dia` | dia | SOFA total e SOFA total − SOFA renal |
| `sofa_cv_dia`, `sofa_nao_renal_nao_cv_dia` | dia | SOFA cardiovascular; SOFA total − renal − cardiovascular |
| `tfge_dia` | dia | CKD-EPI 2021 com a creatinina mínima do dia |
| `lra_dia`, `lra_estadio_dia` | dia | Critério KDIGO de creatinina cumprido nesse dia, e estádio |
| `sem_lra_janela` | dia | Sem critério de LRA nos ±k dias (`lra.janela_estavel_dias`) |
| `dias_ate_lra` | dia | Dias até ao início da LRA (positivo = antes) |
| `dias_obs_apos` | dia | Dias de seguimento de creatinina após esse dia |
| `lra_subida`, `lra_admissao`, `lra_qualquer` | doente | LRA por subida no internamento / presente na admissão / qualquer |
| `lra_inicio`, `lra_estadio_max` | doente | Data do primeiro critério; estádio máximo |
| `tfge_basal_proxy` | doente | CKD-EPI 2021 com a creatinina mínima do internamento |
| `duracao_uci_dias`, `dias_com_dados`, `n_amostras_doente` | doente | Duração e quantidade de dados |

## Métodos

- **Testes** (Fonseca et al., *Comput Biol Med* 2025; Araújo et al., *Metabolites* 2025): contínuas em
  mediana (P25–P75); Shapiro–Wilk em cada grupo → t de Student (variâncias iguais) se ambos normais, senão Mann–Whitney U (aproximação normal com correção de empates, sem correção de continuidade — como o SPSS/PSPP).
  Categóricas em n (%); qui-quadrado, ou exato de Fisher se alguma frequência esperada < 5. Em tabelas maiores que 2×2, o p de Fisher é estimado por Monte Carlo (9 999 reamostragens, semente fixa), para ser reprodutível.
  Bilateral, α configurável, sem correção para comparações múltiplas.
- **SMD**: acrescentado porque "p ≥ α" com grupos pequenos não demonstra equilíbrio. |SMD| < 0,1 é o
  limiar habitual; com n ≈ 30 por grupo, valores até ~0,2–0,3 são frequentes e devem ser discutidos.
- **TFGe**: CKD-EPI 2021 sem raça. Em UCI é descritiva (a creatinina não está estável).
- **LRA**: critério de creatinina da KDIGO 2012 sobre valores diários (subida ≥1,5× face ao mínimo de 7 dias
  ou ≥0,3 mg/dL em 48 h; presente na admissão se ≥1,5× o nadir dos 7 dias seguintes). Sem débito urinário
  nem creatinina pré-admissão: é uma aproximação.
- **Emparelhamento**: guloso global sem reposição, por etapas, com desempate aleatório reprodutível.
  Não é ótimo no sentido estrito (minimizar a distância total); para os tamanhos em causa a diferença é
  pequena, e a transparência e a reprodutibilidade são maiores.

## Limitações e cuidados

- O reequilíbrio por p-valores é uma ajuda prática, não um critério científico. As variáveis de
  emparelhamento devem ser escolhidas por plausibilidade (confundidores do sinal espetral e do desfecho).
- Variáveis posteriores à amostra-índice (duração do internamento, óbito, LRA posterior) são consequências
  e não confundidores: devem ser descritivas (`testar: false`), sobretudo em estudos de prognóstico.
- Exigir controlos "sem qualquer LRA" tende a selecionar doentes menos graves. Ver o SOFA na Tabela 1.
- A folha Qualidade_dados não corrige nada: indica registos a confirmar na fonte.
