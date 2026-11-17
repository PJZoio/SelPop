# SelPop — seleção de populações e comparação de grupos

Ferramenta em Python, com interface no browser, para:

1. **selecionar** a população-alvo (casos) e os controlos a partir de uma base de dados (Excel ou CSV);
2. **emparelhar** controlos aos casos (ex.: mesmo sexo, idade ±5 anos);
3. **escolher e aplicar os testes estatísticos** certos, com explicação passo a passo;
4. **exportar** a lista de amostras, a tabela de características para o artigo (Word, inglês ou português) e os
   dados para validação em SPSS, PSPP, jamovi ou JASP.

Funciona com qualquer base em que cada linha seja um participante (estudo transversal) ou uma observação de um
participante (estudo longitudinal, ex.: dias de internamento). Tudo fica registado num ficheiro de configuração
(YAML), por isso a mesma base e a mesma configuração dão sempre a mesma seleção.

---

## ⚠️ Privacidade: use a aplicação no seu computador

**Com dados de doentes, corra a aplicação localmente.** Assim os dados nunca saem do seu computador.

- A versão publicada na nuvem (Streamlit Cloud) serve **apenas para demonstração**, com os dados de exemplo
  fictícios incluídos. A própria aplicação avisa, no topo da página, se está a correr na nuvem.
- Nunca coloque bases com dados de doentes no GitHub nem noutros serviços na nuvem sem autorização do
  encarregado de proteção de dados. O `.gitignore` do projeto impede que `dados/`, `resultados/` e ficheiros
  Excel/CSV entrem no repositório (exceto a base de exemplo fictícia em `exemplos/`).

## Instalação (uma vez)

1. Instalar o **Python 3.10 ou mais recente** ([python.org](https://www.python.org/downloads/); no Windows,
   marcar *Add Python to PATH*).
2. Descarregar o SelPop: `git clone https://github.com/PJZoio/SelPop.git` — ou, no GitHub, *Code → Download ZIP*.
3. Na pasta do SelPop:

```bash
pip install -r requirements.txt
```

## Utilização

```bash
streamlit run app.py
```

Abre no browser em `http://localhost:8501` (*localhost* = este computador). Na barra lateral escolhe-se a base
— **Exemplo (fictícios)** para experimentar, ou **Os meus dados** — e um modelo de estudo (ou *Novo estudo em
branco*). Depois seguem-se os passos:

| Passo | O que se faz |
|---|---|
| **1 · Colunas** | Indicar a coluna do ID (obrigatória) e, se existirem, data, nº de amostra, idade, sexo e creatinina. Com estas, a ferramenta calcula variáveis derivadas (ex.: TFGe). Há campos opcionais para bases de UCI (admissão, VMI, ECMO, SOFA). |
| **2 · Grupos** | Definir casos e controlos. O **assistente rápido** faz isto escolhendo a coluna que separa os grupos e os valores; critérios mais finos escrevem-se como expressões. O funil mostra, em tempo real, quantos participantes ficam após cada critério. |
| **3 · Emparelhamento** | Controlos por caso, suplentes, variáveis de emparelhamento (exato e calibres) e reequilíbrio automático. |
| **4 · Variáveis** | Variáveis a comparar, o tipo (sugerido automaticamente) e o papel de cada uma (confundidor, preditor, interferente). |
| **5 · Escolha dos testes** | Diagnóstico de cada variável, teste recomendado com justificação, histogramas e gráficos Q–Q; possibilidade de escolher outro teste. |
| **6 · Resultados** | Números finais, tabela para o artigo, comparação detalhada, gráfico de equilíbrio, lista de amostras e botões para descarregar. |

### Como a ferramenta escolhe o teste (passo 5)

1. **Tipo de variável.** Categórica (sexo, diabetes, centro) ou contínua (idade, creatinina). Uma variável
   discreta com poucos valores pode ser categórica (códigos) ou ordinal (escala, ex.: SOFA): no segundo caso
   trata-se como contínua com teste não paramétrico. A ferramenta sugere o tipo e avisa nos casos ambíguos.
2. **Contínua → a distribuição é normal em cada grupo?** Teste de Shapiro–Wilk (p > 0,05 → compatível com a
   normal), histograma e gráfico Q–Q.
   - Normal nos dois grupos → **t de Student**; se as variâncias diferirem (Levene p < 0,05), **t de Welch**.
   - Não normal → **Mann–Whitney U**.
   - Avisos: com n < 15 por grupo o Shapiro–Wilk tem pouca potência; com n > 200 deteta desvios irrelevantes.
3. **Categórica → há poucos casos por célula?** Frequências esperadas da tabela grupo × categoria.
   - Todas ≥ 5 → **qui-quadrado de Pearson** (sem correção de Yates).
   - Alguma < 5 → **teste exato de Fisher** (por Monte Carlo, com semente fixa, quando há mais de 2 categorias).

A opção *Automático* aplica estas regras (com Student em vez de Welch, como nos artigos do grupo). Qualquer
escolha manual fica registada na configuração (`teste:` na variável) e nas notas de rodapé da tabela.

### Base de exemplo (fictícia)

`exemplos/exemplo_ficticio.csv` tem 400 participantes inventados (uma linha cada), com casos propositadamente
mais velhos e com mais diabetes e hipertensão do que os controlos. Serve para aprender a ferramenta e para
demonstrações. A configuração `estudos/exemplo_generico.yaml` mostra o emparelhamento e o reequilíbrio em ação
(72 → 51 casos emparelhados ao equilibrar IMC, diabetes e hipertensão). Gera-se de novo com
`python scripts/gerar_exemplo.py`.

## Adaptar a outra base

Na interface não é preciso editar ficheiros: escolha *Novo estudo (em branco)*, carregue a base e siga os
passos. No fim, descarregue o YAML (separador *Configuração*) para repetir a análise mais tarde ou na linha
de comandos. Para começar a partir de um ficheiro, copie `estudos/exemplo_generico.yaml` e altere:

- `dados.ficheiro` e `dados.id` (único campo obrigatório);
- `colunas` — mapeamento opcional (`n_amostra`, `data_colheita`, `idade`, `sexo` + `sexo_feminino`,
  `creatinina`; e, para UCI, `data_admissao`, `vmi_*`, `ecmo_*`, `sofa_*`, `creatinina_max/min`, …);
- `grupos.casos` / `grupos.controlos` — critérios (expressões pandas);
- `emparelhamento` e `equilibrio.variaveis`.

## Linha de comandos

```bash
python -m selpop estudos/exemplo_generico.yaml --publicacao --sav
python -m selpop.consolidar estudos/prognostico_lra.yaml estudos/diagnostico_lra.yaml estudos/diagnostico_drc.yaml \
    --rotulos M1 M2 M3 --saida resultados/Selecao_amostras.xlsx --publicacao --sav
```

`--publicacao` grava a tabela de características em Word (`--lingua en|pt`, inglês por omissão); `--sav` grava
os dados em SPSS e a sintaxe de validação. `selpop.consolidar` junta vários estudos num só Excel, com a lista
única de amostras a pedir ao biobanco.

## Saídas

| Saída | Conteúdo |
|---|---|
| Excel | LEIA-ME (métodos e ficheiros de entrada com SHA-256), Funil, Qualidade_dados, Comparacao_grupos, Tabela_publicacao, Equilibrio_antes_depois, Selecao (com colunas para o biobanco), Suplentes, Casos_sem_controlo, Iteracoes, Config |
| Word | Tabela de características: secções, mediana [IIQ] ou n (%), n por grupo, p com a letra do teste, notas de rodapé (testes, emparelhamento, dados em falta). Sem numeração: o número da tabela é dado no artigo. |
| `.sav` | Amostras selecionadas com `grupo` (1 = caso), `conjunto` (ID do caso emparelhado) e as variáveis comparadas, com rótulos. |
| `.sps` | Sintaxe SPSS/PSPP que repete os testes. Comparar com *Pearson Chi-Square*, *Fisher's Exact Test* e *Equal variances assumed* (ou *not assumed* para Welch). |

Validação com o PSPP 2.0 nos três estudos de UCI: **71/71 testes com p idêntico** ao SelPop (as diferenças
ficam só nas variáveis com mais de 2 categorias, em que o PSPP não tem Fisher exato).

## Papéis das variáveis

| Papel | Significado | Conta para o reequilíbrio? |
|---|---|---|
| `confundidor` (por omissão) | Deve estar equilibrado entre grupos | Sim |
| `preditor` | Candidato a explicar o desfecho: testa-se mas **não se emparelha** (emparelhar apagaria o sinal) | Não |
| `interferente` | Pode alterar a medição (ex.: hemólise, bilirrubina) — monitorizar | Não |
| `testar: false` | Descritiva (posterior à amostra, ex.: óbito, duração do internamento) | Não |

Outras opções por variável: `log: true` (SMD em escala log), `teste` (`t`, `welch`, `mw`, `chi2`, `fisher`),
`nome_en`, `rotulos`, `rotulos_en`, `secao` (secção na tabela do artigo).

## Variáveis derivadas

Calculadas só se as colunas necessárias estiverem mapeadas.

| Variável | Requer | Definição |
|---|---|---|
| `amostra`, `n_amostras_doente`, `dias_com_dados` | ID (e `n_amostra`) | Linha com amostra; amostras e linhas por participante |
| `tfge_dia`, `tfge_basal_proxy` | creatinina, idade, sexo | CKD-EPI 2021 (sem raça); basal = creatinina mínima do participante |
| `cr_unico`, `cr_dia`, `ur_unico`, `ur_dia` | creatinina/ureia (máx./mín. do dia) | Valor único no dia (referência inequívoca) |
| `dias_desde_admissao`, `duracao_uci_dias` | datas de colheita, admissão, alta | Dias desde a admissão; duração do internamento |
| `em_vmi_dia`, `dias_vmi_ate_amostra`, `em_ecmo_dia` | períodos de VMI/ECMO | Em VMI/ECMO no dia; dias de VMI à data da amostra |
| `sofa_total_dia`, `sofa_nao_renal_dia`, `sofa_cv_dia`, `sofa_nao_renal_nao_cv_dia` | SOFA total, renal, cardiovascular | Componentes e subtotais do SOFA |
| `lra_dia`, `lra_estadio_dia`, `sem_lra_janela`, `dias_ate_lra`, `dias_obs_apos`, `lra_subida`, `lra_admissao`, `lra_qualquer`, `lra_estadio_max` | datas e creatinina diária; `lra.calcular: true` | Critério de creatinina KDIGO 2012 sobre valores diários (aproximação: sem débito urinário nem creatinina pré-admissão) |

## Métodos

- **Testes**: como no passo 5. Contínuas em mediana (P25–P75); categóricas em n (%). Mann–Whitney com
  aproximação normal, correção de empates e sem correção de continuidade (como o SPSS/PSPP). Bilateral; α
  configurável; sem correção para comparações múltiplas (tabela descritiva).
- **SMD** (diferença média padronizada): não depende do tamanho da amostra — com grupos pequenos, "p ≥ α" não
  demonstra equilíbrio. Referência: |SMD| < 0,1.
- **Emparelhamento**: guloso global sem reposição, por etapas (exato + calibres), desempate aleatório com semente
  fixa. Não é ótimo no sentido estrito, mas é transparente e reprodutível.
- **Reequilíbrio**: se um confundidor candidato tiver p < α, é acrescentado ao emparelhamento (contínua →
  calibre de 0,5 DP; categórica → exato); fica a iteração com menos confundidores diferentes.

## Limitações e cuidados

- O reequilíbrio por p-valores é uma ajuda prática; as variáveis de emparelhamento devem ser escolhidas por
  plausibilidade científica.
- Casos e controlos são tratados como amostras independentes (prática habitual numa tabela de características).
  Análises que modelem o emparelhamento (ex.: regressão logística condicional) fazem-se à parte.
- Exigir controlos muito "puros" (ex.: sem nenhuma LRA) tende a selecionar participantes menos graves — confirme
  na tabela.
- A folha Qualidade_dados não corrige nada: indica registos a confirmar na fonte.

## Estrutura

```
app.py                 interface (Streamlit)
selpop/                biblioteca: io, derive, cohort, match, stats, report, publish, consolidar, pipeline
estudos/*.yaml         configurações (exemplo genérico; estudos de UCI)
exemplos/              base de exemplo fictícia
scripts/               gerar a base de exemplo
tests/                 testes automáticos (python -m pytest tests -q)
```
