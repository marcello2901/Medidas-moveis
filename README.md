# PBRTQC Planner

App Streamlit para configurar a medida móvel de pacientes a partir de uma planilha exportada.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fluxo

**Barra lateral:** upload (xlsx/xls/csv) → aba → colunas por lista suspensa (resultado, data, hora, idade, sexo) → filtros de teste e equipamento.
Idade aceita `11M` (meses), `15D` (dias) e `2A`. Resultados com vírgula decimal são aceitos; textos como `<0,5` são descartados e contados no resumo.

**1 · Estatística descritiva**
- Tamanho, tendência central (média, mediana, moda, média geométrica, média aparada), dispersão (DP, variância, CV, erro-padrão, IQR, mín/máx e acúmulo nos extremos), percentis (P1 a P99) e forma (assimetria, curtose).
- Histograma com boxplot, Q-Q normal, resumos por sexo, equipamento, faixa etária, dia da semana e turno. Exporta em .xlsx.

**2 · Preparo dos dados**
- Estatística descritiva e histogramas.
- Outliers: transformação (Box-Cox, log ou nenhuma) + cercas de Tukey (k configurável, opção iterativa). As cercas viram a sugestão de truncamento.
- Exclusão dos outliers é **opcional** (chave "Excluir outliers…"). Desligada, as etapas 3 e 4 usam todos os dados.
- Média/mediana por idade (painel de cima) × nº de resultados por idade (painel de baixo), opcionalmente por sexo. Perfil por hora do dia.

**3 · Configuração**
- Truncamento (excluir ou winsorizar), sexo, faixa de idade. Origem dos limites: intervalo de referência lido da planilha (coluna detectada automaticamente, ex.: "Ref. ranges"; se houver vários, escolhe-se qual), cercas de Tukey da etapa 2, percentis dos resultados ou manual (com campo obrigatório de fonte).
- Estatística: média móvel, mediana móvel, média móvel ponderada (pesos lineares, o mais recente pesa N) e EWMA (λ padrão = 2/(N+1)).
- Bloco **N-n**: bloco de N resultados elegíveis, novo ponto a cada n resultados.
- Limites alvo ± k·DP (alvo/DP da própria série ou manuais, como no Lab Operations); regra de pontos consecutivos.
- Média-alvo e DP: média/DP dos blocos plotados ou manuais (com campo de fonte).
- Gráfico com alarmes, taxa de falso alarme (% pontos, episódios, episódios/dia).
- "Estratificar por" (nenhuma, equipamento, sexo, equipamento e sexo): cada estrato com fluxo, alvo e DP próprios, como cada perfil do Lab Operations. Cartões de métricas por estrato (em abas por equipamento no modo equipamento e sexo), gráfico com uma cor por estrato (um painel por sexo no modo combinado), tabela de comparação e volume diário.
- "Elegíveis / dia": resultados dentro do truncamento e da população por dia com resultado, com o tempo aproximado para renovar um bloco de N (alerta acima de 48 h).

- Relatório em PDF (A4): identificação, preparo dos dados (exclusão de outliers, filtros), gráfico média × nº de resultados por idade, configuração (tipo, bloco, pulo, truncamento e fontes), alvo/DP/limites e fontes (por equipamento, se ativado), gráfico da medida móvel e campos de assinatura. Não inclui a simulação de bias.

**4 · Simulação de bias**
- Bias (% ou unidade) injetado em posições aleatórias do fluxo real e mantido até o fim do horizonte. Entra antes do truncamento.
- NPed = nº de resultados de pacientes liberados entre o início do erro e o primeiro alarme no sentido do erro.
- Tabela por bias: % detectado, ANPed (média dos detectados), MNPed (mediana), P25/P75/P95. Não detectado = "> horizonte".
- Curva de detecção (MNPed × bias, escala log); comparação opcional das 4 estatísticas com os mesmos pontos de início.
- Visualização de uma injeção específica e exportação (.xlsx) da configuração e dos resultados.

## Arquivos
- `app.py`: interface
- `mm_core.py`: cálculos (sem dependência do Streamlit, reutilizável em scripts)
- `report.py`: geração do relatório PDF (reportlab + matplotlib)
