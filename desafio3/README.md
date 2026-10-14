# Desafio 3 — O melhor treino para o nosso problema
GBC073 · Inteligência Computacional · FACOM/UFU
Otimizadores e agendas de taxa de aprendizado para o **preditor de rodada** do
BTD6 Magic AI, observados pelos cinco sinais do treino.
## O problema
O projeto tem duas partes. Um **algoritmo genético** escolhe onde colocar as
torres, e a função de aptidão que ele consulta é uma **rede neural** que
estima a probabilidade de a rodada passar sem perda de vida. O AG não tem
gradiente — quem treina por gradiente é a rede, e é ela que este desafio
estuda.
Entrada: a composição de bloons da rodada (46 valores) e até 12 torres
codificadas em 81 dimensões cada. Arquitetura *Deep Sets*: uma camada
compartilhada processa cada torre, a soma das 12 torna a saída invariante à
ordem, e o resultado se junta às features da rodada. 6.721 parâmetros.
Saída: um logit; perda de entropia cruzada binária.
Dados: 92.312 rodadas após limpeza, 39,0% positivas, divididas 80/10/10.
## Três decisões de implementação
1. **A taxa de aprendizado é argumento de linha de comando.** O enunciado diz
  que *comparar otimizadores com η não ajustada invalida a comparação*, então
  otimizador, η, agenda e semente são todos parâmetros explícitos de
  `treino.py`, e cada linha de `experimentos.csv` carrega os seus.
2. **O rótulo é `y = 1` para a rodada que passou SEM perder vida**
  (`Health Lost == 0`). Vale fixar isso porque a intuição puxa para o
  contrário: o modelo estima a chance de *sobreviver* intacto, não a de
  perder vida.
3. **O gargalo é o pré-processamento, não o treino.** O cálculo de cobertura
  é O(|pontos do mapa|) e seria chamado cerca de um milhão de vezes, mas
  depende só de (torre, upgrade, posição): 281.960 combinações distintas.
  Memoizado e guardado em cache, para que as ~56 execuções do protocolo não
  paguem o custo de novo.
A saída da rede é o **logit** e não a probabilidade, para usar
`BCEWithLogitsLoss` — numericamente estável. Para o AG, `modelo.probabilidade()`
## Onde isto mora
Os dados brutos (`data/`) e os utilitários de alcance das torres (`utils/`)
ficam um nível acima, na raiz do pacote. Só o passo `--construir` precisa
deles — gerado o cache, todo o resto roda sozinho. Para apontar para outro
lugar, use `BTD6_RAIZ`:
```bash
BTD6_RAIZ=/caminho/que/contem/data/e/utils python -m desafio3.dados --construir
```
O cache (`cache/dados.pt`, 362 MB) é gerado por esse comando e pode ser
apagado a qualquer momento — basta rodá-lo de novo.
## Como reproduzir
Rode a partir da raiz de `gbc073/`:
```bash
pip install -r ../requirements.txt
python -m desafio3.dados --construir       # cache de tensores (~4 min, 362 MB)
python -m desafio3.protocolo --etapa tudo  # as quatro etapas do protocolo
python -m desafio3.painel                  # painel.pdf
python -m desafio3.slide                   # slide.pdf
```
Qualquer linha de `experimentos.csv` se reproduz sozinha:
```bash
python -m desafio3.treino --otimizador muon --lr 0.02 \
      --agenda warmup_cosseno --semente 0 --passos 3000 --lote 256
```
A divisão dos dados é determinística (`semente_divisao=0`), a ordem dos lotes
vem da semente da execução, e os pesos iniciais de `torch.manual_seed(semente)`.
## O protocolo
| etapa | o que | orçamento |
|---|---|---|
| 1 Baseline | receita atual do grupo, 3 sementes | 3000 passos, lote 256 |
| 2 Varredura de η | ≥5 valores log por otimizador, warmup+cosseno, 1 semente | 1500 passos |
| 3 Agendas | 2 melhores otimizadores + Muon, 4 agendas, 1 semente | 1500 passos |
| 4 Confirmação | 3 melhores combinações, 3 sementes | 3000 passos |
O orçamento reduzido das etapas 2–3 é **igual para todos** e está declarado em
`protocolo.py`. A seleção usa sempre a perda de validação; o conjunto de teste
fica intocado até a leitura final. Toda execução entra em `experimentos.csv`,
inclusive as que divergiram.
Se o melhor η cair na borda da grade, o script **estende a grade** automaticamente,
como o enunciado pede.
## Instrumentação
A cada `k = 25` passos (o enunciado pede 10 ≤ k ≤ 50), global e nas camadas
`torre` (primeira), `mlp1` (meio) e `saida` (última):
| sinal | onde | por quê |
|---|---|---|
| perda de treino | por lote | convergência, platôs, picos |
| perda de validação | a cada 250 passos | sobreajuste |
| ‖∇θ L‖₂ | **antes** do clipping | estabilidade |
| ‖θ‖₂ | global e por camada | efeito de η, weight decay |
| ‖θₜ₊₁ − θₜ‖/‖θₜ‖ | por camada | quanto cada camada muda; ~10⁻³ é saudável |
| η efetiva | do otimizador | confirma que a agenda fez o esperado |
## Muon
Obrigatório pelo enunciado. Usa `torch.optim.Muon` (PyTorch ≥ 2.9; aqui 2.11),
que ortogonaliza o momento por Newton–Schulz antes do passo, de modo que todas
as direções singulares recebem o mesmo tamanho — inclusive as raras, que SGD e
Adam quase ignoram.
Vale só para matrizes 2-D das camadas ocultas. Em `modelo.py`,
`parametros_muon_e_adamw()` separa: `torre`, `perm`, `mlp1` e `mlp2` vão para o
Muon (6.624 parâmetros); vieses e a camada `saida` ficam no AdamW (97
parâmetros). **A mesma agenda** é aplicada aos dois otimizadores.
## Arquivos
```
desafio3/
├── dados.py        pré-processamento, cache de tensores, divisão determinística
├── modelo.py       Deep Sets em PyTorch
├── treino.py       treino configurável e instrumentado  ← entregável
├── protocolo.py    as quatro etapas
├── painel.py       gera painel.pdf
├── slide.py        gera slide.pdf
├── experimentos.csv  uma linha por execução (74)
├── painel.pdf      três páginas, os sinais e a varredura
├── slide.pdf       um slide, 3 minutos
├── logs/           série temporal dos sinais, uma por execução
└── cache/          tensores pré-processados (ignorado pelo git)
```
## Resultados
74 execuções, 38 min de CPU, nenhuma com NaN.
| | configuração | perda val. | acurácia |
|---|---|---|---|
| baseline | adam · η=0,001 · constante | 0,4593 ± 0,0056 | 77,6% |
| **vencedora** | **adamw · η=0,03 · warmup+cosseno** | **0,2590 ± 0,0069** | **88,5%** |
| 2º | adamw · η=0,03 · wsd | 0,2635 ± 0,0041 | 88,4% |
| 3º | muon · η=0,03 · warmup+cosseno | 0,2674 ± 0,0046 | 88,3% |
**Δ perda = −0,2003**, contra ±0,0125 de variação entre sementes: o ganho é real.
Entre a vencedora e o Muon a diferença é 0,0084 contra ±0,0115 de variação —
**empatam**, e o AdamW é preferido por ser ~2,3× mais rápido por passo nesta CPU.
### O diagnóstico
O baseline **não estava mal otimizado, estava sobre-regularizado**. O sinal que
denuncia é ‖θ‖: despenca de 11 para 4 e estaciona, enquanto a perda empaca em
0,46 — o espelho da linha *"‖θ‖ cresce sem estabilizar → falta regularização"*
da tabela de sintomas do enunciado. Na vencedora, ‖θ‖ sobe até 80.
A causa é a penalidade L2 **acoplada** do Adam: em PyTorch, `weight_decay` no
Adam entra no gradiente e é depois dividido por √v, o que a amplifica quando os
gradientes são pequenos. Ablação controlada, mesmo η=0,01 e mesmo orçamento:
| configuração | perda val. |
|---|---|
| adam, wd=0,01 | 0,4391 |
| adam, **wd=0** | **0,2737** |
| adamw, wd=0 | 0,2768 |
Zerar o weight decay recupera quase toda a distância até o AdamW. Ou seja, o
ganho não vem da adaptatividade nem da agenda — vem de **desacoplar a
penalidade**, o que por sua vez permite subir η de 0,001 para 0,03 sem esmagar
os pesos.
### Correção de protocolo feita no meio do caminho
A primeira rodada do Muon usou η=3e-4 na parte AdamW (o padrão do código), e
não *"o melhor η já encontrado para o AdamW"* como o §3 determina. O erro era
visível no sinal: a razão de atualização da camada `saida` ficava duas ordens de
grandeza abaixo de `torre` e `mlp1`. O braço do Muon foi **refeito** com
η=0,03 na parte AdamW (execuções `e2b`, `e3b`, `e4b`); as execuções erradas
continuam no CSV, como manda a regra de reportar tudo. Depois da correção as
três camadas passam a andar em sincronia — que é o efeito esperado da
ortogonalização — e a perda melhorou de 0,2706 para 0,2674.
### Instabilidade
Nenhuma execução do protocolo divergiu, então três foram rodadas de propósito
fora da faixa (`e5_*`, com clipping desligado) para o painel ter o contraste
exigido. A de `muon η=1` é a ilustração do sintoma: ‖∇θL‖ salta para 5,4·10²,
a perda para 93, e a acurácia colapsa para 61,47% — exatamente a fração da
classe majoritária, isto é, o modelo passa a prever sempre a mesma classe.
### Reprodutibilidade
`e4_adamw_warmup_cosseno_s0` registrou 0,258661; refeito pela linha de comando
do CSV, devolveu 0,2587.