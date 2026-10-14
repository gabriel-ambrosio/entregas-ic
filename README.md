# BTD6 Magic AI — modelo em PyTorch
Uma rede neural estima a chance de uma combinação de torres passar a rodada, e
um algoritmo genético usa essa estimativa para montar o plano da partida.
## Como funciona
1. **O preditor de rodada** — rede *Deep Sets* que recebe a composição de
  bloons da rodada (46 valores) e até 12 torres (81 dimensões cada), e estima
  a probabilidade de passar a rodada **sem perder vida**. Uma camada
  compartilhada processa cada torre e a soma das 12 torna a saída invariante à
  ordem em que elas aparecem. 6.721 parâmetros.
2. **O algoritmo genético** — procura a melhor combinação de 12 torres para
  cada rodada, usando a saída do preditor como função de aptidão. O AG não tem
  gradiente; quem treina por gradiente é a rede.
---
# Passo a passo
Todos os comandos rodam **a partir desta pasta** (`btd6-model/`). Os passos 1 a
4 são a sequência mínima para ver o modelo jogando. O passo 5 é o estudo de
otimizadores e não depende dos passos 3 e 4.
## 1 · Instalar as dependências
```bash
pip install -r requirements.txt
```
Python 3.10 ou mais novo, e **torch 2.9 ou superior** — o otimizador Muon,
usado no estudo do passo 5, só existe a partir dessa versão. Confira:
```bash
python -c "import torch; print(torch.__version__, hasattr(torch.optim,'Muon'))"
# 2.11.0+cu130 True
```
## 2 · Preparar os dados · uma vez só · ~4 min
```bash
python -m desafio3.dados --construir
```
Lê as 92 mil rodadas de `data/`, calcula a cobertura de cada torre e grava
tensores prontos em `desafio3/cache/dados.pt` (362 MB).
É o passo mais lento, e é por isso que ele existe: o cálculo de cobertura é
O(pontos do mapa) e seria chamado cerca de um milhão de vezes ao longo dos
treinos. Feito uma vez e guardado, deixa de ser o gargalo.
Para conferir depois:
```bash
python -m desafio3.dados --info
# dim_outros=46  dim_torre=81  n_torres=12
# tamanhos: {'treino': 73849, 'val': 9231, 'teste': 9232}
```
O cache pode ser apagado quando quiser — basta rodar `--construir` de novo.
## 3 · Treinar o modelo · ~35 s
```bash
python -m desafio3.treino --otimizador adamw --lr 0.03 --agenda warmup_cosseno \
      --passos 3000 --salvar-pesos checkpoints/preditor_rodada.pt
```
Esta é a receita vencedora do estudo do passo 5. Ao fim:
```
 passo  2999  treino 0.2075  val 0.2587  acc 0.8873  lr 0.00e+00
 pesos salvos em checkpoints/preditor_rodada.pt
```
O arquivo guarda os pesos, a arquitetura e a receita que o produziu, então o
passo seguinte reconstrói o modelo sozinho.
Para experimentar outras configurações:
```bash
python -m desafio3.treino --help
#  --otimizador {sgd,sgd_nesterov,adam,adamw,muon,rmsprop}
#  --agenda {constante,cosseno,warmup_cosseno,onecycle,wsd}
#  --lr  --passos  --lote  --wd  --semente  --clip  --k-registro
```
Cada execução grava a série dos sinais em `desafio3/logs/<tag>.csv` e uma linha
em `desafio3/experimentos.csv`.
## 4 · Rodar a simulação
```bash
python simular.py --ate-rodada 20                 # trecho curto, para ver funcionando
python simular.py                                 # partida CHIMPS completa, rodadas 6-100
python simular.py --iteracoes 50 --populacao 100  # busca mais exaustiva
```
Saída, uma linha por trecho de rodadas:
```
modelo: adamw η=0.03 warmup_cosseno | val perda 0.2587 acurácia 0.8873
rodada 5 — chance 93.70% — custo $480
 torres: Druid-[0, 0, 0] cobertura 0.30 (-36.0, -8.2)
rodada 6 — chance 92.02% (mesma composição)
```
O tempo depende de `--iteracoes` e `--populacao`: o AG avalia a rede uma vez
por indivíduo a cada geração, e a partida inteira encadeia dezenas de buscas.
Comece com `--ate-rodada 20` para calibrar antes de soltar as 100 rodadas.
Para usar outros pesos: `--pesos caminho/para/outro.pt`.
## 5 · Reproduzir o estudo de otimizadores · ~25 min
Precisa apenas do cache do passo 2.
```bash
python -m desafio3.protocolo --etapa tudo  # 56 execuções
python -m desafio3.painel                  # gera desafio3/painel.pdf
python -m desafio3.slide                   # gera desafio3/slide.pdf
```
Dá para rodar uma etapa por vez com `--etapa 1` a `4`. Os resultados já
acompanham o pacote em `desafio3/experimentos.csv` e `desafio3/logs/`, então
`painel` e `slide` funcionam **sem** refazer o protocolo.
O protocolo, os sinais registrados e o diagnóstico estão no
[README do desafio 3](desafio3/README.md).
---
## Se algo der errado
| mensagem | o que fazer |
|---|---|
| `cache ausente: .../dados.pt` | rode o passo 2 |
| `não há pesos em .../preditor_rodada.pt` | rode o passo 3 |
| `utilitários de torre não encontrados` | rode a partir de `btd6-model/`, ou defina `BTD6_RAIZ` apontando para a pasta que contém `data/` e `utils/` |
| `No module named 'func_timeout'` | rode o passo 1 |
| `module 'torch.optim' has no attribute 'Muon'` | atualize o torch para 2.9+ |
## O estudo, em uma frase
A receita de partida (`adam`, η=0,001, sem agenda) perdia para `adamw` η=0,03
com warmup+cosseno por **0,4593 → 0,2590** de perda de validação (77,6% →
88,5% de acurácia). O diagnóstico — lido em ‖θ‖ e confirmado por ablação — é
que o treino de partida estava **sobre-regularizado** pela penalidade L2
acoplada do Adam.
## Estrutura
```
btd6-model/
├── simular.py              AG + inferência
├── genetic.py              algoritmo genético (numpy)
├── requirements.txt
├── checkpoints/            pesos gerados no passo 3 (não existe no início)
├── data/                   rodadas, mapa, posições possíveis, histórico
├── utils/
│   ├── tower_ranges.py     geometria e alcance das torres
│   ├── tower_encodings.py  codificação de 81 dimensões
│   └── mapa.py             leitura do mapa e das rodadas
└── desafio3/               estudo de otimizadores — ver README de lá
   ├── dados.py            pré-processamento e cache
   ├── modelo.py           o preditor de rodada
   ├── treino.py           treino configurável e instrumentado
   ├── protocolo.py        as quatro etapas do estudo
   ├── painel.py           gera painel.pdf
   ├── slide.py            gera slide.pdf
   ├── experimentos.csv    74 execuções
   ├── painel.pdf · slide.pdf
   └── logs/               série temporal dos sinais por execução
```