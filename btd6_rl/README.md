# Pipeline de RL para Bloons TD 6

Agente de deep reinforcement learning (MaskablePPO, Stable-Baselines3) que joga BTD6 pelos mods **Exportador** e **Player**.

## Arquivos

| Arquivo | Para que serve |
|---|---|
| `mods/Exportador.cs` | Mod v2: grava `estado.json` e `mapa.json`, congela o jogo, detecta derrota, acelera o jogo |
| `mods/Player.cs` | Mod v2: lê `acoes.json` sozinho, confirma cada ação, reinicia a partida |
| `config.py` | Todas as configurações (pasta do jogo, torres, grade, recompensa, PPO) |
| `ponte.py` | Troca de arquivos com os mods |
| `mapa.py` | Grade de posições e geometria do mapa |
| `ambiente.py` | Ambiente Gymnasium: observação, ações, máscara, recompensa |
| `treinar.py` | Treino |
| `jogar.py` | Joga com o modelo treinado ou com ações aleatórias |
| `simulador_falso.py` | Jogo de mentira com o mesmo protocolo, para testar sem abrir o BTD6 |

## Como funciona

1. O Exportador grava o estado (vida, dinheiro, torres, bloons da próxima rodada, loja com custos) e congela o jogo.
2. O Python lê o estado, a rede escolhe **uma** ação e grava `acoes.json`.
3. O Player executa, espera a simulação confirmar e grava `ultimo_resultado.json`.
4. Se foi uma compra, o Exportador exporta de novo na hora. Se foi "iniciar rodada", exporta quando a rodada acabar (ou na derrota).
5. No fim da partida o Python manda `reiniciar` e começa outro episódio.

**Ações do agente:** iniciar rodada, colocar torre (tipo × célula da grade), melhorar torre (slot × caminho) e, opcionalmente, vender.

**Recompensa:** +1 por rodada concluída, menos 5 × fração da vida perdida, +10 na vitória, −5 na derrota, −0,05 por ação recusada.

**Máscara de ações:** o agente só vê ações possíveis (dinheiro suficiente, posição livre, fora da pista e da água, ao alcance da pista, upgrade permitido). Posições que o jogo recusa são memorizadas em `saida/celulas_bloqueadas.json` e nunca mais tentadas.

## Preparação

### 1. Mods
1. Substitua o código dos dois mods pelos arquivos de `mods/` e compile.
2. No menu de mods do jogo:
   - **Player → Modo automático:** ligado (padrão).
   - **Exportador → Velocidade do jogo no treino:** comece com 0, depois suba (ex.: 5 ou 10).
3. Nas configurações do BTD6, **desligue o início automático de rodada** (auto start).

### 2. Python
```
pip install -r requirements.txt
```
Em `config.py`, ajuste:
- `pasta_userdata`: pasta `UserData` do seu BTD6;
- `rodada_final`: 40 (fácil), 60 (médio) ou 80 (difícil);
- `torres`: lista de torres que o agente pode usar.

## Ordem de teste recomendada

```
python jogar.py --aleatorio --simulador        # 1. Python funcionando (não precisa do jogo)
python treinar.py --simulador --passos 20000   # 2. Stable-Baselines3 funcionando
python jogar.py --aleatorio                    # 3. mods funcionando (abra uma partida antes)
python treinar.py                              # 4. treino de verdade
```

No passo 3, abra o mapa e a dificuldade escolhidos e deixe a partida na rodada inicial. O agente deve comprar torres aleatórias, iniciar rodadas e reiniciar sozinho ao perder.

Outros comandos:
```
python treinar.py --continuar          # continua do último modelo salvo
python jogar.py                        # joga com o último modelo
python jogar.py --modelo saida/modelos/final.zip --episodios 5
tensorboard --logdir saida/tensorboard
```

O treino salva um modelo a cada 2.000 passos em `saida/modelos/` e uma linha por partida em `saida/episodios.monitor.csv`. Ctrl+C salva e sai.

## O que foi testado

- **Testado:** ambiente, ponte, grade, máscaras, recompensa, reinício, vitória, derrota e tempo esgotado, todos contra o simulador falso com agente aleatório.
- **Não testado:** os dois mods não foram compilados nem rodados (não tenho o jogo) e `treinar.py` não foi executado com o Stable-Baselines3 (não estava instalado no meu ambiente). Rode os quatro passos acima na ordem para pegar qualquer problema cedo.

## Pontos dos mods que podem precisar de ajuste

Tudo que depende da versão do jogo está isolado. Se algo não compilar ou não funcionar, olhe aqui primeiro:

| Onde | O que | Se falhar |
|---|---|---|
| Player, `IdJogador` | `bridge.GetInputId()` | Procure no `UnityToSimulation` o membro que devolve o id de input local |
| Player, `ReiniciarNoJogo` | Chama `InGame.Restart` por reflexão; a assinatura usada aparece no console | Se o reinício não acontecer, veja a assinatura no console e troque os argumentos |
| Exportador, `VelocidadeTreino` | `ModSettingInt` com `min`, `max`, `slider` | Remova as propriedades que não existirem |
| Exportador, loja | `towerModel.range` e `towerModel.areaTypes` | Remova o campo; o Python usa valores padrão |
| Exportador, `LiberarPassosDeSimulacao` | Procura `TimeManager.maxSimulationStepsPerUpdate` por reflexão | Só um aviso no console; a velocidade pode ficar limitada a ~3x |
| Exportador, derrota | Detectada quando a vida chega a 0 | Se o modo não zera a vida, o Python desiste após `timeout_rodada` e reinicia |

Também vale saber:
- A vitória é decidida pelo Python (`proxima_rodada > rodada_final`), então `rodada_final` precisa estar certo.
- `ultimo_resultado.json` mudou de formato: agora é `{ "id": ..., "resultados": [...] }`.
- O histórico `partida_*.json` só é salvo se a opção do Exportador estiver ligada.
- Janelas do jogo (subida de nível, recompensas) podem travar a partida. Se o treino parar, o Python reinicia a partida sozinho depois do tempo limite.

## Expectativa realista

O gargalo é a velocidade do jogo, não a rede. Uma partida de 40 rodadas leva alguns minutos mesmo acelerada, então espere **dias** de treino para algo consistente. O que mais ajuda:

- subir a **velocidade do jogo** no Exportador;
- usar **poucas torres** em `torres` (4 a 6);
- usar uma **grade menor** (ex.: 16 × 12);
- treinar primeiro no **fácil**.

Se mudar `torres`, a grade, `max_torres` ou `permitir_venda`, o formato da rede muda e os modelos antigos deixam de servir.

## Próximos passos possíveis

- Vários mapas: trocar o vetor de observação por planos de imagem (pista, água, torres) com uma rede convolucional.
- Heróis e habilidades: precisam de novas ações no Player.
- Pré-treino por imitação a partir de partidas suas (o histórico do Exportador já grava os estados).
