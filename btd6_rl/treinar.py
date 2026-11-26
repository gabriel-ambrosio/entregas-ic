"""Treina o agente com MaskablePPO (sb3-contrib / Stable-Baselines3).

    python treinar.py                      treina no jogo (partida aberta, mods carregados)
    python treinar.py --simulador          treina no jogo de mentira, para testar a instalação
    python treinar.py --continuar          continua do último modelo salvo
    python treinar.py --continuar caminho.zip --passos 50000
    python treinar.py --quieto             só uma linha por partida, sem as rodadas
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os

from config import Config


def ultimo_modelo(pasta_modelos: str) -> str | None:
    arquivos = glob.glob(os.path.join(pasta_modelos, "*.zip"))
    return max(arquivos, key=os.path.getmtime) if arquivos else None


def main() -> None:
    p = argparse.ArgumentParser(description="Treino de RL para Bloons TD 6")
    p.add_argument("--passos", type=int, default=None, help="total de passos (ações) de treino")
    p.add_argument("--pasta", type=str, default=None, help="pasta UserData do BTD6 (substitui o config.py)")
    p.add_argument("--simulador", action="store_true", help="usa o simulador falso em vez do jogo")
    p.add_argument("--quieto", action="store_true", help="mostra só uma linha por partida (sem as rodadas)")
    p.add_argument("--continuar", nargs="?", const="", default=None,
                   help="continua de um modelo salvo (sem caminho = o mais recente)")
    args = p.parse_args()

    cfg = Config()
    if args.pasta:
        cfg.pasta_userdata = args.pasta
    if args.simulador:
        from simulador_falso import ligar_simulador
        ligar_simulador(cfg)
        print(f"Simulador falso ligado em: {cfg.pasta_userdata}")
    passos = args.passos or cfg.passos_totais

    # Importados aqui para o --help funcionar mesmo sem as bibliotecas instaladas
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    from ambiente import AmbienteBTD6

    pasta_modelos = os.path.join(cfg.pasta_saida, "modelos")
    os.makedirs(pasta_modelos, exist_ok=True)

    caminho_modelo = None
    if args.continuar is not None:
        caminho_modelo = args.continuar or ultimo_modelo(pasta_modelos)
        if not caminho_modelo or not os.path.exists(caminho_modelo):
            raise SystemExit("Nenhum modelo salvo para continuar.")

    base = AmbienteBTD6(cfg)
    env = Monitor(
        base,
        filename=os.path.join(cfg.pasta_saida, "episodios"),   # -> episodios.monitor.csv
        info_keywords=("rodada", "vida", "vitoria"),
        override_existing=caminho_modelo is None,
    )
    env = ActionMasker(env, lambda _: base.action_masks())

    class LogEpisodios(BaseCallback):
        """Uma linha no console por rodada concluída e uma por partida terminada.

        A loss só é recalculada quando o PPO atualiza a rede (a cada n_steps passos), então
        o valor mostrado é o da última atualização e se repete até a próxima.
        """

        def __init__(self, por_rodada: bool):
            super().__init__()
            self.n = 0
            self.por_rodada = por_rodada
            self.rodada_anterior = None
            self.recompensa_rodada = 0.0     # soma desde o fim da rodada anterior (compras + rodada)
            self.recompensa_partida = 0.0
            self.acoes_rodada = 0
            self.loss = None
            self.loss_valor = None

        def _ler_loss(self) -> None:
            # O PPO grava as métricas no logger ao treinar; guardamos a última leitura
            try:
                valores = self.logger.name_to_value
                if "train/loss" in valores:
                    self.loss = float(valores["train/loss"])
                if "train/value_loss" in valores:
                    self.loss_valor = float(valores["train/value_loss"])
            except Exception:
                pass

        def _texto_loss(self) -> str:
            if self.loss is None:
                return "loss - (antes da 1ª atualização)"
            return f"loss {self.loss:.4f}" + (f" (valor {self.loss_valor:.4f})" if self.loss_valor is not None else "")

        def _on_rollout_start(self) -> None:
            self._ler_loss()

        def _on_step(self) -> bool:
            self._ler_loss()
            infos = self.locals.get("infos", [])
            fins = self.locals.get("dones", [])
            recompensas = self.locals.get("rewards", [0.0] * len(infos))
            for info, fim, recompensa in zip(infos, fins, recompensas):
                self.recompensa_rodada += float(recompensa)
                self.recompensa_partida += float(recompensa)
                self.acoes_rodada += 1

                rodada = info.get("rodada")
                rodada_acabou = (self.rodada_anterior is not None and rodada is not None
                                 and rodada > self.rodada_anterior)
                if rodada_acabou and not fim and self.por_rodada:
                    print(f"  [partida {self.n + 1}] rodada {rodada - 1} concluída | vida {info.get('vida'):.0f} | "
                          f"$ {info.get('dinheiro'):.0f} | {info.get('torres')} torres | "
                          f"recompensa {self.recompensa_rodada:+.2f} (partida {self.recompensa_partida:+.2f}) | "
                          f"{self._texto_loss()} | passo {self.num_timesteps} ({self.acoes_rodada} na rodada)")
                if rodada_acabou or fim:
                    self.recompensa_rodada = 0.0
                    self.acoes_rodada = 0
                self.rodada_anterior = None if fim else rodada

                if fim:
                    self.n += 1
                    ep = info.get("episode", {})
                    print(f"[partida {self.n}] chegou à rodada {info.get('rodada')} | vida {info.get('vida'):.0f} | "
                          f"vitória: {info.get('vitoria')} | retorno {ep.get('r', self.recompensa_partida):.2f} | "
                          f"{ep.get('l', 0)} ações | {self._texto_loss()} | passo {self.num_timesteps}")
                    self.recompensa_partida = 0.0
            return True

    tensorboard = os.path.join(cfg.pasta_saida, "tensorboard") if importlib.util.find_spec("tensorboard") else None

    if caminho_modelo:
        print(f"Continuando de: {caminho_modelo}")
        modelo = MaskablePPO.load(caminho_modelo, env=env, tensorboard_log=tensorboard)
    else:
        modelo = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            ent_coef=cfg.ent_coef,
            policy_kwargs=dict(net_arch=list(cfg.camadas)),
            tensorboard_log=tensorboard,
            seed=cfg.semente,
            verbose=1,
        )

    callbacks = CallbackList([
        CheckpointCallback(save_freq=cfg.salvar_a_cada, save_path=pasta_modelos, name_prefix="ppo_btd6"),
        LogEpisodios(por_rodada=not args.quieto),
    ])

    final = os.path.join(pasta_modelos, "final.zip")
    try:
        modelo.learn(total_timesteps=passos, callback=callbacks, reset_num_timesteps=caminho_modelo is None)
    except KeyboardInterrupt:
        print("\nTreino interrompido pelo teclado.")
    finally:
        modelo.save(final)
        print(f"Modelo salvo em: {final}")


if __name__ == "__main__":
    main()
