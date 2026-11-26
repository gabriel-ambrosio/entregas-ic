"""Joga partidas com o modelo treinado, ou com ações aleatórias para testar a comunicação.

    python jogar.py --aleatorio --simulador      teste rápido, sem jogo e sem Stable-Baselines3
    python jogar.py --aleatorio                  testa os mods no jogo de verdade
    python jogar.py                              joga com o último modelo salvo
    python jogar.py --modelo saida/modelos/final.zip --episodios 5
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from config import Config


def descrever(env, a: int) -> str:
    acao = env._decodificar(a)
    if acao[0] == "iniciar":
        return "iniciar rodada"
    if acao[0] == "colocar":
        x, y = env._grade.centros[acao[2]]
        return f"colocar {env.cfg.torres[acao[1]]} em ({x:.0f}, {y:.0f})"
    torre = env._slots[acao[1]] or {}
    alvo = f"{torre.get('tipo', '?')} em ({torre.get('x', 0):.0f}, {torre.get('y', 0):.0f})"
    return f"melhorar {alvo}, caminho {acao[2]}" if acao[0] == "melhorar" else f"vender {alvo}"


def main() -> None:
    p = argparse.ArgumentParser(description="Joga BTD6 com o agente")
    p.add_argument("--modelo", type=str, default=None, help="arquivo .zip do modelo (padrão: o mais recente)")
    p.add_argument("--episodios", type=int, default=1)
    p.add_argument("--aleatorio", action="store_true", help="ações aleatórias válidas (não precisa de modelo)")
    p.add_argument("--simulador", action="store_true", help="usa o simulador falso em vez do jogo")
    p.add_argument("--pasta", type=str, default=None, help="pasta UserData do BTD6 (substitui o config.py)")
    p.add_argument("--quieto", action="store_true", help="não mostra cada ação")
    args = p.parse_args()

    cfg = Config()
    if args.pasta:
        cfg.pasta_userdata = args.pasta
    if args.simulador:
        from simulador_falso import ligar_simulador
        ligar_simulador(cfg)

    from ambiente import AmbienteBTD6
    env = AmbienteBTD6(cfg)

    modelo = None
    if not args.aleatorio:
        from sb3_contrib import MaskablePPO
        from treinar import ultimo_modelo
        caminho = args.modelo or ultimo_modelo(os.path.join(cfg.pasta_saida, "modelos"))
        if not caminho or not os.path.exists(caminho):
            raise SystemExit("Nenhum modelo encontrado. Treine antes ou use --aleatorio.")
        print(f"Modelo: {caminho}")
        modelo = MaskablePPO.load(caminho)

    rng = np.random.default_rng()
    for ep in range(1, args.episodios + 1):
        obs, info = env.reset()
        retorno, passos = 0.0, 0
        while True:
            mascara = env.action_masks()
            if modelo is None:
                a = int(rng.choice(np.flatnonzero(mascara)))
            else:
                a, _ = modelo.predict(obs, action_masks=mascara, deterministic=True)
                a = int(a)
            texto = descrever(env, a)
            obs, r, terminou, truncou, info = env.step(a)
            retorno += r
            passos += 1
            if not args.quieto:
                print(f"  rodada {info['rodada']:3d} | vida {info['vida']:4.0f} | $ {info['dinheiro']:6.0f} | "
                      f"{texto} | recompensa {r:+.2f}")
            if terminou or truncou:
                break
        fim = "vitória" if info["vitoria"] else "derrota" if info["derrota"] else "interrompida"
        print(f"[partida {ep}] {fim} | próxima rodada {info['rodada']} | vida {info['vida']:.0f} | "
              f"retorno {retorno:.2f} | {passos} ações")


if __name__ == "__main__":
    main()
