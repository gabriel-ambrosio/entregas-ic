"""Jogo de mentira que fala o mesmo protocolo de arquivos dos mods Exportador e Player.

Serve só para testar o pipeline (ambiente, máscaras, treino) sem abrir o BTD6.
NÃO é uma simulação fiel do jogo: a "força" das torres e dos bloons é inventada.

Uso direto:   python simulador_falso.py PASTA
No treino:    python treinar.py --simulador
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

MOTIVO_POSICAO = "o jogo recusou a posição (pista, água, obstáculo ou fora do mapa)"

# tipo: (custo, raio, alcance, áreas, força)
LOJA: Dict[str, Tuple[float, float, float, List[str], float]] = {
    "DartMonkey":   (170, 6, 32, ["land"], 3.0),
    "TackShooter":  (240, 6, 23, ["land"], 4.5),
    "BombShooter":  (445, 7, 40, ["land"], 8.0),
    "SniperMonkey": (300, 6, 20, ["land"], 4.0),
    "NinjaMonkey":  (340, 6, 40, ["land"], 6.5),
    "WizardMonkey": (320, 6, 40, ["land"], 6.0),
    "MonkeySub":    (275, 7, 42, ["water"], 5.5),
}

LIMITES = (-150.0, 150.0, -110.0, 110.0)
PISTA = [(-150, 60), (60, 60), (60, -10), (-80, -10), (-80, -70), (150, -70)]
MEIA_LARGURA_PISTA = 9.0
AGUA = [(85, 5), (140, 5), (140, 50), (85, 50)]
OBSTACULO = [(-140, -40), (-110, -40), (-110, -5), (-140, -5)]
NOMES_BLOONS = ["Red", "Blue", "Green", "Yellow", "Pink", "Black", "White",
                "Lead", "Zebra", "Rainbow", "Ceramic", "Moab"]


def _dentro(pol: List[Tuple[float, float]], x: float, y: float) -> bool:
    dentro = False
    j = len(pol) - 1
    for i in range(len(pol)):
        xi, yi = pol[i]
        xj, yj = pol[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            dentro = not dentro
        j = i
    return dentro


def _dist_pista(x: float, y: float) -> float:
    menor = float("inf")
    for (ax, ay), (bx, by) in zip(PISTA[:-1], PISTA[1:]):
        abx, aby = bx - ax, by - ay
        t = max(0.0, min(1.0, ((x - ax) * abx + (y - ay) * aby) / (abx * abx + aby * aby)))
        menor = min(menor, math.hypot(x - (ax + t * abx), y - (ay + t * aby)))
    return menor


def _retangulos_da_pista() -> List[List[Tuple[float, float]]]:
    m = MEIA_LARGURA_PISTA
    rets = []
    for (ax, ay), (bx, by) in zip(PISTA[:-1], PISTA[1:]):
        x0, x1 = min(ax, bx) - m, max(ax, bx) + m
        y0, y1 = min(ay, by) - m, max(ay, by) + m
        rets.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return rets


class SimuladorFalso(threading.Thread):
    def __init__(self, pasta: str, rodada_final: int = 40, atraso: float = 0.0):
        super().__init__(daemon=True)
        self.pasta = pasta
        self.rodada_final = rodada_final
        self.atraso = atraso
        self._parar = threading.Event()
        self._seq = 0
        self._episodio = 0
        os.makedirs(pasta, exist_ok=True)
        self._arq = {n: os.path.join(pasta, n) for n in
                     ("estado.json", "mapa.json", "acoes.json", "ultimo_resultado.json")}
        self._gravar("mapa.json", {
            "limites": {"x_min": LIMITES[0], "x_max": LIMITES[1], "y_min": LIMITES[2], "y_max": LIMITES[3]},
            "pistas": [{"inicio": list(PISTA[0]), "fim": list(PISTA[-1]), "pontos": [list(p) for p in PISTA]}],
            "invalidas": [{"tipo": "track", "poligono": [list(p) for p in r]} for r in _retangulos_da_pista()]
                         + [{"tipo": "unplaceable", "poligono": [list(p) for p in OBSTACULO]}],
            "agua": [{"poligono": [list(p) for p in AGUA]}],
        })
        self._reiniciar()

    # ------------------------------------------------------------------ arquivos
    def _gravar(self, nome: str, dados: Any) -> None:
        destino = self._arq.get(nome) or os.path.join(self.pasta, nome)
        tmp = destino + ".sim.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dados, f)
        for _ in range(200):
            try:
                os.replace(tmp, destino)
                return
            except PermissionError:
                time.sleep(0.005)

    def parar(self) -> None:
        self._parar.set()

    # ------------------------------------------------------------------ regras do jogo de mentira
    def _reiniciar(self) -> None:
        self._episodio += 1
        self.vida = 200.0
        self.dinheiro = 650.0
        self.proxima_rodada = 1
        self.torres: List[Dict[str, Any]] = []
        self.fim = False
        self._exportar(inicio=True)

    @staticmethod
    def _forca_da_rodada(r: int) -> float:
        return 4.0 * (1.10 ** r) + 2.0 * r

    def _forca_das_torres(self) -> float:
        total = 0.0
        for t in self.torres:
            _, _, alcance, _, forca = LOJA[t["tipo"]]
            d = _dist_pista(t["x"], t["y"])
            if t["tipo"] == "SniperMonkey":
                cobertura = 0.8
            elif d <= alcance:
                cobertura = 1.0 - 0.5 * d / alcance
            else:
                cobertura = 0.0
            total += forca * (1.0 + 0.7 * sum(t["caminhos"])) * cobertura
        return total

    @staticmethod
    def _custo_upgrade(t: Dict[str, Any], caminho: int) -> Optional[float]:
        niveis = t["caminhos"]
        novo = niveis[caminho] + 1
        if novo > 5:
            return None
        if niveis[caminho] == 0 and sum(1 for n in niveis if n > 0) >= 2:
            return None
        if novo > 2 and any(n > 2 for i, n in enumerate(niveis) if i != caminho):
            return None
        return round(LOJA[t["tipo"]][0] * (0.6 + 0.5 * niveis[caminho]))

    def _bloons(self, r: int) -> List[Dict[str, Any]]:
        if r > self.rodada_final:
            return []
        i = min(r // 4, len(NOMES_BLOONS) - 1)
        grupos = [{"bloon": NOMES_BLOONS[i], "quantidade": 10 + 2 * r, "inicio": 0.0, "fim": 600.0}]
        if i > 0:
            grupos.append({"bloon": NOMES_BLOONS[i - 1], "quantidade": 20 + r, "inicio": 100.0, "fim": 900.0})
        if r >= 24:
            grupos.append({"bloon": NOMES_BLOONS[i - 2] + "Camo", "quantidade": r // 2, "inicio": 300.0, "fim": 700.0})
        return grupos

    def _torre_em(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        melhor, menor = None, float("inf")
        for t in self.torres:
            d = math.hypot(t["x"] - x, t["y"] - y)
            if d < menor:
                melhor, menor = t, d
        return melhor if menor <= 1.0 else None

    # ------------------------------------------------------------------ estado
    def _exportar(self, inicio: bool = False) -> None:
        self._seq = max(self._seq + 1, int(time.time() * 1000))
        self._gravar("estado.json", {
            "seq": self._seq,
            "episodio": self._episodio,
            "fase": "fim" if self.fim else "decisao",
            "resultado": "derrota" if self.fim else None,
            "inicio": inicio,
            "proxima_rodada": self.proxima_rodada,
            "vida": self.vida,
            "dinheiro": self.dinheiro,
            "bloons_proxima_rodada": self._bloons(self.proxima_rodada),
            "torres": [{
                "tipo": t["tipo"], "caminhos": list(t["caminhos"]), "x": t["x"], "y": t["y"],
                "raio": LOJA[t["tipo"]][1],
                "upgrades": [self._custo_upgrade(t, c) for c in range(3)],
            } for t in self.torres],
            "loja": [{"tipo": tipo, "custo": c, "raio": r, "alcance": a, "areas": ar}
                     for tipo, (c, r, a, ar, _) in LOJA.items()],
        })

    # ------------------------------------------------------------------ ações
    def _colocar(self, a: Dict[str, Any]) -> Optional[str]:
        tipo, x, y = a.get("tipo"), a.get("x"), a.get("y")
        if tipo not in LOJA:
            return f"torre '{tipo}' não disponível nesta partida"
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return "campos 'x' e 'y' obrigatórios"
        custo, raio, _, areas, _ = LOJA[tipo]
        if self.dinheiro < custo:
            return f"dinheiro insuficiente ({self.dinheiro:.0f} < {custo:.0f})"
        for t in self.torres:
            if math.hypot(t["x"] - x, t["y"] - y) < raio + LOJA[t["tipo"]][1]:
                return "posição ocupada por outra torre"

        # daqui para baixo é o que só a "simulação" sabe
        fora = not (LIMITES[0] + raio <= x <= LIMITES[1] - raio and LIMITES[2] + raio <= y <= LIMITES[3] - raio)
        na_pista = _dist_pista(x, y) < MEIA_LARGURA_PISTA + raio * 0.5
        no_obstaculo = _dentro(OBSTACULO, x, y)
        na_agua = _dentro(AGUA, x, y)
        area_ok = ("water" in areas) if na_agua else ("land" in areas)
        if fora or na_pista or no_obstaculo or not area_ok:
            return MOTIVO_POSICAO

        self.dinheiro -= custo
        self.torres.append({"tipo": tipo, "x": float(x), "y": float(y), "caminhos": [0, 0, 0], "gasto": custo})
        return None

    def _alvo(self, a: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        alvo = a.get("torre")
        if not isinstance(alvo, dict) or "x" not in alvo or "y" not in alvo:
            return None, "campo 'torre' com 'x' e 'y' obrigatório"
        t = self._torre_em(alvo["x"], alvo["y"])
        return (t, None) if t else (None, "nenhuma torre nessa posição")

    def _melhorar(self, a: Dict[str, Any]) -> Optional[str]:
        t, erro = self._alvo(a)
        if t is None:
            return erro
        caminho = a.get("caminho")
        if caminho not in (0, 1, 2):
            return "'caminho' deve ser 0, 1 ou 2"
        custo = self._custo_upgrade(t, caminho)
        if custo is None:
            return "upgrade indisponível para esse caminho"
        if self.dinheiro < custo:
            return f"dinheiro insuficiente ({self.dinheiro:.0f} < {custo:.0f})"
        self.dinheiro -= custo
        t["caminhos"][caminho] += 1
        t["gasto"] += custo
        return None

    def _vender(self, a: Dict[str, Any]) -> Optional[str]:
        t, erro = self._alvo(a)
        if t is None:
            return erro
        self.dinheiro += round(0.7 * t["gasto"])
        self.torres.remove(t)
        return None

    def _rodar_rodada(self) -> None:
        r = self.proxima_rodada
        forca = self._forca_da_rodada(r)
        vazou = max(0.0, forca - self._forca_das_torres())
        self.vida -= math.ceil(vazou)
        if self.vida <= 0:
            self.fim = True
            return
        self.dinheiro += 100 + 15 * r + round(0.5 * forca)
        self.proxima_rodada += 1

    def _processar(self, lote: Dict[str, Any]) -> None:
        resultados = []
        rodou = reiniciou = False
        for i, a in enumerate(lote.get("acoes", []), start=1):
            nome = a.get("acao", "(sem nome)")
            if self.fim and nome != "reiniciar":
                erro: Optional[str] = "a partida acabou"
            elif nome == "colocar_torre":
                erro = self._colocar(a)
            elif nome == "melhorar_torre":
                erro = self._melhorar(a)
            elif nome == "vender_torre":
                erro = self._vender(a)
            elif nome == "iniciar_rodada":
                erro = None
                rodou = True
            elif nome == "reiniciar":
                erro = None
                reiniciou = True
            else:
                erro = "ação desconhecida"
            resultados.append({"indice": i, "acao": nome, "aceita": erro is None, "motivo": erro or "ok"})
            if reiniciou:
                break

        if self.atraso:
            time.sleep(self.atraso)
        self._gravar("ultimo_resultado.json", {"id": lote.get("id"), "resultados": resultados})

        if reiniciou:
            self._reiniciar()
            return
        if rodou:
            if self.atraso:
                time.sleep(self.atraso)
            self._rodar_rodada()
        self._exportar()

    # ------------------------------------------------------------------ laço
    def run(self) -> None:
        arq = self._arq["acoes.json"]
        while not self._parar.is_set():
            lote = None
            if os.path.exists(arq):
                try:
                    with open(arq, "r", encoding="utf-8") as f:
                        lote = json.load(f)
                    os.remove(arq)
                except (OSError, ValueError):
                    lote = None
            if isinstance(lote, list):
                lote = {"id": None, "acoes": lote}
            if isinstance(lote, dict):
                self._processar(lote)
            else:
                time.sleep(0.001)


def ligar_simulador(cfg) -> SimuladorFalso:
    """Aponta a configuração para uma pasta de mentira e liga o simulador nela."""
    cfg.pasta_saida = cfg.pasta_saida.rstrip("/\\") + "_simulador"  # não mistura com o treino real
    cfg.pasta_userdata = os.path.join(cfg.pasta_saida, "userdata_falso")
    sim = SimuladorFalso(cfg.pasta_userdata, cfg.rodada_final)
    sim.start()
    return sim


if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else "userdata_falso"
    sim = SimuladorFalso(pasta)
    sim.start()
    print(f"Simulador falso rodando em: {os.path.abspath(pasta)}  (Ctrl+C para sair)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sim.parar()
