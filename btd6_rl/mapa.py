"""Geometria do mapa: grade de posições candidatas, distância até a pista e polígonos."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _dentro_do_poligono(pontos: np.ndarray, poligono: np.ndarray) -> np.ndarray:
    """Para cada ponto (N,2), True se está dentro do polígono (M,2). Ray casting."""
    x, y = pontos[:, 0], pontos[:, 1]
    dentro = np.zeros(len(pontos), dtype=bool)
    n = len(poligono)
    if n < 3:
        return dentro
    j = n - 1
    for i in range(n):
        xi, yi = poligono[i]
        xj, yj = poligono[j]
        cruza = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_corte = (xj - xi) * (y - yi) / (yj - yi) + xi
        dentro ^= cruza & (x < x_corte)
        j = i
    return dentro


def _distancia_ate_segmentos(pontos: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Menor distância de cada ponto (N,2) até os segmentos a[i]-b[i] (S,2)."""
    if len(a) == 0:
        return np.full(len(pontos), np.inf)
    ab = b - a                                           # (S,2)
    ap = pontos[:, None, :] - a[None, :, :]              # (N,S,2)
    comprimento2 = np.maximum((ab * ab).sum(axis=1), 1e-9)
    t = np.clip((ap * ab[None, :, :]).sum(axis=2) / comprimento2[None, :], 0.0, 1.0)
    mais_proximo = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(pontos[:, None, :] - mais_proximo, axis=2)
    return d.min(axis=1)


class Grade:
    """Grade colunas x linhas de posições onde o agente pode tentar colocar torres."""

    def __init__(
        self,
        mapa: Dict[str, Any],
        colunas: int,
        linhas: int,
        limites: Optional[Tuple[float, float, float, float]] = None,
        margem_pista: float = 40.0,
    ):
        self.colunas = colunas
        self.linhas = linhas
        self.n = colunas * linhas

        pistas: List[np.ndarray] = []
        for pista in mapa.get("pistas", []):
            pts = np.asarray(pista.get("pontos", []), dtype=np.float64).reshape(-1, 2)
            if len(pts) > 0:
                pistas.append(pts)
        if not pistas:
            raise ValueError("mapa.json sem pistas: o mapa foi exportado nesta partida?")

        todos = np.concatenate(pistas)
        if limites is None:
            x_min, y_min = todos.min(axis=0) - margem_pista
            x_max, y_max = todos.max(axis=0) + margem_pista
            lim = mapa.get("limites") or {}
            chaves = ("x_min", "x_max", "y_min", "y_max")
            # não sai da área conhecida do mapa
            if all(isinstance(lim.get(k), (int, float)) and np.isfinite(lim[k]) for k in chaves):
                x_min, x_max = max(x_min, lim["x_min"]), min(x_max, lim["x_max"])
                y_min, y_max = max(y_min, lim["y_min"]), min(y_max, lim["y_max"])
        else:
            x_min, x_max, y_min, y_max = limites
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"limites da grade inválidos: {(x_min, x_max, y_min, y_max)}")
        self.limites = (float(x_min), float(x_max), float(y_min), float(y_max))

        self.largura_celula = (x_max - x_min) / colunas
        self.altura_celula = (y_max - y_min) / linhas
        xs = x_min + (np.arange(colunas) + 0.5) * self.largura_celula
        ys = y_min + (np.arange(linhas) + 0.5) * self.altura_celula
        gx, gy = np.meshgrid(xs, ys)                      # índice = linha * colunas + coluna
        self.centros = np.stack([gx.ravel(), gy.ravel()], axis=1)

        # distância de cada célula até a pista
        a = np.concatenate([p[:-1] for p in pistas if len(p) > 1] or [np.zeros((0, 2))])
        b = np.concatenate([p[1:] for p in pistas if len(p) > 1] or [np.zeros((0, 2))])
        self.dist_pista = _distancia_ate_segmentos(self.centros, a, b)

        # polígonos do mapa
        self.em_invalida = np.zeros(self.n, dtype=bool)
        for area in mapa.get("invalidas", []):
            pol = np.asarray(area.get("poligono", []), dtype=np.float64).reshape(-1, 2)
            self.em_invalida |= _dentro_do_poligono(self.centros, pol)

        self.em_agua = np.zeros(self.n, dtype=bool)
        for area in mapa.get("agua", []):
            pol = np.asarray(area.get("poligono", []), dtype=np.float64).reshape(-1, 2)
            self.em_agua |= _dentro_do_poligono(self.centros, pol)

    # ------------------------------------------------------------------
    def celula_de(self, x: float, y: float) -> int:
        """Índice da célula que contém (x, y), ou -1 se está fora da grade."""
        x_min, x_max, y_min, y_max = self.limites
        if not (x_min <= x < x_max and y_min <= y < y_max):
            return -1
        col = int((x - x_min) / self.largura_celula)
        lin = int((y - y_min) / self.altura_celula)
        return min(lin, self.linhas - 1) * self.colunas + min(col, self.colunas - 1)

    def normalizar(self, x: float, y: float) -> Tuple[float, float]:
        x_min, x_max, y_min, y_max = self.limites
        return (
            float(np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)),
            float(np.clip((y - y_min) / (y_max - y_min), 0.0, 1.0)),
        )

    def mascara_estatica(self, areas: Sequence[str]) -> np.ndarray:
        """Células onde uma torre com essas áreas (ex.: ['land'], ['water']) pode ficar,
        só pelo centro da célula. É uma aproximação: o jogo dá a palavra final."""
        areas = {str(a).lower() for a in areas} or {"land"}
        pode_terra = "land" in areas
        pode_agua = "water" in areas
        ok = np.zeros(self.n, dtype=bool)
        if pode_terra:
            ok |= ~self.em_agua
        if pode_agua:
            ok |= self.em_agua
        return ok & ~self.em_invalida
