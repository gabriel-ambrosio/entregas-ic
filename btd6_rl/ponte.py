"""Ponte de arquivos entre o Python e os mods Exportador e Player.

Fluxo de um passo:
  1. o Python grava acoes.json (com um "id");
  2. o mod Player executa e grava ultimo_resultado.json (com o mesmo "id");
  3. o mod Exportador grava estado.json com um "seq" maior que o anterior.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional


class TempoEsgotado(RuntimeError):
    """O jogo não respondeu dentro do prazo."""


class Ponte:
    def __init__(self, pasta: str, intervalo: float = 0.02):
        self.pasta = pasta
        self.intervalo = intervalo
        self.arq_estado = os.path.join(pasta, "estado.json")
        self.arq_mapa = os.path.join(pasta, "mapa.json")
        self.arq_acoes = os.path.join(pasta, "acoes.json")
        self.arq_resultado = os.path.join(pasta, "ultimo_resultado.json")
        # Ids únicos mesmo entre execuções, para nunca confundir com um resultado antigo
        self._id = int(time.time() * 1000) % 1_000_000_000

        if not os.path.isdir(pasta):
            raise FileNotFoundError(
                f"Pasta não encontrada: {pasta}\n"
                "Ajuste 'pasta_userdata' em config.py para a pasta UserData do BTD6."
            )

    # ------------------------------------------------------------------ leitura
    @staticmethod
    def _ler_json(caminho: str) -> Optional[Any]:
        """Lê um JSON; devolve None se o arquivo não existe ou está sendo trocado."""
        for _ in range(5):
            try:
                with open(caminho, "r", encoding="utf-8-sig") as f:
                    return json.load(f)
            except FileNotFoundError:
                return None
            except (json.JSONDecodeError, PermissionError, OSError):
                time.sleep(0.01)
        return None

    def ler_estado(self) -> Optional[Dict[str, Any]]:
        estado = self._ler_json(self.arq_estado)
        return estado if isinstance(estado, dict) and "seq" in estado else None

    def ler_mapa(self) -> Optional[Dict[str, Any]]:
        mapa = self._ler_json(self.arq_mapa)
        return mapa if isinstance(mapa, dict) else None

    def esperar_estado(
        self,
        depois_de: int,
        timeout: float,
        condicao: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """Espera um estado com seq > depois_de (e que satisfaça a condição, se houver)."""
        limite = time.monotonic() + timeout
        while True:
            estado = self.ler_estado()
            if estado is not None and estado["seq"] > depois_de and (condicao is None or condicao(estado)):
                return estado
            if time.monotonic() > limite:
                raise TempoEsgotado(f"nenhum estado novo em {timeout:.0f} s (seq > {depois_de})")
            time.sleep(self.intervalo)

    # ------------------------------------------------------------------ escrita
    @staticmethod
    def _apagar(caminho: str) -> None:
        try:
            os.remove(caminho)
        except OSError:
            pass

    def _gravar_atomico(self, caminho: str, dados: Any) -> None:
        tmp = caminho + ".py.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dados, f)
        for _ in range(200):
            try:
                os.replace(tmp, caminho)
                return
            except PermissionError:  # o mod está com o arquivo aberto neste instante
                time.sleep(0.01)
        raise TempoEsgotado(f"não foi possível gravar {caminho}")

    def enviar(self, acoes: List[Dict[str, Any]], timeout: float) -> List[Dict[str, Any]]:
        """Envia um lote de ações e devolve a lista de resultados do mod Player."""
        self._id += 1
        lote = self._id

        self._apagar(self.arq_resultado)
        self._gravar_atomico(self.arq_acoes, {"id": lote, "acoes": acoes})

        limite = time.monotonic() + timeout
        while True:
            r = self._ler_json(self.arq_resultado)
            if isinstance(r, dict) and r.get("id") == lote:
                return list(r.get("resultados") or [])
            if time.monotonic() > limite:
                self._apagar(self.arq_acoes)  # não deixa uma ação velha para depois
                raise TempoEsgotado(
                    f"o mod Player não respondeu em {timeout:.0f} s. "
                    "Confira se há uma partida aberta e se o 'Modo automático' do mod está ligado."
                )
            time.sleep(self.intervalo)

    def limpar(self) -> None:
        """Remove ações e resultados que sobraram de execuções anteriores."""
        self._apagar(self.arq_acoes)
        self._apagar(self.arq_resultado)
