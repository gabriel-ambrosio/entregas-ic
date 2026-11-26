"""Ambiente Gymnasium do Bloons TD 6, conversando com os mods por arquivos.

Um passo do agente = UMA ação no jogo:
    0                      iniciar a próxima rodada (encerra o turno de compras)
    colocar(tipo, célula)  coloca uma torre no centro de uma célula da grade
    melhorar(slot, caminho)
    vender(slot)           só se permitir_venda = True

A recompensa só aparece quando a rodada termina (rodada concluída, vidas perdidas,
vitória ou derrota). Ações impossíveis ficam fora pela máscara (action_masks).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config
from mapa import Grade
from ponte import Ponte, TempoEsgotado

# Bloons na ordem do jogo, com o RBE aproximado de cada um
BLOONS: List[Tuple[str, float]] = [
    ("Red", 1), ("Blue", 2), ("Green", 3), ("Yellow", 4), ("Pink", 5),
    ("Black", 11), ("White", 11), ("Purple", 11), ("Lead", 23), ("Zebra", 23),
    ("Rainbow", 47), ("Ceramic", 104), ("Moab", 616), ("Bfb", 3164),
    ("Zomg", 16656), ("Ddt", 816), ("Bad", 55760),
]
N_GLOBAIS = 6
N_BLOONS = len(BLOONS) + 4  # contagem por tipo + camo + regrow + fortificado + RBE total
TOLERANCIA_POSICAO = 1.0


class AmbienteBTD6(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, ponte: Optional[Ponte] = None):
        super().__init__()
        self.cfg = cfg
        self.ponte = ponte or Ponte(cfg.pasta_userdata, cfg.intervalo_leitura)

        self.T = len(cfg.torres)
        self.G = cfg.grade_colunas * cfg.grade_linhas
        self.S = cfg.max_torres

        # ---- espaço de ações (Discrete, com máscara)
        self._base_colocar = 1
        self._base_melhorar = self._base_colocar + self.T * self.G
        self._base_vender = self._base_melhorar + self.S * 3
        self.n_acoes = self._base_vender + (self.S if cfg.permitir_venda else 0)
        self.action_space = spaces.Discrete(self.n_acoes)

        # ---- espaço de observação (vetor)
        self._n_por_torre = 1 + self.T + 3 + 2 + 3
        self.n_obs = N_GLOBAIS + N_BLOONS + 2 * self.T + self.S * self._n_por_torre + self.G
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(self.n_obs,), dtype=np.float32)

        # ---- estado do episódio
        self._estado: Optional[Dict[str, Any]] = None
        self._grade: Optional[Grade] = None
        self._loja: Dict[str, Dict[str, Any]] = {}
        self._validas: List[np.ndarray] = []          # por tipo: células onde vale tentar colocar
        self._slots: List[Optional[Dict[str, Any]]] = [None] * self.S
        self._vida_inicial = 1.0
        self._acoes_no_turno = 0
        self._terminou = False
        self._precisa_reiniciar = False
        self._avisou_loja = False

        # ---- posições que o jogo já recusou (aprendidas e guardadas em disco)
        os.makedirs(cfg.pasta_saida, exist_ok=True)
        self._arq_bloqueadas = os.path.join(cfg.pasta_saida, "celulas_bloqueadas.json")
        self._bloqueadas: Dict[str, set] = {}
        self._assinatura_grade: Optional[str] = None

    # ================================================================== API do Gymnasium
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        estado = self._obter_estado_inicial()
        self._preparar_partida(estado)
        return self._observacao(), self._info()

    def step(self, acao):
        if self._estado is None:
            raise RuntimeError("chame reset() antes de step()")
        try:
            return self._passo(int(acao))
        except TempoEsgotado as e:
            # O jogo parou de responder: encerra o episódio e reinicia a partida no próximo reset
            print(f"[ambiente] tempo esgotado: {e}")
            self._precisa_reiniciar = True
            info = self._info()
            info["erro"] = str(e)
            return self._observacao(), 0.0, False, True, info

    def action_masks(self) -> np.ndarray:
        """Máscara usada pelo MaskablePPO: True = ação permitida agora."""
        m = np.zeros(self.n_acoes, dtype=bool)
        m[0] = True  # iniciar a rodada é sempre permitido
        e = self._estado
        if e is None or self._terminou or self._acoes_no_turno >= self.cfg.max_acoes_por_turno:
            return m

        dinheiro = float(e.get("dinheiro", 0))
        torres = e.get("torres") or []

        # ---- colocar
        if any(s is None for s in self._slots) and self._grade is not None:
            centros = self._grade.centros
            if torres:
                pos = np.array([[t["x"], t["y"]] for t in torres], dtype=np.float64)
                raios = np.array([self._raio_da_torre(t) for t in torres], dtype=np.float64)
                dist = np.linalg.norm(centros[:, None, :] - pos[None, :, :], axis=2)  # (G, n)
            for ti, tipo in enumerate(self.cfg.torres):
                item = self._loja.get(tipo)
                if item is None or dinheiro < item["custo"]:
                    continue
                ok = self._validas[ti].copy()
                if torres:
                    ok &= (dist >= item["raio"] + raios[None, :] + 0.5).all(axis=1)
                ini = self._base_colocar + ti * self.G
                m[ini:ini + self.G] = ok

        # ---- melhorar / vender
        for s, t in enumerate(self._slots):
            if t is None:
                continue
            custos = t.get("upgrades") or [None, None, None]
            for c in range(3):
                custo = custos[c] if c < len(custos) else None
                if custo is not None and dinheiro >= custo:
                    m[self._base_melhorar + s * 3 + c] = True
            if self.cfg.permitir_venda:
                m[self._base_vender + s] = True
        return m

    def render(self):
        return None

    def close(self):
        return None

    # ================================================================== passo
    def _decodificar(self, a: int) -> Tuple:
        if a == 0:
            return ("iniciar",)
        if a < self._base_melhorar:
            a -= self._base_colocar
            return ("colocar", a // self.G, a % self.G)
        if a < self._base_vender:
            a -= self._base_melhorar
            return ("melhorar", a // 3, a % 3)
        return ("vender", a - self._base_vender)

    def _passo(self, a: int):
        cfg = self.cfg
        anterior = self._estado
        assert anterior is not None

        if self._terminou:
            # Episódio já acabou: nada a fazer até o reset
            return self._observacao(), 0.0, True, False, self._info()

        if not (0 <= a < self.n_acoes) or not self.action_masks()[a]:
            # Ação fora da máscara (agente sem máscara): conta como recusada, sem tocar no jogo
            self._acoes_no_turno += 1
            return self._observacao(), -cfg.penalidade_acao_recusada, False, False, self._info()

        acao = self._decodificar(a)
        recompensa = 0.0

        if acao[0] == "iniciar":
            self.ponte.enviar([{"acao": "iniciar_rodada"}], cfg.timeout_resultado)
            novo = self.ponte.esperar_estado(anterior["seq"], cfg.timeout_rodada)
            rodadas = max(0, int(novo["proxima_rodada"]) - int(anterior["proxima_rodada"]))
            perdidas = max(0.0, float(anterior["vida"]) - max(float(novo["vida"]), 0.0))
            recompensa += cfg.recompensa_rodada * rodadas
            recompensa -= cfg.penalidade_vidas * perdidas / self._vida_inicial
            self._acoes_no_turno = 0
        else:
            if acao[0] == "colocar":
                _, ti, cel = acao
                x, y = self._grade.centros[cel]
                comando = {"acao": "colocar_torre", "tipo": cfg.torres[ti], "x": float(x), "y": float(y)}
            else:
                torre = self._slots[acao[1]]
                alvo = {"x": float(torre["x"]), "y": float(torre["y"])}
                if acao[0] == "melhorar":
                    comando = {"acao": "melhorar_torre", "torre": alvo, "caminho": int(acao[2])}
                else:
                    comando = {"acao": "vender_torre", "torre": alvo}

            resultados = self.ponte.enviar([comando], cfg.timeout_resultado)
            novo = self.ponte.esperar_estado(anterior["seq"], cfg.timeout_estado)
            r = resultados[0] if resultados else {}
            if not r.get("aceita"):
                recompensa -= cfg.penalidade_acao_recusada
                if acao[0] == "colocar" and "recusou a posi" in str(r.get("motivo", "")):
                    self._bloquear(acao[1], acao[2])
            self._acoes_no_turno += 1

        self._estado = novo
        self._sincronizar_slots(novo.get("torres") or [])

        derrota = novo.get("fase") == "fim" or float(novo["vida"]) <= 0
        vitoria = not derrota and int(novo["proxima_rodada"]) > cfg.rodada_final
        if derrota:
            recompensa -= cfg.penalidade_derrota
        if vitoria:
            recompensa += cfg.recompensa_vitoria
        self._terminou = derrota or vitoria
        if self._terminou:
            self._precisa_reiniciar = True

        info = self._info()
        info["vitoria"] = vitoria
        info["derrota"] = derrota
        return self._observacao(), float(recompensa), self._terminou, False, info

    # ================================================================== início de episódio
    def _eh_inicio(self, e: Optional[Dict[str, Any]]) -> bool:
        """Estado de começo de partida de verdade: sem torres e dentro das rodadas do modo.
        (Evita aceitar um "início" falso quando o jogo não reiniciou e seguiu para o freeplay.)"""
        return (
            bool(e)
            and e.get("inicio") is True
            and e.get("fase") == "decisao"
            and float(e.get("vida", 0)) > 0
            and not e.get("torres")
            and int(e.get("proxima_rodada", 0)) <= self.cfg.rodada_final
        )

    def _obter_estado_inicial(self) -> Dict[str, Any]:
        cfg = self.cfg
        self.ponte.limpar()
        estado = self.ponte.ler_estado()

        # Partida recém-aberta e ainda intocada: aproveita sem reiniciar
        if not self._precisa_reiniciar and self._eh_inicio(estado):
            return estado

        ultimo_erro: Optional[Exception] = None
        for _ in range(3):
            atual = self.ponte.ler_estado()
            seq0 = atual["seq"] if atual else -1
            try:
                resultados = self.ponte.enviar([{"acao": "reiniciar"}], cfg.timeout_resultado)
                if resultados and not resultados[0].get("aceita"):
                    raise RuntimeError(f"o mod Player não conseguiu reiniciar: {resultados[0].get('motivo')}")
                self.ponte.esperar_estado(seq0, cfg.timeout_reinicio, self._eh_inicio)
                time.sleep(cfg.espera_apos_reinicio)  # o mod pode exportar o início duas vezes
                novo = self.ponte.ler_estado()
                if self._eh_inicio(novo):
                    self._precisa_reiniciar = False
                    return novo
                ultimo_erro = RuntimeError(
                    f"o estado depois do reinício não é de começo de partida (rodada "
                    f"{(novo or {}).get('proxima_rodada')}, {len((novo or {}).get('torres') or [])} torres)")
            except TempoEsgotado as e:
                ultimo_erro = e
                print(f"[ambiente] reinício sem resposta, tentando de novo: {e}")
        raise RuntimeError(
            "Não foi possível obter o estado inicial da partida. Confira se o BTD6 está com uma "
            "partida aberta, se os mods Exportador e Player (v2) estão carregados e se "
            f"'pasta_userdata' está certa. Último erro: {ultimo_erro}"
        )

    def _preparar_partida(self, estado: Dict[str, Any]) -> None:
        cfg = self.cfg
        mapa = self.ponte.ler_mapa()
        if mapa is None:
            raise RuntimeError("mapa.json não encontrado: o mod Exportador (v2) exporta o mapa no início da partida.")
        self._grade = Grade(mapa, cfg.grade_colunas, cfg.grade_linhas, cfg.limites, cfg.margem_pista)

        self._estado = estado
        self._vida_inicial = max(float(estado["vida"]), 1.0)
        self._acoes_no_turno = 0
        self._terminou = False
        self._slots = [None] * self.S
        self._sincronizar_slots(estado.get("torres") or [])

        # ---- loja
        self._loja = {}
        for item in estado.get("loja") or []:
            self._loja[item["tipo"]] = {
                "custo": float(item.get("custo", 0)),
                "raio": float(item.get("raio", 8)),
                "alcance": float(item.get("alcance", cfg.alcance_minimo)),
                "areas": [str(a).lower() for a in (item.get("areas") or ["land"])],
            }
        if not self._loja and not self._avisou_loja:
            self._avisou_loja = True
            print("[ambiente] AVISO: o estado veio sem 'loja'. Atualize o mod Exportador para a v2; "
                  "sem os custos, nenhuma torre pode ser colocada.")

        # ---- posições candidatas por tipo
        self._carregar_bloqueadas()
        self._validas = []
        for tipo in cfg.torres:
            item = self._loja.get(tipo)
            if item is None:
                self._validas.append(np.zeros(self.G, dtype=bool))
                continue
            ok = self._grade.mascara_estatica(item["areas"]) if cfg.mascara_estatica else np.ones(self.G, dtype=bool)
            if cfg.filtrar_por_alcance:
                ok &= self._grade.dist_pista <= max(item["alcance"], cfg.alcance_minimo)
            for cel in self._bloqueadas.get(self._chave_areas(item), ()):
                if 0 <= cel < self.G:
                    ok[cel] = False
            self._validas.append(ok)

    # ================================================================== slots de torres
    def _sincronizar_slots(self, torres: List[Dict[str, Any]]) -> None:
        """Mantém cada torre no mesmo slot enquanto ela existir (identificada pela posição)."""
        novos: List[Optional[Dict[str, Any]]] = [None] * self.S
        restantes = list(torres)
        for i, antiga in enumerate(self._slots):
            if antiga is None:
                continue
            for t in restantes:
                if (abs(t["x"] - antiga["x"]) <= TOLERANCIA_POSICAO
                        and abs(t["y"] - antiga["y"]) <= TOLERANCIA_POSICAO):
                    novos[i] = t
                    restantes.remove(t)
                    break
        for t in restantes:
            if None not in novos:
                break  # mais torres do que slots: as excedentes ficam sem upgrade
            novos[novos.index(None)] = t
        self._slots = novos

    def _raio_da_torre(self, t: Dict[str, Any]) -> float:
        if t.get("raio") is not None:
            return float(t["raio"])
        item = self._loja.get(t.get("tipo", ""))
        return float(item["raio"]) if item else 8.0

    # ================================================================== posições bloqueadas
    @staticmethod
    def _chave_areas(item: Dict[str, Any]) -> str:
        return ",".join(sorted(item["areas"]))

    def _carregar_bloqueadas(self) -> None:
        g = self._grade
        assinatura = f"{g.colunas}x{g.linhas}:" + ",".join(f"{v:.1f}" for v in g.limites)
        if assinatura == self._assinatura_grade:
            return
        self._assinatura_grade = assinatura
        self._bloqueadas = {}
        try:
            with open(self._arq_bloqueadas, "r", encoding="utf-8") as f:
                dados = json.load(f)
            if dados.get("grade") == assinatura:
                self._bloqueadas = {k: set(v) for k, v in dados.get("celulas", {}).items()}
        except (OSError, ValueError):
            pass

    def _bloquear(self, ti: int, cel: int) -> None:
        """O jogo recusou essa posição: nenhuma torre com as mesmas áreas tenta de novo."""
        item = self._loja.get(self.cfg.torres[ti])
        if item is None:
            return
        chave = self._chave_areas(item)
        self._bloqueadas.setdefault(chave, set()).add(int(cel))
        for tj, tipo in enumerate(self.cfg.torres):
            outro = self._loja.get(tipo)
            if outro is not None and self._chave_areas(outro) == chave:
                self._validas[tj][cel] = False
        try:
            with open(self._arq_bloqueadas, "w", encoding="utf-8") as f:
                json.dump({"grade": self._assinatura_grade,
                           "celulas": {k: sorted(v) for k, v in self._bloqueadas.items()}}, f)
        except OSError:
            pass

    # ================================================================== observação
    def _observacao(self) -> np.ndarray:
        cfg = self.cfg
        e = self._estado or {}
        v = np.zeros(self.n_obs, dtype=np.float32)
        dinheiro = max(float(e.get("dinheiro", 0)), 0.0)
        torres = e.get("torres") or []

        # ---- globais
        v[0] = max(float(e.get("vida", 0)), 0.0) / self._vida_inicial
        v[1] = np.log1p(dinheiro) / 12.0
        v[2] = min(dinheiro / 2000.0, 2.0)
        v[3] = float(e.get("proxima_rodada", 0)) / cfg.rodada_final
        v[4] = 1.0 - self._acoes_no_turno / cfg.max_acoes_por_turno
        v[5] = min(len(torres) / self.S, 1.0)
        k = N_GLOBAIS

        # ---- bloons da próxima rodada
        rbe_total = 0.0
        camo = regrow = fortificado = 0.0
        contagem = np.zeros(len(BLOONS))
        for grupo in e.get("bloons_proxima_rodada") or []:
            nome = str(grupo.get("bloon", ""))
            qtd = float(grupo.get("quantidade", 0))
            minusculo = nome.lower()
            for i, (base, rbe) in enumerate(BLOONS):
                if minusculo.startswith(base.lower()):
                    contagem[i] += qtd
                    rbe_total += qtd * rbe
                    break
            if "camo" in minusculo:
                camo += qtd
            if "regrow" in minusculo:
                regrow += qtd
            if "fortified" in minusculo:
                fortificado += qtd
        v[k:k + len(BLOONS)] = np.minimum(np.log1p(contagem) / np.log1p(100.0), 2.0)
        k += len(BLOONS)
        v[k] = min(np.log1p(camo) / np.log1p(100.0), 2.0)
        v[k + 1] = min(np.log1p(regrow) / np.log1p(100.0), 2.0)
        v[k + 2] = min(np.log1p(fortificado) / np.log1p(100.0), 2.0)
        v[k + 3] = np.log1p(rbe_total) / 12.0
        k += 4

        # ---- loja: dá para comprar? quanto custa?
        for ti, tipo in enumerate(cfg.torres):
            item = self._loja.get(tipo)
            if item is not None:
                v[k + 2 * ti] = 1.0 if dinheiro >= item["custo"] else 0.0
                v[k + 2 * ti + 1] = min(item["custo"] / 2000.0, 2.0)
        k += 2 * self.T

        # ---- torres (um bloco por slot)
        for s, t in enumerate(self._slots):
            if t is None:
                continue
            b = k + s * self._n_por_torre
            v[b] = 1.0
            if t.get("tipo") in cfg.torres:
                v[b + 1 + cfg.torres.index(t["tipo"])] = 1.0
            niveis = t.get("caminhos") or [0, 0, 0]
            for c in range(3):
                v[b + 1 + self.T + c] = float(niveis[c]) / 5.0
            if self._grade is not None:
                nx, ny = self._grade.normalizar(t["x"], t["y"])
                v[b + 1 + self.T + 3] = nx
                v[b + 1 + self.T + 4] = ny
            custos = t.get("upgrades") or [None, None, None]
            for c in range(3):
                custo = custos[c] if c < len(custos) else None
                if custo is not None and dinheiro >= custo:
                    v[b + 1 + self.T + 5 + c] = 1.0
        k += self.S * self._n_por_torre

        # ---- ocupação da grade
        if self._grade is not None:
            for t in torres:
                cel = self._grade.celula_de(t["x"], t["y"])
                if cel >= 0:
                    v[k + cel] = 1.0

        return np.clip(v, 0.0, 10.0)

    def _info(self) -> Dict[str, Any]:
        e = self._estado or {}
        return {
            "rodada": int(e.get("proxima_rodada", 0)),
            "vida": float(e.get("vida", 0)),
            "dinheiro": float(e.get("dinheiro", 0)),
            "torres": len(e.get("torres") or []),
            "vitoria": False,
            "derrota": False,
        }
