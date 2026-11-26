"""Configuração do pipeline. Edite os valores padrão aqui."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ jogo
    # Pasta UserData do BTD6 (onde os mods leem e escrevem os arquivos)
    pasta_userdata: str = r"C:\Program Files (x86)\Steam\steamapps\common\BloonsTD6\UserData"

    # Última rodada do modo em que você treina (fácil = 40, médio = 60, difícil = 80)
    rodada_final: int = 40

    # Torres que o agente pode comprar (baseId do jogo). Menos torres = aprende mais rápido.
    # As que não existirem na loja da partida ficam bloqueadas sozinhas.
    torres: List[str] = field(default_factory=lambda: [
        "DartMonkey",
        "TackShooter",
        "BombShooter",
        "SniperMonkey",
        "NinjaMonkey",
        "WizardMonkey",
    ])

    # ------------------------------------------------------------------ ações
    # Grade de posições onde o agente pode colocar torres (colunas x linhas)
    grade_colunas: int = 20
    grade_linhas: int = 15
    # Limites da grade (x_min, x_max, y_min, y_max). None = caixa em volta da pista + margem.
    limites: Optional[Tuple[float, float, float, float]] = None
    margem_pista: float = 40.0

    max_torres: int = 20            # torres que o agente consegue acompanhar/melhorar
    max_acoes_por_turno: int = 8    # compras por rodada; depois disso só resta iniciar a rodada
    permitir_venda: bool = False

    # Só oferece posições de onde a torre alcança a pista (alcance da loja, com este mínimo)
    filtrar_por_alcance: bool = True
    alcance_minimo: float = 40.0
    # Bloqueia de antemão pista/obstáculos/água usando os polígonos do mapa.json
    mascara_estatica: bool = True

    # ------------------------------------------------------------------ recompensa
    recompensa_rodada: float = 1.0        # por rodada concluída
    penalidade_vidas: float = 5.0         # vezes a fração da vida inicial perdida na rodada
    recompensa_vitoria: float = 10.0
    penalidade_derrota: float = 5.0
    penalidade_acao_recusada: float = 0.05

    # ------------------------------------------------------------------ comunicação (segundos)
    intervalo_leitura: float = 0.02
    timeout_resultado: float = 30.0       # o Player responder a uma ação
    timeout_estado: float = 30.0          # o Exportador mandar o estado depois de uma compra
    timeout_rodada: float = 600.0         # uma rodada inteira terminar
    timeout_reinicio: float = 90.0        # a partida reiniciar
    espera_apos_reinicio: float = 1.0     # deixa o estado inicial assentar

    # ------------------------------------------------------------------ treino (MaskablePPO)
    pasta_saida: str = "saida"
    passos_totais: int = 200_000
    salvar_a_cada: int = 2_000
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.02
    camadas: List[int] = field(default_factory=lambda: [256, 256])
    semente: int = 0
