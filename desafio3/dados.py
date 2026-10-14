"""
dados.py — pipeline de dados do preditor de rodada.
O pré-processamento roda UMA vez e fica em cache. O cálculo de cobertura de
cada torre é O(pontos do mapa) e seria chamado cerca de um milhão de vezes;
como depende só de (torre, upgrade, posição), é memoizado — são ~282 mil
combinações distintas. Sem o cache, cada execução do protocolo pagaria esse
custo de novo.
A divisão é determinística (80/10/10 por semente), para que qualquer linha de
experimentos.csv seja reproduzível.
Uso:
   python -m desafio3.dados --construir      # gera o cache
   python -m desafio3.dados --info           # inspeciona o cache
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
AQUI = Path(__file__).resolve().parent
CACHE = AQUI / "cache" / "dados.pt"
# Os dados brutos e os utilitários de cobertura ficam um nível acima. Só
# `--construir` precisa deles: gerado o cache, o resto roda sozinho.
RAIZ = Path(os.environ.get("BTD6_RAIZ", AQUI.parent))
CAMINHO_RODADAS = RAIZ / "data" / "btd6-bloon-rounds.csv"
CAMINHO_MAPA = RAIZ / "data" / "map_points.json"
CAMINHO_SAIDAS = RAIZ / "data" / "magic-meadows-round-outcomes.gz"

def _importa_utilitarios():
   """Carrega os utilitários de alcance das torres, só na construção do cache."""
   if not (RAIZ / "utils" / "tower_ranges.py").exists():
       raise FileNotFoundError(
           f"utilitários de torre não encontrados em {RAIZ / 'utils'}.\n"
           f"Defina BTD6_RAIZ para a pasta que contém data/ e utils/."
       )
   if str(RAIZ) not in sys.path:
       sys.path.insert(0, str(RAIZ))
   from utils.tower_ranges import TOWER_NAMES, TOWER_RANGES, Point
   return TOWER_NAMES, TOWER_RANGES, Point

N_TORRES = 12          # slots de torre por rodada
DIM_TORRE = 81         # codificação de uma torre
MAX_RODADA = 100

# --------------------------------------------------------------------------
# pré-processamento
# --------------------------------------------------------------------------
def pontos_do_mapa(caminho: Path, Point) -> list:
   """Lê os pontos do caminho do mapa e interpola as lacunas."""
   mapa = json.loads(Path(caminho).read_text())
   pontos, dist = [], 0
   for i, p in enumerate(mapa):
       if i != len(mapa) - 1 and i != 0:
           a = Point(p["x"], p["y"])
           b = Point(mapa[i + 1]["x"], mapa[i + 1]["y"])
           trecho, dist = a.interpolate(b, dist)
           pontos.extend(trecho)
   return pontos

def bloons_por_rodada(caminho: Path) -> list[list[float]]:
   """Vetor de composição de bloons para cada rodada (entrada `other`)."""
   df = pd.read_csv(caminho)
   colunas = [c for c in df.columns if c not in ("Round", "RBE")]
   d = df.to_dict()
   return [[d[c][r] / 50 for c in colunas] for r in range(MAX_RODADA)]

def limpar(df: pd.DataFrame, TOWER_NAMES) -> pd.DataFrame:
   """Descarta rodadas fora do esquema; normaliza PermaPhoenix."""
   descartar = set()
   for i, (torres, rodada) in enumerate(zip(df["Towers"], df["Round"])):
       if rodada >= MAX_RODADA or len(torres) > N_TORRES:
           descartar.add(i)
           continue
       for t in torres:
           if t["name"] == "PermaPhoenix":
               t["name"] = "WizardMonkey-050"
           elif t["name"].split("-")[0] not in TOWER_NAMES:
               descartar.add(i)
               break
   return df.drop(index=list(descartar)).reset_index(drop=True)

def _separa_nome(nome: str) -> tuple[str, str]:
   partes = nome.split("-")
   return (partes[0], partes[1]) if len(partes) == 2 else (partes[0], "000")

def cobertura_memoizada(pontos_mapa: list, TOWER_RANGES, Point):
   """Devolve uma função de cobertura com cache.
   `find_tower_coverage` custa O(|map_points|) e é chamada ~1M de vezes, mas
   depende só de (torre, upgrade, x, y) — há muito menos combinações
   distintas do que chamadas.
   """
   memo: dict[tuple, float] = {}
   def cobertura(nome: str, x: float, y: float) -> float:
       base, upg = _separa_nome(nome)
       chave = (base, upg, x, y)
       if chave not in memo:
           memo[chave] = TOWER_RANGES[base].find_tower_coverage(
               upg, Point(x, y), pontos_mapa
           )
       return memo[chave]
   cobertura.memo = memo
   return cobertura

def codifica_torre(nome: str, cobertura: float, TOWER_NAMES) -> list[float]:
   """Codificação de 81 dimensões: 5 torres x (1 presença + 15 upgrades) + cobertura."""
   base, upg = _separa_nome(nome)
   i = TOWER_NAMES.index(base)
   multi_hot: list[float] = []
   for passo in upg:
       n = int(passo)
       multi_hot += [1] * n + [0] * (5 - n)
   return (
       [0] * (i * 16) + [1] + multi_hot + [0] * ((4 - i) * 16) + [cobertura / 100]
   )

# --------------------------------------------------------------------------
# construção do cache
# --------------------------------------------------------------------------
def construir_cache(verboso: bool = True) -> dict:
   TOWER_NAMES, TOWER_RANGES, Point = _importa_utilitarios()
   df = (
       pd.read_csv(CAMINHO_SAIDAS, compression="gzip")
       .drop_duplicates()
       .reset_index(drop=True)
   )
   df["Towers"] = df["Towers"].apply(json.loads)
   df = limpar(df, TOWER_NAMES)
   if verboso:
       print(f"rodadas após limpeza: {len(df)}")
   bpr = bloons_por_rodada(CAMINHO_RODADAS)
   dim_other = len(bpr[0])
   cobertura = cobertura_memoizada(
       pontos_do_mapa(CAMINHO_MAPA, Point), TOWER_RANGES, Point)
   n = len(df)
   x_outros = np.zeros((n, dim_other), dtype=np.float32)
   x_torres = np.zeros((n, N_TORRES, DIM_TORRE), dtype=np.float32)
   # y = 1 quando a rodada passou SEM perder vida (Health Lost == 0).
   y = (df["Health Lost"].to_numpy() >= 0).astype(np.float32)
   rodada = df["Round"].to_numpy().astype(np.int64)
   for i, (torres, r) in enumerate(zip(df["Towers"], rodada)):
       x_outros[i] = bpr[r]
       for j, t in enumerate(torres[:N_TORRES]):
           c = cobertura(t["name"], t["position"]["x"], t["position"]["y"])
           x_torres[i, j] = codifica_torre(t["name"], c, TOWER_NAMES)
       if verboso and i % 10000 == 0 and i:
           print(f"  {i}/{n} ({len(cobertura.memo)} coberturas distintas)")
   if verboso:
       print(f"coberturas distintas memoizadas: {len(cobertura.memo)}")
   dados = {
       "outros": torch.from_numpy(x_outros),
       "torres": torch.from_numpy(x_torres),
       "y": torch.from_numpy(y),
       "rodada": torch.from_numpy(rodada),
       "dim_outros": dim_other,
       "dim_torre": DIM_TORRE,
       "n_torres": N_TORRES,
   }
   CACHE.parent.mkdir(parents=True, exist_ok=True)
   torch.save(dados, CACHE)
   if verboso:
       print(f"cache salvo em {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)")
   return dados

def carregar(semente_divisao: int = 0):
   """Carrega o cache e devolve as três partições, 80/10/10, determinísticas."""
   if not CACHE.exists():
       raise FileNotFoundError(
           f"cache ausente: {CACHE}\nRode:  python -m desafio3.dados --construir"
       )
   d = torch.load(CACHE, weights_only=True)
   n = len(d["y"])
   g = torch.Generator().manual_seed(semente_divisao)
   perm = torch.randperm(n, generator=g)
   n_tr, n_val = int(0.8 * n), int(0.1 * n)
   cortes = {
       "treino": perm[:n_tr],
       "val": perm[n_tr : n_tr + n_val],
       "teste": perm[n_tr + n_val :],
   }
   partes = {
       nome: torch.utils.data.TensorDataset(
           d["outros"][idx], d["torres"][idx], d["y"][idx]
       )
       for nome, idx in cortes.items()
   }
   meta = {k: d[k] for k in ("dim_outros", "dim_torre", "n_torres")}
   meta["tamanhos"] = {k: len(v) for k, v in partes.items()}
   meta["prop_positiva"] = float(d["y"].mean())
   return partes, meta

def carregadores(lote: int, semente_divisao: int = 0, semente_ordem: int = 0):
   """DataLoaders com ordem de embaralhamento controlada pela semente."""
   partes, meta = carregar(semente_divisao)
   g = torch.Generator().manual_seed(semente_ordem)
   treino = torch.utils.data.DataLoader(
       partes["treino"], batch_size=lote, shuffle=True, generator=g, drop_last=True
   )
   val = torch.utils.data.DataLoader(partes["val"], batch_size=1024)
   teste = torch.utils.data.DataLoader(partes["teste"], batch_size=1024)
   return treino, val, teste, meta

if __name__ == "__main__":
   p = argparse.ArgumentParser(description="Cache de dados do Desafio 3.")
   p.add_argument("--construir", action="store_true", help="gera o cache de tensores")
   p.add_argument("--info", action="store_true", help="inspeciona o cache existente")
   a = p.parse_args()
   if a.construir:
       construir_cache()
   if a.info or not a.construir:
       partes, meta = carregar()
       print(f"dim_outros={meta['dim_outros']}  dim_torre={meta['dim_torre']}  "
             f"n_torres={meta['n_torres']}")
       print(f"tamanhos: {meta['tamanhos']}")
       print(f"proporção de y=1 (rodada sem perda de vida): {meta['prop_positiva']:.3f}")