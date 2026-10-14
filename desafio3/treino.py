"""
treino.py — Desafio 3 (GBC073): treino configurável e instrumentado.
O que o desafio exige está aqui:
 * otimizador, taxa de aprendizado, agenda e semente são argumentos de linha
   de comando, para que cada comparação do protocolo seja reproduzível;
 * registro, a cada k passos, dos cinco sinais do §1 do enunciado:
     perda (treino por lote, validação periódica)
     ‖∇θ L‖₂  — global e por camada, SEMPRE antes do clipping
     ‖θ‖₂     — global e por camada
     razão de atualização ‖θ_{t+1} − θ_t‖ / ‖θ_t‖, por camada
     taxa efetiva η_t
 * orçamento em PASSOS, não em épocas, para que toda comparação use o mesmo
   orçamento mesmo com otimizadores de custo diferente.
Uso:
   python -m desafio3.treino --otimizador adamw --lr 3e-4 --agenda warmup_cosseno
   python -m desafio3.treino --help
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
from desafio3.dados import carregadores
from desafio3.modelo import CAMADAS_MONITORADAS, PreditorDeRodada, parametros_muon_e_adamw
PASTA = Path(__file__).resolve().parent
LOGS = PASTA / "logs"
CSV_EXPERIMENTOS = PASTA / "experimentos.csv"
OTIMIZADORES = ("sgd", "sgd_nesterov", "adam", "adamw", "muon", "rmsprop")
AGENDAS = ("constante", "cosseno", "warmup_cosseno", "onecycle", "wsd")

# --------------------------------------------------------------------------
# otimizadores e agendas
# --------------------------------------------------------------------------
def monta_otimizador(nome: str, modelo: nn.Module, lr: float, wd: float,
                    lr_adamw_muon: float = 3e-4):
   """Devolve a lista de otimizadores (o Muon precisa de dois grupos)."""
   ps = [p for p in modelo.parameters() if p.requires_grad]
   if nome == "sgd":
       return [torch.optim.SGD(ps, lr=lr, momentum=0.9, weight_decay=wd)]
   if nome == "sgd_nesterov":
       return [torch.optim.SGD(ps, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)]
   if nome == "adam":
       return [torch.optim.Adam(ps, lr=lr, weight_decay=wd)]
   if nome == "adamw":
       return [torch.optim.AdamW(ps, lr=lr, weight_decay=wd)]
   if nome == "rmsprop":
       return [torch.optim.RMSprop(ps, lr=lr, momentum=0.9, weight_decay=wd)]
   if nome == "muon":
       # Muon só nas matrizes 2-D ocultas; vieses e camada de saída no AdamW.
       muon_p, adamw_p = parametros_muon_e_adamw(modelo)
       return [
           torch.optim.Muon(muon_p, lr=lr, momentum=0.95, weight_decay=wd),
           torch.optim.AdamW(adamw_p, lr=lr_adamw_muon, weight_decay=wd),
       ]
   raise ValueError(f"otimizador desconhecido: {nome}")

def monta_agenda(nome: str, opt, total: int, aquecimento: int = 0):
   """A MESMA agenda é aplicada a todos os otimizadores da lista."""
   S = torch.optim.lr_scheduler
   if nome == "constante":
       return S.LambdaLR(opt, lambda _: 1.0)
   if nome == "cosseno":
       return S.CosineAnnealingLR(opt, T_max=total)
   if nome == "warmup_cosseno":
       n_aq = max(1, aquecimento)
       return S.SequentialLR(
           opt,
           [S.LinearLR(opt, start_factor=0.01, total_iters=n_aq),
            S.CosineAnnealingLR(opt, T_max=max(1, total - n_aq))],
           milestones=[n_aq],
       )
   if nome == "onecycle":
       return S.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                           total_steps=total, pct_start=0.3)
   if nome == "wsd":
       # warmup -> estável -> decaimento final (Hägele et al., 2024)
       n_aq = max(1, aquecimento)
       inicio_decai = int(0.8 * total)
       def fator(t):
           if t < n_aq:
               return 0.01 + 0.99 * t / n_aq
           if t < inicio_decai:
               return 1.0
           p = (t - inicio_decai) / max(1, total - inicio_decai)
           return max(0.01, 1.0 - p)
       return S.LambdaLR(opt, fator)
   raise ValueError(f"agenda desconhecida: {nome}")

# --------------------------------------------------------------------------
# instrumentação
# --------------------------------------------------------------------------
def normas_por_camada(modelo: nn.Module) -> tuple[dict, dict, float, float]:
   """‖∇θ‖ e ‖θ‖, globais e das camadas monitoradas. Chame ANTES do clipping."""
   g2_total, t2_total = 0.0, 0.0
   g_cam, t_cam = {}, {}
   for nome, p in modelo.named_parameters():
       t2 = float((p.detach() ** 2).sum())
       t2_total += t2
       g2 = float((p.grad.detach() ** 2).sum()) if p.grad is not None else 0.0
       g2_total += g2
       raiz = nome.split(".")[0]
       if raiz in CAMADAS_MONITORADAS:
           g_cam[raiz] = g_cam.get(raiz, 0.0) + g2
           t_cam[raiz] = t_cam.get(raiz, 0.0) + t2
   return ({k: math.sqrt(v) for k, v in g_cam.items()},
           {k: math.sqrt(v) for k, v in t_cam.items()},
           math.sqrt(g2_total), math.sqrt(t2_total))

@torch.no_grad()
def avalia(modelo: nn.Module, carregador, perda_fn) -> dict:
   modelo.eval()
   soma, n, acertos, tp, pos = 0.0, 0, 0, 0, 0
   for outros, torres, y in carregador:
       logit = modelo(outros, torres)
       soma += float(perda_fn(logit, y)) * len(y)
       pred = (logit > 0).float()
       acertos += int((pred == y).sum())
       tp += int(((pred == 1) & (y == 1)).sum())
       pos += int((y == 1).sum())
       n += len(y)
   modelo.train()
   return {"perda": soma / n, "acuracia": acertos / n,
           "revocacao": tp / pos if pos else float("nan")}

# --------------------------------------------------------------------------
# laço de treino
# --------------------------------------------------------------------------
def treinar(otimizador="adamw", lr=3e-4, agenda="warmup_cosseno", semente=0,
           passos=1500, lote=256, wd=0.01, k_registro=25, val_a_cada=250,
           clip=1.0, lr_adamw_muon=3e-4, aquecimento_frac=0.1,
           tag=None, salvar_log=True, verboso=True,
           salvar_pesos: str | None = None) -> dict:
   torch.manual_seed(semente)
   t_inicio = time.perf_counter()
   treino_dl, val_dl, teste_dl, meta = carregadores(
       lote, semente_divisao=0, semente_ordem=semente
   )
   modelo = PreditorDeRodada(dim_outros=meta["dim_outros"],
                             dim_torre=meta["dim_torre"],
                             n_torres=meta["n_torres"])
   perda_fn = nn.BCEWithLogitsLoss()
   opts = monta_otimizador(otimizador, modelo, lr, wd, lr_adamw_muon)
   aquec = int(aquecimento_frac * passos)
   scheds = [monta_agenda(agenda, o, passos, aquec) for o in opts]
   nome = tag or f"{otimizador}_lr{lr:g}_{agenda}_s{semente}"
   registro, melhor_val, passo_melhor, divergiu = [], float("inf"), -1, False
   it = iter(treino_dl)
   for t in range(passos):
       try:
           outros, torres, y = next(it)
       except StopIteration:
           it = iter(treino_dl)
           outros, torres, y = next(it)
       logit = modelo(outros, torres)
       perda = perda_fn(logit, y)
       perda_v = perda.detach().item()
       for o in opts:
           o.zero_grad(set_to_none=True)
       perda.backward()
       # --- sinais, SEMPRE antes do clipping ---------------------------
       anotar = (t % k_registro == 0) or (t == passos - 1)
       if anotar:
           g_cam, t_cam, g_tot, th_tot = normas_por_camada(modelo)
           antes = {n_: p.detach().clone()
                    for n_, p in modelo.named_parameters()
                    if n_.split(".")[0] in CAMADAS_MONITORADAS and n_.endswith("weight")}
       if clip and clip > 0:
           torch.nn.utils.clip_grad_norm_(modelo.parameters(), clip)
       for o in opts:
           o.step()
       for s in scheds:
           s.step()
       if anotar:
           rho = {}
           for n_, p in modelo.named_parameters():
               if n_ in antes:
                   base = antes[n_].norm()
                   rho[n_.split(".")[0]] = float(
                       (p.detach() - antes[n_]).norm() / (base + 1e-12)
                   )
           linha = {
               "passo": t,
               "perda_treino": perda_v,
               "grad": g_tot,
               "theta": th_tot,
               "lr": opts[0].param_groups[0]["lr"],
               **{f"grad/{c}": g_cam.get(c, 0.0) for c in CAMADAS_MONITORADAS},
               **{f"theta/{c}": t_cam.get(c, 0.0) for c in CAMADAS_MONITORADAS},
               **{f"rho/{c}": rho.get(c, 0.0) for c in CAMADAS_MONITORADAS},
           }
           if not math.isfinite(perda_v) or not math.isfinite(g_tot):
               divergiu = True
           registro.append(linha)
       if (t % val_a_cada == 0 and t) or t == passos - 1:
           m = avalia(modelo, val_dl, perda_fn)
           registro[-1].update({"perda_val": m["perda"],
                                "acuracia_val": m["acuracia"],
                                "revocacao_val": m["revocacao"]})
           if m["perda"] < melhor_val:
               melhor_val, passo_melhor = m["perda"], t
           if verboso:
               print(f"  passo {t:5d}  treino {perda_v:.4f}  "
                     f"val {m['perda']:.4f}  acc {m['acuracia']:.4f}  "
                     f"lr {opts[0].param_groups[0]['lr']:.2e}")
           if not math.isfinite(m["perda"]):
               divergiu = True
               break
   dur = time.perf_counter() - t_inicio
   final = avalia(modelo, val_dl, perda_fn)
   if salvar_pesos:
       destino = Path(salvar_pesos)
       destino.parent.mkdir(parents=True, exist_ok=True)
       torch.save({"state_dict": modelo.state_dict(),
                   "arquitetura": {"dim_outros": meta["dim_outros"],
                                   "dim_torre": meta["dim_torre"],
                                   "n_torres": meta["n_torres"]},
                   "receita": {"otimizador": otimizador, "lr": lr,
                               "agenda": agenda, "semente": semente,
                               "passos": passos, "lote": lote, "wd": wd},
                   "metrica_val": final}, destino)
       if verboso:
           print(f"  pesos salvos em {destino}")
   if salvar_log:
       LOGS.mkdir(parents=True, exist_ok=True)
       campos = sorted({c for l in registro for c in l})
       with open(LOGS / f"{nome}.csv", "w", newline="") as f:
           w = csv.DictWriter(f, fieldnames=campos)
           w.writeheader()
           w.writerows(registro)
   return {"nome": nome, "otimizador": otimizador, "lr": lr, "agenda": agenda,
           "semente": semente, "passos": passos, "lote": lote, "wd": wd,
           "perda_val": final["perda"], "acuracia_val": final["acuracia"],
           "revocacao_val": final["revocacao"],
           "melhor_perda_val": melhor_val, "passo_melhor": passo_melhor,
           "divergiu": divergiu, "segundos": round(dur, 1),
           "registro": registro}

def anexa_csv(res: dict, caminho: Path = CSV_EXPERIMENTOS) -> None:
   """Uma linha por execução, inclusive as que divergiram."""
   linha = {k: v for k, v in res.items() if k != "registro"}
   novo = not caminho.exists()
   caminho.parent.mkdir(parents=True, exist_ok=True)
   with open(caminho, "a", newline="") as f:
       w = csv.DictWriter(f, fieldnames=list(linha))
       if novo:
           w.writeheader()
       w.writerow(linha)

def get_args():
   p = argparse.ArgumentParser(description="Treino instrumentado — Desafio 3.")
   p.add_argument("--otimizador", "-O", default="adamw", choices=OTIMIZADORES)
   p.add_argument("--lr", type=float, default=3e-4)
   p.add_argument("--agenda", "-A", default="warmup_cosseno", choices=AGENDAS)
   p.add_argument("--semente", "-S", type=int, default=0)
   p.add_argument("--passos", type=int, default=1500)
   p.add_argument("--lote", type=int, default=256)
   p.add_argument("--wd", type=float, default=0.01)
   p.add_argument("--k-registro", type=int, default=25, help="10 <= k <= 50")
   p.add_argument("--val-a-cada", type=int, default=250)
   p.add_argument("--clip", type=float, default=1.0, help="0 desliga")
   p.add_argument("--lr-adamw-muon", type=float, default=3e-4)
   p.add_argument("--tag", default=None)
   p.add_argument("--sem-csv", action="store_true")
   p.add_argument("--salvar-pesos", default=None,
                  help="caminho .pt para gravar o modelo treinado")
   return p.parse_args()

if __name__ == "__main__":
   a = get_args()
   assert 10 <= a.k_registro <= 50, "o enunciado exige 10 <= k <= 50"
   res = treinar(otimizador=a.otimizador, lr=a.lr, agenda=a.agenda,
                 semente=a.semente, passos=a.passos, lote=a.lote, wd=a.wd,
                 k_registro=a.k_registro, val_a_cada=a.val_a_cada,
                 clip=a.clip, lr_adamw_muon=a.lr_adamw_muon, tag=a.tag,
                 salvar_pesos=a.salvar_pesos)
   if not a.sem_csv:
       anexa_csv(res)
   print(f"\n{res['nome']}: perda_val={res['perda_val']:.4f} "
         f"acc={res['acuracia_val']:.4f} divergiu={res['divergiu']} "
         f"({res['segundos']}s)")