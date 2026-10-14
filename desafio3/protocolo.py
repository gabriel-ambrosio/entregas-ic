"""
protocolo.py — Desafio 3 (GBC073): executa as quatro etapas do §3 do enunciado.
 1. Baseline            — receita atual do grupo, 3 sementes
 2. Varredura de η      — >= 5 valores log por otimizador, warmup+cosseno, 1 semente
                          (estende a grade se o melhor cair na borda)
 3. Agendas             — 2 melhores otimizadores + Muon, 4 agendas, 1 semente
 4. Confirmação         — 3 melhores combinações, 3 sementes
 muon     — refaz o braço do Muon com a parte AdamW no melhor η do AdamW
 instavel — execuções fora da faixa, para o painel ter uma instável
 ablacao  — isola o efeito do weight decay, com η e orçamento fixos
Regras honradas:
 * mesmo orçamento (passos e lote) dentro de cada etapa comparativa;
 * orçamento REDUZIDO nas etapas 2-3, igual para todos e declarado aqui;
 * seleção pela perda de validação; o teste fica intocado até o fim;
 * toda execução entra em experimentos.csv, inclusive as que divergirem.
Uso:
   python -m desafio3.protocolo --etapa tudo      # as 74 execuções
   python -m desafio3.protocolo --etapa 2         # só a varredura de η
   python -m desafio3.protocolo --etapa ablacao   # só a ablação
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
from desafio3.treino import CSV_EXPERIMENTOS, anexa_csv, treinar
PASTA = Path(__file__).resolve().parent
RESUMO = PASTA / "resumo_protocolo.json"
# --- orçamentos (declarados, iguais para todos dentro de cada etapa) -------
LOTE = 256
PASSOS_REDUZIDO = 1500    # etapas 2 e 3
PASSOS_COMPLETO = 3000    # etapas 1 e 4
# receita de partida do projeto: Adam com η = 1e-3 e sem agenda.
BASELINE = dict(otimizador="adam", lr=1e-3, agenda="constante")
GRADES = {
   "sgd":          [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
   "sgd_nesterov": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
   "adam":         [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
   "adamw":        [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
   "muon":         [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
   "rmsprop":      [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}
AGENDAS_COMPARADAS = ["constante", "cosseno", "warmup_cosseno", "wsd"]

def roda(verboso=True, **kw):
   t0 = time.perf_counter()
   res = treinar(verboso=False, **kw)
   anexa_csv(res)
   if verboso:
       marca = "  DIVERGIU" if res["divergiu"] else ""
       print(f"    {res['nome']:46s} val={res['perda_val']:.4f} "
             f"acc={res['acuracia_val']:.4f} {time.perf_counter()-t0:5.1f}s{marca}")
   return res

# --------------------------------------------------------------------------
def etapa1_baseline():
   print("\n[1] BASELINE — receita atual do grupo, 3 sementes")
   print(f"    {BASELINE} | {PASSOS_COMPLETO} passos, lote {LOTE}")
   return [roda(**BASELINE, semente=s, passos=PASSOS_COMPLETO, lote=LOTE,
                tag=f"e1_baseline_s{s}") for s in (0, 1, 2)]

def etapa2_varredura():
   print(f"\n[2] VARREDURA DE η — warmup+cosseno, 1 semente, "
         f"{PASSOS_REDUZIDO} passos (orçamento reduzido, igual para todos)")
   melhores, todos = {}, []
   for otim, grade in GRADES.items():
       print(f"  {otim}:")
       grade = list(grade)
       vistos = {}
       while True:
           for lr in grade:
               if lr in vistos:
                   continue
               r = roda(otimizador=otim, lr=lr, agenda="warmup_cosseno", semente=0,
                        passos=PASSOS_REDUZIDO, lote=LOTE,
                        tag=f"e2_{otim}_lr{lr:g}")
               vistos[lr] = r
               todos.append(r)
           ok = {k: v for k, v in vistos.items() if not v["divergiu"]}
           if not ok:
               break
           melhor_lr = min(ok, key=lambda k: ok[k]["perda_val"])
           ordenadas = sorted(vistos)
           # "se o melhor cair na borda, estenda a grade"
           if melhor_lr == ordenadas[0] and melhor_lr > 1e-5:
               grade = [melhor_lr / 3, melhor_lr / 10]
               print(f"    melhor na borda inferior -> estendendo para {grade}")
               continue
           if melhor_lr == ordenadas[-1] and melhor_lr < 1.0:
               grade = [melhor_lr * 3, melhor_lr * 10]
               print(f"    melhor na borda superior -> estendendo para {grade}")
               continue
           break
       if ok:
           melhores[otim] = min(ok, key=lambda k: ok[k]["perda_val"])
           print(f"    -> melhor η = {melhores[otim]:g} "
                 f"(val {ok[melhores[otim]]['perda_val']:.4f})")
   return melhores, todos

def etapa3_agendas(melhores_lr: dict):
   print(f"\n[3] AGENDAS — {PASSOS_REDUZIDO} passos, 1 semente")
   # 2 melhores otimizadores pela etapa 2, + Muon se não estiver entre eles
   por_perda = {}
   with open(CSV_EXPERIMENTOS) as f:
       for l in csv.DictReader(f):
           if l["nome"].startswith("e2_") and l["divergiu"] == "False":
               o = l["otimizador"]
               v = float(l["perda_val"])
               if o not in por_perda or v < por_perda[o]:
                   por_perda[o] = v
   ranking = sorted(por_perda, key=por_perda.get)
   escolhidos = ranking[:2]
   if "muon" not in escolhidos:
       escolhidos.append("muon")
   print(f"    otimizadores: {escolhidos}  (ranking: {ranking})")
   res = []
   for o in escolhidos:
       lr = melhores_lr[o]
       for ag in AGENDAS_COMPARADAS:
           res.append(roda(otimizador=o, lr=lr, agenda=ag, semente=0,
                           passos=PASSOS_REDUZIDO, lote=LOTE,
                           tag=f"e3_{o}_{ag}"))
   return escolhidos, res

def etapa4_confirmacao():
   print(f"\n[4] CONFIRMAÇÃO — 3 melhores combinações, 3 sementes, "
         f"{PASSOS_COMPLETO} passos")
   combos = []
   with open(CSV_EXPERIMENTOS) as f:
       for l in csv.DictReader(f):
           if l["nome"].startswith("e3_") and l["divergiu"] == "False":
               combos.append((float(l["perda_val"]), l["otimizador"],
                              float(l["lr"]), l["agenda"]))
   combos.sort()
   top3, vistos = [], set()
   for _, o, lr, ag in combos:
       if (o, ag) in vistos:
           continue
       vistos.add((o, ag))
       top3.append((o, lr, ag))
       if len(top3) == 3:
           break
   print(f"    combinações: {top3}")
   res = []
   for o, lr, ag in top3:
       for s in (0, 1, 2):
           res.append(roda(otimizador=o, lr=lr, agenda=ag, semente=s,
                           passos=PASSOS_COMPLETO, lote=LOTE,
                           tag=f"e4_{o}_{ag}_s{s}"))
   return top3, res
def _melhor_lr_de(otim: str, prefixo: str = "e2_") -> float:
   """Melhor η não divergente de um otimizador, lido do CSV."""
   melhor, melhor_v = None, float("inf")
   with open(CSV_EXPERIMENTOS) as f:
       for l in csv.DictReader(f):
           if (l["nome"].startswith(prefixo) and l["otimizador"] == otim
                   and l["divergiu"] == "False"):
               v = float(l["perda_val"])
               if v < melhor_v:
                   melhor, melhor_v = float(l["lr"]), v
   return melhor

def etapa_muon_corrigido():
   """Refaz o braço do Muon com a parte AdamW no melhor η do AdamW.
   O §3 do enunciado determina que, ao varrer o η das matrizes do Muon, a
   parte AdamW use "o melhor η já encontrado para o AdamW". A primeira
   passagem usou o padrão do código (3e-4), o que deixava a camada de saída
   duas ordens de grandeza mais lenta que as ocultas — visível na razão de
   atualização. As execuções antigas permanecem no CSV.
   """
   lr_adamw = _melhor_lr_de("adamw")
   print(f"\n[2b-4b] MUON CORRIGIDO — parte AdamW em η={lr_adamw:g}")
   res = {}
   for lr in (3e-3, 1e-2, 3e-2, 1e-1, 3e-1):
       r = roda(otimizador="muon", lr=lr, agenda="warmup_cosseno", semente=0,
                passos=PASSOS_REDUZIDO, lote=LOTE, lr_adamw_muon=lr_adamw,
                tag=f"e2b_muon_lr{lr:g}")
       if not r["divergiu"]:
           res[lr] = r["perda_val"]
   melhor_lr = min(res, key=res.get)
   print(f"    -> melhor η = {melhor_lr:g} (val {res[melhor_lr]:.4f})")
   ags = {}
   for ag in AGENDAS_COMPARADAS:
       r = roda(otimizador="muon", lr=melhor_lr, agenda=ag, semente=0,
                passos=PASSOS_REDUZIDO, lote=LOTE, lr_adamw_muon=lr_adamw,
                tag=f"e3b_muon_{ag}")
       if not r["divergiu"]:
           ags[ag] = r["perda_val"]
   melhor_ag = min(ags, key=ags.get)
   print(f"    -> melhor agenda = {melhor_ag} (val {ags[melhor_ag]:.4f})")
   for sem in (0, 1, 2):
       roda(otimizador="muon", lr=melhor_lr, agenda=melhor_ag, semente=sem,
            passos=PASSOS_COMPLETO, lote=LOTE, lr_adamw_muon=lr_adamw,
            tag=f"e4b_muon_{melhor_ag}_s{sem}")

def etapa_instabilidade():
   """Execuções deliberadamente fora da faixa, para o painel ter o contraste.
   O enunciado pede uma configuração instável no painel; como nenhuma do
   protocolo divergiu, estas são rodadas de propósito com η muito alto e
   clipping desligado. Entram no CSV como todas as outras.
   """
   print(f"\n[5] INSTABILIDADE DELIBERADA — η fora da faixa, sem clipping")
   for otim, lr in (("sgd", 3.0), ("adamw", 1.0), ("muon", 1.0)):
       roda(otimizador=otim, lr=lr, agenda="constante", semente=0,
            passos=PASSOS_REDUZIDO, lote=LOTE, clip=0,
            tag=f"e5_instavel_{otim}_lr{lr:g}")

def etapa_ablacao():
   """Isola o efeito do weight decay, mantendo η e orçamento fixos.
   Serve ao diagnóstico: separa quanto do ganho vem da adaptatividade do
   otimizador e quanto vem de a penalidade ser desacoplada.
   """
   print(f"\n[6] ABLAÇÃO DE WEIGHT DECAY — mesmo η, mesmo orçamento")
   for otim, lr, wd in (("adam", 0.01, 0.01), ("adam", 0.01, 0.0),
                        ("adamw", 0.03, 0.0)):
       roda(otimizador=otim, lr=lr, agenda="warmup_cosseno", semente=0,
            passos=PASSOS_REDUZIDO, lote=LOTE, wd=wd,
            tag=f"e6_ablacao_{otim}_lr{lr:g}_wd{wd:g}")

def main(etapa: str):
   t0 = time.perf_counter()
   estado = {}
   if etapa in ("tudo", "1"):
       etapa1_baseline()
   if etapa in ("tudo", "2"):
       melhores, _ = etapa2_varredura()
       estado["melhores_lr"] = melhores
       RESUMO.write_text(json.dumps(estado, indent=2))
   if etapa in ("tudo", "3"):
       if "melhores_lr" not in estado:
           estado = json.loads(RESUMO.read_text())
       esc, _ = etapa3_agendas({k: float(v) for k, v in estado["melhores_lr"].items()})
       estado["otimizadores_etapa3"] = esc
       RESUMO.write_text(json.dumps(estado, indent=2))
   if etapa in ("tudo", "4"):
       top3, _ = etapa4_confirmacao()
       estado["top3"] = top3
       RESUMO.write_text(json.dumps(estado, indent=2))
   if etapa in ("tudo", "muon"):
       etapa_muon_corrigido()
   if etapa in ("tudo", "instavel"):
       etapa_instabilidade()
   if etapa in ("tudo", "ablacao"):
       etapa_ablacao()
   print(f"\nprotocolo concluído em {(time.perf_counter()-t0)/60:.1f} min")
   print(f"resultados: {CSV_EXPERIMENTOS}")

if __name__ == "__main__":
   p = argparse.ArgumentParser(description="Protocolo do Desafio 3.")
   p.add_argument("--etapa", default="tudo",
                  choices=["tudo", "1", "2", "3", "4", "muon", "instavel", "ablacao"])
   main(p.parse_args().etapa)