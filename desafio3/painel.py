"""
painel.py — Desafio 3 (GBC073): gera painel.pdf.
Conteúdo exigido pelo §5 do enunciado:
 pág. 1  os quatro sinais alinhados no mesmo eixo de passos, para quatro
         configurações: melhor, baseline, melhor com Muon e uma instável
 pág. 2  métrica de validação x η, por otimizador
 pág. 3  razão de atualização por camada — Muon contra AdamW, que é onde o
         enunciado diz que o Muon se diferencia
Decisão de cor: as figuras usam posição (facetas) para identificar
configuração e otimizador, não matiz. Seis cores categóricas com separação
suficiente sob daltonismo não existem nesta faixa de luminosidade — verde e
magenta, por exemplo, ficam a ΔE 1,9 sob protanopia. Onde há duas séries
sobrepostas (Muon x AdamW) o par usado é #3F6DC4 / #A86E00, validado em
ΔE 24,9 sob CVD e contraste 5,00:1 e 4,29:1.
Uso:
   python -m desafio3.painel
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
PASTA = Path(__file__).resolve().parent
LOGS = PASTA / "logs"
CSV_EXP = PASTA / "experimentos.csv"
SAIDA = PASTA / "painel.pdf"
AZUL, DOURADO, TINTA, CINZA, GRADE = "#3F6DC4", "#A86E00", "#1B2233", "#55607A", "#DDE3EE"
VERMELHO = "#B3121B"
CAMADAS = ("torre", "mlp1", "saida")
ESTILO_CAMADA = {"torre": "-", "mlp1": "--", "saida": ":"}
plt.rcParams.update({
   "font.family": "DejaVu Sans", "font.size": 8.5,
   "axes.edgecolor": CINZA, "axes.labelcolor": TINTA, "text.color": TINTA,
   "xtick.color": CINZA, "ytick.color": CINZA, "axes.linewidth": .8,
   "pdf.fonttype": 42, "figure.dpi": 110,
})

def le_experimentos() -> list[dict]:
   with open(CSV_EXP) as f:
       linhas = list(csv.DictReader(f))
   for l in linhas:
       for k in ("lr", "perda_val", "acuracia_val", "melhor_perda_val", "segundos"):
           l[k] = float(l[k]) if l[k] not in ("", "nan") else float("nan")
       l["divergiu"] = l["divergiu"] == "True"
       l["passos"] = int(l["passos"])
       l["semente"] = int(l["semente"])
   return linhas

def le_log_linhas(nome: str) -> list[dict]:
   """Linhas do log preservando o alinhamento entre colunas."""
   p = LOGS / f"{nome}.csv"
   if not p.exists():
       return []
   with open(p) as f:
       return list(csv.DictReader(f))

def pares(linhas, chave):
   """(passo, valor) só onde a coluna foi registrada — a validação é periódica."""
   xs, ys = [], []
   for l in linhas:
       v = l.get(chave, "")
       if v not in ("", None):
           xs.append(float(l["passo"])); ys.append(float(v))
   return xs, ys

def le_log(nome: str) -> dict[str, list[float]]:
   p = LOGS / f"{nome}.csv"
   if not p.exists():
       return {}
   col: dict[str, list[float]] = {}
   with open(p) as f:
       for linha in csv.DictReader(f):
           for k, v in linha.items():
               if v == "":
                   continue
               col.setdefault(k, []).append(float(v))
   return col

def escolhe_configuracoes(exp: list[dict]) -> list[tuple[str, str]]:
   """(rótulo, nome do log) para: melhor, baseline, melhor com Muon, instável."""
   ok = [e for e in exp if not e["divergiu"] and e["nome"].startswith(("e4_", "e4b_"))]
   if not ok:
       ok = [e for e in exp if not e["divergiu"]]
   melhor = min(ok, key=lambda e: e["perda_val"])
   base = next(e for e in exp if e["nome"].startswith("e1_baseline_s0"))
   muons = [e for e in exp if e["otimizador"] == "muon" and not e["divergiu"]]
   melhor_muon = min(muons, key=lambda e: e["perda_val"]) if muons else melhor
   # "instável": prioriza quem divergiu; senão, a execução com o maior PICO de
   # ‖∇θ L‖ relativo à mediana — o sintoma de passo grande demais.
   div = [e for e in exp if e["divergiu"]]
   if div:
       instavel = div[0]
   else:
       def pico(e):
           log = le_log(e["nome"])
           g = log.get("grad") or [0]
           med = sorted(g)[len(g) // 2] or 1e-12
           return max(g) / med
       candidatos = [e for e in exp if e["nome"].startswith("e5_")] or exp
       instavel = max(candidatos, key=pico)
   cfgs = [
       (f"melhor\n{melhor['otimizador']} η={melhor['lr']:g} {melhor['agenda']}", melhor["nome"]),
       (f"baseline\n{base['otimizador']} η={base['lr']:g} {base['agenda']}", base["nome"]),
       (f"melhor com Muon\nη={melhor_muon['lr']:g} {melhor_muon['agenda']}", melhor_muon["nome"]),
       (f"instável\n{instavel['otimizador']} η={instavel['lr']:g}", instavel["nome"]),
   ]
   return cfgs

# --------------------------------------------------------------- página 1
def pagina_sinais(pdf, exp):
   cfgs = escolhe_configuracoes(exp)
   sinais = [("perda_treino", "perda (treino)", False),
             ("grad", "‖∇θ L‖₂", True),
             ("theta", "‖θ‖₂", False),
             ("lr", "η efetiva", False)]
   fig, axs = plt.subplots(4, 4, figsize=(11.7, 8.3), sharex="col")
   fig.suptitle("Os quatro sinais alinhados no mesmo eixo de passos",
                fontsize=13, fontweight="bold", color=TINTA, y=.975)
   fig.text(.5, .943, "cada coluna é uma configuração; ler a mesma coluna vertical "
            "nos quatro gráficos é de onde sai o diagnóstico",
            ha="center", fontsize=8.5, color=CINZA)
   for c, (rotulo, nome) in enumerate(cfgs):
       log = le_log(nome)
       axs[0, c].set_title(rotulo, fontsize=9, color=TINTA, pad=7)
       for r, (chave, ylab, logy) in enumerate(sinais):
           ax = axs[r, c]
           if log.get(chave) and log.get("passo"):
               n = min(len(log["passo"]), len(log[chave]))
               ax.plot(log["passo"][:n], log[chave][:n], color=AZUL, lw=1.2)
               if chave == "perda_treino":
                   xv, yv = pares(le_log_linhas(nome), "perda_val")
                   if xv:
                       ax.plot(xv, yv, color=DOURADO, lw=1.4, marker="o", ms=3.5,
                               mec="white", mew=.8, label="validação", zorder=4)
                       ax.legend(fontsize=7, frameon=False, labelcolor=CINZA)
           else:
               ax.text(.5, .5, "sem registro", ha="center", va="center",
                       transform=ax.transAxes, color=CINZA, fontsize=8)
           if logy:
               ax.set_yscale("log")
           if c == 0:
               ax.set_ylabel(ylab, fontsize=9)
           if r == 3:
               ax.set_xlabel("passo", fontsize=9)
           ax.grid(color=GRADE, lw=.6)
           ax.set_axisbelow(True)
           for s in ("top", "right"):
               ax.spines[s].set_visible(False)
   fig.tight_layout(rect=[0, 0, 1, .93])
   pdf.savefig(fig)
   plt.close(fig)
   # --------------------------------------------------------------- página 2
def pagina_varredura(pdf, exp):
   e2 = [e for e in exp if e["nome"].startswith("e2_")]
   otims = sorted({e["otimizador"] for e in e2})
   base = [e["perda_val"] for e in exp if e["nome"].startswith("e1_")]
   media_base = sum(base) / len(base) if base else None
   n = len(otims)
   cols = 3
   rows = math.ceil(n / cols)
   fig, axs = plt.subplots(rows, cols, figsize=(11.7, 3.3 * rows), squeeze=False)
   fig.suptitle("Métrica de validação × η, por otimizador",
                fontsize=13, fontweight="bold", color=TINTA, y=.98)
   fig.text(.5, .945, "pequenos múltiplos: a identidade do otimizador vem da posição, "
            "não da cor — o eixo y é comum, então as alturas se comparam entre painéis",
            ha="center", fontsize=8.5, color=CINZA)
   finitos = [e["perda_val"] for e in e2 if math.isfinite(e["perda_val"])]
   ymax = min(max(finitos), (media_base or max(finitos)) * 1.8) if finitos else 1
   ymin = min(finitos) * .98 if finitos else 0
   for i, o in enumerate(otims):
       ax = axs[i // cols][i % cols]
       pts = sorted([(e["lr"], e["perda_val"], e["divergiu"]) for e in e2
                     if e["otimizador"] == o])
       xs = [p[0] for p in pts]
       ys = [p[1] if math.isfinite(p[1]) else ymax for p in pts]
       ax.plot(xs, ys, "-o", color=AZUL, lw=1.6, ms=5, mec="white", mew=1,
               zorder=3)
       div = [(x, y) for x, y, d in zip(xs, ys, [p[2] for p in pts]) if d]
       if div:
           ax.plot([d[0] for d in div], [d[1] for d in div], "x",
                   color=VERMELHO, ms=9, mew=2, zorder=4, label="divergiu")
           ax.legend(fontsize=7, frameon=False, labelcolor=CINZA)
       bons = [(x, y) for x, y, d in zip(xs, ys, [p[2] for p in pts]) if not d]
       if bons:
           bx, by = min(bons, key=lambda t: t[1])
           ax.plot([bx], [by], "o", color=DOURADO, ms=8, mec="white", mew=1.5, zorder=5)
           ax.annotate(f"η={bx:g}\n{by:.4f}", (bx, by), textcoords="offset points",
                       xytext=(7, 6), fontsize=7.5, color=DOURADO, fontweight="bold")
       if media_base:
           ax.axhline(media_base, color=CINZA, ls="--", lw=1, zorder=2)
           if i == 0:
               ax.text(xs[0], media_base, " baseline", va="bottom", fontsize=7.5,
                       color=CINZA)
       ax.set_xscale("log")
       ax.set_ylim(ymin, ymax)
       ax.set_title(o, fontsize=10, color=TINTA, fontweight="bold")
       ax.set_xlabel("η (escala log)", fontsize=8.5)
       if i % cols == 0:
           ax.set_ylabel("perda de validação", fontsize=8.5)
       ax.grid(color=GRADE, lw=.6)
       ax.set_axisbelow(True)
       for s in ("top", "right"):
           ax.spines[s].set_visible(False)
   for j in range(n, rows * cols):
       axs[j // cols][j % cols].axis("off")
   fig.tight_layout(rect=[0, 0, 1, .93])
   pdf.savefig(fig)
   plt.close(fig)

# --------------------------------------------------------------- página 3
def pagina_muon(pdf, exp):
   """Razão de atualização por camada: onde o Muon se separa do AdamW."""
   def melhor_de(o):
       c = [e for e in exp if e["otimizador"] == o and not e["divergiu"]
            and e["nome"].startswith(("e3_", "e3b_", "e4_", "e4b_"))]
       if not c:
           c = [e for e in exp if e["otimizador"] == o and not e["divergiu"]]
       return min(c, key=lambda e: e["perda_val"]) if c else None
   muon, adamw = melhor_de("muon"), melhor_de("adamw")
   fig, axs = plt.subplots(1, 2, figsize=(11.7, 4.6), sharey=True)
   fig.suptitle("Razão de atualização por camada  ‖θₜ₊₁ − θₜ‖ / ‖θₜ‖",
                fontsize=13, fontweight="bold", color=TINTA, y=.97)
   fig.text(.5, .905, "a referência de um treino saudável é ~10⁻³; o Muon iguala o "
            "tamanho do passo entre as direções singulares, o AdamW não",
            ha="center", fontsize=8.5, color=CINZA)
   for ax, (e, cor, titulo) in zip(axs, [(muon, AZUL, "Muon"), (adamw, DOURADO, "AdamW")]):
       if e is None:
           ax.text(.5, .5, "sem execução", ha="center", va="center",
                   transform=ax.transAxes, color=CINZA)
           continue
       log = le_log(e["nome"])
       for cam in CAMADAS:
           chave = f"rho/{cam}"
           if log.get(chave):
               n = min(len(log["passo"]), len(log[chave]))
               ax.plot(log["passo"][:n], log[chave][:n], ESTILO_CAMADA[cam],
                       color=cor, lw=1.3, label=cam)
       ax.axhline(1e-3, color=CINZA, ls=":", lw=1)
       ax.text(0.01, 1.05e-3, "referência 10⁻³", fontsize=7.5, color=CINZA,
               transform=ax.get_yaxis_transform())
       ax.set_yscale("log")
       ax.set_title(f"{titulo} — η={e['lr']:g}, {e['agenda']}", fontsize=10,
                    color=TINTA, fontweight="bold")
       ax.set_xlabel("passo", fontsize=9)
       ax.legend(fontsize=8, frameon=False, labelcolor=CINZA, title="camada",
                 title_fontsize=8)
       ax.grid(color=GRADE, lw=.6)
       ax.set_axisbelow(True)
       for s in ("top", "right"):
           ax.spines[s].set_visible(False)
   axs[0].set_ylabel("razão de atualização", fontsize=9)
   fig.tight_layout(rect=[0, 0, 1, .89])
   pdf.savefig(fig)
   plt.close(fig)

def main():
   exp = le_experimentos()
   with PdfPages(SAIDA) as pdf:
       pagina_sinais(pdf, exp)
       pagina_varredura(pdf, exp)
       pagina_muon(pdf, exp)
   print(f"painel salvo: {SAIDA}")

if __name__ == "__main__":
   main()