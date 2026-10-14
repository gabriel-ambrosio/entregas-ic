"""
slide.py — Desafio 3 (GBC073): gera slide.pdf (um slide, 3 minutos).
O enunciado pede três coisas no slide: a receita vencedora, o ganho ± desvio,
e um diagnóstico lido nos sinais que justifique a escolha. O texto do
diagnóstico fica em DIAGNOSTICO, abaixo — ele é escrito a partir da leitura
dos logs, não gerado automaticamente, porque é exatamente o item que a rubrica
avalia como "interpretação sustentada pelos sinais".
Uso:
   python -m desafio3.slide
"""
from __future__ import annotations
import csv
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
PASTA = Path(__file__).resolve().parent
CSV_EXP = PASTA / "experimentos.csv"
LOGS = PASTA / "logs"
SAIDA = PASTA / "slide.pdf"
AZUL_ESC, AZUL, DOURADO = "#16357C", "#3F6DC4", "#A86E00"
TINTA, CINZA, GRADE, BOXBG = "#1B2233", "#55607A", "#DDE3EE", "#EDF2FB"
W, H, ML, MR = 13.333, 7.5, .72, .72
# preenchido por main() a partir dos dados; o texto do diagnóstico é autoral
DIAGNOSTICO = (
   "O baseline não estava mal otimizado — estava sobre-regularizado. ‖θ‖ despenca de 11 para 4 "
   "e estaciona enquanto a perda empaca em 0,46. A causa é a L2 ACOPLADA do Adam: com o mesmo "
   "η=0,01, zerar o weight decay leva a perda de 0,4391 a 0,2737 e recupera quase toda a "
   "distância até o AdamW. Desacoplar permite subir η a 0,03 sem esmagar os pesos."
)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

def le(caminho):
   with open(caminho) as f:
       return list(csv.DictReader(f))

def le_log(nome):
   p = LOGS / f"{nome}.csv"
   if not p.exists():
       return {}
   col = {}
   for linha in le(p):
       for k, v in linha.items():
           if v != "":
               col.setdefault(k, []).append(float(v))
   return col

def apura():
   exp = le(CSV_EXP)
   for e in exp:
       e["perda_val"] = float(e["perda_val"])
       e["acuracia_val"] = float(e["acuracia_val"])
       e["lr"] = float(e["lr"])
       e["div"] = e["divergiu"] == "True"
   base = [e for e in exp if e["nome"].startswith("e1_baseline")]
   conf = [e for e in exp if e["nome"].startswith(("e4_", "e4b_")) and not e["div"]]
   if not conf:
       conf = [e for e in exp if not e["div"]]
   grupos = {}
   for e in conf:
       grupos.setdefault((e["otimizador"], e["lr"], e["agenda"]), []).append(e)
   vencedor = min(grupos.items(),
                  key=lambda kv: st.mean(x["perda_val"] for x in kv[1]))
   chave, runs = vencedor
   return {
       "exp": exp,
       "base_perda": [e["perda_val"] for e in base],
       "base_acc": [e["acuracia_val"] for e in base],
       "base_cfg": (base[0]["otimizador"], base[0]["lr"], base[0]["agenda"]) if base else None,
       "venc_cfg": chave,
       "venc_perda": [e["perda_val"] for e in runs],
       "venc_acc": [e["acuracia_val"] for e in runs],
       "venc_nome": runs[0]["nome"],
       "grupos": grupos,
   }

def mm(v):
   return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

def main():
   d = apura()
   bp, bps = mm(d["base_perda"]); ba, bas = mm(d["base_acc"])
   vp, vps = mm(d["venc_perda"]); va, vas = mm(d["venc_acc"])
   o, lr, ag = d["venc_cfg"]
   ganho = bp - vp
   sep = abs(ganho) > (bps + vps)   # ganho maior que a variação entre sementes
   fig = plt.figure(figsize=(W, H))
   ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
   ax.axis("off"); ax.invert_yaxis()
   ax.text(ML, .60, f"Receita vencedora: {o} · η = {lr:g} · agenda {ag}",
           fontsize=23, fontweight="bold", color=AZUL_ESC, va="baseline")
   ax.text(ML, .94, "BTD6 Magic AI — preditor de rodada · Deep Sets, 6.721 parâmetros · "
           "3000 passos, lote 256, 3 sementes",
           fontsize=12, color=CINZA, va="baseline")
   ax.add_patch(Rectangle((ML, 1.12), .64, .035, color=AZUL_ESC, lw=0))
   # ---- números (cartões empilhados: evita colisão entre colunas)
   bo, blr, bag = d["base_cfg"]
   ax.text(ML, 1.72, "GANHO SOBRE A BASELINE", fontsize=10.5, color=AZUL_ESC,
           fontweight="bold", va="baseline")
   cartoes = [("baseline", f"{bo} · η={blr:g} · {bag}", bp, bps, ba, CINZA),
              ("vencedora", f"{o} · η={lr:g} · {ag}", vp, vps, va, AZUL)]
   for i, (rot, cfg, pm, ps, am, cor) in enumerate(cartoes):
       y = 2.22 + i * 1.05
       ax.add_patch(Rectangle((ML, y - 0.30), 0.045, 0.80, color=cor, lw=0))
       ax.text(ML + 0.20, y, rot, fontsize=12.5, fontweight="bold", color=cor,
               va="baseline")
       ax.text(ML + 1.35, y, cfg, fontsize=11.5, color=TINTA, va="baseline")
       ax.text(ML + 0.20, y + 0.38, f"perda {pm:.4f} ± {ps:.4f}", fontsize=12,
               fontweight="bold", color=cor, va="baseline")
       ax.text(ML + 3.05, y + 0.38, f"acurácia {am*100:.1f}%", fontsize=12,
               color=TINTA, va="baseline")
   ax.text(ML, 4.42, f"Δ perda = {ganho:+.4f}", fontsize=14, fontweight="bold",
           color=AZUL if sep else DOURADO, va="baseline")
   ax.text(ML + 2.62, 4.42, f"(± {bps + vps:.4f} entre sementes)",
           fontsize=11.5, color=CINZA, va="baseline")
   ax.text(ML, 4.78,
           "o ganho supera a variação entre sementes" if sep else
           "o ganho NÃO supera a variação — não há vencedor declarável",
           fontsize=11, color=AZUL if sep else DOURADO, va="baseline")
   outros = sorted(((st.mean(x["perda_val"] for x in v), k)
                    for k, v in d["grupos"].items()))
   if len(outros) > 1:
       p2, k2 = outros[1]
       s2 = st.stdev([x["perda_val"] for x in d["grupos"][k2]]) if len(d["grupos"][k2]) > 1 else 0
       empata = abs(p2 - vp) <= (vps + s2)
       ax.text(ML, 5.08, f"2º  {k2[0]} η={k2[1]:g} {k2[2]} — {p2:.4f} ± {s2:.4f}",
               fontsize=10.5, color=CINZA, va="baseline")
       if empata:
           ax.text(ML, 5.36, "empata com a vencedora dentro da variação entre sementes",
                   fontsize=10.5, color=DOURADO, va="baseline")
   # ---- figura: razão de atualização, Muon x AdamW
   gx = (ML + 6.05) / W
   g = fig.add_axes([gx, 1 - 5.05 / H, (W - MR) / W - gx, 3.10 / H])
   def melhor(ot):
       c = [e for e in d["exp"] if e["otimizador"] == ot and not e["div"]
            and e["nome"].startswith(("e3_", "e3b_", "e4_", "e4b_"))]
       return min(c, key=lambda e: e["perda_val"]) if c else None
   for ot, cor in (("muon", AZUL), ("adamw", DOURADO)):
       e = melhor(ot)
       if not e:
           continue
       log = le_log(e["nome"])
       for cam, ls in (("torre", "-"), ("mlp1", "--"), ("saida", ":")):
           ch = f"rho/{cam}"
           if log.get(ch):
               n = min(len(log["passo"]), len(log[ch]))
               g.plot(log["passo"][:n], log[ch][:n], ls, color=cor, lw=1.3,
                      label=f"{ot} · {cam}")
   g.axhline(1e-3, color=CINZA, ls=":", lw=1)
   g.text(.012, 1.12e-3, "referência 10⁻³", fontsize=8.5, color=CINZA,
          transform=g.get_yaxis_transform())
   g.set_yscale("log")
   g.set_xlabel("passo", fontsize=10, color=CINZA)
   g.set_ylabel("razão de atualização  ‖Δθ‖/‖θ‖", fontsize=10, color=CINZA)
   g.set_title("onde o Muon se separa do AdamW", fontsize=11,
               color=TINTA, fontweight="bold")
   g.tick_params(labelsize=9, colors=CINZA)
   g.grid(color=GRADE, lw=.6); g.set_axisbelow(True)
   for s in ("top", "right"):
       g.spines[s].set_visible(False)
   for s in ("left", "bottom"):
       g.spines[s].set_color(CINZA)
   g.legend(fontsize=8, frameon=False, labelcolor=CINZA, ncol=2, loc="upper right")
   # ---- diagnóstico
   import textwrap
   linhas_d = textwrap.wrap(DIAGNOSTICO, 116)
   alt = .30 + len(linhas_d) * .225 + .14
   ax.add_patch(FancyBboxPatch((ML, 5.62), W - ML - MR, alt,
                boxstyle="round,pad=0,rounding_size=.05",
                facecolor=BOXBG, edgecolor="none"))
   ax.add_patch(Rectangle((ML, 5.62), .05, alt, color=AZUL, lw=0))
   ax.text(ML + .22, 5.88, "DIAGNÓSTICO LIDO NOS SINAIS", fontsize=9.5,
           color=AZUL, fontweight="bold", va="baseline")
   for i, ln in enumerate(linhas_d):
       ax.text(ML + .22, 6.17 + i * .225, ln, fontsize=12, color=TINTA, va="baseline")
   ax.text(ML, H - .30, "Desafio 3 — GBC073 · Inteligência Computacional",
           fontsize=9.5, color=CINZA, va="baseline")
   n_exec = len(d["exp"])
   n_div = sum(1 for e in d["exp"] if e["div"])
   ax.text(W - MR, H - .30, f"{n_exec} execuções · {n_div} divergiram",
           fontsize=9.5, color=CINZA, va="baseline", ha="right")
   with PdfPages(SAIDA) as pdf:
       pdf.savefig(fig)
   plt.close(fig)
   print(f"slide salvo: {SAIDA}")
   print(f"  vencedora: {o} η={lr:g} {ag}  val={vp:.4f}±{vps:.4f}")
   print(f"  baseline:  {bo} η={blr:g} {bag}  val={bp:.4f}±{bps:.4f}")
   print(f"  ganho={ganho:+.4f}  supera variação entre sementes: {sep}")

if __name__ == "__main__":
   main()