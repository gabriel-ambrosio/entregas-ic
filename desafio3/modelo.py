"""
modelo.py — o preditor de rodada.
Arquitetura Deep Sets: as 12 torres passam por uma MESMA camada (peso
compartilhado), são somadas, e só então se juntam às features da rodada. A
soma é o que torna a saída invariante à ordem das torres, que é a propriedade
correta — um layout não muda de valor porque as torres foram listadas em
outra ordem.
   torres (B, 12, 81) --[Linear compartilhada]--> (B, 12, 32)
                      --[soma sobre as torres]--> (B, 32)
   rodada (B, 46) ----------[concatena]---------> (B, 78)
                      --[32 -> 16 -> 1]---------> logit
A saída é o LOGIT, não a probabilidade: o treino usa `BCEWithLogitsLoss`, que
funde o log-sum-exp e é estável. Use `probabilidade()` para a leitura do AG.
Inicialização: Glorot uniforme com viés zero nas camadas densas, e
normal(0, 0.05) sem viés na camada de agregação.
"""
from __future__ import annotations
import torch
import torch.nn as nn
# camadas registradas no log por-camada (primeira, meio, última), como pede o
# enunciado do desafio
CAMADAS_MONITORADAS = ("torre", "mlp1", "saida")

class PreditorDeRodada(nn.Module):
   def __init__(self, dim_outros: int = 46, dim_torre: int = 81,
                n_torres: int = 12, oculta: int = 32):
       super().__init__()
       self.n_torres = n_torres
       self.torre = nn.Linear(dim_torre, oculta)           # compartilhada
       self.perm = nn.Linear(oculta, oculta, bias=False)   # agrega a soma
       self.mlp1 = nn.Linear(dim_outros + oculta, oculta)
       self.mlp2 = nn.Linear(oculta, 16)
       self.saida = nn.Linear(16, 1)
       self.reiniciar_parametros()
   def reiniciar_parametros(self) -> None:
       for m in (self.torre, self.mlp1, self.mlp2, self.saida):
           nn.init.xavier_uniform_(m.weight)
           nn.init.zeros_(m.bias)
       nn.init.normal_(self.perm.weight, mean=0.0, std=0.05)  # random_normal
   def forward(self, outros: torch.Tensor, torres: torch.Tensor) -> torch.Tensor:
       # torres: (B, 12, 81) -> peso compartilhado -> (B, 12, 32)
       h = torch.relu(self.torre(torres))
       # soma sobre as torres: invariância a permutação
       h = torch.relu(self.perm(h.sum(dim=1)))              # (B, 32)
       x = torch.cat([outros, h], dim=1)                    # (B, 46+32)
       x = torch.relu(self.mlp1(x))
       x = torch.relu(self.mlp2(x))
       return self.saida(x).squeeze(-1)                     # logit (B,)
   @torch.no_grad()
   def probabilidade(self, outros: torch.Tensor, torres: torch.Tensor) -> torch.Tensor:
       """Probabilidade de passar a rodada sem perder vida — o que o AG consome."""
       return torch.sigmoid(self(outros, torres))

def parametros_muon_e_adamw(modelo: nn.Module):
   """Separa os parâmetros conforme a regra do desafio.
   Muon vale só para matrizes 2-D de camadas ocultas. Vieses, a camada de
   saída (`saida`) e qualquer embedding ficam no AdamW.
   """
   muon, adamw = [], []
   for nome, p in modelo.named_parameters():
       if not p.requires_grad:
           continue
       eh_oculta_2d = p.ndim == 2 and "saida" not in nome and "emb" not in nome
       (muon if eh_oculta_2d else adamw).append(p)
   return muon, adamw

if __name__ == "__main__":
   m = PreditorDeRodada()
   n = sum(p.numel() for p in m.parameters())
   print(m)
   print(f"\nparâmetros: {n}")
   mu, ad = parametros_muon_e_adamw(m)
   print(f"Muon (2-D ocultas): {sum(p.numel() for p in mu)} em {len(mu)} tensores")
   print(f"AdamW (resto):      {sum(p.numel() for p in ad)} em {len(ad)} tensores")
   o, t = torch.randn(4, 46), torch.randn(4, 12, 81)
   print(f"\nsaída (logits): {tuple(m(o, t).shape)}")
   # invariância a permutação: embaralhar as torres não muda a predição
   perm = torch.randperm(12)
   d = (m(o, t) - m(o, t[:, perm])).abs().max().item()
   print(f"invariância a permutação, erro máximo: {d:.2e}")