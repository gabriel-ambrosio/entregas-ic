using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using BTD_Mod_Helper;
using BTD_Mod_Helper.Api.ModOptions;
using BTD_Mod_Helper.Extensions;
using Il2CppAssets.Scripts.Unity.UI_New.InGame;
using MelonLoader;
using UnityEngine;

[assembly: MelonInfo(typeof(RoundExporterMod.RoundExporter), "Exportador", "2.0.0", "Gabriel Valentin")]
[assembly: MelonGame("Ninja Kiwi", "BloonsTD6")]

namespace RoundExporterMod;

/// <summary>
/// Exporta o estado do jogo para o pipeline de RL e congela o jogo enquanto o agente decide.
///
/// Arquivos em UserData:
///   estado.json        estado mais recente (sempre o mesmo nome; "seq" cresce a cada exportação)
///   mapa.json          mapa da partida atual
///   partida_*.json     histórico completo (só se "Salvar histórico" estiver ligado)
///
/// Quando exporta:
///   - início da partida (e depois de cada reinício)      -> fase "decisao", inicio = true
///   - fim de cada rodada                                  -> fase "decisao"
///   - pedido do mod Player (arquivo exportar.sinal)       -> fase "decisao"
///   - vida chegou a zero                                  -> fase "fim", resultado "derrota"
///
/// Sinais (arquivos vazios criados pelo mod Player):
///   retomar.sinal      descongela o jogo
///   exportar.sinal     exporta o estado de novo e congela
///   reiniciado.sinal   a partida foi reiniciada: exporta o estado inicial quando estiver pronto
/// </summary>
public class RoundExporter : BloonsTD6Mod
{
    // ---------- opções (menu de mods do jogo) ----------
    public static readonly ModSettingHotkey TeclaRetomar = new(KeyCode.F2)
    {
        displayName = "Retomar jogo"
    };

    public static readonly ModSettingInt VelocidadeTreino = new(0)
    {
        displayName = "Velocidade do jogo no treino (0 = não mexer)",
        min = 0,
        max = 20,
        slider = true
    };

    public static readonly ModSettingBool SalvarHistorico = new(false)
    {
        displayName = "Salvar histórico (partida_*.json)"
    };

    // ---------- configuração ----------
    private const float EsperaInicio = 1.0f; // segundos reais entre o (re)início e a 1ª exportação

    // ---------- estado ----------
    private string _arquivoHistorico = "";
    private readonly List<object> _historico = new();

    private int _proximaRodada = 1;        // rodada que vai começar a seguir
    private float _exportarInicioEm = -1f; // Time.unscaledTime a partir do qual exporta o estado inicial
    private bool _partidaAtiva;            // já exportou o estado inicial desta partida
    private bool _fimExportado;            // derrota já exportada
    private int _episodio;
    private long _seq;

    private bool _pausadoPeloMod;
    private float _velocidadeAnterior = 1f;

    private static string Pasta => MelonLoader.Utils.MelonEnvironment.UserDataDirectory;
    private static string ArquivoEstado => Path.Combine(Pasta, "estado.json");
    private static string ArquivoMapa => Path.Combine(Pasta, "mapa.json");
    private static string SinalRetomar => Path.Combine(Pasta, "retomar.sinal");
    private static string SinalExportar => Path.Combine(Pasta, "exportar.sinal");
    private static string SinalReiniciado => Path.Combine(Pasta, "reiniciado.sinal");

    // Ponto [x,y] escrito em uma linha só, mesmo com o JSON indentado
    [JsonConverter(typeof(PontoConverter))]
    public class Ponto
    {
        public float X, Y;
        public Ponto(float x, float y) { X = x; Y = y; }
    }

    public class PontoConverter : JsonConverter<Ponto>
    {
        public override Ponto Read(ref Utf8JsonReader reader, Type tipo, JsonSerializerOptions opcoes) =>
            throw new NotSupportedException();

        public override void Write(Utf8JsonWriter writer, Ponto p, JsonSerializerOptions opcoes) =>
            writer.WriteRawValue(FormattableString.Invariant($"[{p.X},{p.Y}]"));
    }

    // ---------- ciclo de vida ----------
    public override void OnMatchStart()
    {
        PrepararPartida("início da partida");
    }

    public override void OnUpdate()
    {
        var jogo = InGame.instance;
        if (jogo?.bridge != null)
        {
            // O mod Player reiniciou a partida
            if (Consumir(SinalReiniciado))
                PrepararPartida("reinício");

            if (_exportarInicioEm >= 0f)
            {
                // Estado inicial: espera um pouco e só exporta quando a partida nova está pronta
                if (Time.unscaledTime >= _exportarInicioEm && PartidaPronta())
                {
                    _exportarInicioEm = -1f;
                    _partidaAtiva = true;
                    _episodio++;
                    _proximaRodada = jogo.bridge.GetCurrentRound() + 1;
                    LiberarPassosDeSimulacao();
                    ExportarMapa();
                    Exportar("decisao", null, inicio: true, pausar: true);
                }
            }
            else if (_partidaAtiva && !_fimExportado)
            {
                if (Vida() <= 0)
                {
                    // Derrota: o jogo não chama o fim de rodada, então avisamos por aqui
                    _fimExportado = true;
                    Exportar("fim", "derrota", inicio: false, pausar: false);
                }
                else if (Consumir(SinalExportar))
                {
                    Exportar("decisao", null, inicio: false, pausar: true);
                }
            }
        }

        ControlarTempo();
    }

    // Reforça a pausa/velocidade depois que o jogo atualizou o timeScale no frame
    public override void OnLateUpdate()
    {
        ControlarTempo();
    }

    // Fim da rodada N: exporta o estado atual + bloons da rodada N+1 e pausa
    public override void OnRoundEnd()
    {
        if (!_partidaAtiva || _fimExportado) return;
        _proximaRodada++;
        Exportar("decisao", null, inicio: false, pausar: true);
    }

    public override void OnMatchEnd()
    {
        _partidaAtiva = false;
        _pausadoPeloMod = false;
        Time.timeScale = 1f; // nunca deixa o jogo congelado ao sair da partida
    }

    private void PrepararPartida(string motivo)
    {
        _partidaAtiva = false;
        _fimExportado = false;
        _pausadoPeloMod = false;
        _historico.Clear();
        _exportarInicioEm = Time.unscaledTime + EsperaInicio;
        Apagar(SinalRetomar);
        Apagar(SinalExportar);

        string carimbo = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        _arquivoHistorico = Path.Combine(Pasta, $"partida_{carimbo}.json");
        MelonLogger.Msg($"Exportador: {motivo}.");
    }

    // A partida nova já tem vida e modelo carregados? (logo após um reinício ainda não tem)
    private static bool PartidaPronta()
    {
        try
        {
            var jogo = InGame.instance;
            return jogo.GetGameModel() != null && jogo.GetHealth() > 0;
        }
        catch { return false; }
    }

    private static double Vida()
    {
        try { return InGame.instance.GetHealth(); }
        catch { return 1; } // sem leitura: não declara derrota
    }

    // ---------- arquivos ----------
    private static void Apagar(string arquivo)
    {
        try { if (File.Exists(arquivo)) File.Delete(arquivo); } catch { }
    }

    // true se o sinal existia (e já foi apagado)
    private static bool Consumir(string arquivo)
    {
        if (!File.Exists(arquivo)) return false;
        Apagar(arquivo);
        return true;
    }

    // Escreve em arquivo temporário e troca de uma vez: o Python nunca lê um JSON pela metade
    private static void EscreverAtomico(string destino, string texto)
    {
        string tmp = destino + ".tmp";
        File.WriteAllText(tmp, texto);
        for (int tentativa = 0; ; tentativa++)
        {
            try { File.Move(tmp, destino, true); return; }
            catch (IOException) when (tentativa < 20) { System.Threading.Thread.Sleep(5); }
            catch (UnauthorizedAccessException) when (tentativa < 20) { System.Threading.Thread.Sleep(5); }
        }
    }

    // ---------- pausa e velocidade ----------
    private void ControlarTempo()
    {
        if (_pausadoPeloMod)
        {
            if (TeclaRetomar.JustPressed() || File.Exists(SinalRetomar))
            {
                Apagar(SinalRetomar);
                _pausadoPeloMod = false;
                Time.timeScale = _velocidadeAnterior;
                MelonLogger.Msg("Jogo retomado.");
            }
            else
            {
                // O jogo reescreve o timeScale sozinho; por isso forçamos 0 todo frame
                Time.timeScale = 0f;
            }
            return;
        }

        // Fora da pausa: acelera o jogo durante o treino (o jogo também reescreve esse valor todo frame)
        int velocidade = VelocidadeTreino;
        if (velocidade > 0 && _partidaAtiva && InGame.instance?.bridge != null)
            Time.timeScale = velocidade;
    }

    // O jogo limita quantos passos de simulação roda por frame (o que segura a velocidade em ~3x).
    // Tentamos aumentar esse limite por reflexão; se o campo não existir nesta versão, só avisamos.
    private static PropertyInfo? _propLimitePassos;
    private static bool _limiteProcurado;

    private static void LiberarPassosDeSimulacao()
    {
        int velocidade = VelocidadeTreino;
        if (velocidade <= 3) return;

        try
        {
            if (!_limiteProcurado)
            {
                _limiteProcurado = true;
                Type[] tipos;
                try { tipos = typeof(InGame).Assembly.GetTypes(); }
                catch (ReflectionTypeLoadException e) { tipos = e.Types.Where(t => t != null).ToArray()!; }

                var tipo = tipos.FirstOrDefault(t => t.Name == "TimeManager");
                _propLimitePassos = tipo?.GetProperty("maxSimulationStepsPerUpdate",
                    BindingFlags.Public | BindingFlags.Static);
                if (_propLimitePassos == null)
                    MelonLogger.Warning("TimeManager.maxSimulationStepsPerUpdate não encontrado: " +
                                        "a velocidade pode ficar limitada a ~3x.");
            }

            if (_propLimitePassos != null && _propLimitePassos.CanWrite)
            {
                _propLimitePassos.SetValue(null, Convert.ChangeType(velocidade, _propLimitePassos.PropertyType));
                MelonLogger.Msg($"Limite de passos de simulação por frame: {velocidade}.");
            }
        }
        catch (Exception e)
        {
            MelonLogger.Warning($"Não foi possível liberar a velocidade: {e.GetType().Name}: {e.Message}");
        }
    }

    // =====================================================================
    //  MAPA (a cada início de partida) -> mapa.json
    //  Usa o modelo do mapa do jogo (MapModel): pistas (paths) e áreas (areas).
    //  Se algum nome de campo não compilar na sua versão, o ajuste é só aqui.
    // =====================================================================
    private void ExportarMapa()
    {
        try
        {
            var mapa = InGame.instance.GetGameModel().map;

            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;
            void Atualiza(float x, float y)
            {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }

            // Pistas: onde começam, onde terminam e o traçado completo
            var pistas = new List<object>();
            foreach (var caminho in mapa.paths)
            {
                var pontos = new List<Ponto>();
                foreach (var p in caminho.points)
                {
                    pontos.Add(new Ponto(p.point.x, p.point.y));
                    Atualiza(p.point.x, p.point.y);
                }
                if (pontos.Count > 0)
                    pistas.Add(new { inicio = pontos.First(), fim = pontos.Last(), pontos });
            }

            // Áreas: água, e tudo o que não permite colocar torre (pista, obstáculos, etc.)
            var invalidas = new List<object>();
            var agua = new List<object>();
            var tiposVistos = new HashSet<string>();
            foreach (var area in mapa.areas)
            {
                string tipo = area.type.ToString();
                tiposVistos.Add(tipo);

                var poligono = new List<Ponto>();
                foreach (var v in area.polygon.points)
                {
                    poligono.Add(new Ponto(v.x, v.y));
                    Atualiza(v.x, v.y);
                }
                if (poligono.Count == 0) continue;

                string t = tipo.ToLowerInvariant();
                if (t == "water") agua.Add(new { poligono });
                else if (t != "land") invalidas.Add(new { tipo, poligono });
            }

            var saida = new
            {
                limites = new { x_min = minX, x_max = maxX, y_min = minY, y_max = maxY },
                pistas,
                invalidas,
                agua
            };

            EscreverAtomico(ArquivoMapa,
                JsonSerializer.Serialize(saida, new JsonSerializerOptions { WriteIndented = true }));

            MelonLogger.Msg($"Mapa exportado: {ArquivoMapa} | pistas: {pistas.Count}, " +
                            $"áreas inválidas: {invalidas.Count}, água: {agua.Count} | tipos de área: {string.Join(", ", tiposVistos)}");
        }
        catch (Exception e)
        {
            MelonLogger.Warning($"Falha ao exportar o mapa: {e.GetType().Name}: {e.Message}");
        }
    }

    // =====================================================================
    //  ESTADO -> estado.json
    // =====================================================================
    private void Exportar(string fase, string? resultado, bool inicio, bool pausar)
    {
        var jogo = InGame.instance;
        if (jogo?.bridge == null) return;

        Apagar(SinalRetomar); // um sinal velho não pode despausar o jogo logo de cara

        try
        {
            var gameModel = jogo.GetGameModel();
            var rounds = gameModel.roundSet.rounds;

            // Bloons programados para a PRÓXIMA rodada (vazio se o jogo acabou)
            var grupos = _proximaRodada - 1 < rounds.Length
                ? rounds[_proximaRodada - 1].groups
                    .Select(g => new
                    {
                        bloon = g.bloon,
                        quantidade = g.count,
                        inicio = g.start,
                        fim = g.end
                    })
                    .Cast<object>()
                    .ToList()
                : new List<object>();

            // Custo do próximo upgrade em cada caminho (null = não pode melhorar esse caminho).
            // Mesmas regras que o mod Player usa para validar.
            double?[] CustosDeUpgrade(Il2CppAssets.Scripts.Simulation.Towers.Tower t)
            {
                var custos = new double?[3];
                try
                {
                    int[] tiers = { t.towerModel.tiers[0], t.towerModel.tiers[1], t.towerModel.tiers[2] };
                    foreach (var u in t.towerModel.upgrades)
                    {
                        var upgrade = gameModel.GetUpgrade(u.upgrade);
                        if (upgrade == null) continue;
                        int caminho = upgrade.path;
                        if (caminho < 0 || caminho > 2) continue;

                        int novo = tiers[caminho] + 1;
                        if (novo > 5) continue;
                        if (tiers[caminho] == 0 && tiers.Count(x => x > 0) >= 2) continue;
                        if (novo > 2 && tiers.Where((x, i) => i != caminho).Any(x => x > 2)) continue;

                        custos[caminho] = upgrade.cost;
                    }
                }
                catch { }
                return custos;
            }

            var torres = jogo.GetTowers()
                .Select(t => new
                {
                    tipo = t.towerModel.baseId,
                    caminhos = new[] { t.towerModel.tiers[0], t.towerModel.tiers[1], t.towerModel.tiers[2] },
                    x = t.Position.X,
                    y = t.Position.Y,
                    raio = t.towerModel.radius,
                    upgrades = CustosDeUpgrade(t)
                })
                .ToList();

            // Loja: torres disponíveis nesta partida, com custo e tamanho
            var loja = new List<object>();
            try
            {
                foreach (var d in gameModel.towerSet)
                {
                    var m = gameModel.GetTowerFromId(d.towerId);
                    if (m == null) continue;

                    var areas = new List<string>();
                    for (int i = 0; i < m.areaTypes.Length; i++)
                        areas.Add(m.areaTypes[i].ToString());

                    loja.Add(new
                    {
                        tipo = d.towerId,
                        custo = m.cost,
                        raio = m.radius,
                        alcance = m.range,
                        areas
                    });
                }
            }
            catch (Exception e)
            {
                MelonLogger.Warning($"Falha ao exportar a loja: {e.GetType().Name}: {e.Message}");
            }

            // seq sempre cresce, mesmo depois de fechar e abrir o jogo
            _seq = Math.Max(_seq + 1, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());

            var estado = new
            {
                seq = _seq,
                episodio = _episodio,
                fase,
                resultado,
                inicio,
                proxima_rodada = _proximaRodada,
                vida = jogo.GetHealth(),
                dinheiro = jogo.GetCash(),
                bloons_proxima_rodada = grupos,
                torres,
                loja
            };

            EscreverAtomico(ArquivoEstado,
                JsonSerializer.Serialize(estado, new JsonSerializerOptions { WriteIndented = true }));

            if (SalvarHistorico)
            {
                _historico.Add(estado);
                File.WriteAllText(_arquivoHistorico,
                    JsonSerializer.Serialize(_historico, new JsonSerializerOptions { WriteIndented = true }));
            }
        }
        catch (Exception e)
        {
            MelonLogger.Warning($"Falha ao exportar o estado: {e.GetType().Name}: {e.Message}");
        }

        // Pausa DEPOIS de salvar
        if (pausar)
        {
            if (!_pausadoPeloMod)
                _velocidadeAnterior = Time.timeScale > 0 ? Time.timeScale : 1f;
            _pausadoPeloMod = true;
            Time.timeScale = 0f;
        }

        MelonLogger.Msg($"Estado exportado (fase: {fase}, próxima rodada: {_proximaRodada}, seq: {_seq}).");
    }
}
