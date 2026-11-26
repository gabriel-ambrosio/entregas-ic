using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using BTD_Mod_Helper;
using BTD_Mod_Helper.Api.ModOptions;
using BTD_Mod_Helper.Extensions;
using Il2CppAssets.Scripts;
using Il2CppAssets.Scripts.Simulation.Towers;
using Il2CppAssets.Scripts.Unity.UI_New.InGame;
using MelonLoader;
using UnityEngine;

[assembly: MelonInfo(typeof(RoundPlayerMod.RoundPlayer), "Player", "2.1.0", "Gabriel Valentin")]
[assembly: MelonGame("Ninja Kiwi", "BloonsTD6")]

namespace RoundPlayerMod;

/// <summary>
/// Lê UserData/acoes.json, valida cada ação contra o estado REAL do jogo e executa
/// uma por vez. O resultado de cada lote vai para ultimo_resultado.json.
///
/// Formato do acoes.json ("id" é opcional e volta igual no resultado):
/// {
///   "id": 17,
///   "acoes": [
///     { "acao": "colocar_torre",  "tipo": "DartMonkey", "x": 10.5, "y": -3.0 },
///     { "acao": "melhorar_torre", "torre": { "x": 10.5, "y": -3.0 }, "caminho": 0 },
///     { "acao": "vender_torre",   "torre": { "x": 10.5, "y": -3.0 } },
///     { "acao": "iniciar_rodada" },
///     { "acao": "reiniciar" }
///   ]
/// }
/// (também aceita um array puro: [ {...}, {...} ])
///
/// Formato do ultimo_resultado.json:
/// { "id": 17, "resultados": [ { "indice": 1, "acao": "...", "aceita": true, "motivo": "ok" } ] }
///
/// Modo automático (opção do mod, ligado por padrão): o arquivo é lido assim que aparece,
/// sem precisar apertar a tecla. É o modo usado pelo pipeline de RL.
/// </summary>
public class RoundPlayer : BloonsTD6Mod
{
    // ---------- configuração ----------
    private const float IntervaloEntrePassos = 0.05f;  // segundos reais entre passos do lote
    private const float EsperaAposRetomar = 0.15f;     // espera o Exportador descongelar
    private const double ToleranciaPosicao = 1.0;      // x,y da torre no JSON vs. posição real
    private const float PrazoJogo = 2f;                // segundos de JOGO (param quando pausado) para a simulação responder
    private const float PrazoMinimoReal = 1f;          // ...e pelo menos isto em segundos reais (jogo acelerado)
    private const float PrazoReal = 15f;               // limite em segundos reais, para nunca travar o lote
    private const float PrazoReinicio = 8f;            // segundos reais para a partida voltar ao início, por tentativa

    // Tecla que lê o acoes.json e executa as ações (configurável nas opções do mod)
    public static readonly ModSettingHotkey TeclaLer = new(KeyCode.Return)
    {
        displayName = "Ler e executar acoes.json"
    };

    public static readonly ModSettingBool ModoAutomatico = new(true)
    {
        displayName = "Modo automático (lê acoes.json assim que aparece)"
    };

    private static bool Automatico => ModoAutomatico;

    private static string Pasta => MelonLoader.Utils.MelonEnvironment.UserDataDirectory;
    private static string ArquivoAcoes => Path.Combine(Pasta, "acoes.json");
    private static string ArquivoProcessado => Path.Combine(Pasta, "acoes.processado.json");
    private static string ArquivoResultado => Path.Combine(Pasta, "ultimo_resultado.json");
    private static string SinalRetomar => Path.Combine(Pasta, "retomar.sinal");
    private static string SinalExportar => Path.Combine(Pasta, "exportar.sinal");
    private static string SinalReiniciado => Path.Combine(Pasta, "reiniciado.sinal");

    // ---------- estado ----------
    private readonly Queue<JsonElement> _fila = new();
    private readonly List<Resultado> _resultados = new();
    private readonly List<Pendente> _pendentes = new();
    private float _proximoPasso;
    private int _indice;
    private bool _rodadaAtiva;

    private bool _loteAberto;       // há um lote lido que ainda não teve o resultado salvo
    private long? _idLote;          // "id" do acoes.json, devolvido no resultado
    private bool _rodadaIniciada;   // o lote iniciou uma rodada (o Exportador exporta no fim dela)
    private bool _reiniciou;        // o lote reiniciou a partida (confirmado)

    private Reinicio? _reinicio;            // reinício em andamento, esperando a confirmação
    private int _rodadaInicial = -1;        // GetCurrentRound() no começo da partida
    private bool _capturarRodadaInicial;

    // Reinício pedido: tenta um método de cada vez e só aceita quando a partida voltou mesmo ao início
    private class Reinicio
    {
        public Resultado resultado = null!;
        public int proximoMetodo;           // índice em MetodosDeReinicio
        public int rodadaAntes;
        public float prazo;
        public string historico = "";
    }

    private class Resultado
    {
        public int indice { get; set; }
        public string acao { get; set; } = "";
        public bool aceita { get; set; }
        public string motivo { get; set; } = "";
    }

    // Ação enviada ao jogo que ainda espera a confirmação da simulação
    private class Pendente
    {
        public Resultado resultado = null!;
        public Func<bool> confirmou = () => false;  // o estado do jogo já mostra o efeito da ação?
        public string motivoRecusa = "";            // texto usado quando o callback do jogo responde false
        public string callback = "(não chamado)";   // preenchido pelo callback do jogo
        public float prazoJogo;                     // Time.time limite (não anda com o jogo pausado)
        public float prazoMinimoReal;               // antes disto nunca desiste
        public float prazoReal;                     // Time.unscaledTime limite
    }

    // ---------- ciclo de vida ----------
    public override void OnMatchStart()
    {
        _rodadaAtiva = false;
        // Reinício em andamento: o lote continua aberto até a confirmação (não limpa nada)
        if (_reinicio != null) return;

        _capturarRodadaInicial = true;
        _fila.Clear();
        _resultados.Clear();
        _pendentes.Clear();
        _indice = 0;
        _rodadaAtiva = false;
        _proximoPasso = 0;
        _loteAberto = false;
        _idLote = null;
        _rodadaIniciada = false;
        _reiniciou = false;

        // Limpa restos de partidas anteriores. No modo automático o acoes.json fica:
        // quem controla a fila é o programa que escreve o arquivo.
        try { if (File.Exists(SinalRetomar)) File.Delete(SinalRetomar); } catch { }
        if (!Automatico)
            try { if (File.Exists(ArquivoAcoes)) File.Delete(ArquivoAcoes); } catch { }
    }

    public override void OnRoundStart() => _rodadaAtiva = true;
    public override void OnRoundEnd() => _rodadaAtiva = false;

    public override void OnUpdate()
    {
        if (InGame.instance?.bridge == null) return;

        if (_capturarRodadaInicial)
        {
            _capturarRodadaInicial = false;
            try { _rodadaInicial = InGame.instance.bridge.GetCurrentRound(); } catch { }
        }

        // Tecla: precisa ser conferida todo frame
        if (TeclaLer.JustPressed())
        {
            if (_loteAberto)
                MelonLogger.Warning("Ainda executando o lote anterior: aguarde terminar.");
            else
                LerArquivoDeAcoes(avisarSeFaltar: true);
            return;
        }

        if (Time.unscaledTime < _proximoPasso) return; // unscaled: funciona mesmo com o jogo pausado
        _proximoPasso = Time.unscaledTime + IntervaloEntrePassos;

        if (_reinicio != null)
        {
            AcompanharReinicio();
            return;
        }

        if (!_loteAberto)
        {
            if (Automatico && File.Exists(ArquivoAcoes))
                LerArquivoDeAcoes(avisarSeFaltar: false);
            return;
        }

        // Avança o lote: só executa a próxima ação depois que a anterior foi confirmada ou recusada
        VerificarPendentes();
        if (_pendentes.Count > 0) return;

        if (_fila.Count > 0) ExecutarProxima();
        else FecharLote();
    }

    // ---------- leitura do JSON ----------
    private void LerArquivoDeAcoes(bool avisarSeFaltar)
    {
        if (!File.Exists(ArquivoAcoes))
        {
            if (avisarSeFaltar) MelonLogger.Warning($"Arquivo não encontrado: {ArquivoAcoes}");
            return;
        }

        string texto;
        try { texto = File.ReadAllText(ArquivoAcoes); }
        catch (IOException)
        {
            // Em uso por outro programa: tenta de novo no próximo passo (ou na próxima tecla)
            if (avisarSeFaltar) MelonLogger.Warning("acoes.json está em uso por outro programa: aperte a tecla de novo.");
            return;
        }

        _fila.Clear();
        _resultados.Clear();
        _pendentes.Clear();
        _indice = 0;
        _idLote = null;
        _rodadaIniciada = false;
        _reiniciou = false;

        JsonDocument doc;
        try { doc = JsonDocument.Parse(texto); }
        catch (JsonException)
        {
            MelonLogger.Warning("acoes.json inválido: arquivo ignorado.");
            RejeitarArquivo("JSON inválido");
            return;
        }

        using (doc)
        {
            var raiz = doc.RootElement;
            if (raiz.ValueKind == JsonValueKind.Object)
            {
                if (raiz.TryGetProperty("id", out var id) && id.ValueKind == JsonValueKind.Number
                    && id.TryGetInt64(out var n))
                    _idLote = n;
                if (raiz.TryGetProperty("acoes", out var lista))
                    raiz = lista;
            }

            if (raiz.ValueKind != JsonValueKind.Array)
            {
                MelonLogger.Warning("acoes.json sem lista de ações: arquivo ignorado.");
                RejeitarArquivo("arquivo sem lista de ações");
                return;
            }

            foreach (var a in raiz.EnumerateArray())
                _fila.Enqueue(a.Clone());
        }

        if (_fila.Count == 0)
        {
            RejeitarArquivo("lista de ações vazia");
            return;
        }

        Consumir();
        _loteAberto = true;
        SinalizarRetomada();
        _proximoPasso = Time.unscaledTime + EsperaAposRetomar;
        MelonLogger.Msg($"{_fila.Count} ação(ões) recebida(s)" + (_idLote != null ? $" (id {_idLote})." : "."));
    }

    // Arquivo inteiro recusado (inválido/vazio): registra o motivo; o lote fecha no próximo passo
    private void RejeitarArquivo(string motivo)
    {
        Consumir();
        _fila.Clear();
        _resultados.Add(new Resultado { indice = 0, acao = "(arquivo)", aceita = false, motivo = motivo });
        _loteAberto = true;
        SinalizarRetomada();
        _proximoPasso = Time.unscaledTime + EsperaAposRetomar;
    }

    private static void Consumir()
    {
        try { File.Move(ArquivoAcoes, ArquivoProcessado, true); }
        catch { try { File.Delete(ArquivoAcoes); } catch { } }
    }

    private static void CriarSinal(string arquivo)
    {
        try { File.WriteAllText(arquivo, "1"); } catch { }
    }

    // Pede ao mod Exportador para descongelar o jogo
    private static void SinalizarRetomada() => CriarSinal(SinalRetomar);

    // ---------- execução + validação ----------
    private void ExecutarProxima()
    {
        var a = _fila.Dequeue();
        string nome = Texto(a, "acao") ?? "(sem nome)";
        var r = new Resultado { indice = ++_indice, acao = nome };

        if (nome == "reiniciar")
        {
            IniciarReinicio(r);
            return;
        }

        bool aguardaConfirmacao = false;
        string? erro;
        try
        {
            switch (nome)
            {
                case "colocar_torre":
                    erro = ColocarTorre(a, r);
                    aguardaConfirmacao = erro == null;
                    break;
                case "melhorar_torre":
                    erro = MelhorarTorre(a, r);
                    aguardaConfirmacao = erro == null;
                    break;
                case "vender_torre":
                    erro = VenderTorre(a, r);
                    aguardaConfirmacao = erro == null;
                    break;
                case "iniciar_rodada":
                    erro = IniciarRodada();
                    if (erro == null) _rodadaIniciada = true;
                    break;
                default:
                    erro = "ação desconhecida";
                    break;
            }
        }
        catch (Exception e)
        {
            erro = "exceção: " + e.Message;
        }

        if (aguardaConfirmacao)
        {
            // VerificarPendentes decide se foi aceita ou não
            r.aceita = false;
            r.motivo = "aguardando confirmação da simulação";
        }
        else
        {
            r.aceita = erro == null;
            r.motivo = erro ?? "ok";
        }
        _resultados.Add(r);

        if (erro != null) MelonLogger.Warning($"Ação #{r.indice} '{nome}' ignorada: {erro}");
        else MelonLogger.Msg($"Ação #{r.indice} '{nome}' enviada ao jogo.");
    }

    // ---------- reinício da partida ----------
    // O reinício só conta como feito quando o jogo mostra que voltou ao início
    // (rodada inicial, nenhuma torre, vida > 0). Se um método não funcionar, tenta o próximo.
    private void IniciarReinicio(Resultado r)
    {
        _fila.Clear();
        _pendentes.Clear();
        _reinicio = new Reinicio { resultado = r, rodadaAntes = RodadaAtual() };
        TentarProximoMetodo();
    }

    private void TentarProximoMetodo()
    {
        var re = _reinicio!;
        while (re.proximoMetodo < MetodosDeReinicio.Length)
        {
            var (nome, chamar) = MetodosDeReinicio[re.proximoMetodo++];
            string? erro;
            try { erro = chamar(); }
            catch (Exception e) { erro = "exceção: " + (e.InnerException?.Message ?? e.Message); }

            if (erro == null)
            {
                re.historico += $"{nome}: chamado, mas a partida não voltou ao início; ";
                re.prazo = Time.unscaledTime + PrazoReinicio;
                return; // AcompanharReinicio confere o efeito
            }
            re.historico += $"{nome}: {erro}; ";
            MelonLogger.Warning($"Reinício por {nome} falhou: {erro}");
        }
        ConcluirReinicio(false);
    }

    private void AcompanharReinicio()
    {
        var re = _reinicio!;
        if (PartidaVoltouAoInicio(re))
            ConcluirReinicio(true);
        else if (Time.unscaledTime > re.prazo)
            TentarProximoMetodo();
    }

    private bool PartidaVoltouAoInicio(Reinicio re)
    {
        try
        {
            var jogo = InGame.instance;
            int rodada = jogo.bridge.GetCurrentRound();
            bool rodadaVoltou = _rodadaInicial >= 0
                ? rodada <= _rodadaInicial
                : rodada == 0 || rodada < re.rodadaAntes;
            return rodadaVoltou && !jogo.GetTowers().Any() && jogo.GetHealth() > 0;
        }
        catch { return false; }
    }

    private static int RodadaAtual()
    {
        try { return InGame.instance.bridge.GetCurrentRound(); }
        catch { return 0; }
    }

    private void ConcluirReinicio(bool ok)
    {
        var re = _reinicio!;
        _reinicio = null;
        re.resultado.aceita = ok;
        re.resultado.motivo = ok ? "ok" : "a partida não reiniciou. Tentativas: " + re.historico;
        _resultados.Add(re.resultado);
        _reiniciou = ok;
        _rodadaAtiva = false;
        _loteAberto = true;

        if (ok) MelonLogger.Msg("Partida reiniciada (confirmado).");
        else MelonLogger.Warning($"Ação #{re.resultado.indice} 'reiniciar' falhou: {re.resultado.motivo}");
        FecharLote();
    }

    // Cada método retorna null se a ação foi válida (e enviada) ou o motivo da rejeição.

    private string? ColocarTorre(JsonElement a, Resultado r)
    {
        var jogo = InGame.instance;
        string? tipo = Texto(a, "tipo");
        if (string.IsNullOrWhiteSpace(tipo)) return "campo 'tipo' obrigatório";
        if (!Numero(a, "x", out var x) || !Numero(a, "y", out var y)) return "campos 'x' e 'y' obrigatórios";

        var gameModel = jogo.GetGameModel();

        // 1) o tipo existe e está disponível nesta partida?
        if (!gameModel.towerSet.Any(d => d.towerId == tipo)) return $"torre '{tipo}' não disponível nesta partida";
        var modelo = gameModel.GetTowerFromId(tipo);
        if (modelo == null) return $"torre '{tipo}' não encontrada";

        // 2) dinheiro suficiente?
        if (jogo.GetCash() < modelo.cost) return $"dinheiro insuficiente ({jogo.GetCash():0} < {modelo.cost:0})";

        // 3) não sobrepõe outra torre?
        foreach (var t in jogo.GetTowers())
        {
            double d = Distancia(t.Position.X, t.Position.Y, x, y);
            if (d < modelo.radius + t.towerModel.radius) return "posição ocupada por outra torre";
        }

        // 4) pista/água/obstáculo só dá para saber pela simulação: confirmamos depois
        int antes = Contar(tipo!, x, y);
        var p = NovoPendente(r, () => Contar(tipo!, x, y) > antes,
            "o jogo recusou a posição (pista, água, obstáculo ou fora do mapa)");
        CriarTorreNoJogo(modelo, (float)x, (float)y, p);
        return null;
    }

    private string? MelhorarTorre(JsonElement a, Resultado r)
    {
        var jogo = InGame.instance;
        var torre = AcharTorre(a, out var erro, out var tx, out var ty);
        if (torre == null) return erro;

        if (!Numero(a, "caminho", out var c) || c != Math.Floor(c) || c < 0 || c > 2)
            return "'caminho' deve ser 0, 1 ou 2";
        int caminho = (int)c;

        int[] tiers = { torre.towerModel.tiers[0], torre.towerModel.tiers[1], torre.towerModel.tiers[2] };
        int novo = tiers[caminho] + 1;

        // Regras de upgrade do BTD6
        if (novo > 5) return "caminho já está no nível máximo";
        if (tiers[caminho] == 0 && tiers.Count(t => t > 0) >= 2) return "só 2 caminhos podem ser usados";
        if (novo > 2 && tiers.Where((t, i) => i != caminho).Any(t => t > 2))
            return "só um caminho pode passar do nível 2";

        // O upgrade existe para essa torre e o dinheiro cobre?
        var gameModel = jogo.GetGameModel();
        var opcao = torre.towerModel.upgrades
            .FirstOrDefault(u => gameModel.GetUpgrade(u.upgrade).path == caminho);
        if (opcao == null) return "upgrade indisponível para esse caminho";

        var upgrade = gameModel.GetUpgrade(opcao.upgrade);
        if (jogo.GetCash() < upgrade.cost) return $"dinheiro insuficiente ({jogo.GetCash():0} < {upgrade.cost:0})";

        // Confirmado quando o nível daquele caminho subir na torre daquela posição
        int nivelAntes = tiers[caminho];
        var p = NovoPendente(r, () =>
        {
            var atual = TorreEm(tx, ty);
            return atual != null && atual.towerModel.tiers[caminho] > nivelAntes;
        }, "o jogo recusou o upgrade");
        MelhorarTorreNoJogo(torre, caminho, p);
        return null;
    }

    private string? VenderTorre(JsonElement a, Resultado r)
    {
        var torre = AcharTorre(a, out var erro, out var tx, out var ty);
        if (torre == null) return erro;

        // Confirmado quando não houver mais torre naquela posição
        NovoPendente(r, () => TorreEm(tx, ty) == null, "o jogo recusou a venda");
        VenderTorreNoJogo(torre);
        return null;
    }

    private string? IniciarRodada()
    {
        if (_rodadaAtiva) return "já existe uma rodada em andamento";
        IniciarRodadaNoJogo();
        return null;
    }

    private Pendente NovoPendente(Resultado r, Func<bool> confirmou, string motivoRecusa)
    {
        var p = new Pendente
        {
            resultado = r,
            confirmou = confirmou,
            motivoRecusa = motivoRecusa,
            prazoJogo = Time.time + PrazoJogo,
            prazoMinimoReal = Time.unscaledTime + PrazoMinimoReal,
            prazoReal = Time.unscaledTime + PrazoReal
        };
        _pendentes.Add(p);
        return p;
    }

    // Uma ação pendente só sai da lista quando:
    //  - o callback do jogo respondeu (true/false), ou
    //  - o efeito apareceu no jogo, ou
    //  - o prazo estourou (contado em tempo de jogo, então pausa não gera falso negativo).
    private void VerificarPendentes()
    {
        for (int i = _pendentes.Count - 1; i >= 0; i--)
        {
            var p = _pendentes[i];
            bool confirmou;
            try { confirmou = p.confirmou(); }
            catch { confirmou = false; }

            if (confirmou || p.callback == "true")
            {
                p.resultado.aceita = true;
                p.resultado.motivo = "ok";
                MelonLogger.Msg($"Ação #{p.resultado.indice} '{p.resultado.acao}' confirmada.");
            }
            else if (p.callback == "false")
            {
                p.resultado.aceita = false;
                p.resultado.motivo = p.motivoRecusa;
                MelonLogger.Warning($"Ação #{p.resultado.indice} '{p.resultado.acao}' recusada pelo jogo.");
            }
            else if ((Time.time > p.prazoJogo && Time.unscaledTime > p.prazoMinimoReal)
                     || Time.unscaledTime > p.prazoReal)
            {
                p.resultado.aceita = false;
                p.resultado.motivo = $"a simulação não respondeu (callback: {p.callback}, timeScale: {Time.timeScale})";
                MelonLogger.Warning($"Ação #{p.resultado.indice} '{p.resultado.acao}' sem resposta | " +
                                    $"callback: {p.callback} | timeScale: {Time.timeScale}");
            }
            else
            {
                continue; // ainda aguardando: confere de novo no próximo passo
            }

            _pendentes.RemoveAt(i);
        }
    }

    // Salva o resultado do lote e avisa o Exportador do que vem a seguir
    private void FecharLote()
    {
        try
        {
            string texto = JsonSerializer.Serialize(new { id = _idLote, resultados = _resultados },
                new JsonSerializerOptions { WriteIndented = true });
            string tmp = ArquivoResultado + ".tmp";
            File.WriteAllText(tmp, texto);
            File.Move(tmp, ArquivoResultado, true);
        }
        catch (Exception e) { MelonLogger.Warning("Falha ao salvar resultado: " + e.Message); }

        if (_reiniciou)
        {
            // O Exportador exporta o estado inicial quando a partida nova estiver pronta
            CriarSinal(SinalReiniciado);
        }
        else if (!_rodadaIniciada && Automatico)
        {
            // Nenhuma rodada começou: pede o estado atualizado (e o jogo congela de novo).
            // Se uma rodada começou, o Exportador exporta sozinho no fim dela.
            CriarSinal(SinalExportar);
        }
        else if (!_rodadaIniciada)
        {
            // Modo manual: só garante que o jogo está despausado
            SinalizarRetomada();
        }

        _resultados.Clear();
        _loteAberto = false;
        _idLote = null;
        _rodadaIniciada = false;
        _reiniciou = false;
    }

    // =====================================================================
    //  CAMADA DO JOGO: únicas chamadas que dependem da versão do BTD6 /
    //  Mod Helper. Se algo não compilar, o ajuste é só aqui.
    // =====================================================================

    // Id de input do jogador local, fornecido pelo próprio jogo (o mesmo usado ao
    // colocar torre com o mouse). Não dá para adivinhar: não é um índice 0..4.
    private static int IdJogador => InGame.instance.bridge.GetInputId();

    // Callback do jogo ligado a uma ação pendente
    private static Il2CppSystem.Action<bool> CallbackDe(Pendente p) => new Action<bool>(ok =>
    {
        p.callback = ok ? "true" : "false";
        MelonLogger.Msg($"Callback do jogo (ação #{p.resultado.indice}): {ok}");
    });

    private static void CriarTorreNoJogo(Il2CppAssets.Scripts.Models.Towers.TowerModel modelo, float x, float y,
        Pendente p)
    {
        int id = IdJogador;
        MelonLogger.Msg($"CreateTowerAt: {modelo.baseId} em ({x}, {y}) | inputId: {id} | timeScale: {Time.timeScale}");
        InGame.instance.bridge.CreateTowerAt(id, new Vector2(x, y), modelo, IdInvalido(), false, CallbackDe(p), true);
    }

    // ObjectId "vazio": procura um membro estático do tipo (InvalidId, Invalid...) e,
    // se não achar, usa default. Também lista os membros estáticos no console para diagnóstico.
    private static object? _idInvalidoObj;
    private static bool _idResolvido;

    private static ObjectId IdInvalido()
    {
        if (!_idResolvido)
        {
            _idResolvido = true;
            var tipo = typeof(ObjectId);
            const BindingFlags flags = BindingFlags.Public | BindingFlags.Static;
            foreach (var nome in new[] { "InvalidId", "Invalid", "invalid", "NullId", "None", "Empty" })
            {
                object? v = null;
                try { v = tipo.GetProperty(nome, flags)?.GetValue(null) ?? tipo.GetField(nome, flags)?.GetValue(null); }
                catch { }
                if (v is ObjectId id)
                {
                    _idInvalidoObj = id;
                    MelonLogger.Msg($"ObjectId inválido encontrado em '{nome}'.");
                    break;
                }
            }
            MelonLogger.Msg("ObjectId, membros estáticos: " +
                            string.Join(", ", tipo.GetMembers(flags).Select(m => m.Name).Distinct()));
        }
        return _idInvalidoObj is ObjectId o ? o : default(ObjectId);
    }

    private static void MelhorarTorreNoJogo(Tower torre, int caminho, Pendente p)
    {
        InGame.instance.bridge.UpgradeTower(IdJogador, torre.Id, caminho, 0.0, CallbackDe(p));
    }

    private static void VenderTorreNoJogo(Tower torre)
    {
        InGame.instance.bridge.SellTower(IdJogador, torre.Id);
    }

    private static void IniciarRodadaNoJogo()
    {
        InGame.instance.bridge.StartRound();
    }

    // Métodos de reinício, na ordem em que são tentados. Cada um devolve null se foi chamado
    // (o efeito é conferido depois) ou o motivo de não ter sido possível chamar.
    //  1) InGame.Restart: o mesmo do botão "Reiniciar" do jogo (também limpa a tela de fim).
    //  2) bridge.Restart: reinicia só a simulação.
    private static readonly (string nome, Func<string?> chamar)[] MetodosDeReinicio =
    {
        ("InGame.Restart", () => ChamarRestart(InGame.instance, typeof(InGame))),
        ("bridge.Restart", () => ChamarRestart(InGame.instance.bridge,
            typeof(Il2CppAssets.Scripts.Unity.Bridge.UnityToSimulation))),
    };

    // Chama "Restart" por reflexão, para funcionar com ou sem parâmetros (a assinatura muda entre versões).
    // Parâmetro bool com "save" no nome (removeSave) vai como true: false faz o jogo voltar do save
    // em vez de recomeçar do zero.
    private static string? ChamarRestart(object alvo, Type tipo)
    {
        const BindingFlags flags = BindingFlags.Public | BindingFlags.Instance;
        var candidatos = tipo.GetMethods(flags).Where(m => m.Name.Contains("Restart")).ToList();
        MelonLogger.Msg($"Métodos com 'Restart' em {tipo.Name}: " + (candidatos.Count == 0 ? "(nenhum)" :
            string.Join(" | ", candidatos.Select(m => m.Name + "(" +
                string.Join(", ", m.GetParameters().Select(prm => $"{prm.ParameterType.Name} {prm.Name}")) + ")"))));

        var metodo = candidatos.Where(m => m.Name == "Restart")
            .OrderBy(m => m.GetParameters().Length)
            .FirstOrDefault();
        if (metodo == null) return "método não existe nesta versão do jogo";

        var argumentos = metodo.GetParameters().Select(prm =>
        {
            if (prm.ParameterType == typeof(bool) && (prm.Name ?? "").ToLowerInvariant().Contains("save"))
                return (object?)true;
            if (prm.HasDefaultValue) return prm.DefaultValue;
            return prm.ParameterType.IsValueType ? Activator.CreateInstance(prm.ParameterType) : null;
        }).ToArray();

        MelonLogger.Msg($"Chamando {tipo.Name}.Restart(" + string.Join(", ", argumentos.Select(a => a?.ToString() ?? "null")) + ")");
        metodo.Invoke(alvo, argumentos);
        return null;
    }

    // ---------- utilitários ----------
    private static string? Texto(JsonElement a, string chave) =>
        a.ValueKind == JsonValueKind.Object && a.TryGetProperty(chave, out var v) && v.ValueKind == JsonValueKind.String
            ? v.GetString() : null;

    private static bool Numero(JsonElement a, string chave, out double n)
    {
        n = 0;
        return a.ValueKind == JsonValueKind.Object && a.TryGetProperty(chave, out var v)
               && v.ValueKind == JsonValueKind.Number && v.TryGetDouble(out n);
    }

    private static double Distancia(double x1, double y1, double x2, double y2) =>
        Math.Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

    private static int Contar(string tipo, double x, double y) =>
        InGame.instance.GetTowers()
            .Count(t => t.towerModel.baseId == tipo && Distancia(t.Position.X, t.Position.Y, x, y) <= ToleranciaPosicao);

    // A torre mais próxima de (x, y), dentro da tolerância; null se não houver
    private static Tower? TorreEm(double x, double y)
    {
        Tower? melhor = null;
        double menor = double.MaxValue;
        foreach (var tw in InGame.instance.GetTowers())
        {
            double d = Distancia(tw.Position.X, tw.Position.Y, x, y);
            if (d < menor) { menor = d; melhor = tw; }
        }
        return melhor != null && menor <= ToleranciaPosicao ? melhor : null;
    }

    // Localiza a torre pelo campo "torre": { "x": .., "y": .. }
    private static Tower? AcharTorre(JsonElement a, out string? erro, out double x, out double y)
    {
        erro = null;
        x = 0;
        y = 0;
        if (a.ValueKind != JsonValueKind.Object
            || !a.TryGetProperty("torre", out var t) || t.ValueKind != JsonValueKind.Object
            || !Numero(t, "x", out x) || !Numero(t, "y", out y))
        {
            erro = "campo 'torre' com 'x' e 'y' obrigatório";
            return null;
        }

        var torre = TorreEm(x, y);
        if (torre == null) erro = "nenhuma torre nessa posição";
        return torre;
    }
}
