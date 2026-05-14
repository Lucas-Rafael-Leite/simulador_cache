# Importações necessárias para funcionalidades diversas
import numpy as np  # Biblioteca para computação numérica
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # Biblioteca para visualização gráfica
import random  # Biblioteca padrão para geração de números aleatórios
from collections import deque  # Deque (fila dupla) para simulações de políticas de cache
import io, base64  # Para manipulação de fluxos de bytes e codificação
from PIL import Image  # Para manipulação de imagens
from io import BytesIO  # Para fluxo de bytes em memória
import csv  # Para leitura/escrita de arquivos CSV
import dearpygui.dearpygui as dpg  # Biblioteca GUI para interface gráfica
import time, sys, os  # Utilitários do sistema e tempo
from datetime import datetime  # Para manipulação de datas e horários


# ------------------------------------------------------------------------------
# Redirecionador de saída padrão (print) para uma tag do DearPyGUI
class DPGRedirector:
    def __init__(self, tag):
        self.tag = tag
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        current_text = dpg.get_value(self.tag)
        dpg.set_value(self.tag, current_text + text)

    def flush(self):
        pass  # Método necessário para compatibilidade com sys.stdout


# Lista global para armazenar tags de séries de plotagem (caso visualizações sejam usadas)
plot_series_tags = []


# ------------------------------------------------------------------------------
# Verifica se um número é potência de 2 (útil para parâmetros de cache válidos)
def is_power_of_two(x):
    return (x != 0) and ((x & (x - 1)) == 0)


# ------------------------------------------------------------------------------
# Gera um padrão de acessos à memória simulando comportamentos realistas
def gerar_padrao_realista(acessos, memory_size, regioes_quentes, prob_temporal, prob_espacial, prob_quente,
                          bloco_tamanho):
    padrao = []
    endereco_anterior = random.randint(0, memory_size - 1)
    for _ in range(acessos):
        r = random.random()
        if r < prob_temporal:
            # Repetição do endereço anterior (localidade temporal)
            endereco = endereco_anterior
        elif r < prob_temporal + prob_espacial:
            # Acesso a um endereço próximo ao anterior (localidade espacial)
            deslocamento = random.randint(-16, 16)
            endereco = max(0, min(memory_size - 1, endereco_anterior + deslocamento))
        elif r < prob_temporal + prob_espacial + prob_quente:
            # Acesso a uma região quente
            base = random.choice(regioes_quentes)
            deslocamento = random.randint(0, 3)
            endereco = min(memory_size - 1, base + deslocamento)
        else:
            # Acesso completamente aleatório
            endereco = random.randint(0, memory_size - 1)

        padrao.append(endereco)
        endereco_anterior = endereco

    return padrao


# ------------------------------------------------------------------------------
# Simulação de cache com política de substituição FIFO
def simular_cache_FIFO(padrao_acesso, cache_lines, associatividade, bloco_tamanho):
    num_conjuntos = cache_lines // associatividade
    cache = [[] for _ in range(num_conjuntos)]  # Lista de conjuntos de cache
    hits, misses = 0, 0
    conjunto_log, hit_log = [], []

    for endereco in padrao_acesso:
        bloco = endereco // bloco_tamanho
        conjunto = bloco % num_conjuntos
        conjunto_atual = cache[conjunto]

        if bloco in conjunto_atual:
            hits += 1
            hit_log.append(1)
        else:
            misses += 1
            hit_log.append(0)
            if len(conjunto_atual) < associatividade:
                conjunto_atual.append(bloco)
            else:
                conjunto_atual.pop(0)  # Remove o mais antigo
                conjunto_atual.append(bloco)

        conjunto_log.append(conjunto)

    return conjunto_log, hit_log


# ------------------------------------------------------------------------------
# Simulação de cache com política de substituição LRU (Least Recently Used)
def simular_cache_LRU(padrao_acesso, cache_lines, associatividade, bloco_tamanho):
    num_conjuntos = cache_lines // associatividade
    cache = [deque() for _ in range(num_conjuntos)]
    hits, misses = 0, 0
    conjunto_log, hit_log = [], []

    for endereco in padrao_acesso:
        bloco = endereco // bloco_tamanho
        conjunto = bloco % num_conjuntos
        conjunto_atual = cache[conjunto]

        if bloco in conjunto_atual:
            hits += 1
            hit_log.append(1)
            conjunto_atual.remove(bloco)  # Remove e reinsere no fim (mais recente)
            conjunto_atual.append(bloco)
        else:
            misses += 1
            hit_log.append(0)
            if len(conjunto_atual) >= associatividade:
                conjunto_atual.popleft()  # Remove o menos recentemente usado
            conjunto_atual.append(bloco)

        conjunto_log.append(conjunto)

    return conjunto_log, hit_log


# ------------------------------------------------------------------------------
# Simulação de cache com política de substituição LFU (Least Frequently Used)
def simular_cache_LFU(padrao_acesso, cache_lines, associatividade, bloco_tamanho):
    num_conjuntos = cache_lines // associatividade
    cache = [{} for _ in range(num_conjuntos)]  # Dict: bloco -> frequência
    hits, misses = 0, 0
    conjunto_log, hit_log = [], []

    for endereco in padrao_acesso:
        bloco = endereco // bloco_tamanho
        conjunto = bloco % num_conjuntos
        conjunto_atual = cache[conjunto]

        if bloco in conjunto_atual:
            hits += 1
            hit_log.append(1)
            conjunto_atual[bloco] += 1
        else:
            misses += 1
            hit_log.append(0)
            if len(conjunto_atual) < associatividade:
                conjunto_atual[bloco] = 1
            else:
                bloco_remover = min(conjunto_atual, key=conjunto_atual.get)
                del conjunto_atual[bloco_remover]
                conjunto_atual[bloco] = 1

        conjunto_log.append(conjunto)

    return conjunto_log, hit_log


# ------------------------------------------------------------------------------
# Simulação de cache com política de substituição aleatória (RANDOM)
def simular_cache_RANDOM(padrao_acesso, cache_lines, associatividade, bloco_tamanho):
    num_conjuntos = cache_lines // associatividade
    cache = [[] for _ in range(num_conjuntos)]
    hits, misses = 0, 0
    conjunto_log, hit_log = [], []

    for endereco in padrao_acesso:
        bloco = endereco // bloco_tamanho
        conjunto = bloco % num_conjuntos
        conjunto_atual = cache[conjunto]

        if bloco in conjunto_atual:
            hits += 1
            hit_log.append(1)
        else:
            misses += 1
            hit_log.append(0)
            if len(conjunto_atual) < associatividade:
                conjunto_atual.append(bloco)
            else:
                idx_remover = random.randint(0, associatividade - 1)
                conjunto_atual[idx_remover] = bloco

        conjunto_log.append(conjunto)

    return conjunto_log, hit_log


# ------------------------------------------------------------------------------
# Variável global que armazena o algoritmo de substituição selecionado
algoritmo_escolhido = "FIFO"


# Callback para seleção do algoritmo via GUI
def selecionar_algoritmo(sender, app_data):
    global algoritmo_escolhido
    algoritmo_escolhido = app_data


# ------------------------------------------------------------------------------
# Geração de mapa de calor (heatmap) dos acessos por bloco ao longo do tempo (implementação antiga)
def mapa_temporal_blocos(padrao_acesso, memory_size, bloco_tamanho, resolucao_temporal=100):
    num_janelas = len(padrao_acesso) // resolucao_temporal
    num_blocos = memory_size // bloco_tamanho
    heatmap = np.zeros((num_blocos, num_janelas), dtype=int)

    for i, endereco in enumerate(padrao_acesso):
        tempo = i // resolucao_temporal
        bloco = endereco // bloco_tamanho
        if bloco < num_blocos and tempo < num_janelas:
            heatmap[bloco][tempo] += 1

    plt.figure(figsize=(10, 4))
    plt.imshow(heatmap, cmap='hot', aspect='auto', origin='lower')
    plt.colorbar(label="Número de acessos por bloco")
    plt.title("Evolução dos Acessos à Memória por Bloco")
    plt.xlabel(f"Grupos de {resolucao_temporal} Acessos")
    plt.ylabel("Bloco de Memória")
    plt.show()


# ------------------------------------------------------------------------------
# Execução de várias simulações (Monte Carlo) para avaliar desempenho do algoritmo escolhido
def simulacao_monte_carlo(n_simulacoes, acessos, memory_size, cache_lines, associatividade, regioes_quentes, probs,
                          bloco_tamanho):
    global ultimo_padrao_acesso
    taxas_acerto = []
    hits_totais = []
    misses_totais = []

    for i in range(n_simulacoes):
        padrao = gerar_padrao_realista(acessos, memory_size, regioes_quentes, *probs, bloco_tamanho)
        ultimo_padrao_acesso = padrao

        # Seleciona e executa o algoritmo de substituição
        if algoritmo_escolhido == 'FIFO':
            conjunto_log, hit_log = simular_cache_FIFO(padrao, cache_lines, associatividade, bloco_tamanho)
        elif algoritmo_escolhido == 'LRU':
            conjunto_log, hit_log = simular_cache_LRU(padrao, cache_lines, associatividade, bloco_tamanho)
        elif algoritmo_escolhido == 'LFU':
            conjunto_log, hit_log = simular_cache_LFU(padrao, cache_lines, associatividade, bloco_tamanho)
        elif algoritmo_escolhido == 'Random':
            conjunto_log, hit_log = simular_cache_RANDOM(padrao, cache_lines, associatividade, bloco_tamanho)
        else:
            raise ValueError(f"Algoritmo de substituição desconhecido: {algoritmo_escolhido}")

        hits = sum(hit_log)
        misses = len(hit_log) - hits
        taxa_acerto = hits / len(hit_log)

        # Armazena resultados desta simulação
        taxas_acerto.append(taxa_acerto)
        hits_totais.append(hits)
        misses_totais.append(misses)

    # Exibe estatísticas gerais
    print(f"--- Resultados: {acessos} Acessos - Bloco de {bloco_tamanho} ---\n")
    print(f"Média da Taxa de Acerto: {np.mean(taxas_acerto):.2f}")
    print(f"Desvio Padrão da Taxa de Acerto: {np.std(taxas_acerto):.2f}")
    print(f"Máximo: {max(taxas_acerto):.2f}, Mínimo: {min(taxas_acerto):.2f}")

    # Parte do plot do mapa de acessos. COmentada porque NÃO FUNCIONA!!!!!
    # if i == 0:
    # mapa_temporal_blocos(padrao, memory_size, bloco_tamanho, resolucao_temporal=100)

    print(f"--- Resultados: {acessos} Acessos - Bloco de {bloco_tamanho} ---\n")
    print(f"Média da Taxa de Acerto: {np.mean(taxas_acerto):.4f}")
    print(
        f"Desvio padrão da Taxa de Acerto: {np.std(taxas_acerto):.4f}\n")  # print(f"Total médio de acessos: {acessos}")

    # Parte comentada ANTIGA para debug!!!!!
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # print(f"Média de Cache Hits: {np.mean(hits_totais):.2f}")
    # print(f"Variância Taxa de Acerto: {np.var(hits_totais):.2f}")
    # print(f"Média Taxa de Erro: {np.mean(misses_totais):.2f}")
    # print(f"Variância Taxa de Erro: {np.var(misses_totais):.2f}")
    # print(f"Média da Taxa de Acerto: {np.mean(taxas_acerto):.4f}")
    # print(f"Desvio padrão da Taxa de Acerto: {np.std(taxas_acerto):.4f}\n")

    # Atualiza o conteúdo da caixa de texto 'Resumo' na interface
    # ATENÇÃO: Após a alteração de desvio da saída padrão (DPGRedirector), tanto faz usar a
    # caixa de texto 'Resumo' ou a função 'print'

    # Texto = f"Total de acessos: {acessos}\n"
    # Texto += f"Média de Cache Hits: {np.mean(hits_totais):.2f}\n"
    # Texto += f"Variância de Cache Hits: {np.var(hits_totais):.2f}\n"
    # Texto += f"Média de Cache Misses: {np.mean(misses_totais):.2f}\n"
    # Texto += f"Variância de Cache Misses: {np.var(misses_totais):.2f}\n"
    # Texto += f"Média da Taxa de Acerto: {np.mean(taxas_acerto):.4f}\n"
    # Texto += f"Desvio padrão da Taxa de Acerto: {np.std(taxas_acerto):.4f}\n"

    # Atualiza dados da simulação na Caixa de tetxto Resumo
    # dpg.set_value("Resumo", Texto)
    return np.mean(taxas_acerto)


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

import dearpygui.dearpygui as dpg

from collections import deque

resultados = []
ultimo_padrao_acesso = None
ultimo_bloco_tamanho = 1

def rodar_simulacao_callback():
    start_time = time.time()
    print(f"            ---   Algoritmo: {algoritmo_escolhido} ---\n")
    global resultados
    dpg.set_value("mensagem_erro", "")  # Limpa mensagem antiga

    try:
        # Leitura dos valores
        memory_size = dpg.get_value("memory_size")
        acessos = dpg.get_value("acessos")
        tamanho_cache_bytes = dpg.get_value("tamanho_cache")
        associatividade = dpg.get_value("associatividade")
        n_simulacoes = dpg.get_value("n_simulacoes")
        prob_temporal = dpg.get_value("prob_temporal")
        prob_espacial = dpg.get_value("prob_espacial")
        prob_quente = dpg.get_value("prob_quente")
        blocos = dpg.get_value("blocos")
        blocos = [int(b.strip()) for b in blocos.split(",")]

        regioes_quentes = [
            64, 1024, 8192, 32768, 131072, 262144, 524288, 786432, 983040
        ]

        # --- VERIFICAÇÕES ---

        if not (2 ** 20 <= memory_size <= 2 ** 30) or (memory_size & (memory_size - 1)) != 0:
            dpg.set_value("mensagem_erro", "Erro: Memory Size deve ser potência de 2 entre 2^20 e 2^30.")
            return

        if not is_power_of_two(associatividade):
            dpg.set_value("mensagem_erro", "Erro: Associatividade deve ser potência de 2 maior que zero.")
            return

        if (tamanho_cache_bytes & (tamanho_cache_bytes - 1)) != 0 or tamanho_cache_bytes >= memory_size:
            dpg.set_value("mensagem_erro", "Erro: Tamanho da Cache deve ser potência de 2 e menor que Memory Size.")
            return

        for bloco in blocos:
            if bloco <= 0 or bloco >= memory_size or not is_power_of_two(bloco):
                dpg.set_value("mensagem_erro",
                              f"Tamanho do Bloco deve ser potência de 2 e menor que Memory Size. Valor fornecido: {bloco}")
                return
            cache_lines = tamanho_cache_bytes // bloco
            num_conjuntos = cache_lines // associatividade
            if num_conjuntos < 1:
                dpg.set_value("mensagem_erro", f"Erro: Associatividade {associatividade} inválida para bloco {bloco}.")
                return

        if not (0 <= prob_temporal <= 1) or not (0 <= prob_espacial <= 1) or not (0 <= prob_quente <= 1):
            dpg.set_value("mensagem_erro", "Erro: Probabilidades devem ser entre 0 e 1.")
            return

        # --- FIM VERIFICAÇÕES ---

        resultados.clear()
        contador_barra = 0
        progresso = 0.01  # Mostra um andamento mínimo na barra de progresso para indicar que a nova simulação iniciou
        dpg.set_value("barra", progresso)
        dpg.set_value("texto", "Simulação Iniciada")
        for bt in blocos:
            contador_barra += 1
            progresso = contador_barra / len(blocos)
            cache_lines = tamanho_cache_bytes // bt
            taxas_acerto = simulacao_monte_carlo(
                n_simulacoes,
                acessos,
                memory_size,
                cache_lines,
                associatividade,
                regioes_quentes,
                (prob_temporal, prob_espacial, prob_quente),
                bt
            )
            global ultimo_bloco_tamanho
            ultimo_bloco_tamanho = bt  # Salva o último tamanho de bloco processado
            dpg.set_value("barra", progresso)
            dpg.set_value("texto", f"{int(progresso * 100)}% concluído")
            resultados.append((bt, taxas_acerto))
            dpg.split_frame()  # Permite que a interface atualize
        # ao final, exibir resultados
        atualizar_plot()

        if resultados:
            tamanhos, taxas = zip(*resultados)
            texto = f"Tamanhos_de_bloco = [{', '.join(str(int(t)) for t in tamanhos)}];\n"
            texto += f"Taxa_media_de_acerto = [{', '.join(f'{float(t):.6f}' for t in taxas)}];"

            # Atualiza o conteúdo da caixa de texto na interface
            dpg.set_value("resultados_box", texto)

            # Opcional: também salva os resultados em um arquivo CSV

            # Define o nome da subpasta
            subpasta = "Resultados da Simulacao"
            # Cria a subpasta se não existir
            os.makedirs(subpasta, exist_ok=True)
            # Captura data e hora atual
            agora = datetime.now()
            # Formata para uma string segura para nome de arquivo
            timestamp = agora.strftime("%Y-%m-%d_%H-%M-%S")
            # Caminho completo do arquivo
            nome_arquivo = f"Resultados da Simulacao - {timestamp}.csv"
            caminho_arquivo = os.path.join(subpasta, nome_arquivo)
            with open(caminho_arquivo, "w") as f:
                f.write(
                    f"- Algoritmo = {algoritmo_escolhido}\n- Associatividade = {associatividade}\n- Acessos = {acessos}\n- Cache = {tamanho_cache_bytes}\n- P_tem = {prob_temporal}\n- P_espa = {prob_espacial}\n- P_reg_quente = {prob_quente}\n")
                f.write("\nTamanho_Bloco,Taxa_Acerto\n")
                for bloco, taxa in resultados:
                    f.write(f"{bloco},{taxa:.6f}\n")

    except Exception as e:
        dpg.set_value("mensagem_erro", f"Erro inesperado: {str(e)}")
    # Mede tempo de simulação:
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo de execução: {elapsed_time:.2f} segundos\n")
    print("          ------------++-------------     \n")


# Nova implementação do heatmap
def mostrar_heatmap_callback(sender, app_data):
    global ultimo_padrao_acesso
    dpg.set_value("mensagem_erro", "")

    # 1. Trava de segurança: Garante que só desenhe SE houve simulação
    if not ultimo_padrao_acesso:
        dpg.set_value("mensagem_erro", "Erro: Execute a simulação ('Simular') antes de gerar o Heatmap!")
        return

    try:
        # 2. Configura os dados para o gráfico
        resolucao_temporal = 100
        x = [i // resolucao_temporal for i in range(len(ultimo_padrao_acesso))]
        y = [addr // ultimo_bloco_tamanho for addr in ultimo_padrao_acesso]

        # Usando hist2d, agrupamos os dados e resolvemos o problema do gráfico "preto" invisível
        fig, ax = plt.subplots(figsize=(8, 4))
        num_janelas = max(1, len(x) // resolucao_temporal)

        # Divide a tela em 100 divisões (bins) visíveis
        h = ax.hist2d(x, y, bins=[num_janelas, 100], cmap='hot', cmin=1)

        # Fundo preto para regiões sem acesso para contrastar
        ax.set_facecolor('black')
        if h[3] is not None:
            fig.colorbar(h[3], ax=ax, label="Número de Acessos")

        ax.set_title(f"Acessos por Bloco (Tamanho: {ultimo_bloco_tamanho})")
        ax.set_xlabel(f"Grupos de {resolucao_temporal} Acessos")
        ax.set_ylabel("Índice do Bloco de Memória")

        # 3. Salvar figura direto na memória (Buffer) em vez de criar arquivo PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)  # Libera o gráfico do Matplotlib
        buf.seek(0)

        # 4. Converter a imagem em Bytes para Textura (RGBA) legível pelo DearPyGui
        img = Image.open(buf).convert('RGBA')
        img_width, img_height = img.size
        data = np.array(img, dtype=np.float32) / 255.0
        data_flat = data.flatten().tolist()

        # 5. PRIMEIRO: Deletar a janela antiga se ela existir (isso solta a textura)
        if dpg.does_alias_exist("janela_heatmap"):
            dpg.delete_item("janela_heatmap")

        # 6. SEGUNDO: Atualizar/Recriar a textura
        if not dpg.does_alias_exist("registro_texturas"):
            dpg.add_texture_registry(tag="registro_texturas")

        # Deleta a textura antiga e limpa o nome dela do registro
        if dpg.does_alias_exist("textura_heatmap"):
            dpg.delete_item("textura_heatmap")
            if dpg.does_alias_exist("textura_heatmap"):  # Garantia extra de limpeza
                dpg.remove_alias("textura_heatmap")

        # Usamos dynamic_texture que é otimizado para gráficos que se atualizam
        dpg.add_dynamic_texture(width=img_width, height=img_height, default_value=data_flat, tag="textura_heatmap",
                                parent="registro_texturas")

        # 7. TERCEIRO: Criar a nova janela com a textura recriada
        with dpg.window(label="Heatmap de Acessos", tag="janela_heatmap", width=img_width + 20, height=img_height + 40,
                        pos=[250, 150], no_collapse=True):
            dpg.add_image("textura_heatmap")

        dpg.set_value("mensagem_erro", "Heatmap atualizado com sucesso!")

    except Exception as e:
        dpg.set_value("mensagem_erro", f"Erro ao gerar Heatmap: {str(e)}")


def mostrar_especificacoes_callback():
    try:
        # Coleta de dados da interface
        M = dpg.get_value("memory_size")  # Tamanho da Memória (Bytes)
        C = dpg.get_value("tamanho_cache")  # Tamanho da Cache (Bytes)
        A = dpg.get_value("associatividade")  # Associatividade

        # Como o campo 'blocos' é uma lista, será utilizado o primeiro valor para o cálculo de hardware
        blocos_input = dpg.get_value("blocos")
        B = int(blocos_input.split(",")[0].strip())

        # 1. Cálculos de Endereçamento
        # Endereço total em bits = log2(M)
        bits_endereco = int(math.log2(M))

        # Bits de Offset (palavra) = log2(B)
        bits_offset = int(math.log2(B))

        # Número de Linhas (Slots) = C / B
        num_linhas = C // B

        # Número de Conjuntos (Sets) = Linhas / Associatividade
        num_conjuntos = num_linhas // A

        # Bits de Índice (conjunto/linha) = log2(Conjuntos)
        bits_indice = int(math.log2(num_conjuntos))

        # Bits de Tag (rótulo) = Total - Índice - Offset
        bits_tag = bits_endereco - bits_indice - bits_offset

        # 2. Tamanho Físico (Hardware)
        # Cálculo das linhas de comparação em hardware (equivalente ao número de bits enviado ao comparador)
        linhas_hardware = bits_tag + (bits_tag * A)



        # Gerar Texto de Saída
        specs_texto = (
            f"--- CONFIGURAÇÃO DE HARDWARE ---\n"
            f"Endereçamento: {bits_endereco} bits\n"
            f"Linhas de Cache: {num_linhas}\n"
            f"Estrutura: {num_conjuntos} conjuntos de {A} linhas\n\n"
            f"--- DIVISÃO DO ENDEREÇO ---\n"
            f"Tag: {bits_tag} bits\n"
            f"Índice: {bits_indice} bits\n"
            f"Offset: {bits_offset} bits\n\n"
            f"--- IMPLEMENTAÇÃO FÍSICA ---\n"
            f"Linhas de Hardware: {linhas_hardware} linhas de comparação\n\n"
        )

        # Criar ou atualizar janela de especificações
        if dpg.does_alias_exist("janela_specs"):
            dpg.delete_item("janela_specs")

        with dpg.window(label="Especificações de Engenharia", tag="janela_specs", width=400, height=450,
                        pos=[500, 100]):
            dpg.add_text(specs_texto)
            dpg.add_button(label="Fechar", callback=lambda: dpg.delete_item("janela_specs"))

    except Exception as e:
        dpg.set_value("mensagem_erro", f"Erro no cálculo: {str(e)}")

# Limpa plots e elementos graficos (barra e caixas de texto)
def limpar_plots():
    global plot_series_tags
    plot_series_tags = []
    dpg.delete_item("y_axis", children_only=True)
    dpg.set_value("barra", 0.0)
    dpg.set_value("texto", f"0% concluído")
    dpg.set_value("Resumo", "\n")
    dpg.set_value("resultados_box", "\n")
    dpg.set_value("mensagem_erro", " ")


def limpar_ultimo_plot():
    global plot_series_tags
    # Limpa dados da simulação na Caixa de texto Resumo
    # dpg.set_value("Resumo", "\n")
    dpg.set_value("mensagem_erro", " ")
    dpg.set_value("barra", 0.0)
    dpg.set_value("texto", f"0% concluído")
    dpg.set_value("resultados_box", "\n")
    if plot_series_tags:
        ultimo_tag = plot_series_tags.pop()
        dpg.delete_item(ultimo_tag)
        print(f"Apagado: {ultimo_tag}")


import math


def atualizar_plot():
    global plot_series_tags
    if not dpg.does_item_exist("y_axis"):
        print("Erro: 'y_axis' não existe.")
        return

    # dpg.delete_item("plot_series", children_only=True)
    tamanhos, taxas = zip(*resultados)

    # Calcula log2 dos tamanhos
    tamanhos_log2 = [math.log2(tam) for tam in tamanhos]

    # Verifica consistência
    if len(tamanhos_log2) != len(taxas):
        print("Erro: tamanhos_log2 e taxas têm tamanhos diferentes")
        return

    # dpg.set_axis_limits("x_axis", min(tamanhos_log2), max(tamanhos_log2))
    # dpg.set_axis_limits("y_axis", min(taxas), max(taxas))
    plot_series = f"plot_{len(plot_series_tags)}"
    print(f"\nPlot Atual: {plot_series}\n")
    plot_series_tags.append(plot_series)
    dpg.add_line_series(
        tamanhos_log2,
        taxas,
        label=f"{algoritmo_escolhido}",
        parent="y_axis",
        tag=plot_series,
        show=True
    )
    dpg.fit_axis_data("x_axis")
    dpg.fit_axis_data("y_axis")


# Interface
dpg.create_context()
sys.stdout = DPGRedirector("Resumo")  # Redireciona todos os prints

# Pega a resolução da tela
# viewport_width, viewport_height = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()


with dpg.window(label="Simulação de Cache", width=1400, height=900):
    dpg.add_input_int(label="Memory Size", default_value=1048576, tag="memory_size", width=200)
    dpg.add_input_int(label="Acessos", default_value=10000, tag="acessos", width=200)
    dpg.add_input_int(label="Tamanho Cache (Bytes)", default_value=8192, tag="tamanho_cache", width=200)
    dpg.add_input_int(label="Associatividade", default_value=16, tag="associatividade", width=200)
    dpg.add_input_int(label="N Simulações", default_value=10, tag="n_simulacoes", width=200)

    dpg.add_separator()
    dpg.add_input_float(label="Probabilidade Temporal", default_value=0.2, tag="prob_temporal", width=200)
    dpg.add_input_float(label="Probabilidade Espacial", default_value=0.2, tag="prob_espacial", width=200)
    dpg.add_input_float(label="Probabilidade Região Quente", default_value=0.4, tag="prob_quente", width=200)

    dpg.add_separator()
    dpg.add_input_text(label="Tamanhos de Bloco", default_value="2,4,8,16,32,64,128,256,512", tag="blocos", width=400)

    with dpg.group(horizontal=True):  # Inicia um grupo horizontal
        dpg.add_button(label="Simular", callback=rodar_simulacao_callback)
        dpg.add_button(label="Mostrar Especificações", callback=mostrar_especificacoes_callback)
        dpg.add_button(label="Limpar Último", callback=limpar_ultimo_plot)
        dpg.add_button(label="Limpar Plots", callback=limpar_plots)
        dpg.add_progress_bar(tag="barra", default_value=0.0, width=300)
        dpg.add_text("0% concluído", tag="texto")
        # Combobox escolha do algoritmo de substituição
        # dpg.add_text("Algoritmo de Substituição:")
        dpg.add_combo(items=["FIFO", "LRU", "LFU", "Random"], default_value='FIFO',
                      label="<-- Algoritmo de Substituição", width=100, tag="combo_algoritmo",
                      callback=selecionar_algoritmo)
        # Salvar Grafico está com erro!!!
        # dpg.add_button(label="Salvar Simulação", callback=export_callback)
        dpg.add_button(label="Mostrar Heatmap", callback=mostrar_heatmap_callback)
    dpg.add_separator()
    dpg.add_input_text(label="<-- Resultado da Última Simulação", multiline=True, readonly=True, height=35,
                       tag="resultados_box")

    dpg.add_separator()
    dpg.add_text("", tag="mensagem_erro")
    dpg.add_separator()

    with dpg.group(horizontal=True):  # Inicia um grupo horizontal
        with dpg.plot(label="Taxa de acerto vs Tamanho do Bloco", tag="plot", height=380, width=600):
            dpg.add_plot_legend()
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Tamanho do Bloco", tag="x_axis")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Taxa de Acerto", tag="y_axis")
        dpg.add_input_text(label="<-- Resumo da Simulação", multiline=True, readonly=True, height=380, width=360,
                           default_value="", tag="Resumo")

dpg.create_viewport(title='Simulação de Cache', width=800, height=600)
dpg.setup_dearpygui()
dpg.maximize_viewport()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
