import matplotlib.pyplot as plt

# ATENÇÃO: Altere este caminho para apontar para o ficheiro da sua melhor run
# Baseado nos seus resultados, a Run 3 da Experiência 3 foi excelente.
caminho_ficheiro = 'Meta2_Vento_Final/log_vento_run1.txt'

geracoes = []
fitness_historico = []

try:
    # Ler o ficheiro e extrair o fitness de cada geração
    with open(caminho_ficheiro, 'r') as f:
        for i, linha in enumerate(f):
            # A linha tem o formato: fitness \t shape \t genotype
            partes = linha.split('\t')
            fitness = float(partes[0])  # O fitness é o primeiro valor da linha
            geracoes.append(i)
            fitness_historico.append(fitness)

    # Configurar o tamanho e estilo do gráfico
    plt.figure(figsize=(10, 6))
    
    # Desenhar a linha principal
    plt.plot(geracoes, fitness_historico, color='#2ca02c', linewidth=2.5, label='Melhor Indivíduo da Geração')


    # Personalização visual
    plt.title('Curva de Aprendizagem - Com vento (Experiência 1)', fontsize=14, fontweight='bold')
    plt.xlabel('Geração', fontsize=12)
    plt.ylabel('Valor de Fitness', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')

    plt.tight_layout()
    
    # Guardar a imagem
    plt.savefig('curva_aprendizagem_com_vento.png', dpi=300)
    print("Gráfico 'curva_aprendizagem.png' gerado com sucesso!")
    plt.show()

except FileNotFoundError:
    print(f"Erro: Não foi possível encontrar o ficheiro '{caminho_ficheiro}'. Verifique o caminho.")