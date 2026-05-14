import matplotlib.pyplot as plt
import numpy as np

# Dados exatos recolhidos das suas 8 experiências
experiencias = ['Exp 1', 'Exp 2', 'Exp 3', 'Exp 4', 'Exp 5', 'Exp 6', 'Exp 7', 'Exp 8']
medias_sucesso = [42.40, 91.40, 97.00, 95.60, 24.80, 74.20, 89.40, 85.60]
std_sucesso = [29.03, 6.28, 1.79, 3.83, 33.62, 27.02, 7.76, 15.37]

# Configurar o tamanho do gráfico
plt.figure(figsize=(10, 6))

# Criar o gráfico de barras com as linhas de erro (yerr)
bars = plt.bar(experiencias, medias_sucesso, yerr=std_sucesso, capsize=5, 
               color='skyblue', edgecolor='black', alpha=0.8)

# Destacar a melhor experiência (Exp 3) a verde
bars[2].set_color('limegreen')
bars[2].set_edgecolor('black')

# Títulos e rótulos
plt.ylabel('Taxa de Sucesso Média (%)', fontsize=12)
plt.title('Comparação da Taxa de Sucesso - Experiências da Tabela 2', fontsize=14, fontweight='bold')
plt.ylim(0, 115) # Dá espaço para as barras de erro não cortarem no topo

# Adicionar o valor numérico em cima de cada barra
for bar, media in zip(bars, medias_sucesso):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{media:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Guardar a imagem na pasta
plt.savefig('grafico_taxa_sucesso.png', dpi=300)
print("Gráfico 'grafico_taxa_sucesso.png' gerado com sucesso!")
plt.show()