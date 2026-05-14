import matplotlib.pyplot as plt
import numpy as np

# Dados a comparar
cenarios = ['Meta 1\n(Sem Vento)', 'Baseline Meta 2\n(Com Vento)']
sucesso_medios = [97.00, 58.20]
sucesso_erros = [1.79, 16.22]

fitness_medios = [1005.28, 577.81]
fitness_erros = [18.92, 171.04]

# Criar a figura com 2 subgráficos lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
cores = ['limegreen', 'tomato']

# Gráfico 1: Taxa de Sucesso
barras1 = ax1.bar(cenarios, sucesso_medios, yerr=sucesso_erros, capsize=8, color=cores, edgecolor='black', alpha=0.8)
ax1.set_ylabel('Taxa de Sucesso Média (%)', fontsize=12)
ax1.set_title('Impacto do Vento na Taxa de Sucesso', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 110)

# Adicionar valores em cima das barras (Gráfico 1)
for bar, media in zip(barras1, sucesso_medios):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{media:.1f}%', ha='center', va='bottom', fontweight='bold')

# Gráfico 2: Fitness Médio
barras2 = ax2.bar(cenarios, fitness_medios, yerr=fitness_erros, capsize=8, color=cores, edgecolor='black', alpha=0.8)
ax2.set_ylabel('Fitness Médio Final', fontsize=12)
ax2.set_title('Impacto do Vento no Fitness', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1200)

# Adicionar valores em cima das barras (Gráfico 2)
for bar, media in zip(barras2, fitness_medios):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
             f'{media:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('impacto_vento.png', dpi=300)
print("Gráfico 'impacto_vento.png' gerado com sucesso!")
plt.show()