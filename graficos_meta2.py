import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# GRÁFICO 1: COMPARAÇÃO (BASELINE vs FINAL)
# ==========================================
# (Ajuste o valor da Baseline se preferir usar os 19.80% da tentativa suicida)
cenarios = ['Meta 2: Baseline\n(Função Antiga)', 'Meta 2: Final\n(Corredor Central)']
medias_sucesso = [58.20, 72.80]
erros_sucesso = [16.22, 6.49]

plt.figure(figsize=(8, 6))
barras = plt.bar(cenarios, medias_sucesso, yerr=erros_sucesso, capsize=8, 
                 color=['#ff9999', '#2ca02c'], edgecolor='black', alpha=0.85)

plt.ylabel('Taxa de Sucesso Média (%)', fontsize=12, fontweight='bold')
plt.title('Impacto da Nova Função Objetivo no Vento', fontsize=14, fontweight='bold')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionar valores em cima das barras
for bar, media in zip(barras, medias_sucesso):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{media:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('meta2_comparacao.png', dpi=300)
print("Gráfico 'meta2_comparacao.png' gerado com sucesso!")

# ==========================================
# GRÁFICO 2: CONSISTÊNCIA DAS 5 RUNS FINAIS
# ==========================================
runs = ['Run 0', 'Run 1', 'Run 2', 'Run 3', 'Run 4']
sucesso_runs = [65.0, 80.0, 66.0, 73.0, 80.0]

plt.figure(figsize=(9, 5))
# Destacar as melhores runs a azul escuro e a pior a azul claro
cores = ['#99ccff' if s < 60 else '#1f77b4' for s in sucesso_runs]
barras2 = plt.bar(runs, sucesso_runs, color=cores, edgecolor='black', alpha=0.85)

# Linha da média
plt.axhline(y=72.80, color='red', linestyle='--', linewidth=2, label='Média Global (72.8%)')

plt.ylabel('Taxa de Sucesso (%)', fontsize=12, fontweight='bold')
plt.title('Desempenho Individual (5 Execuções Independentes)', fontsize=14, fontweight='bold')
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionar valores em cima das barras
for bar, val in zip(barras2, sucesso_runs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('meta2_runs.png', dpi=300)
print("Gráfico 'meta2_runs.png' gerado com sucesso!")
plt.show()