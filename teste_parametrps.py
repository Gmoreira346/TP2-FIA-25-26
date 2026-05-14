import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
TEST_EPISODES = 1000
STEPS = 500

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()

nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100
PROB_CROSSOVER = 0.9

PROB_MUTATION = 1.0/GENOTYPE_SIZE
STD_DEV = 0.1

ELITE_SIZE = 1

def network(shape, observation,ind):
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x

def check_successful_landing(observation):
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1
    on_landing_pad = abs(x) <= 0.2
    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        return True
    return False

def objective_function(observation_history):
    final_obs = observation_history[-2]
    x, y, vx, vy, angle, v_angle, left_leg, right_leg = final_obs
    
    distance = np.sqrt(x**2 + y**2)
    fitness = -distance * 100 
    speed = np.sqrt(vx**2 + vy**2)
    fitness -= speed * 100
    fitness -= abs(angle) * 100
    fitness -= abs(v_angle) * 50
    fitness += (left_leg + right_leg) * 20
    
    success = check_successful_landing(final_obs)
    if success:
        fitness += 1000
        
    return fitness, success

def simulate(genotype, render_mode = None, seed=None, env = None):
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
        
    observation, info = env.reset(seed=seed)

    observation_history = [observation]
    for _ in range(STEPS):
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        
        observation_history.append(observation)

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective_function(observation_history)

def evaluate(evaluationQueue, evaluatedQueue):
    env = gym.make("LunarLander-v3", render_mode =None, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
    while True:
        ind = evaluationQueue.get()
        if ind is None:
            break
        ind['fitness'] = simulate(ind['genotype'], seed = None, env = env)[0]
        evaluatedQueue.put(ind)
    env.close()
    
def evaluate_population(population):
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

def generate_initial_population():
    population = []
    for i in range(POPULATION_SIZE):
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

# ==========================================
# OPERADORES DE SELEÇÃO
# ==========================================
def tournament_selection(population):
    tournament_size = 3
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda x: x['fitness'], reverse=True)
    return copy.deepcopy(tournament[0])

def best_selection(population):
    # Escolhe aleatoriamente de entre os 10% melhores (assume que a população vem ordenada)
    top_10 = max(1, int(len(population) * 0.1))
    return copy.deepcopy(random.choice(population[:top_10]))

def random_selection(population):
    # Escolha 100% aleatória (geralmente serve apenas como baseline de controlo)
    return copy.deepcopy(random.choice(population))


# ==========================================
# OPERADORES DE CROSSOVER
# ==========================================
def uniform_crossover(p1, p2):
    child_genotype = []
    for i in range(GENOTYPE_SIZE):
        if random.random() < 0.5:
            child_genotype.append(p1['genotype'][i])
        else:
            child_genotype.append(p2['genotype'][i])
    return {'genotype': child_genotype, 'fitness': None}

def one_point_crossover(p1, p2):
    # Escolhe um ponto de corte aleatório no genótipo
    corte = random.randint(1, GENOTYPE_SIZE - 1)
    child_genotype = p1['genotype'][:corte] + p2['genotype'][corte:]
    return {'genotype': child_genotype, 'fitness': None}


# ==========================================
# OPERADORES DE MUTAÇÃO
# ==========================================
def gaussian_mutation(p):
    for i in range(GENOTYPE_SIZE):
        if random.random() < PROB_MUTATION:
            p['genotype'][i] += random.gauss(0, STD_DEV)
            p['genotype'][i] = max(-1.0, min(1.0, p['genotype'][i]))
    return p

def uniform_mutation(p):
    for i in range(GENOTYPE_SIZE):
        if random.random() < PROB_MUTATION:
            # Substitui completamente o gene por um novo valor aleatório
            p['genotype'][i] = random.uniform(-1.0, 1.0)
    return p

# ==========================================

def survival_selection(population, offspring):
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution(sel_func, cx_func, mut_func):
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                # Usar a função de seleção dinâmica
                p1 = sel_func(population)
                p2 = sel_func(population)
                # Usar a função de crossover dinâmica
                ni = cx_func(p1, p2)
            else:
                ni = sel_func(population)
                
            # Usar a função de mutação dinâmica
            ni = mut_func(ni)
            offspring.append(ni)
            
        offspring = evaluate_population(offspring)
        population = survival_selection(population, offspring)
        
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)

    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    return bests

if __name__ == '__main__':
    # Para gerar o gráfico, importe o matplotlib
    import matplotlib.pyplot as plt

    # Definição dos operadores a testar
    selections = [("Torneio", tournament_selection), 
                  ("Top10%", best_selection), 
                  ("Aleatorio", random_selection)]
    
    crossovers = [("Unif", uniform_crossover), 
                  ("1Ponto", one_point_crossover)]
    
    mutations = [("Gauss", gaussian_mutation), 
                 ("Uniforme", uniform_mutation)]

    # Reduzimos para 3 runs por combinação apenas para este teste preliminar não demorar horas
    n_runs = 3
    seeds = [964, 952, 364]
    
    resultados_finais = []

    print("==================================================")
    print("A INICIAR BATERIA DE TESTES DE OPERADORES GENÉTICOS")
    print("Total de combinações a testar: 12")
    print("==================================================")

    for sel_name, sel_func in selections:
        for cx_name, cx_func in crossovers:
            for mut_name, mut_func in mutations:
                combo_name = f"{sel_name} | {cx_name} | {mut_name}"
                print(f"\nA testar combinação: {combo_name}")
                
                fitness_runs = []
                for i in range(n_runs):
                    random.seed(seeds[i])
                    np.random.seed(seeds[i])
                    
                    # Corre a evolução passando as funções como argumento
                    bests = evolution(sel_func, cx_func, mut_func)
                    
                    # Guarda o fitness do melhor indivíduo da ÚLTIMA geração
                    melhor_fitness_final = bests[-1][1]
                    fitness_runs.append(melhor_fitness_final)
                    print(f"  Run {i+1}/{n_runs}: Fitness = {melhor_fitness_final:.2f}")
                
                media_combo = np.mean(fitness_runs)
                resultados_finais.append((combo_name, media_combo))
                print(f" -> Média da Combinação: {media_combo:.2f}")

    # ==========================================
    # GERAÇÃO DO GRÁFICO DE RESULTADOS
    # ==========================================
    # Ordenar resultados do pior para o melhor para o gráfico ficar bonito
    resultados_finais.sort(key=lambda x: x[1])
    
    nomes = [r[0] for r in resultados_finais]
    pontuacoes = [r[1] for r in resultados_finais]

    plt.figure(figsize=(12, 8))
    # Destacar a barra com melhor pontuação a verde e as outras a azul
    cores = ['skyblue' if i < len(pontuacoes)-1 else 'limegreen' for i in range(len(pontuacoes))]
    
    barras = plt.barh(nomes, pontuacoes, color=cores)
    plt.xlabel('Fitness Médio Final (Após 100 Gerações)')
    plt.title('Comparação de Desempenho: Combinações de Operadores Genéticos')
    
    # Adicionar os valores numéricos à frente das barras
    for barra in barras:
        plt.text(barra.get_width(), barra.get_y() + barra.get_height()/2, 
                 f' {barra.get_width():.1f}', 
                 va='center', ha='left')

    plt.tight_layout()
    plt.savefig('comparacao_operadores.png')
    print("\n==================================================")
    print("Teste concluído! O gráfico foi guardado como 'comparacao_operadores.png'.")
    print("Use a melhor combinação verificada neste gráfico para o seu algoritmo final.")
    print("==================================================")