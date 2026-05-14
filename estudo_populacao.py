import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt

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
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x

def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
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
    # Vai buscar o estado imediatamente antes do fim do episódio
    final_obs = observation_history[-2]
    
    # Descompactar as variáveis da observação para ser mais legível
    x, y, vx, vy, angle, v_angle, left_leg, right_leg = final_obs
    
    # Usamos a distância Euclidiana para uma aproximação suave à plataforma
    distance = np.sqrt(x**2 + y**2)
    fitness = -distance * 100 
    
    # Fundamental para que a nave aprenda a usar os motores para travar
    speed = np.sqrt(vx**2 + vy**2)
    fitness -= speed * 100
    
    # Queremos que a nave desça direita e estabilizada
    fitness -= abs(angle) * 100
    fitness -= abs(v_angle) * 50
    
    # Dá uns pontos extra por cada perna que tocar no solo de forma segura
    fitness += (left_leg + right_leg) * 20
    
    # Se a nave cumprir todos os requisitos de uma aterragem perfeita, 
    # recebe um bónus gigante para "carimbar" a excelência desse genótipo.
    success = check_successful_landing(final_obs)
    if success:
        fitness += 1000
        
    return fitness, success

def simulate(genotype, render_mode = None, seed=None, env = None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
        
    observation, info = env.reset(seed=seed)

    observation_history = [observation]
    for _ in range(STEPS):
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        
        observation_history.append(observation)

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective_function(observation_history)

def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes
    
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
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def parent_selection(population):
    """
    Seleção por Torneio (Tournament Selection)
    Escolhe 3 indivíduos aleatórios da população e o que tiver melhor fitness vence.
    """
    tournament_size = 3
    # Escolhe um grupo aleatório (torneio)
    tournament = random.sample(population, tournament_size)
    
    # Ordena o grupo do melhor para o pior
    tournament.sort(key=lambda x: x['fitness'], reverse=True)
    
    return copy.deepcopy(tournament[0])

def crossover(p1, p2):
    """
    Crossover Uniforme (Uniform Crossover)
    Como o genótipo é uma lista de pesos isolados, o crossover uniforme é ideal.
    Para cada gene, atira-se uma "moeda" (50% de probabilidade) para decidir
    se o descendente herda a característica do progenitor 1 ou do progenitor 2.
    """
    child_genotype = []
    
    for i in range(GENOTYPE_SIZE):
        if random.random() < 0.5:
            child_genotype.append(p1['genotype'][i])
        else:
            child_genotype.append(p2['genotype'][i])
            
    # Retorna o novo indivíduo com o fitness reiniciado a None
    return {'genotype': child_genotype, 'fitness': None}

def mutation(p):
    """
    Mutação Gaussiana (Gaussian Mutation)
    Percorre o genótipo e, com base na PROB_MUTATION, adiciona um pequeno
    ruído gaussiano (baseado no STD_DEV) ao peso. Limitamos os valores a [-1, 1].
    """
    for i in range(GENOTYPE_SIZE):
        # Verifica se este gene específico vai sofrer mutação
        if random.random() < PROB_MUTATION:
            # Adiciona o ruído normal/gaussiano
            p['genotype'][i] += random.gauss(0, STD_DEV)
            
            # Garante que o peso não ultrapassa os limites de -1 e 1
            # (Ajuda a evitar que a rede dispare valores caóticos)
            p['genotype'][i] = max(-1.0, min(1.0, p['genotype'][i]))
            
    return p
    
def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution():
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    #Create initial population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)

            else:
                ni = parent_selection(population)
                
            ni = mutation(ni)
            offspring.append(ni)
            
        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        
        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        print(f'Best of generation {gen}: {best[1]}')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return bests

def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Definições base (A Melhor Combinação: Exp 3)
    PROB_MUTATION = 0.008
    PROB_CROSSOVER = 0.9
    ELITE_SIZE = 0
    
    tamanhos_populacao = [10, 30, 50, 100, 150]
    n_runs = 3 # Reduzido para 3 para poupar tempo de computação
    seeds = [964, 952, 364]
    ntests = 50 # Número de simulações para avaliar o fitness final
    
    fitness_medio_por_populacao = []

    print("==================================================")
    print(" A INICIAR ESTUDO DO TAMANHO DA POPULAÇÃO")
    print(" Parâmetros fixos: Mut = 0.008 | CX = 0.9 | Elite = 0")
    print("==================================================")

    for pop_size in tamanhos_populacao:
        POPULATION_SIZE = pop_size # Atualiza a variável global
        print(f"\n-> A testar População de tamanho: {POPULATION_SIZE}")
        
        fitness_runs = []
        
        for i in range(n_runs):    
            print(f"   Run {i+1}/{n_runs} (Seed: {seeds[i]})...")
            random.seed(seeds[i])
            np.random.seed(seeds[i])
            
            # Evoluir com este tamanho de população
            bests = evolution()
            
            # Avaliar o melhor indivíduo da última geração
            melhor_genotipo = bests[-1][0]
            
            fit_total = 0
            for _ in range(ntests):
                f, _ = simulate(melhor_genotipo, render_mode=None, seed=None)
                fit_total += f
                
            fitness_final_run = fit_total / ntests
            fitness_runs.append(fitness_final_run)
            print(f"   -> Fitness da Run: {fitness_final_run:.2f}")
            
        media_pop = np.mean(fitness_runs)
        fitness_medio_por_populacao.append(media_pop)
        print(f" => Média para População {POPULATION_SIZE}: {media_pop:.2f}")

    print("\n==================================================")
    print(" ESTUDO CONCLUÍDO! A GERAR GRÁFICO...")
    print("==================================================")

    # ---------------------------------------------------------
    # GERAÇÃO DO GRÁFICO (CURVA DE COTOVELO)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Criar a linha do gráfico
    plt.plot(tamanhos_populacao, fitness_medio_por_populacao, marker='o', markersize=8, 
             linewidth=2.5, color='#ff7f0e', label='Fitness Médio Final')

    # Destacar o ponto "100" que é o nosso cotovelo
    if 100 in tamanhos_populacao:
        idx_100 = tamanhos_populacao.index(100)
        fit_100 = fitness_medio_por_populacao[idx_100]
        plt.annotate('Ponto de Cotovelo\n(População = 100)', 
                     xy=(100, fit_100), xytext=(120, fit_100 - 200),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     fontsize=11, fontweight='bold', ha='center')

    # Linha de referência (1000)
    plt.axhline(y=1000, color='gray', linestyle='--', alpha=0.7, label='Limiar de Sucesso (~1000)')

    # Formatação do gráfico
    plt.title('Estudo do Tamanho da População (Curva de Cotovelo)', fontsize=14, fontweight='bold')
    plt.xlabel('Tamanho da População', fontsize=12)
    plt.ylabel('Fitness Médio (Após Treino)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.xticks(tamanhos_populacao)

    plt.tight_layout()
    plt.savefig('estudo_populacao_real.png', dpi=300)
    print("Gráfico guardado como 'estudo_populacao_real.png'!")
    plt.show()