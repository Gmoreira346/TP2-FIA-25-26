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
    
    # 1. Penalização pela distância ao centro (0,0)
    # Usamos a distância Euclidiana para uma aproximação suave à plataforma
    distance = np.sqrt(x**2 + y**2)
    fitness = -distance * 100 
    
    # 2. Penalização por velocidade alta
    # Fundamental para que a nave aprenda a usar os motores para travar
    speed = np.sqrt(vx**2 + vy**2)
    fitness -= speed * 100
    
    # 3. Penalização por má postura (inclinação e velocidade angular)
    # Queremos que a nave desça direita e estabilizada
    fitness -= abs(angle) * 100
    fitness -= abs(v_angle) * 50
    
    # 4. Recompensa de contacto
    # Dá uns pontos extra por cada perna que tocar no solo de forma segura
    fitness += (left_leg + right_leg) * 20
    
    # 5. O Grande Bónus de Sucesso
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

    # --to evolve the controller--    
    evolve = True
    render_mode = None

    if evolve:
        # Configurações das 8 experiências exigidas na Tabela 2 do enunciado
        experiencias = [
            {'id': 1, 'mut': 0.008, 'cx': 0.5, 'elite': 0},
            {'id': 2, 'mut': 0.050, 'cx': 0.5, 'elite': 0},
            {'id': 3, 'mut': 0.008, 'cx': 0.9, 'elite': 0},
            {'id': 4, 'mut': 0.050, 'cx': 0.9, 'elite': 0},
            {'id': 5, 'mut': 0.008, 'cx': 0.5, 'elite': 1},
            {'id': 6, 'mut': 0.050, 'cx': 0.5, 'elite': 1},
            {'id': 7, 'mut': 0.008, 'cx': 0.9, 'elite': 1},
            {'id': 8, 'mut': 0.050, 'cx': 0.9, 'elite': 1},
        ]

        n_runs = 5
        # Selecionamos 5 seeds da lista original para garantir reprodutibilidade
        seeds = [964, 952, 364, 913, 140] 

        for exp in experiencias:
            print(f"\n=========================================")
            print(f" A INICIAR EXPERIÊNCIA {exp['id']}")
            print(f" Mutação: {exp['mut']} | Crossover: {exp['cx']} | Elitismo: {exp['elite']}")
            print(f"=========================================")
            
            # 1. Atualizar as variáveis globais dinamicamente
            PROB_MUTATION = exp['mut']
            PROB_CROSSOVER = exp['cx']
            ELITE_SIZE = exp['elite']
            
            # 2. Criar uma pasta específica para esta experiência
            pasta_exp = f"Experiencia_{exp['id']}"
            os.makedirs(pasta_exp, exist_ok=True)

            # 3. Correr as 5 repetições
            for i in range(n_runs):    
                print(f"  -> A executar Run {i+1}/{n_runs} (Seed: {seeds[i]})...")
                
                # Definir as seeds para garantir que os resultados são iguais se repetir
                random.seed(seeds[i])
                np.random.seed(seeds[i])
                
                # Correr o Algoritmo Evolucionário
                bests = evolution()
                
                # 4. Guardar o log dentro da pasta da experiência
                caminho_ficheiro = os.path.join(pasta_exp, f'log{i}.txt')
                with open(caminho_ficheiro, 'w') as f:
                    for b in bests:
                        f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')
                        
            print(f" Experiência {exp['id']} concluída! Logs guardados em '{pasta_exp}/'")

    else:
        # --to test the evolved controller without/with visualisation--
        # Exemplo: testar o melhor indivíduo da run 0 da Experiência 8
        filename = 'Experiencia_8/log0.txt' 
        render_mode = 'human'
        
        print(f"A testar o ficheiro: {filename}")
        bests = load_bests(filename)
        b = bests[-1] # Vai buscar o último (melhor da última geração)
        SHAPE = b[1]
        ind = b[2]
            
        ind = {'genotype': ind, 'fitness': None}
            
        ntests = TEST_EPISODES

        fit, success = 0, 0
        for i in range(1, ntests+1):
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed=None)
            fit += f
            success += s
            
        print(f"Resultados de Teste - Fitness Médio: {fit/ntests:.2f} | Taxa de Sucesso: {(success/ntests)*100:.2f}%")