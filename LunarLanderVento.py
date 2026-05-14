import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# ==========================================
# CONFIGURAÇÕES (O VENTO ESTÁ LIGADO!)
# ==========================================
ENABLE_WIND = True
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
TEST_EPISODES = 100
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

# ==========================================
# MELHORES PARÂMETROS DA META 1
# ==========================================
POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 150 
PROB_CROSSOVER = 0.9
PROB_MUTATION = 0.008
STD_DEV = 0.1
ELITE_SIZE = 0

def network(shape, observation, ind):
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
    # Vamos buscar o estado final (antes do choque/pouso)
    final_obs = observation_history[-2]
    x, y, vx, vy, angle, v_angle, left_leg, right_leg = final_obs
    
    # --- Avaliação Base (Focada no Pouso) ---
    distance = np.sqrt(x**2 + y**2)
    fitness = -distance * 100 
    
    speed = np.sqrt(vx**2 + vy**2)
    fitness -= speed * 100
    
    fitness -= abs(angle) * 100
    fitness -= abs(v_angle) * 50
    
    fitness += (left_leg + right_leg) * 20
    
    # --- Avaliação de Trajetória ---
    # Premiamos a permanência no centro.
    # Se o vento soprar para a esquerda, a rede aprenderá a inclinar para a direita.
    # Se soprar para a direita, aprenderá a inclinar para a esquerda.
    estabilidade_trajetoria = 0
    for obs in observation_history:
        # Recompensa por frame se estiver no corredor central (X entre -0.3 e 0.3)
        if abs(obs[0]) < 0.3:
            estabilidade_trajetoria += 2.0
        else:
            # Penaliza se for arrastado pelo vento para fora do corredor
            estabilidade_trajetoria -= 1.0
            
    fitness += estabilidade_trajetoria
    
    # --- Bónus de Sucesso ---
    success = check_successful_landing(final_obs)
    if success:
        fitness += 1000
        
    return fitness, success

def simulate(genotype, render_mode = None, seed=None, env = None):
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode=render_mode, 
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
    env = gym.make("LunarLander-v3", render_mode=None, 
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
        genotype = [random.uniform(-1,1) for _ in range(GENOTYPE_SIZE)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def parent_selection(population):
    tournament_size = 3
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda x: x['fitness'], reverse=True)
    return copy.deepcopy(tournament[0])

def crossover(p1, p2):
    child_genotype = []
    for i in range(GENOTYPE_SIZE):
        if random.random() < 0.5:
            child_genotype.append(p1['genotype'][i])
        else:
            child_genotype.append(p2['genotype'][i])
    return {'genotype': child_genotype, 'fitness': None}

def mutation(p):
    for i in range(GENOTYPE_SIZE):
        if random.random() < PROB_MUTATION:
            p['genotype'][i] += random.gauss(0, STD_DEV)
            p['genotype'][i] = max(-1.0, min(1.0, p['genotype'][i]))
    return p
    
def survival_selection(population, offspring):
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution():
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key=lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)
            else:
                ni = parent_selection(population)
                
            ni = mutation(ni)
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

def load_bests(fname):
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests

if __name__ == '__main__':
    evolve = True  # Mude para False se quiser apenas reavaliar logs já gerados
    
    pasta_baseline = "Meta2_Vento_Final"
    n_runs = 5
    seeds = [964, 952, 364, 913, 140]
    
    if evolve:
        os.makedirs(pasta_baseline, exist_ok=True)
        print("==================================================")
        print(" A TREINAR FINAL PARA A META 2 (VENTO LIGADO)")
        print("==================================================")
        
        for i in range(n_runs):
            print(f" -> A executar Run {i+1}/{n_runs} (Seed: {seeds[i]})...")
            random.seed(seeds[i])
            np.random.seed(seeds[i])
            
            bests = evolution()
            
            caminho_ficheiro = os.path.join(pasta_baseline, f'log_vento_run{i}.txt')
            with open(caminho_ficheiro, 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')
                    
        print("\nTreino concluído! A iniciar testes de avaliação...")

    # ==========================================
    # AVALIAÇÃO E ESTATÍSTICAS
    # ==========================================
    print("\n==================================================")
    print(" RESULTADOS FINAIS META2 (VENTO LIGADO)")
    print("==================================================")
    
    fitness_runs = []
    sucesso_runs = []
    
    for i in range(n_runs):
        filename = f'{pasta_baseline}/log_vento_run{i}.txt'
        bests = load_bests(filename)
        b = bests[-1] 
        ind = {'genotype': b[2], 'fitness': None}
        
        fit_total, success_total = 0, 0
        for _ in range(TEST_EPISODES):
            f, s = simulate(ind['genotype'], render_mode=None, seed=None)
            fit_total += f
            success_total += s
            
        taxa_sucesso = (success_total / TEST_EPISODES) * 100
        fitness_medio_run = fit_total / TEST_EPISODES
        
        fitness_runs.append(fitness_medio_run)
        sucesso_runs.append(taxa_sucesso)
        print(f"  Run {i}: Sucesso = {taxa_sucesso}% | Fitness = {fitness_medio_run:.2f}")

    media_sucesso = np.mean(sucesso_runs)
    std_sucesso = np.std(sucesso_runs)
    media_fit = np.mean(fitness_runs)
    std_fit = np.std(fitness_runs)
    
    print("\n--- RESUMO ESTATÍSTICO ---")
    print(f"Taxa de Sucesso: {media_sucesso:.2f}% ± {std_sucesso:.2f}%")
    print(f"Fitness Médio:   {media_fit:.2f} ± {std_fit:.2f}")
    print("==================================================")