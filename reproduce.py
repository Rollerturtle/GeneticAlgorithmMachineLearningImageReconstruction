import numpy as np
import cv2
import random
from deap import base, creator, tools, algorithms
from sklearn.linear_model import LinearRegression
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import time

# Parameters
k = 10000     # Number of generations
n = 100          # Population size
pc = 0.9         # Crossover probability
pm = 0.6         # Mutation probability
v = 80           # Buffer size
epsilon = 0.1    # Parameter threshold
pfc = 0.7        # Crossover probability factor
plc = 69         # Crossover probability level
pfm = 0.6        # Mutation probability factor
plm = 97         # Mutation probability level
N = 100          # Number of polygons
E = 3            # Number of vertices per polygon

# Load original image
original_image = cv2.imread('vangogh1.jpg', cv2.IMREAD_UNCHANGED)
original_image = cv2.resize(original_image, (395, 480))  # Resize the image to 395x480 pixels
H, W, _ = original_image.shape

# Fitness function
def evaluate(individual):
    reconstructed_image = np.zeros_like(original_image)
    for polygon in individual:
        vertices = polygon['vertices']
        color = polygon['color']
        absolute_vertices = [(int(v[0] * W), int(v[1] * H)) for v in vertices]
        cv2.fillPoly(reconstructed_image, [np.array(absolute_vertices)], color)
    mse = mean_squared_error(original_image, reconstructed_image)
    return 1 / (1 + mse),

# GA initialization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_polygon():
    vertices = [(random.random(), random.random()) for _ in range(E)]
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return {'vertices': vertices, 'color': color}

def mutate_polygon(individual):
    index = random.randrange(len(individual))
    if random.random() < 0.5:
        individual[index]['vertices'] = [(random.random(), random.random()) for _ in range(E)]
    else:
        individual[index]['color'] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return individual

def crossover_polygon(parent1, parent2):
    child1, child2 = [], []
    for p1, p2 in zip(parent1, parent2):
        new_poly1 = {'vertices': p1['vertices'][:E//2] + p2['vertices'][E//2:], 'color': p1['color']}
        new_poly2 = {'vertices': p2['vertices'][:E//2] + p1['vertices'][E//2:], 'color': p2['color']}
        child1.append(new_poly1)
        child2.append(new_poly2)
    return child1, child2

toolbox = base.Toolbox()
toolbox.register("individual", tools.initRepeat, creator.Individual, create_polygon, N)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", crossover_polygon)
toolbox.register("mutate", mutate_polygon)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    population = toolbox.population(n=n)
    buffer = []
    pm_current = pm  # Initialize current mutation probability

    # Initial evaluation
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    start_time = time.time()
    # Evolutionary process
    for gen in range(k):  # Loop will run until gen reaches k
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < pc:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < pm_current:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        # Store fitness in buffer and adjust parameters
        avg_fitness = np.mean([ind.fitness.values[0] for ind in population])
        buffer.append(avg_fitness)
        if len(buffer) > v:
            buffer.pop(0)

        if len(buffer) >= v:
            X = np.arange(len(buffer)).reshape(-1, 1)
            y = np.array(buffer)
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]

            if slope > epsilon:
                pm_current = max(pm_current - pfm / plm, 0.01)
            elif slope < -epsilon:
                pm_current = min(pm_current + pfm / plm, 0.9)
            else:
                pm_current = pm_current + (pfm / plm) if pm_current < 0.5 else pm_current - (pfm / plm)

        # Display progress
        best_fitness = max(ind.fitness.values[0] for ind in population)
        print(f"Generation: {gen+1}/{k}, Best Fitness: {best_fitness:.6f}")

    end_time = time.time()
    best_ind = tools.selBest(population, 1)[0]
    print(f"Execution time: {end_time - start_time} seconds")
    return best_ind

if __name__ == "__main__":
    best_solution = main()
    print("Best Solution:", best_solution)
    # Save or display the reconstructed image
    reconstructed_image = np.zeros_like(original_image)
    for polygon in best_solution:
        vertices = polygon['vertices']
        color = polygon['color']
        absolute_vertices = [(int(v[0] * W), int(v[1] * H)) for v in vertices]
        cv2.fillPoly(reconstructed_image, [np.array(absolute_vertices)], color)
    cv2.imwrite('reconstructed_vangogh1.png', reconstructed_image)
    
    # Evaluate quality of the solution
    mse = mean_squared_error(original_image, reconstructed_image)
    psnr = peak_signal_noise_ratio(original_image, reconstructed_image)
    ssim = structural_similarity(original_image, reconstructed_image, win_size=5, channel_axis=-1)
    print(f"MSE: {mse}, PSNR: {psnr}, SSIM: {ssim}")
