import numpy as np
import matplotlib.pyplot as plt


def min_max_norm(val, min_val, max_val, new_min, new_max):
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


class Chromosome:
    def __init__(self, length, array=None):
        self.length = length
        if array is None:
            self.array = np.random.randint(0, 2, size=length)
        else:
            self.array = np.array(array)

    def decode(self, bits_per_arg, lower_bound, upper_bound, aoi):
        decoded_args = []
        # Split the chromosome into segments for each variable
        for i in range(0, self.length, bits_per_arg):
            segment = self.array[i:i+bits_per_arg]
            # Convert binary array segment to integer
            val = int("".join(str(bit) for bit in segment), 2)
            # Normalize to the Area of Interest (aoi)
            norm_val = min_max_norm(val, lower_bound, upper_bound, aoi[0], aoi[1])
            decoded_args.append(norm_val)
        return decoded_args

    def mutation(self, probability):
        for i in range(self.length):
            if np.random.rand() < probability:
                self.array[i] = 1 - self.array[i]  # Flip the bit

    def crossover(self, other):
        # One-point crossover
        crossover_point = np.random.randint(1, self.length)
        child1_array = np.concatenate((self.array[:crossover_point], other.array[crossover_point:]))
        child2_array = np.concatenate((other.array[:crossover_point], self.array[crossover_point:]))
        
        return Chromosome(self.length, child1_array), Chromosome(self.length, child2_array)


def objective_function(*args):
    # Himmelblau's function
    x1, x2 = args[0], args[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


class GeneticAlgorithm:
    def __init__(self, chromosome_length, obj_func_num_args, objective_function, aoi,
                 population_size=100, tournament_size=2, mutation_probability=0.05,
                 crossover_probability=0.8, num_steps=50):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
        self.chromosome_length = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = int(chromosome_length / obj_func_num_args)
        self.objective_function = objective_function
        self.aoi = aoi
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps

    def eval_objective_func(self, chromosome):
        upper_bound = (1 << self.bits_per_arg) - 1
        decoded_args = chromosome.decode(self.bits_per_arg, 0, upper_bound, self.aoi)
        return self.objective_function(*decoded_args)

    def tournament_selection(self, population, fitnesses):
        # Select 'tournament_size' random individuals and return the best one (lowest fitness)
        tournament_indices = np.random.choice(self.population_size, self.tournament_size, replace=False)
        best_idx = min(tournament_indices, key=lambda idx: fitnesses[idx])
        # Return a copy of the winner to avoid modifying the current generation directly
        return Chromosome(self.chromosome_length, population[best_idx].array.copy())

    def reproduce(self, parents):
        next_generation = []
        # Process parents in pairs
        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            # Handle odd population sizes by pairing the last individual with the first one
            parent2 = parents[i+1] if i+1 < self.population_size else parents[0]
            
            if np.random.rand() < self.crossover_probability:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1 = Chromosome(self.chromosome_length, parent1.array.copy())
                child2 = Chromosome(self.chromosome_length, parent2.array.copy())
            
            # Apply mutation
            child1.mutation(self.mutation_probability)
            child2.mutation(self.mutation_probability)
            
            next_generation.extend([child1, child2])
            
        return next_generation[:self.population_size]

    def plot_func(self, trace):
        x = np.linspace(self.aoi[0], self.aoi[1], 400)
        y = np.linspace(self.aoi[0], self.aoi[1], 400)
        X, Y = np.meshgrid(x, y)
        Z = self.objective_function(X, Y)

        plt.figure(figsize=(10, 8))
        # Plotting the landscape
        contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='viridis')
        plt.colorbar(contour, label='Objective Value')

        # Extract trace coordinates
        trace_x = [pt[0] for pt in trace]
        trace_y = [pt[1] for pt in trace]

        # Plotting the trace trajectory
        plt.plot(trace_x, trace_y, marker='o', color='red', markersize=3, linestyle='-', linewidth=1, alpha=0.7, label='Search Trajectory')
        plt.plot(trace_x[0], trace_y[0], marker='s', color='cyan', markersize=8, label='Start')
        plt.plot(trace_x[-1], trace_y[-1], marker='*', color='magenta', markersize=12, label='End (Best Found)')

        plt.title("Genetic Algorithm Optimization on Himmelblau's Function")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def run(self):
        # Initialize population
        population = [Chromosome(self.chromosome_length) for _ in range(self.population_size)]
        trace = []

        for step in range(self.num_steps):
            # Evaluate all fitnesses once per generation
            fitnesses = [self.eval_objective_func(ind) for ind in population]
            
            # Record the best individual for the trace
            best_idx = np.argmin(fitnesses)
            best_ind = population[best_idx]
            upper_bound = (1 << self.bits_per_arg) - 1
            best_decoded = best_ind.decode(self.bits_per_arg, 0, upper_bound, self.aoi)
            trace.append(best_decoded)
            
            # Print progress every 10 steps
            if step % 10 == 0 or step == self.num_steps - 1:
                print(f"Generation {step:3d} | Best f(x1,x2) = {fitnesses[best_idx]:.6f} at x={best_decoded}")

            # 1. Selection
            parents = [self.tournament_selection(population, fitnesses) for _ in range(self.population_size)]
            
            # 2. Reproduction (Crossover and Mutation)
            population = self.reproduce(parents)

        # Plot the final trajectory
        self.plot_func(trace)
        return trace


# Configuration specific to the Group B variant setup
if __name__ == "__main__":
    ga = GeneticAlgorithm(
        chromosome_length=32,          # 16 bits * 2 arguments = 32 bits
        obj_func_num_args=2,           # x1 and x2
        objective_function=objective_function,
        aoi=[-5.0, 5.0],               # Bounded domain [-5, 5]
        population_size=50,            # Population size of 50
        tournament_size=3,             # Tournament size of 3
        mutation_probability=0.01,     # Mutation probability pm = 0.01
        crossover_probability=0.8,     # Crossover probability pc = 0.8
        num_steps=100                  # Generations (you can adjust this as needed for your parameter study)
    )
    ga.run()