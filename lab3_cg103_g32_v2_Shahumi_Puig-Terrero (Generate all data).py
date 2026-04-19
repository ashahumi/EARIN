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

    def decode(self, lower_bound, upper_bound, aoi):
        segment = self.array[lower_bound:upper_bound]
        val = int("".join(str(bit) for bit in segment), 2)
        max_int_val = (1 << len(segment)) - 1
        return min_max_norm(val, 0, max_int_val, aoi[0], aoi[1])

    def mutation(self, probability):
        if np.random.rand() < probability:
            gene_idx = np.random.randint(self.length)
            self.array[gene_idx] = 1 - self.array[gene_idx]

    def crossover(self, other):
        crossover_point = np.random.randint(1, self.length)
        child1_array = np.concatenate((self.array[:crossover_point], other.array[crossover_point:]))
        child2_array = np.concatenate((other.array[:crossover_point], self.array[crossover_point:]))
        return Chromosome(self.length, child1_array), Chromosome(self.length, child2_array)


# Group B: Himmelblau's function
def objective_function(*args):
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
        self.population = []

    def eval_objective_func(self, chromosome):
        args = []
        for i in range(self.obj_func_num_args):
            lb = i * self.bits_per_arg
            ub = (i + 1) * self.bits_per_arg
            args.append(chromosome.decode(lb, ub, self.aoi))
        return self.objective_function(*args)

    def tournament_selection(self):
        parents = []
        fitnesses = [self.eval_objective_func(ind) for ind in self.population]
        
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(self.population_size, self.tournament_size, replace=False)
            best_idx = min(tournament_indices, key=lambda idx: fitnesses[idx])
            parents.append(Chromosome(self.chromosome_length, self.population[best_idx].array.copy()))
            
        return parents

    def reproduce(self, parents):
        next_generation = []
        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i+1] if i+1 < self.population_size else parents[0]
            
            if np.random.rand() < self.crossover_probability:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1 = Chromosome(self.chromosome_length, parent1.array.copy())
                child2 = Chromosome(self.chromosome_length, parent2.array.copy())
            
            child1.mutation(self.mutation_probability)
            child2.mutation(self.mutation_probability)
            next_generation.extend([child1, child2])
            
        return next_generation[:self.population_size]

    def plot_func(self, trace, title="GA Optimization"):
        x = np.linspace(self.aoi[0], self.aoi[1], 400)
        y = np.linspace(self.aoi[0], self.aoi[1], 400)
        X, Y = np.meshgrid(x, y)
        Z = self.objective_function(X, Y)

        plt.figure(figsize=(8, 6))
        contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='viridis')
        plt.colorbar(contour, label='Objective Value')

        num_points = len(trace)
        trace_x = [pt[0] for pt in trace]
        trace_y = [pt[1] for pt in trace]

        plt.plot(trace_x, trace_y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

        for i, pt in enumerate(trace):
            color = plt.cm.Reds(1.0 - (i / num_points) * 0.7)
            plt.scatter(pt[0], pt[1], color=color, s=20, zorder=2)

        plt.plot(trace_x[0], trace_y[0], marker='s', color='black', markersize=8, label='Start')
        plt.plot(trace_x[-1], trace_y[-1], marker='*', color='blue', markersize=12, label='End (Best Found)')

        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def run(self, verbose=True):
        self.population = [Chromosome(self.chromosome_length) for _ in range(self.population_size)]
        trace = []

        for step in range(self.num_steps):
            best_ind = min(self.population, key=self.eval_objective_func)
            best_val = self.eval_objective_func(best_ind)
            
            best_args = []
            for i in range(self.obj_func_num_args):
                lb = i * self.bits_per_arg
                ub = (i + 1) * self.bits_per_arg
                best_args.append(best_ind.decode(lb, ub, self.aoi))
                
            trace.append(best_args)
            
            if verbose and (step % 10 == 0 or step == self.num_steps - 1):
                print(f"Generation {step:3d} | Best f(x1, x2) = {best_val:.6f} at x={best_args}")

            parents = self.tournament_selection()
            self.population = self.reproduce(parents)

        # Return the trace and the very last (best) value found in the final generation
        return trace, best_val


# -------------------------------------------------------------------
# AUTOMATED PARAMETER STUDY FOR REPORT
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Define the baseline configuration
    baseline_kwargs = {
        "chromosome_length": 32,
        "obj_func_num_args": 2,
        "objective_function": objective_function,
        "aoi": [-5.0, 5.0],
        "population_size": 50,
        "tournament_size": 3,
        "mutation_probability": 0.01,
        "crossover_probability": 0.8,
        "num_steps": 100
    }

    # 2. Define the exact configurations requested in the assignment
    parameter_studies = {
        "mutation_probability": [0.001, 0.01, 0.1],
        "tournament_size": [2, 3, 10],
        "population_size": [10, 50, 200],
        "crossover_probability": [0.2, 0.8, 1.0]
    }

    num_runs_per_config = 10  # Run 10 times to get mean and standard deviation

    print("Starting Automated Parameter Study...")

    for param_name, test_values in parameter_studies.items():
        print(f"\n{'='*60}")
        print(f"TESTING PARAMETER: {param_name}")
        print(f"{'='*60}")

        for val in test_values:
            print(f"Running 10 tests for {param_name} = {val} ...")
            
            # Prepare configuration for this specific test
            kwargs = baseline_kwargs.copy()
            kwargs[param_name] = val
            
            best_results = []
            trace_for_plot = None
            
            # Run the GA 10 times
            for i in range(num_runs_per_config):
                ga = GeneticAlgorithm(**kwargs)
                # verbose=False so it doesn't print every generation 100 times
                trace, final_best_val = ga.run(verbose=False) 
                best_results.append(final_best_val)
                
                # Save the trace from the first run to generate the plot
                if i == 0:
                    trace_for_plot = trace
            
            # Calculate Mean and Standard Deviation
            mean_val = np.mean(best_results)
            std_val = np.std(best_results)
            
            print(f"  -> RESULTS | Mean Minimum: {mean_val:.6f} | Standard Deviation: {std_val:.6f}")
            
            # Plot the representative trace for this setting
            plot_title = f"Himmelblau Optimization ( {param_name} = {val} )"
            ga.plot_func(trace_for_plot, title=plot_title)

    print("\nAll studies complete!")