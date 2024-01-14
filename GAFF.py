import numpy as np

class FireflyAlgorithm(object):
    
    def __init__(self):
        self.sudoku_board = None
        self.population_size = 100
        self.num_generations = 400
        self.crossover_rate = 0.8
        self.mutation_rate = 0.01
        self.population = None
        self.fitness_values = None
        self.brightness_values = None
        self.pheromone_values = None
        return

    def load(self, path):
        with open(path, "r") as f:
            self.sudoku_board = np.loadtxt(f).astype(int)
        print("INPUT\n", self.sudoku_board)
        return

    def print_sudoku(self, sudoku):
        for row in sudoku:
            print(' '.join(map(str, row)))

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            individual = self.generate_random_individual()
            self.population.append(individual)
        self.population = np.array(self.population)

    def generate_random_individual(self):
        individual = np.copy(self.sudoku_board)
        empty_cells = np.where(individual == 0)
        values = np.random.permutation(np.arange(1, 10))

        for i in range(len(empty_cells[0])):
            row, col = empty_cells[0][i], empty_cells[1][i]
            individual[row, col] = values[i % 9]

        return individual

    def calculate_fitness(self, individuals=None):
        if individuals is None:
            individuals = self.population 
        fitness_values = []
        for individual in individuals:
            row_errors = 0
            col_errors = 0
       
            for row in range(9):
                individual_array = np.array(individual)
                freq_row = np.bincount(individual_array[row, :].astype(int))
                row_errors += np.sum(freq_row[freq_row > 1] - 1)
        
            for col in range(9):
                freq_col = np.bincount(individual[:, col])
                col_errors += np.sum(freq_col[freq_col > 1] - 1)
        
            total_errors = row_errors + col_errors
            fitness = 100 - total_errors
            fitness_values.append(fitness)
    
        self.fitness_values = np.array(fitness_values)
        return self.fitness_values

    def update_brightness(self):
        self.brightness_values = np.zeros((self.population_size, self.population_size))
        for i in range(self.population_size):
            for j in range(self.population_size):
                distance = self.calculate_distance(self.population[i], self.population[j])
                self.brightness_values[i, j] = self.fitness_values[j] * np.exp(-0.1 * distance**2)

    def update_pheromone(self):
        self.pheromone_values = np.zeros((self.population_size, self.population_size))
        for i in range(self.population_size):
            for j in range(self.population_size):
                random_w = np.random.rand()
                self.pheromone_values[i, j] = np.abs(self.fitness_values[i] * self.calculate_distance(self.population[i], self.population[j]) * random_w)

    def mutual_attraction(self):
        MP = (self.population_size / 2)**2
        mutual_attraction_value = 0
        for i in range(self.population_size):
            for j in range(self.population_size):
                mutual_attraction_value += self.pheromone_values[i, j] + self.brightness_values[i, j]
        return MP * mutual_attraction_value

    def mating_capability(self):
        mating_capability_values = np.zeros(self.population_size)
        for i in range(self.population_size):
            mating_capability_values[i] = int((self.fitness_values[i] / 100) * np.random.randint(0, self.population_size))
        return mating_capability_values

    def calculate_distance(self, individual1, individual2):
 
        distance = 0
        for i in range(9):
        
            distance += np.sum(individual1[i, :] == individual2[i, :])
      
            distance += np.sum(individual1[:, i] == individual2[:, i])
        return distance


    def selection_and_crossover(self):
        selected_parents = self.select_parents()
        for i in range(0, self.population_size, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            crossover_prob = np.random.rand()

            if crossover_prob < self.crossover_rate:
                crossover_point = np.random.randint(1, 9)
                child1 = np.hstack((parent1[:, :crossover_point], parent2[:, crossover_point:]))
                child2 = np.hstack((parent2[:, :crossover_point], parent1[:, crossover_point:]))
                self.population[i] = child1
                self.population[i + 1] = child2
    def mutate_population(self, mutation_probability):
        mutated_population = []

        for individual in self.population:
            mutated_individual = np.copy(individual)

            # Mutate each subgrid
            for subgrid_start_row in range(0, 9, 3):
                for subgrid_start_col in range(0, 9, 3):
                    if np.random.rand() < mutation_probability:
                        location1 = np.random.choice(9, size=2, replace=False)
                        location2 = np.random.choice(9, size=2, replace=False)

                        if individual[subgrid_start_row + location1[0]//3, subgrid_start_col + location1[0]%3] == 0 and \
                           individual[subgrid_start_row + location1[1]//3, subgrid_start_col + location1[1]%3] == 0 and \
                           individual[subgrid_start_row + location2[0]//3, subgrid_start_col + location2[0]%3] == 0 and \
                           individual[subgrid_start_row + location2[1]//3, subgrid_start_col + location2[1]%3] == 0:
                            
                            mutated_individual[subgrid_start_row + location1[0]//3, subgrid_start_col + location1[0]%3], \
                            mutated_individual[subgrid_start_row + location1[1]//3, subgrid_start_col + location1[1]%3] = \
                            mutated_individual[subgrid_start_row + location1[1]//3, subgrid_start_col + location1[1]%3], \
                            mutated_individual[subgrid_start_row + location1[0]//3, subgrid_start_col + location1[0]%3]

                            mutated_individual[subgrid_start_row + location2[0]//3, subgrid_start_col + location2[0]%3], \
                            mutated_individual[subgrid_start_row + location2[1]//3, subgrid_start_col + location2[1]%3] = \
                            mutated_individual[subgrid_start_row + location2[1]//3, subgrid_start_col + location2[1]%3], \
                            mutated_individual[subgrid_start_row + location2[0]//3, subgrid_start_col + location2[0]%3]
                            
            mutated_population.append(mutated_individual)

        self.population = np.array(mutated_population)

    def select_parents(self):

        total_fitness = np.sum(self.fitness_values)
        probabilities = self.fitness_values / total_fitness
        selected_indices = np.random.choice(np.arange(self.population_size), size=self.population_size, p=probabilities)
        selected_parents = [self.population[i] for i in selected_indices]
        return selected_parents

    def run(self):
        self.initialize_population()

        for generation in range(self.num_generations):
            self.calculate_fitness(self.population)
            self.update_brightness()
            self.update_pheromone()

            mutual_attraction_value = self.mutual_attraction()
            mating_capability_value = self.mating_capability()
            print(f"Generation {generation}: Mutual Attraction: {mutual_attraction_value}, Mating Capability: {mating_capability_value}")
           
            self.selection_and_crossover()
  
            self.mutate_population(self.mutation_rate)
        

        best_solution_index = np.argmax(self.fitness_values)
        best_solution = self.population[best_solution_index]
        print("\nBest Solution:")
        self.print_sudoku(best_solution)
        

        return best_solution


   
s = FireflyAlgorithm()
s.load("sudokueasy.txt")
s.run()
