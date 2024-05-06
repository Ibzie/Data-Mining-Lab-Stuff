import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import random
from deap import base, creator, tools

# Load data from Parquet files
training_df = pd.read_parquet("training.parquet")
testing_df = pd.read_parquet("testing.parquet")

# Drop specified features
drop_features = ['service', 'state', 'attack_cat', 'swin', 'dwin', 'sloss', 'dloss',
                 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
                 'synack', 'ackdat', 'dmean', 'trans_depth', 'response_body_len',
                 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_ftp_cmd',
                 'ct_flw_http_mthd', 'is_sm_ips_ports']
training_df.drop(columns=drop_features, inplace=True)
testing_df.drop(columns=drop_features, inplace=True)

# Standardize 'dur' column
training_df['dur'] = training_df['dur'] * 1000
testing_df['dur'] = testing_df['dur'] * 1000

# Standardize 'sload' column
training_df['sload'] = training_df['sload'] * 1000000 if training_df['sload'].max() < 1000 else training_df['sload']
testing_df['sload'] = testing_df['sload'] * 1000000 if testing_df['sload'].max() < 1000 else testing_df['sload']

# Encode categorical variables (e.g., 'proto') using one-hot encoding
training_df = pd.get_dummies(training_df, columns=['proto'])
testing_df = pd.get_dummies(testing_df, columns=['proto'])

# Get missing columns in the testing set that are present in the training set
missing_cols = set(training_df.columns) - set(testing_df.columns)

# Add missing columns to testing set with default values (0)
for col in missing_cols:
    testing_df[col] = 0

# Ensure the order of columns in testing set is the same as in training set
testing_df = testing_df[training_df.columns]

# Split the datasets into features and labels
X_train = training_df.drop(columns=['label'])
y_train = training_df['label']
X_test = testing_df.drop(columns=['label'])
y_test = testing_df['label']

# Standardize features (optional but recommended for many models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Example fitness function
def evaluate(individual):
    # Count the number of features selected (number of 1s in the individual)
    num_features_selected = sum(individual)

    # Return the absolute value of the negative count of selected features
    return abs(-num_features_selected),


# Define the individual (chromosome) and fitness classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Genetic Algorithm parameters
NUM_GENERATIONS = 50
NUM_FEATURES = X_train_scaled.shape[1]

# Toolbox setup
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NUM_FEATURES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


# Main Genetic Algorithm function
def genetic_algorithm(population_size, crossover_probability, mutation_probability):
    POPULATION_SIZE = population_size
    CROSSOVER_PROBABILITY = crossover_probability
    MUTATION_PROBABILITY = mutation_probability

    population = toolbox.population(n=POPULATION_SIZE)

    for generation in range(NUM_GENERATIONS):
        offspring = toolbox.select(population, len(population))

        # Optimized cloning process using list comprehension
        offspring = [toolbox.clone(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_PROBABILITY:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_individuals))
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    return population


def conduct_experiment(experiment_name, population_size, crossover_probability, mutation_probability):
    print(f"Experiment: {experiment_name}")
    final_population = genetic_algorithm(population_size, crossover_probability, mutation_probability)
    best_individual = tools.selBest(final_population, k=1)[0]
    best_fitness = best_individual.fitness.values[0]
    print("Best individual:", best_individual)
    print("Best fitness:", best_fitness)


if __name__ == "__main__":
    # Experiment 1: Population=50, Crossover=0.6, Mutation=0.1
    conduct_experiment("Experiment 1", 50, 0.6, 0.1)

    # Experiment 2: Population=100, Crossover=0.8, Mutation=0.2
    conduct_experiment("Experiment 2", 100, 0.8, 0.2)

    # Experiment 3: Population=200, Crossover=1, Mutation=0.5
    conduct_experiment("Experiment 3", 200, 1, 0.5)