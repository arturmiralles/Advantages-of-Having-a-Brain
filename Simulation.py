import NeuralNetwork
import numpy as np
import csv
import time
import random

number_of_networks = 50 # Parameter N.
number_of_generations = 10
game = "IPD" # Options: "IPD" (Iterated Prisoner's Dilemma), "ISD" (Iterated Snowdrift Dilemma).
mutation_value_probability = 0.1 # Parameter p_v.
mutation_structure_probability = 0.02 # Parameter p_s.
offspring_rate = 1
reproduction_method = "sexual_TE"  # Options: "asexual_TE" (asexual Truncated Elitism), "sexual_TE" (sexual Truncated Elitism), "proportionate_selection" (asexual Proportionate Selection).
csv_filename = f"{game}_{number_of_networks}_nets_{number_of_generations}_gens.csv"

class Simulation:
    '''This simulation file is for reproduction visualisation purposes. Each individual produces '''
    def __init__(self):
        '''
        CHARACTERISTICS:
            population: list of neural networks containing the whole population in the current generation.
            number_of_rounds: list storing how many rounds each individual has played during the generation.
            sum_of_payoffs: list containing the sum of the payoffs for all games that each individual has played in the given generation.
            population_cognitive_nodes: list containing the number of cognitive nodes of each individual in the generation.
            population_context_nodes: list containing the number of context nodes of each individual in the generation.
            population_intelligence: list containing the sum of the number of cognitive and context nodes of each individual in the generation.
            population_fitness: list containing the fitness of each individual in the generation.
            total_cooperations: number of times individuals of a given generation have decided to cooperate.
            total_defections: number of times individuals of a given generation have decided to defect.
            population_cooperations: number of times each individual cooperated in the generation.
            population_defections: number of times each individual defected in the generation.
            reproduction_counts: list storing how many times each individual has reproduced.
            reproduction_probabilities: list of probabilities for each individual to reproduce.
            test_set: list of test scenarios for evaluating strategies.
            pure_strategies: list of pure strategies for comparison.
            data: matrix containing the data for each individual of each generation.
            strategy_data: list storing strategy data for each individual.
            reproduction_data: list storing reproduction data for each generation.
        '''
        self.population = [NeuralNetwork.NeuralNetwork() for i in range(number_of_networks)]
        self.number_of_rounds = list(np.zeros(number_of_networks))
        self.sum_of_payoffs = list(np.zeros(number_of_networks))
        self.population_cognitive_nodes = []
        self.population_context_nodes = []
        self.population_intelligence = self.count_nodes()
        self.population_fitness = []
        self.total_cooperations = 0
        self.total_defections = 0
        self.population_cooperations = list(np.zeros(number_of_networks))
        self.population_defections = list(np.zeros(number_of_networks))
        self.reproduction_counts = list(np.zeros(number_of_networks))
        self.reproduction_probabilities = list(np.zeros(number_of_networks))
        self.test_set = self.initialise_test_set()
        self.pure_strategies = self.pure_strategy()
        self.data = []
        self.strategy_data = []
        self.reproduction_data = []

    def count_nodes(self):
        '''Function that returns a list containing the total number of inner nodes for each of the individuals in the population.'''
        num_of_nodes = []
        for network in self.population:
            num_context = network.context_node_count
            num_cognitive = len(network.inner_nodes)
            count = num_context + num_cognitive
            self.population_cognitive_nodes.append(num_cognitive)
            self.population_context_nodes.append(num_context)
            num_of_nodes.append(count)
        return num_of_nodes

    def game_payoffs(self, game):
        '''Function that, given the name of a game (either "IPD" or "ISD"), gives out the payoffs for all the possible outcomes.
            "IPD": Iterated Prisoner's Dilemma.
            "ISD": Iterated Snowdrift Dilemma.'''
        if game == "IPD":
            return {'both_cooperate': 6, 'both_defect': 2, 'self_cooperates_opponent_defects': 1, 'self_defects_opponent_cooperates': 7}
        elif game == "ISD":
            return {'both_cooperate': 5, 'both_defect': 1, 'self_cooperates_opponent_defects': 2, 'self_defects_opponent_cooperates': 8}
        else:
            raise ValueError("Unknown game.")

    def run(self, game, number_of_generations):
        '''
        Function that, when called, runs a full simulation of a given game for a number of generations previously determined.

        INPUT:
            game: name of the game that we we want to play:
                "IPD": Iterated Prisoner's Dilemma.
                "ISD": Iterated Snowdrift Dilemma.
            number_of_generations: total number of generations of individuals we want to produce.
        '''
        start_time = time.time()
        payoff_game = self.game_payoffs(game)

        for gen in range(number_of_generations):
            print(f"Running generation {gen}...")
            self.play_everyone(payoff_game)
            self.population_fitness = self.fitness()
            self.update_reproduction_probabilities()
            self.store_generation_data(gen)
            self.characterise_networks(gen, payoff_game, self.pure_strategies)
            self.reproduction()
        self.save_data()
        print("It took", time.time() - start_time, "seconds.")

    def play_everyone(self, payoff_game):
        '''Function that, when called, makes each individual of the population play a game against every other individual.

        INPUT:
            payoff_game: dictionary of payoffs for all the possible outcomes in the game.
        '''
        for index_a in range(number_of_networks - 1):
            for index_b in range(index_a + 1, number_of_networks):
                self.play_game(index_a, index_b, payoff_game)

    def play_game(self, index_a, index_b, payoff_game):
        '''
        Function that makes two individual play a game against each other. "The number of rounds to be played in each interaction between
        individuals is determined by taking a random number from a negative binomial distribution with a probability of 0.98 of success
        and failure number of 1, and adding one to this value. This gives a mean length of each interaction of 50 rounds."

        INPUTS:
            index_a: index of the first individual in the total population list.
            index_b: index of the second individual in the total population list.
            payoff_game: dictionary of payoffs for all the possible outcomes in the game.
        '''
        player_a = self.population[index_a]
        player_b = self.population[index_b]

        player_a_cooperates = player_a.first_round
        player_b_cooperates = player_b.first_round

        number_of_rounds = np.random.negative_binomial(1, 0.02) + 1
        
        self.number_of_rounds[index_a] += number_of_rounds
        self.number_of_rounds[index_b] += number_of_rounds

        for round in range(number_of_rounds):
            if player_a_cooperates:
                self.total_cooperations += 1
                self.population_cooperations[index_a] += 1
                
            else:
                self.total_defections += 1
                self.population_defections[index_a] += 1

            if player_b_cooperates:
                self.total_cooperations += 1
                self.population_cooperations[index_b] += 1
            else:
                self.total_defections += 1
                self.population_defections[index_b] += 1

            player_a_payoff, player_b_payoff = self.assign_payoffs(player_a_cooperates, player_b_cooperates, payoff_game)
            self.sum_of_payoffs[index_a] += player_a_payoff
            self.sum_of_payoffs[index_b] += player_b_payoff

            player_a_cooperates = player_a.forward(player_a_payoff, player_b_payoff)
            player_b_cooperates = player_b.forward(player_b_payoff, player_a_payoff)
        

    def assign_payoffs(self, a_cooperates, b_cooperates, payoff_game):
        '''
        Function that, given the decision made by the players of a game, assigns the corresponding payoff to each of them.

        INPUTS:
            a_cooperates: boolean indicating whether player A cooperated ('True') or defected ('False').
            b_cooperates: boolean indicating whether player B cooperated ('True') or defected ('False').
            payoff_game: dictionary of payoffs for all the possible outcomes in the game.
        '''
        if a_cooperates and b_cooperates:
            return payoff_game['both_cooperate'], payoff_game['both_cooperate']
        elif a_cooperates and not b_cooperates:
            return payoff_game['self_cooperates_opponent_defects'], payoff_game['self_defects_opponent_cooperates']
        elif not a_cooperates and b_cooperates:
            return payoff_game['self_defects_opponent_cooperates'], payoff_game['self_cooperates_opponent_defects']
        else:
            return payoff_game['both_defect'], payoff_game['both_defect']

    def fitness(self):
        '''Function that calculates the fitness of all individuals in a population. "The fitness of each individual is calculated as their 
        mean payoff per round during that generation minus a fitness penalty of 0.01 * i. The metric i is a measure of the intelligence
        of the network, defined as the sum of their cognitive and context nodes."'''
        mean_payoff_per_round = [sum_payoff / num_rounds if num_rounds > 0 else 0 for sum_payoff, num_rounds in zip(self.sum_of_payoffs, self.number_of_rounds)]
        return [mean_payoff - 0.01 * intelligence for mean_payoff, intelligence in zip(mean_payoff_per_round, self.population_intelligence)]

    def update_reproduction_probabilities(self):
        '''Function that updates the reproduction probabilities based on the current fitness values.'''
        total_fitness_population = sum(self.population_fitness)
        if total_fitness_population <= 0:
            self.reproduction_probabilities = [1.0 / number_of_networks] * number_of_networks
        else:
            self.reproduction_probabilities = [fitness / total_fitness_population for fitness in self.population_fitness]

    def store_generation_data(self, generation):
        '''Function that stores the data of each individual in a population in the matrix self.data. This is later used to create a csv 
        file.'''
        for i, network in enumerate(self.population):
            data_point = {
                'generation': generation,
                'network_id': i,
                'parent_id': network.parent_id if hasattr(network, 'parent_id') else None,
                'intelligence': self.population_intelligence[i],
                'cognitive_nodes': self.population_cognitive_nodes[i],
                'context_nodes': self.population_context_nodes[i],
                'fitness': self.population_fitness[i],
                'first_round_cooperation': network.first_round,
                'number_of_rounds': self.number_of_rounds[i],
                'sum_of_payoffs': self.sum_of_payoffs[i],
                'total_cooperations': self.total_cooperations,
                'total_defections': self.total_defections,
                'individual_cooperations': self.population_cooperations[i],
                'individual_defections': self.population_defections[i],
                'reproduction_count': 0,
                'reproduction_probability': self.reproduction_probabilities[i],
                'best_strategy': None,
                'sse': None}
            self.data.append(data_point)

    def save_data(self):
        '''Function that saves the data from the self.data matrix into a csv file. This allows for easier plotting in the future without
        having to run long simulations every time.'''
        keys = self.data[0].keys()
        with open(csv_filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.data)
            
    def reproduction(self):
        '''Function that creates a new generation based on the selected reproduction method.'''
        if reproduction_method == "asexual_TE":
            self.reproduction_asexual_TE()
        elif reproduction_method == "sexual_TE":
            self.reproduction_sexual_TE()
        elif reproduction_method == "proportionate_selection":
            self.reproduction_proportionate_selection()
        else:
            raise ValueError("Unknown reproduction method.")

    def reproduction_asexual_TE(self):
        num_parents = number_of_networks // 5
        num_children = number_of_networks // 5

        parent_indices = np.random.choice(number_of_networks, size=num_parents, replace=False, p=self.reproduction_probabilities)
        child_indices = np.random.choice(list(set(range(number_of_networks)) - set(parent_indices)), size=num_children, replace=False)

        new_population = [None] * number_of_networks

        for index in parent_indices:
            new_population[index] = self.population[index]

        for parent_idx, child_idx in zip(parent_indices, child_indices):
            new_population[child_idx] = NeuralNetwork.NeuralNetwork()
            new_population[child_idx].copy(self.population[parent_idx])
            new_population[child_idx].mutation(mutation_value_probability, mutation_structure_probability)
            new_population[child_idx].parent_id = parent_idx
            self.reproduction_counts[parent_idx] += 1

        remaining_indices = list(set(range(number_of_networks)) - set(parent_indices) - set(child_indices))
        for index in remaining_indices:
            new_population[index] = self.population[index]

        self.reset_population(new_population)
        
    def reproduction_sexual_TE(self):
        num_parents = number_of_networks // 5
        num_children = number_of_networks // 5

        parent_indices = np.random.choice(number_of_networks, size=num_parents, replace=False, p=self.reproduction_probabilities)
        initial_child_indices = list(np.random.choice(list(set(range(number_of_networks)) - set(parent_indices)), size=num_children, replace=False))

        child_indices = initial_child_indices.copy()

        new_population = [None] * number_of_networks

        for index in parent_indices:
            new_population[index] = self.population[index]

        for _ in range(num_children):
            parent1_idx, parent2_idx = np.random.choice(parent_indices, size=2, replace=False)
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            child = NeuralNetwork.NeuralNetwork()
            child.crossover(parent1, parent2)
            child.mutation(mutation_value_probability, mutation_structure_probability)
            new_population[child_indices.pop()] = child

        remaining_indices = list(set(range(number_of_networks)) - set(parent_indices) - set(initial_child_indices))
        for index in remaining_indices:
            new_population[index] = self.population[index]

        self.reset_population(new_population)

    def reproduction_proportionate_selection(self):
        global number_of_networks

        self.reproduction_counts = list(np.zeros(number_of_networks))

        index_networks_reproducing = np.random.choice(number_of_networks, size=number_of_networks * offspring_rate, p=self.reproduction_probabilities)

        new_population = [NeuralNetwork.NeuralNetwork() for i in range(number_of_networks * offspring_rate)]

        for i in range(number_of_networks * offspring_rate):
            index = index_networks_reproducing[i]
            new_population[i].copy(self.population[index])
            new_population[i].mutation(mutation_value_probability, mutation_structure_probability)
            new_population[i].parent_id = index
            self.reproduction_counts[index] += 1

        number_of_networks *= offspring_rate
        self.reset_population(new_population)
        
    def reset_population(self, new_population):
        self.population = new_population
        self.number_of_rounds = list(np.zeros(number_of_networks))
        self.sum_of_payoffs = list(np.zeros(number_of_networks))
        self.population_cognitive_nodes = []
        self.population_context_nodes = []
        self.population_intelligence = self.count_nodes()
        self.population_fitness = []
        self.total_cooperations = 0
        self.total_defections = 0
        self.population_cooperations = list(np.zeros(number_of_networks))
        self.population_defections = list(np.zeros(number_of_networks))
        self.reproduction_data.append(list(self.reproduction_counts))

    def merge_reproduction_data(self):
        '''Function to merge stored reproduction data into the main data list.'''
        for generation in range(len(self.reproduction_data)):
            for i in range(len(self.reproduction_data[generation])):
                self.data[generation * len(self.reproduction_data[0]) + i]['reproduction_count'] = self.reproduction_data[generation][i]

    def initialise_test_set(self):
        '''Function that generates a test set to be compared against the networks. "The test-set consisted of the moves of theoretical 
        opponents that are characterised simply by their probability of cooperating in each round of the prisoner's dilemma, this
        probability varying from 0 to 1 in steps of 0.25. The sequence of moves of each of these opponents for 5 replicates of a 20 
        round IPD or ISD was generated."'''
        probabilities_of_cooperating = [0, 0.25, 0.5, 0.75, 1]
        test_set = []
        for probability in probabilities_of_cooperating:
            for replicate in range(5):
                test_set.append([random.random() < probability for round in range(20)])
        return test_set

    def pure_strategy(self):
        strategies = [
            PureStrategy('tit_for_tat'),
            PureStrategy('tit_for_two_tats'),
            PureStrategy('pavlov'),
            PureStrategy('always_defect'),
            PureStrategy('always_cooperate')]

        pure_strategies = []

        for strategy in strategies:
            strategy_list = []
            
            for test_sequences in self.test_set:
                pure_behaviour = []

                for round_number in range(len(test_sequences)):
                    if round_number > 1:
                        opponent_last_move = test_sequences[round_number - 1]
                        opponent_second_last_move = test_sequences[round_number - 2]
                        self_last_move = pure_behaviour[-1]
                    
                    elif round_number == 1:
                        opponent_last_move = test_sequences[round_number - 1]
                        opponent_second_last_move = True
                        self_last_move = pure_behaviour[-1]
                        
                    else:
                        opponent_last_move = True
                        opponent_second_last_move = True
                        self_last_move = True
                    
                    strategy_move = strategy.play(round_number, self_last_move, opponent_last_move, opponent_second_last_move)
                    pure_behaviour.append(int(strategy_move))
                
                strategy_list.append(pure_behaviour)

            pure_strategies.append(strategy_list)
        
        return pure_strategies
            
    def characterise_networks(self, generation, payoff_game, pure_strategies):
        strategies = {0: 'tit_for_tat', 1: 'tit_for_two_tats', 2: 'pavlov', 3: 'always_defect', 4: 'always_cooperate'}
        
        for i, network in enumerate(self.population):
            network_behaviours = []
            for j, test_sequences in enumerate(self.test_set):
                network_behaviour = [int(network.first_round)]
                for round_number in range(1, len(test_sequences)):
                    self_payoff, test_payoff = self.assign_payoffs(network_behaviour[-1], test_sequences[round_number - 1], payoff_game)
                    network_move = network.forward(self_payoff, test_payoff)
                    network_behaviour.append(int(network_move))
                network_behaviours.append(network_behaviour)
            
            min_sse = float('inf')
            
            for strategy_type in range(5):
                sse = 0
                for j in range(len(self.test_set)):
                    network_behaviour = network_behaviours[j]
                    pure_behaviour = pure_strategies[strategy_type][j]
                    for k in range(len(network_behaviour)):
                        sse += (pure_behaviour[k] - network_behaviour[k])**2
                        
                if sse < min_sse:
                    min_sse = sse
                    best_strategy = strategies[strategy_type]

            self.strategy_data.append({'generation': generation, 'network_id': i, 'best_strategy': best_strategy, 'sse': min_sse})

            
    def merge_strategy_data(self):
        for strategy in self.strategy_data:
            generation = strategy['generation']
            network_id = strategy['network_id']
            self.data[generation * number_of_networks + network_id]['best_strategy'] = strategy['best_strategy']
            self.data[generation * number_of_networks + network_id]['sse'] = strategy['sse']

class PureStrategy:
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name

    def play(self, round_number, self_last_move, opponent_last_move, opponent_second_last_move):
        '''Function that, depending on the strategy and given the movement of the opponent in the previous round, returns either cooperation
        or defection in the next round.
        
            INPUT:
                round_number: integer corresponding to the round that the individuals are currently playing.
                opponent_last_move: boolean indicating if the opponent cooperated ('True') or defected ('False') in the previous round.
                
            OUTPUT:
                boolean indicating if the individual will cooperate ('True') or not ('False') in the following round based on its strategy.
        '''
        if self.strategy_name == 'always_defect':
            # We always defect.
            return False
        
        elif self.strategy_name == 'always_cooperate':
            # We always cooperate.
            return True
        
        elif self.strategy_name == 'tit_for_tat':
            # We cooperate in the first round, then we mimic the opponent's last move.
            return opponent_last_move if round_number > 0 else True
        
        elif self.strategy_name == 'tit_for_two_tats':
            if round_number == 0 or round_number == 1:
                # We cooperate in the first two rounds.
                return True
            else:
                # We cooperate unless the opponent defected in both last rounds.
                return not (not opponent_last_move and not opponent_second_last_move)
            
        elif self.strategy_name == 'pavlov':
            if round_number == 0:
                # We cooperate in the first round.
                return True
            else:
                if (self_last_move and opponent_last_move) or (not self_last_move and not opponent_last_move):
                    # We cooperate if both players did the same in the previous round.
                    return True
                else:
                    # We defect if the players did different things in the previous round.
                    return False

if __name__ == '__main__':
    simulation = Simulation()
    simulation.run(game, number_of_generations)
    simulation.merge_strategy_data()
    simulation.merge_reproduction_data()
    simulation.save_data()
