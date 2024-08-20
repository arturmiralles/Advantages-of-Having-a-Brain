import random
import numpy as np
import scipy

sigma_0 = 1

class InnerNode:
    '''This class represents a cognitive node, which has a threshold value and optionally a context node.'''
    
    def __init__(self):
        '''
        CHARACTERISTICS:
            cognitive_threshold: threshold of the cognitive node.
            context_node: boolean indicating whether the cognitive node has an associated context node.
            context_value: value stored in the associated context node.
            weight_cognitive_context: weight of the link between the context node and its cognitive node.
        '''
        self.cognitive_threshold = np.random.normal(0, sigma_0)
        self.context_node = False
        self.context_value = 0.0
        self.weight_cognitive_context = 0.0
    
    def add_context_node(self):
        '''Function that creates a context node for the given cognitive node.'''
        assert not self.context_node
        self.context_node = True
        self.weight_cognitive_context = np.random.normal(0, sigma_0)
        
    def delete_context_node(self):
        '''Function that deletes the context node of the given cognitive node and restores its values.'''
        assert self.context_node
        self.context_node = False
        self.context_value = 0.0
        self.weight_cognitive_context = 0.0
        
    def copy(self, other):
        '''
        Function that copies all the attributes from another InnerNode.
        
        INPUT:
            self: inner node to which we want to copy the information.
            other: inner node from which we want to copy the information.
        '''
        self.cognitive_threshold = other.cognitive_threshold
        self.context_node = other.context_node
        self.weight_cognitive_context = other.weight_cognitive_context
        
    def forward(self, input):
        '''This method calculates the output of the InnerNode. If it has a context node, it incorporates the
        context node's value into the input.'''
        if self.context_node:
            node_value = input + self.context_value * self.weight_cognitive_context + self.cognitive_threshold
            output = scipy.special.expit(node_value)
            self.context_value = output
        else:
            output = scipy.special.expit(input + self.cognitive_threshold)
        return output
    
    
class NeuralNetwork:
    '''This class represents the entire neural network. Each individual in the simulations is represented by a neural
    network, with their hidden layer varying in structure.
    
    STRUCTURE:
        -A network always possesses two input nodes, with inputs of the individual's and their opponent's scores in the 
        previous round of the game.
        
        -A network always possesses one output node giving the probability that the individual will cooperate in the 
        current round.

        -Individuals can then possess between 0 and 10 cognitive nodes.
        
        -Each cognitive node has the potential to have a context node with a recurrent connection.
    
    INTERPRETATION:
        The addition of extra cognitive nodes gives networks the potential to perform complex computation based on payoffs
        by increasing the dimensions of internal representation of the network.
    
        The addition of context nodes gives the potential for the use of longer-term memory of previous interactions in 
        these computations.
        
    As the network would not function without an input, each individual is given an additional trait, which encodes
    whether they cooperate or defect on the first round of any interaction. If an individual does not possess any 
    cognitive nodes, this trait decides their behaviour in all interactions.'''

    def __init__(self):
        '''
        CHARACTERISTICS:
            first_round: 'True' means the individual will cooperate on the first round. 'False' means it will defect.
            inner_nodes: list that will contain the inner nodes, initially empty.
            output_threshold: output node's threshold, which is inherited as a genetic variable along with all weights in the network, the
                network structure and the individual's behaviour on the first round of an interaction.
            weights_self_payoff_inner_node: list of weights linking the payoff of the individual to the cognitive nodes.
            weights_opponent_payoff_inner_node: list of weights linking the payoff of the opponent to the cognitive nodes.
            weights_inner_node_output: list of weights linking the cognitive node to the output.
            context_node_count: integer indicating the number of context nodes in the network.
            initialise_inner_nodes: function that initialises the inner nodes of the network and adds them to the inner_nodes list.
        '''
        # We initialise the first move of the network as either cooperation or defection.
        self.first_round = random.choice([True, False])
        
        # We initialise a list to hold the inner node objects.
        self.inner_nodes = []
        
        # We initialise the output threshold with a random value from a normal distribution.
        self.output_threshold = np.random.normal(0, sigma_0)
        
        # We initialise lists to hold the weights for the connections between nodes.
        self.weights_self_payoff_inner_node = []
        self.weights_opponent_payoff_inner_node = []
        self.weights_inner_node_output = []
        
        # We initialise the count for the number of context nodes.
        self.context_node_count = 0
        
        # We initialise the inner nodes with a random structure.
        self.initialise_inner_nodes()
        
    def initialise_inner_nodes(self):
        '''Function that initialises the inner nodes of the given network. At the beginning of a simulation run, individuals are randomly 
        assigned network structures of 0, 1, 2 or 3 cognitive nodes. The number of cognitive nodes throughout the simulation is always 
        between 0 and 10.'''
        # We randomly choose 0, 1, 2 or 3 cognitive nodes.
        num_cognitive_nodes = random.randint(0, 3)
        
        # We add the chosen number of cognitive nodes to the network.
        for num in range(num_cognitive_nodes):
            self.add_cognitive_node()
            
    def add_cognitive_node(self):
        '''Function that adds a cognitive node to the neural network, with a threshold and with random weights linking it to the inputs and 
        to the other inner nodes. These weights are added to the lists containing all other inner nodes' weights, with the position
        in the list corresponding to the position of the node in the inner_nodes list.'''
        # We create a new cognitive node.
        node = InnerNode()
        
        # We add the new node to the list of inner nodes.
        self.inner_nodes.append(node)
        
        # We initialise the weights for the new node.
        self.weights_self_payoff_inner_node.append(np.random.normal(0, sigma_0))
        self.weights_opponent_payoff_inner_node.append(np.random.normal(0, sigma_0))
        self.weights_inner_node_output.append(np.random.normal(0, sigma_0))
        
    def delete_cognitive_node(self, chosen_cognitive_node):
        '''Function that selects a random cognitive node from the neural network and deletes it, together with its corresponding weights
        and context node (if it has one).'''
        # We check if the node that we are going to delete has a context node assigned to it.
        if self.inner_nodes[chosen_cognitive_node].context_node:
            # We decrease the network's context node count.
            self.context_node_count -= 1
            # We delete the context node.
            self.inner_nodes[chosen_cognitive_node].delete_context_node()
            
        # We delete the cognitive node and its associated weights.
        del self.inner_nodes[chosen_cognitive_node]
        del self.weights_self_payoff_inner_node[chosen_cognitive_node]
        del self.weights_opponent_payoff_inner_node[chosen_cognitive_node]
        del self.weights_inner_node_output[chosen_cognitive_node]

    def mutation(self, mutation_value_probability, mutation_structure_probability):
        '''This function mutates the network by potentially modifying values, such as weights or threshold values, and possibly adding or
        removing nodes. Each potential mutation happens with a predefined probability:
            mutation_value_probability: probability that the network modifies the weights or threshold values.
            mutation_structure_probability: probability that the network modifies its structure by adding ot removing nodes).
        Mutations in a network structure lead to the addition or removal of a node with equal probability.'''
        
        # We mutate the first move of the network with a certain probability.
        if random.random() < mutation_value_probability:
            self.first_round = not self.first_round
            
        # We iterate over all the inner nodes of the network and potentially mutate their weights and thresholds.
        for i in range(len(self.inner_nodes)):
            if random.random() < mutation_value_probability: #Mutate all of them.
                self.weights_self_payoff_inner_node[i] += np.random.normal(0, 0.5)
            if random.random() < mutation_value_probability:
                self.weights_opponent_payoff_inner_node[i] += np.random.normal(0, 0.5)
            if random.random() < mutation_value_probability:
                self.weights_inner_node_output[i] += np.random.normal(0, 0.5)
            if random.random() < mutation_value_probability:
                self.inner_nodes[i].cognitive_threshold += np.random.normal(0, 0.5)
            if random.random() < mutation_value_probability:
                self.inner_nodes[i].weight_cognitive_context += np.random.normal(0, 0.5)
                
        # We mutate the output threshold with a certain probability.
        if random.random() < mutation_value_probability:
            self.output_threshold += np.random.normal(0, 0.5)
            
        # We mutate the structure of the network by adding or removing nodes with a certain probability.
        if random.random() < mutation_structure_probability:
            if random.choice([True, False]):
                self.add_node()
            else:
                if len(self.inner_nodes) != 0:
                    self.delete_node()
                    
    def add_node(self):
        '''Function that, if the network allows it (i.e. the network has not already reached the maximum number of nodes allowed), adds a 
        new node. The choice between context and cognitive nodes is random if both choices are allowed.'''
        context_node_count = self.context_node_count
        cognitive_node_count = len(self.inner_nodes)   
                
        # We check if the number of cognitive nodes equals the number of context nodes.
        if cognitive_node_count == context_node_count:
            if cognitive_node_count == 10:
                # If we have reached the maximum number of context and cognitive nodes allowed in the network, do nothing.
                return

            else:
                # If all cognitive nodes have already a context node associated to them but we have not reached the maximum number of
                # cognitive nodes allowed, add a new one.
                self.add_cognitive_node()
                
        # If we have less context nodes than cognitive nodes.
        else:
            if cognitive_node_count != 10:
                # If we have not reached the maximum number of cognitive nodes allowed in the network, there is a 50-50 chance of adding
                # a cognitive or a context node.
                choose_context = random.choice([True, False])
                
                if choose_context:
                    # If we have chosen to add a context node, we check for cognitive nodes that do not already possess a context node
                    # and choose a random one.
                    not_context = [node for node in self.inner_nodes if not node.context_node]
                    chosen_cognitive_node = random.choice(not_context)
                    
                    # We add a context node to the chosen cognitive node.
                    chosen_cognitive_node.add_context_node()
                    
                    # We increase the network's context node count.
                    self.context_node_count += 1
                    
                else:
                    # If we have chosen not to add a context node, we add a cognitive node.
                    self.add_cognitive_node()
                    
            else:
                # If we have reached the maximum number of cognitive nodes allowed in the network but we still have space for context nodes,
                # we check for cognitive nodes without associated context nodes and choose a random one.
                not_context = [node for node in self.inner_nodes if not node.context_node]
                chosen_cognitive_node = random.choice(not_context)
                
                # We add a context node to the chosen cognitive node.
                chosen_cognitive_node.add_context_node()
                
                # We increase the network's context node count.
                self.context_node_count += 1
        
    def delete_node(self):
        '''Function that, unless the network already has zero inner nodes, removes a node. The choice between context and cognitive nodes 
        is random if both choices are allowed.'''
        
        # We have 50-50 chance of deleting a cognitive or a context node.
        remove_context = random.choice([True, False])
        
        if remove_context:
            if self.context_node_count > 0:
                # If we choose to delete a context node, we check for cognitive nodes with associated context nodes and choose a random one.
                context_nodes = [node for node in self.inner_nodes if node.context_node]
                chosen_context_node = random.choice(context_nodes)
                
                # We delete the chosen context node.
                chosen_context_node.delete_context_node()
                
                # We decrease the network's context node count.
                self.context_node_count -= 1
            
        else:
            if len(self.inner_nodes) > 0:
                # If we have chosen to delete a cognitive node, we select a random one and delete it.
                chosen_cognitive_node = random.randint(0, len(self.inner_nodes) - 1)
                self.delete_cognitive_node(chosen_cognitive_node)
            
    def copy(self, other):
        '''
        Function that copies all attributes from another neural network.
        
        INPUT:
            self: neural network to which we want to copy the information.
            other: neural network from which we want to copy the information.
        '''        
        # We copy the basic properties of the network.
        self.first_round = other.first_round
        self.output_threshold = other.output_threshold
        self.context_node_count = other.context_node_count

        # We initialise lists to which we will append other properties of the network.
        self.inner_nodes = []
        self.weights_self_payoff_inner_node = []
        self.weights_opponent_payoff_inner_node = []
        self.weights_inner_node_output = []
        
        # We copy all the inner nodes to the new network.
        for node in other.inner_nodes:
            new_node = InnerNode()
            new_node.copy(node)
            self.inner_nodes.append(new_node)

        # We copy all the weights form the other network to the new one.
        for weight in other.weights_self_payoff_inner_node:
            self.weights_self_payoff_inner_node.append(weight)

        for weight in other.weights_opponent_payoff_inner_node:
            self.weights_opponent_payoff_inner_node.append(weight)
        
        for weight in other.weights_inner_node_output:
            self.weights_inner_node_output.append(weight)
                
    def forward(self, self_payoff, opponent_payoff):
        '''
       "Computation in the network is implemented via synchronous updating of nodes. The value of each input node is passed to each of the
        network's cognitive nodes, multiplied by the weight linking the two nodes. Each cognitive node is also passed the current value of 
        their associated context node (if it possesses one) multiplied by the weight linking the two nodes. The cognitive nodes sum across
        all of the weighted values that they receive and pass this value through a sigmoidal squashing function, resulting in a value
        between 0 and 1, analogous to a probability of activation. All context nodes are then passed the value of their associated cognitive
        nodes. [...] Finally, the values at all cognitive nodes are then passed to the output node (multiplied by their weights), summed and
        again passed through a sigmoidal squashing function. This output gives the probability that the individual will cooperate in the
        current round."
        
        INPUTS:
            self: individual whose probability we are computing.
            self_payoff: individual's scores in the previous round of the game.
            opponent_payoff: opponent's scores in the previous round of the game.

        OUTPUT:
            a boolean, with 'True' indicating that the individual will cooperate and 'False' indicating it will defect. It follows from
            computing the probability that the individual ('self') will cooperate in the current round. Since the sigmoidal function 
            asymptotes to 0 at -inf and 1 at +inf, there will always be inherent noise in the network's probabilistic decision.
            
            It is calculated using the sigmoid function applied to (x + output_threshold), where
                x: The input to the node, which is simply the sum of the states of each of its connected nodes multiplied by
                the weights of those connections.
                output_threshold: The node's threshold, which is inherited as a genetic variable along with all weights in the network, the
                network structure and the individual's behaviour on the first round of an interaction.
        '''
        self_payoff = np.copy(self_payoff)
        opponent_payoff = np.copy(opponent_payoff)
        
        # If there are no inner nodes, the decision to either cooperate or defect is based on the first round behaviour.
        if len(self.inner_nodes) == 0:
            return self.first_round
        
        x = 0.0
        
        # If there are inner nodes, we perform the network's calculations by passing the inputs through all cognitive nodes.
        for i, inner_node in enumerate(self.inner_nodes):
            self_input = self_payoff * self.weights_self_payoff_inner_node[i]
            opponent_input = opponent_payoff * self.weights_opponent_payoff_inner_node[i]
            x += inner_node.forward(self_input + opponent_input) * self.weights_inner_node_output[i]
        
        # We compute the cooperation probability using the sigmoid function, which will map the previous output to a number between 0 and 1.
        cooperate_probability = scipy.special.expit(x + self.output_threshold)
        
        # Based on the cooperation probability, the network makes a choice.
        decision = random.random() < cooperate_probability

        return decision