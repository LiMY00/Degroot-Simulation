import networkx as nx
import random
import numpy as np
from abc import abstractmethod
import scipy.stats as stats

# Initialize the class for doctor agents
class Doctor:
    def __init__(self, Doc_id, agent_type, initial_opinion_ratio):
        self.agent_id = Doc_id
        if random.random() < initial_opinion_ratio:
            self.opinion = 0
        else:
            self.opinion = 1
        self.agent_type = agent_type  

# Initialize the class for common user agents
class Non_Doc:
    def __init__(self, NonDoc_id,  agent_type):
        self.agent_id = NonDoc_id
        lower, upper = 0, 1
        mu, sigma = 0.54, 0.29
        self.opinion = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        # self.opinion = np.random.normal(mu, sigma)
        self.agent_type = agent_type  

# Initialize the class for the ER network environment
class ERNetwork:
    def __init__(self, num_Doc, num_NonDoc, num_titled_doc, prob, initial_opinion_ratio, random_state) -> None:
        self.graph = nx.erdos_renyi_graph(num_Doc+num_NonDoc, prob, seed=random_state, directed=True)
        self.initial_opinion_ratio = initial_opinion_ratio
        self.agents = {}
        self.num_of_nodes = num_Doc+num_NonDoc
        self.degree = self.graph.degree
        self.nodes = []
        for node in self.graph.nodes():
            self.nodes.append(node)

        # randomly sample the nodes that represent doctor agent from all the nodes on the network
        self.doc_nodes = random.sample(self.nodes, num_Doc)
        # randomly sample the nodes that represent doctor agents with titles from the doctor agent nodes
        self.doc_nodes_title = random.sample(self.doc_nodes, num_titled_doc)

       
        for node in self.graph.nodes():
            # Add a self-loop edge to every nodes without self-loop, in order to calculate the weight of self-influence
            if not self.graph.has_edge(node, node):
                self.graph.add_edge(node, node)

            # Create the agents of every node
            if node in self.doc_nodes:
                if node in self.doc_nodes_title:
                    agent = Doctor(node, "Doctor", initial_opinion_ratio)
                    self.agents[node] = agent
                else:
                    agent = Doctor(node, "Notitle_doctor", initial_opinion_ratio)
                    self.agents[node] = agent
                
            else:
                agent = Non_Doc(node, "Non_Doc")
                self.agents[node] = agent
        
        self.calculate_edge_weights()
        self.weight_matrix = self.generate_weight_matrix() 
    
    def calculate_edge_weights(self):
        ''' This is a function to calculate the weight of edges based on the start nodes' connections and the type of its neighbors. '''

        for edge in self.graph.edges():
            node1, node2 = edge
            agent_type1 = self.agents[node1].agent_type
            agent_type2 = self.agents[node2].agent_type
            # calculate the number of agents in the neighbors
            num_DocNeighbors = 0
            num_NonDocNeighbors = 0
            # Calculate the numbers of doctors neighbors and non-doctor neighbors for node1
            for neighbor in self.graph.neighbors(node1):
                if self.agents[neighbor].agent_type == "Doctor" and neighbor != node1:
                    num_DocNeighbors = num_DocNeighbors + 1
                elif self.agents[neighbor].agent_type == "Non_Doc" and neighbor != node1:
                    num_NonDocNeighbors = num_NonDocNeighbors + 1
                elif self.agents[neighbor].agent_type == "Notitle_doctor":
                    num_NonDocNeighbors = num_NonDocNeighbors + 1

            # Assign weights based on agent types
            if agent_type1 == "Doctor" or agent_type1 == "Notitle_doctor":
                if node1 == node2:
                    weight = 1
                else:
                    weight = 0

            else:                
                if node1 != node2:
                    if agent_type2 =="Non_Doc" or agent_type2 == "Notitle_doctor":
                        weight = 0.003/(0.183*num_DocNeighbors + 0.003*num_NonDocNeighbors + 0.906)
                    else:
                        weight = 0.183/(0.183*num_DocNeighbors + 0.003*num_NonDocNeighbors + 0.906)
                else:                         
                    weight = 0.906/(0.183*num_DocNeighbors + 0.003*num_NonDocNeighbors + 0.906)
        
            self.graph[node1][node2]['weight'] = weight
    
    def generate_weight_matrix(self):
        ''' This is a function to generate a weight matrix based on the weight of every edge in the network.'''

        weight_matrix = np.zeros((self.num_of_nodes, self.num_of_nodes))
        for edge in self.graph.edges():
            node1, node2 = edge
            weight_matrix[node1][node2] = self.graph[node1][node2]['weight']

        return weight_matrix


    # def update_opinions(self):
    #     for node in self.graph.nodes():
    #         neighbors = list(self.graph.neighbors(node))
            
    #         weighted_sum = sum(self.weight_matrix[node][n] * self.agents[n].opinion for n in neighbors)
    #         new_opinion = weighted_sum / sum(self.weight_matrix[node][n] for n in neighbors)
    #         self.agents[node].opinion = new_opinion

    def consensus_reached(self, tolerance=0.01):
        ''' This is a function to examine if the opinions of all the agents on the network have converged. '''
        opinions = [self.agents[node].opinion for node in self.graph.nodes()]
        consensus_opinion = sum(opinions) / len(opinions)
        return all(abs(opinion - consensus_opinion) <= tolerance for opinion in opinions)
