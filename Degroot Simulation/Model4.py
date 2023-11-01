import networkx as nx
import random
import numpy as np
from abc import abstractmethod
import scipy.stats as stats

class Stub_Agent:
    def __init__(self, Doc_id, agent_type, initial_opinion_ratio):
        if random.random() < initial_opinion_ratio:
            self.opinion = 0
        else:
            self.opinion = 1
        self.agent_type = agent_type 

class Doc:
    def __init__(self, Doc_id, agent_type):
        self.agent_id = Doc_id
        self.opinion = 1
        self.agent_type = agent_type  

class Norm_Agent:
    def __init__(self, NonDoc_id,  agent_type):
        self.agent_id = NonDoc_id
        lower, upper = 0, 1
        mu, sigma = 0.37, 0.25
        self.opinion = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        self.agent_type = agent_type  

class ERNetwork:

    def __init__(self, num_Doc, num_NonDoc, prob, initial_doc_ratio, initial_opinion_ratio, random_state) -> None:
        self.graph = nx.erdos_renyi_graph(num_Doc+num_NonDoc, prob, seed=random_state, directed=True)
        self.initial_doc_ratio = initial_doc_ratio
        self.initial_opinion_ratio = initial_opinion_ratio
        self.agents = {}
        self.num_of_nodes = num_Doc+num_NonDoc
        self.degree = self.graph.degree

        for node in self.graph.nodes():
            self.graph.add_edge(node, node)
            if random.random() < num_Doc/(num_Doc+num_NonDoc):
                if random.random() < self.initial_doc_ratio:
                    agent = Stub_Agent(node, "Stubborn", self.initial_opinion_ratio)
                    self.agents[node] = agent
                else:
                    agent = Doc(node, "Doc")
                    self.agents[node] = agent
            else:
                agent = Norm_Agent(node, "Normal")
                self.agents[node] = agent
        
        self.calculate_edge_weights()
        self.weight_matrix = self.generate_weight_matrix() 
    
    def calculate_edge_weights(self):

        for edge in self.graph.edges():
            node1, node2 = edge
            agent_type1 = self.agents[node1].agent_type
            agent_type2 = self.agents[node2].agent_type
            # num_neighbor = self.graph[node1].degree
            num_Stub = 0
            num_Doc = 0
            num_Norm = 0
            for neighbor in self.graph.neighbors(node1):
                if self.agents[neighbor].agent_type == "Stubborn":
                    num_Stub = num_Stub + 1
                elif self.agents[neighbor].agent_type == "Doc":
                    num_Doc = num_Doc + 1
                else:
                    num_Norm = num_Norm + 1
            if agent_type1 == "Stubborn" or agent_type1 == "Doc":
                if node1 == node2:
                    weight = 1
                else:
                    weight = 0
            else:                
            # Assign weights based on agent types
                if node1 != node2:
                    if agent_type2 =="Stubborn" or agent_type2 == "Normal":
                        weight = 0.09/(0.19*num_Doc + 0.09*(num_Stub+num_Norm) + 0.8)
                    else:
                        weight = 0.19/(0.19*num_Doc + 0.09*(num_Stub+num_Norm) + 0.8)
                else:                         
                    weight = 0.8/(0.19*num_Doc + 0.09*(num_Stub+num_Norm) + 0.8)
        
            self.graph[node1][node2]['weight'] = weight
    
    def generate_weight_matrix(self):

        weight_matrix = np.zeros((self.num_of_nodes, self.num_of_nodes))

        for edge in self.graph.edges():
            node1, node2 = edge
            weight_matrix[node1][node2] = self.graph[node1][node2]['weight']

        return weight_matrix


    def update_opinions(self):
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            
            weighted_sum = sum(self.weight_matrix[node][n] * self.agents[n].opinion for n in neighbors)
            new_opinion = weighted_sum / sum(self.weight_matrix[node][n] for n in neighbors)
            self.agents[node].opinion = new_opinion

    def consensus_reached(self, tolerance=0.01):
        opinions = [self.agents[node].opinion for node in self.graph.nodes()]
        consensus_opinion = sum(opinions) / len(opinions)
        return all(abs(opinion - consensus_opinion) <= tolerance for opinion in opinions)
