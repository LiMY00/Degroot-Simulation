import networkx as nx
import random
import numpy as np
from abc import abstractmethod
import scipy.stats as stats
# import cugraph as cnx

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
        self.opinion = random.random()
        # lower, upper = 0, 1
        # mu, sigma = 0.37, 0.25
        # self.opinion = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        self.agent_type = agent_type  

class SBMNetwork:

    def __init__(self, blocks, prob, num_doc, num_bot, num_titled_doc, initial_opinion_ratio) -> None:
        self.graph = nx.stochastic_block_model(blocks, prob, nodelist=None, directed=True, selfloops=True, sparse=True)
        # self.initial_doc_ratio = initial_doc_ratio
        self.initial_opinion_ratio = initial_opinion_ratio
        self.agents = {}
        self.block = blocks
        self.block_nodes = [list(range(sum(blocks[:i]), sum(blocks[:i + 1]))) for i in range(len(blocks))]
        self.degree = self.graph.degree
        self.num_of_nodes = np.sum(self.block)
        self.nodes = []
        for node in self.graph.nodes():
            self.nodes.append(node)


        # doc_block = 0
        # bot_block = 1
        # self.all_nodes = list(self.graph.nodes)

        self.doc_nodes = random.sample(self.nodes, num_doc)
        self.doc_nodes_title = random.sample(self.doc_nodes, num_titled_doc)

        for node in self.graph.nodes():
            if not self.graph.has_edge(node, node):
                self.graph.add_edge(node, node)
            if node in self.doc_nodes:
                if node in self.doc_nodes_title:
                    agent = Doc(node, "Doctor")
                    self.agents[node] = agent
                else:
                    agent = Doc(node, "Notitle_doctor")
                    self.agents[node] = agent
                
            else:
                agent = Norm_Agent(node, "Non_Doc")
                self.agents[node] = agent
        # stubborn_nodes = random.sample(self.block_nodes[bot_block], num_bot)
        # for node in self.graph.nodes():
        #     if node in stubborn_nodes:
        #         agent = Stub_Agent(node, "Stubborn", self.initial_opinion_ratio)
        #         self.agents[node] = agent
        #     if node in doc_nodes:
        #         agent = Doc(node, "Doc")
        #         self.agents[node] = agent
        #     else:
        #         agent = Norm_Agent(node, "Normal")
        #         self.agents[node] = agent
            


        # for node in self.graph.nodes():
        #     self.graph.add_edge(node, node)
        #     if random.random() < num_Doc/(num_Doc+num_NonDoc):
        #         if random.random() < self.initial_doc_ratio:
        #             agent = Stub_Agent(node, "Stubborn", self.initial_opinion_ratio)
        #             self.agents[node] = agent
        #         else:
        #             agent = Doc(node, "Doc")
        #             self.agents[node] = agent
        #     else:
        #         agent = Norm_Agent(node, "Normal")
        #         self.agents[node] = agent
        
        self.calculate_edge_weights()
        self.weight_matrix = self.generate_weight_matrix() 
    
    def calculate_edge_weights(self):
        for edge in self.graph.edges():
            node1, node2 = edge
            agent_type1 = self.agents[node1].agent_type
            agent_type2 = self.agents[node2].agent_type
            num_DocNeighbors = 0
            num_NonDocNeighbors = 0
            for neighbor in self.graph.neighbors(node1):
                if self.agents[neighbor].agent_type == "Doc" and neighbor != node1:
                    num_DocNeighbors = num_DocNeighbors + 1
                elif self.agents[neighbor].agent_type == "Non_Doc" and "Notitle_doctor" and neighbor != node1:
                    num_NonDocNeighbors = num_NonDocNeighbors + 1
            if agent_type1 == "Doctor" or agent_type1 == "Notitle_doctor":
                if node1 == node2:
                    weight = 1
                # elif agent_type2 == "Doctor" and num_DocNeighbors !=0:
                #     weight = 0.4/num_DocNeighbors
                else:
                    weight = 0
            else:  
            # Assign weights based on agent types
                if node1 != node2:
                    if agent_type2 == "Non_Doc" or agent_type2 == "Notitle_doctor":
                        weight = 0.003/(0.186*num_DocNeighbors + 0.003*num_NonDocNeighbors+0.906)
                    else:
                        weight = 0.186/(0.186*num_DocNeighbors + 0.003*num_NonDocNeighbors+0.906)
                else:                         
                    weight = 0.906/(0.186*num_DocNeighbors + 0.003*num_NonDocNeighbors + 0.906)
        
            self.graph[node1][node2]['weight'] = weight
    # def calculate_edge_weights(self):

    #     for edge in self.graph.edges():
    #         node1, node2 = edge
    #         agent_type1 = self.agents[node1].agent_type
    #         agent_type2 = self.agents[node2].agent_type
    #         # num_neighbor = self.graph[node1].degree
    #         num_Stub = 0
    #         num_Doc = 0
    #         num_Norm = 0
    #         for neighbor in self.graph.neighbors(node1):
    #             if self.agents[neighbor].agent_type == "Stubborn":
    #                 num_Stub = num_Stub + 1
    #             elif self.agents[neighbor].agent_type == "Doc":
    #                 num_Doc = num_Doc + 1
    #             else:
    #                 num_Norm = num_Norm + 1
    #         if agent_type1 == "Stubborn" or agent_type1 == "Doc":
    #             if node1 == node2:
    #                 weight = 1
    #             else:
    #                 weight = 0
    #         else:                
    #         # Assign weights based on agent types
    #             if node1 != node2:
    #                 if agent_type2 =="Stubborn" or agent_type2 == "Normal":
    #                     weight = 0.003/(0.186*num_Doc + 0.003*(num_Stub+num_Norm) + 0.906)
    #                 else:
    #                     weight = 0.186/(0.186*num_Doc + 0.003*(num_Stub+num_Norm) + 0.906)
    #             else:                         
    #                 weight = 0.906/(0.186*num_Doc + 0.003*(num_Stub+num_Norm) + 0.906)
        
    #         self.graph[node1][node2]['weight'] = weight
    
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
