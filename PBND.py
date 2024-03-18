import networkx as nx
import random
import numpy as np
import scipy.stats as stats
# import cugraph as cnx

class Influencer:
    def __init__(self, Doc_id, agent_type, initial_opinion):
        # if random.random() < initial_opinion_ratio:
        #     self.opinion = 0
        # else:
        #     self.opinion = 1
        self.opinion = initial_opinion
        self.agent_id = Doc_id
        self.agent_type = agent_type 

class Stub:
    def __init__(self, Doc_id, agent_type):
        # if random.random() < initial_opinion_ratio:
        #     self.opinion = 0
        # else:
        #     self.opinion = 1
        self.opinion = 0
        self.agent_id = Doc_id
        self.agent_type = agent_type 

class Stub_no_title:
    def __init__(self, Doc_id, agent_type):
        # if random.random() < initial_opinion_ratio:
        #     self.opinion = 0
        # else:
        #     self.opinion = 1
        self.opinion = 0
        self.agent_id = Doc_id
        self.agent_type = agent_type 

class Doc:
    def __init__(self, Doc_id, agent_type):
        self.agent_id = Doc_id
        self.opinion = 1
        self.agent_type = agent_type  

class Doc_no_title:
    def __init__(self, Doc_id, agent_type):
        self.agent_id = Doc_id
        self.opinion = 1
        self.agent_type = agent_type  

class Norm_Agent:
    def __init__(self, NonDoc_id,  agent_type):
        self.agent_id = NonDoc_id
        # self.opinion = random.random()
        lower, upper = 0, 1
        mu, sigma = 0.54, 0.29
        self.opinion = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        # self.opinion = np.random.normal(mu, sigma)
        self.agent_type = agent_type  

class PBNDNetwork:

    def __init__(self, num_I1, num_I2, num_doc_b1, num_doc_b2, num_stub_b1, num_stub_b2, num_ntitle_doc_b1, num_ntitle_stub, initial_opinion_ratio, threshold) -> None:
        self.graph = self.generate_PBN()
        # self.initial_doc_ratio = initial_doc_ratio
        self.initial_opinion_ratio = initial_opinion_ratio
        self.agents = {}
        self.neighbors = []
        # self.block = [num_nodes_per_community, num_nodes_per_community]
        # self.block_nodes = [list(range(sum(self.block[:i]), sum(self.block[:i + 1]))) for i in range(len(self.block))]
        # self.B1_list = []
        # self.B2_list = []
        
        # check if there is self-loop for every node, if not, add self-loop
        for node in self.graph.nodes():
            if not self.graph.has_edge(node, node):
                self.graph.add_edge(node, node)
            # if self.graph.nodes[node]['value'] == 0:
            #     self.B1_list.append(node)
            # else:
            #     self.B2_list.append(node)
        self.B1_nodes = [n for n, attr in self.graph.nodes(data=True) if attr['value'] == 0]
        self.B2_nodes = [n for n, attr in self.graph.nodes(data=True) if attr['value'] == 1]
        self.degrees_1 = self.graph.in_degree(self.B1_nodes)
        self.degrees_2 = self.graph.in_degree(self.B2_nodes)
        self.degree = self.graph.degree
        self.edges = self.graph.edges()
        for node in self.graph.nodes():
            self.neighbors.append(self.graph.neighbors(node))
        self.num_of_nodes = self.graph.number_of_nodes()

        # Block_1 = 1
        # Block_2 = 2
        I1 = sorted(self.degrees_1, key=lambda x: x[1], reverse=True)[:50]
        I2 = sorted(self.degrees_2, key=lambda x: x[1], reverse=True)[:50]
        influencer_b1 = sorted(self.degrees_1, key=lambda x: x[1], reverse=True)[:num_I1]
        influencer_b2 = sorted(self.degrees_2, key=lambda x: x[1], reverse=True)[:num_I2]
        self.influencer_b1 = [item[0] for item in influencer_b1]
        self.influencer_b2 = [item[0] for item in influencer_b2]
        # all_nodes = list(self.graph.nodes)

        # influencer_b1 = np.array(self.influencer1)[:,0]
        # influencer_b2 = np.array(self.influencer2)[:,0]
        low_degree_b1 = [x for x in self.B1_nodes if x not in I1]
        low_degree_b2 = [x for x in self.B2_nodes if x not in I2]
        self.doc_nodes_b1 = random.sample(low_degree_b1, num_doc_b1)
        self.doc_nodes_b2 = random.sample(low_degree_b2, num_doc_b2)
        self.doc_nodes_ntitle_b1 = random.sample(self.doc_nodes_b1, num_ntitle_doc_b1)
        self.doc_nodes_ntitle_b2 = random.sample(self.doc_nodes_b2, num_doc_b2)
        low_degree_b1_nd = [x for x in low_degree_b1 if x not in self.doc_nodes_b1]
        low_degree_b2_nd = [x for x in low_degree_b2 if x not in self.doc_nodes_b2]
        self.stub_nodes_b1 = random.sample(low_degree_b1_nd, num_stub_b1)
        self.stub_nodes_b2 = random.sample(low_degree_b2_nd, num_stub_b2)
        self.stub_nodes_ntitle_b1 = random.sample(self.stub_nodes_b1, 0)
        self.stub_nodes_ntitle_b2 = random.sample(self.stub_nodes_b2, num_ntitle_stub)
        for node in self.graph.nodes():

            if node in self.influencer_b2:
                agent = Influencer(node, "Influencer", 0)
                self.agents[node] = agent
            elif node in self.influencer_b1:
                agent = Influencer(node, "Influencer", 1)
                self.agents[node] = agent
            elif node in self.doc_nodes_b1 and not self.doc_nodes_ntitle_b1:
                agent = Doc(node, "Doc")
                self.agents[node] = agent
            elif node in self.doc_nodes_b1 and self.doc_nodes_ntitle_b1:
                agent = Doc_no_title(node, "Stubborn")
                self.agents[node] = agent
            elif node in self.doc_nodes_b2 and not self.doc_nodes_ntitle_b2:
                agent = Doc(node, "Doc")
                self.agents[node] = agent
            elif node in self.doc_nodes_b2 and self.doc_nodes_ntitle_b2:
                agent = Doc_no_title(node, "Stubborn")
                self.agents[node] = agent
            # elif node in self.doc_nodes_b2:
            #     agent = Doc(node, "Doc")
            #     self.agents[node] = agent
            elif node in self.stub_nodes_b1 and self.stub_nodes_ntitle_b1:
                agent = Stub_no_title(node, "Stubborn")
                self.agents[node] = agent
            elif node in self.stub_nodes_b1 and not self.stub_nodes_ntitle_b1:
                agent = Stub(node, "Doc")
                self.agents[node] = agent
            elif node in self.stub_nodes_b2 and self.stub_nodes_ntitle_b2:
                agent = Stub_no_title(node, "Stubborn")
                self.agents[node] = agent
            elif node in self.stub_nodes_b2 and not self.stub_nodes_ntitle_b2:
                agent = Stub(node, "Doc")
                self.agents[node] = agent
            # elif node in self.stub_nodes_b2:
            #     agent = Stub(node, "Stubborn")
            #     self.agents[node] = agent
            else:
                agent = Norm_Agent(node, "Normal")
                self.agents[node] = agent
            


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

        self.calculate_edge_weights(threshold)
        self.weight_matrix = self.generate_weight_matrix()
    
    def calculate_edge_weights(self, threshold):

        for edge in self.graph.edges():
            node1, node2 = edge
            # if node1 in self.stub_nodes_b2:
            #     print("yes")
            agent_type1 = self.agents[node1].agent_type
            agent_type2 = self.agents[node2].agent_type
            
            # num_neighbor = self.graph[node1].degree
            num_Stub = 0
            num_Doc = 0
            num_Norm = 0
            num_Influencer = 0
            
            for neighbor in self.graph.neighbors(node1):
                if self.agents[neighbor].agent_type == "Stubborn" and neighbor != node1:
                    num_Stub = num_Stub + 1
                elif self.agents[neighbor].agent_type == "Doc" and neighbor != node1:
                    num_Doc = num_Doc + 1
                elif self.agents[neighbor].agent_type == "Influencer" and neighbor != node1:
                    num_Influencer = num_Influencer + 1
                elif self.agents[neighbor].agent_type == "Normal" and neighbor != node1:
                    num_Norm = num_Norm + 1

            if agent_type1 == "Stubborn" or agent_type1 == "Doc" or agent_type1 == "Influencer":
                if node1 == node2:
                    weight = 1
                else:
                    weight = 0
            else:                
            # Assign weights based on agent types
                if self.graph.degree(node1)<threshold:
                    if node1 != node2:
                        if agent_type2 == "Normal" or agent_type2 == "Influencer":
                            weight = 0.003/(0.183*(num_Doc+num_Stub) + 0.003*(num_Influencer+num_Norm) + 0.906)
                        else:
                            weight = 0.183/(0.183*(num_Doc+num_Stub) + 0.003*(num_Influencer+num_Norm) + 0.906)
                    else:                         
                        weight = 0.906/(0.183*(num_Doc+num_Stub) + 0.003*(num_Influencer+num_Norm) + 0.906)
                else:
                    if node1 != node2:
                        if agent_type2 == "Normal" or agent_type2 == "Influencer" or "Stubborn":
                            weight = 0.003/(0.183*(num_Doc) + 0.003*(num_Influencer+num_Norm+num_Stub) + 0.906)
                        # elif agent_type2 == "Stubborn":
                        #     weight = 0
                        # if agent_type2 == "Normal" or agent_type2 == "Influencer":
                        #     weight = 0.003/(0.183*num_Doc + 0.003*(num_Influencer+num_Norm) + 0.906)
                        # elif agent_type2 == "Stubborn":
                        #     weight = 0                            
                        else:
                            weight = 0.183/(0.183*num_Doc + 0.003*(num_Influencer+num_Norm+num_Stub) + 0.906)
                    else:                         
                        weight = 0.906/(0.183*num_Doc + 0.003*(num_Influencer+num_Norm+num_Stub) + 0.906)
                
        
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
    
    def generate_PBN(self):
        gml_path = 'C://Users/limoy/OneDrive/Desktop/polblogs.gml'
        self.G = nx.read_gml(gml_path, label='id')

        if min(self.G.nodes()) == 1:
            mapping = {node: node - 1 for node in self.G.nodes()}
            self.G = nx.relabel_nodes(self.G, mapping)

        # # Create a graph with two communities using the block model
        # G = nx.stochastic_block_model(
        #     sizes=[num_nodes_per_community, num_nodes_per_community],
        #     p=[[p_intra, p_inter], [p_inter, p_intra]],
        #     directed=True
        # )

        # # Generate nodes with power-law degree distribution for each community
        # degrees_community1 = self.generate_power_law_nodes(num_nodes_per_community, exponent)
        # degrees_community2 = self.generate_power_law_nodes(num_nodes_per_community, exponent)

        # # Assign degrees to nodes
        # for node, degree in zip(range(num_nodes_per_community), degrees_community1):
        #     G.nodes[node]['degree'] = degree

        # for node, degree in zip(range(num_nodes_per_community, 2 * num_nodes_per_community), degrees_community2):
        #     G.nodes[node]['degree'] = degree
        # top_nodes_community1 = sorted([(node, G.nodes[node]['degree']) for node in range(num_nodes_per_community)], key=lambda x: x[1], reverse=True)[:num_influencer1]
        # top_nodes_community2 = sorted([(node, G.nodes[node]['degree']) for node in range(num_nodes_per_community, 2 * num_nodes_per_community)], key=lambda x: x[1], reverse=True)[:num_influencer2]

        return self.G



    # def generate_power_law_nodes(self, num_nodes, exponent):
    #     self.degrees = powerlaw.Power_Law(xmin=1, parameters=[exponent]).generate_random(num_nodes).astype(int)
    #     return self.degrees