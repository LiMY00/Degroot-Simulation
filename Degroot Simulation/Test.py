import Model1
import Model2
import Model3
import Model4
import matplotlib.pyplot as plt
import numpy
# import Simulation as Sim

# num_doc = 100
# num_NonDoc = 900
max_iteration = 200
random_state = numpy.random.RandomState(42)
G = Model2.ERNetwork(50, 4950, 0.01, 0.2, random_state)
num_iteration = 0
opinion_history = []
# while num_iteration < max_iteration:
while num_iteration < max_iteration and not G.consensus_reached(0.001):
    for node in G.graph.nodes():
        neighbors = list(G.graph.neighbors(node))
            
        weighted_sum = sum(G.weight_matrix[node][n] * G.agents[n].opinion for n in neighbors)
        new_opinion = weighted_sum / sum(G.weight_matrix[node][n] for n in neighbors)
        G.agents[node].opinion = new_opinion
    num_iteration += 1
    opinions_snapshot = [G.agents[node].opinion for node in G.graph.nodes()]
    opinion_history.append(opinions_snapshot)
num_nodes = G.num_of_nodes
time_steps = range(num_iteration)

doc_1 = []
for i in range(max_iteration):
    doc_1.append(numpy.average(opinion_history[i]))
plt.figure(figsize=(12, 8))
for node_idx in range(num_nodes):
    opinion_values = [opinions[node_idx] for opinions in opinion_history]
    plt.plot(time_steps, opinion_values, alpha=0.1, color = "grey")
plt.plot(time_steps, doc_1, alpha = 1, color = "black")
plt.xlabel('Iteration')
plt.ylabel('Opinion')
plt.title('Opinion Dynamics Over Time')
plt.legend()
plt.grid(True)
plt.show()

# simulation = Sim.Simulation(max_iteration)
# simulation.run
# print(f"Consensus reached in {simulation.num_iteration} iterations.")
# simulation.plot_consensus()