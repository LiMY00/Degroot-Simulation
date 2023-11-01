import Model1
import Model2
import Model3
import Model4
import matplotlib.pyplot as plt
import numpy
import random
# import Simulation as Sim


def run_sim(G, max_iteration):
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
    return opinion_history, num_iteration
    
    # num_nodes = G.num_of_nodes
# num_doc = 100
# num_NonDoc = 900
overall_iteration = 2000
random_state = numpy.random.RandomState(42)
result1 = []
result2 = []
result3 = []
result4 = []
result5 = []
result6 = []
time_reaching_consensus1 = []
time_reaching_consensus2 = []
time_reaching_consensus3 = []
time_reaching_consensus4 = []
time_reaching_consensus5 = []
time_reaching_consensus6 = []
max_iter1 = 250
max_iter2 = 50
max_iter3 = 400
max_iter4 = 50
max_iter5 = 1000
max_iter6 = 2000
for _ in range(10):
    seed = random.randint(0, 1000)  # Generate a random seed for each run
    random.seed(seed)  # Set the random seed
    G1 = Model2.ERNetwork(50, 4950, 0.01, 0, random_state)
    G2 = Model2.ERNetwork(500, 4500, 0.01, 0, random_state)
    G3 = Model3.ERNetwork(50, 4950, 0.01, 0, random_state)
    G4 = Model3.ERNetwork(500, 4500, 0.01, 0, random_state)
    G5 = Model2.ERNetwork(10, 4990, 0.01, 0, random_state)
    G6 = Model3.ERNetwork(10, 4990, 0.01, 0, random_state)
    # num_iteration = 0
    # opinion_history = []

    opinion_hist1, num_iter1 = run_sim(G1, max_iter1)
    opinion_hist2, num_iter2 = run_sim(G2, max_iter2)
    opinion_hist3, num_iter3 = run_sim(G3, max_iter3)
    opinion_hist4, num_iter4 = run_sim(G4, max_iter4)
    opinion_hist5, num_iter5 = run_sim(G5, max_iter5)
    opinion_hist6, num_iter6 = run_sim(G6, max_iter6)
    compensation1 = [0.999]*G1.num_of_nodes
    compensation2 = [0.999]*G2.num_of_nodes
    compensation3 = [0.999]*G3.num_of_nodes
    compensation4 = [0.999]*G4.num_of_nodes
    compensation5 = [0.999]*G5.num_of_nodes
    compensation6 = [0.999]*G6.num_of_nodes
    if len(opinion_hist1) < overall_iteration:
        ones = overall_iteration - len(opinion_hist1)
        for delta in range(ones):
            opinion_hist1.append(compensation1)
    if len(opinion_hist2) < overall_iteration:
        ones = overall_iteration - len(opinion_hist2)
        for delta in range(ones):
            opinion_hist2.append(compensation2)
    if len(opinion_hist3) < overall_iteration:
        ones = overall_iteration - len(opinion_hist3)
        for delta in range(ones):
            opinion_hist3.append(compensation3)
    if len(opinion_hist4) < overall_iteration:
        ones = overall_iteration - len(opinion_hist4)
        for delta in range(ones):
            opinion_hist4.append(compensation4)
    if len(opinion_hist5) < overall_iteration:
        ones = overall_iteration - len(opinion_hist5)
        for delta in range(ones):
            opinion_hist5.append(compensation5)
    if len(opinion_hist6) < overall_iteration:
        ones = overall_iteration - len(opinion_hist6)
        for delta in range(ones):
            opinion_hist6.append(compensation6)
    result1.append(opinion_hist1)
    time_reaching_consensus1.append(num_iter1)
    result2.append(opinion_hist2)
    time_reaching_consensus2.append(num_iter2)
    result3.append(opinion_hist3)
    time_reaching_consensus3.append(num_iter3)
    result4.append(opinion_hist4)
    time_reaching_consensus4.append(num_iter4)
    result5.append(opinion_hist5)
    time_reaching_consensus5.append(num_iter5)
    result6.append(opinion_hist6)
    time_reaching_consensus6.append(num_iter6)


average_time1 = numpy.average(time_reaching_consensus1)
num_nodes1 = len(G1.graph.nodes())
avg_opinions1 = numpy.zeros((overall_iteration, num_nodes1))

average_time2 = numpy.average(time_reaching_consensus2)
num_nodes2 = len(G2.graph.nodes())
avg_opinions2 = numpy.zeros((overall_iteration, num_nodes2))

average_time3 = numpy.average(time_reaching_consensus3)
num_nodes3 = len(G3.graph.nodes())
avg_opinions3 = numpy.zeros((overall_iteration, num_nodes3))

average_time4 = numpy.average(time_reaching_consensus4)
num_nodes4 = len(G4.graph.nodes())
avg_opinions4 = numpy.zeros((overall_iteration, num_nodes4))

average_time5 = numpy.average(time_reaching_consensus5)
num_nodes5 = len(G5.graph.nodes())
avg_opinions5 = numpy.zeros((overall_iteration, num_nodes5))

average_time6 = numpy.average(time_reaching_consensus6)
num_nodes6 = len(G6.graph.nodes())
avg_opinions6 = numpy.zeros((overall_iteration, num_nodes6))

for i in range(overall_iteration):
    for j in range(num_nodes1):
        avg_opinion = numpy.mean([result[i][j] for result in result1])
        avg_opinions1[i][j] = avg_opinion

for i in range(overall_iteration):
    for j in range(num_nodes2):
        avg_opinion = numpy.mean([result[i][j] for result in result2])
        avg_opinions2[i][j] = avg_opinion

for i in range(overall_iteration):
    for j in range(num_nodes3):
        avg_opinion = numpy.mean([result[i][j] for result in result3])
        avg_opinions3[i][j] = avg_opinion

for i in range(overall_iteration):
    for j in range(num_nodes4):
        avg_opinion = numpy.mean([result[i][j] for result in result4])
        avg_opinions4[i][j] = avg_opinion

for i in range(overall_iteration):
    for j in range(num_nodes5):
        avg_opinion = numpy.mean([result[i][j] for result in result5])
        avg_opinions5[i][j] = avg_opinion

for i in range(overall_iteration):
    for j in range(num_nodes6):
        avg_opinion = numpy.mean([result[i][j] for result in result6])
        avg_opinions6[i][j] = avg_opinion



# print(average_time)

doc_50 = []
doc_500 = []
bot_50 = []
bot_500 = []
doc_10 = []
bot_10 = []
# doc_50_above = []
# doc_50_below = []
plt.figure(figsize=(12, 8))
for i in range(overall_iteration):
    doc_50.append(numpy.average(avg_opinions1[i]))
    doc_500.append(numpy.average(avg_opinions2[i]))
    bot_50.append(numpy.average(avg_opinions3[i]))
    bot_500.append(numpy.average(avg_opinions4[i]))
    doc_10.append(numpy.average(avg_opinions5[i]))
    bot_10.append(numpy.average(avg_opinions6[i]))
    # opinion_values = [opinions[node_idx] for opinions in opinion_history]
for node_idx in range(num_nodes1):
    plt.plot(range(overall_iteration), avg_opinions1[:, node_idx], alpha=0.1, color = "plum",   linewidth=2)
# plt.fill_between(range(overall_iteration), doc_50_below, doc_50_above, alpha = 0.3, color = "plum")
for node_idx in range(num_nodes2):
    plt.plot(range(overall_iteration), avg_opinions2[:, node_idx], alpha=0.1, color = "skyblue",   linewidth=2)
for node_idx in range(num_nodes3):
    plt.plot(range(overall_iteration), avg_opinions3[:, node_idx], alpha=0.1, color = "lightsalmon",   linewidth=2)
for node_idx in range(num_nodes4):
    plt.plot(range(overall_iteration), avg_opinions4[:, node_idx], alpha=0.1, color = "khaki",   linewidth=2)
for node_idx in range(num_nodes5):
    plt.plot(range(overall_iteration), avg_opinions5[:, node_idx], alpha=0.1, color = "cyan", linewidth=2)
for node_idx in range(num_nodes5):
    plt.plot(range(overall_iteration), avg_opinions6[:, node_idx], alpha=0.1, color = "pink",   linewidth=2)
plt.plot(range(overall_iteration), doc_50, alpha=1, color = "purple",   linewidth=2)
plt.plot(range(overall_iteration), doc_500, alpha=1, color = "steelblue", linewidth=2)
plt.plot(range(overall_iteration), bot_50, alpha=1, color = "red",   linewidth=2)
plt.plot(range(overall_iteration), bot_500, alpha=1, color = "gold",   linewidth=2)
plt.plot(range(overall_iteration), doc_10, alpha=1, color = "darkcyan",   linewidth=2)
plt.plot(range(overall_iteration), bot_10, alpha=1, color = "mediumvioletred",   linewidth=2)
plt.plot(average_time1,1,marker='o', color = "purple", label = "with 1% Doctors")
plt.plot(average_time2,1,marker='o', color = "steelblue", label = "with 10% Doctors")
plt.plot(average_time3,1,marker='o', color = "red", label = "with 1% Bots")
plt.plot(average_time4,1,marker='o', color = "gold", label = "with 10% Bots")
plt.plot(average_time5,1,marker='o', color = "darkcyan", label = "with 0.2% Doctors")
plt.plot(average_time6,1,marker='o', color = "mediumvioletred", label = "with 0.2% Bots")
# plt.plot(range(overall_iteration),1)
plt.axhline(y = 1, color = 'grey', linestyle = 'dashed')
plt.axvline(x = average_time1, color = 'grey', linestyle = 'dashed')
plt.axvline(x = average_time2, color = 'grey', linestyle = 'dashed')
plt.axvline(x = average_time3, color = 'grey', linestyle = 'dashed')
plt.axvline(x = average_time4, color = 'grey', linestyle = 'dashed')
plt.axvline(x = average_time5, color = 'grey', linestyle = 'dashed')
plt.axvline(x = average_time6, color = 'grey', linestyle = 'dashed')
plt.xlabel('Iteration', size=30)
plt.ylabel('Opinion', size=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.locator_params(nbins=4)
leg = plt.legend(prop={"size":20})
plt.grid(False)
plt.legend()



plt.show()

# simulation = Sim.Simulation(max_iteration)
# simulation.run
# print(f"Consensus reached in {simulation.num_iteration} iterations.")
# simulation.plot_consensus()