import Model1
import Model2
import Model3
import Model4
import matplotlib.pyplot as plt
import numpy
import random
# import Simulation as Sim


def run_sim(G):
    num_iteration = 0
    opinion_history = []
    # while num_iteration < max_iteration:
    while num_iteration < 300:
        for node in G.graph.nodes():
            neighbors = list(G.graph.neighbors(node))
    
            weighted_sum = sum(G.weight_matrix[node][n] * G.agents[n].opinion for n in neighbors)
            new_opinion = weighted_sum / sum(G.weight_matrix[node][n] for n in neighbors)
            G.agents[node].opinion = new_opinion
        num_iteration += 1
        opinions_snapshot = [G.agents[node].opinion for node in G.graph.nodes()]
        opinion_history.append(opinions_snapshot)
    return opinion_history
    

overall_iteration = 300
random_state = numpy.random.RandomState(42)
result1 = []
result2 = []
result3 = []
result4 = []
result5 = []
# time_reaching_consensus1 = []
# time_reaching_consensus2 = []
# time_reaching_consensus3 = []
# time_reaching_consensus4 = []
# max_iter1 = 250
# max_iter2 = 50
# max_iter3 = 400
# max_iter4 = 50
for _ in range(1):
    seed = random.randint(0, 1000)  # Generate a random seed for each run
    random.seed(seed)  # Set the random seed
    G1 = Model2.ERNetwork(50, 4950, 0.01, 0.1, random_state)
    # G2 = Model2.ERNetwork(50, 495, 0.01, 0.3, random_state)
    # G3 = Model2.ERNetwork(50, 495, 0.01, 0.5, random_state)
    # G4 = Model2.ERNetwork(50, 495, 0.01, 0.7, random_state)
    # G5 = Model2.ERNetwork(50, 495, 0.01, 0.9, random_state)
    # num_iteration = 0
    # opinion_history = []

    opinion_hist1 = run_sim(G1)
    # opinion_hist2= run_sim(G2)
    # opinion_hist3= run_sim(G3)
    # opinion_hist4= run_sim(G4)
    # opinion_hist5= run_sim(G5)
    # compensation1 = [0.999]*G1.num_of_nodes
    # compensation2 = [0.999]*G2.num_of_nodes
    # compensation3 = [0.999]*G3.num_of_nodes
    # compensation4 = [0.999]*G4.num_of_nodes
    # if len(opinion_hist1) < overall_iteration:
    #     ones = overall_iteration - len(opinion_hist1)
    #     for delta in range(ones):
    #         opinion_hist1.append(compensation1)
    # if len(opinion_hist2) < overall_iteration:
    #     ones = overall_iteration - len(opinion_hist2)
    #     for delta in range(ones):
    #         opinion_hist2.append(compensation2)
    # if len(opinion_hist3) < overall_iteration:
    #     ones = overall_iteration - len(opinion_hist3)
    #     for delta in range(ones):
    #         opinion_hist3.append(compensation3)
    # if len(opinion_hist4) < overall_iteration:
    #     ones = overall_iteration - len(opinion_hist4)
    #     for delta in range(ones):
    #         opinion_hist4.append(compensation4)
    result1.append(opinion_hist1)
    # # time_reaching_consensus1.append(num_iter1)
    # result2.append(opinion_hist2)
    # # time_reaching_consensus2.append(num_iter2)
    # result3.append(opinion_hist3)
    # # time_reaching_consensus3.append(num_iter3)
    # result4.append(opinion_hist4)
    # # time_reaching_consensus4.append(num_iter4)
    # result5.append(opinion_hist5)


# average_time1 = numpy.average(time_reaching_consensus1)
num_nodes1 = len(G1.graph.nodes())
avg_opinions1 = numpy.zeros((overall_iteration, num_nodes1))

# # average_time2 = numpy.average(time_reaching_consensus2)
# num_nodes2 = len(G2.graph.nodes())
# avg_opinions2 = numpy.zeros((overall_iteration, num_nodes2))

# # average_time3 = numpy.average(time_reaching_consensus3)
# num_nodes3 = len(G3.graph.nodes())
# avg_opinions3 = numpy.zeros((overall_iteration, num_nodes3))

# # average_time4 = numpy.average(time_reaching_consensus4)
# num_nodes4 = len(G4.graph.nodes())
# avg_opinions4 = numpy.zeros((overall_iteration, num_nodes4))

# # average_time5 = numpy.average(time_reaching_consensus5)
# num_nodes5 = len(G5.graph.nodes())
# avg_opinions5 = numpy.zeros((overall_iteration, num_nodes5))

for i in range(overall_iteration):
    for j in range(num_nodes1):
        avg_opinion = numpy.mean([result[i][j] for result in result1])
        avg_opinions1[i][j] = avg_opinion

# for i in range(overall_iteration):
#     for j in range(num_nodes2):
#         avg_opinion = numpy.mean([result[i][j] for result in result2])
#         avg_opinions2[i][j] = avg_opinion

# for i in range(overall_iteration):
#     for j in range(num_nodes3):
#         avg_opinion = numpy.mean([result[i][j] for result in result3])
#         avg_opinions3[i][j] = avg_opinion

# for i in range(overall_iteration):
#     for j in range(num_nodes4):
#         avg_opinion = numpy.mean([result[i][j] for result in result4])
#         avg_opinions4[i][j] = avg_opinion

# for i in range(overall_iteration):
#     for j in range(num_nodes5):
#         avg_opinion = numpy.mean([result[i][j] for result in result5])
#         avg_opinions5[i][j] = avg_opinion

# print(average_time)

doc_1 = []
doc_1_above = []
doc_1_below = []
doc_1_var = []
# doc_3 = []
# doc_5 = []
# doc_7 = []
# doc_9 = []
plt.figure(figsize=(12, 8))
for i in range(overall_iteration):
    doc_1.append(numpy.average(avg_opinions1[i]))
    doc_1_var.append(numpy.var(avg_opinions1[i]))
    doc_1_above.append(numpy.average(avg_opinions1[i])+100*numpy.var(avg_opinions1[i]))
    doc_1_below.append(numpy.average(avg_opinions1[i])-100*numpy.var(avg_opinions1[i]))
    # doc_3.append(numpy.average(avg_opinions2[i]))
    # doc_5.append(numpy.average(avg_opinions3[i]))
    # doc_7.append(numpy.average(avg_opinions4[i]))
    # doc_9.append(numpy.average(avg_opinions5[i]))
    # opinion_values = [opinions[node_idx] for opinions in opinion_history]
# for node_idx in range(num_nodes1):
#     plt.plot(range(overall_iteration), avg_opinions1[:, node_idx], alpha=0.05, color = "plum")
# for node_idx in range(num_nodes2):
#     plt.plot(range(overall_iteration), avg_opinions2[:, node_idx], alpha=0.05, color = "skyblue")
# for node_idx in range(num_nodes3):
#     plt.plot(range(overall_iteration), avg_opinions3[:, node_idx], alpha=0.05, color = "lightsalmon")
# for node_idx in range(num_nodes4):
#     plt.plot(range(overall_iteration), avg_opinions4[:, node_idx], alpha=0.05, color = "khaki")
# for node_idx in range(num_nodes5):
#     plt.plot(range(overall_iteration), avg_opinions5[:, node_idx], alpha=0.05, color = "mediumspringgreen")
plt.plot(range(overall_iteration), doc_1, alpha=1, color = "purple")
# plt.plot(range(overall_iteration), doc_1_var, alpha=1, color = "yellow")
# plt.plot(range(overall_iteration), doc_1_below, alpha=1, color = "red")


# plt.plot(range(overall_iteration), doc_3, alpha=1, color = "steelblue")
# plt.plot(range(overall_iteration), doc_5, alpha=1, color = "red")
# plt.plot(range(overall_iteration), doc_7, alpha=1, color = "gold")
# plt.plot(range(overall_iteration), doc_9, alpha=1, color = "green")
# plt.axhline(y = doc_1[overall_iteration-1], xmin = 0, xmax = overall_iteration, linestyle = 'dashed', color = "grey")
plt.fill_between(range(overall_iteration), doc_1_below, doc_1_above, color = "plum", alpha = 0.3)
# plt.axhline(y = doc_3[overall_iteration-1], xmin = 0, xmax = overall_iteration, linestyle = 'dashed', color = "grey")
# plt.axhline(y = doc_5[overall_iteration-1], xmin = 0, xmax = overall_iteration, linestyle = 'dashed', color = "grey")
# plt.axhline(y = doc_7[overall_iteration-1], xmin = 0, xmax = overall_iteration, linestyle = 'dashed', color = "grey")
# plt.axhline(y = doc_9[overall_iteration-1], xmin = 0, xmax = overall_iteration, linestyle = 'dashed', color = "grey")
plt.annotate(str('%.2f' % doc_1[overall_iteration-1]),xy=(0,doc_1[overall_iteration-1]))
# plt.annotate(str('%.2f' % doc_3[overall_iteration-1]),xy=(0,doc_3[overall_iteration-1]))
# plt.annotate(str('%.2f' % doc_5[overall_iteration-1]),xy=(0,doc_5[overall_iteration-1]))
# plt.annotate(str('%.2f' % doc_7[overall_iteration-1]),xy=(0,doc_7[overall_iteration-1]))
# plt.annotate(str('%.2f' % doc_9[overall_iteration-1]),xy=(0,doc_9[overall_iteration-1]))
# plt.plot(average_time1,1,marker='o', color = "purple", label = "Non-Doc opinion with 1% Doctors")
# plt.plot(average_time2,1,marker='o', color = "steelblue", label = "Non-Doc opinion with 10% Doctors")
# plt.plot(average_time3,1,marker='o', color = "red", label = "Non-Doc opinion with 1% Bots")
# plt.plot(average_time4,1,marker='o', color = "gold", label = "Non-Doc opinion with 10% Doctors")
# plt.plot(range(overall_iteration),1)
# plt.axhline(range(overall_iteration), 1, linestyle='dashed', color = 'grey')
# plt.axhline(range(overall_iteration), 0, linestyle='dashed', color = 'grey')
plt.xlabel('Iteration')
plt.ylabel('Opinion')
plt.title('Opinion Dynamics Over Time')
plt.grid(False)
plt.legend()



plt.show()

