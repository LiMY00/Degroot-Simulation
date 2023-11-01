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
    ave_every_run = []
    # while num_iteration < max_iteration:
    while num_iteration < 500:
        for node in G.graph.nodes():
            neighbors = list(G.graph.neighbors(node))
    
            weighted_sum = sum(G.weight_matrix[node][n] * G.agents[n].opinion for n in neighbors)
            new_opinion = weighted_sum / sum(G.weight_matrix[node][n] for n in neighbors)
            G.agents[node].opinion = new_opinion
        num_iteration += 1
        opinions_snapshot = [G.agents[node].opinion for node in G.graph.nodes()]
        opinion_history.append(opinions_snapshot)
        ave_every_run.append(numpy.average(opinions_snapshot))
    return opinion_history, ave_every_run
    
overall_iteration = 500
random_state = numpy.random.RandomState(42)
result1 = []
result2 = []
result3 = []
result4 = []
time_reaching_consensus1 = []
time_reaching_consensus2 = []
time_reaching_consensus3 = []
time_reaching_consensus4 = []
# max_iter1 = 250
# max_iter2 = 50
# max_iter3 = 400
# max_iter4 = 50
var1 = []
var2 = []
var3 = []
for _ in range(10):
    seed = random.randint(0, 1000)  # Generate a random seed for each run
    random.seed(seed)  # Set the random seed
    # G1: only bots, 50% with initial opinion 0 and 50% with initial opinion 1
    G1 = Model4.ERNetwork(50, 4950, 0.01, 1, 0.5, random_state)
    # G2: only bots, all with initial opinion 0
    G2 = Model4.ERNetwork(50, 4950, 0.01, 1, 1, random_state)
    # # G3: 50% doctors with initial opinion 1 and 50% bots with 0.
    G3 = Model4.ERNetwork(50, 4950, 0.01, 0.5, 1, random_state)
    # G4 = Model3.ERNetwork(500, 4500, 0.01, 0, random_state)
    # num_iteration = 0
    # opinion_history = []

    opinion_hist1, ave_every_run1 = run_sim(G1)
    opinion_hist2, ave_every_run2 = run_sim(G2)
    opinion_hist3, ave_every_run3 = run_sim(G3)
    # opinion_hist4, num_iter4 = run_sim(G4, max_iter4)
    # compensation1 = [0.999]*G1.num_of_nodes
    # compensation2 = [0.999]*G2.num_of_nodes
    # compensation3 = [0.999]*G3.num_of_nodes
    # # compensation4 = [0.999]*G4.num_of_nodes
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
    # var1.append(ave_every_run1)
    # var2.append(ave_every_run2)
    # var3.append(ave_every_run3)
    # # time_reaching_consensus1.append(num_iter1)
    result2.append(opinion_hist2)
    # # time_reaching_consensus2.append(num_iter2)
    result3.append(opinion_hist3)
    # time_reaching_consensus3.append(num_iter3)
    # result4.append(opinion_hist4)
    # time_reaching_consensus4.append(num_iter4)
# var_rounds1 = numpy.var(var1)
# var_rounds2 = numpy.var(var2)
# var_rounds3 = numpy.var(var3)
# print(var_rounds1, var_rounds2, var_rounds3)


# average_time1 = numpy.average(time_reaching_consensus1)
num_nodes1 = len(G1.graph.nodes())
avg_opinions1 = numpy.zeros((overall_iteration, num_nodes1))

# average_time2 = numpy.average(time_reaching_consensus2)
num_nodes2 = len(G2.graph.nodes())
avg_opinions2 = numpy.zeros((overall_iteration, num_nodes2))

# average_time3 = numpy.average(time_reaching_consensus3)
num_nodes3 = len(G3.graph.nodes())
avg_opinions3 = numpy.zeros((overall_iteration, num_nodes3))

# average_time4 = numpy.average(time_reaching_consensus4)
# num_nodes4 = len(G4.graph.nodes())
# avg_opinions4 = numpy.zeros((overall_iteration, num_nodes4))

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

# for i in range(overall_iteration):
#     for j in range(num_nodes4):
#         avg_opinion = numpy.mean([result[i][j] for result in result4])
#         avg_opinions4[i][j] = avg_opinion


# print(average_time)

doc_50 = []
doc_500 = []
bot_50 = []
doc_50_above = []
doc_500_above = []
bot_50_above = []
doc_50_below = []
doc_500_below = []
bot_50_below = []
# bot_500 = []
plt.figure(figsize=(12, 8))
for i in range(overall_iteration):
    doc_50.append(numpy.average(avg_opinions1[i]))
    doc_50_above.append(numpy.average(avg_opinions1[i]) + numpy.var(avg_opinions1[i]))
    doc_500_above.append(numpy.average(avg_opinions2[i]) + numpy.var(avg_opinions2[i]))
    bot_50_above.append(numpy.average(avg_opinions3[i]) + numpy.var(avg_opinions3[i]))
    doc_500.append(numpy.average(avg_opinions2[i]))
    bot_50.append(numpy.average(avg_opinions3[i]))
    doc_50_below.append(numpy.average(avg_opinions1[i]) - numpy.var(avg_opinions1[i]))
    doc_500_below.append(numpy.average(avg_opinions2[i]) - numpy.var(avg_opinions2[i]))
    bot_50_below.append(numpy.average(avg_opinions3[i]) - numpy.var(avg_opinions3[i]))
#     # bot_500.append(numpy.average(avg_opinions4[i]))
#     # opinion_values = [opinions[node_idx] for opinions in opinion_history]
# for node_idx in range(num_nodes1):
#     plt.plot(range(overall_iteration), avg_opinions1[:, node_idx], alpha=0.1, color = "plum")
# for node_idx in range(num_nodes2):
#     plt.plot(range(overall_iteration), avg_opinions2[:, node_idx], alpha=0.1, color = "skyblue")
# for node_idx in range(num_nodes3):
#     plt.plot(range(overall_iteration), avg_opinions3[:, node_idx], alpha=0.1, color = "lightsalmon")
# # for node_idx in range(num_nodes4):
# #     plt.plot(range(overall_iteration), avg_opinions4[:, node_idx], alpha=0.1, color = "khaki")
plt.plot(range(overall_iteration), doc_50, alpha=1, color = "yellowgreen", label = "25 stubborn users with 0 and 25 stubborn users with 1", linewidth=5)
plt.plot(range(overall_iteration), doc_500, alpha=1, color = "teal", label = "only stubborn users with opinion 0", linewidth=5)
plt.plot(range(overall_iteration), bot_50, alpha=1, color = "darkred", label = "25 doctors with 0 and 25 stubborn users with 1", linewidth=5)
plt.fill_between(range(overall_iteration), doc_50_below, doc_50_above, alpha = 0.3, color = "palegreen")
plt.fill_between(range(overall_iteration), doc_500_below, doc_500_above, alpha = 0.3, color = "darkturquoise")
plt.fill_between(range(overall_iteration), bot_50_below, bot_50_above, alpha = 0.3, color = "coral")
# # plt.plot(range(overall_iteration), bot_500, alpha=1, color = "gold")
# # plt.plot(overall_iteration,1,marker='o', color = "purple", label = "Non-Doc opinion with 1% Doctors")
# # plt.plot(overall_iteration,1,marker='o', color = "steelblue", label = "Non-Doc opinion with 10% Doctors")
# # plt.plot(overall_iteration,1,marker='o', color = "red", label = "Non-Doc opinion with 1% Bots")
# # plt.plot(average_time4,1,marker='o', color = "gold", label = "Non-Doc opinion with 10% Doctors")
# # plt.plot(range(overall_iteration),1)
plt.axhline(y = doc_50[overall_iteration-1], color = 'grey', linestyle = 'dashed')
plt.axhline(y = bot_50[overall_iteration-1], color = 'grey', linestyle = 'dashed')
# plt.axhline(y = bot_50[overall_iteration-1], color = 'grey')
plt.annotate(str('%.2f' % doc_50[overall_iteration-1]),xy=(overall_iteration,doc_50[overall_iteration-1]), size=30)
plt.annotate(str('%.2f' % bot_50[overall_iteration-1]),xy=(overall_iteration,bot_50[overall_iteration-1]), size=30)
# plt.annotate(str('%.2f' % doc_5[overall_iteration-1]),xy=(overall_iteration,doc_5[overall_iteration-1]))
plt.xlabel('Iteration', size=30)
plt.ylabel('Opinion', size=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
leg = plt.legend(prop={"size":20})

# plt.title('Opinion Dynamics Over Time')
# plt.grid(False)
# plt.legend()

plt.show()
