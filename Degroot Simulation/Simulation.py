import Model
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, max_iteration) -> None:
        self.G = Model.ERNetwork(100, 900, 0.1)
        self.max_iteration = max_iteration
        self.num_iteration = 0
        self.opinion_history = []

    def run(self, err=0.01):
        while self.num_iteration < self.max_iteration and not self.G.consensus_reached(err):
            self.G.update_opinions
            self.num_iteration += 1
            opinions_snapshot = [self.G.agents[node].opinion for node in self.G.graph.nodes()]
            self.opinion_history.append(opinions_snapshot)
    
    def plot_consensus(self):
        num_nodes = self.G.num_of_nodes
        time_steps = range(self.num_iteration)

        plt.figure(figsize=(12, 8))
        opinion_values = [opinions[1] for opinions in self.opinion_history]
        plt.plot(time_steps, opinion_values, alpha = 0.7)
        # for node_idx in range(num_nodes):
        #     opinion_values = [opinions[node_idx] for opinions in self.opinion_history]
        #     plt.plot(time_steps, opinion_values, alpha=0.7)

        plt.xlabel('Iteration')
        plt.ylabel('Opinion')
        plt.title('Opinion Dynamics Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()







        