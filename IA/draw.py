import matplotlib.pyplot as plt
from IPython import display

plt.ion()

class Draw:
    def __init__(self):
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0
        self.n_games = 0
        self.show()
        plt.pause(1)

    def plot(self, score):
        self.n_games += 1
        self.plot_scores.append(score)
        self.total_score += score
        mean_score = self.total_score / self.n_games
        self.plot_mean_scores.append(mean_score)
        self.show()

    def show(self):
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.ylim(ymin=0)
        plt.plot(self.plot_scores)
        plt.plot(self.plot_mean_scores)
        if len(self.plot_scores) != 0:
            plt.text(len(self.plot_scores)-1, self.plot_scores[-1], str(self.plot_scores[-1]))
            plt.text(len(self.plot_mean_scores)-1, self.plot_mean_scores[-1], str(self.plot_mean_scores[-1]))
        plt.draw()
        plt.pause(0.0001)
