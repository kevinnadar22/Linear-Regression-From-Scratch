"""
Gradient Descent Animation

Initially we have linear relationship between experience and income.
We want to find the best fit line for the data.

Y = mX + b is the equation of the line.

We want to find the best values of m and b.

Initially we have random values of m and b.

We need to reduce the error between the predicted line and the actual data and 
the m and b corresponding to the minimum error is the best fit.

For that we plot the loss function with respect to m and b and check the rate of change.

We update the values of m and b in the opposite direction of the rate of change to reach the minimum error.

This is done in an iterative manner until we reach the minima.

This is gradient descent.

m_grad = -2/n * sum(x * (y - (m*x + b))) #Convergence theorem for m (basically the derivative of loss function with respect to m)
b_grad = -2/n * sum(y - (m*x + b)) #Convergence theorem for b (basically the derivative of loss function with respect to b)

Note: m is weight and b is bias (the intercept on y-axis).
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class LinearRegression:
    def __init__(self, data_file, learning_rate, epochs, x_key, y_key):
        self.data = pd.read_csv(data_file)
        self.x = self.data[x_key]
        self.y = self.data[y_key]
        self.L = learning_rate
        self.epochs = epochs
        self.x_key = x_key
        self.y_key = y_key

        # Initialize parameters
        self.m = 0
        self.b = 0

        # For storing parameters and losses for animation
        self.m_history = []
        self.b_history = []
        self.losses = []

    def gd(self, m, b, L):
        n = len(self.data)
        m_grad = 0
        b_grad = 0

        for i in range(len(self.data)):
            x = self.data[self.x_key].iloc[i]
            y = self.data[self.y_key].iloc[i]

            m_grad += -2 / n * x * (y - (m * x + b))  # Convergence theorem for m
            b_grad += -2 / n * (y - (m * x + b))  # Convergence theorem for b

        m -= m_grad * L
        b -= b_grad * L

        return m, b

    def train(self):
        for i in range(self.epochs):
            self.m, self.b = self.gd(self.m, self.b, self.L)
            self.m_history.append(self.m)
            self.b_history.append(self.b)

            if i % 100 == 0:
                print(f"Epoch {i}: m={self.m:.2f}, b={self.b:.2f}")

            loss = sum(
                (y - (self.m * x + self.b)) ** 2 for x, y in zip(self.x, self.y)
            ) / len(self.data)
            self.losses.append(loss)

    def predict(self, x):
        return self.m * x + self.b

    def animate(self, frame, ax1, ax2):
        ax1.clear()
        ax2.clear()

        # Plot the data and current line on the first subplot
        ax1.scatter(self.x, self.y, label="Actual Data", s=10)  # Reduce marker size
        x_line = np.linspace(min(self.x), max(self.x), 100)
        y_line = self.m_history[frame] * x_line + self.b_history[frame]
        ax1.plot(
            x_line, y_line, "r-", label=f"Epoch {frame}", linewidth=1
        )  # Reduce line width
        ax1.legend(fontsize="small")  # Smaller legend text
        ax1.set_title(f"Fitting Line for {self.x_key} and {self.y_key}", fontsize=10)
        ax1.set_xlabel(self.x_key.capitalize(), fontsize=8)
        ax1.set_ylabel(self.y_key.capitalize(), fontsize=8)
        ax1.tick_params(axis="both", which="major", labelsize=7)

        # Plot the loss curve on the second subplot
        ax2.plot(self.losses[: frame + 1], "b-", linewidth=1)  # Reduce line width
        ax2.set_title("Loss vs Epoch", fontsize=10)
        ax2.set_xlabel("Epoch", fontsize=8)
        ax2.set_ylabel("Loss", fontsize=8)
        ax2.tick_params(axis="both", which="major", labelsize=7)

        # Add current parameters as text
        ax1.text(
            0.02,
            0.85,  # Move text lower to fit smaller plot
            f"m={self.m_history[frame]:.2f}\nb={self.b_history[frame]:.2f}",
            transform=ax1.transAxes,
            fontsize=8,
        )

    def create_animation(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # Sample epochs exponentially to show more detail early on
        epochs_to_show = [
            int(np.exp(i)) for i in np.linspace(0, np.log(self.epochs), 100)
        ]
        epochs_to_show = [e for e in epochs_to_show if e < self.epochs]
        epochs_to_show.append(self.epochs - 1)  # Add final epoch

        anim = FuncAnimation(
            fig,
            self.animate,
            frames=epochs_to_show,
            fargs=(ax1, ax2),
            interval=50,
            repeat=False,
        )

        plt.tight_layout()
        plt.show()
        anim.save("linear_regression.mp4", writer="ffmpeg")


# Usage

file_path = "data.csv"
learning_rate = 0.001
epochs = 30000
x_key = "experience"
y_key = "income"

linear_regression = LinearRegression(file_path, learning_rate, epochs, x_key, y_key)
linear_regression.train()
linear_regression.create_animation()
