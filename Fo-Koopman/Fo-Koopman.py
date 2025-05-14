import math
import random

import numpy as np
import gc
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.special import gamma
from scipy.special import digamma



def generalized_binomial(alpha, k):
    if k < 0 or not isinstance(k, int):
        raise ValueError("k must be a non-negative integer")
    G = gamma(alpha + 1) / (gamma(k + 1) * gamma(alpha - k + 1))
    return G

class liftNet(nn.Module):
    def __init__(self, input_size, hidden_size, high_size):
        super(liftNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, high_size)
        )

    def forward(self, x):
        y = self.encoder(x).to(device)
        return y


class ReplayBuffer:
    def __init__(self, state_dim=2, capacity=20000, k=5):
        self.state_buf = np.zeros((capacity, 1, state_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, k, state_dim), dtype=np.float32)
        self.ptr, self.size, self.capacity = 0, 0, capacity

    def push(self, state, state_next):
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = state_next
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.choice(self.size, batch_size, replace=False)
        batch = dict(state=self.state_buf[ind],
                     state_next=self.next_state_buf[ind])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}

    def __len__(self):
        return self.size


class koopman:
    def __init__(self, X_dim, hidden_size, high_size, order):
        self.X_dim = X_dim
        self.hidden_dim = hidden_size
        self.high_dim = high_size
        self.uplift = liftNet(self.X_dim, self.hidden_dim, self.high_dim).to(device)
        self.K = torch.ones([self.X_dim + self.high_dim, self.X_dim + self.high_dim]).to(device)
        self.alpha = order  #B(alpha,j)
        self.precomputed_c = torch.ones([1, k + 2]).to(device)

    def calc_z(self, x):
        z = (torch.cat((x, self.uplift.forward(x)), 1)).to(device)
        return z

    def precompute_coefficients(self, max_k):
        for j in range(max_k + 2):
            self.precomputed_c[0,j:j+1] = ((-1) ** j) * generalized_binomial(self.alpha, j)
    def predict(self, x, k):
            z = self.calc_z(x)
            z = z.T
            z_pred = torch.zeros([self.X_dim + self.high_dim, k + 1]).to(device)
            z_pred[:, 0:1] = z[:, 0:1]
            h = 0.005
            c = self.precomputed_c
            for count in range(k):
                sum = 0
                for j in range(1, count + 2):
                    sum = (c[0, j:j + 1] * torch.eye(self.X_dim + self.high_dim, self.X_dim + self.high_dim).to(
                        device)) @ z_pred[:, count - j + 1:count - j + 2] + sum
                z_pred[:, count + 1:count + 2] = h**(self.alpha) * self.K @ (z_pred[:, count:count + 1]) - sum
            return z_pred


class TrainThread:
    def __init__(self, X_dim, hidden_size, high_size, order):
        self.X_dim = X_dim
        self.hidden_dim = hidden_size
        self.high_dim = high_size
        self.koopman = koopman(X_dim, hidden_size, high_size, order)

    def Caculate_psi_phi(self, current_data, next_data, N):
        with torch.no_grad():
            N = N - 1
            A_j = torch.zeros([N + 1, self.X_dim + self.high_dim, self.X_dim + self.high_dim]).to(device)
            con_x = torch.cat((current_data, next_data), 0)
            z = self.koopman.calc_z(con_x[:, :]).T
            phi = z[:, 0:N + 1]
            c=self.koopman.precomputed_c
            psi = torch.zeros(([self.X_dim + self.high_dim, N])).to(device)
            for count in range(1, N + 1):
                A_j[count, :, :] = -c[0, count + 1] * torch.eye(self.X_dim + self.high_dim,
                                                                self.X_dim + self.high_dim).to(device)
                s = 0
                for n in range(1, count + 1):
                    s = s + A_j[n] @ z[:, count - n:count - n + 1]
                psi[:, count - 1:count] = z[:, count + 1:count + 2] - s
            PSI = torch.cat((z[:, 1:2], psi), 1)
            return PSI, phi

    def Solve_Koopman_K(self, current_data, next_data, k):
        with torch.no_grad():
            psi, phi = self.Caculate_psi_phi(current_data, next_data, k)
            h = 0.005
            A0 = psi @ torch.linalg.pinv(phi)
            m = -generalized_binomial(self.koopman.alpha, 1) * torch.eye(self.X_dim + self.high_dim,
                                                                 self.X_dim + self.high_dim).to(device)
            f = pow(h, self.koopman.alpha)
            self.koopman.K = (A0 + m) / f
            # return self.koopman.K

    def learning(self, current_data, next_data, k):
        self.koopman.precompute_coefficients(k)
        lift_all_next_z = self.koopman.calc_z(next_data).T
        x_next = lift_all_next_z[:self.X_dim, :]
        z_pred = self.koopman.predict(current_data, k)
        z_pred = z_pred[:, 1:]
        x_pred = z_pred[0:self.X_dim, :]
        optimizer.zero_grad()
        loss = loss_fn(lift_all_next_z[:, :k], z_pred) + loss_fn(x_next[:, :k], x_pred) + self.l2_regularization()
        loss.backward()
        optimizer.step()
        return loss

    def l2_regularization(self):
        l2_loss = []
        for module in self.koopman.uplift.modules():
            if type(module) is nn.Linear:
                l2_loss.append((module.weight ** 2).sum())
        return 1e-6 * sum(l2_loss)


def test(input_size, k):
    filename = 'FOChen_train_0.65.csv'
    batch_size = 2000
    epochs = 5
    buffer_capacity = 8000
    buffer = ReplayBuffer(state_dim=input_size, capacity=buffer_capacity, k=k)
    data = np.loadtxt(filename, delimiter=',', dtype=np.float32)
    for t in range(8000):
        current_state = data[(k + 1) * t:(k + 1) * t + 1, 0:input_size]
        next_state = data[(k + 1) * t + 1:(k + 1) * t + k + 1, 0:input_size]
        buffer.push(current_state, next_state)
    for epoch in tqdm(range(epochs), desc="Training"):
        total_loss = 0
        num_batches = len(buffer) // batch_size

        for _ in range(num_batches):
            batch = buffer.sample(batch_size)
            for j in range(batch_size):
                current_data = batch['state'][j, :, :].to(device)
                next_data = batch['state_next'][j, :, :].to(device)
                train.Solve_Koopman_K(current_data, next_data, k)
                loss = train.learning(current_data, next_data, k)
                total_loss += loss
        print(total_loss)

def evaluate_alpha(alpha_value):
    train.koopman.alpha = alpha_value

    test(input_size, k)

    # Data Loading
    datatest = np.loadtxt('FOChen_test_0.65.csv', delimiter=',', dtype=np.float32)
    total_rmse = 0.0
    group_count = 0
    for i in range(0, len(datatest), 6):
        if i + 6 > len(datatest):
            break
        test_group = datatest[i:i + 6]
        # Starting from the original state
        input_data = torch.tensor(test_group[0:1, 0:input_size]).to(device)

        # k-step prediction
        prediction = train.koopman.predict(input_data, 5)
        pred = prediction[:input_size, 1:].T.cpu().detach().numpy()
        torch.set_printoptions(precision=5)
        print(prediction[:input_size,:].T)
        # True state
        true_values = test_group[1:6, 0:input_size]

        # Calculate RMSE
        RMSE = np.sqrt(np.mean((pred - true_values)**2))
        total_rmse += RMSE
        group_count += 1

    return total_rmse / group_count


if __name__ == "__main__":
    # Parameter Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 3
    hidden_size =64
    high_size = 360
    learning_rate = 1e-4
    k = 5
    torch.manual_seed(21)
    np.random.seed(21)
    loss_fn = nn.SmoothL1Loss()

    alpha_values = [0.65]
    results = {}

    # Test
    for alpha in alpha_values:
        print(f"\n=== Testing alpha={alpha} ===")
        train = TrainThread(input_size, hidden_size, high_size, alpha)
        model = liftNet(input_size, hidden_size, high_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train.koopman.uplift = model

        avg_error = evaluate_alpha(alpha)
        results[alpha] = avg_error
        print(f"Alpha {alpha} RMSE: {avg_error:.6f}")

    # Save results
    df = pd.DataFrame(list(results.items()), columns=['Alpha', 'RMSE'])
    df.to_csv('alpha_errors.csv', index=False)
    print("\nResults have been saved to alpha_errors.csv")
