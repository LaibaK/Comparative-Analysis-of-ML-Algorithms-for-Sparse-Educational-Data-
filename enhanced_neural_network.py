import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
import matplotlib.pyplot as plt #check if this can be added 

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        initial_out = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(initial_out))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out

class ModifiedAutoEncoder(nn.Module):
    def __init__(self, num_question, k1=100, k2=50):
        super(ModifiedAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_question, k1),
            nn.BatchNorm1d(k1),
            nn.ReLU(),
            nn.Linear(k1, k2),
            nn.BatchNorm1d(k2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(k2, k1),
            nn.BatchNorm1d(k1),
            nn.ReLU(),
            nn.Linear(k1, num_question),
            nn.Sigmoid()
        )

    def get_weight_norm(self):
        norm = 0
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                norm += torch.norm(layer.weight, 2) ** 2
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                norm += torch.norm(layer.weight, 2) ** 2
        return norm

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_users = train_data.shape[0]

    training_loss_log = []
    validation_accuracy_log = []

    for epoch in range(num_epoch):
        epoch_loss = 0.0

        for user_id in range(num_users):
            user_input = Variable(zero_train_data[user_id]).unsqueeze(0)
            user_target = user_input.clone()

            optimizer.zero_grad()
            predicted_output = model(user_input)

            missing_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            user_target[missing_mask] = predicted_output[missing_mask]

            reconstruction_loss = torch.sum((predicted_output - user_target) ** 2)
            l2_penalty = (lamb / 2) * model.get_weight_norm()
            total_loss = reconstruction_loss + l2_penalty

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        val_accuracy = evaluate(model, zero_train_data, valid_data)
        training_loss_log.append(epoch_loss)
        validation_accuracy_log.append(val_accuracy)

        print(f"EPOCH: {epoch} | Train Loss: {epoch_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

    return training_loss_log, validation_accuracy_log
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

def train_modified(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch,
                   denoising=False, early_stopping=True, scheduler_on=True, batch_size=16):
    model.train()
    adam_optim = optim.Adam(model.parameters(), lr=lr)
    plateau_sched = optim.lr_scheduler.ReduceLROnPlateau(adam_optim, mode='max', patience=3, factor=0.5)
    n_students = train_data.shape[0]

    loss_history, acc_history = [], []
    best_acc, best_state, best_epoch_seen = 0, None, 0

    for ep_num in range(num_epoch):
        running_loss = 0.0

        for idx in range(0, n_students, batch_size):
            # Grab batch slices
            x_batch = zero_train_data[idx: idx + batch_size].clone()
            y_batch = train_data[idx: idx + batch_size].clone()

            if denoising:
                mask_known = (y_batch != 0)
                dropout_mask = torch.rand_like(x_batch) > 0.1
                x_batch[mask_known] *= dropout_mask[mask_known]

            # Fill missing targets with model's own predictions
            y_batch = Variable(y_batch)
            x_batch = Variable(x_batch)
            nan_idx = torch.isnan(y_batch)
            if nan_idx.any():
                y_batch[nan_idx] = model(x_batch)[nan_idx]

            adam_optim.zero_grad()
            pred = model(x_batch)
            total_loss = torch.sum((pred - y_batch) ** 2) + (lamb / 2) * model.get_weight_norm()
            total_loss.backward()
            adam_optim.step()

            running_loss += total_loss.item()

        # Evaluate and log
        val_metric = evaluate(model, zero_train_data, valid_data)
        loss_history.append(running_loss)
        acc_history.append(val_metric)

        if scheduler_on:
            plateau_sched.step(val_metric)

        print(f"[Epoch {ep_num}] Loss: {running_loss:.4f} | Val: {val_metric:.4f}")

        # Track best performance
        if val_metric > best_acc:
            best_acc = val_metric
            best_state = model.state_dict()
            best_epoch_seen = ep_num

        # Early stopping rule
        if early_stopping and ep_num - best_epoch_seen > 5:
            print("No progress — halting training.")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    return loss_history, acc_history

def compare_baseline_vs_enhanced():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    input_dim = train_matrix.shape[1]
    num_epochs = 50
    lr = 0.01
    lamb = 0.01
    k = 100

    print("\n=== Training Baseline AutoEncoder (SGD, no enhancements) ===")
    baseline_model = AutoEncoder(input_dim, k)
    base_train_losses, base_valid_accuracies = train(
        model=baseline_model,
        lr=lr,
        lamb=lamb,
        train_data=train_matrix,
        zero_train_data=zero_train_matrix,
        valid_data=valid_data,
        num_epoch=num_epochs
    )
    base_test_acc = evaluate(baseline_model, zero_train_matrix, test_data)
    print(f"Baseline Test Accuracy: {base_test_acc:.4f}")

    print("\n=== Training Enhanced Deep AutoEncoder (denoising + all enhancements) ===")
    enhanced_model = ModifiedAutoEncoder(input_dim, 100, 50)
    enh_train_losses, enh_valid_accuracies = train_modified(
        model=enhanced_model,
        lr=lr,
        lamb=lamb,
        train_data=train_matrix,
        zero_train_data=zero_train_matrix,
        valid_data=valid_data,
        num_epoch=num_epochs,
        denoising=True,
        early_stopping=True,
        scheduler_on=True,
        batch_size=16
    )
    enh_test_acc = evaluate(enhanced_model, zero_train_matrix, test_data)
    print(f"Enhanced Test Accuracy: {enh_test_acc:.4f}")

    # Plotting comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(base_train_losses, label="Baseline Train Loss")
    plt.plot(enh_train_losses, label="Enhanced Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(base_valid_accuracies, label="Baseline Valid Acc")
    plt.plot(enh_valid_accuracies, label="Enhanced Valid Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def run_comparisons():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    num_epochs = 50
    lr = 0.01
    lamb = 0.01
    k = 100
    input_dim = train_matrix.shape[1]

    models = [
        {
            "name": "Baseline (SGD)",
            "model": AutoEncoder(input_dim, k),
            "kwargs": {
                "denoising": False,
                "early_stopping": False,
                "scheduler_on": False,
            },
            "use_train_modified": False
        },
        {
            "name": "Deeper + Denoising (Adam)",
            "model": ModifiedAutoEncoder(input_dim, 100, 50),
            "kwargs": {
                "denoising": True,
                "early_stopping": False,
                "scheduler_on": False,
            },
            "use_train_modified": True
        },
        {
            "name": "Enhanced (All Combined)",
            "model": ModifiedAutoEncoder(input_dim, 100, 50),
            "kwargs": {
                "denoising": True,
                "early_stopping": True,
                "scheduler_on": True,
            },
            "use_train_modified": True
        },
    ]
        
    results = []

    for model in models:
        print(f"\n--- Training: {model['name']} ---")

        if model["use_train_modified"]:
            train_modified(
                model=model["model"],
                lr=lr,
                lamb=lamb,
                train_data=train_matrix,
                zero_train_data=zero_train_matrix,
                valid_data=valid_data,
                num_epoch=num_epochs,
                denoising=model["kwargs"]["denoising"],
                early_stopping=model["kwargs"]["early_stopping"],
                scheduler_on=model["kwargs"]["scheduler_on"],
                batch_size=16,
            )
        else:
            train(
                model=model["model"],
                lr=lr,
                lamb=lamb,
                train_data=train_matrix,
                zero_train_data=zero_train_matrix,
                valid_data=valid_data,
                num_epoch=num_epochs,
            )

        test_acc = evaluate(model["model"], zero_train_matrix, test_data)
        print(f"{model['name']} → Test Accuracy: {test_acc:.4f}")
        results.append((model["name"], test_acc))

    print("\n===== Summary of Test Accuracies =====")
    for name, acc in results:
        print(f"{name}: {acc:.4f}")


def main():
        
    #######    BASELINE MODEL VS MODIFIED MODEL #######
    compare_baseline_vs_enhanced()
    run_comparisons()
   

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
