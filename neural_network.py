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

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    #k = None
    #model = None

    # Set optimization hyperparameters.
    #lr = None
    #num_epoch = None
    #lamb = None

    #train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    # Next, evaluate your network on validation/test data


    # Set model hyperparameters.

    ########################### PART C #####################################
    
    k_val = [10, 50, 100, 200, 500]
    lr = 0.01
    num_epoch = 80

    validation_accuracies = []
    models = []

    for k in k_val:
        print(f"\nTraining with k = {k}")
        model = AutoEncoder(train_matrix.shape[1], k)
        train(model, lr=lr, lamb=0.0, train_data=train_matrix,
              zero_train_data=zero_train_matrix, valid_data=valid_data,
              num_epoch=num_epoch)
        
        validation_accuracy = evaluate(model, zero_train_matrix, valid_data)
        print(f"Validation Accuracy: {validation_accuracy:.4f}")
        validation_accuracies.append(validation_accuracy)
        models.append(model)

    # Plot validation accuracy vs k
    plt.figure()
    plt.plot(k_val, validation_accuracies, marker='o')
    plt.title("Validation Accuracy vs Latent Dimension k")
    plt.xlabel("Latent Dimension k")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.show()

    # Choose best k
    best_index = validation_accuracies.index(max(validation_accuracies))
    best_k = k_val[best_index]
    best_model = models[best_index]

    print(f"The best k is {best_k} with the validation accuracy, {validation_accuracies[best_index]:.4f}")
    test_accuracy = evaluate(best_model, zero_train_matrix, test_data)
    print(f"The test accuracy with best k={best_k} is {test_accuracy:.4f}")    
    

    ########################## PART D ################################ 

    """

    # Set optimization hyperparameters.
    k = 10
    lr = 0.01
    num_epochs = 80
    lamb = 0

    print(f'Configuration: k:{k}, learning rate: {lr}, number of epochs:{num_epoch}, lambda:{lamb}')
    
    model = AutoEncoder(train_matrix.shape[1], k)
    train_losses, valid_accuracies = train(
        model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epochs
    )

    # Plot Training Loss and Validation Accuracy
    epochs = list(range(num_epochs))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, valid_accuracies, label="Validation Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy over Epochs")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f'Test Accuracy: {test_acc}')
    

    """
    
    ####################### PART E ############################
    """

    # Set hyperparameters 
    best_k = 10
    lr = 0.01
    num_epochs = 80
    lambdas = [0, 0.001, 0.01, 0.1, 1]  

    best_val_acc = 0
    best_lamb = None
    best_model = None

    results = {}

    for lamb in lambdas:
        print(f"\nTraining with λ = {lamb}")
        model = AutoEncoder(train_matrix.shape[1], best_k)
        train_losses, valid_accuracies = train(
            model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epochs
        )

        final_val_acc = valid_accuracies[-1]
        test_acc = evaluate(model, zero_train_matrix, test_data) 

        results[lamb] = {
            'train_losses': train_losses,
            'valid_accuracies': valid_accuracies,
            'val_acc': final_val_acc,
            'test_acc': test_acc,
        }

        print(f" λ = {lamb} has a Final Validation Accuracy of {final_val_acc:.4f} and Test Accuracy of {test_acc:.4f}")

        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_lamb = lamb
            best_model = model


    print(f"\n The best λ is {best_lamb} with Validation Accuracy: {best_val_acc:.4f}")

    # Plot comparison of validation accuracy across different λ
    plt.figure(figsize=(8, 6))
    for lamb in lambdas:
        plt.plot(results[lamb]['valid_accuracies'], label=f"λ={lamb}")
    plt.title("Validation Accuracy vs. Epoch for Different λ")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evaluate best model on test set
    test_acc = evaluate(best_model, zero_train_matrix, test_data)
    print(f" The best λ is {best_lamb} with Final Test Accuracy of {test_acc:.4f}")

    """
    
   

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
