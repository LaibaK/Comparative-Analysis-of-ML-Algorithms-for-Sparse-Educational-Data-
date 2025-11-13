from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Apply sigmoid function."""
    return 1 / (1 + np.exp(-x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    user_idx = np.array(data["user_id"])
    question_idx = np.array(data["question_id"])
    c = np.array(data["is_correct"])

    x = theta[user_idx] - beta[question_idx]  
    p = sigmoid(x)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1 - epsilon)

    log_lklihood = np.sum(c * np.log(p) + (1 - c) * np.log(1 - p))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_idx = np.array(data["user_id"])
    question_idx = np.array(data["question_id"])
    c = np.array(data["is_correct"])

    x = theta[user_idx] - beta[question_idx]
    p = sigmoid(x)
    diff = c - p

    d_theta = np.zeros_like(theta)
    d_beta = np.zeros_like(beta)

    np.add.at(d_theta, user_idx, diff)
    np.add.at(d_beta, question_idx, -diff)

    theta += lr * d_theta
    beta += lr * d_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.normal(0, 0.01, max(data["user_id"]) + 1)
    beta = np.random.normal(0, 0.01, max(data["question_id"]) + 1)

    train_lld_lst = []
    val_lld_lst = []
    train_acc_lst = []
    val_acc_lst = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta, beta)
        val_neg_lld = neg_log_likelihood(val_data, theta, beta)
        train_acc = evaluate(data, theta, beta)
        val_acc = evaluate(val_data, theta, beta)
        train_avg_nll = train_neg_lld / len(data["user_id"])
        val_avg_nll = val_neg_lld / len(val_data["user_id"])

        train_lld_lst.append(-train_avg_nll)
        val_lld_lst.append(-val_avg_nll)
        train_acc_lst.append(train_acc)
        val_acc_lst.append(val_acc)

        print(f"Iter {i+1}: Train Avg LL = {-train_avg_nll:.4f}, Val Avg LL = {-val_avg_nll:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_lld_lst, val_lld_lst, train_acc_lst, val_acc_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    user_idx = np.array(data["user_id"])
    question_idx = np.array(data["question_id"])
    true_labels = np.array(data["is_correct"])

    x = theta[user_idx] - beta[question_idx]
    preds = sigmoid(x) >= 0.5
    return np.mean(preds == true_labels)


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.001
    iterations = 140

    theta, beta, train_lld, val_lld, train_acc, val_acc = irt(train_data, val_data, lr, iterations)

    print(f"Final Train Accuracy: {train_acc[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_acc[-1]:.4f}")
    test_acc = evaluate(test_data, theta, beta)
    print(f"Test Accuracy: {test_acc:.4f}")

    iterations = range(1, len(train_lld) + 1)
    plt.figure(figsize=(10,6))
    plt.plot(iterations, train_lld, label="Training Log-Likelihood")
    plt.plot(iterations, val_lld, label="Validation Log-Likelihood")
    plt.xlabel("Iteration")
    plt.ylabel("Average Log-Likelihood")
    plt.title("Average Training and Validation Log-Likelihood over Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    questions = [5, 100, 200]
    theta_range = (-4, 4)
    num_points = 100
    theta_values = np.linspace(theta_range[0], theta_range[1], num_points)

    plt.figure(figsize=(8, 6))
    for j in questions:
        p_correct = 1 / (1 + np.exp(-(theta_values - beta[j])))
        plt.plot(theta_values, p_correct, label=f"Question {j}")

    plt.xlabel("User Ability θ")
    plt.ylabel("Probability of Correct Response")
    plt.title("Probability of Correct Response vs. User Ability (θ)")
    plt.legend()
    plt.grid(True)
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
