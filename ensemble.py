from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
from item_response import irt 
from sklearn.impute import KNNImputer
from neural_network import AutoEncoder, load_data, train
import torch
from torch.autograd import Variable

def evaluate_preds(data, preds):
    """
    Compute accuracy by comparing predictions to ground truth labels.

    :param data: dict with "is_correct"
    :param preds: array of predicted probabilities or bools
    :return: accuracy (float)
    """
    binary_preds = np.array(preds) >= 0.5
    return np.mean(binary_preds == np.array(data["is_correct"]))

def ensemble_knn_user(train_matrix, val_data, test_data, k=11, frac=0.8):
    """
    Train an ensemble of user-based KNN imputation models with bootstrapped training data.

    Each model is trained on a bootstrapped version of the training matrix with 20% missing data masked.

    :param train_matrix: np.array, training data matrix with missing values as np.nan
    :param val_data: dict, validation dataset
    :param test_data: dict, test dataset
    :param k: int, number of neighbors for KNN imputation
    :param frac: fraction of data to sample for bootstrap (default 0.8)
    :return: tuple (val_preds, test_preds) validation and test predictions after ensemble
    """

    # Bootstrap by masking 20% of entries
    mask = torch.rand(train_matrix.shape) < frac
    boot_matrix = train_matrix.copy()
    boot_matrix[~mask.numpy()] = np.nan

    imputer = KNNImputer(n_neighbors=k)
    imputed_matrix = imputer.fit_transform(boot_matrix)

    # Predict on validation and test sets
    val_preds = [imputed_matrix[u, q] >= 0.5 for u, q in zip(val_data["user_id"], val_data["question_id"])]
    val_acc = np.mean([pred == actual for pred, actual in zip(val_preds, val_data["is_correct"])])
    test_preds = [imputed_matrix[u, q] >= 0.5 for u, q in zip(test_data["user_id"], test_data["question_id"])]
    test_acc = np.mean([pred == actual for pred, actual in zip(test_preds, test_data["is_correct"])])
    print("Validation Accuracy (User-based KNN): {}".format(val_acc))
    print("Test Accuracy (User-based KNN): {}".format(test_acc))
    return val_preds, test_preds

def ensemble_nn(train_matrix, zero_train_matrix, val_data, test_data,
                k, lr, lamb, num_epochs, frac=0.8):
    """
    Train an ensemble of neural network AutoEncoder models with bootstrapped training data.

    Each model is trained on a bootstrapped training matrix with 20% of entries masked.

    :param train_matrix: torch.FloatTensor, training data with NaNs for missing entries
    :param zero_train_matrix: torch.FloatTensor, training data with missing entries filled as zeros
    :param val_data: dict, validation dataset
    :param test_data: dict, test dataset
    :param k: int, latent dimension size of AutoEncoder hidden layer
    :param lr: float, learning rate
    :param lamb: float, regularization parameter
    :param num_epochs: int, number of epochs to train each model
    :param frac: fraction of data to sample for bootstrap (default 0.8)
    :return: tuple (val_preds, test_preds) validation and test predictions after ensemble
    """
    mask = torch.rand(train_matrix.shape) < frac
    boot_matrix = train_matrix.clone()
    boot_matrix[~mask] = float('nan')
    boot_matrix = torch.FloatTensor(boot_matrix)
        
    zero_boot_matrix = boot_matrix.clone()
    zero_boot_matrix[torch.isnan(zero_boot_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
        
    # Train the AutoEncoder model
    model = AutoEncoder(train_matrix.shape[1], k)
    train(model, lr, lamb, boot_matrix, zero_boot_matrix, val_data, num_epochs)

    # Get predictions on validation and test data
    model.eval()
    val_preds = []
    for i, u in enumerate(val_data["user_id"]):
        inputs = Variable(zero_train_matrix[u]).unsqueeze(0)
        output = model(inputs)
        val_preds.append(output[0][val_data["question_id"][i]].item())

    test_preds = []
    for i, u in enumerate(test_data["user_id"]):
        inputs = Variable(zero_train_matrix[u]).unsqueeze(0)
        output = model(inputs)
        test_preds.append(output[0][test_data["question_id"][i]].item())

    print("Test Accuracy (Neural Network): {}".format(np.mean(np.array(test_preds) >= 0.5)))
    return val_preds, test_preds

def ensemble_irt(train_data, val_data, test_data, lr, iterations, frac=0.8):
    """
    Train an ensemble of IRT models with bootstrapped training data.

    Each model is trained on a bootstrapped training matrix with 20% of entries masked.

    :param train_data: dict, training dataset
    :param val_data: dict, validation dataset
    :param test_data: dict, test dataset
    :param lr: float, learning rate
    :param iterations: int, number of iterations to train each model
    :param frac: fraction of data to sample for bootstrap (default 0.8)
    :return: tuple (val_preds, test_preds) validation and test predictions after ensemble"""
    n = len(train_data["user_id"])
    indices = np.random.choice(n, int(n * frac), replace=True)
    indices = np.array(indices)

    boot_data = {
        "user_id": np.array(train_data["user_id"])[indices].tolist(),
        "question_id": np.array(train_data["question_id"])[indices].tolist(),
        "is_correct": np.array(train_data["is_correct"])[indices].tolist(),
    }

    theta, beta, *_ = irt(boot_data, val_data, lr, iterations)

    val_u = np.array(val_data["user_id"])
    val_q = np.array(val_data["question_id"])
    val_preds = 1 / (1 + np.exp(-(theta[val_u] - beta[val_q])))

    test_u = np.array(test_data["user_id"])
    test_q = np.array(test_data["question_id"])
    test_preds = 1 / (1 + np.exp(-(theta[test_u] - beta[test_q])))

    print("Test Accuracy (IRT): {:.4f}".format(np.mean(test_preds >= 0.5)))
    return val_preds.tolist(), test_preds.tolist()

def main():
    # Load data
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    sparse_matrix = load_train_sparse("./data").toarray()
    zero_train_matrix, train_matrix, val_data_nn, test_data_nn = load_data()

    val_knn_acc = 0.0
    val_nn_acc = 0.0
    val_irt_acc = 0.0

    print("Training KNN (user-based)...")
    val_knn_preds, test_knn_preds = ensemble_knn_user(sparse_matrix, val_data, test_data, k=11)
    val_knn_preds = np.array(val_knn_preds, dtype=float)
    test_knn_preds = np.array(test_knn_preds, dtype=float)
    val_knn_acc = evaluate_preds(val_data, val_knn_preds)

    print("\nTraining IRT...")
    val_irt_preds, test_irt_preds = ensemble_irt(train_data, val_data, test_data,
                                                lr=0.001, iterations=140)
    val_irt_preds = np.array(val_irt_preds)
    test_irt_preds = np.array(test_irt_preds)
    val_irt_acc = evaluate_preds(val_data, val_irt_preds)

    print("\nTraining Neural Network...")
    val_nn_preds, test_nn_preds = ensemble_nn(train_matrix, zero_train_matrix, val_data_nn, test_data_nn,
                                            k=100, lr=0.01, lamb=0.001, num_epochs=80)
    val_nn_preds = np.array(val_nn_preds)
    test_nn_preds = np.array(test_nn_preds)
    val_nn_acc = evaluate_preds(val_data, val_nn_preds)

    print(f"\nValidation Accuracy KNN: {val_knn_acc:.4f}")
    print(f"Validation Accuracy IRT: {val_irt_acc:.4f}")
    print(f"Validation Accuracy NN: {val_nn_acc:.4f}")

    # Use validation accuracies as weights
    weights = np.array([val_knn_acc, val_irt_acc, val_nn_acc])
    weights_sum = np.sum(weights)
    normalized_weights = weights / weights_sum

    print(f"Normalized Weights: {normalized_weights}")

    # Weighted average ensemble predictions
    val_avg = (normalized_weights[0] * val_knn_preds +
               normalized_weights[1] * val_irt_preds +
               normalized_weights[2] * val_nn_preds)

    test_avg = (normalized_weights[0] * test_knn_preds +
                normalized_weights[1] * test_irt_preds +
                normalized_weights[2] * test_nn_preds)

    val_acc = np.mean(val_avg >= 0.5)
    test_acc = np.mean(test_avg >= 0.5)

    print(f"\nEnsemble Validation Accuracy: {val_acc:.4f}")
    print(f"Ensemble Test Accuracy: {test_acc:.4f}")

    
if __name__ == "__main__":
    main()
