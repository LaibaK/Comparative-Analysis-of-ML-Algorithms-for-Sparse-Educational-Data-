import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # used NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    matrix_transposed = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    mat_transposed = nbrs.fit_transform(matrix_transposed)
    mat = mat_transposed.T

    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy (Item-based): {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    
    user_based_accuracies = []
    item_based_accuracies = []
    
    print("=== USER-BASED COLLABORATIVE FILTERING ===")
    for k in k_values:
        print(f"\nTesting k = {k}")
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_based_accuracies.append(acc)
    
    print("\n=== ITEM-BASED COLLABORATIVE FILTERING ===")
    for k in k_values:
        print(f"\nTesting k = {k}")
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        item_based_accuracies.append(acc)
    
    # find best k for each method
    best_k_user_idx = np.argmax(user_based_accuracies)
    best_k_user = k_values[best_k_user_idx]
    best_acc_user = user_based_accuracies[best_k_user_idx]
    
    best_k_item_idx = np.argmax(item_based_accuracies)
    best_k_item = k_values[best_k_item_idx]
    best_acc_item = item_based_accuracies[best_k_item_idx]
    
    print(f"\n=== RESULTS ===")
    print(f"Best k for user-based: {best_k_user} (validation accuracy: {best_acc_user:.4f})")
    print(f"Best k for item-based: {best_k_item} (validation accuracy: {best_acc_item:.4f})")
    
    print(f"\n=== TEST SET EVALUATION ===")
    print(f"User-based (k={best_k_user}):")
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    
    print(f"Item-based (k={best_k_item}):")
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)
    
    print(f"\nFinal test accuracies:")
    print(f"User-based: {test_acc_user:.4f}")
    print(f"Item-based: {test_acc_item:.4f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, user_based_accuracies, 'b-o', label='User-based', linewidth=2, markersize=8)
    plt.plot(k_values, item_based_accuracies, 'r-s', label='Item-based', linewidth=2, markersize=8)
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Validation Accuracy')
    plt.title('KNN Performance vs k')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    methods = ['User-based', 'Item-based']
    test_accuracies = [test_acc_user, test_acc_item]
    colors = ['blue', 'red']
    plt.bar(methods, test_accuracies, color=colors, alpha=0.7)
    plt.ylabel('Test Accuracy')
    plt.title('Test Set Performance Comparison')
    plt.ylim(0, max(test_accuracies) * 1.1)
    
    for i, v in enumerate(test_accuracies):
        plt.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('knn_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== SUMMARY FOR REPORT ===")
    print(f"User-based collaborative filtering:")
    print(f"  - Best k*: {best_k_user}")
    print(f"  - Validation accuracy: {best_acc_user:.4f}")
    print(f"  - Test accuracy: {test_acc_user:.4f}")
    print(f"\nItem-based collaborative filtering:")
    print(f"  - Best k*: {best_k_item}")
    print(f"  - Validation accuracy: {best_acc_item:.4f}")
    print(f"  - Test accuracy: {test_acc_item:.4f}")
    
    if test_acc_user > test_acc_item:
        print(f"\nUser-based performs better by {test_acc_user - test_acc_item:.4f}")
    else:
        print(f"\nItem-based performs better by {test_acc_item - test_acc_user:.4f}")
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
