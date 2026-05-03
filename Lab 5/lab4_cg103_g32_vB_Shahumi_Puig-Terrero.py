from abc import abstractmethod, ABC
from typing import List
import numpy as np

class Layer(ABC):
    """Basic building block of the Neural Network"""

    def __init__(self) -> None:
        self._learning_rate = 0.01

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through layer"""
        pass

    @abstractmethod
    def backward(self, output_error_derivative) -> np.ndarray:
        """Backward propagation of output_error_derivative through layer"""
        pass

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert learning_rate < 1, f"Given learning_rate={learning_rate} is larger than 1"
        assert learning_rate > 0, f"Given learning_rate={learning_rate} is smaller than 0"
        self._learning_rate = learning_rate


class FullyConnected(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        # Initialization works well for stability with these activations
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, output_error_derivative) -> np.ndarray:
        # Compute derivatives
        input_error_derivative = np.dot(output_error_derivative, self.weights.T)
        weights_derivative = np.dot(self.input.T, output_error_derivative)
        bias_derivative = np.sum(output_error_derivative, axis=0, keepdims=True)
        
        # Apply gradients directly
        self.weights -= self.learning_rate * weights_derivative
        self.bias -= self.learning_rate * bias_derivative
        
        return input_error_derivative


# ---------------------------------------------------------------------------
# Implementation of Activation functions (Tanh, Sigmoid, ReLU, LeakyReLU)
# ---------------------------------------------------------------------------
class ActivationLayer(Layer):
    """Helper class to prevent repeating code for all activation layers"""
    def __init__(self, activation: callable, activation_prime: callable):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return self.activation(self.input)

    def backward(self, output_error_derivative) -> np.ndarray:
        return self.activation_prime(self.input) * output_error_derivative


class Tanh(ActivationLayer):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)


class Sigmoid(ActivationLayer):
    def __init__(self):
        # Using np.clip to prevent overflow errors in exp
        sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        sig_prime = lambda x: sig(x) * (1 - sig(x))
        super().__init__(sig, sig_prime)


class ReLU(ActivationLayer):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0).astype(float)
        super().__init__(relu, relu_prime)


class LeakyReLU(ActivationLayer):
    def __init__(self, alpha=0.01):
        lrelu = lambda x: np.where(x > 0, x, alpha * x)
        lrelu_prime = lambda x: np.where(x > 0, 1.0, alpha)
        super().__init__(lrelu, lrelu_prime)


# ---------------------------------------------------------------------------
# Loss & Network Architecture
# ---------------------------------------------------------------------------
class Loss:
    def __init__(self, loss_function: callable, loss_function_derivative: callable) -> None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss_function(y_pred, y_true)

    def loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.loss_function_derivative(y_pred, y_true)

# Providing a basic MSE implementation so the code can compile and run natively
def mse_loss(y_pred, y_true):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size


class Network:
    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        for layer in self.layers:
            layer.learning_rate = self.learning_rate

    def compile(self, loss: Loss) -> None:
        self.loss = loss

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            learning_rate: float,
            verbose: int = 0) -> List[float]:
        
        # Ensure all layers are using the fit-provided learning rate
        for layer in self.layers:
            layer.learning_rate = learning_rate
            
        history = []
        
        for epoch in range(epochs):
            err = 0
            # Basic SGD: iteration sample-by-sample
            for j in range(len(x_train)):
                # Slicing like [j:j+1] keeps the matrix shape as (1, input_size)
                x = x_train[j:j+1] 
                y = y_train[j:j+1]
                
                # Forward pass
                output = self.__call__(x)
                
                # Compute loss
                err += self.loss.loss(output, y)
                
                # Backward pass
                error = self.loss.loss_derivative(output, y)
                for layer in reversed(self.layers):
                    error = layer.backward(error)
                    
            err /= len(x_train)
            history.append(err)
            
            if verbose and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {err:.6f}")
                
        return history

# ---------------------------------------------------------------------------
# Data loading & Execution wrapper (Group B - 3 Seeds & Scalability)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    print("Fetching Fashion-MNIST dataset... (This might take a moment)")
    fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
    
    X = fmnist.data / 255.0
    y = fmnist.target.astype(int)
    y_onehot = np.eye(10)[y]
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=0)

    # NOTE: Subsetting data for faster execution while testing, original subet contains 
    #       60000 samples for training and 10000 for testing
    # COMMENT OUT the two lines below to train on the full dataset for full final numbers!
    X_train, y_train = X_train[:10000], y_train[:10000]
    X_test, y_test = X_test[:2000], y_test[:2000]

    epochs = 10
    learning_rate = 0.1
    seeds = [42, 7, 99] # The 3 seeds we will use for rigorous testing
    
    # Helper function to calculate classification accuracy
    def get_accuracy(net, X, y_onehot):
        predictions = net(X)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_onehot, axis=1)
        return np.mean(pred_labels == true_labels) * 100
    
    # =======================================================================
    # PHASE 1: ACTIVATION FUNCTION COMPARISON (OVER 3 SEEDS)
    # =======================================================================
    activations_to_test = {
        "Tanh": Tanh,
        "Sigmoid": Sigmoid,
        "ReLU": ReLU,
        "LeakyReLU": LeakyReLU
    }
    
    # Dictionaries to store the aggregated results
    final_accuracies = {name: [] for name in activations_to_test}
    avg_histories = {name: np.zeros(epochs) for name in activations_to_test}
    
    print(f"\n--- PHASE 1: Training Activations over {len(seeds)} Seeds ---")
    
    for name, ActClass in activations_to_test.items():
        print(f"\nEvaluating {name}...")
        for seed in seeds:
            np.random.seed(seed) # Set the seed for this specific run
            
            if name == "LeakyReLU":
                act_layer1, act_layer2 = ActClass(alpha=0.01), ActClass(alpha=0.01)
            else:
                act_layer1, act_layer2 = ActClass(), ActClass()

            net = Network([
                FullyConnected(784, 128), act_layer1,
                FullyConnected(128, 64), act_layer2,
                FullyConnected(64, 10), Sigmoid()
            ], learning_rate=learning_rate)

            mse = Loss(mse_loss, mse_prime)
            net.compile(mse)

            # Train silently (verbose=0) so the terminal isn't flooded by 3x runs
            history = net.fit(X_train, y_train, epochs=epochs, learning_rate=learning_rate, verbose=0)
            
            # Evaluate accuracy
            acc = get_accuracy(net, X_test, y_test)
            final_accuracies[name].append(acc)
            avg_histories[name] += np.array(history)
            
            print(f"  Seed {seed} -> Accuracy: {acc:.2f}% | Final Loss: {history[-1]:.6f}")
            
        # Average the loss history for plotting
        avg_histories[name] /= len(seeds)

    # Calculate Mean and Std for terminal output
    print("\n--- Phase 1 Final Results (Mean ± Std) ---")
    mean_results = {}
    for name in activations_to_test:
        mean_acc = np.mean(final_accuracies[name])
        std_acc = np.std(final_accuracies[name])
        mean_results[name] = mean_acc
        print(f"{name}: {mean_acc:.2f}% ± {std_acc:.2f}%")

    # Generate Matplotlib Figure (Averaged Activations)
    plt.figure(figsize=(10, 6))
    colors = {'Tanh': 'blue', 'Sigmoid': 'red', 'ReLU': 'green', 'LeakyReLU': 'orange'}
    for name, history in avg_histories.items():
        plt.plot(range(1, epochs + 1), history, label=name, color=colors[name], marker='o', linewidth=2)
        
    plt.title('Average Training Loss over Epochs (Combined across 3 Seeds)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig("activation_comparison_graph_averaged.png", dpi=300)
    
    # =======================================================================
    # ANNOUNCEMENT OF THE WINNER
    # =======================================================================
    # Winner is now based on highest average accuracy across the 3 seeds
    winner_name = max(mean_results, key=mean_results.get)
    
    print("\n" + "="*60)
    print(f" WINNER ANNOUNCEMENT ")
    print(f"Based on 3 distinct seeds, the most robust activation is: {winner_name}!")
    print("Proceeding to Phase 2: Architectural Scalability...")
    print("="*60)

    # =======================================================================
    # PHASE 2: ARCHITECTURAL SCALABILITY (Using winning activation)
    # =======================================================================
    # Since Phase 1 dynamically picks the winner, we just map it here:
    WinningActClass = activations_to_test[winner_name]
    
    def get_winning_act():
        return WinningActClass(alpha=0.01) if winner_name == "LeakyReLU" else WinningActClass()

    topologies = {
        "Baseline (128-64)": [
            FullyConnected(784, 128), get_winning_act(),
            FullyConnected(128, 64), get_winning_act(),
            FullyConnected(64, 10), Sigmoid()
        ],
        "Shallow/Wide (256)": [
            FullyConnected(784, 256), get_winning_act(),
            FullyConnected(256, 10), Sigmoid()
        ],
        "Deep/Narrow (64-32-16)": [
            FullyConnected(784, 64), get_winning_act(),
            FullyConnected(64, 32), get_winning_act(),
            FullyConnected(32, 16), get_winning_act(),
            FullyConnected(16, 10), Sigmoid()
        ]
    }
    
    accuracies = {}
    
    for name, layers in topologies.items():
        print(f"\n--- Training {name} Topology ---")
        np.random.seed(42) # Keep seed fixed for fair topology comparison
        
        net = Network(layers, learning_rate=learning_rate)
        mse = Loss(mse_loss, mse_prime)
        net.compile(mse)
        
        net.fit(X_train, y_train, epochs=epochs, learning_rate=learning_rate, verbose=1)
        
        acc = get_accuracy(net, X_test, y_test)
        accuracies[name] = acc
        print(f">>> {name} Test Accuracy: {acc:.2f}%")
        
    # Generate Matplotlib Bar Chart (Topologies)
    plt.figure(figsize=(9, 6))
    bars = plt.bar(list(accuracies.keys()), list(accuracies.values()), color=['#4C72B0', '#55A868', '#C44E52'])
    plt.title(f'Test Accuracy by Network Topology (Using {winner_name})')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100) 
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.savefig("topology_comparison_bar.png", dpi=300)
    print("\nAll experiments complete! Displaying graphs on screen...")
    plt.show()
