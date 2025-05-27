import tensorflow as tf
import numpy as np
import time

# Set random seed for reproducibility
np.random.seed(42)

# Function to run a benchmark on a specific device
def run_benchmark(device, spatial_size=128, n_channels=32, kernel_size=3, n_iter=100):
    # Generate random input data
    input_data = np.random.rand(1, spatial_size, spatial_size, n_channels).astype(np.float32)
    
    # Create a convolutional layer
    conv_layer = tf.keras.layers.Conv2D(filters=n_channels, kernel_size=kernel_size, padding='same')

    # Transfer input data to a TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_data)

    # Set up the computation on the specified device
    with tf.device(device):
        # Warm-up to ensure the GPU is ready
        for _ in range(10):
            _ = conv_layer(input_tensor)
        
        # Benchmark the operation
        start_time = time.time()
        for _ in range(n_iter):
            _ = conv_layer(input_tensor)
        avg_time = (time.time() - start_time) / n_iter

    print(f"Average execution time on {device}: {avg_time:.4f} seconds per iteration")

# Running the benchmark on CPU and GPU
print("TensorFlow Benchmark - CPU vs GPU")
print("=" * 50)

# Run on CPU
run_benchmark(device='/CPU:0')

# Check if a GPU is available
if tf.config.list_physical_devices('GPU'):
    # Run on GPU
    run_benchmark(device='/GPU:0')
else:
    print("No GPU available. Please ensure that TensorFlow is set up with GPU support.")
