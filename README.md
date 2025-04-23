# Python Hello World Complexity Benchmark

This project is a Streamlit-based web application that demonstrates different algorithms to generate the string "Hello, World!" with varying computational complexities. It serves as both an educational tool and a benchmarking suite to compare the performance of creative, humorous, and intentionally over-engineered approaches to a simple task.

## Features

- **10 Unique Algorithms**: Ranging from a simple direct return (O(1)) to a highly inefficient Bogo Sort permutation (O(n · n!) average case), showcasing a spectrum of computational complexities.
- **Interactive UI**: Select algorithms via a slider, run benchmarks, and view intermediate steps, final results, and performance metrics.
- **Performance Metrics**: Measures execution time and peak memory usage using Python's `tracemalloc` and `time.perf_counter`.
- **Theoretical Complexity**: Displays Big-O time and space complexities for each algorithm.
- **Explanations and Code Snippets**: Provides simplified explanations and code examples for educational purposes.
- **Generator-Based Algorithms**: Algorithms use Python generators (`yield`) to show intermediate steps, enhancing visualization of the process.

## Algorithms Included

1. **Direct Return**: Returns "Hello, World!" directly (O(1)).
2. **Random Choice Loop**: Combines random greetings and names until the target is hit.
3. **String Concatenation**: Builds the string character by character.
4. **List Join**: Uses a list of characters and joins them.
5. **Quantum Simulation**: Simulates quantum gates to generate characters (highly simplified).
6. **FFT Noise**: Applies FFT, adds noise, and recovers the string.
7. **Hash Nonce (Proof-of-Work)**: Finds a nonce for a hash prefix, mimicking crypto mining.
8. **Genetic Algorithm**: Evolves a population of strings toward the target.
9. **Random Guessing (Incremental)**: Guesses characters one by one.
10. **Bogo Sort (Permutation)**: Shuffles characters until the correct order is found.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/iamkairoth/hello-python.git
   cd hello-python
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include:
   ```
   streamlit
   numpy
   scipy
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

   This will launch the app in your default web browser.

## Usage

- **Select an Algorithm**: Use the slider to choose an algorithm (1 = simplest, 10 = most complex).
- **Run the Benchmark**: Click the "Run Benchmark" button to execute the selected algorithm.
- **View Results**:
  - **Intermediate Steps**: Displayed in a scrollable text area (last 20 steps).
  - **Final Result**: Shown in a code block.
  - **Performance Metrics**: Execution time (seconds) and peak memory usage (KiB).
  - **Theoretical Complexity**: Big-O notation for time and space.
  - **Explanation**: Expand the section to read about the algorithm and see a simplified code snippet.

## Project Structure

```
hello-python/
├── app.py              # Main Streamlit application script
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore file
```

## Notes

- **Performance Variability**: Benchmark times may vary due to system load, random algorithm outcomes, or Python's internal optimizations.
- **Educational Purpose**: Some algorithms (e.g., Quantum Simulation, FFT Noise) are intentionally whimsical and not practical for real-world use.
- **Safety Limits**: Algorithms like Bogo Sort and Random Guessing include maximum attempt limits to prevent infinite loops.

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## Issues and Feedback

Report bugs or suggest improvements via the [GitHub Issues page](https://github.com/iamkairoth/hello-python).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## About

Created by Kai Roth. For more details, visit the [project repository](https://github.com/iamkairoth/hello-python).