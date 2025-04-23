import streamlit as st
import time
import tracemalloc
import random
import string
import hashlib
import numpy as np
from scipy.fft import fft, ifft
import math # Needed for factorial in complexity string


st.set_page_config(
    page_title="Hello Python - KaiRoth",
    page_icon="🤍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/iamkairoth/hello-python',
        'Report a bug': "https://github.com/iamkairoth/hello-python",
        'About': "# Python Hello World Complexity Benchmark by Kai Roth"
    }
)

st.title("Python Hello World Complexity Benchmark")

TARGET_STRING = "Hello, World"
TARGET_LEN = len(TARGET_STRING)

# --- Algorithm Definitions ---
# NOTE: Algorithms modified to be generators (using yield)
#       to show intermediate steps where applicable.

def algo_direct_return():
    """1. Direct Return: The simplest way."""
    yield TARGET_STRING # Yield final result

def algo_random_choice_loop():
    """2. Random Choice Loop: Randomly pick greetings and names until target is hit (or max attempts)."""
    salutations = ["Hello,", "Hi,", "Hola,", "Hey,", "Greetings,"] # Added commas
    addressees = ["World", "Sir", "Ma’am", "Friend", "Lady", "Pal"] # Added exclamation
    attempts = 0
    max_attempts = 1000 # Safety break

    while attempts < max_attempts:
        sal = random.choice(salutations)
        addr = random.choice(addressees)
        current_string = f"{sal} {addr}".replace(",  ", ", ") # Basic formatting
        yield current_string # Show attempt
        if current_string == TARGET_STRING:
            return # Stop if target found
        attempts += 1
    yield TARGET_STRING # Fallback if not found

def algo_string_concat():
    """3. String Concatenation: Build the string character by character using +=."""
    s = ""
    for c in TARGET_STRING:
        s += c
        # yield s # Optional: yield intermediate build-up if desired
    yield s # Yield final result

def algo_list_join():
    """4. List Join: Put characters in a list and join them."""
    chars = ["H", "e", "l", "l", "o", ",", " ", "W", "o", "r", "l", "d", "!"]
    yield "".join(chars) # Yield final result

def algo_quantum_sim():
    """5. Quantum Simulation: Simulate quantum gates based on char codes (highly simplified)."""
    class QuantumSimulator:
        def __init__(self):
             # Simulate a simple state, not true qubits
            self.state = random.random() * 2 * np.pi

        def apply_gate(self, char):
            # Rotate state based on char code - arbitrary simulation
            self.state += (ord(char) % 128) / 128 * np.pi
            self.state %= (2 * np.pi)

        def measure(self):
            # Simulate measurement based on state angle - arbitrary
            # More likely to be '1' if closer to pi, '0' if closer to 0/2pi
            prob_1 = (np.sin(self.state) + 1) / 2 # Map sin to [0, 1]
            return '1' if random.random() < prob_1 else '0'

    result = ""
    for target_char in TARGET_STRING:
        sim = QuantumSimulator()
        sim.apply_gate(target_char)
        binary = ''
        # Simulate getting 8 bits for the char code
        for _ in range(8):
            # In a real scenario, measurement collapses state, here we simplify
            # Re-applying gate and measuring is not how quantum computing works!
            # This is just for achieving a 'random-like' process influenced by the char
            sim.apply_gate(target_char) # Simplified re-application
            binary += sim.measure()

        try:
            char_code = int(binary, 2) % 128
            generated_char = chr(char_code)
            # Use the target character if generated one isn't printable or likely wrong
            result += generated_char if generated_char in string.printable else target_char
        except:
             result += target_char # Fallback if int conversion fails

        # yield result # Optional: yield intermediate build-up

    # Ensure the final result is correct, overriding simulation errors
    yield TARGET_STRING

def algo_fft_noise():
    """6. FFT Noise: Convert to signal, add noise in frequency domain, convert back."""
    signal = np.array([ord(c) for c in TARGET_STRING], dtype=float)
    freq = fft(signal)

    # Add scaled noise
    noise_level = 0.1 # Can be adjusted
    noise = np.random.normal(0, noise_level, len(freq)) + 1j * np.random.normal(0, noise_level, len(freq))
    noisy_freq = freq + noise

    recovered_signal = ifft(noisy_freq).real

    # Round and convert back to characters, clamping to valid ASCII range
    result_chars = []
    for x in recovered_signal:
        char_code = int(round(x))
        # Clamp to prevent errors with chr()
        clamped_code = max(0, min(127, char_code))
        try:
             result_chars.append(chr(clamped_code))
        except ValueError:
             # Fallback for safety, though clamping should prevent this
             result_chars.append('?')

    result = "".join(result_chars)

    # Yield the potentially corrupted result first? No, let's just yield the corrected one.
    # yield result # Optional: show noisy version first

    yield TARGET_STRING # Return correct target as fallback/guarantee

def algo_hash_nonce():
    """7. Hash Nonce: Find a nonce N such that hash(target + N) has a specific prefix."""
    # Use a simpler/shorter target hash prefix for faster demo
    target_hash_prefix = hashlib.sha256(TARGET_STRING.encode()).hexdigest()[:4]
    nonce = 0
    max_nonce = 500000 # Increased safety break

    while nonce <= max_nonce:
        block = f"{TARGET_STRING}{nonce}".encode()
        current_hash = hashlib.sha256(block).hexdigest()
        # if nonce % 1000 == 0: # Optionally yield progress
        #     yield f"Nonce: {nonce}, Hash: {current_hash[:10]}..."
        if current_hash.startswith(target_hash_prefix):
            yield TARGET_STRING # Found it
            return
        nonce += 1

    yield TARGET_STRING # Fallback if not found

def algo_genetic():
    """8. Genetic Algorithm: Evolve a population of strings towards the target."""
    POPULATION_SIZE = 100
    GENERATIONS = 200
    MUTATION_RATE = 0.1
    ELITISM = 10 # Keep top N individuals

    # Valid characters to use
    CHAR_SET = string.ascii_letters + string.digits + ",.! "

    def fitness(s):
        return sum(1 for a, b in zip(s, TARGET_STRING) if a == b)

    # Initialize population
    population = [''.join(random.choices(CHAR_SET, k=TARGET_LEN)) for _ in range(POPULATION_SIZE)]

    for gen in range(GENERATIONS):
        # Calculate fitness for all
        pop_with_fitness = [(ind, fitness(ind)) for ind in population]

        # Sort by fitness (descending)
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)

        best_individual, best_fitness = pop_with_fitness[0]
        yield f"Gen {gen}: {best_individual} (Fitness: {best_fitness}/{TARGET_LEN})" # Show best of generation

        if best_individual == TARGET_STRING:
            return # Target found

        # Selection (Elitism + Roulette Wheel/Tournament could be added, using simple top N here)
        parents = [ind for ind, fit in pop_with_fitness[:POPULATION_SIZE // 2]] # Select top half as potential parents

        # Elitism: Carry over the best directly
        new_population = [ind for ind, fit in pop_with_fitness[:ELITISM]]

        # Crossover & Mutation
        while len(new_population) < POPULATION_SIZE:
            # Choose parents (allow duplicates for simplicity)
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            # Crossover
            crossover_point = random.randint(1, TARGET_LEN - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]

            # Mutation
            mutated_child = ""
            for char in child:
                if random.random() < MUTATION_RATE:
                    mutated_child += random.choice(CHAR_SET)
                else:
                    mutated_child += char
            new_population.append(mutated_child)

        population = new_population

    yield TARGET_STRING # Fallback if not found within generations

def algo_random_guess_incremental():
    """9. Random Guessing (Incremental): Guess characters one by one, only keeping correct ones."""
    s = ""
    attempts = 0
    max_attempts_per_char = 5000 # Safety break per character
    char_set = string.ascii_letters + string.digits + ",.! " # Include possible chars

    while len(s) < TARGET_LEN:
        target_char = TARGET_STRING[len(s)]
        char_attempts = 0
        while char_attempts < max_attempts_per_char:
            guess = random.choice(char_set)
            attempts += 1
            if guess == target_char:
                s += guess
                yield s # Show progress
                break # Move to next character
            # Optional: yield failed attempts? Could be too verbose.
            # yield f"Trying: {s + guess}"
            char_attempts += 1
        else: # If loop finished without break (max_attempts_per_char reached)
            # print(f"Max attempts reached for char {len(s)}. Falling back.")
            yield TARGET_STRING # Fallback if stuck
            return

    # yield s # Final correct string already yielded in loop

def algo_bogo_sort():
    """10. Bogo Sort (Permutation): Randomly shuffle characters until correct."""
    chars = list(TARGET_STRING)
    attempts = 0
    max_attempts = 2025 # Safety break - Factorial is too large! 13! is huge.

    while attempts < max_attempts:
        random.shuffle(chars)
        current_string = "".join(chars)
        yield current_string # Show attempt
        if current_string == TARGET_STRING:
            return # Found it
        attempts += 1
        # Throttle yielding if it's too fast/overwhelming
        # if attempts % 100 != 0: continue

    yield TARGET_STRING # Fallback


# --- Algorithm Mapping and Complexity (Ordered by Complexity) ---

# Pair the function with its description and a simplified code snippet
# Descriptions match the function docstrings now.
ALGO_INFO = {
    1: {
        "func": algo_direct_return,
        "name": "Direct Return",
        "time": "$O(1)$", "space": "$O(1)$",
        "explanation": "The most straightforward method: simply return the hardcoded string.",
        "snippet": 'return "Hello, World!"'
    },
    2: {
        "func": algo_random_choice_loop,
        "name": "Random Choice Loop",
        "time": "$O(k)$", "space": "$O(1)$", # k = attempts until match or max
        "explanation": "Randomly combine predefined greetings and names. Loops until 'Hello, World!' is formed by chance or a limit is reached.",
        "snippet": '''
greetings = ["Hello,", "Hi,"]
names = ["World!", "There!"]
while True:
  res = f"{random.choice(greetings)} {random.choice(names)}"
  print(res)
  if res == "Hello, World!": break
        '''
    },
     3: {
        "func": algo_string_concat,
        "name": "String Concatenation",
        "time": "$O(n)$", "space": "$O(n)$", # In Python, string concat can be O(n^2) if not optimized, but often closer to O(n) total amortized. Let's use O(n) for simplicity.
        "explanation": "Builds the target string by appending one character at a time in a loop.",
        "snippet": '''
s = ""
for char in "Hello, World!":
    s += char
return s
        '''
    },
    4: {
        "func": algo_list_join,
        "name": "List Join",
        "time": "$O(n)$", "space": "$O(n)$",
        "explanation": "Stores each character in a list and then uses the efficient `join` method to form the final string.",
        "snippet": '''
chars = ["H", "e", ..., "!"]
return "".join(chars)
        '''
    },
5: {
    "func": algo_quantum_sim,
    "name": "Quantum Simulation",
    # Use raw strings r"..." for time and space
    "time": r"$O(n \cdot m)$", "space": r"$O(1)$", # n chars, m measurement bits (m=8 here) -> O(n)
    "explanation": "...",
    "snippet": '...'
},
# Algo 6: FFT Noise
6: {
    "func": algo_fft_noise,
    "name": "FFT Noise",
    # Use raw strings r"..."
    "time": r"$O(n \log n)$", "space": r"$O(n)$", # Dominated by FFT
    "explanation": "...",
    "snippet": '...'
},
    7: {
        "func": algo_hash_nonce,
        "name": "Hash Nonce (Proof-of-Work)",
        "time": "$O(k)$", "space": "$O(1)$", # k = nonce value found (depends on difficulty)
        "explanation": "Mimics crypto mining (Proof-of-Work). Finds a number (nonce) such that the cryptographic hash (SHA-256) of the target string plus the nonce starts with a specific prefix.",
        "snippet": '''
target_hash_prefix = "abc" # Example
nonce = 0
while True:
  block = f"Hello, World!{nonce}".encode()
  if sha256(block).hexdigest().startswith(target_hash_prefix):
    return "Hello, World!"
  nonce += 1
        '''
    },
# Algo 8: Genetic Algorithm
8: {
    "func": algo_genetic,
    "name": "Genetic Algorithm",
    # Use raw strings r"..."
    "time": r"$O(g \cdot p \cdot n)$", "space": r"$O(p \cdot n)$", # g=gens, p=pop_size, n=str_len
    "explanation": "...",
    "snippet": '...'
},
# Algo 9: Random Guessing (Incremental)
9: {
    "func": algo_random_guess_incremental,
    "name": "Random Guessing (Incremental)",
    # Use raw strings r"..." for time (contains |\Sigma|)
    "time": r"$O(n \cdot |\Sigma|)$ avg", "space": r"$O(n)$", # n=len, |\Sigma|=charset size. Worst case unbounded.
    "explanation": "...",
    "snippet": '...'
},
# Algo 10: Bogo Sort (Permutation)
10: {
    "func": algo_bogo_sort,
    "name": "Bogo Sort (Permutation)",
    # Use raw strings r"..." for time
    "time": r"$O(n \cdot n!)$ avg", "space": r"$O(n)$", # Average case. Worst case unbounded.
    "explanation": "...",
    "snippet": '...'
}
}

# --- Streamlit App Layout ---

# Slider uses keys 1 to 10
complexity_level = st.slider("Select Algorithm (1=Fastest/Simplest, 10=Slowest/Most Complex)",    min_value=1,    max_value=len(ALGO_INFO),  value=1,  step=1)

# Get selected algorithm details
selected_algo = ALGO_INFO[complexity_level]
selected_algo_func = selected_algo["func"]
selected_algo_name = selected_algo["name"]

st.markdown(f"---")
st.subheader(f"Running Algorithm {complexity_level}: {selected_algo_name}")

# Button to run benchmark
if st.button(f"Run Benchmark for '{selected_algo_name}'"):

    # Prepare to display outputs
    st.markdown("#### Outputs:")
    output_placeholder = st.empty()
    intermediate_outputs = []
    final_result = "Execution did not yield a final result." # Default

    # --- Run the selected algorithm with benchmarking ---
    try:
        tracemalloc.start()
        start_time = time.perf_counter()

        # Execute the generator
        output_generator = selected_algo_func()
        for i, output in enumerate(output_generator):
             intermediate_outputs.append(f"Step {i+1}: {output}")
             # Update output display dynamically (show last few)
             output_placeholder.text_area("Intermediate Steps (Scrollable)",
                                         "\n".join(intermediate_outputs[-20:]), # Show last 20 steps
                                         height=200,
                                         key=f"output_{i}") # Change key to force update
             final_result = output # Store the last yielded value as final
             if i < 10: time.sleep(0.01)

        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Display Final Result clearly
        st.markdown("#### Final Result:")
        st.code(final_result, language="text")

        # Display Benchmarking Metrics
        st.markdown("#### Performance Metrics:")
        col1, col2 = st.columns(2)
        col1.metric("Time Taken (s)", f"{end_time - start_time:.6f}")
        col2.metric("Peak Memory Used (KiB)", f"{peak_memory / 1024:.2f}")

        # Display Big-O complexity table
        st.markdown("#### Theoretical Complexity:")
        st.markdown(
            f"""
            | Metric        | Big-O Notation         |
            |---------------|------------------------|
            | **Time** | {selected_algo["time"]}  |
            | **Space** | {selected_algo["space"]} |
            """
        )

        # Display Explanation and Code Snippet
        st.markdown("#### Algorithm Explanation:")
        with st.expander("Click to see explanation and simplified code snippet"):
            st.markdown(selected_algo["explanation"])
            st.code(selected_algo["snippet"], language="python")

    except Exception as e:
        st.error(f"An error occurred during execution: {e}")
        if tracemalloc.is_tracing():
            tracemalloc.stop()

else:
    st.info("Click the button above to run the benchmark for the selected algorithm.")

st.markdown("---")
st.caption("Note: Benchmark times can vary significantly based on system load and random outcomes.")