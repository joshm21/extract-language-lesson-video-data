from core import detect
from core import prepare
import config
from functools import partial
import random
import copy
import csv
import cv2
import os
print("starting...")

# Import workshop modules

# --- CONFIGURATION VARIABLES ---
FRAMES_DIR = "training_frames"
DATA_CSV = "data.csv"

VIDEO_ID = None  # which video_id to train against, or all if = None
# VIDEO_ID = "14QbqkeiSDtU62syzgaOVXhXRzBJWhaNN"

# how many config "genomes" / configs to try?
POPULATION_SIZE = 20
# percent of best scores that always make it to the next generation with no mutations
ELITE_SURVIVORS = 0.1
# ensure we have at least one survivor
ELITISM_COUNT = max(1, round(POPULATION_SIZE * ELITE_SURVIVORS))
# how many times to evaluate and evolve?
GENERATIONS = 15

# how likely functions are added, removed, or swapped?
MUTATION_RATE_STRUCTURAL = 0.30
# how likely function arg values are mutated?
MUTATION_RATE_ARG = 0.30

# --- GENE POOLS & CONSTRAINTS ---
POOLS = {
    "pre": [prepare.to_grayscale, prepare.to_blurred],
    "binary": [prepare.at_global_threshold, prepare.at_adaptive_threshold, prepare.at_canny_edges],
    "morph": [prepare.do_dilation, prepare.do_erosion, prepare.do_opening, prepare.do_closing],
    "detect": [detect.find_quads]
}

DEFAULT_PARAMS = {
    prepare.to_grayscale: {},
    prepare.to_blurred: {"ksize": 5},
    prepare.at_global_threshold: {"threshold": 127},
    prepare.at_adaptive_threshold: {"block_size": 11, "c_val": 2},
    prepare.at_canny_edges: {"low": 50, "high": 150},
    prepare.do_dilation: {"kernel_size": 3, "iterations": 1},
    prepare.do_erosion: {"kernel_size": 3, "iterations": 1},
    prepare.do_opening: {"kernel_size": 3},
    prepare.do_closing: {"kernel_size": 3},
    detect.find_quads: {"min_area": 300, "epsilon": 0.02}
}


# --- 1. DATA LOADING ---
def load_cached_frames(directory, csv_filename):
    """Loads images into RAM and parses expected quad counts from data.csv."""
    frames = []

    csv_path = os.path.join(directory, csv_filename)

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV {csv_path} not found.")
        return frames

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'filename' not in row:
                print("ERROR: CSV does not contain 'filename' column")
                return frames
            filepath = row['filename']

            # filter out if not our focus video
            if VIDEO_ID:
                if VIDEO_ID not in filepath:
                    continue

            # Parse full + partial columns
            # could change this to include any combination of 'full', 'covered', and 'partial' columns
            # expected_count = int(row.get('full', 0)) + int(row.get('partial', 0))
            expected_count = int(row.get('full', 0))

            # resizing images to improve speed without losing too much accuracy
            img = cv2.imread(filepath)
            if img is not None:
                # Resize to a max width of 640 while maintaining aspect ratio
                scale = 640 / img.shape[1]
                if scale < 1.0:
                    new_size = (640, int(img.shape[0] * scale))
                    img = cv2.resize(
                        img, new_size, interpolation=cv2.INTER_AREA)

                frames.append(
                    {"image": img, "expected": expected_count, "name": filepath})
    return frames


# --- 2. SEEDING FROM CONFIG ---
def get_seed_genome():
    """Extracts the Phase 1 manual pipeline from config.py into structured segments."""
    seed = {
        "pre": [],
        "binary": [],
        "morph": [],
        "detect": [{"func": detect.find_quads, "params": copy.deepcopy(DEFAULT_PARAMS[detect.find_quads])}]
    }

    if not hasattr(config, 'FRAME_PIPELINE'):
        return seed

    for step in config.FRAME_PIPELINE:
        func = step.func if isinstance(step, partial) else step
        params = dict(step.keywords) if isinstance(step, partial) else {}

        if func in POOLS["pre"]:
            seed["pre"].append({"func": func, "params": params})
        elif func in POOLS["binary"]:
            seed["binary"].append({"func": func, "params": params})
        elif func in POOLS["morph"]:
            seed["morph"].append({"func": func, "params": params})
        elif func == detect.find_quads:
            seed["detect"][0]["params"]["min_area"] = params.get(
                "min_area", 300)
            seed["detect"][0]["params"]["epsilon"] = params.get(
                "epsilon", 0.02)
            break

    return seed


# --- 3. GA MECHANICS ---
def get_random_gene(pool_name):
    """Creates a gene with a random function and randomized initial parameters."""
    func = random.choice(POOLS[pool_name])
    params = copy.deepcopy(DEFAULT_PARAMS[func])

    # Randomize each parameter immediately so the gene is unique at birth
    for param in params:
        # We apply multiple nudges or a larger range to ensure diversity
        for _ in range(random.randint(1, 5)):
            params[param] = nudge_value(param, params[param])

    return {"func": func, "params": params}


def evaluate(genome, frames):
    """Calculates fitness: 100 points per frame, minus error margin."""
    total_score = 0

    for frame_data in frames:
        state = {"current_image": frame_data["image"].copy()}

        try:
            # Run the structured pipeline sequentially
            for section in ["pre", "binary", "morph", "detect"]:
                for gene in genome[section]:
                    updates = gene["func"](state, **gene["params"])
                    if updates:
                        state.update(updates)

            detected_count = len(state.get("quads", []))
            expected_count = frame_data["expected"]

            # Max 100 per frame. Penalty for missing or hallucinating quads.
            frame_score = max(
                0, 100 - (abs(expected_count - detected_count) * 10))
            total_score += frame_score

        except Exception:
            return 0  # Crash = 0 fitness

    return total_score


def nudge_value(param_name, current_val):
    """Mutates numeric arguments safely."""
    if param_name in ["ksize", "block_size"]:
        new_val = current_val + random.choice([-2, 2])
        return max(3, new_val)
    elif param_name in ["threshold", "low", "high"]:
        new_val = current_val + random.randint(-15, 15)
        return max(0, min(500, new_val))
    elif param_name == "iterations":
        new_val = current_val + random.choice([-1, 1])
        return max(1, min(5, new_val))
    elif param_name == "c_val":
        return current_val + random.choice([-1, 1])
    elif param_name == "kernel_size":
        new_val = current_val + random.choice([-2, 2])
        new_val = new_val if new_val % 2 != 0 else new_val + 1
        return max(1, new_val)
    elif param_name == "min_area":
        new_val = current_val + random.randint(-50, 50)
        return max(300, new_val)
    elif param_name == "epsilon":
        new_val = current_val + random.uniform(-0.005, 0.005)
        return max(0.001, min(0.1, new_val))

    return current_val


def mutate(genome):
    """Applies structural and argument mutations specifically tailored per pool."""
    mutant = copy.deepcopy(genome)

    # 1. Structural Mutations
    if random.random() < MUTATION_RATE_STRUCTURAL:

        # PRE POOL: 0, 1, or 2 functions
        if random.random() < 0.5 and len(mutant["pre"]) > 0:
            mutant["pre"].pop(random.randrange(len(mutant["pre"])))
        elif len(mutant["pre"]) < len(POOLS["pre"]):
            existing_funcs = [g["func"] for g in mutant["pre"]]
            available = [f for f in POOLS["pre"] if f not in existing_funcs]
            if available:
                func = random.choice(available)
                mutant["pre"].append(
                    {"func": func, "params": copy.deepcopy(DEFAULT_PARAMS[func])})

        # BINARY POOL: Exactly 1 or 0 functions
        if mutant["binary"] and random.random() < 0.3:
            mutant["binary"] = []  # make it None
        else:
            mutant["binary"] = [get_random_gene("binary")]

        # MORPH POOL: Any number, any order
        action = random.choice(["add", "remove", "swap"])
        if action == "add":
            mutant["morph"].insert(random.randint(
                0, len(mutant["morph"])), get_random_gene("morph"))
        elif action == "remove" and mutant["morph"]:
            mutant["morph"].pop(random.randrange(len(mutant["morph"])))
        elif action == "swap" and len(mutant["morph"]) >= 2:
            idx1, idx2 = random.sample(range(len(mutant["morph"])), 2)
            mutant["morph"][idx1], mutant["morph"][idx2] = mutant["morph"][idx2], mutant["morph"][idx1]

    # 2. Argument Mutations
    for section in ["pre", "binary", "morph", "detect"]:
        for gene in mutant[section]:
            if random.random() < MUTATION_RATE_ARG:
                for param in gene["params"]:
                    gene["params"][param] = nudge_value(
                        param, gene["params"][param])

    return mutant


def crossover(parent1, parent2):
    """Uniform crossover applied section by section."""
    child = {
        "pre": copy.deepcopy(parent1["pre"] if random.random() < 0.5 else parent2["pre"]),
        "binary": copy.deepcopy(parent1["binary"] if random.random() < 0.5 else parent2["binary"]),
        "morph": copy.deepcopy(parent1["morph"] if random.random() < 0.5 else parent2["morph"]),
        "detect": copy.deepcopy(parent1["detect"])
    }

    # Crossover parameters for the detection function
    for param in child["detect"][0]["params"]:
        if random.random() < 0.5:
            child["detect"][0]["params"][param] = parent2["detect"][0]["params"][param]

    return child


# --- 4. THE EVOLUTIONARY LOOP ---
def run_evolution():
    frames = load_cached_frames(FRAMES_DIR, DATA_CSV)
    if not frames:
        return

    max_possible_score = len(frames) * 100
    print(f"Max Possible Fitness Score: {max_possible_score}")

    # Store history for final reporting
    generation_history = []

    # 1. Initialize Population from Seed
    seed_genome = get_seed_genome()
    population = [seed_genome]
    for _ in range(POPULATION_SIZE - 1):
        population.append(mutate(mutate(copy.deepcopy(seed_genome))))

    print(f"=== GENERATION STATS ===")

    # 2. Evolution
    for gen in range(GENERATIONS):
        scored_pop = []

        # Persistent progress lines
        for i, genome in enumerate(population):
            # The \r keeps these lines updating in place
            print(
                f"Evaluating Generation: {gen+1} / {GENERATIONS}, Genome: {i + 1} / {POPULATION_SIZE}    ", end="\r")

            score = evaluate(genome, frames)
            scored_pop.append((score, genome))

        # Sort and calculate stats
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        best_score, best_genome = scored_pop[0]
        avg_fitness = sum(x[0] for x in scored_pop) / POPULATION_SIZE

        generation_history.append((gen + 1, avg_fitness, best_score))

        # Log Generation Stats (this stays in terminal history)
        # Clear the 'Evaluating' line before printing the summary
        print(" " * 50, end="\r")
        print(
            f"Gen {gen + 1:02d} | Avg Fitness: {int(avg_fitness):5d} | Best Fitness: {int(best_score):5d}")

        # Selection & Next Generation
        next_gen = []
        for i in range(ELITISM_COUNT):
            next_gen.append(copy.deepcopy(scored_pop[i][1]))

        while len(next_gen) < POPULATION_SIZE:
            tournament = random.sample(scored_pop, 3)
            tournament.sort(key=lambda x: x[0], reverse=True)
            child = crossover(tournament[0][1], tournament[1][1])
            child = mutate(child)
            next_gen.append(child)

        population = next_gen

    # --- END OF EVOLUTION LOGGING ---
    print("\n" + "="*40)
    print("EVOLUTION COMPLETE - GENOMES BY RANK")
    print("="*40)

    # Re-evaluate or just take from the final sorted population
    scored_pop.sort(key=lambda x: x[0], reverse=True)

    for rank in range(len(scored_pop)):
        score, genome = scored_pop[rank]
        print(f"\n[RANK {rank + 1}] Fitness: {score}/{max_possible_score}")

        step_idx = 1
        for section in ["pre", "binary", "morph", "detect"]:
            for gene in genome[section]:
                func_name = gene['func'].__name__
                params = ", ".join([f"{k}={v}" if not isinstance(v, float) else f"{k}={v:.3f}"
                                   for k, v in gene['params'].items()])
                print(f"  {step_idx}. {func_name}({params})")
                step_idx += 1


if __name__ == "__main__":
    run_evolution()
