# UG Machine Learning Experiments
This repo contains code for experimenting with machine learning:
1. reinforcement learning with a genetic algorithm
2. unsupervised learning with k-means clusters
3. supervised learning with k nearest neighbors

## Quickstart
### Google Collab
1. [Open in collab](https://colab.research.google.com/drive/1VeZ7CzaUeZCQ1YL0RBKHnW3CBnSOoDCr?usp=sharing)
2. Run the cells in order
3. To alter pipeline or evolve starting point, edit config.py cell, and run it to save changes
4. If at any point your runtime disconnects, you need to re-run the install and warm up cells

### Local Setup
```bash
# 1. Clone the repository and enter the directory
git clone git@github.com:joshm21/ug-workshop1.git
cd ug-workshop1

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configuration
# Open config.py in your editor to make any necessary changes
# nano config.py  # (Example for terminal users)

# 6. Run the project
# To run the main application:
python run.py
# Or to run the evolution script:
python evolve.py
