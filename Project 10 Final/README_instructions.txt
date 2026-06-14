========================================================================
EARIN PROJECT 10: REVIEW SCORE PREDICTION - CODE EXECUTION MANUAL
========================================================================
This manual outlines the environmental configurations, workspace design,
and execution workflows required to replicate the experimental milestones 
and data analysis detailed in the final project report (Project 10.pdf).

------------------------------------------------------------------------
1. PROJECT STORAGE ARCHITECTURE
------------------------------------------------------------------------
Before initiating execution sequences, ensure your root project directory
adheres strictly to the following layout:

Project-Root/
│
├── dataset/
│   └── README.txt              <-- Open this first to extract raw data!
│   └── kaggle_script           <-- Script required to extract the data from kaggle
│   └── my_50k_reviews.csv      <-- Will be created after running data_loader.py and data extractions script from kaggle
│
├── code/
│   ├── data_loader.py          <-- Balanced 5-class splitting engine
│   ├── preprocessing.py        <-- Text processing & BERT extraction
│   ├── preprocessing_final.py  <-- Post-GridSearch optimized vectorizer
│   ├── models.py               <-- Baseline architectures (with BERT tweaks)
│   ├── model_final.py          <-- Post-GridSearch optimized neural network
│   ├── main.py                 <-- Historical Baseline & Ablation script
│   ├── optimize.py             <-- Phase 6: TF-IDF Grid Search Optimizer
│   ├── optimize_bert.py        <-- Phase 6: Dense BERT Grid Search Optimizer
│   ├── main_(5-40k_samples).py <-- Phase 7: Macro Learning Curve Analysis
│   └── main_(500-5k_samples).py<-- Phase 7: Micro Learning Curve (Low-Data Cliff)
│
├── output/                     <-- [DYNAMICAL] Automatically generated on first run
│   ├── output_x.txt            <-- Will be created after running the code
│
├── Project 10.pdf              <-- Main Final Project Report
└── README_instructions.txt     <-- This manual file

------------------------------------------------------------------------
2. SYSTEM REQUIREMENTS & RUNTIME ENVIRONMENT
------------------------------------------------------------------------
This workspace is built using Python 3.10+. 

A. Initialize Virtual Environment:
   Open a terminal shell inside the "Project-Root/" folder and run:
       python -m venv venv
    
B. Activate Environment:
   - Windows:          .\venv\Scripts\activate
   - Linux / macOS:    source venv/bin/activate

C. Package Installations:
   Execute the following command to download core matrix processing, 
   machine learning pipelines, and contextual deep learning embeddings:
       pip install numpy pandas scikit-learn sentence-transformers

------------------------------------------------------------------------
3. PIPELINE STEP-BY-STEP REPRODUCTION GUIDE
------------------------------------------------------------------------
Crucial Step: Navigate directly into the code subdirectory before execution:
    cd code

All runnable python targets are equipped with a `DualLogger` module. When
invoked, they will dynamically create the "/output" folder one level up if
it does not exist, and generate auto-incrementing text logs (e.g., output_1.txt,
output_2.txt) so previous presentation run records are never overwritten.

--- RUN 1: Historical Baseline & Ablation Studies (main.py) ---
Description: Fits default configurations for Logistic Regression, Random Forests,
and MLPs on the corpus, isolates the top model, and handles the stop-word ablation.
    Command:  python main.py
    Output:   Generates "output_x.txt" in the /output folder.

--- RUN 2: TF-IDF Hyperparameter Grid Search (optimize.py) ---
Description: Launches a 3-fold cross-validation grid search testing 144 matrix 
and neural configurations (432 total fits) to isolate optimal TF-IDF parameters.
    Command:  python optimize.py
    Output:   Saves metrics and optimal parameters to "output_x.txt" in /output.

--- RUN 3: BERT Embedding Grid Search (optimize_bert.py) ---
Description: Evaluates dense continuous vector spaces using pre-computed BERT 
sentence embeddings across various hidden layer depths and alpha regularization parameters.
    Command:  python optimize_bert.py
    Output:   Saves metrics and parameter paths to "output_bert_x.txt" in /output.

--- RUN 4: Macro Learning Curve Data Saturation (main_(5-40k_samples).py) ---
Description: Implements the optimized grid parameters and steps down the data pool
from 40,000 down to 5,000 samples to map standard data saturation boundaries.
    Command:  python main_(5-40k_samples).py
    Output:   Generates "output_final_x.txt" with summary tables in /output.

--- RUN 5: Micro Learning Curve Breakdown Cliff (main_(500-5k_samples).py) ---
Description: Zooms directly into high-resolution low-data allocations (5k down to 500)
to mathematically demonstrate the exact sample size cliff where system viability collapses.
    Command:  python main_(500-5k_samples).py
    Output:   Generates "output_final_x.txt" with micro-zoom tables in /output.

------------------------------------------------------------------------
4. DETERMINISTIC REPRODUCIBILITY GUARANTEES
------------------------------------------------------------------------
To comply with scientific validation policies, pseudorandom state parameters 
across all classification algorithms and text splitters are hard-locked 
utilizing an immutable seed configuration (random_state=42) within the files:
data_loader.py, models.py, model_final.py, and optimize.py. 

Re-running these files will consistently produce identical evaluation marks 
on matching data slices, preserving the peak optimized TF-IDF accuracy.
========================================================================