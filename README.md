# Applying Differential Privacy to Training Data
## Project Overview
This project investigates the integration of Differential Privacy (DP) into machine learning workflows, focusing on how privacy-preserving randomization affects model performance when training on sensitive healthcare data. By applying both the Laplace and Exponential mechanisms to a synthetic COVID‑19 patient dataset, the study quantifies the trade‑offs between privacy guarantees and predictive accuracy, and evaluates post‑processing corrections (e.g., one‑hot encoding) and data balancing techniques (SMOTE).

## Objectives
- Privacy Protection: Implement DP mechanisms to sanitize numeric and categorical clinical features, ensuring that individual patient information cannot be reverse‑engineered from released datasets.

- Performance Evaluation: Measure the impact of varying privacy budgets (ϵ values) on model convergence, test accuracy, and feature importance using a convolutional neural network (CNN).

- Data Integrity Analysis: Compare anonymized outputs against original data to quantify distortion and validate DP implementations.

- Class Imbalance Mitigation: Apply SMOTE to investigate how oversampling interacts with DP noise and its effect on model fairness.

## Dataset
- Source: Synthea™ COVID‑19 synthetic patient population simulator, containing demographics, medical conditions, care plans, observations, allergies, immunizations, and more .

- Scope: Subsets selected for their clinical relevance to COVID‑19 outcomes, including lab measurements, binary indicators (e.g., virus presence), and patient metadata.

- Preprocessing: Initial cleaning removed zeros as missing values, and split into numeric vs. categorical columns for mechanism assignment .

## Differential Privacy Mechanisms
- Laplace Mechanism (Numeric Data)
- Implementation (Normal): LaplaceMech-Normal.py loads a CSV, excludes the patient ID, and adds Laplace noise scaled by sensitivity/ϵ, rounding results to non‑negative integers to preserve domain semantics .

- Implementation (One‑Hot): LaplaceMech-OneHot.py uses Dask for scalable noise addition across large tables, followed by a post‑processing step that enforces valid one‑hot encodings on categorical indicator columns .

## Exponential Mechanism (Categorical Data)
- Embedded in ProcessObservations.py, categories are resampled based on utility scores proportional to their original frequencies. This ensures ε‑differential privacy while preserving plausible distributional properties of binary and multi‑category fields 


## Data Integrity Assessment
- Table Comparison: compareTables.py reads the original and noisy CSVs, verifies column alignment, and computes the overall percentage of unchanged cell values, providing a quantitative measure of distortion introduced by DP 

## Model Training & Evaluation
- Architecture: A CNN classifier trained on both original and DP‑noised datasets, with and without SMOTE balancing.

- Privacy Budgets Tested: ϵ = 0.5, 1.0, and 3.0, reflecting stronger to weaker privacy guarantees.

- Metrics: Training/validation accuracy and loss curves; test accuracy/loss; precision, recall, and F₁‑scores per class; and confusion matrices. Higher ϵ values reduced noise, requiring fewer epochs for convergence and yielding performance closer to the non‑private baseline, but at the cost of weaker privacy guarantees 

## Key Findings
- Performance Degradation: Lower ϵ (more noise) led to slower convergence and higher loss, particularly affecting underrepresented classes (e.g., COVID + Died).

- SMOTE Interaction: Balancing improved overall accuracy but risked inflating metrics via synthetic patterns, especially when combined with DP noise.

- Feature Importance Shift: In private models, reliance shifted from granular biomarkers (e.g., specific viral RNA measurements) to broader clinical features (e.g., age, ejection fraction), indicating noise‑induced dilution of high‑signal attributes 


- Integrity Trade‑off: Table comparison revealed a non‑trivial fraction of altered values, underscoring the balance between privacy and data utility.

## Tools & Environment
- Languages/Libraries: Python, Pandas, NumPy, Dask (for large‑scale processing), scikit‑learn (SMOTE), and TensorFlow/Keras (CNN).

- Workflow: Jupyter notebooks (main.ipynb, dataprocess.ipynb) orchestrate end‑to‑end data processing, DP application, and model training.

## Conclusion & Future Directions
This work demonstrates a reproducible pipeline for integrating DP into healthcare ML, highlighting the practical considerations of mechanism selection, budget allocation, and downstream impacts on model interpretability. Future enhancements include adaptive privacy budgets based on feature sensitivity, integration with federated learning, and exploration of alternative DP frameworks (e.g., Gaussian DP) to further optimize the privacy–utility trade‑off.
