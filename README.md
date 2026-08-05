# Predicting the Unpredictable: Testing the Null Hypothesis with XGBoost on Powerball Data

An end-to-end Machine Learning pipeline built to test a foundational statistical concept: **Can historical "draw shapes" predict a completely random physical event?** 

This project web-scrapes historical Powerball data, engineers lagged temporal features, and utilizes a GPU-accelerated dual-model XGBoost architecture to evaluate the Null Hypothesis.

## Read the Full Project Breakdown on Medium
* **Part 1:** [Scraping and Feature Engineering the Powerball] *([Part1](https://medium.com/@heathdingusa/can-machine-learning-hack-the-lottery-part-1-scraping-and-augmenting-20-years-of-data-8d410fb6df61))*
* **Part 2:** [Building a Dual-Model XGBoost Architecture] *([Part 2](https://medium.com/@heathdingusa/can-machine-learning-hack-the-lottery-part-2-building-a-dual-model-xgboost-architecture-23954e71d1c5)*

## Technical Architecture
Because the Powerball draws the five white balls and the single red ball from **two separate drums** forcing a single model to predict all six numbers violates the mechanics of the game. This repository implements a split-pipeline architecture:
1. **Model A (White Balls):** Formulated as a **Multi-Label Classification** problem. It leverages `MultiLabelBinarizer` to map a 69-dimensional binary target matrix and uses `predict_proba` with probability sorting to output the top 5 predicted numbers.
2. **Model B (Red Ball):** Formulated as a **Multi-Class Classification** problem predicting 1 target out of 26 possible classes.

### Key Data Science Principles Implemented:
* **Data Leakage Prevention:** All engineered features (sums, high/low limits, odd/even ratios) are strictly lagged by one period (`shift(1)`) to ensure the model only trains on information available *prior* to the draw.
* **Sequential Time-Series Splitting:** Train/test sets are split chronologically (`shuffle=False`) to prevent future data from contaminating historical training batches.
* **GPU Acceleration:** Built to leverage CUDA-enabled hardware (`device='cuda'`) for rapid gradient boosting iterations.

---

## Repository Structure
```text
Powerball-Project/
|
|-- Data/
|     |-- powerball26-05.html   # Raw scraped HTML draw history
|-- Dataset/
|     |-- Powerball_Dataset.csv   # Processed tabular export
|-- src/
|     |-- powerball_numbers.py   # Main end to end Python script
|-- requirements.txt   # Project dependencies
|-- README.md   # Project Documentation
