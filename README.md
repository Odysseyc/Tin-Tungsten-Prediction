# Predicting Tin-Tungsten Deposits with Machine Learning

Final project for Physics 5680, Big Data Analytics in Physics (The Ohio State University, Autumn 2025). Author: Adam Sabet.

Finding new mineral deposits usually takes expensive surveys and drilling. This project trains classifiers on geophysical
survey data to map where tin-tungsten (Sn-W) deposits are most likely in northeast Tasmania, Australia, so that exploration
can focus on the most promising ground first.

**Result:** a Random Forest reached **0.93 ROC-AUC** on held-out test data. At its decision threshold, **43% of the cells it flagged
were real deposits**, when only about 4% of the test cells contain one.

The full analysis, with figures and discussion, is in [`TinTungstenProject(5)-1`](TinTungstenProject(5)-1).

## Data

The study area is divided into 251,138 grid cells. Each cell holds the average of several geophysical layers:

| Feature | What it measures |
|---|---|
| `TMI_mean`, `TMI_1VD_mean` | Total magnetic intensity and its first vertical derivative |
| `IR_mean`, `IR_1VD_mean` | Isostatic residual gravity and its first vertical derivative |
| `K_pct_mean`, `Th_ppm_mean`, `U_ppm_mean` | Radiometric potassium, thorium and uranium |
| `elevation` | Height from a digital elevation model, which I added to the grid |

I engineered three more features: the K/Th ratio, IR × U, and TMI × elevation. The IR × U term follows a geological
hypothesis: Sn-W deposits form near granites, which show up as gravity lows, where hot fluids have enriched the rock in uranium.

Labels come from 222 known Sn-W occurrences. I sampled 5,000 cells without a deposit as background, so the dataset has about
20 empty cells for every deposit cell.

The data are public geoscience layers from Mineral Resources Tasmania and Geoscience Australia, provided through the course.
They are not included in this repo, so the notebook is meant to be read rather than run.

## Method

1. **Split and scale:** 70% training, 30% test, stratified by class. Features are standardized using statistics from the
   training set only.
2. **Balance:** SMOTE-ENN on the training set only. It creates synthetic deposit examples and removes noisy boundary points,
   giving close to a 1:1 class ratio. The test set keeps the real imbalance.
3. **Tune:** 10-fold cross-validation. For the Random Forest I tuned tree depth (best: 22). For the RBF-kernel SVM I tuned
   the regularization parameter C (best: 100).
4. **Threshold:** I raised the decision threshold from 0.5 to 0.9. At 0.5 the models found most deposits but flagged far too
   many empty cells, which would waste exploration money.
5. **Map:** the Random Forest's probabilities across all grid cells form a continuous prospectivity map.

## Results (held-out test set: 1,560 cells, 60 with deposits)

| Model | ROC-AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| **Random Forest** | **0.931** | **0.434** | 0.600 | **0.504** | 0.955 |
| SVM (RBF) | 0.916 | 0.339 | 0.650 | 0.446 | 0.938 |

Precision, recall and F1 are for the deposit class. Accuracy alone is misleading here: a model that always predicts "no
deposit" would score about 96%.

**Most important features (Random Forest, Gini importance):**

1. Isostatic residual gravity (0.36)
2. IR × U, the engineered feature (0.22)
3. Gravity first vertical derivative (0.12)
4. Uranium (0.11)

The engineered IR × U feature ranked second, which supports the granite and uranium-enrichment hypothesis.

## Limitations and next steps

- Most false positives look like barren granites. Adding structural features, such as distance to mapped faults and
  geological contacts, is the most promising way to separate them from mineralized ones.
- Gradient boosting (XGBoost or LightGBM) may improve precision and recall at the same time.
- The labels only include known occurrences. Unexplored areas with deposits count as "no deposit", which biases the model
  toward places that have already been explored.

## Tools

Python 3.9, pandas, NumPy, GeoPandas, rasterio, scikit-learn, imbalanced-learn, Matplotlib, seaborn, Plotly and Folium.
I ran the notebook on the Ohio Supercomputer Center cluster.
