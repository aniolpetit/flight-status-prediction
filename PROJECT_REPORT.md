# Flight Delay Prediction & Visual Analytics Report

## 1. Problem Statement
Flight delays are a persistent challenge in the aviation industry, costing billions of dollars annually and causing significant frustration for passengers. While some delays are caused by unpredictable events like severe weather, many are driven by systemic factors such as airport congestion, scheduling inefficiencies, and airline operational practices.

The core problem this project addresses is the **uncertainty surrounding flight reliability**. Passengers and stakeholders often lack accessible tools to:
1.  Assess the risk of a specific flight being delayed based on historical patterns.
2.  Understand the *drivers* behind these delays (e.g., is it the airline, the time of day, or the season?).

This project aims to bridge this gap by developing a **visual analytics platform** that combines interactive storytelling with machine learning to predict flight delay risks and explain the underlying factors.

---

## 2. Dataset Overview
**Source & Scope:**
The analysis uses a dataset of **US domestic flights from January 2018 to July 2022**. To ensure computational efficiency while maintaining statistical validity, we utilized **reservoir sampling** to curate a representative dataset of approximately **500,000 flights** from the raw multi-year data.

**Key Features:**
-   **Temporal:** Flight Date, Scheduled Departure Time (`CRSDepTime`), Month, Day of Week.
-   **Categorical:** Airline (`Reporting_Airline`), Origin Airport, Destination Airport.
-   **Operational:** Distance, Air Time, Taxi Times (used for analysis, excluded from prediction to prevent leakage).
-   **Target Variable:** `IsArrDelayed` (Binary: 1 if Arrival Delay > 15 minutes, 0 otherwise).

**Basic Exploratory Analysis (EDA):**
-   **Class Imbalance:** The dataset is highly imbalanced, with **~83% of flights arriving on time** and **~17% delayed**.
-   **The "Long Tail":** While most delays are short, a small percentage of flights experience extreme delays (3+ hours), creating a disproportionate impact on passenger perception.
-   **Temporal Trends:** Delay rates increase steadily throughout the day, peaking in the evening due to cascading effects.
-   **Seasonality:** Distinct peaks in delay/cancellation rates occur in **summer** (convective weather/volume) and **winter** (snowstorms), with **September-November** being the most reliable period.

---

## 3. Business Questions and Objectives
The project was designed to answer the following key questions:

**1. What are the primary drivers of flight delays?**
   - Does the airline matter more than the route?
   - How does the time of day influence reliability?

**2. Can we predict delays using *only* information available before departure?**
   - We aim to build a model that does not rely on "cheating" features (like Departure Delay or Taxi Out time), making it useful for future scheduling.

**3. How can we build trust in black-box predictive models?**
   - Providing a raw probability is insufficient; users need to know *why* a flight is flagged as high-risk.

**Objectives:**
-   **Interactive Exploration:** Build a Streamlit dashboard allowing users to filter and visualize data dynamically.
-   **Predictive Modeling:** Develop a robust binary classification model to predict arrival delays (>15 min).
-   **Explainable AI (XAI):** Implement SHAP values and surrogate models to make predictions transparent.

---

## 4. Methodology

### Data Preprocessing
-   **Feature Engineering:** Derived features like `DepTimeOfDay` (Morning, Afternoon, etc.), `DayOfWeekName`, and `seasonality` indicators.
-   **Leakage Prevention:** Strictly removed post-departure features (e.g., `DepDelay`, `TaxiOut`, `ActualElapsedTime`) from the training set.
-   **Handling Imbalance:** Used **oversampling** of the minority class (delayed flights) in the training set to achieve a 50/50 balance, preventing the model from biasing towards the "On-Time" majority. The test set remained representative of the real-world distribution.

### Machine Learning Strategy
-   **Model Selection:** Evaluated **Random Forest** and **XGBoost**. XGBoost was selected for its superior performance (ROC-AUC ~0.72) and efficiency.
-   **Evaluation Metrics:** Focused on **ROC-AUC** and **Precision-Recall** trade-offs rather than raw accuracy.
-   **Threshold Tuning:** Implemented a dual-threshold system:
    -   **Default (0.5):** Balanced approach.
    -   **High-Recall (0.4):** Prioritizes catching delays (reducing false negatives) at the cost of more false alarms.

### Visualization & Application (Streamlit)
The solution is deployed as a multi-page Streamlit app:
1.  **Explore Data:** Interactive histograms, heatmaps (Hour × Weekday), and correlation matrices.
2.  **Key Insights:** A curated "storytelling" page highlighting major findings (e.g., "The Delay Paradox").
3.  **Predict Delays:** An interface for users to input flight details and receive a risk assessment with a gauge chart.
4.  **Explainability:** Integration of **SHAP (SHapley Additive exPlanations)** to show feature contributions (Waterfall plots) and a **Surrogate Decision Tree** to visualize decision logic.

---

## 5. Conclusions
The analysis and modeling yield several critical insights:

1.  **Time is the Critical Factor:** Departure time is the strongest predictor of delays. "Cascading effects" mean that early morning flights are significantly safer than evening flights, regardless of the airline or route.
2.  **Airline Disparity:** There is a consistent performance gap (~15-20%) between top-performing airlines and budget/regional carriers.
3.  **Predictability Limits:** While the model effectively identifies high-risk scenarios (AUC > 0.7), the inherent randomness of aviation (weather, air traffic control) puts a ceiling on precision. The model is best used as a **risk assessment tool** rather than a crystal ball.
4.  **Value of Explainability:** By exposing *why* a prediction was made (e.g., "High risk because it's a Friday evening in July"), the tool transforms from a black box into an actionable decision support system for travelers.

**Final Recommendation:**
For the most reliable travel, passengers should prioritize **early morning flights** and **mid-week travel**, while avoiding the peak summer and winter holiday operational windows when possible.

