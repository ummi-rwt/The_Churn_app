# The_Churn_app
Predicting which telecom customers will leave, and whether a retention offer is worth it. Python, scikit-learn, SHAP, Streamlit.

# The_Churn_telecom
Predicting which telecom customers will leave, and whether a retention offer is worth it. Python, scikit-learn, SHAP, Streamlit.

# Telecom customer churn: who is about to leave, and is it worth calling them?

**Live demo:** https://thechurntelecom-refl7liuc73zdbwp95mgqb.streamlit.app/

<!-- Write 2-3 sentences here in your own words: why this project, what you wanted to learn. Delete this comment when done. -->

A telecom operator wants to keep customers who are about to leave. This project predicts who is likely to churn and, more importantly, decides **whom to contact** once you put a price on the retention offer.

## The data

The IBM Telco customer churn dataset: 7,043 customers and 50 columns (contract, services, monthly charge, tenure, reasons for leaving...). It is a **fictional** dataset made by IBM for teaching, so the results say nothing about a real operator. The file is not included in this repo.

I kept **5,992 customers** (21% of them churned):

- I removed the 454 customers marked "Joined" (they signed up in the quarter, so we cannot know yet whether they will stay).
- I also removed customers with 3 months of tenure or less (see "A mistake I fixed" below).
- I dropped the columns that describe the outcome or may be computed from it (`Customer Status`, `Churn Label`, `Churn Score`, `Churn Category`, `Churn Reason`, `Satisfaction Score`, `CLTV`), plus identifiers, location columns and cumulative totals that overlap with tenure and monthly charge.

## What I did

1. Explored the data and checked which columns could leak the answer.
2. Built 36 features in one function (yes/no columns, one-hot categories, numeric columns).
3. Set aside 20% of the customers as a final test set, opened only once at the end.
4. Compared five models with 5-fold cross-validation, against a "nobody leaves" baseline.
5. Chose the decision threshold with a simple cost model instead of the default 0.5.
6. Explained the model with SHAP.
7. Evaluated once on the test set, then built a small Streamlit app.

## What the data shows

| Contract | Customers | Share who left |
|---|---|---|
| Month-to-month | 2,607 | 41% |
| One year | 1,524 | 11% |
| Two years | 1,861 | 3% |

| Tenure | Customers | Share who left |
|---|---|---|
| 4-6 months | 419 | 45% |
| 7-12 months | 716 | 35% |
| 13-24 months | 1,024 | 29% |
| More than 24 months | 3,833 | 14% |

Fiber optic customers left more (35%) than DSL (13%) or cable (19%) customers.

For the 1,272 customers who left, the company recorded a reason. 45% went to a competitor (better devices, better offer, more data, faster speeds), 17% were dissatisfied, 16% blamed the attitude of the staff, and 11% cited price. These reasons come from customers who already left, so they are not necessarily the real causes.

## Results

Accuracy alone is misleading: predicting that nobody leaves is already right 79% of the time. I looked at **recall** (how many leavers I catch), **precision** (how many contacted customers really leave) and **AUC** (how well customers are ranked by risk).

Cross-validation on the training set:

| Model | Accuracy | Recall | Precision | AUC |
|---|---|---|---|---|
| Baseline (nobody leaves) | 0.788 | 0.000 | 0.000 | 0.500 |
| Gradient boosting | 0.868 | 0.585 | 0.738 | 0.910 |
| Logistic regression | 0.805 | 0.854 | 0.525 | 0.902 |
| Random forest | 0.852 | 0.700 | 0.640 | 0.898 |
| SVM (RBF) | 0.808 | 0.814 | 0.531 | 0.890 |
| KNN | 0.808 | 0.533 | 0.549 | 0.812 |

Gradient boosting has the best AUC, but the gap with logistic regression is small, within the noise of cross-validation. Logistic regression, random forest and SVM were trained with class weights and gradient boosting was not, so their recall is not directly comparable; AUC is the fairer comparison. Removing `Total Refunds` and `Total Extra Data Charges` left the AUC unchanged (0.910).

### Choosing who to contact

With **made-up** costs (offer 100 €, 30% of leavers accept to stay, a kept customer is worth about 800 €, i.e. 12 months of revenue), the expected profit on the 4,793 training customers (cross-validated predictions) is:

| Strategy | Profit |
|---|---|
| Contact nobody | 0 € |
| Contact everyone | -234,562 € |
| Threshold 0.50 (default) | 62,385 € |
| Threshold 0.45 (best) | 65,080 € |

The best threshold (0.45) is close to the simple rule *offer cost / (success rate x customer value)* = 0.42, and the curve is almost flat between 0.35 and 0.50. These amounts are assumptions, not real data: change them in the app and watch the decision move.

![Expected profit by threshold](images/profit_by_threshold.png)

### Final test

I opened the test set once, with the model and the threshold (0.45) already fixed: 1,199 customers, about 255 of whom left.

| Metric | Result |
|---|---|
| AUC | 0.890 |
| Recall | 0.573 |
| Precision | 0.679 |
| Customers contacted | 215 (146 of them really left) |
| Expected profit | 11.4 € per customer |

This is a little lower than in cross-validation (AUC 0.910, about 13.6 € per customer), which is expected because the model and threshold were chosen on that training data.

### What drives churn (SHAP)

![SHAP summary](images/shap_beeswarm.png)

The contract type (two years, one year), the number of referrals, the number of dependents and the tenure push the risk down. A high monthly charge, a higher age and fiber optic push it up.

<!-- Write 2-3 sentences here in your own words: which variables dominate, what surprised you, what you would do first. Delete this comment when done. -->

## A mistake I fixed

My first version removed the "Joined" customers but kept customers with 3 months of tenure or less. Among those, only the ones who had already left remained, so the data said that **100% of customers with up to 3 months of tenure leave**, which is an artifact of my filter and not real behavior. The model learned it: AUC 0.935 and a profit of about 128,000 €, both inflated. I noticed it on the SHAP plot (a cluster of points far from all the others) and checked it with a tenure table. After removing these customers, the numbers dropped to the ones above.

## Limits

- The dataset is fictional and the costs are invented. In real life, the offer's success rate would be measured with an A/B test.
- The model only applies to customers with more than 3 months of tenure.
- The test set has about 255 leavers, so its results can move by a few points with another split.
- The threshold was chosen on the training data (cross-validation), so the profit there is a little optimistic.
- SHAP explains the model, not the real world: a long contract keeps customers partly because it is hard to leave, not necessarily because they are happy.
- `Satisfaction Score` and `CLTV` were excluded as a precaution because they may depend on the outcome; I did not test this.
- The app only asks 10 questions. The other variables are set to typical values of the training data.

## Run it yourself

```
pip install -r requirements.txt
streamlit run app.py
```

Files: `app.py` (Streamlit app), `telco_model.joblib` (trained model), `Telco_Churn.ipynb` (analysis), `images/` (charts).
