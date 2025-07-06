import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, norm, beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# -----------------------------------
# 1. Load Dataset
# -----------------------------------
df = pd.read_csv("ab_test_synthetic_data.csv")

# -----------------------------------
# 2. Overview & Conversion Rates
# -----------------------------------
print(df.head())
print("\n🔢 Group Sizes:\n", df['Group'].value_counts())
print("\n✅ Conversion Rates:\n", df.groupby('Group')['Converted'].mean())

# -----------------------------------
# 3. Barplot for Conversion Rate
# -----------------------------------
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x='Group', y='Converted', estimator=np.mean, ci=95)
plt.title("Conversion Rate by Group")
plt.ylabel("Conversion Rate")
plt.tight_layout()
plt.show()

# -----------------------------------
# 4. Chi-Square Test
# -----------------------------------
contingency = pd.crosstab(df['Group'], df['Converted'])
chi2, p, _, _ = chi2_contingency(contingency)
print(f"\n📊 Chi-Square Test:\nChi2 = {chi2:.4f}, p-value = {p:.4f}")

# -----------------------------------
# 5. Revenue Comparison
# -----------------------------------
rev_a = df[(df['Group'] == 'A') & (df['Converted'] == 1)]['Revenue']
rev_b = df[(df['Group'] == 'B') & (df['Converted'] == 1)]['Revenue']

plt.figure(figsize=(8, 4))
sns.boxplot(data=df[df['Converted'] == 1], x='Group', y='Revenue')
plt.title("Revenue (Converted Users Only)")
plt.tight_layout()
plt.show()

t_stat, p_val = ttest_ind(rev_a, rev_b, equal_var=False)
print(f"\n💰 Revenue T-Test:\nt-stat = {t_stat:.4f}, p = {p_val:.4f}")

# -----------------------------------
# 6. Lift Metrics
# -----------------------------------
conv_lift = df[df['Group'] == 'B']['Converted'].mean() - df[df['Group'] == 'A']['Converted'].mean()
rev_lift = rev_b.mean() - rev_a.mean()
print(f"\n📈 Conversion Rate Lift: {conv_lift:.2%}")
print(f"📈 Revenue per Converter Lift: ${rev_lift:.2f}")

# -----------------------------------
# 7. Power Analysis (Frequentist)
# -----------------------------------
baseline = df[df['Group'] == 'A']['Converted'].mean()
target = df[df['Group'] == 'B']['Converted'].mean()
alpha = 0.05
power = 0.8
p1 = baseline
p2 = target
pooled_prob = (p1 + p2) / 2
z_alpha = norm.ppf(1 - alpha / 2)
z_beta = norm.ppf(power)
n_required = ((z_alpha + z_beta)**2 * (2 * pooled_prob * (1 - pooled_prob))) / ((p2 - p1)**2)
print(f"\n🧪 Required Sample Size per group (80% power, alpha=0.05): {int(np.ceil(n_required))}")

# -----------------------------------
# 8. Bayesian A/B Testing (Beta Posterior)
# -----------------------------------
success_a = df[df['Group'] == 'A']['Converted'].sum()
success_b = df[df['Group'] == 'B']['Converted'].sum()
total_a = df[df['Group'] == 'A'].shape[0]
total_b = df[df['Group'] == 'B'].shape[0]

samples = 100000
posterior_a = beta.rvs(success_a + 1, total_a - success_a + 1, size=samples)
posterior_b = beta.rvs(success_b + 1, total_b - success_b + 1, size=samples)
prob_b_better = np.mean(posterior_b > posterior_a)

plt.figure(figsize=(10, 4))
sns.kdeplot(posterior_a, label='Posterior A')
sns.kdeplot(posterior_b, label='Posterior B')
plt.title("Bayesian Posterior Distributions")
plt.xlabel("Conversion Rate")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\n📊 Bayesian Inference: P(B > A) = {prob_b_better:.2%}")

# -----------------------------------
# 9. Uplift Modeling / Segment-wise Insights
# -----------------------------------
# Encode group as binary
df['GroupFlag'] = df['Group'].map({'A': 0, 'B': 1})

# Optional: simulate a segment column
np.random.seed(42)
df['Segment'] = np.random.choice(['Mobile', 'Web', 'Email'], size=len(df))
le = LabelEncoder()
df['SegmentCode'] = le.fit_transform(df['Segment'])

# Uplift logistic model
features = ['GroupFlag', 'SegmentCode']
X = df[features]
y = df['Converted']
uplift_model = LogisticRegression()
uplift_model.fit(X, y)

# Predict probabilities and plot by segment
df['Predicted_Prob'] = uplift_model.predict_proba(X)[:, 1]

plt.figure(figsize=(8, 4))
sns.barplot(data=df, x='Segment', y='Predicted_Prob', hue='Group')
plt.title("Segment-wise Conversion Prediction (Uplift Modeling)")
plt.ylabel("Predicted Conversion Probability")
plt.tight_layout()
plt.show()
