import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load the fitted model
with open('fitted_cox_model.pkl', 'rb') as f:
    cph = pickle.load(f)

# Load the dataset used for fitting
cox_data = pd.read_csv('heart_valve.csv')

# Plot the partial effects of 'age' and 'sex'
cph.plot_partial_effects_on_outcome('age', values=[40, 50, 60, 70, 80])
plt.title("Partial Effect of Age")
plt.savefig('partial_effect_age.png')
plt.close()

sex_values = [0, 1]  # 0 for Male, 1 for Female
cph.plot_partial_effects_on_outcome('sex', values=sex_values)
plt.title("Partial Effect of Sex")
plt.xticks([0, 1], ['Male', 'Female'])
plt.savefig('partial_effect_sex.png')
plt.close()

# Plot the hazard ratio for each covariate
cph.plot()
plt.title("Hazard Ratios for Covariates")
plt.tight_layout()
plt.savefig('hazard_ratios.png')
plt.close()

# Plot survival curves for different age groups
median_age = cox_data['age'].median()
young = cox_data.copy()
young['age'] = median_age - 10
old = cox_data.copy()
old['age'] = median_age + 10

cph.plot_partial_effects_on_outcome('age', [median_age - 10, median_age, median_age + 10])
plt.title("Survival Curves for Different Age Groups")
plt.savefig('survival_curves_age.png')
plt.close()
