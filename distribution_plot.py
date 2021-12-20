# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Read in data

morphed_attack = pd.read_csv(r'comparison_score_morphed.csv')
genuine = pd.read_csv(r'comparison_score_genuine.csv')
imposter = pd.read_csv(r'comparison_score_imposter.csv')

# Plot the data
sns.kdeplot(1-genuine['score'],
            color="g", label='Genuine')
sns.kdeplot(1-morphed_attack['score'],
            color='r', label='Morphing attack')
sns.kdeplot(1-imposter['score'],
            color='b', label='Imposter')


plt.title('Vulnerability assessment - ArcFace')
plt.xlabel('Scores')
plt.ylabel('Density')
plt.savefig('Vulnerability assessment - ArcFace')
plt.legend()
plt.show()