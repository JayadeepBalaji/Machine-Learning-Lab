import pandas as pd
import matplotlib.pyplot as plt

# Read Iris dataset from CSV
df = pd.read_csv("C:\\Users\\pc\\OneDrive\\Desktop\\ML LAB\\DATASETS\\Iris.csv")

# Features and species
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
species = df['Species'].unique()
colors = ['blue', 'red', 'green']

# Plot histograms for each feature by species
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    # Calculate common bins for all species
    data_all = df[feature]
    bins = plt.hist(data_all, bins=10)[1]  # Get bin edges, don't plot
    plt.cla()  # Clear the plot so only our histograms show
    for j, specie in enumerate(species):
        plt.hist(
            df[df['Species'] == specie][feature],
            bins=bins,
            alpha=0.5,
            color=colors[j],
            label=f'class {specie}',
            edgecolor='black',
            histtype='bar'  # Overlap bars, not stack
        )
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    plt.title(f'Iris histogram #{i+1}')
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.legend()
plt.tight_layout()
plt.show()