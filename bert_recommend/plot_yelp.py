import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [0.141, 0.125, 0.083, 0.072, 0.0593]

# Create the line graph
plt.figure(figsize=(6, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b')

# Title and labels
plt.title('Yelp', fontsize=16)
plt.xlabel('Number of Training Reviews', fontsize=16)
plt.ylabel('Reduction in MSE', fontsize=16)

# Set x-ticks to only display the specific values [1, 2, 3, 4, 5]
plt.xticks(x)

# save plot
plt.savefig('./reduction_in_mse_vs_training_reviews_yelp.png')

# Show grid
plt.grid(True)

# Show the plot
plt.show()