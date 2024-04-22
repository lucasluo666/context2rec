import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [0.3160400986671448, 0.26670220494270325, 0.19867345690727234, 0.035958562046289444, 0.0046129412949085236]

# Create the line graph
plt.figure(figsize=(6, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b')

# Title and labels
plt.title('Amazon (Music Instruments)', fontsize=16)
plt.xlabel('Number of Training Reviews', fontsize=16)
plt.ylabel('Reduction in MSE', fontsize=16)

# Set x-ticks to only display the specific values [1, 2, 3, 4, 5]
plt.xticks(x)

# save plot
plt.savefig('./reduction_in_mse_vs_training_reviews_amazon.png')

# Show grid
plt.grid(True)

# Show the plot
plt.show()