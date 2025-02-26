from modules import *
# probability practice
arr_1 = [1,3,5,7,9]
prob_1 = [0.1, 0.2, 0, 0.45, 0.25]
prob_arr_1 = np.random.choice(arr_1, p=prob_1, size=(100))
# Count occurrences of each number
unique, counts = np.unique(prob_arr_1, return_counts=True)

# Display results
count_dict = dict(zip(unique, counts))
print(count_dict)