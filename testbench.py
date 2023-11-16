import noise
import numpy as np
import random
print(random.randint(0, 255))
# def calculate_average_difference():
#     noised_data = []
#     counter = 0
#     for i in range(10, 15000, 15):
#         flatLength = 100
#         startingPoint = random.uniform(0, 100000)
#         steepness_Level = np.interp(i, [0, 14000], [130, 140])
#         noisedY = abs(noise.pnoise1(startingPoint + (i - flatLength) / (700 - steepness_Level), octaves=4,
#                                         persistence=0.5,
#                                         lacunarity=0.1))
#         if noisedY < 0.1:
#             print(noisedY)
#             counter = counter + 1
#         noised_data.append(noisedY)
#
#     # Calculate the average difference of adjacent values in the list
#     differences = [noised_data[i] - noised_data[i-1] for i in range(1, len(noised_data))]
#     average_difference = sum(differences) / len(differences)
#     print(counter)
#     return average_difference
#
# # Number of runs to perform
# num_runs = 1000
#
# # Perform multiple runs and calculate the average of the average differences
# total_average_difference = 0
# for run in range(num_runs):
#     average_difference = calculate_average_difference()
#     total_average_difference += average_difference
#
# final_average_difference = total_average_difference / num_runs
# print("Final Average Difference over", num_runs, "runs:", final_average_difference)