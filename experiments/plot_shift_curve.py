import numpy as np
import matplotlib.pyplot as plt

data = np.load("shift_results.npy", allow_pickle=True)

sigmas = [x[0] for x in data]
accs = [x[1] for x in data]

plt.plot(sigmas, accs, marker='o')
plt.xlabel("Shift magnitude (sigma)")
plt.ylabel("Accuracy")
plt.title("ShiftLab: Performance under controlled shift")
plt.show()
