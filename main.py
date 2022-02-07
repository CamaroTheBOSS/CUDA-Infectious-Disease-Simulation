import matplotlib.pyplot as plt
import pandas as pd
import os
path = os.path.abspath("C:/Users/CamaroTheBOSS/Desktop/Code projects/C++/CUDA Infectious Disease Simulation/outputs.txt")
data = pd.read_csv(path)
if data.iloc[-1][0] < 0:
    data.drop(data.tail(1).index, inplace=True)  # drop last n rows

infected, = plt.plot(data['Infected'], label='Infected')
dead, = plt.plot(data['Dead'], label='Dead')
plt.legend(handles=[infected, dead])

# plt.plot(data)
# plt.legend(data.columns)
plt.show()
