import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(font_scale=1.3)
sns.set_style("whitegrid")
# Data
models = ['w/ cursor map', 'w/o cursor map', 'Random']
dx = np.array([1.569, 130.041, 175.427])
dy = np.array([1.447, 95.758, 126.878])
dr = np.array([2.282, 173.727, 237.840])
#dr = np.sqrt(dx**2 + dy**2)

# X locations
x = np.arange(len(models))
width = 0.25  # bar width

fig, ax = plt.subplots()
ax.bar(x - width, dx, width, label='Δx')
ax.bar(x,        dy, width, label='Δy')
ax.bar(x + width, dr, width, label='Δr')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Error (pixels)')
#ax.set_title('Cursor Position Error Comparison')
ax.legend()

plt.tight_layout()
plt.savefig('cursor_error_comparison.png')

