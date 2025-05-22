import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load confusion matrix from pickle file
filename = 'cont20allnewdatapsearch_a_vis_norm_standard_contextnewdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_ckpt116000/test_32/confusion_matrix.pkl'
filename = 'heatmapcont20allnewdatapsearch_a_vis_norm_standard_contextnewdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_ckpt116000/test_32/confusion_matrix.pkl'
filename = '73heatmapcont20allnewdatapsearch_a_vis_norm_standard_contextnewdata_pretrainchallenging_addc_allnew_more_c_alldata_diffusion_c_alldata_joint_noss_4Xb_ss005_cont_lr2e5_context64_b16_ckpt116000/test_32/confusion_matrix.pkl'
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Seaborn style
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# 2. Load confusion matrix
with open(filename, 'rb') as f:
    confusion = pickle.load(f)  # assumed shape (K, K)

# 3. Avoid division by zero: ensure each row sum ≥ 1
row_sums = confusion.sum(axis=1, keepdims=True)
row_sums = np.maximum(row_sums, 1)

# 4. Compute per-true-cluster accuracy percentages
accuracy = confusion / row_sums * 100

# 5. Truncate to maximum 16 clusters
K = accuracy.shape[0]
if False and K > 16:
    accuracy = accuracy[:16, :16]
    labels = list(range(16))
else:
    labels = list(range(K))

# 6. Create DataFrame for heatmap
df = pd.DataFrame(accuracy, index=labels, columns=labels)

# 7. Custom annotation formatter
def custom_annot(x):
    if x == 100:
        return f"{int(x)}"
    return f"{x:.1f}"
cmap='RdYlGn'
cmap= 'YlGnBu'
# 8. Plot heatmap
plt.figure(figsize=(12*1.0, 7.4*1.0))
ax = sns.heatmap(
    df,
    #annot=df.applymap(custom_annot),
    fmt="",
    cmap=cmap,
    cbar=True,
    linewidths=0.5,
    annot_kws={"size": 11},
    cbar_kws={"shrink": 0.75},
    vmin=0
)
ax.set_aspect('equal', 'box')
# 9. Format axes
ax.set_xlabel("Predicted Cluster")
ax.set_ylabel("Ground Truth Cluster")
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.tick_params(axis="x", which="both", length=0)
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
plt.xticks([])
plt.yticks([])
# 10. Colorbar label
cbar = ax.collections[0].colorbar
cbar.set_label("Percentage")

# 11. Save and show
plt.tight_layout()
plt.savefig("73state_transition_heatmap_full.png", dpi=300)
plt.show()

