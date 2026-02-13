import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Define your data
"""
data = {
    "Terminal pair": [
        ("South", "North"),
        ("KUO V0941", "South"),
        ("APT V0002", "North"),
        ("South", "SIJ V0611"),
        ("North", "KUO V0002"),
        ("KUO V0002", "SOR V0001"),
        ("SIJ V0642", "SKM V0271"),
        ("North", "East"),
        ("SIJ V0632", "North"),
        ("North", "LNA V0002"),
        ("KUO V0002", "SIJ V0611"),
        ("South", "LNA V0001"),
        ("East", "SIJ V0642"),
        ("North", "SOR V0001"),
        ("LNA V0001", "East"),
        ("KUO V0002", "East"),
        ("South", "TOI V0001"),
        ("JKI V0422", "East"),
        ("TE V0002", "North"),
        ("SIJ V0632", "APT V0001"),
        ("TOI V0002", "SIJ V0611"),
        ("KUO V0002", "TOI V0001"),
        ("South", "JKI V0411"),
        ("SKM V0262", "East"),
        ("South", "East"),
        ("East", "TOI V0002"),
        ("APT V0001", "TOI V0002"),
        ("LNA V0001", "APT V0002"),
        ("TOI V0002", "North"),
        ("LNA V0001", "KUO V0002"),
        ("SIJ V0632", "LNA V0001"),
        ("TE V0001", "SIJ V0632"),
        ("South", "APT V0001"),
        ("South", "SOR V0001"),
        ("SOR V0001", "TOI V0001"),
        ("KUO V0002", "APT V0001")
    ],
    "Yearly traffic volume": [
        7614, 2116, 1917, 1824, 766, 528, 525, 212, 190, 143, 133, 98, 16, 13, 
        9, 8, 8, 7, 7, 5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1
    ]
}"""

data = {
    "Terminal pair": [
        ("South", "North"),
        ("KUO", "South"),  # KUO V0941 → KUO
        ("APT", "North"),  # APT V0002 → APT
        ("South", "SIJ"),  # SIJ V0611 → SIJ
        ("North", "KUO"),  # KUO V0002 → KUO
        ("KUO", "SOR"),  # KUO V0002, SOR V0001 → KUO, SOR
        ("SIJ", "SKM"),  # SIJ V0642, SKM V0271 → SIJ, SKM
        ("North", "East"),
        ("SIJ", "North"),  # SIJ V0632 → SIJ
        ("North", "LNA"),  # LNA V0002 → LNA
        ("KUO", "SIJ"),  # KUO V0002, SIJ V0611 → KUO, SIJ
        ("South", "LNA"),  # LNA V0001 → LNA
        ("East", "SIJ"),  # SIJ V0642 → SIJ
        ("North", "SOR"),  # SOR V0001 → SOR
        ("LNA", "East"),  # LNA V0001 → LNA
        ("KUO", "East"),  # KUO V0002 → KUO
        ("South", "TOI"),  # TOI V0001 → TOI
        ("JKI", "East"),  # JKI V0422 → JKI
        ("TE", "North"),  # TE V0002 → TE
        ("SIJ", "APT"),  # SIJ V0632, APT V0001 → SIJ, APT
        ("TOI", "SIJ"),  # TOI V0002, SIJ V0611 → TOI, SIJ
        ("KUO", "TOI"),  # KUO V0002, TOI V0001 → KUO, TOI
        ("South", "JKI"),  # JKI V0411 → JKI
        ("SKM", "East"),  # SKM V0262 → SKM
        ("South", "East"),
        ("East", "TOI"),  # TOI V0002 → TOI
        ("APT", "TOI"),  # APT V0001, TOI V0002 → APT, TOI
        ("LNA", "APT"),  # LNA V0001, APT V0002 → LNA, APT
        ("TOI", "North"),  # TOI V0002 → TOI
        ("LNA", "KUO"),  # LNA V0001, KUO V0002 → LNA, KUO
        ("SIJ", "LNA"),  # SIJ V0632, LNA V0001 → SIJ, LNA
        ("TE", "SIJ"),  # TE V0001, SIJ V0632 → TE, SIJ
        ("South", "APT"),  # APT V0001 → APT
        ("South", "SOR"),  # SOR V0001 → SOR
        ("SOR", "TOI"),  # SOR V0001, TOI V0001 → SOR, TOI
        ("KUO", "APT")  # KUO V0002, APT V0001 → KUO, APT
    ],
    "Yearly traffic volume": [
        7614, 2116, 1917, 1824, 766, 528, 525, 212, 190, 143, 133, 98, 16, 13, 
        9, 8, 8, 7, 7, 5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Extract all unique terminals
all_terminals = set()
for pair in df["Terminal pair"]:
    all_terminals.add(pair[0])
    all_terminals.add(pair[1])

terminals = list(all_terminals)
print(f"Unique terminals: {len(terminals)}")

# Create empty matrix with original order
matrix = np.zeros((len(terminals), len(terminals)))
terminal_to_idx = {terminal: i for i, terminal in enumerate(terminals)}

# Fill the matrix
for (source, target), volume in zip(df["Terminal pair"], df["Yearly traffic volume"]):
    i = terminal_to_idx[source]
    j = terminal_to_idx[target]
    matrix[i, j] = volume

for i in range(len(terminals)):
    for j in range(len(terminals)):
        #if i < j:  # Above diagonal
        if matrix[i, j] > 0:
            # Mirror it to below diagonal
            matrix[j, i] = matrix[i, j]

print(matrix)

# Calculate total traffic for each terminal (incoming + outgoing)
terminal_traffic = {}
for terminal in terminals:
    idx = terminal_to_idx[terminal]
    # Outgoing traffic (sum of row)
    outgoing = matrix[idx, :].sum()
    # Incoming traffic (sum of column)
    incoming = matrix[:, idx].sum()
    terminal_traffic[terminal] = outgoing + incoming

# Sort terminals by total traffic in ASCENDING order
# This puts low-traffic terminals first (top/left), high-traffic terminals last (bottom/right)
sorted_terminals_asc = [t for t, _ in sorted(terminal_traffic.items(), key=lambda x: x[1])]
print("\n=== Terminals in ASCENDING order (for lower left placement) ===")
for i, terminal in enumerate(sorted_terminals_asc):
    print(f"{i+1:2d}. {terminal:15s}: {terminal_traffic[terminal]:6,.0f}")

# Create new matrix with terminals sorted in ascending order
sorted_indices = [terminal_to_idx[t] for t in sorted_terminals_asc]
sorted_matrix = matrix[sorted_indices][:, sorted_indices]

# Create DataFrame with sorted terminals
sorted_heatmap_df = pd.DataFrame(sorted_matrix, 
                                index=sorted_terminals_asc, 
                                columns=sorted_terminals_asc)

print(sorted_heatmap_df)

# REVERSE THE X-AXIS ORDERING
# Keep y-axis (rows) as ascending, but reverse x-axis (columns)
reversed_columns = sorted_terminals_asc[::-1]  # Reverse the column order
reversed_df = sorted_heatmap_df[reversed_columns]  # Reorder columns

print("\n=== Column order REVERSED ===")
print("Y-axis (rows): Ascending order (low→high traffic)")
print("X-axis (columns): Descending order (high→low traffic)")
print("\nFirst few columns (left side):")
for i, col in enumerate(reversed_columns[:5]):
    print(f"  Col {i}: {col} ({terminal_traffic[col]:,.0f} total traffic)")

# Plot heatmap with lower triangle only and reversed x-axis
plt.figure(figsize=(16, 14))

# Create mask for lower triangle (based on the ORIGINAL ordering before reversal)
# We need to be careful: after reversing columns, what's "lower triangle" changes
# Let's create mask based on the ORIGINAL matrix ordering

# Create a mask matrix with same shape
mask_matrix = np.zeros_like(reversed_df.values, dtype=bool)
n = len(reversed_columns)

# For lower triangle in the DISPLAYED matrix (after column reversal):
# We want to mask cells where column index > row index in the DISPLAYED order
for i in range(n):
    for j in range(n):
        # In displayed matrix: rows i, columns j
        # Column j corresponds to terminal reversed_columns[j]
        # Row i corresponds to terminal sorted_terminals_asc[i]
        
        # Find positions in original sorted order
        row_terminal = sorted_terminals_asc[i]
        col_terminal = reversed_columns[j]
        
        # Get original indices in sorted order
        row_idx_original = sorted_terminals_asc.index(row_terminal)
        col_idx_original = sorted_terminals_asc.index(col_terminal)
        
        # Mask if original column index > original row index (upper triangle in original)
        mask_matrix[i, j] = col_idx_original > row_idx_original

# Alternative simpler approach: Just mask where displayed column index > displayed row index
# This gives us lower triangle in the VISUAL display
mask_simple = np.triu(np.ones((n, n), dtype=bool))

# Create the heatmap with reversed x-axis
ax = sns.heatmap(
    reversed_df,
    mask=mask_matrix, #mask_simple,  # Use simple mask for visual lower triangle
    annot=True,
    fmt='g',
    cmap='YlOrRd',
    norm=LogNorm(vmin=1, vmax=10000),
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={
        'label': 'Yearly Traffic Volume (log scale)',
        'extend': 'max'
    },
    annot_kws={'size': 20, 'weight': 'bold'},
    square=True
)

cbar = ax.collections[0].colorbar
cbar.set_label('Yearly Traffic Volume (log scale)', size=24)
cbar.ax.tick_params(labelsize=20)

# Customize the plot
#plt.title('Traffic Volumes', fontsize=20, fontweight='bold', pad=20)
#plt.xlabel('Terminal Node', fontsize=28)
#plt.ylabel('Terminal Node', fontsize=28)
#plt.xticks(rotation=90, fontsize=9)
#plt.yticks(rotation=0, fontsize=9)

# Add traffic volume annotations on axes
for i, terminal in enumerate(sorted_terminals_asc):  # y-axis in ascending order
    traffic = terminal_traffic[terminal]
    ax.get_yticklabels()[i].set_text(f"{terminal}\n({traffic:,.0f})")
    ax.get_yticklabels()[i].set_fontsize(20)

for i, terminal in enumerate(reversed_columns):  # x-axis in reversed (descending) order
    traffic = terminal_traffic[terminal]
    ax.get_xticklabels()[i].set_text(f"{terminal}\n({traffic:,.0f})")
    ax.get_xticklabels()[i].set_fontsize(20)


plt.xticks(rotation=0, fontsize=20)  # Changed from 9 to 11
plt.yticks(rotation=0, fontsize=20)   # Changed from 9 to 11

# Redraw the plot with updated labels
plt.draw()

# Add diagonal line for reference (now this will be from top-left to bottom-right)
#ax.plot([0, n], [0, n], color='blue', linewidth=1, linestyle='--', alpha=0.7)
ax.plot([0, n], [n, 0], color='blue', linewidth=1, linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])

# Save the figure
plt.savefig('traffic_volume_heatmap.pdf', 
            dpi=300, bbox_inches='tight')
plt.savefig('traffic_volume_heatmap.png', 
            dpi=300, bbox_inches='tight')

plt.show()