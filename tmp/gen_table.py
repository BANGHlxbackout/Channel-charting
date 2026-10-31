import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

columns = ['方法', 'CT ↑', 'TW ↑', 'KS ↓', 'Mean(m) ↓', 'P95(m) ↓', 'Max(m) ↓']
data = [
    ['ADP Fused',    '0.9845', '0.9899', '0.1063', '0.5275', '1.1303', '9.1438'],
    ['ADP-Siamese',  '0.9957', '0.9960', '0.0840', '0.4093', '0.8558', '7.8936'],
    ['Graph FCF',    '0.9969', '0.9968', '0.0793', '0.3998', '0.8463', '1.4632'],
]

fig, ax = plt.subplots(figsize=(10, 2.2))
ax.axis('off')

n_rows = len(data)
n_cols = len(columns)

col_widths = [0.16, 0.12, 0.12, 0.12, 0.14, 0.14, 0.14]
x_starts = []
x = 0.02
for w in col_widths:
    x_starts.append(x)
    x += w

y_top = 0.92
row_h = 0.18
header_y = y_top

ax.plot([0.01, 0.97], [y_top + 0.06, y_top + 0.06], color='black', linewidth=1.2, transform=ax.transAxes, clip_on=False)

for j, col in enumerate(columns):
    ax.text(x_starts[j] + col_widths[j] / 2, header_y, col,
            ha='center', va='center', fontsize=11, fontweight='bold',
            transform=ax.transAxes)

line_y = header_y - 0.09
ax.plot([0.01, 0.97], [line_y, line_y], color='black', linewidth=0.8, transform=ax.transAxes, clip_on=False)

for i, row in enumerate(data):
    y = header_y - 0.12 - i * row_h
    is_best = (i == n_rows - 1)
    weight = 'bold' if is_best else 'normal'
    for j, val in enumerate(row):
        ax.text(x_starts[j] + col_widths[j] / 2, y, val,
                ha='center', va='center', fontsize=10.5, fontweight=weight,
                transform=ax.transAxes)
    sep_y = y - 0.09
    if i < n_rows - 1:
        ax.plot([0.01, 0.97], [sep_y, sep_y], color='black', linewidth=0.4, transform=ax.transAxes, clip_on=False)

bottom_y = header_y - 0.12 - (n_rows - 1) * row_h - 0.09
ax.plot([0.01, 0.97], [bottom_y, bottom_y], color='black', linewidth=1.2, transform=ax.transAxes, clip_on=False)

fig.patch.set_facecolor('white')
plt.savefig(r'd:\diffusion_refined_charting\tmp\method_comparison.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.close()
print('done')
