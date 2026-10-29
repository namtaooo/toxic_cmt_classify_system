import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42})
rows = [("Tier 1: TF-IDF", 0.93, 0.8690, 0), ("PhoBERT-CNN (tier 2)", 97.4, 0.9367, 0.0012), ("ViHateT5-base", 315.1, 0.9279, 0.0022),
        ("ViSoBERT-CNN", 113.0, 0.9000, 0.0019), ("Three-tier system", 36.5, 0.9335, 0)]
GREY, BLUE, INK = "#a3a29c", "#2a78d6", "#0b0b0b"
fig, ax = plt.subplots(figsize=(3.5, 2.45))
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
off = {"Tier 1: TF-IDF": (6, -2, "left"), "PhoBERT-CNN (tier 2)": (6, 3, "left"), "ViHateT5-base": (0, -11, "center"),
       "ViSoBERT-CNN": (6, -2, "left"), "Three-tier system": (-6, -2, "right")}
for name, ms, f1, sd in rows:
    c = BLUE if name.startswith("Three") else GREY
    if sd: ax.errorbar(ms, f1, yerr=sd, fmt="none", ecolor=c, elinewidth=1.2, capsize=0, zorder=2)
    ax.scatter([ms], [f1], s=30 if c == BLUE else 24, color=c, zorder=3, edgecolor="white", linewidth=0.8)
    dx, dy, ha = off[name]
    ax.annotate(name, (ms, f1), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=6.5, color=INK if c == BLUE else "#52514e", fontweight="bold" if c == BLUE else "normal")
ax.annotate("", xy=(36.5 * 1.12, 0.9335), xytext=(97.4 / 1.12, 0.9367),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8, linestyle=(0, (3, 2))))
ax.text(60, 0.9445, "2.7× faster, macro-F1 −0.003", ha="center", fontsize=6.3, color=INK)
ax.set_xscale("log"); ax.set_xlim(0.5, 900); ax.set_ylim(0.855, 0.955)
ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"]); ax.minorticks_off()
ax.set_xlabel("Mean CPU latency per comment (ms, log scale)"); ax.set_ylabel("Test macro-F1")
plt.tight_layout(pad=0.3); plt.savefig("results/figure/f1_vs_latency.pdf"); plt.savefig("results/figure/f1_vs_latency.png", dpi=250)
