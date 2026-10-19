import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TEXT, TEXT_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelcolor": TEXT_MUTED,
    "axes.edgecolor": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "text.color": TEXT,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.markersize": 5,
})


def load(filename):
    """Read a history CSV as a list of dicts of floats, or None if it is missing."""
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"  (missing: results/{filename})")
        return None
    with open(path, newline="") as file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]


def column(rows, name, percent=False):
    values = [row[name] for row in rows]
    return [v * 100 for v in values] if percent else values


def label_line_end(ax, x, y, text, color, dy=0.0):
    """Write a label just after the last point of a line (direct labelling)."""
    ax.annotate(
        text, xy=(x[-1], y[-1]), xytext=(6, dy), textcoords="offset points",
        color=TEXT, fontsize=9, va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1),
    )


def mark_stages(ax, boundary=3.5, y_text=None):
    """Shade stage 1 (frozen backbone) and label both stages."""
    ax.axvspan(0.5, boundary, color="#f2f1ee", zorder=0)
    ax.axvline(boundary, color=TEXT_MUTED, lw=1, ls=":", zorder=1)
    top = ax.get_ylim()[1] if y_text is None else y_text
    ax.text(2.0, top, "Stage 1: frozen", ha="center", va="top", fontsize=8, color=TEXT_MUTED)
    ax.text(6.0, top, "Stage 2: fine-tuning", ha="center", va="top", fontsize=8, color=TEXT_MUTED)


def save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: results/figures/{name}")


v1 = load("baseline_history_seed42.csv")
v2 = load("baseline_v2_history_seed42.csv")
resnet = load("resnet18_subset_history_seed42.csv")
resnet_random = load("resnet18_subset_random_init_history_seed42.csv")
resnet_ft_high = load("resnet18_subset_ft_lr1e-3_history_seed42.csv")
resnet_head_high = load("resnet18_subset_head_lr1e-2_history_seed42.csv")
resnet_s123 = load("resnet18_subset_history_seed123.csv")
resnet_s2026 = load("resnet18_subset_history_seed2026.csv")


# Figure 1: validation accuracy per epoch, all four models (seed 42).
if all([v1, v2, resnet_random, resnet]):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = [
        (v1, "Baseline v1", BLUE, -9),
        (v2, "Baseline v2", ORANGE, 9),
        (resnet_random, "ResNet-18, random weights", AQUA, 0),
        (resnet, "ResNet-18, ImageNet weights", YELLOW, 0),
    ]
    for rows, name, color, dy in series:
        x = column(rows, "epoch")
        y = column(rows, "validation_accuracy", percent=True)
        ax.plot(x, y, color=color, marker="o", label=name)
        label_line_end(ax, x, y, f"{y[-1]:.1f}%", color, dy=dy)
    ax.axhline(50, color=TEXT_MUTED, lw=1, ls="--")
    ax.text(1, 51, "chance (50%)", va="bottom", fontsize=8, color=TEXT_MUTED)
    ax.set_xlim(0.5, 22)
    ax.set_ylim(45, 102)
    ax.set_xticks(range(1, 21))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Validation accuracy per epoch (same 4,000 training / 1,000 validation images)")
    # The band between about 78% and 92% has no data, so the legend goes there.
    ax.legend(loc="center", bbox_to_anchor=(0.62, 0.71), ncol=2)
    save(fig, "fig1_model_comparison.png")

# Figure 2: best validation accuracy per model (one series, one colour).
if all([v1, v2, resnet_random, resnet]):
    stage1_best = max(r["validation_accuracy"] for r in resnet if r["stage"] == 1) * 100
    models = [
        ("Baseline v1\n(3 conv blocks)", max(column(v1, "validation_accuracy")) * 100),
        ("Baseline v2\n(4 blocks, BN,\ndropout)", max(column(v2, "validation_accuracy")) * 100),
        ("ResNet-18\nrandom weights", max(column(resnet_random, "validation_accuracy")) * 100),
        ("ResNet-18\nImageNet,\nfrozen backbone", stage1_best),
        ("ResNet-18\nImageNet,\nfine-tuned", max(column(resnet, "validation_accuracy")) * 100),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [m[0] for m in models]
    values = [m[1] for m in models]
    bars = ax.bar(names, values, color=BLUE, width=0.6, edgecolor="white", linewidth=2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%",
                ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title("Best validation accuracy by model (seed 42)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig2_best_accuracy.png")

# Figure 3: ResNet main model, accuracy and loss in two separate panels.
if resnet:
    x = column(resnet, "epoch")
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))

    for key, name, color, style in [
        ("train_accuracy", "Training", ORANGE, "--"),
        ("validation_accuracy", "Validation", BLUE, "-"),
    ]:
        y = column(resnet, key, percent=True)
        ax_acc.plot(x, y, color=color, ls=style, marker="o", label=name)
    ax_acc.set_ylim(90, 101.5)
    mark_stages(ax_acc)
    ax_acc.set_xticks(x)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend(loc="lower right")

    for key, name, color, style in [
        ("train_loss", "Training", ORANGE, "--"),
        ("validation_loss", "Validation", BLUE, "-"),
    ]:
        ax_loss.plot(x, column(resnet, key), color=color, ls=style, marker="o", label=name)
    ax_loss.set_ylim(0, 0.2)
    mark_stages(ax_loss)
    ax_loss.set_xticks(x)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88))

    fig.suptitle("ResNet-18 (ImageNet weights), seed 42", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_resnet_training.png")

# Figure 4: learning-rate comparison (validation accuracy).
if all([resnet, resnet_ft_high, resnet_head_high]):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = [
        (resnet, "Chosen: stage 1 LR 1e-3, stage 2 LR 1e-4", BLUE, 9),
        (resnet_ft_high, "Stage 2 LR 1e-3 (fine-tuning rate too high)", ORANGE, 0),
        (resnet_head_high, "Stage 1 LR 1e-2 (higher head rate)", AQUA, -9),
    ]
    for rows, name, color, dy in series:
        x = column(rows, "epoch")
        y = column(rows, "validation_accuracy", percent=True)
        ax.plot(x, y, color=color, marker="o", label=name)
        label_line_end(ax, x, y, f"{y[-1]:.1f}%", color, dy=dy)
    ax.set_ylim(88, 101)
    mark_stages(ax)
    ax.set_xlim(0.5, 9)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("ResNet-18: effect of the learning rate (seed 42)")
    ax.legend(loc="lower left")
    save(fig, "fig4_learning_rates.png")

# Figure 5: main model over three seeds.
if all([resnet, resnet_s123, resnet_s2026]):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = [
        (resnet, "Seed 42", BLUE, -9),
        (resnet_s123, "Seed 123", ORANGE, 4),
        (resnet_s2026, "Seed 2026", AQUA, 6),
    ]
    best = []
    for rows, name, color, dy in series:
        x = column(rows, "epoch")
        y = column(rows, "validation_accuracy", percent=True)
        best.append(max(y))
        ax.plot(x, y, color=color, marker="o", label=name)
        label_line_end(ax, x, y, f"{y[-1]:.1f}%", color, dy=dy)
    mean = sum(best) / len(best)
    std = (sum((b - mean) ** 2 for b in best) / (len(best) - 1)) ** 0.5
    ax.set_ylim(92, 100.5)
    mark_stages(ax)
    ax.set_xlim(0.5, 9)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"ResNet-18 over three seeds: best {mean:.2f}% ± {std:.2f}%")
    ax.legend(loc="lower right")
    save(fig, "fig5_seeds.png")

# Figure 6: baselines, training vs validation accuracy (two panels).
if v1 and v2:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, rows, title in [(axes[0], v1, "Baseline v1"), (axes[1], v2, "Baseline v2")]:
        x = column(rows, "epoch")
        ax.plot(x, column(rows, "train_accuracy", percent=True),
                color=ORANGE, ls="--", marker="o", label="Training")
        ax.plot(x, column(rows, "validation_accuracy", percent=True),
                color=BLUE, marker="o", label="Validation")
        ax.set_title(title)
        ax.set_xticks(range(2, 21, 2))
        ax.set_xlabel("Epoch")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(50, 80)
    fig.suptitle("Baseline CNNs trained from scratch, seed 42", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_baselines.png")