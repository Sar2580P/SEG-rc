import yaml
import matplotlib.pyplot as plt
import textwrap

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def create_plot(images, titles, rows, cols, save_path):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), constrained_layout=True)
    axes = axes.flatten() if rows * cols > 1 else [axes]  # Flatten axes for easy iteration

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis("off")

            # Wrap text to avoid horizontal overlap
            wrapped_title = textwrap.fill(titles[i], width=30)  # Adjust width for better readability

            # Place title as text below the image
            ax.text(0.5, -0.1, wrapped_title, fontsize=9, ha="center", va="top", transform=ax.transAxes, wrap=True)
        else:
            ax.axis("off")  # Hide extra subplots if images are fewer than grid cells

    # Adjust layout to prevent overlapping
    plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.3, hspace=0.6)  # Increase wspace & hspace
    plt.savefig(save_path, bbox_inches="tight", dpi=300)  # Save with proper bounding box
    plt.close()