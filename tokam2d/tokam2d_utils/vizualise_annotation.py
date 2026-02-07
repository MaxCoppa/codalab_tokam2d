import matplotlib.pyplot as plt
import matplotlib.patches as patches


def vizualise_annotation(img, boxes_true, boxes_pred):

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="inferno", origin="upper")

    if boxes_true is not None:
        for box in boxes_true:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
            ax.add_patch(rect)

    if boxes_pred is not None:
        for box in boxes_pred:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

    ax.axis("off")
    plt.show()
