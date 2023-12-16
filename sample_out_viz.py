import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def img_plot(mat_path, img_save_path):
    """
    Plotting original image, segmentation mask and prediction for a random slice of a random image in the batch
    """
    data = loadmat(mat_path)
    orig_imgs, seg, pred = data["all_test_images"], data["all_test_labels"], data["all_test_outputs"]

    # Choose four random indices from the length of the batch
    indices = np.random.choice(len(orig_imgs), 4, replace=False)

    fig, axs = plt.subplots(4, 7, figsize=(30, 15))  # Create a 4x3 subplot grid

    for i, idx in enumerate(indices):
        slice_id = np.random.randint(30, 91)
        # Plotting original image
        axs[i, 0].imshow(orig_imgs[idx][1,1,slice_id], cmap="gray")
        axs[i, 0].set_title("Original Image")

        # Plotting segmentation mask
        axs[i, 1].imshow(seg[idx][1,0,slice_id]>0, cmap="gray")
        axs[i, 1].set_title("Segmentation Mask (Whole Tumor)")

        # Plotting prediction
        axs[i, 2].imshow(pred[idx][1,0,slice_id]>0, cmap="gray")
        axs[i, 2].set_title("Prediction for Whole Tumor Mask")
        
        # Plotting segmentation mask
        axs[i, 3].imshow(((seg[idx][1,0,slice_id]==1) + (seg[idx][1,0,slice_id]==3))>0, cmap="gray")
        axs[i, 3].set_title("Segmentation Mask (Tumor Core)")

        # Plotting prediction
        axs[i, 4].imshow(pred[idx][1,1,slice_id]>0, cmap="gray")
        axs[i, 4].set_title("Prediction for Tumor Core Mask")

        # Plotting segmentation mask
        axs[i, 5].imshow(seg[idx][1,0,slice_id]==3, cmap="gray")
        axs[i, 5].set_title("Segmentation Mask (Enhancing Tumor)")

        # Plotting prediction
        axs[i, 6].imshow(pred[idx][1,2,slice_id]>0, cmap="gray")
        axs[i, 6].set_title("Prediction for Enhancing Tumor Mask")

    # Hide x and y ticks for all subplots
    for ax in axs.flatten():
        ax.axis('off')

    plt.tight_layout()
    
    plt.savefig(img_save_path)  
    
def img_plot_deep_supervision(mat_path, img_save_path):
    data = loadmat(mat_path)
    orig_imgs, seg, pred, ds_outs = data["all_test_images"], data["all_test_labels"], data["all_test_outputs"], data["ds_outs"]

    # Choose four random indices from the length of the batch
    indices = np.random.choice(len(orig_imgs), 4, replace=False)

    fig, axs = plt.subplots(4, 13, figsize=(90, 20))  # Create a 4x3 subplot grid

    for i, idx in enumerate(indices):
        slice_id = np.random.randint(30, 91)
        # Plotting original image
        axs[i, 0].imshow(orig_imgs[idx][1,1,slice_id], cmap="gray")
        axs[i, 0].set_title("Original Image")

        # Plotting segmentation mask
        axs[i, 1].imshow(seg[idx][1,0,slice_id]>0, cmap="gray")
        axs[i, 1].set_title("Segmentation Mask (Whole Tumor)")
        
        # Plotting prediction
        axs[i, 2].imshow(ds_outs[idx][2][1,0,slice_id], cmap="gray")
        axs[i, 2].set_title("ds2 for Whole Tumor Mask")
        axs[i, 3].imshow(ds_outs[idx][1][1,0,slice_id], cmap="gray")
        axs[i, 3].set_title("ds1 for Whole Tumor Mask")
        axs[i, 4].imshow(pred[idx][1,0,slice_id]>0, cmap="gray")
        axs[i, 4].set_title("Prediction for Whole Tumor Mask")
        

        # Plotting segmentation mask
        axs[i, 5].imshow(((seg[idx][1,0,slice_id]==1) + (seg[idx][1,0,slice_id]==3))>0, cmap="gray")
        axs[i, 5].set_title("Segmentation Mask (Tumor Core)")

        # Plotting prediction
        axs[i, 6].imshow(ds_outs[idx][2][1,1,slice_id], cmap="gray")
        axs[i, 6].set_title("ds2 for Whole Tumor Mask")
        axs[i, 7].imshow(ds_outs[idx][1][1,1,slice_id], cmap="gray")
        axs[i, 7].set_title("ds1 for Whole Tumor Mask")
        axs[i, 8].imshow(pred[idx][1,1,slice_id]>0, cmap="gray")
        axs[i, 8].set_title("Prediction for Whole Tumor Mask")

        # Plotting segmentation mask
        axs[i, 9].imshow(seg[idx][1,0,slice_id]==3, cmap="gray")
        axs[i, 9].set_title("Segmentation Mask (Enhancing Tumor)")

        # Plotting prediction
        axs[i, 10].imshow(ds_outs[idx][2][1,2,slice_id], cmap="gray")
        axs[i, 10].set_title("ds2 for Whole Tumor Mask")
        axs[i, 11].imshow(ds_outs[idx][1][1,2,slice_id], cmap="gray")
        axs[i, 11].set_title("ds1 for Whole Tumor Mask")
        axs[i, 12].imshow(pred[idx][1,2,slice_id]>0, cmap="gray")
        axs[i, 12].set_title("Prediction for Whole Tumor Mask")

    # Hide x and y ticks for all subplots
    for ax in axs.flatten():
        ax.axis('off')

    plt.tight_layout()
    
    plt.savefig(img_save_path) 
    
def img_plot_deep_supervision2(mat_path, img_save_path):
    """
    Plotting original image, segmentation mask and prediction from deep supervision for a random slice of a random image in the batch
    """
    data = loadmat(mat_path)
    orig_imgs, seg, pred, ds_outs = data["all_test_images"], data["all_test_labels"], data["all_test_outputs"], data["ds_outs"]

    idx = np.random.randint(len(orig_imgs))  # Choose a random index

    slice_id = np.random.randint(30, 91)

    fig, axs = plt.subplots(3, 4, figsize=(40, 30))  # Create a 3x4 subplot grid

    # Plotting for Whole Tumor
    axs[0, 0].imshow(seg[idx][1, 0, slice_id] > 0, cmap="gray")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(ds_outs[idx][2][1, 0, slice_id], cmap="gray")
    axs[0, 1].axis('off')
    axs[0, 2].imshow(ds_outs[idx][1][1, 0, slice_id], cmap="gray")
    axs[0, 2].axis('off')
    axs[0, 3].imshow(pred[idx][1, 0, slice_id] > 0, cmap="gray")
    axs[0, 3].axis('off')

    # Plotting for Tumor Core
    axs[1, 0].imshow(((seg[idx][1, 0, slice_id] == 1) + (seg[idx][1, 0, slice_id] == 3)) > 0, cmap="gray")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(ds_outs[idx][2][1, 1, slice_id], cmap="gray")
    axs[1, 1].axis('off')
    axs[1, 2].imshow(ds_outs[idx][1][1, 1, slice_id], cmap="gray")
    axs[1, 2].axis('off')
    axs[1, 3].imshow(pred[idx][1, 1, slice_id] > 0, cmap="gray")
    axs[1, 3].axis('off')

    # Plotting for Enhancing Tumor
    axs[2, 0].imshow(seg[idx][1, 0, slice_id] == 3, cmap="gray")
    axs[2, 0].axis('off')
    axs[2, 1].imshow(ds_outs[idx][2][1, 2, slice_id], cmap="gray")
    axs[2, 1].axis('off')
    axs[2, 2].imshow(ds_outs[idx][1][1, 2, slice_id], cmap="gray")
    axs[2, 2].axis('off')
    axs[2, 3].imshow(pred[idx][1, 2, slice_id] > 0, cmap="gray")
    axs[2, 3].axis('off')


    for ax, col in zip(axs[0], ['Segmentation mask', 'First DS Output', 'Second DS Output', 'Prediction']):
        ax.set_title(col, fontsize=60)

    for ax, row in zip(axs[:,0], ['Class Whole Tumor', 'Class Tumor Core', 'Class Enhancing Tumor']):
        ax.set_ylabel(row, fontsize=60, labelpad=1000)

    plt.subplots_adjust(wspace=5, hspace=5)  # Adjust spacing between subplots
    
    plt.tight_layout()

    # Save the grid of subplots
    plt.savefig(img_save_path + "_grid.png", bbox_inches='tight')

    # Show the original image separately
    plt.figure(figsize=(5, 5))
    plt.imshow(orig_imgs[idx][1, 1, slice_id], cmap="gray")
    plt.title("Original Image")
    plt.axis('off')

    # Save the original image separately
    plt.savefig(img_save_path + "_original.png", bbox_inches='tight')