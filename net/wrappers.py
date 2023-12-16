import torch
import os
import numpy as np
from tqdm import tqdm
from scipy.io import savemat, loadmat
from net.loss import DiceCoeff, LossBraTS, compute_loss
import matplotlib.pyplot as plt

def trainer(num_epochs, train_loader, val_loader, model, optimizer, criterion,
             best_model_path='DL_results', dim3d=False, deep_supervision=False, device='cuda:0', amp=True):

    best_model_save_path = ""
    best_val_loss = float('inf')  # Initialize with a high value
    best_epoch = -1
    scaler = torch.cuda.amp.GradScaler()
    train_loss_list = []
    val_loss_list = []
    val_dice_0_list = []
    val_dice_1_list = []
    val_dice_2_list = []

    print(f"Training started for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch_idx, (batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")):
            if dim3d:
                

                images, labels = batch['image'].float().to(device), batch['label'].to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = compute_loss(outputs, labels, criterion, deep_supervision)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                # loss.backward()
                # optimizer.step()

                scaler.update()

                train_loss += loss.item() * images.size(0)
                train_samples += images.size(0)
            else:
                # for img2d, label2d in zip(batch['image'],batch['label']):
                for idx in range(batch['image'].shape[2]):
                    img2d, label2d = batch['image'][:,:,idx], batch['label'][:,:,idx]
                    images, labels = img2d.float().to(device), label2d.to(device)

                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=amp):
                        outputs = model(images)
                        loss = compute_loss(outputs, labels, criterion, deep_supervision)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    # loss.backward()
                    # optimizer.step()

                    scaler.update()

                    train_loss += loss.item() * images.size(0)
                    train_samples += images.size(0)
                    
            
        train_loss /= train_samples
        train_loss_list.append(train_loss)


        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_dice_0 = 0.0
        val_dice_1 = 0.0
        val_dice_2 = 0.0
        

        with torch.no_grad():
            for batch_idx, (val_batch) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")):
                if dim3d:
                    val_images, val_labels = val_batch['image'].float().to(device), val_batch['label'].to(device)

                    with torch.cuda.amp.autocast():
                        val_outputs = model(val_images)
                        val_loss_batch = compute_loss(val_outputs, val_labels, criterion, deep_supervision)
                        val_dice_cal0, val_dice_cal1, val_dice_cal2 = DiceCoeff()(val_outputs[0] if deep_supervision else val_outputs, val_labels)

                    val_loss += val_loss_batch.item() * val_images.size(0)
                    val_samples += val_images.size(0)
                    val_dice_0 += val_dice_cal0.item() * val_images.size(0)
                    val_dice_1 += val_dice_cal1.item() * val_images.size(0)
                    val_dice_2 += val_dice_cal2.item() * val_images.size(0)
                else:
                    # for img2d, label2d in zip(val_batch['image'],val_batch['label']):
                    for idx in range(val_batch['image'].shape[2]):
                        img2d, label2d = val_batch['image'][:,:,idx], val_batch['label'][:,:,idx]
                        val_images, val_labels = img2d.float().to(device), label2d.to(device)

                        with torch.cuda.amp.autocast(enabled=amp):
                            val_outputs = model(val_images)
                            val_loss_batch = compute_loss(val_outputs, val_labels, criterion, deep_supervision)
                            val_dice_cal0, val_dice_cal1, val_dice_cal2 = DiceCoeff()(val_outputs[0] if deep_supervision else val_outputs, val_labels)


                        val_loss += val_loss_batch.item() * val_images.size(0)
                        val_samples += val_images.size(0)
                        val_dice_0 += val_dice_cal0.item() * val_images.size(0)
                        val_dice_1 += val_dice_cal1.item() * val_images.size(0)
                        val_dice_2 += val_dice_cal2.item() * val_images.size(0)

            val_loss /= val_samples
            val_loss_list.append(val_loss)
            val_dice_0 /= val_samples
            val_dice_1 /= val_samples
            val_dice_2 /= val_samples
            val_dice_0_list.append(val_dice_0)
            val_dice_1_list.append(val_dice_1)
            val_dice_2_list.append(val_dice_2)

        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}, Validation Dice 0: {val_dice_0:.4f}, Validation Dice 1: {val_dice_1:.4f}, Validation Dice 2: {val_dice_2:.4f}")

        # Save the model if it achieves the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(best_model_path, f"best_model_{epoch + 1}_loss_{best_val_loss:.4f}.pth"))
            if os.path.exists(best_model_save_path):
                os.remove(best_model_save_path)
            best_model_save_path = os.path.join(best_model_path, f"best_model_{epoch + 1}_loss_{best_val_loss:.4f}.pth")
            print(f"Model saved at Epoch {best_epoch} with Validation Loss: {best_val_loss:.4f}")
        

    print(f"Training completed. Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    return {'model': model, 'train_loss': train_loss_list, 'val_loss': val_loss_list,
             'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'optimizer': optimizer,
               'val_dice_0': val_dice_0_list, 'val_dice_1': val_dice_1_list, 'val_dice_2': val_dice_2_list}
     
def tester(test_loader, model, criterion, device='cuda:0', dim3d=False, deep_supervision=False, save_path="results2"):
    model.eval()
    test_loss = 0.0
    test_samples = 0
    test_dice_0 = 0.0
    test_dice_1 = 0.0
    test_dice_2 = 0.0

    all_test_images = []
    all_test_labels = []
    all_test_outputs = []
    
    ds_outputs = []
    
    with torch.no_grad():
        for batch_idx, (test_batch) in enumerate(tqdm(test_loader, desc=f"Testing")):
            if dim3d:
                test_images, test_labels = test_batch['image'].float().to(device), test_batch['label'].to(device)
                
                with torch.cuda.amp.autocast():
                    test_outputs = model(test_images)
                    test_loss_batch = compute_loss(test_outputs, test_labels, criterion, deep_supervision)
                    test_dice_cal0, test_dice_cal1, test_dice_cal2 = DiceCoeff()(test_outputs[0] if deep_supervision else test_outputs, test_labels)

                all_test_images.append(test_images.cpu().numpy())
                all_test_labels.append(test_labels.cpu().numpy())
                temp = test_outputs[0] if deep_supervision else test_outputs
                all_test_outputs.append(temp.cpu().numpy()) 
                
                if deep_supervision:
                    print(type(test_outputs))
                    ds_outputs.append([out.cpu().numpy() for out in test_outputs])

                test_loss += test_loss_batch.item() * test_images.size(0)
                test_samples += test_images.size(0)
                test_dice_0 += test_dice_cal0.item() * test_images.size(0)
                test_dice_1 += test_dice_cal1.item() * test_images.size(0)
                test_dice_2 += test_dice_cal2.item() * test_images.size(0)  
                
            else:
                # for img2d, label2d in zip(test_batch['image'],test_batch['label']):
                for idx in range(test_batch['image'].shape[2]):
                    img2d, label2d = test_batch['image'][:,:,idx], test_batch['label'][:,:,idx]
                    test_images, test_labels = img2d.float().to(device), label2d.to(device)

                    with torch.cuda.amp.autocast():
                        test_outputs = model(test_images)
                        test_loss_batch = compute_loss(test_outputs, test_labels, criterion, deep_supervision)
                        test_dice_cal0, test_dice_cal1, test_dice_cal2 = DiceCoeff()(test_outputs[0] if deep_supervision else test_outputs, test_labels)

                    all_test_images.append(test_images.cpu().numpy())
                    all_test_labels.append(test_labels.cpu().numpy())
                    all_test_outputs.append(test_outputs.cpu().numpy()) 

                    test_loss += test_loss_batch.item() * test_images.size(0)
                    test_samples += test_images.size(0)
                    test_dice_0 += test_dice_cal0.item() * test_images.size(0)
                    test_dice_1 += test_dice_cal1.item() * test_images.size(0)
                    test_dice_2 += test_dice_cal2.item() * test_images.size(0)


        test_loss /= test_samples
        test_dice_0 /= test_samples
        test_dice_1 /= test_samples
        test_dice_2 /= test_samples

    print(f"Testing Loss: {test_loss:.4f}, Testing Dice 0: {test_dice_0:.4f}, Testing Dice 1: {test_dice_1:.4f}, Testing Dice 2: {test_dice_2:.4f}")
    i=5
    savemat(os.path.join(save_path, 'output.mat'), {'all_test_images': all_test_images[:i], 
                                                    'all_test_labels': all_test_labels[:i],
                                                    'all_test_outputs': all_test_outputs[:i], 
                                                    'ds_outs': ds_outputs[:i],
                                                    'test_loss': test_loss, 'test_dice_0': test_dice_0, 'test_dice_1': test_dice_1, 'test_dice_2': test_dice_2})
    
    img_plot(os.path.join(save_path, 'output.mat'), os.path.join(save_path, 'output.png'))
    return {'test_loss': test_loss, 'test_dice_0': test_dice_0, 'test_dice_1': test_dice_1, 'test_dice_2': test_dice_2}


def plot_loss(train_loss, val_loss, best_epoch, best_val_loss, save_path='DL_results'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(x=best_epoch - 1, color='r', linestyle='--', label=f'Best Model at Epoch {best_epoch} with Validation Loss: {best_val_loss:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.show()
    
def img_plot(mat_path, img_save_path):
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

    fig, axs = plt.subplots(4, 13, figsize=(45, 10))  # Create a 4x3 subplot grid

    for i, idx in enumerate(indices):
        slice_id = np.random.randint(30, 91)
        # Plotting original image
        axs[i, 0].imshow(orig_imgs[idx][1,1,slice_id], cmap="gray")
        axs[i, 0].set_title("Original Image")

        # Plotting segmentation mask
        axs[i, 1].imshow(seg[idx][1,0,slice_id]>0, cmap="gray")
        axs[i, 1].set_title("Segmentation Mask (Whole Tumor)")
        
        # Plotting prediction
        axs[i, 2].imshow(ds_outs[idx][2][1,0,slice_id]>0, cmap="gray")
        axs[i, 2].set_title("ds2 for Whole Tumor Mask")
        axs[i, 3].imshow(ds_outs[idx][1][1,0,slice_id]>0, cmap="gray")
        axs[i, 3].set_title("ds1 for Whole Tumor Mask")
        axs[i, 4].imshow(pred[idx][1,0,slice_id]>0, cmap="gray")
        axs[i, 4].set_title("Prediction for Whole Tumor Mask")
        

        # Plotting segmentation mask
        axs[i, 5].imshow(((seg[idx][1,0,slice_id]==1) + (seg[idx][1,0,slice_id]==3))>0, cmap="gray")
        axs[i, 5].set_title("Segmentation Mask (Tumor Core)")

        # Plotting prediction
        axs[i, 6].imshow(ds_outs[idx][2][1,1,slice_id]>0, cmap="gray")
        axs[i, 6].set_title("ds2 for Whole Tumor Mask")
        axs[i, 7].imshow(ds_outs[idx][1][1,1,slice_id]>0, cmap="gray")
        axs[i, 7].set_title("ds1 for Whole Tumor Mask")
        axs[i, 8].imshow(pred[idx][1,1,slice_id]>0, cmap="gray")
        axs[i, 8].set_title("Prediction for Whole Tumor Mask")

        # Plotting segmentation mask
        axs[i, 9].imshow(seg[idx][1,0,slice_id]==3, cmap="gray")
        axs[i, 9].set_title("Segmentation Mask (Enhancing Tumor)")

        # Plotting prediction
        axs[i, 10].imshow(ds_outs[idx][2][1,2,slice_id]>0, cmap="gray")
        axs[i, 10].set_title("ds2 for Whole Tumor Mask")
        axs[i, 11].imshow(ds_outs[idx][1][1,2,slice_id]>0, cmap="gray")
        axs[i, 11].set_title("ds1 for Whole Tumor Mask")
        axs[i, 12].imshow(pred[idx][1,2,slice_id]>0, cmap="gray")
        axs[i, 12].set_title("Prediction for Whole Tumor Mask")

    # Hide x and y ticks for all subplots
    for ax in axs.flatten():
        ax.axis('off')

    plt.tight_layout()
    
    plt.savefig(img_save_path) 