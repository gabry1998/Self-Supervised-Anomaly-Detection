from self_supervised.datasets import  MVTecDatamodule
from self_supervised.model import *
from self_supervised.support.dataset_generator import *
from self_supervised.support.functional import *
from self_supervised.support.cutpaste_parameters import CPP
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import random
from torchvision import transforms
from sklearn import preprocessing
import self_supervised.metrics as mtr
from self_supervised.support.visualization import localize, plot_heatmap, plot_curve 

    
      
def test_GDE_image_level():

    datamodule = MVTecDatamodule(
      'dataset/bottle/',
      'bottle',
      (256,256),
      64,
      0
    )
    datamodule.setup()
    
    sslm = SSLM.load_from_checkpoint('outputs/computations/bottle/image_level/best_model.ckpt')
    sslm.eval()
    sslm.to('cuda')
    
    train_embed = []
    for x, _ in datamodule.train_dataloader():
      y_hat, embeddings = sslm(x.to('cuda'))
      embeddings = embeddings.to('cpu')
      train_embed.append(embeddings)
    train_embed = torch.cat(train_embed).to('cpu').detach()

    
    print(train_embed.shape)
    
    test_labels = []
    test_embeds = []
    with torch.no_grad():
        for x, label in datamodule.test_dataloader():
            y_hat, embeddings = sslm(x.to('cuda'))

            # save 
            test_embeds.append(embeddings.to('cpu').detach())
            test_labels.append(label.to('cpu').detach())
    test_labels = torch.cat(test_labels)
    test_embeds = torch.cat(test_embeds)
    
    print(test_embeds.shape)
    print(test_labels.shape)
    #test_embeds = torch.nn.functional.normalize(test_embeds, p=2, dim=1)
    #train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)
    
    gde = GDE()
    gde.fit(train_embed)
    scores = gde.predict(test_embeds)
    
    print(scores.shape)
    print(scores)

    
    
    int_labels = []
    for x in test_labels:
      if torch.sum(x) == 0:
        int_labels.append(0)
      else:
        int_labels.append(1)
    print(int_labels)
    test_labels = torch.tensor(int_labels)
    
    plot_curve(test_labels, scores)


def test_1d_to_2d():
    x = torch.randn((3249,128))

    print(x.shape)
    print(x)
    x1 = np.resize(np.array(x), (int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0]))))
    print(x1.shape)
    print(x1)


def test_patch_level():
    imsize = (256,256)
    train_img = Image.open('dataset/bottle/test/good/000.png').resize(imsize).convert('RGB')
    test_img = Image.open('dataset/bottle/test/broken_large/001.png').resize(imsize).convert('RGB') 
    
    train_img_tensor = transforms.ToTensor()(train_img)
    train_img_tensor_norm = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(train_img_tensor)
    train_img_tensor_norm = train_img_tensor_norm.unsqueeze(0)
    
    test_img_tensor = transforms.ToTensor()(test_img)
    test_img_tensor_norm = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(test_img_tensor)
    test_img_tensor_norm = test_img_tensor_norm.unsqueeze(0)
    
    sslm = SSLM().load_from_checkpoint('outputs/computations/bottle/patch_level/best_model.ckpt')
    sslm.to('cuda')
    sslm.eval()
    sslm.unfreeze_layers(False)
    
    patches = extract_patches(test_img_tensor_norm, 32, 4)
    
    print('inferencing')
    start = time.time()
    y_hat, embeddings = sslm(patches.to('cuda'))
    y_hat = get_prediction_class(y_hat.to('cpu'))
    print(torch.sum(y_hat)> 0)
    end = time.time() - start
    print('done in', end, 'sec')
    
    print('getting train embeds')
    train_patches = extract_patches(train_img_tensor_norm, 32, 4)
    _, train_embedding = sslm(train_patches.to('cuda'))
    
    gde = GDE()
    gde.fit(train_embedding.to('cpu'))
    print('predicting')
    start = time.time()
    embeddings = embeddings.to('cpu')
    mvtec_test_scores = gde.predict(embeddings)
    end = time.time() - start
    print('done in', end, 'sec')
    
    dim = int(np.sqrt(embeddings.shape[0]))
    out = torch.reshape(mvtec_test_scores, (dim, dim))
    saliency_map_min, saliency_map_max = out.min(), out.max()
    out = (out - saliency_map_min).div(saliency_map_max - saliency_map_min).data
    out[out < 0.35] = 0
    gs = GaussianSmooth(device='cpu')
    out = gs.upsample(out[None, None, :])
    saliency_map_min, saliency_map_max = out.min(), out.max()
    out = (out - saliency_map_min).div(saliency_map_max - saliency_map_min).data

    heatmap = localize(test_img_tensor[None, :], out)
    print(heatmap.min(), heatmap.max())
    image = imagetensor2array(test_img_tensor)
    print(image.min(), image.max())
    heatmap = np.uint8(255 * heatmap)
    image = np.uint8(255 * image)
    plot_heatmap(image, heatmap)


def test_reshape():
    gt = get_mvtec_gt_filename_counterpart(
        'dataset/bottle/test/good/000.png', 
        'dataset/bottle/ground_truth')
    gt = ground_truth(gt)

    gt = transforms.ToTensor()(gt).unsqueeze(0)
    print(gt.shape)
    gt_patches = extract_mask_patches(gt, 32, 4)
    print(gt_patches.shape)
    gt_labels = gt2label(gt_patches)
    print(len(gt_labels))
    x = torch.randn((4,3,3))
    print(x)
    print(x.shape)
    num_batch, num_patches, embedding_size = x.shape
    y = torch.reshape(x, (num_patches, num_batch*embedding_size))
    
    x1 = x.flatten()
    x2 = x1.split(embedding_size)
    start = time.time()
    for h in range(num_patches):
        t = torch.tensor([])
        for i in range(len(x2)):
            if (i+h)%num_patches == 0:
                t = torch.cat([t, x2[i]])
        y[h] = t
    end = time.time() - start
    print('done in', end, 'sec')
    print('')
    print(y)
    print(y.shape)


def test_metrics():
    y = torch.tensor([0, 1, 0, 0, 1, 1])
    y_hat = torch.tensor([0, 1, 0, 0, 0, 1])
    
    score = mtr.compute_f1(y, y_hat)
    print(score)
    
    fpr, tpr, _ = mtr.compute_roc(y, y_hat)
    auc = mtr.compute_auc(fpr, tpr)
    print(auc)
    
    objects = np.array(['bottle', 'grid', 'screw', 'tile'])
    auc_scores = np.array([0.99, 0.97, 0.86, 0.95])
    f1_scores = np.array([0.99, 0.98, 0.97, 0.85])
    metric_dict = {
        'auc':auc_scores,
        'f1':f1_scores
    }
    report = mtr.metrics_to_dataframe(metric_dict, objects)
    mtr.export_dataframe(report, saving_path='brutta_copia', name='bho.csv')


def test_pixel_level_metrics():
    gt = get_mvtec_gt_filename_counterpart(
        'dataset/bottle/test/broken_large/000.png', 
        'dataset/bottle/ground_truth/')
    gt = ground_truth(gt)

    gt = transforms.ToTensor()(gt)
    gt = gt.flatten()

    pred_gt = torch.randn((1,256,256))
    pred_gt = normalize(pred_gt)
    mask = heatmap2mask(pred_gt)
    plt.imshow(mask[0])
    plt.savefig('brutta_copia/bho.png', bbox_inches='tight')
    pred_gt = pred_gt.flatten()
    
    fpr, tpr, thresholds = mtr.compute_roc(gt, pred_gt)
    auc = mtr.compute_auc(fpr, tpr)
    print(auc)
    f1 = mtr.compute_f1(gt, mask.flatten())
    print(f1)


def get_container(imsize:tuple, scaling_factor:float):
    center = int(imsize[0]/2)
    container_dim = int(imsize[0]/scaling_factor)
    container_left = int(center-(container_dim/scaling_factor))
    container_top = int(center-(container_dim/scaling_factor))
    container_right = int(center+(container_dim/scaling_factor))
    container_bottom = int(center+(container_dim/scaling_factor))
    width = container_right - container_left
    height = container_bottom - container_top
    return (container_left, container_top, container_right, container_bottom), (width, height)


def find_white_background(img, threshold=0.8):
    """remove images with transparent or white background"""
    imgArr = np.array(img)
    print(imgArr.mean(axis=(0,1)))
    background = np.array([180, 180, 180])
    percent = (imgArr > background).sum() / imgArr.size
    print(percent)
    if percent >= threshold:
        return True
    else:
        return False


def test_centering():
    patch_localization = False
    
    imsize=(256,256)
    img = Image.open('dataset/screw/train/good/000.png').resize(imsize).convert('RGB')
    
    if patch_localization:
        cropper = Container(imsize, 1.5)
        img = img.crop((cropper.left, cropper.top, cropper.right, cropper.bottom))
        img = transforms.RandomCrop((64,64))(img)
        y, mask, coords = generate_patch_new(img, polygoned=True, distortion=False, factor=1)
        y = apply_jittering(y, CPP.jitter_transforms)
        x = paste_patch(img, y, coords, mask)
        
        container = Container((64,64), 1)
        draw = ImageDraw.Draw(x)
        draw.rectangle(
            ((container.center-1, container.center-1), (container.center+1, container.center+1)),
            fill='green')
        
        draw.text(
            (container.left, container.top), 
            text=str(container.left)+', '+str(container.top),
            fill='black')
        draw.text(
            (container.right, container.bottom), 
            text=str(container.right)+', '+str(container.bottom),
            fill='black')
        draw.rectangle(
            ((container.left+1, container.top+1), (container.right-1, container.bottom-1)),
            outline='red')

        plt.imshow(x)
        plt.savefig('patch.png', bbox_inches='tight')
        plt.close()
        
        factor = 1
        y, coords = generate_scar_centered(img,augs=CPP.jitter_transforms, with_padding=False, colorized=True, factor=factor)
        #y = apply_jittering(y, CPP.jitter_transforms)
        x = paste_patch(img, y, coords, y)
        
        
        container = Container((64,64), 1)
        draw = ImageDraw.Draw(x)
        draw.rectangle(
            ((container.center-1, container.center-1), (container.center+1, container.center+1)),
            fill='green')
        
        draw.text(
            (container.left, container.top), 
            text=str(container.left)+', '+str(container.top),
            fill='black')
        draw.text(
            (container.right, container.bottom), 
            text=str(container.right)+', '+str(container.bottom),
            fill='black')
        draw.rectangle(
            ((container.left, container.top), (container.right-1, container.bottom-1)),
            outline='red')
        
        plt.imshow(x)
        plt.savefig('scar.png', bbox_inches='tight')
    else:
        factor = 1.5
        y, mask, coords = generate_patch_new(img, polygoned=True, distortion=False, factor=factor)
        y = apply_jittering(y, CPP.jitter_transforms)
        x = paste_patch(img, y, coords, mask)
        
        
        container = Container(imsize, factor)
        print(container.left, container.top, container.right, container.bottom)
        draw = ImageDraw.Draw(x)
        draw.rectangle(
            ((container.center-1, container.center-1), (container.center+1, container.center+1)),
            fill='green')
        
        draw.text(
            (container.left, container.top), 
            text=str(container.left)+', '+str(container.top),
            fill='black')
        draw.text(
            (container.right, container.bottom), 
            text=str(container.right)+', '+str(container.bottom),
            fill='black')
        
        draw.rectangle(
            ((container.left+1, container.top+1), (container.right-1, container.bottom-1)),
            outline='red')

        plt.imshow(x)
        plt.savefig('patch.png', bbox_inches='tight')
        plt.close()
        
        factor = 2.25
        y, coords = generate_scar_centered(img,augs=CPP.jitter_transforms, with_padding=False, colorized=True, factor=factor)
        #y = apply_jittering(y, CPP.jitter_transforms)
        x = paste_patch(img, y, coords, y)
        
        
        container = Container(imsize, factor)
        draw = ImageDraw.Draw(x)
        draw.rectangle(
            ((container.center-1, container.center-1), (container.center+1, container.center+1)),
            fill='green')
        
        draw.text(
            (container.left, container.top), 
            text=str(container.left)+', '+str(container.top),
            fill='black')
        draw.text(
            (container.right, container.bottom), 
            text=str(container.right)+', '+str(container.bottom),
            fill='black')
        draw.rectangle(
            ((container.left, container.top), (container.right, container.bottom)),
            outline='red')
        
        plt.imshow(x)
        plt.savefig('scar.png', bbox_inches='tight')

test_centering()