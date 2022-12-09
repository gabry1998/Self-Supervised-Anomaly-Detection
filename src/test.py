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


def read_test():
    x = get_image_filenames('dataset/bottle/train/good/')
    print(x[0])


def duplicate_test():
    dataset_dir = '/home/ubuntu/TesiAnomalyDetection/dataset/'
    subject = 'toothbrush'

    x = get_image_filenames(dataset_dir+subject+'/train/good/')
    y = get_image_filenames(dataset_dir+subject+'/test/good/')

    print(x.shape)
    print(y.shape)

    x = duplicate_filenames(x)
    y = duplicate_filenames(y)

    print(x.shape)
    print(y.shape)


def gt_test():
    dataset_dir = 'dataset/'
    subject = 'bottle'
    
    y = get_mvtec_anomaly_classes(dataset_dir+subject+'/test/')
    print(y)
    
    x = get_mvtec_test_images(dataset_dir+subject+'/test/')
    print(x)
    y_hat1 = get_mvtec_gt_filename_counterpart(x[0], dataset_dir+subject+'/ground_truth/')
    y_hat2 = get_mvtec_gt_filename_counterpart(x[-1], dataset_dir+subject+'/ground_truth/')
    y_hat3 = get_mvtec_gt_filename_counterpart(
        'dataset/bottle/test/good/000.png', 
        dataset_dir+subject+'/ground_truth/')
    print(y_hat1)
    print(y_hat2)
    print(y_hat3)
    
    y_hat4 = get_mvtec_gt_filename_counterpart(
        'dataset/bottle/train/good/000.png', 
        dataset_dir+subject+'/ground_truth/')
    
    print(y_hat4)


def tqdm_test():
    
    for i1 in tqdm(range(5),leave=False):
        print('tante cose belle')
        imsize=(256,256)
        batch_size = 64
        
        print('image size:', imsize)
        print('batch size:', batch_size)
    
        for i2 in tqdm(range(300), leave=False):    
            time.sleep(0.01)
        
        os.system('clear')
     
      
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

#test_GDE_image_level()

def test_1d_to_2d():
    x = torch.randn((3249,128))

    print(x.shape)
    print(x)
    x1 = np.resize(np.array(x), (int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0]))))
    print(x1.shape)
    print(x1)

#test_1d_to_2d()

def test_gaussian():
    x = [
        [0,0,0,0],
        [0,0,1,0],
        [0,0,1,0],
        [0,0,0,0]
        ]
    x = torch.tensor(x)[None, None, :]
    print(x)
    gs = GaussianSmooth(kernel_size=16, stride=1, device='cpu')
    x1 = gs.upsample(x)
    min_max_scaler = preprocessing.MinMaxScaler()
    x1 = x1.squeeze().squeeze()
    x1 = min_max_scaler.fit_transform(x1)
    
    
    print(x1.shape)
    print(x1)
    plt.imshow(x1)
    plt.savefig('bho.png')


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
#test_patch_level()


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
#test_reshape()


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


def other_tests():
    gt = get_mvtec_gt_filename_counterpart(
        'dataset/bottle/test/broken_large/000.png', 
        'dataset/bottle/ground_truth/')
    gt = ground_truth(gt)

    gt = transforms.ToTensor()(gt)[None, :]
    gt_patches = extract_mask_patches(gt, 32,4)
    patches_labels = torch.tensor(gt2label(gt_patches))

    patches_scores = torch.randn(57*57)
    dim = int(np.sqrt(patches_scores.shape[0]))
    predictions = normalize(patches_scores)

    print(patches_labels.shape)
    print(patches_scores.shape)

    fpr, tpr, thresholds = mtr.compute_roc(patches_labels, patches_scores)
    auc = mtr.compute_auc(fpr, tpr)
    print(auc)  
#test_pixel_level_metrics()


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


class Container:
    def __init__(self, imsize:tuple, scaling_factor:float) -> None:
        self.center = int(imsize[0]/2)
        self.dim = int(imsize[0]/scaling_factor)
        self.left = int(self.center-(self.dim/scaling_factor))
        self.top = int(self.center-(self.dim/scaling_factor))
        self.right = int(self.center+(self.dim/scaling_factor))
        self.bottom = int(self.center+(self.dim/scaling_factor))
        self.width = self.right - self.left
        self.height = self.bottom - self.top


def generate_patch_new(
        image, 
        area_ratio:tuple=(0.02, 0.15), 
        aspect_ratio:tuple=((0.3, 1),(1, 3.3)),
        polygoned=False,
        distortion=False):

    img_area = image.size[0] * image.size[1]
    patch_area = random.uniform(area_ratio[0], area_ratio[1]) * img_area
    patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
    patch_w  = int(np.sqrt(patch_area*patch_aspect))
    patch_h = int(np.sqrt(patch_area/patch_aspect))
    org_w, org_h = image.size
    container = Container(image.size, scaling_factor=2)

    # parte da tagliare
    patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
    patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
    # coordinate
    if container.right-patch_w > container.left:
        paste_left = random.randint(container.left, container.right-patch_w)
    else:
        paste_left = container.left
    if container.bottom-patch_h > container.top:
        paste_top = random.randint(container.top, container.bottom-patch_h)
    else:
        paste_top = container.top
    mask = None
    
    if polygoned:
        mask = Image.new('RGBA', (patch_w, patch_h), (255,255,255,0)) 
        draw = ImageDraw.Draw(mask)
        
        points = get_random_points(
            mask.size[0],
            mask.size[1],
            5,
            15)
        draw.polygon(points, fill='black')
        
    if distortion:
        deformer = Deformer(imsize=image.size, points=(patch_left, patch_top, patch_right, patch_bottom))
        deformed_image = ImageOps.deform(image, deformer)
        cropped_patch = deformed_image.crop((patch_left, patch_top, patch_right, patch_bottom))
    else:
        cropped_patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
    return cropped_patch, mask, (paste_left, paste_top)



def generate_scar_centered(image, w_range=(2,16), h_range=(10,25), augs=None, with_padding=False):
    img_w, img_h = image.size
    right = 1
    left = 1
    top = 1
    bottom = 1
    container = Container(image.size, scaling_factor=2.5)
    scar_w = random.randint(w_range[0], w_range[1])
    scar_h = random.randint(h_range[0], h_range[1])
    new_width = scar_w + right + left
    new_height = scar_h + top + bottom
    patch_left, patch_top = random.randint(0, img_w - scar_w), random.randint(0, img_h - scar_h)
    patch_right, patch_bottom = patch_left + scar_w, patch_top + scar_h
    
    scar = image.crop((patch_left, patch_top, patch_right, patch_bottom))
    if with_padding:
        scar_with_pad = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
        scar = apply_jittering(scar, augs)
        scar_with_pad.paste(scar, (left, top))
    else:
        scar_with_pad = Image.new(image.mode, (scar_w, scar_h), (255, 255, 255))
        scar = apply_jittering(scar, augs)
        scar_with_pad.paste(scar, (0, 0))
    scar = scar_with_pad.convert('RGBA')
    angle = random.randint(-45, 45)
    scar = scar.rotate(angle, expand=True)

    #posizione casuale della sezione
    if container.right-scar_w > container.left:
        left = random.randint(container.left, container.right-scar_w)
    else:
        left = container.left
    if container.bottom-scar_h > container.top:
        top = random.randint(container.top, container.bottom-scar_h)
    else:
        top = container.top
    #left, top = random.randint(0, img_w - scar_w), random.randint(0, img_h - scar_h)
    return scar, (left, top)



def test_centering():
    imsize=(256,256)
    img = Image.open('dataset/screw/train/good/000.png').resize(imsize).convert('RGB')
    
    y, mask, coords = generate_patch_new(img, polygoned=True)
    y = apply_jittering(y, CPP.jitter_transforms)
    x = paste_patch(img, y, coords, mask)
    
    factor = 1.75
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
    plt.savefig('patch.png', bbox_inches='tight')
    
    
    y, coords = generate_scar_centered(img,augs=CPP.jitter_transforms, with_padding=False)
    #y = apply_jittering(y, CPP.jitter_transforms)
    x = paste_patch(img, y, coords, y)
    
    factor = 2.5
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
    