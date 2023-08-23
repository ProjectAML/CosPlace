
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T

import test
import util
import parser
import commons
import cosface_loss
import sphereface_loss
import arcface_loss
import center_loss
import augmentations
from cosplace_model import cosplace_network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
from datasets.target_dataset import TargetDataset, DomainAdaptationDataLoader
from itertools import chain

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(args.output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

#### Model
model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim, domain_adaptation = args.domain_adaptation)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()

#### Optimizer
criterion = torch.nn.CrossEntropyLoss()

if args.domain_adaptation:
    target_dataset=TargetDataset(args.target_dataset_folder)
    criterion_da = torch.nn.CrossEntropyLoss()

model_parameters=chain(model.backbone.parameters(), model.aggregation.parameters(), model.discriminator.parameters() if args.domain_adaptation else [])
model_optimizer = torch.optim.Adam(model_parameters, lr=args.lr)

#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]

if args.pseudo_target_folder:
        pseudo_groups = [TrainDataset(args, args.pseudo_target_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                        current_group=n, min_images_per_class=args.min_images_per_class, 
                        pseudo_target=True
                        ) for n in range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
use_gpu = torch.cuda.is_available()

if args.loss == "cosface": 
    logging.info("cosface loss is used")
    classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
elif args.loss == "sphereface":
    logging.info("spherefaCe loss is used")
    classifiers = [sphereface_loss.SphereFaceLoss(args.fc_output_dim, len(group)) for group in groups]
elif args.loss == "arcface":
    logging.info("arcface loss is used")
    classifiers = [arcface_loss.ArcFaceLoss(args.fc_output_dim, len(group)) for group in groups]
elif args.loss == "center":
    logging.info("center loss is used")
    for group in groups:
        calssifiers = [center_loss.CenterLoss(args.fc_output_dim, len(group), use_gpu)]
    
else:
    logging.debug("No valid loss, please try again typing 'cosface', 'sphereface' or 'arcface' or 'cosface_center'")
    exit

classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries",
                      positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")

#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, args.output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0

#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")


if args.augmentation_device == "cuda":
    augmentation_applied=True
    if args.augmentation_type=="colorjitter":
        augmentation_type=augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                    contrast=args.contrast,
                                                    saturation=args.saturation,
                                                    hue=args.hue)
    elif args.augmentation_type=="brightness":
        logging.info("data augmentation brightness")
        augmentation_type = augmentations.DeviceAgnosticBrightness(args.reduce_brightness)
    elif args.augmentation_type=="contrast":
        logging.info("data augmentation contrast")
        augmentation_type = augmentations.DeviceAgnosticContrast(args.increase_contrast)
    elif args.augmentation_type=="saturation":
        logging.info("data augmentation saturation")
        augmentation_type = augmentations.DeviceAgnosticSaturation(args.increase_saturation)
    elif args.augmentation_type=="brightness_contrast_saturation":
        logging.info("data augmentation brightness_contrast_saturation")
        augmentation_type = augmentations.DeviceAgnosticBrightnessContrastSaturation(args.reduce_brightness, args.increase_contrast, args.increase_saturation)
    elif args.augmentation_type=="none":
        augmentation_applied=False
    else:
        logging.debug("No valid augmentation type, please try again typing 'colorjitter', 'brightness' , 'contrast' , 'saturation' or 'none'")
        exit
    
if augmentation_applied:
    gpu_augmentation = T.Compose([
      augmentation_type,
      augmentations.DeviceAgnosticRandomResizedCrop([224, 224],
                                                    scale=[1-args.random_resized_crop, 1]),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    gpu_augmentation = T.Compose([
      augmentations.DeviceAgnosticRandomResizedCrop([224, 224],
                                                    scale=[1-args.random_resized_crop, 1]),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

for epoch_num in range(start_epoch_num, args.epochs_num):
    
    #### Train
    epoch_start_time = datetime.now()
    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % args.groups_num
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)
    
    batch_size = args.batch_size
    if args.pseudo_target_folder:
      batch_size = int(batch_size / 2)

    dataloader = commons.InfiniteDataLoader(groups[current_group_num], pseudo_dataset = pseudo_groups[current_group_num] if args.pseudo_target_folder else None, num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True)
    
    if args.domain_adaptation:
        dataloader_da=DomainAdaptationDataLoader(groups[current_group_num], target_dataset, pseudo_dataset = pseudo_groups[current_group_num] if args.pseudo_target_folder else None,
                                                pseudo=args.pseudo_da,num_workers=args.num_workers,
                                                batch_size=16, shuffle=True,
                                                pin_memory=(args.device == "cuda"), drop_last=True)

    dataloader_iterator = iter(dataloader)
    model = model.train()
    
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
        images, targets, _= next(dataloader_iterator)
        images, targets = images.to(args.device), targets.to(args.device)
        
        if args.domain_adaptation:
            images_da, targets_da = next(dataloader_da)
            images_da, targets_da = images_da.to(args.device), targets_da.to(args.device)

        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)
        
        model_optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()
        
        if not args.use_amp16:
            descriptors = model(images)
            output = classifiers[current_group_num](descriptors, targets)
            loss = criterion(output, targets)
            loss.backward()

            loss_da=0
            if args.domain_adaptation:
                output_da=model(images_da, grl=True)
                loss_da=criterion(output_da,targets_da)
                loss_da=loss_da*args.loss_weight_grl
                loss_da.backward()
                loss_da=loss_da.item()
                del output_da, images_da
            
            epoch_losses = np.append(epoch_losses, loss.item()+loss_da)
            del loss, output, images, loss_da
            model_optimizer.step()
            classifiers_optimizers[current_group_num].step()
        else:  # Use AMP 16
            with torch.cuda.amp.autocast():
                descriptors = model(images)
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)
            scaler.scale(loss).backward()

            loss_da=0
            if args.domain_adaptation:
                with torch.cuda.amp.autocast():
                    output_da=model(images_da, grl=True)
                    loss_da=criterion(output_da,targets_da)
                    loss_da=loss_da*args.loss_weight_grl
                scaler.scale(loss_da).backward()
                del output_da, images_da

            epoch_losses = np.append(epoch_losses, loss.item()+loss_da.item())
            del loss, output, images, loss_da
            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()
    
    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
    
    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {epoch_losses.mean():.4f}")
    
    #### Evaluation
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, args.output_folder)


logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set v1
best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model, args.num_preds_to_save)
logging.info(f"{test_ds}: {recalls_str}")

logging.info("Experiment finished (without any errors)")
