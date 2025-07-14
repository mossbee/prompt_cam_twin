import torch
from data.dataset.cub import get_cub
from data.dataset.dog import get_dog
from data.dataset.pet import get_pet
from data.dataset.car import get_car
from data.dataset.birds_525 import get_birds_525
from data.twin_dataset import TwinPairDataset, TwinIdentityDataset
import torchvision.transforms as transforms
import os


def get_dataset(data, params, logger):
    dataset_train, dataset_val, dataset_test = None, None, None

    if data.startswith("cub"):
        logger.info("Loading CUB data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for cub)...")
            dataset_train = get_cub(params, 'trainval_combined')
            dataset_test = get_cub(params, 'test')
        else:
            raise NotImplementedError 
    elif data.startswith("dog"):
        logger.info("Loading Standford Dogs data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for dog)...")
            dataset_train = get_dog(params, 'trainval_combined')
            dataset_test = get_dog(params, 'test')
        else:
            raise NotImplementedError
    elif data.startswith("pet"):
        logger.info("Loading Oxford Pet data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for pet)...")
            dataset_train = get_pet(params, 'trainval_combined')
            dataset_test = get_pet(params, 'test')
        else:
            raise NotImplementedError
    elif data.startswith("car"):
        logger.info("Loading Stanford Car data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for car)...")
            dataset_train = get_car(params, 'trainval_combined')
            dataset_test = get_car(params, 'test')
        else:
            raise NotImplementedError
    elif data.startswith("birds_525"):
        logger.info("Loading Birds 525 data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for birds_525)...")
            dataset_train = get_birds_525(params, 'trainval_combined')
            dataset_test = get_birds_525(params, 'test')
        else:
            raise NotImplementedError
    elif data.startswith("twin"):
        logger.info("Loading Twin Face data ...")
        return get_twin_datasets(params, logger)
    else:
        raise Exception("Dataset '{}' not supported".format(data))
    return dataset_train, dataset_val, dataset_test


def get_loader(params, logger):
    if 'test_data' in params:
        dataset_train, dataset_val, dataset_test = get_dataset(params.test_data, params, logger)
    else:
        dataset_train, dataset_val, dataset_test = get_dataset(params.data, params, logger)

    if isinstance(dataset_train, list):
        train_loader, val_loader, test_loader = [], [], []
        for i in range(len(dataset_train)):
            tmp_train, tmp_val, tmp_test = gen_loader(params, dataset_train[i], dataset_val[i], None)
            train_loader.append(tmp_train)
            val_loader.append(tmp_val)
            test_loader.append(tmp_test)
    else:
        train_loader, val_loader, test_loader = gen_loader(params, dataset_train, dataset_val, dataset_test)

    logger.info("Finish setup loaders")
    return train_loader, val_loader, test_loader


def gen_loader(params, dataset_train, dataset_val, dataset_test):
    train_loader, val_loader, test_loader = None, None, None
    if params.debug:
        num_workers = 1
    else:
        num_workers = 4
    if dataset_train is not None:
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    if dataset_val is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    if dataset_test is not None:
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True

        )
    return train_loader, val_loader, test_loader


def get_twin_datasets(params, logger):
    """Get twin face datasets for verification task"""
    
    # Default transform for face images - minimal since they're already 224x224
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset paths
    data_dir = getattr(params, 'data_dir', 'data')
    train_info_path = os.path.join(data_dir, 'train_dataset_infor.json')
    train_pairs_path = os.path.join(data_dir, 'train_twin_pairs.json')
    test_info_path = os.path.join(data_dir, 'test_dataset_infor.json')
    test_pairs_path = os.path.join(data_dir, 'test_twin_pairs.json')
    
    # Check if this is for Stage 1 (identity classification) or Stage 2 (verification)
    stage1_training = getattr(params, 'stage1_training', False)
    
    if stage1_training:
        logger.info("Loading datasets for Stage 1: Identity Classification")
        # For Stage 1, use identity classification datasets
        dataset_train = TwinIdentityDataset(train_info_path, transform=train_transform)
        dataset_val = TwinIdentityDataset(test_info_path, transform=val_transform)
        dataset_test = None
    else:
        logger.info("Loading datasets for Stage 2: Verification")
        # For Stage 2, use pair datasets
        dataset_train = TwinPairDataset(
            train_info_path, train_pairs_path, 
            transform=train_transform,
            positive_ratio=0.5,
            hard_negative_ratio=0.7
        )
        dataset_val = TwinPairDataset(
            test_info_path, test_pairs_path,
            transform=val_transform,
            positive_ratio=0.5,
            hard_negative_ratio=0.8
        )
        # For evaluation, we also provide the test dataset
        dataset_test = TwinPairDataset(
            test_info_path, test_pairs_path,
            transform=val_transform,
            positive_ratio=0.5,
            hard_negative_ratio=0.8
        )
    
    logger.info(f"Train dataset size: {len(dataset_train)}")
    logger.info(f"Val dataset size: {len(dataset_val)}")
    if dataset_test is not None:
        logger.info(f"Test dataset size: {len(dataset_test)}")
    
    return dataset_train, dataset_val, dataset_test
