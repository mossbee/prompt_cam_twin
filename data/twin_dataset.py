import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from itertools import combinations
import os


class TwinPairDataset(Dataset):
    """
    Dataset for twin face verification task.
    Generates pairs of images with same/different person labels.
    """
    
    def __init__(self, dataset_info_path, twin_pairs_path, transform=None, 
                 positive_ratio=0.5, hard_negative_ratio=0.7):
        """
        Args:
            dataset_info_path: Path to dataset info JSON (person_id -> image_paths)
            twin_pairs_path: Path to twin pairs JSON (list of twin pairs)
            transform: Image transforms
            positive_ratio: Ratio of positive pairs (same person)
            hard_negative_ratio: Ratio of hard negatives (twins) among negative pairs
        """
        
        # Load dataset information
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        with open(twin_pairs_path, 'r') as f:
            self.twin_pairs = json.load(f)
        
        self.transform = transform or self._default_transform()
        self.positive_ratio = positive_ratio
        self.hard_negative_ratio = hard_negative_ratio
        
        # Create person ID to index mapping
        self.person_ids = list(self.dataset_info.keys())
        self.person_to_idx = {pid: idx for idx, pid in enumerate(self.person_ids)}
        
        # Create twin pairs mapping for hard negatives
        self.twin_map = {}
        for pair in self.twin_pairs:
            self.twin_map[pair[0]] = pair[1]
            self.twin_map[pair[1]] = pair[0]
        
        # Generate all possible pairs for the epoch
        self.pairs = self._generate_pairs()
    
    def _default_transform(self):
        """Default transform for face images - minimal since images are already 224x224"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _generate_pairs(self):
        """Generate balanced positive and negative pairs"""
        pairs = []
        
        # Calculate number of pairs needed
        total_images = sum(len(paths) for paths in self.dataset_info.values())
        num_pairs = total_images // 2  # Approximate number of pairs per epoch
        
        num_positive = int(num_pairs * self.positive_ratio)
        num_negative = num_pairs - num_positive
        num_hard_negative = int(num_negative * self.hard_negative_ratio)
        num_easy_negative = num_negative - num_hard_negative
        
        # Generate positive pairs (same person)
        for _ in range(num_positive):
            person_id = random.choice(self.person_ids)
            if len(self.dataset_info[person_id]) >= 2:
                img1, img2 = random.sample(self.dataset_info[person_id], 2)
                pairs.append({
                    'img1_path': img1,
                    'img2_path': img2,
                    'person1_id': person_id,
                    'person2_id': person_id,
                    'label': 1,  # Same person
                    'is_twin_pair': False
                })
        
        # Generate hard negative pairs (twin siblings)
        for _ in range(num_hard_negative):
            if self.twin_pairs:
                twin_pair = random.choice(self.twin_pairs)
                person1_id, person2_id = twin_pair
                
                if (person1_id in self.dataset_info and 
                    person2_id in self.dataset_info):
                    
                    img1 = random.choice(self.dataset_info[person1_id])
                    img2 = random.choice(self.dataset_info[person2_id])
                    
                    pairs.append({
                        'img1_path': img1,
                        'img2_path': img2,
                        'person1_id': person1_id,
                        'person2_id': person2_id,
                        'label': 0,  # Different person
                        'is_twin_pair': True
                    })
        
        # Generate easy negative pairs (random different people)
        for _ in range(num_easy_negative):
            person1_id, person2_id = random.sample(self.person_ids, 2)
            
            # Ensure they are not twins
            if (person2_id != self.twin_map.get(person1_id) and
                person1_id != self.twin_map.get(person2_id)):
                
                img1 = random.choice(self.dataset_info[person1_id])
                img2 = random.choice(self.dataset_info[person2_id])
                
                pairs.append({
                    'img1_path': img1,
                    'img2_path': img2,
                    'person1_id': person1_id,
                    'person2_id': person2_id,
                    'label': 0,  # Different person
                    'is_twin_pair': False
                })
        
        # Shuffle pairs
        random.shuffle(pairs)
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load images
        img1 = Image.open(pair['img1_path']).convert('RGB')
        img2 = Image.open(pair['img2_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # Get person indices for prompts
        person1_idx = self.person_to_idx[pair['person1_id']]
        person2_idx = self.person_to_idx[pair['person2_id']]
        
        return {
            'img1': img1,
            'img2': img2,
            'label': torch.tensor(pair['label'], dtype=torch.long),
            'person1_idx': torch.tensor(person1_idx, dtype=torch.long),
            'person2_idx': torch.tensor(person2_idx, dtype=torch.long),
            'is_twin_pair': pair['is_twin_pair'],
            'person1_id': pair['person1_id'],
            'person2_id': pair['person2_id']
        }
    
    def regenerate_pairs(self):
        """Regenerate pairs for new epoch"""
        self.pairs = self._generate_pairs()


class TwinIdentityDataset(Dataset):
    """
    Dataset for identity classification (used for Stage 1 training).
    Standard single-image classification with person labels.
    """
    
    def __init__(self, dataset_info_path, transform=None):
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        self.transform = transform or self._default_transform()
        
        # Create flat list of (image_path, person_id) pairs
        self.samples = []
        self.person_ids = list(self.dataset_info.keys())
        self.person_to_idx = {pid: idx for idx, pid in enumerate(self.person_ids)}
        
        for person_id, image_paths in self.dataset_info.items():
            for img_path in image_paths:
                self.samples.append((img_path, person_id))
    
    def _default_transform(self):
        """Default transform for identity dataset - minimal since images are already 224x224"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, person_id = self.samples[idx]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        person_idx = self.person_to_idx[person_id]
        
        return {
            'image': img,
            'label': torch.tensor(person_idx, dtype=torch.long),
            'person_id': person_id
        }
    
    @property
    def num_classes(self):
        return len(self.person_ids)
