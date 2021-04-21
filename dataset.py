import cv2

class ImageWoofDataset(Dataset):
    
    def __init__(self, images_paths, labels, transforms):
        assert len(images_paths) == len(labels)
        self.images_paths = images_paths
        self.labels = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx])
        image = self.transforms(image=image)["image"]
        label = self.labels[idx]
        return (image, label)
