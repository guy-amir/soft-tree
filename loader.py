import torch
import torchvision
import torchvision.transforms as transforms


def class_numerator(y,n):
    idx = (y == 0)
    for i in range(1,n):
        idx = idx + (y == i)
    return idx

def mnist_loader(path = './data', nClasses = 2, batch_size = 64, train = True):
        # Transformations
    # RC   = transforms.RandomCrop(32, padding=4)
    RHF  = transforms.RandomHorizontalFlip()
    RVF  = transforms.RandomVerticalFlip()
    NRM  = transforms.Normalize((0.1307,), (0.3081,)) #NRM  = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT  = transforms.ToTensor()
    TPIL = transforms.ToPILImage()

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([RHF, TT, NRM]) #TPIL,RC, 
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([TT, NRM])

    
    
    if train is True:
        dataset = torchvision.datasets.MNIST(path, train=train, download=True, transform = transform_with_aug)
        # idx = class_numerator(dataset.train_labels,nClasses)
        # idx = (dataset.train_labels==0)+(dataset.train_labels==1)+(dataset.train_labels==2)+(dataset.train_labels==3)
        # dataset.train_labels = dataset.train_labels[idx]
        # dataset.train_data = dataset.train_data[idx]
    else:
        dataset = torchvision.datasets.MNIST('./data', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))
        idx = class_numerator(dataset.test_labels,nClasses)
        # dataset.test_labels = dataset.test_labels[idx]
        # dataset.test_data = dataset.test_data[idx]

    m_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return m_loader