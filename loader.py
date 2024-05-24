import numpy as np
import random
import torch

#from .brats2021data import BrainDataset
#from .brats_configs import get_default_configs
from brats2021data import BrainDataset, CLRDataset
from brats_configs import get_default_configs


def seed_worker(worker_id):
    np.random.seed(worker_id)
    random.seed(0)

g = torch.Generator()
g.manual_seed(0)


def get_data_loader(dataset, config, split_set, generator = True, clr_data = None, no_img=None, shuffle=True):
    if dataset == "mnist":
        loader = get_data_loader_mnist(config.data.path, config.sampling.batch_size, 
                                    split_set=split_set, which_label=config.classifier.label)
    elif dataset == "brats":
        loader = get_data_loader_brats(config.data.path, config.fa.batch_size, split_set=split_set,
                                            sequence_translation = config.data.sequence_translation,
                                            healthy_data_percentage = config.data.healthy_data_percentage, 
                                            clr_data= clr_data, no_img=no_img, shuffle=shuffle)
    else:
        raise Exception("Dataset does exit")
    
    return get_generator_from_loader(loader) if generator else loader

def get_data_loader_mnist(path, batch_size, split_set: str = 'train', which_label: str = "class"):
    assert split_set in ["train", "val", "test"]
    default_kwargs = {"shuffle": True, "num_workers": 1, "drop_last": True, "batch_size": batch_size}
    dataset = MNIST_dataset(root_dir=path, train=split_set != "test")

    if split_set != "test":
        val_ratio = 0.1
        split = torch.utils.data.random_split(dataset,
                                              [int(len(dataset) * (1 - val_ratio)), int(len(dataset) * val_ratio)],
                                              generator=torch.Generator().manual_seed(42))
        dataset = split[0] if split_set == "train" else split[1]

    return torch.utils.data.DataLoader(dataset, **default_kwargs)


def get_data_loader_brats(path, batch_size, split_set: str = 'train',
                             sequence_translation : bool = False, 
                             healthy_data_percentage : float = 1.0, clr_data = False,
                             no_img = None, shuffle = True,):

    assert split_set in ["train", "val", "test"]
    default_kwargs = {"drop_last": True, "batch_size": batch_size, "pin_memory" : True, "num_workers": 8,
                    "prefetch_factor" : 8, "worker_init_fn" : seed_worker, "generator": g,}
    default_kwargs["shuffle"] = shuffle
    default_kwargs["num_workers"] = 1
    if clr_data:
        dataset = CLRDataset(path, split = split_set, pad=(8,8,8,8), resize_factor=None, 
                               no_img=no_img, shuffle = shuffle
                            )
        print(f"dataset length: {len(dataset)}")
        return torch.utils.data.DataLoader(dataset, **default_kwargs),len(dataset),dataset
    else:
        if split_set == "test":    
            #get all data
            dataset = BrainDataset(path, n_tumour_patients = None,
                               n_healthy_patients = 0, split = split_set, pad=(8,8,8,8), resize_factor=None, #, centercrop=20,
                               sequence_translation = sequence_translation,
                               )
            print(f"dataset length: {len(dataset)}")
            return torch.utils.data.DataLoader(dataset, **default_kwargs),len(dataset)
        ###if sp
        else:
            default_kwargs["num_workers"] = 8
            dataset_healthy = BrainDataset(path, split = split_set,
                n_tumour_patients=0, n_healthy_patients=None, pad=(8,8,8,8), resize_factor=None, #,  centercrop=20
                skip_healthy_s_in_tumour=True,skip_tumour_s_in_healthy=True,
                )
            dataset_unhealthy = BrainDataset(path, split = split_set,
                n_tumour_patients=None, n_healthy_patients=0, pad=(8,8,8,8), resize_factor=None, #,centercrop=20, 
                skip_healthy_s_in_tumour=True,skip_tumour_s_in_healthy=True,
                )
        # float [0,1]; 1 for training using full data; None for training with healthy data only
        #healthy_data_percentage = None

            if healthy_data_percentage is not None:
                healthy_size = int(len(dataset_healthy)*healthy_data_percentage)
                unhealthy_size = len(dataset_unhealthy)
                total_size = healthy_size + unhealthy_size
                samples_weight = torch.cat([torch.ones(healthy_size)   * total_size / healthy_size,torch.ones(unhealthy_size) * total_size / unhealthy_size]
                                        ).double()
                sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
                default_kwargs["sampler"] = sampler
                default_kwargs.pop('shuffle', None) # shuffle and sampler are mutually exclusive

                dataset = torch.utils.data.dataset.ConcatDataset([torch.utils.data.Subset(dataset_healthy, range(0, int(len(dataset_healthy)*healthy_data_percentage))),
                                 dataset_unhealthy])
            else:
                dataset = dataset_healthy

    print(f"dataset length: {len(dataset)}")
    return torch.utils.data.DataLoader(dataset, **default_kwargs),len(dataset)


def get_generator_from_loader(loader):
    while True:
        yield from loader

if __name__ == "__main__":
    from tqdm import tqdm 
    config = get_default_configs()
    dataset = "brats"
    split_set = "test"
    data_loader = get_data_loader(dataset, config, split_set, generator = True)
    data_loader  = iter(data_loader)
    #pbar = tqdm(data_loader)
    for batch in data_loader:
        print(batch['image'].shape,batch['gt'].shape(),batch['y'])
    '''
        if batch['gt'].sum()>0:
            print("break:",batch['gt'].sum())
            break
    '''
    '''
    for (idx,data) in enumerate(data_loader):
        print(idx) 
        #data = next(data_loader)
        print("output:",data['image'].shape)#,torch.max(data['image']),torch.min(data['image']))
    '''