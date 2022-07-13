from conv_acn_vq_models import ConvACNVQVAE, PriorNetwork
import numpy as np
import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import imageio
import time

import random

def seed_everything(seed=1234):
    random.seed(seed)
    tseed = random.randint(1, 1E6)
    tcseed = random.randint(1, 1E6)
    npseed = random.randint(1, 1E6)
    ospyseed = random.randint(1, 1E6)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(tseed)
    torch.cuda.manual_seed_all(tcseed)
    np.random.seed(npseed)
    # cannot set these inside colab :(
    #torch.use_deterministic_algorithms(True)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ['PYTHONHASHSEED'] = str(ospyseed)

def sigmoid_np(x):
    return np.exp(-np.logaddexp(0, -x))

# from jalexvig
def parse_flags():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='reconstruct', choices=['reconstruct', 'encode', 'encode_rotate'], help='Task to do.')
    args = parser.parse_args()
    return args

class IndexedDataset(Dataset):
    def __init__(self, dataset_function, path, train=True, download=True, transform=transforms.ToTensor()):
        """ class to provide indexes into the data -- needed for ACN prior
        """
        self.indexed_dataset = dataset_function(path,
                             download=download,
                             train=train,
                             transform=transform)

    def __getitem__(self, index):
        data, target = self.indexed_dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.indexed_dataset)

# as an alternative to IndexedDataset from the acn/ and even_faster_acn/, use our own data iterator
# mnist from 
# https://github.com/lucastheis/deepbelief/blob/master/data/mnist.npz
# this also lets us load other mnist like datasets easily
#d = np.load("mnist.npz")
#print(d.files)
#['test', 'test_labels', 'train', 'train_labels']

mnist_path = "../mnist.npz"
train_data_f = np.load(mnist_path)
train_data_np = train_data_f[train_data_f.files[2]].T
train_data_labels_np = train_data_f[train_data_f.files[3]].T

test_data_f = np.load(mnist_path)
test_data_np = test_data_f[test_data_f.files[0]].T
test_data_labels_np = test_data_f[test_data_f.files[1]].T
#kuzu_train = "/content/drive/MyDrive/kuzushiji_MNIST/k49-train-imgs.npz"
#kuzu_test = "/content/drive/MyDrive/kuzushiji_MNIST/k49-test-imgs.npz"
def dataset_itr(batch_size, subset_type="train", seed=1234):
    """
    Coroutine of experience replay.
    Provide a new experience by calling send, which in turn yields
    a random batch of previous replay experiences.
    """
    if subset_type == "train":
        data_f = np.load(mnist_path)
        data_np = data_f[data_f.files[2]].T
        label_np = data_f[data_f.files[3]].T
    elif subset_type == "test":
        data_f = np.load(mnist_path)
        data_np = data_f[data_f.files[0]].T
        label_np = data_f[data_f.files[1]].T
    else:
        raise ValueError("Unknown subset_type {}".format(subset_type))
    max_sz = len(data_np)
    random_state = np.random.RandomState(seed)
    while True:
        inds = np.arange(max_sz)
        batch_inds = random_state.choice(inds, size=batch_size, replace=True)
        batch = data_np[batch_inds]
        batch_label = label_np[batch_inds]
        # return in a similar format as pytorch
        yield batch, batch_label, batch_inds

code_len = 32
model_hidden_size = 256
prior_hidden_size = 512
batch_size = 128

n_neighbors = 5
model_save_path = "conv_acn_vq_models"
dataset_len = 60000
testset_len = 10000
learning_rate = 1E-4

n_epochs = 100

train_itr = dataset_itr(batch_size, subset_type="train", seed=123)
# start coroutine
next(train_itr);

test_itr = dataset_itr(batch_size, subset_type="test", seed=1234)
# start coroutine
next(test_itr);

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

FPATH_VAE = os.path.join(model_save_path, 'conv_vae.pth')
FPATH_PRIOR = os.path.join(model_save_path, 'prior.pth')

args = parse_flags()

model = ConvACNVQVAE(model_hidden_size, code_len, batch_size)
prior = PriorNetwork(prior_hidden_size, code_len, dataset_len, k=n_neighbors)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device {}".format(device))
model = model.to(device)
prior = prior.to(device)

if args.task == "reconstruct":
    seed_everything(1234)
    model.load_state_dict(torch.load(FPATH_VAE, map_location=device))
    prior.load_state_dict(torch.load(FPATH_PRIOR, map_location=device))

    model.eval()
    prior.eval()

    d = np.load("sunmask_vq_pred.npz")
    context0_pred = d["context0_pred"]
    context1_pred = d["context1_pred"]
    context2_pred = d["context2_pred"]
    final_pred = d["final_pred"]
    ground_truth_target = d["ground_truth_target"]

    each_pred = []
    each_pred.append(("context0_pred", context0_pred))
    each_pred.append(("context1_pred", context1_pred))
    each_pred.append(("context2_pred", context2_pred))
    each_pred.append(("final_pred", final_pred))
    each_pred.append(("ground_truth_target", ground_truth_target))

    import matplotlib.pyplot as plt

    for el in each_pred:
        code_tmp = model.vq_indices_to_codes(torch.tensor(el[1]).long().to(device))
        out_code = model.decode(code_tmp).detach().cpu().data.numpy()
        out_im = sigmoid_np(out_code)

        for _v in range(context0_pred.shape[0]):
            plt.imshow(out_im[_v, 0], cmap="gray")
            plt.savefig("{}_{}.png".format(_v, el[0]))
            plt.close()
    print("finished reconstruction, wrote out sampled images")
elif "encode" in args.task:
    seed_everything(1234)
    model.load_state_dict(torch.load(FPATH_VAE, map_location=device))
    prior.load_state_dict(torch.load(FPATH_PRIOR, map_location=device))

    model.eval()
    prior.eval()
    idx, (all_data, all_label, all_batch_idx) = next(enumerate(test_itr))
    each_label_match = []
    for _i in range(10):
        is_label = np.where(all_label == _i)[0]
        each_label_match.append(is_label[0])

    # get a sub-batch that has 1 of every digit category
    # then go get neighbors to these
    data = all_data[each_label_match]
    label = all_label[each_label_match]
    batch_idx = all_batch_idx[each_label_match]

    # first, get neighbors using unrotated data
    image_batch = torch.tensor(data.reshape(data.shape[0], 1, 28, 28).astype("float32") / 255.).to(device)
    acn_z_flat, acn_mu_flat, acn_log_std_flat, vq_e_z, vq_q_z, vq_indices = model.encode(image_batch)

    dists, indices = prior.kneighbors(acn_mu_flat, n_neighbors=5)

    neighbor_data_batches = [train_data_np[indices[i]].reshape(-1, 1, 28, 28) for i in range(indices.shape[0])]
    neighbor_data_labels = [train_data_labels_np[indices[i]] for i in range(indices.shape[0])]

    # encode the original data but rotated
    image_batch = torch.tensor(data.reshape(data.shape[0], 1, 28, 28).astype("float32") / 255.).to(device)

    if args.task == "encode_rotate":
        image_batch = image_batch.permute(0, 1, 3, 2)
    acn_z_flat, acn_mu_flat, acn_log_std_flat, vq_e_z, vq_q_z, vq_indices = model.encode(image_batch)

    data_batches = data.reshape(data.shape[0], 1, 28, 28)

    #data_batches = data_batches.transpose(0, 1, 3, 2)
    data_labels = label
    data_batches_quantized = vq_indices.argmax(axis=1).cpu().data.numpy()[:, None]

    # encode rotated neighbors
    neighbor_data_batches_quantized = []
    for _i in range(len(neighbor_data_batches)):
        data = neighbor_data_batches[_i]
        # rotate them all before encoding
        image_batch = torch.tensor(data.reshape(data.shape[0], 1, 28, 28).astype("float32") / 255.).to(device)
        # rotate 90
        if args.task == "encode_rotate":
            image_batch = image_batch.permute(0, 1, 3, 2)
        acn_z_flat, acn_mu_flat, acn_log_std_flat, vq_e_z, vq_q_z, vq_indices = model.encode(image_batch)
        neighbor_data_batches_quantized.append(vq_indices.argmax(axis=1).cpu().data.numpy())

    # rotate neighbor data to match
    if args.task == "encode_rotate":
        neighbor_data_batches = np.concatenate([nb[None].transpose(0, 1, 2, 4, 3) for nb in neighbor_data_batches], axis=0)
    else:
        neighbor_data_batches = np.concatenate([nb[None] for nb in neighbor_data_batches], axis=0)
    neighbor_data_labels = np.concatenate([nl[None] for nl in neighbor_data_labels], axis=0)
    neighbor_data_batches_quantized = np.concatenate([nq[None] for nq in neighbor_data_batches_quantized], axis=0)

    """
    # inspect rotated base data rec
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    code_tmp = model.vq_indices_to_codes(torch.tensor(data_batches_quantized).long().to(device))
    out_code = model.decode(code_tmp).detach().cpu().data.numpy()
    out_im = sigmoid_np(out_code)

    # lets inspect some rotations
    for _v in range(10):
        plt.imshow(out_im[_v, 0].transpose(1, 0), cmap="gray")
        plt.savefig("tmp{}.png".format(_v))
        plt.close()
        plt.imshow(data_batches[_v, 0].transpose(1, 0), cmap="gray")
        plt.savefig("tmp{}gt.png".format(_v))
        plt.close()
    """

    # then save to a file...
    np.savez("encoded_data.npz",
             data_batches=data_batches,
             data_labels=data_labels,
             data_batches_quantized=data_batches_quantized,
             neighbor_data_batches=neighbor_data_batches,
             neighbor_data_labels=neighbor_data_labels,
             neighbor_data_batches_quantized=neighbor_data_batches_quantized)
    print("finished encoding data, saved to encoded_data.npz")

    """
    for el in each_pred:
        code_tmp = model.vq_indices_to_codes(torch.tensor(el[1][None]).long().to(device))
        out_code = model.decode(code_tmp).detach().cpu().data.numpy()
        out_im = sigmoid_np(out_code)

        plt.imshow(out_im[0, 0], cmap="gray")
        plt.savefig("{}.png".format(el[0]))
        plt.close()
    print("finished sampling, wrote out sampled images")
    from IPython import embed; embed(); raise ValueError()
    """


"""
# might need to modify imagemagick policy xml if errors
# https://stackoverflow.com/questions/31407010/cache-resources-exhausted-imagemagick
sed -i '/disable ghostscript format types/,+6d' /etc/ImageMagick-6/policy.xml //this one is just to solve convertion from .tiff to pdf, you may need it some day
sed -i -E 's/name="memory" value=".+"/name="memory" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
sed -i -E 's/name="map" value=".+"/name="map" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
sed -i -E 's/name="area" value=".+"/name="area" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
sed -i -E 's/name="disk" value=".+"/name="disk" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
"""
