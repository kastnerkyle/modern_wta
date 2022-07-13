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
    parser.add_argument('--task', default='train', choices=['train', 'daydream', 'neighbors', 'neighbors_dataset'], help='Task to do.')
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

def train(model,
          prior,
          optimizer,
          idx_epoch):

    model.train()
    train_loss = 0
    steps = 0
    max_steps = int(dataset_len / float(batch_size))
    for idx, (data, label, batch_idx) in enumerate(train_itr):
        inputs = torch.tensor(data.reshape(data.shape[0], 1, 28, 28).astype("float32") / 255.).to(device)
        optimizer.zero_grad()

        outputs, u_q, s_q, vq_e_z, vq_q_z, vq_indices = model(inputs)
        u_p, s_p = prior(u_q)

        xent = F.binary_cross_entropy_with_logits(outputs, inputs, reduction='sum')

        # coding cost
        dkl = torch.sum(s_p - s_q - 0.5 +
                        ((2 * s_q).exp() + (u_q - u_p).pow(2)) /
                        (2 * (2 * s_p).exp()))
        loss = xent + dkl

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        u_q_flat = u_q.view(data.shape[0], code_len)
        prior.update_codebook(batch_idx, u_q_flat.detach())

        if idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                idx_epoch, idx * len(data), dataset_len, 100 * (idx * len(data)) / float(dataset_len),
                loss.item() / dataset_len))

            torch.save(model.state_dict(), FPATH_VAE)
            torch.save(prior.state_dict(), FPATH_PRIOR)
        steps += 1
        if steps > max_steps:
            break

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        idx_epoch, train_loss / dataset_len))
    return steps


def daydream(image,
             model,
             prior,
             num_iters=0,
             save_gif=False):

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    plt.axis('off')

    mpl_images = []

    i = 0
    while i < num_iters or not num_iters:
        image_np = image.clone().detach().cpu().data.numpy().reshape(28, 28)
        mpl_images.append((image_np * 255.).astype("uint8"))

        if not num_iters:
            plt.axis('off')
            plt.show()

        latent, _ = model.encode(image)

        # Generate guess of next latent
        #next_latent, _ = prior.encode(latent)
        next_latent, _ = prior(latent)

        decode_image = model.decode(next_latent).clone().detach()
        image = decode_image.reshape(1, 1, 28, 28)
        i += 1
        if i % 200 == 0:
            print("Daydream step {}".format(i))

    if save_gif:
        print("Saving daydream gif...")
        gif_path = 'animation.gif'
        imageio.mimsave(gif_path, mpl_images, duration=0.001)

def neighbors(image_batch,
              model,
              prior):

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    model.eval()

    acn_z_flat, acn_mu_flat, acn_log_std_flat, vq_e_z, vq_q_z, vq_indices = model.encode(image_batch)

    decode_images = model.decode(vq_q_z).detach().cpu().data.numpy()

    dists, indices = prior.kneighbors(acn_mu_flat, n_neighbors=5)
    neighbor_data_batches = [train_data_np[indices[i]].reshape(-1, 1, 28, 28) for i in range(indices.shape[0])]
    neighbor_data_labels = [train_data_labels_np[indices[i]] for i in range(indices.shape[0])]

    #next_latent, _ = prior(latent)

    # input, its reconstruction and 5 neighbor reconstructions
    f, axarr = plt.subplots(2 * image_batch.shape[0], len(neighbor_data_batches) + 1)
    for _i in range(image_batch.shape[0]):
        # linear out -> sigmoid
        decode_image = decode_images[_i]
        in_image = image_batch[_i].reshape(1, 1, 28, 28)[0, 0]
        out_image = sigmoid_np(decode_image.reshape(1, 1, 28, 28)[0, 0])
        cmap = "gray"
        axarr[2 * _i, 0].imshow(in_image, cmap=cmap)
        axarr[2 * _i + 1, 0].imshow(out_image, cmap=cmap)

        axarr[2 * _i, 0].axis("off")
        axarr[2 * _i, 1].axis("off")

        axarr[2 * _i + 1, 0].axis("off")
        axarr[2 * _i + 1, 1].axis("off")

        n_acn_z_flat, n_acn_mu_flat, n_acn_log_std_flat, n_vq_e_z, n_vq_q_z, n_vq_indices = model.encode(torch.tensor(neighbor_data_batches[_i].astype("float32") / 255.))
        neighbor_decode_images = sigmoid_np(model.decode(n_vq_q_z).cpu().data.numpy().reshape(-1, 1, 28, 28))

        for _j in range(neighbor_data_batches[_i].shape[0]):
            axarr[2 * _i, _j + 1].imshow(neighbor_data_batches[_i][_j][0], cmap=cmap)
            axarr[2 * _i + 1, _j + 1].imshow(neighbor_decode_images[_j, 0], cmap=cmap)
            axarr[2 * _i, _j + 1].axis("off")
            axarr[2 * _i + 1, _j + 1].axis("off")

    plt.savefig("final_arr.png")

    #plt.imshow(in_image, cmap="gray")
    #plt.savefig("in.png")

    #plt.imshow(out_image, cmap="gray")
    #plt.savefig("out.png")

    from IPython import embed; embed(); raise ValueError()


if args.task == 'train':
    params = list(model.parameters()) + list(prior.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    total_steps = 0
    start_time = time.time()
    for i in range(n_epochs):
        this_steps = train(model, prior, optimizer, i)
        total_steps += this_steps
    end_time = time.time()
    run_time = end_time - start_time
    print("Total training time {} seconds".format(run_time))
    print("Dataset size in examples {}".format(dataset_len))
    print("Number of epochs {}".format(n_epochs))
    print("Total train steps taken in batches {}".format(total_steps))
    print("Average time per batch {} seconds".format(run_time / float(total_steps)))
elif args.task == "daydream":
    test_loader = torch.utils.data.DataLoader(
        IndexedDataset(datasets.MNIST, path='../data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    idx, (data, label, batch_idx) = next(enumerate(test_loader))

    model.load_state_dict(torch.load(FPATH_VAE))
    prior.load_state_dict(torch.load(FPATH_PRIOR))

    img_ = data[0].reshape(1, 1, 28, 28).to(device)

    daydream(img_, model, prior, num_iters=10000, save_gif=True)
elif args.task == "neighbors":
    seed_everything(1234)
    test_loader = torch.utils.data.DataLoader(
        IndexedDataset(datasets.MNIST, path='../data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    idx, (data, label, batch_idx) = next(enumerate(test_loader))

    model.load_state_dict(torch.load(FPATH_VAE, map_location=device))
    prior.load_state_dict(torch.load(FPATH_PRIOR, map_location=device))

    image_batch = data[:5].to(device)
    neighbors(image_batch, model, prior)
elif args.task == "neighbors_dataset":
    seed_everything(1234)
    model.load_state_dict(torch.load(FPATH_VAE, map_location=device))
    prior.load_state_dict(torch.load(FPATH_PRIOR, map_location=device))

    model.eval()
    prior.eval()

    process_batch_size = 1000
    from collections import OrderedDict
    train_neighbor_matches = OrderedDict()
    test_neighbor_matches = OrderedDict()

    train_vq_entries = OrderedDict()

    def process_(image_batch_np, global_index, which="train"):
        acn_z_flat, acn_mu_flat, acn_log_std_flat, vq_e_z, vq_q_z, vq_indices = model.encode(torch.tensor(image_batch_np.astype("float32") / 255.).to(device))
        # want 5 neighbors not including self - do top 6 to ensure this
        dists, indices = prior.kneighbors(acn_mu_flat, n_neighbors=5 + 1)
        dists_np = dists.cpu().data.numpy()
        indices_np = indices.cpu().data.numpy()

        vq_q_z_np = vq_q_z.cpu().data.numpy()
        vq_indices_np = vq_indices.cpu().data.numpy()

        for _l in range(image_batch_np.shape[0]):
            # get top 5 but ignore self - don't need to ignore self for test data
            if which == "train":
                _s = indices_np[_l] != _l
            else:
                _s = indices_np[_l] > -1
            assert len(_s) >= 5
            inds_sub = indices_np[_l, _s][:5]
            dists_sub = dists_np[_l, _s][:5]
            if which == "train":
                train_neighbor_matches[global_index + _l] = (inds_sub, dists_sub)
                # 256x16x16 q_z, 16x16 indices
                train_vq_entries[global_index + _l] = (vq_q_z_np[_l], vq_indices_np[_l].argmax(axis=0))
            else:
                test_neighbor_matches[global_index + _l] = (inds_sub, dists_sub)


    _i = 0
    while True:
        print("train pre {}".format(_i))
        if _i > train_data_np.shape[0]:
            break
        image_batch_np = train_data_np[_i:_i + process_batch_size].reshape(-1, 1, 28, 28)
        if image_batch_np.shape[0] < 1:
            break
        process_(image_batch_np, _i, which="train")
        _i = _i + process_batch_size

    _i = 0
    while True:
        print("test pre {}".format(_i))
        if _i > test_data_np.shape[0]:
            break
        image_batch_np = test_data_np[_i:_i + process_batch_size].reshape(-1, 1, 28, 28)
        if image_batch_np.shape[0] < 1:
            break
        process_(image_batch_np, _i, which="test")
        _i = _i + process_batch_size

    # we have the indices form of the dataset, as well as the neighbors for everything
    # just need to make a big numpy array and write it out
    train_vq_indices = np.concatenate([train_vq_entries[el][1][None] for el in range(len(train_vq_entries))], axis=0)

    train_vq_match_indices = np.concatenate([train_neighbor_matches[el][0][None] for el in range(len(train_neighbor_matches))], axis=0)
    train_vq_match_dists = np.concatenate([train_neighbor_matches[el][1][None] for el in range(len(train_neighbor_matches))], axis=0)

    test_vq_match_indices = np.concatenate([test_neighbor_matches[el][0][None] for el in range(len(test_neighbor_matches))], axis=0)
    test_vq_match_dists = np.concatenate([test_neighbor_matches[el][1][None] for el in range(len(test_neighbor_matches))], axis=0)
    np.savez("mnist_vq.npz", train_vq_indices=train_vq_indices,
                             train_vq_match_indices=train_vq_match_indices,
                             train_vq_match_dists=train_vq_match_dists,
                             test_vq_match_indices=test_vq_match_indices,
                             test_vq_match_dists=test_vq_match_dists)
    print("wrote out neighbors dataset to mnist_vq.npz")

    # commented code is an example of how to reconstruct from indices
    # code_tmp = model.vq_indices_to_codes(torch.tensor(train_vq_entries[0][1][None]).to(device))
    # out_code = sigmoid_np(model.decode(code_tmp).detach().cpu().data.numpy())


"""
# might need to modify imagemagick policy xml if errors
# https://stackoverflow.com/questions/31407010/cache-resources-exhausted-imagemagick
sed -i '/disable ghostscript format types/,+6d' /etc/ImageMagick-6/policy.xml //this one is just to solve convertion from .tiff to pdf, you may need it some day
sed -i -E 's/name="memory" value=".+"/name="memory" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
sed -i -E 's/name="map" value=".+"/name="map" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
sed -i -E 's/name="area" value=".+"/name="area" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
sed -i -E 's/name="disk" value=".+"/name="disk" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
"""
