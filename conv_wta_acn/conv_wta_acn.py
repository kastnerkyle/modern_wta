from conv_wta_acn_models import ConvWTAVAE, PriorNetwork
import numpy as np
import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import imageio
import time

# from jalexvig
def parse_flags():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='train', choices=['train', 'daydream', 'iterate_noise', 'sparsity'], help='Task to do.')
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

code_len = 20
batch_size = 128
n_neighbors = 5
model_save_path = "conv_acn_models"
dataset_len = 60000
testset_len = 10000

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

model = ConvWTAVAE(code_len, batch_size)
prior = PriorNetwork(code_len, dataset_len, k=n_neighbors)

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

        outputs, u_q, s_q = model(inputs)
        u_p, s_p = prior(u_q)

        #xent = F.binary_cross_entropy(outputs, inputs, reduction='sum')
        r_loss = .5 * ((outputs - inputs) ** 2).sum() / data.shape[0]

        # coding cost
        dkl = torch.sum(s_p - s_q - 0.5 +
                        ((2 * s_q).exp() + (u_q - u_p).pow(2)) /
                        (2 * (2 * s_p).exp()))
        w_dkl = 1.0 * dkl
        loss = r_loss + w_dkl

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        u_q_flat = u_q.view(data.shape[0], code_len)
        prior.update_codebook(batch_idx, u_q_flat.detach())

        if idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}, {:.3f}, {:.3f}'.format(
                idx_epoch, idx * len(data), dataset_len, 100 * (idx * len(data)) / float(dataset_len),
                loss.item(), r_loss.item(), w_dkl.item()))

            torch.save(model.state_dict(), FPATH_VAE)
            torch.save(prior.state_dict(), FPATH_PRIOR)
        steps += 1
        if steps > max_steps:
            break

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        idx_epoch, train_loss / float(steps)))
    return steps

def sparsity(image,
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

        decode_sparsity_act = model.decode_through_spatial_sparsity(next_latent, lifetime=False).clone().detach()
        from IPython import embed; embed(); raise ValueError()

        decode_image = model.decode(next_latent, lifetime=False).clone().detach()
        image = decode_image.reshape(1, 1, 28, 28)
        i += 1
        if i % 200 == 0:
            print("Daydream step {}".format(i))

    if save_gif:
        print("Saving daydream gif...")
        gif_path = 'animation.gif'
        imageio.mimsave(gif_path, mpl_images, duration=0.001)


def iterate_noise(image,
                  model,
                  prior,
                  num_iters=0,
                  save_gif=False):

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    plt.axis('off')

    comb_mpl_images = []

    i = 0
    while i < num_iters or not num_iters:
        image_np = image.clone().detach().cpu().data.numpy().reshape(-1, 1, 28, 28)
        comb_mpl_images.append(image_np)

        if not num_iters:
            plt.axis('off')
            plt.show()

        latent, _ = model.encode(image, spatial=True, lifetime=False)
        # Generate guess of next latent
        #next_latent, _ = prior.encode(latent)
        #next_latent, _ = prior(latent)
        decode_image = model.decode(latent, spatial=True, lifetime=False).clone().detach()
        image = decode_image.reshape(-1, 1, 28, 28)
        i += 1
        if i % 200 == 0:
            print("Iterate noise step {}".format(i))

    mpl_images_dict = {}
    for j in range(len(comb_mpl_images)):
        for i in range(comb_mpl_images[0].shape[0]):
            if i not in mpl_images_dict:
                mpl_images_dict[i] = []
            mpl_images_dict[i].append((comb_mpl_images[j][i, 0] * 255.).astype("uint8"))

    if save_gif:
        print("Saving iterate mp4...")
        sav_path = 'animation_iter.mp4'
        imageio.mimsave(sav_path, mpl_images_dict[0], fps=10)


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

        decode_image = model.decode(next_latent, lifetime=False).clone().detach()
        image = decode_image.reshape(1, 1, 28, 28)
        i += 1
        if i % 200 == 0:
            print("Daydream step {}".format(i))

    if save_gif:
        print("Saving daydream mp4...")
        gif_path = 'animation.mp4'
        imageio.mimsave(gif_path, mpl_images, fps=1000)

if args.task == 'train':
    params = list(model.parameters()) + list(prior.parameters())
    optimizer = optim.Adam(params)

    total_steps = 0
    start_time = time.time()
    n_epochs = 100
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
elif args.task == "sparsity":
    test_loader = torch.utils.data.DataLoader(
        IndexedDataset(datasets.MNIST, path='../data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    idx, (data, label, batch_idx) = next(enumerate(test_loader))

    model.load_state_dict(torch.load(FPATH_VAE))
    prior.load_state_dict(torch.load(FPATH_PRIOR))

    img_ = data[0].reshape(1, 1, 28, 28).to(device)

    sparsity(img_, model, prior, num_iters=10000, save_gif=True)
elif args.task == "iterate_noise":
    model.load_state_dict(torch.load(FPATH_VAE))
    prior.load_state_dict(torch.load(FPATH_PRIOR))
    random_state = np.random.RandomState(1234)

    img_ = random_state.rand(100 * 1 * 28 * 28).reshape(100, 1, 28, 28).astype("float32")
    img_ = torch.tensor(img_).to(device)
    iterate_noise(img_, model, prior, num_iters=500, save_gif=True)

elif args.task == "daydream":
    test_loader = torch.utils.data.DataLoader(
        IndexedDataset(datasets.MNIST, path='../data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    idx, (data, label, batch_idx) = next(enumerate(test_loader))

    model.load_state_dict(torch.load(FPATH_VAE))
    prior.load_state_dict(torch.load(FPATH_PRIOR))

    img_ = data[0].reshape(1, 1, 28, 28).to(device)

    daydream(img_, model, prior, num_iters=10000, save_gif=True)

"""
# might need to modify imagemagick policy xml if errors
# https://stackoverflow.com/questions/31407010/cache-resources-exhausted-imagemagick
sed -i '/disable ghostscript format types/,+6d' /etc/ImageMagick-6/policy.xml //this one is just to solve convertion from .tiff to pdf, you may need it some day
sed -i -E 's/name="memory" value=".+"/name="memory" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
sed -i -E 's/name="map" value=".+"/name="map" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
sed -i -E 's/name="area" value=".+"/name="area" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
sed -i -E 's/name="disk" value=".+"/name="disk" value="8GiB"/g' /etc/ImageMagick-6/policy.xml
"""
