from conv_wta_models import ConvWTA
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

# from jalexvig
def parse_flags():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='train', choices=['train', 'daydream', 'iterate_noise', 'sparsity'], help='Task to do.')
    args = parser.parse_args()
    return args

def seed_everything(seed=4142, workers=False):
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            rank_zero_warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                rank_zero_warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    print(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed

seed_everything(1257)

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
batch_size = 100
n_neighbors = 5
model_save_path = "conv_acn_models"
dataset_len = 60000
testset_len = 10000
learning_rate = 1E-3

train_itr = dataset_itr(batch_size, subset_type="train", seed=123)
# start coroutine
next(train_itr);

test_itr = dataset_itr(batch_size, subset_type="test", seed=1234)
# start coroutine
next(test_itr);

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

FPATH_VAE = os.path.join(model_save_path, 'conv_vae.pth')

args = parse_flags()

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.normal_(0.0, 0.001)
    #if isinstance(m, nn.ConvTranspose2d):
    #    m.weight.data.normal_(0.0, 0.001)
    #    m.bias.data.normal_(0.0, 0.001)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device {}".format(device))
model = ConvWTA(code_len, batch_size).to(device)
model.apply(weights_init)

def train(model,
          optimizer,
          idx_epoch):

    model.train()
    train_loss = 0
    steps = 0
    max_steps = int(dataset_len / float(batch_size))
    for idx, (data, label, batch_idx) in enumerate(train_itr):
        inputs = torch.tensor(data.reshape(data.shape[0], 1, 28, 28).astype("float32") / 255.).to(device)
        optimizer.zero_grad()

        outputs, latent = model(inputs)
        #err = F.binary_cross_entropy(outputs, inputs, reduction='sum')
        err = .5 * ((outputs - inputs) ** 2).sum() / data.shape[0]
        loss = err

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                idx_epoch, idx * len(data), dataset_len, 100 * (idx * len(data)) / float(dataset_len),
                loss.item()))

            torch.save(model.state_dict(), FPATH_VAE)
        steps += 1
        if steps > max_steps:
            break

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        idx_epoch, train_loss / (int(steps))))
    return steps

def sparsity(image,
             model,
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
                  num_iters=0,
                  save_gif=False):

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    plt.axis('off')

    comb_mpl_images = []

    i = 0
    diff = 1E10
    thresh = 1E-3
    with torch.no_grad():
        while i < num_iters: # and diff > thresh:
            image_np = image.clone().detach().cpu().data.numpy().reshape(-1, 1, 28, 28)
            comb_mpl_images.append(image_np)

            latent = model.encode(image, spatial=True, lifetime=False)
            # Generate guess of next latent
            #next_latent, _ = prior.encode(latent)
            #next_latent, _ = prior(latent)
            decode_image = model.decode(latent, spatial=True, lifetime=False).clone().detach()
            new_image = decode_image.reshape(-1, 1, 28, 28)
            diff = np.max(((new_image - image) ** 2).cpu().data.numpy())
            # renorm
            a = new_image
            a = (a - a.min()) / (a.max() - a.min())
            a = (a * 255.).long()
            image = a.float() / 255.
            i += 1
            if i % 200 == 0:
                print("Iterate noise step {}".format(i))

    mpl_images_dict = {}
    for j in range(len(comb_mpl_images)):
        for i in range(comb_mpl_images[0].shape[0]):
            if i not in mpl_images_dict:
                mpl_images_dict[i] = []
            mpl_images_dict[i].append((comb_mpl_images[j][i, 0] * 255.).astype("uint8"))

    final_plot_images = [None for k in range(len(mpl_images_dict[0]))]
    for i in range(10):
        for k in range(len(mpl_images_dict[0])):
            if final_plot_images[k] is None:
                final_plot_images[k] = mpl_images_dict[i][k]
            else:
                # i is row
                cur_sz = final_plot_images[k].shape
                final_plot_images[k] = np.concatenate((final_plot_images[k], mpl_images_dict[i][k]), axis=1)

    if save_gif:
        print("Saving iterate mp4...")
        sav_path = 'animation_iter.mp4'
        imageio.mimsave(sav_path, final_plot_images, fps=max(1, int(num_iters / 10)))


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
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    total_steps = 0
    start_time = time.time()
    n_epochs = 10
    for i in range(n_epochs):
        this_steps = train(model, optimizer, i)
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
    random_state = np.random.RandomState(1234)

    img_ = random_state.rand(100 * 1 * 28 * 28).reshape(100, 1, 28, 28).astype("float32")
    img_ = torch.tensor(img_).to(device)
    iterate_noise(img_, model, num_iters=5000, save_gif=True)

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
