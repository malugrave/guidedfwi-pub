import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from scipy.signal import filtfilt
from accelerate import Accelerator
from torch.utils.data import TensorDataset
from ema_pytorch import EMA
from tqdm.auto import tqdm
from pathlib import Path
from timm.layers import to_2tuple
from multiprocessing import cpu_count
from torchvision import transforms as T, utils

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, exists, has_int_squareroot, cycle, divisible_by, num_to_groups
from denoising_diffusion_pytorch.version import __version__

def unnormalize_from_zero_to_one(x, xmin, xmax):
    return (x)*(xmax - xmin) + xmin

def normalize_to_zero_to_one(x, xmin, xmax):
    return (x - xmin)/(xmax - xmin)

def normalize_to_minusone_to_one(x, xmin, xmax):
    return 2*(x - xmin)/(xmax - xmin)-1

def unnormalize_from_minusone_to_one(x, xmin, xmax):
    return (x+1)/2*(xmax - xmin) + xmin 

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def tensor_to_image(data, min_, max_, path, nrow, class_samples, label):
    h_f, w_f = to_2tuple(nrow)
    fig, axs = plt.subplots(h_f, w_f, figsize=(nrow*2,nrow*2), sharex = True, sharey = True)
    for i in range(h_f):
        for j in range(w_f):
            ax = axs[i,j]
            im = ax.imshow(unnormalize_from_zero_to_one(data[i*h_f+j].cpu().numpy(), min_, max_).T, cmap='jet', vmin=min_, vmax=max_)
            ax.set_title(f'{label[class_samples[i*h_f + j]]}')
    fig.subplots_adjust(hspace=-0.3) 
    plt.colorbar(im, ax=axs, shrink=0.5)
    plt.savefig(path, bbox_inches='tight', dpi=500)
    
def split_data_to_size(data, size, stride):
    b, c, h, w = data.shape
    h_split, w_split =  (h-size[0]+1)//stride[0], (w-size[1]+1)//stride[1]
    b_new, c_new, h_new, w_new = b, h_split * w_split, size[0],size[1]
    data_split = torch.zeros(b_new,c_new,h_new,w_new)
    for i in range(h_split):
        for j in range(w_split):
            data_split[:,i*w_split+j,:,:] = data[:,0,i*stride[0]:i*stride[0]+h_new,j*stride[1]:j*stride[1]+w_new]
    return data_split.reshape(-1,1,h_new,w_new)

# class Trainer(object):
#     def __init__(
#         self, train_x, train_y,
#         diffusion_model,
#         *,
#         train_batch_size = 16,
#         gradient_accumulate_every = 1,
#         train_lr = 1e-4,
#         train_num_steps = 100000,
#         ema_update_every = 10,
#         ema_decay = 0.995,
#         adam_betas = (0.9, 0.99),
#         save_and_sample_every = 1000,
#         num_samples = 25,
#         results_folder = './results',
#         amp = True,
#         classes = None,
#         save_and_sample=True
#     ):
#         super().__init__()

#         self.accelerator = Accelerator()
#         self.accelerator.native_amp = amp
#         self.model = diffusion_model
#         self.num_samples = num_samples
#         self.save_and_sample_every = save_and_sample_every
#         self.batch_size = train_batch_size
#         self.gradient_accumulate_every = gradient_accumulate_every
#         self.train_num_steps = train_num_steps
#         self.image_size = diffusion_model.image_size

#         self.class_dict = classes
#         self.ds = TensorDataset(train_x,train_y)
        
#         dl = torch.utils.data.DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, num_workers=32, pin_memory=True)

#         self.opt = torch.optim.Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

#         if self.accelerator.is_main_process:
#             self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

#         self.results_folder = Path(results_folder)
#         self.results_folder.mkdir(exist_ok = True)
#         self.step = 0

#         self.model, self.opt, dl = self.accelerator.prepare(self.model, self.opt, dl)
#         self.dl = cycle(dl)
#         self.save_and_sample = save_and_sample
        
#     def save(self, milestone=None):
#         data = {
#             'step': self.step,
#             'model': self.accelerator.get_state_dict(self.model),
#             'opt': self.opt.state_dict(),
#             'ema': self.ema.state_dict(),
#             'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
#         }
#         if milestone is not None:
#             torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
#         else:
#             torch.save(data, str(self.results_folder / 'model.pt'))

#     def load(self, milestone=None):
#         accelerator = self.accelerator
#         device = accelerator.device

#         if milestone is not None:
#             data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
#         else:
#             data = torch.load(str(self.results_folder / f'model.pt'), map_location=device)

#         model = self.accelerator.unwrap_model(self.model)
#         model.load_state_dict(data['model'])

#         self.step = data['step']
#         self.opt.load_state_dict(data['opt'])
#         if self.accelerator.is_main_process:
#             self.ema.load_state_dict(data["ema"])

#         if exists(self.accelerator.scaler) and exists(data['scaler']):
#             self.accelerator.scaler.load_state_dict(data['scaler'])

#     def train(self):
#         accelerator = self.accelerator
#         device = accelerator.device
#         if accelerator.is_main_process:
#             sw = SummaryWriter(str(self.results_folder / f'tensorboard'))

#         with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

#             while self.step < self.train_num_steps:

#                 total_loss = 0.

#                 for _ in range(self.gradient_accumulate_every):
#                     data, label = next(self.dl)
#                     data, label = data.to(device), label.to(device)

#                     with self.accelerator.autocast():
#                         loss = self.model(data, classes=label)
#                         loss = loss / self.gradient_accumulate_every
#                         total_loss += loss.item()

#                     self.accelerator.backward(loss)

#                 pbar.set_description(f'Step: {self.step}; loss: {total_loss:.4f}\n')

#                 accelerator.wait_for_everyone()

#                 self.opt.step()
#                 self.opt.zero_grad()

#                 accelerator.wait_for_everyone()

#                 self.step += 1
#                 if accelerator.is_main_process:
#                     sw.add_scalar('train/loss', total_loss, self.step)
#                     self.ema.to(device)
#                     self.ema.update()

#                     if self.step != 0 and self.step % self.save_and_sample_every == 0:
#                         self.ema.ema_model.eval()

#                         with torch.no_grad():
#                             milestone = self.step // self.save_and_sample_every
#                             classes_sample = torch.randint(0,len(self.class_dict),(self.num_samples,)).to(device) 
#                             # print(self.ema.ema_model.sample(classes=classes_sample, cond_scale=5.).shape)
#                             all_images = self.ema.ema_model.sample(classes=classes_sample, cond_scale=5.)

#                         # all_images = torch.cat(all_images_list, dim = 1)
#                         tensor_to_image(all_images[:,0], 0, 1, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)), class_samples=classes_sample, label=self.class_dict)
                        
#                         if self.save_and_sample:
#                             self.save(milestone)
#                         else:
#                             self.save()

#                 pbar.update(1)

#         accelerator.print('training complete')

#     def cond_fn(self, x, y, mask, alpha_t, weight_factor=1.0):
#         assert y is not None
#         with torch.enable_grad():
#             x_in = x.detach().requires_grad_(True)
#             rec_loss = F.mse_loss(x_in * mask, y, reduction='none').sum()
#             return torch.autograd.grad(rec_loss, x_in)[0] * weight_factor * alpha_t
    
#     def test(self, cond_rec=False, weight_factor=1.0, target=None, target_mask=None, classes_samples=None):
#         accelerator = self.accelerator
#         if accelerator.is_main_process:
#             self.ema.to(accelerator.device)
#             self.ema.ema_model.eval()
            
#             with torch.no_grad():
#                 if classes_samples is None:
#                     classes_sample = torch.randint(0,len(self.class_dict),(self.num_samples,)).to(accelerator.device) 
#                 else:
#                     classes_sample = classes_samples.to(accelerator.device)
#                 if cond_rec:
#                     batch_size, image_size, channels = classes_sample.shape[0], self.image_size, self.ema.ema_model.channels
#                     img = torch.randn((batch_size, channels, image_size[0], image_size[1]), device = accelerator.device)
#                     x_start = None
#                     for t in tqdm(reversed(range(0, self.ema.ema_model.num_timesteps)), desc='sampling with reconstruction guided loop time step', total=self.ema.ema_model.num_timesteps):
#                         batched_times = torch.full((img.shape[0],), t, device = img.device, dtype = torch.long)
#                         model_mean, posterior_variance, posterior_log_variance_clipped, x_start = self.ema.ema_model.p_mean_variance(x = img, t = batched_times, classes = classes_sample, cond_scale = 5., clip_denoised = True)
#                         model_mean = model_mean - self.cond_fn(img, normalize_to_neg_one_to_one(target.to(accelerator.device)), target_mask.to(accelerator.device), posterior_variance, weight_factor=weight_factor)
#                         noise = torch.randn_like(img) if t > 0 else 0. # no noise if t == 0
#                         img = model_mean + (0.5 * posterior_log_variance_clipped).exp() * noise
#                     all_images = unnormalize_to_zero_to_one(img).squeeze(1)
#                 else:
#                     all_images = self.ema.ema_model.sample(classes=classes_sample, cond_scale=5.)
#                     # all_images = torch.cat(all_images_list, dim = 0)
#                 tensor_to_image(all_images[:,0], 0, 1, str(self.results_folder / f'test-sample.png'), nrow = int(math.sqrt(self.num_samples)), class_samples=classes_sample, label=self.class_dict)
#                 tensor_to_image(target[:,0], 0, 1, str(self.results_folder / f'test-target.png'), nrow = int(math.sqrt(self.num_samples)), class_samples=classes_sample, label=self.class_dict)

class Trainer:
    def __init__(
        self,
        diffusion_model,
        train_data,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results-noamp-openfwi',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = TensorDataset(train_data)
        
        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = torch.utils.data.DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 0)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = torch.optim.Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        print("Accelerator device:", self.accelerator.device)
        print("Model device:", next(self.model.parameters()).device)
        if torch.cuda.is_available():
            print("CUDA allocated:", torch.cuda.memory_allocated() / 1024**2, "MiB")
            print("CUDA reserved:", torch.cuda.memory_reserved() / 1024**2, "MiB")

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)[0].to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')