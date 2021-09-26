import os
import subprocess
import sys
import random
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from siren_pytorch import SirenNet, SirenWrapper
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch_optimizer import DiffGrad, AdamP
import numpy as np

from PIL import Image
from imageio import imread, mimsave
import torchvision.transforms as T

import imageio


from tqdm import trange, tqdm

from clip import load, tokenize


# Helpers
#global phi = (1 + math.sqrt(5))/2

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


"""
A = cos with period 1 epoch, starts at 0 
B = cos w/ period 1 epoch, starts at 1
C = cos w/ period 3 epochs, starts at 0 
D = cos w/ period 3 epochs, starts at 1 
E = dna cos, period 3, starts at 0, max 1, mid-min= .0.364, mid-max=0.636
F = dna cos, period 3, starts at 0.385, decreasing, min 0, max 0.77
G = cos with period 2 epochs, starts at 1
 
https://www.desmos.com/calculator/b94pluhvxq
"""    
def flow_fx(x, selector):
    if selector is None:
        fx=1
    elif selector == 'A':
        fx=-0.5*math.cos(2*math.pi*x) + 0.5
    elif selector == 'B':
        fx=0.5*math.cos(2*math.pi*x) + 0.5
    elif selector == 'C':
        fx=-0.5*math.cos((2/3)*math.pi*x) + 0.5
    elif selector == 'D':
        fx=0.5*math.cos((2/3)*math.pi*x) + 0.5
    elif selector == 'E':
        a=-0.5*math.cos(2*math.pi*x)+0.5
        d=0.5*math.cos((2/3)*math.pi*x)+0.5
        fx=0.5*(a-d) + 0.5
    elif selector == 'F':
        a=-0.5*math.cos(2*math.pi*x)+0.5
        c=-0.5*math.cos((2/3)*math.pi*x)+0.5
        fx=-0.5*(a-c) + 0.385
    elif selector == 'G':
        fx=0.5*math.cos(math.pi*x) + 0.5
    elif selector == 'fade':
        fx=1-x
    else:
        fx=1    
    #fx=-0.125*math.cos(2*math.pi*( math.cos(x)-0.5*x )) + 0.875
    #fx=0.125*math.cos( 2*math.pi*math.cos(x) ) + 0.875
    return fx

def interpolate(image, size):
    return F.interpolate(image, (size, size), mode='bicubic', align_corners=True)


def rand_cutout(image, size, center_bias=False, center_focus=2):
    height = image.shape[-2]
    width = image.shape[-1]
    min_offset = 0
    h_max_offset = height - size
    w_max_offset = width - size
    if center_bias:
        # sample around image center
        center = w_max_offset / 2
        std = center / center_focus
        offset_w = int(random.gauss(mu=center, sigma=std))
        offset_h = int(random.gauss(mu=center, sigma=std))
        # resample uniformly if over boundaries
        offset_w = random.randint(min_offset, w_max_offset) if (offset_w > w_max_offset or offset_w < min_offset) else offset_w
        offset_h = random.randint(min_offset, h_max_offset) if (offset_h > h_max_offset or offset_h < min_offset) else offset_h
    else:
        offset_w = random.randint(min_offset, w_max_offset)
        offset_h = random.randint(min_offset, h_max_offset)
    cutout = image[:, :, offset_h:offset_h + size, offset_w:offset_w + size]
    return cutout


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


def create_clip_img_transform(image_width):
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    transform = T.Compose([
                    #T.ToPILImage(),
                    T.Resize(256),
                    T.CenterCrop((image_width, image_width)),
                    T.ToTensor(),
                    T.Normalize(mean=clip_mean, std=clip_std)
            ])
    return transform


def open_folder(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return

    cmd_list = None
    if sys.platform == 'darwin':
        cmd_list = ['open', '--', path]
    elif sys.platform == 'linux2' or sys.platform == 'linux':
        cmd_list = ['xdg-open', path]
    elif sys.platform in ['win32', 'win64']:
        cmd_list = ['explorer', path.replace('/', '\\')]
    if cmd_list is None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


def norm_siren_output(img):
    return ((img + 1) * 0.5).clamp(0.0, 1.0)


def create_text_path(context_length, text=None, img=None, encoding=None, separator=None):
    if text is not None:
        if separator is not None and separator in text:
            #Reduces filename to first epoch text
            text = text[:text.index(separator, )]
        input_name = text.replace(" ", "_")[:context_length]
    elif img is not None:
        if isinstance(img, str):
            input_name = "".join(img.replace(" ", "_").split(".")[:-1])
        else:
            input_name = "PIL_img"
    else:
        input_name = "your_encoding"
    return input_name


#Returns a value truncated to a specific number of decimal places.
def truncate(number, decimals=0):
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor



class DeepDaze(nn.Module):
    def __init__(
            self,
            clip_perceptor,
            clip_norm,
            input_res,
            total_batches,
            batch_size,
            num_layers=8,
            image_width=224,
            image_height=224,
            loss_coef=100,
            theta_initial=None,
            theta_hidden=None,
            lower_bound_cutout=0.1, # should be smaller than 0.8
            upper_bound_cutout=1.0,
            saturate_bound=False,
            gauss_sampling=False,
            gauss_mean=0.6,
            gauss_std=0.2,
            do_cutout=True,
            center_bias=False,
            center_focus=2,
            hidden_size=256,
            averaging_weight=0.3,
            do_vqcuts=False,
    ):
        super().__init__()
        # load clip
        self.perceptor = clip_perceptor
        self.input_resolution = input_res
        self.normalize_image = clip_norm
        
        self.loss_coef = loss_coef
        self.image_width = image_width
        self.image_height = image_height

        self.batch_size = batch_size
        self.total_batches = total_batches
        self.num_batches_processed = 0

        w0 = default(theta_hidden, 30.)
        w0_initial = default(theta_initial, 30.)

        siren = SirenNet(
            dim_in=2,
            dim_hidden=hidden_size,
            num_layers=num_layers,
            dim_out=3,
            #use_bias=False,
            use_bias=True,
            w0=w0,
            w0_initial=w0_initial
        )

        self.model = SirenWrapper(
            siren,
            image_width=image_width,
            image_height=image_height
        )

        self.saturate_bound = saturate_bound
        self.saturate_limit = 0.75  # cutouts above this value lead to destabilization
        self.lower_bound_cutout = lower_bound_cutout
        self.upper_bound_cutout = upper_bound_cutout
        self.gauss_sampling = gauss_sampling
        self.gauss_mean = gauss_mean
        self.gauss_std = gauss_std
        self.do_cutout = do_cutout
        self.center_bias = center_bias
        self.center_focus = center_focus
        self.averaging_weight = averaging_weight
        self.do_vqcuts = do_vqcuts
        
    def sample_sizes(self, lower, upper, height, gauss_mean):
        if self.gauss_sampling:
            gauss_samples = torch.zeros(self.batch_size).normal_(mean=gauss_mean, std=self.gauss_std)
            outside_bounds_mask = (gauss_samples > upper) | (gauss_samples < upper)
            gauss_samples[outside_bounds_mask] = torch.zeros((len(gauss_samples[outside_bounds_mask]),)).uniform_(lower, upper)
            sizes = (gauss_samples * width).int()
        else:
            lower = max(lower*height, 224)
            #lower *= height
            upper *= height
            sizes = torch.randint(int(lower), int(upper), (self.batch_size,))
        return sizes
        
    def vqstyle_cuts(self, image):
        cut_pow=1

        input_res = self.input_resolution
        sideY, sideX = image.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, input_res)
        cutouts = []
        q = random.randint(0,self.batch_size-1)
        the_chosen_one = None
        for i in range(self.batch_size):
            size = int(torch.rand([])**cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = image[:, :, offsety:offsety + size, offsetx:offsetx + size]

        cutouts.append(resample(cutout, (input_res, input_res)))
        return torch.cat(cutouts, dim=0)

    def forward(self, text_embed, return_loss=True, dry_run=False):
        out = self.model()
        out = norm_siren_output(out)

        if not return_loss:
            return out
        
        if self.do_vqcuts:
            image_pieces = [self.vqstyle_cuts(out)]
        else:
            # determine upper and lower sampling bound
            height = out.shape[-2]
            #print(str(height) + "the one in forward")
            lower_bound = self.lower_bound_cutout
            if self.saturate_bound:
                progress_fraction = self.num_batches_processed / self.total_batches
                lower_bound += (self.saturate_limit - self.lower_bound_cutout) * progress_fraction

            # sample cutout sizes between lower and upper bound
            #sizes = self.sample_sizes(lower_bound, self.upper_bound_cutout, width, self.gauss_mean)
            sizes = self.sample_sizes(lower_bound, self.upper_bound_cutout, height, self.gauss_mean)

            # create normalized random cutouts
            if self.do_cutout:   
                image_pieces = [rand_cutout(out, size, center_bias=self.center_bias, center_focus=self.center_focus) for size in sizes]
                image_pieces = [interpolate(piece, self.input_resolution) for piece in image_pieces]
            else:
                image_pieces = [interpolate(out.clone(), self.input_resolution) for _ in sizes]

        # normalize
        image_pieces = torch.cat([self.normalize_image(piece) for piece in image_pieces])
        
        # calc image embedding
        with autocast(enabled=False):
            image_embed = self.perceptor.encode_image(image_pieces)
            
        # calc loss
        # loss over averaged features of cutouts
        avg_image_embed = image_embed.mean(dim=0).unsqueeze(0)
        averaged_loss = -self.loss_coef * torch.cosine_similarity(text_embed, avg_image_embed, dim=-1).mean()
        # loss over all cutouts
        general_loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
        # merge losses
        loss = averaged_loss * (self.averaging_weight) + general_loss * (1 - self.averaging_weight)

        # count batches
        if not dry_run:
            self.num_batches_processed += self.batch_size
        
        return out, loss


class Imagine(nn.Module):
    def __init__(
            self,
            *,
            text=None,
            img=None,
            clip_encoding=None,
            batch_size=4,
            gradient_accumulate_every=4,
            save_every=100,
            image_width=540,
            image_height=540,
            num_layers=16,
            epochs=20,
            iterations=1050,
            save_progress=True,
            seed=None,
            open_folder=True,
            save_date_time=False,
            start_image_path=None,
            start_image_train_iters=10,
            start_image_lr=3e-4,
            theta_initial=None,
            theta_hidden=None,
            lower_bound_cutout=0.1, # should be smaller than 0.8
            upper_bound_cutout=1.0,
            saturate_bound=False,
            averaging_weight=0.3,

            create_story=False,
            story_start_words=5,
            story_words_per_epoch=5,
            story_separator=None,
            gauss_sampling=False,
            gauss_mean=0.6,
            gauss_std=0.2,
            do_cutout=True,
            center_bias=False,
            center_focus=2,
            optimizer="AdamP",
            jit=False,
            hidden_size=256,
            save_gif=False,
            save_video=False,
            model_name="ViT-B/16",
            
            lr=1e-5,
            lr_max=None,
            change_lr=False,
            use_flow=False,
            use_flow_fx=False,
            bg_wt=None,
            bg_wt_range=None,
            flow_txts=None,
            flow_txt_wt=None,
            flow_txt_enc=None,
            flow_txts_enc_list=None,
            flow_imgs=None,
            flow_img_wt=None,
            flow_img_wt_range=None,
            flow_img_enc=None,
            flow_imgs_enc_list=None,
            bg_txt=None,
            bg_txt_wt=None,
            bg_txt_enc=None, 
            bg_img=None,
            bg_img_wt=None,
            bg_img_wt_range=None,
            bg_img_enc=None,
            flow_flavor=None,
            tdex=0,
            idex=0,
            t_num=7,
            i_num=3,
            t_freq=300,
            i_freq=300,
            use_flow_txt_offset=True,
            total_iterations=0,
            current_lr=3e-4,
            do_vqcuts=False,
            seemless=False,
            img_fading=False,
            ifade_range=21,
            ifadex=0,
            txt_fading=False,
            tfade_range=21,
            tfadex=0,
            
            
    ):

        super().__init__()

        if exists(seed):
            tqdm.write(f'setting seed: {seed}')
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            
        self.epochs=epochs
        self.iterations = iterations
        self.total_iterations = self.epochs*self.iterations
        self.lr=lr
        self.lr_max=lr_max
        self.current_lr=lr
        self.change_lr=change_lr
        self.use_flow=use_flow
        self.use_flow_fx=use_flow_fx
        self.bg_wt=bg_wt
        self.bg_wt_range=bg_wt_range
        self.flow_txts=flow_txts
        self.flow_txt_wt=flow_txt_wt
        self.flow_txt_enc=flow_txt_enc
        self.flow_imgs=flow_imgs
        self.flow_img_wt=flow_img_wt
        self.flow_img_wt_range=flow_img_wt_range
        self.flow_img_enc=flow_img_enc
        self.bg_txt=bg_txt
        self.bg_txt_wt=bg_txt_wt
        self.bg_txt_enc=bg_txt_enc
        self.bg_img=bg_img
        self.bg_img_wt=bg_img_wt
        self.bg_img_wt_range=bg_img_wt_range
        self.bg_img_enc=bg_img_enc
        self.tdex=tdex
        self.idex=idex
        self.t_num=t_num
        self.i_num=i_num
        self.t_freq=t_freq
        self.i_freq=i_freq
        self.use_flow_txt_offset = use_flow_txt_offset
        self.do_vqcuts = do_vqcuts
        self.seemless = seemless
        self.img_fading = img_fading
        self.ifade_range = ifade_range
        self.ifadex = ifadex
        self.txt_fading = txt_fading
        self.tfade_range = tfade_range
        self.tfadex = tfadex
        
        if use_flow:
            self.flow_flavor=[0]*4
        else:
            self.flow_flavor=None

        # fields for story creation:
        self.create_story = create_story
        self.words = None
        self.separator = str(story_separator) if story_separator is not None else None
        if self.separator is not None and text is not None:
            #exit if text is just the separator
            if str(text).replace(' ','').replace(self.separator,'') == '':
                print('Exiting because the text only consists of the separator! Needs words or phrases that are separated by the separator.')
                exit()
            #adds a space to each separator and removes double spaces that might be generated
            text = text.replace(self.separator,self.separator+' ').replace('  ',' ').strip()
        self.all_words = text.split(" ") if text is not None else None
        self.num_start_words = story_start_words
        self.words_per_epoch = story_words_per_epoch
        if create_story:
            assert text is not None,  "We need text input to create a story..."
            # overwrite epochs to match story length
            num_words = len(self.all_words)
            self.epochs = 1 + (num_words - self.num_start_words) / self.words_per_epoch
            # add one epoch if not divisible
            self.epochs = int(self.epochs) if int(self.epochs) == self.epochs else int(self.epochs) + 1
            if self.separator is not None:
                if self.separator not in text:
                    print("Separator '"+self.separator+"' will be ignored since not in text!")
                    self.separator = None
                else:
                    self.epochs = len(list(filter(None,text.split(self.separator))))
            print("Running for", self.epochs, "epochs" + (" (split with '"+self.separator+"' as the separator)" if self.separator is not None else ""))
        else: 
            self.epochs = epochs

        # jit models only compatible with version 1.7.1
        if "1.7.1" not in torch.__version__:
            if jit == True:
                print("Setting jit to False because torch version is not 1.7.1.")
            jit = False

        # Load CLIP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_perceptor, norm = load(model_name, jit=jit, device=self.device)
        self.perceptor = clip_perceptor.eval()
        for param in self.perceptor.parameters():
            param.requires_grad = False
        if jit == False:
            input_res = clip_perceptor.visual.input_resolution
        else:
            input_res = clip_perceptor.input_resolution.item()
        self.clip_transform = create_clip_img_transform(input_res)
        
        
        self.image_width = image_width
        self.image_height = image_height
        total_batches = self.epochs * self.iterations * batch_size * gradient_accumulate_every
        model = DeepDaze(
                self.perceptor,
                norm,
                input_res,
                total_batches,
                batch_size=batch_size,
                image_width=image_width,
                image_height=image_height,
                num_layers=num_layers,
                theta_initial=theta_initial,
                theta_hidden=theta_hidden,
                lower_bound_cutout=lower_bound_cutout,
                upper_bound_cutout=upper_bound_cutout,
                saturate_bound=saturate_bound,
                gauss_sampling=gauss_sampling,
                gauss_mean=gauss_mean,
                gauss_std=gauss_std,
                do_cutout=do_cutout,
                center_bias=center_bias,
                center_focus=center_focus,
                hidden_size=hidden_size,
                averaging_weight=averaging_weight,
                do_vqcuts=do_vqcuts,
            ).to(self.device)
        self.model = model
        self.scaler = GradScaler()
        siren_params = model.model.parameters()
        if optimizer == "AdamP":
            self.optimizer = AdamP(siren_params, lr)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(siren_params, lr)
        elif optimizer == "DiffGrad":
            self.optimizer = DiffGrad(siren_params, lr)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every
        self.save_date_time = save_date_time
        self.open_folder = open_folder
        self.save_progress = save_progress
        self.text = text
        self.image = img
        self.textpath = create_text_path(self.perceptor.context_length, text=text, img=img, encoding=clip_encoding, separator=story_separator)
        self.filename = self.image_output_path() 
        
        #initialize encoding lists for flow, which will be set inside create_clip_encoding >> create_flow_enc
        self.flow_txts_enc_list = [None] * self.t_num
        self.flow_imgs_enc_list = [None] * self.i_num

        # create coding to optimize for
        self.clip_encoding = self.create_clip_encoding(text=text, img=img, encoding=clip_encoding)
    
        self.start_image = None
        self.start_image_train_iters = start_image_train_iters
        self.start_image_lr = start_image_lr
        if exists(start_image_path):
            file = Path(start_image_path)
            assert file.exists(), f'file does not exist at given starting image path {self.start_image_path}'
            image = Image.open(str(file))
            start_img_transform = T.Compose([T.Resize(image_width),
                                             T.CenterCrop((image_width, image_width)),
                                             T.ToTensor()])
            image_tensor = start_img_transform(image).unsqueeze(0).to(self.device)
            self.start_image = image_tensor

        self.save_gif = save_gif
        self.save_video = save_video
            
    def create_clip_encoding(self, text=None, img=None, encoding=None):
        self.text = text
        self.img = img
        if encoding is not None:
            encoding = encoding.to(self.device)
        elif self.use_flow:
            encoding = self.create_flow_encoding(ti=0)
        elif self.create_story:
            encoding = self.update_story_encoding(epoch=0, iteration=1)
        elif text is not None and img is not None:
            encoding = (self.create_text_encoding(text) + self.create_img_encoding(img)) / 2
        elif text is not None:
            encoding = self.create_text_encoding(text)
        elif img is not None:
            encoding = self.create_img_encoding(img)
        return encoding
    
    #initializes the flow encoding: requires at least flow_txt or flow_img   
    def create_flow_encoding(self, ti):
        #make the encodings
        has_flow_txts=0
        has_flow_imgs=0
        has_bg_txt=0
        has_bg_img=0
        if self.flow_txts is not None:
            j=0
            while j < self.t_num:
                self.flow_txts_enc_list[j] = self.create_text_encoding( self.flow_txts[j] )
                j+=1
            self.flow_txt_enc = self.flow_txts_enc_list[0]
            has_flow_txts=1
        if self.flow_imgs is not None:
            k=0
            while k< self.i_num:
                self.flow_imgs_enc_list[k] = self.create_img_encoding( self.flow_imgs[k] )
                k+=1	
            self.flow_img_enc = self.flow_imgs_enc_list[0]
            has_flow_imgs=1
        if self.bg_txt is not None:
            self.bg_txt_enc = self.create_text_encoding(self.bg_txt)
            has_bg_txt=1
        if self.bg_img is not None:
            self.bg_img_enc = self.create_img_encoding(self.bg_img)
            has_bg_img=1
        
        self.flow_flavor[0] = has_flow_txts
        self.flow_flavor[1] = has_flow_imgs
        self.flow_flavor[2] = has_bg_txt
        self.flow_flavor[3] = has_bg_img

        return self.update_flow_enc(ti)
            
        
    def create_text_encoding(self, text):
        tokenized_text = tokenize(text).to(self.device)
        with torch.no_grad():
            text_encoding = self.perceptor.encode_text(tokenized_text).detach()
        return text_encoding
    
    def create_img_encoding(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        normed_img = self.clip_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_encoding = self.perceptor.encode_image(normed_img).detach()
        return img_encoding
    
    def index_of_first_separator(self) -> int:
        for c, word in enumerate(self.all_words):
            if self.separator in str(word):
                return c +1

    def update_story_encoding(self, epoch, iteration):
        if self.separator is not None:
            self.words = " ".join(self.all_words[:self.index_of_first_separator()])
            #removes separator from epoch-text
            self.words = self.words.replace(self.separator,'')
            self.all_words = self.all_words[self.index_of_first_separator():]
        else:
            if self.words is None:
                self.words = " ".join(self.all_words[:self.num_start_words])
                self.all_words = self.all_words[self.num_start_words:]
            else:
                # add words_per_epoch new words
                count = 0
                while count < self.words_per_epoch and len(self.all_words) > 0:
                    new_word = self.all_words[0]
                    self.words = " ".join(self.words.split(" ") + [new_word])
                    self.all_words = self.all_words[1:]
                    count += 1
                # remove words until it fits in context length
                while len(self.words) > self.perceptor.context_length:
                    # remove first word
                    self.words = " ".join(self.words.split(" ")[1:])
        # get new encoding
        print("Now thinking of: ", '"', self.words, '"')
        sequence_number = self.get_img_sequence_number(epoch, iteration)
        # save new words to disc
        with open("story_transitions.txt", "a") as f:
            f.write(f"{epoch}, {sequence_number}, {self.words}\n")
        
        encoding = self.create_text_encoding(self.words)
        return encoding
    
    def calc_flow_wts(self, ti):
        x = (ti/self.total_iterations)*self.epochs  #is this wrong?? should it be ti/( totitrs*epochs)???
        fiw = self.flow_img_wt
        bgiw = self.bg_img_wt
        bgw = self.bg_wt
        if self.use_flow_fx == True:
		    #if we have both flow types / else no wts besides the bg_wt are needed
            if self.flow_img_enc is not None and self.flow_txt_enc is not None:
                fix = flow_fx(x, 'E')
                fiw = self.flow_img_wt - self.flow_img_wt_range*fix
                tqdm.write(f'flow_img_wt: "{fiw}"')
				
            if self.bg_img_enc is not None:
                bgix = flow_fx(x, 'F')
                bgiw = self.bg_img_wt - self.bg_img_wt_range*bgix
                tqdm.write(f'bg_img_wt: "{bgiw}"')
            
            #if we have any bg, then we need the bg_wt     
            if self.bg_img_enc is not None or self.bg_txt_enc is not None:
                bgx = flow_fx(x, 'G')
                bgw = self.bg_wt - self.bg_wt_range*bgx
                tqdm.write(f'bg_wt: "{bgw}"')
                
        return fiw, bgiw, bgw

        
    def update_flow_inputs(self, ti):
        #SEEMLESS
        if self.seemless:
            #IMG FADE
            if (ti+self.ifade_range)%self.i_freq == 0:
                self.img_fading = True
            if self.img_fading:
                x = self.ifadex/((2*self.ifade_range)+1)
                img_now = self.flow_imgs_enc_list[self.idex%self.i_num]
                img_next = self.flow_imgs_enc_list[(self.idex+1)%self.i_num]
                ifade_wt = (1-x) #starts and 1, goes to 0 after 2*fade_range+1 steps
                self.flow_img_enc = img_now*ifade_wt + img_next*(1-ifade_wt)
                self.ifadex += 1
                if (ti-self.ifade_range)%self.i_freq == 0:
                    self.img_fading = False
                    self.idex += 1
                    self.ifadex = 0
            
            #TXT FADE
            #for now, no txt offset implemented
            if (ti+self.tfade_range)%self.t_freq == 0:
                self.txt_fading = True
            if self.txt_fading:
                y = self.tfadex/((2*self.tfade_range)+1)
                txt_now = self.flow_txts_enc_list[self.tdex%self.t_num]
                txt_next = self.flow_txts_enc_list[(self.tdex+1)%self.t_num]
                tfade_wt = (1-y) #starts and 1, goes to 0 after 2*fade_range+1 steps
                self.flow_txt_enc = txt_now*tfade_wt + txt_next*(1-tfade_wt)
                self.tfadex += 1
                if (ti-self.tfade_range)%self.t_freq == 0:
                    self.txt_fading = False
                    self.tdex += 1
                    self.tfadex = 0
            
        else:         
            ##OLD METHOD
            if self.flow_img_enc is not None and ti != 0 and ti%self.i_freq == 0:
                self.flow_img_enc = self.flow_imgs_enc_list[self.idex%self.i_num]
                self.idex += 1
        
            if self.use_flow_txt_offset:
                txt_offset = ti - 2*(self.i_freq//3)
                if txt_offset >=0 and txt_offset % self.t_freq == 0:
                    self.flow_txt_enc = self.flow_txts_enc_list[self.tdex%self.t_num]
                    self.tdex += 1
            elif self.flow_txt_enc is not None and ti != 0 and ti%self.t_freq == 0:
                self.flow_txt_enc = self.flow_txts_enc_list[self.tdex%self.t_num]
                self.tdex += 1

  
    def calc_flow_encs(self, fiw, bgiw, bgw):
        #flow encoding
        if self.flow_txt_enc is not None and self.flow_img_enc is not None:
            flow_enc = fiw*self.flow_img_enc + (1-fiw)*self.flow_txt_enc
        elif self.flow_txt_enc is None:
            flow_enc =  self.flow_img_enc
        else:
            flow_enc =  self.flow_txt_enc
        #bg encoding
        if self.bg_txt_enc is not None and self.bg_img_enc is not None:
            bg_enc = bgiw*self.bg_img_enc + (1-bgiw)*self.bg_txt_enc
        elif self.bg_txt_enc is None and self.bg_img_enc is None:
            bg_enc = None
        elif self.bg_txt is None:
            bg_enc = self.bg_img_enc
        elif self.bg_img_enc is None:
            bg_enc = self.flow_txt_enc
            
        if bg_enc is None:
            return flow_enc
        else:
            return  bgw*bg_enc + (1-bgw)*flow_enc
		 
           
    def update_flow_enc(self, ti):
		### NEEDS TO BE TESTED FOR IF IT'S WORKING FOR  NON-ALL OUT FLOW MODES###
		 
		#calculate flow wts >> maybe variable functions used
        fiw, bgiw, bgw = self.calc_flow_wts(ti)
        
        #flow updates of img/txts
        self.update_flow_inputs(ti)
        
        return self.calc_flow_encs(fiw, bgiw, bgw)
    
            
    def image_output_path(self, sequence_number=None):
        if self.flow_txts is None:
            text = self.text
        else:
            if self.txt_fading:
                text = self.flow_txts[(self.tdex) % self.t_num] + ">" + self.flow_txts[(self.tdex+1) % self.t_num]
            else:
                text = self.flow_txts[(self.tdex) % self.t_num]
        text_name = text.replace(" ", "_")
        if self.flow_imgs is None:
            img_name = ""
        else:
            if self.img_fading:
                img_name = self.flow_imgs[(self.idex) % self.i_num] + ">" + self.flow_imgs[(self.idex+1) % self.i_num]
            else:    
                img_name = self.flow_imgs[(self.idex) % self.i_num]
            img_name = img_name.replace(".jpeg", "")
            img_name = img_name.replace("i/", "")
        
        lr_name = str( round(self.current_lr, 8) )
        
        output_path = text_name+"_"+img_name+"_lr="+lr_name
        if sequence_number:
            sequence_number_left_padded = str(sequence_number).zfill(6)
            output_path = f"{sequence_number_left_padded}.{output_path}"
        #return Path(f"results/{output_path}.jpg")
        return Path(f"results/{output_path}.png")
		
		
		
    def image_output_path_original(self, sequence_number=None):
        """
        Returns underscore separated Path.
        A current timestamp is prepended if `self.save_date_time` is set.
        Sequence number left padded with 6 zeroes is appended if `save_every` is set.
        :rtype: Path
        """
        output_path = self.textpath
        if sequence_number:
            sequence_number_left_padded = str(sequence_number).zfill(6)
            output_path = f"{output_path}.{sequence_number_left_padded}"
        if self.save_date_time:
            current_time = datetime.now().strftime("%y%m%d-%H%M%S_%f")
            output_path = f"{current_time}_{output_path}"
        return Path(f"{output_path}.jpg")

    def train_step(self, epoch, iteration):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            with autocast(enabled=True):
                out, loss = self.model(self.clip_encoding)
            loss = loss / self.gradient_accumulate_every
            total_loss += loss
            self.scaler.scale(loss).backward()    
        out = out.cpu().float().clamp(0., 1.)
        #out = out.cpu().float()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if (iteration % self.save_every == 0) and self.save_progress:
            self.save_image(epoch, iteration, img=out)

        return out, total_loss
    
    def get_img_sequence_number(self, epoch, iteration):
        current_total_iterations = epoch * self.iterations + iteration
        sequence_number = current_total_iterations // self.save_every
        return sequence_number

    @torch.no_grad()
    def save_image(self, epoch, iteration, img=None):
        sequence_number = self.get_img_sequence_number(epoch, iteration)

        if img is None:
            img = self.model(self.clip_encoding, return_loss=False).cpu().float().clamp(0., 1.)
        self.filename = self.image_output_path(sequence_number=sequence_number)
        
        pil_img = T.ToPILImage()(img.squeeze())
        pil_img.save(f"results/{self.textpath}.png", mode='png')
        pil_img.save(self.filename, mode='png')
        
        #pil_img.save(self.filename, quality=95, subsampling=0)
        #pil_img.save(f"{self.textpath}.jpg", quality=95, subsampling=0)

     
        tqdm.write(f'image updated at "./{str(self.filename)}"')
        
    def save_img(img, fname=None):
        img = np.array(img)[:,:,:]
        img = np.transpose(img, (1,2,0))  
        img = exposure.equalize_adapthist(np.clip(img, -1., 1.))
        img = np.clip(img*255, 0, 255).astype(np.uint8)
        if fname is not None:
            imageio.imsave(fname, np.array(img))
            imageio.imsave('result.jpg', np.array(img))

    def generate_gif(self):
        images = []
        for file_name in sorted(os.listdir('./')):
            if file_name.startswith(self.textpath) and file_name != f'{self.textpath}.jpg':
                images.append(imread(os.path.join('./', file_name)))

        if self.save_video:
            mimsave(f'{self.textpath}.mp4', images)
            print(f'Generated image generation animation at ./{self.textpath}.mp4')
        if self.save_gif:
            mimsave(f'{self.textpath}.gif', images)
            print(f'Generated image generation animation at ./{self.textpath}.gif')
    
    
    # i don't think i need i here- as long as i stick with this x definition     
    def update_lr(self, ti, i, epoch, lr_diff):
        x = (ti / self.total_iterations)*self.epochs
        do_lr_init = False
        if do_lr_init and ti < 25:
            return self.lr_max
        
        fx = flow_fx(x, 'A')
        new_lr = lr_diff*fx + self.lr
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr
        tqdm.write(f'lr = "{new_lr}". ')	
        return new_lr

    def forward(self):
        if exists(self.start_image):
            tqdm.write('Preparing with initial image...')
            optim = DiffGrad(self.model.model.parameters(), lr = self.start_image_lr)
            pbar = trange(self.start_image_train_iters, desc='iteration')
            try:
                for _ in pbar:
                    loss = self.model.model(self.start_image)
                    loss.backward()
                    pbar.set_description(f'loss: {loss.item():.2f}')

                    optim.step()
                    optim.zero_grad()
            except KeyboardInterrupt:
                print('interrupted by keyboard, gracefully exiting')
                return exit()

            del self.start_image
            del optim

        tqdm.write(f'Imagining "{self.textpath}" from the depths of my weights...')

        with torch.no_grad():
            self.model(self.clip_encoding, dry_run=True) # do one warmup step due to potential issue with CLIP and CUDA

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        try:
            ti = 0
            if self.change_lr:
                lr_diff = (self.lr_max - self.lr)
            for epoch in trange(self.epochs, desc='epochs'):
                pbar = trange(self.iterations, desc='iteration')
                for i in pbar:
                    _, loss = self.train_step(epoch, i)
                    pbar.set_description(f'loss: {loss.item():.2f}')
                    
                    if self.change_lr:
                        self.current_lr = self.update_lr(ti, i, epoch, lr_diff)
                    if self.use_flow:
                        self.clip_encoding = self.update_flow_enc(ti)
                        
                    ti += 1

                # Update clip_encoding per epoch if we are creating a story
                if self.create_story:
                    self.clip_encoding = self.update_story_encoding(epoch, i)
        except KeyboardInterrupt:
            print('interrupted by keyboard, gracefully exiting')
            return

        self.save_image(epoch, i) # one final save at end

        if (self.save_gif or self.save_video) and self.save_progress:
            self.generate_gif()
