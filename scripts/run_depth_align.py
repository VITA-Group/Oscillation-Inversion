import os, sys, yaml, gc
from datetime import datetime
from PIL import Image
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from diffusers.image_processor import VaeImageProcessor
from diffusers_local.pipelines.flux.pipeline_flux import FluxPipeline, retrieve_timesteps, calculate_shift
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import DepthAlignConfig

__ENABLE_VRAM_TRACKING__ = False

def dump_snapshot():
    if __ENABLE_VRAM_TRACKING__:
        torch.cuda.memory._dump_snapshot('cuda.dump')

class DepthAlign(nn.Module):
    def __init__(self, config : DepthAlignConfig):
        super().__init__()
        self.device = config.device
        self.prompt = config.prompt
        self.num_inference_steps = config.num_inference_steps
        self.num_fixed_point_steps = config.num_fixed_point_steps
        self.num_predict_steps = config.num_predict_steps
        self.num_align_steps = config.num_align_steps
        
        self.pipe : FluxPipeline = FluxPipeline.from_pretrained(config.flux_model_id, torch_dtype=torch.bfloat16).to(self.device)
        self.d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to(self.device)
        self.mask_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        
        self.generator = torch.Generator(self.device)
        
        self.noise_optim : torch.nn.Parameter = None
        self.P_optim : torch.nn.Parameter = None
        
        self.image_processor = VaeImageProcessor()
        self.width, self.height = config.image_width, config.image_height
        
        def read_image(file_path, mode='RGB'):
            image_raw = Image.open(file_path).convert(mode).resize((self.width, self.height))
            return self.image_processor.preprocess(image_raw, height=self.height, width=self.width).to(dtype=torch.bfloat16, device=self.device)
        
        self.fg_image = read_image(config.path_fg_image)
        self.bg_image = read_image(config.path_bg_image)
        self.image_target = read_image(config.path_target_image)
        self.mask = transforms.PILToTensor()(Image.open(config.path_fg_image).convert('RGBA')).to(self.device)[3, :, :] / 255.0
        
        self.config_warping_net = config.config_warping_net
        num_layers, hidden_dim = self.config_warping_net.num_layers, self.config_warping_net.hidden_dim
        
        layers = [nn.Linear(2, num_layers - 1), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 2))
        self.warping_net = nn.Sequential(*layers)
        
        self.logdir = os.path.join(os.getcwd(), 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.logdir, exist_ok=True)

    def apply_transformer(self, hidden_states, latent_image_ids, timestep):
        return self.pipe.transformer(
            hidden_states=hidden_states,
            timestep=timestep / 1000,
            guidance=None,
            pooled_projections=self.pipe.pooled_prompt_embeds,
            encoder_hidden_states=self.pipe.prompt_embeds,
            txt_ids=self.pipe.text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.pipe.joint_attention_kwargs,
            return_dict=False,
        )[0]
        
    def image_to_latents(self, image):
        latents_raw = self.pipe.vae.encode(image)[0].sample()
        latents_raw = (latents_raw - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        latents = self.pipe._pack_latents(latents_raw, latents_raw.shape[0], latents_raw.shape[1], latents_raw.shape[2], latents_raw.shape[3])
        return latents_raw, latents
    
    def latents_to_image(self, latents):
        latents = self.pipe._unpack_latents(latents, self.height, self.width, self.pipe.vae_scale_factor)
        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        return self.pipe.vae.decode(latents, return_dict=False)[0]
    
    def image_to_pil(self, image):
        image_pil = self.image_processor.postprocess(image.cpu().detach(), output_type='PIL')
        return Image.fromarray((image_pil[0] * 255).astype(np.uint8))
    
    def get_current_image(self):
        image = self.fg_image * self.mask + self.bg_image * (1 - self.mask)
        return image.to(dtype=torch.bfloat16, device=self.device)
    
    def image_to_depth(self, image):
        return self.d_model(image.float())['metric_depth']
    
    def visualize_latents(self, latents, image_name):
        image = self.latents_to_image(latents)
        image_pil = self.image_to_pil(image)
        image_pil.save(os.path.join(self.logdir, image_name))
    
    def inpaint(self, image):
        raise NotImplementedError()
    
    def initialize_model(self):
        latents_raw, latents = self.image_to_latents(self.get_current_image())
        
        self.pipe(self.prompt, height=self.height, width=self.width, output_type="pil", num_inference_steps=1, generator=self.generator)
        print(f"Pipe initialized with prompt: {self.prompt}")
        
        mu = calculate_shift(
            image_seq_len=latents_raw.shape[1],
            base_seq_len=self.pipe.scheduler.config.base_image_seq_len,
            max_seq_len=self.pipe.scheduler.config.max_image_seq_len,
            base_shift=self.pipe.scheduler.config.base_shift,
            max_shift=self.pipe.scheduler.config.max_shift)
        
        self.timesteps, self.num_inference_steps = retrieve_timesteps(scheduler=self.pipe.scheduler,
                            num_inference_steps=self.num_inference_steps,
                            device=self.device,
                            timesteps=None,
                            sigmas=np.linspace(1.0, 1 / self.num_inference_steps, self.num_inference_steps),
                            mu=mu)
        self.timestep = self.timesteps[self.num_predict_steps].expand(latents_raw.shape[0]).to(latents_raw.dtype)
        
        self.num_warmup_steps = max(len(self.timesteps) - self.num_inference_steps * self.pipe.scheduler.order, 0)
        self.pipe._num_timesteps = len(self.timesteps)
        
        # Prepare latent variables
        _, self.latent_image_ids = self.pipe.prepare_latents(
            1 * 1,
            num_channels_latents=self.pipe.transformer.config.in_channels // 4,
            height=self.height,
            width=self.width,
            dtype=self.pipe.prompt_embeds.dtype,
            device=self.pipe.device,
            generator=self.generator,
            latents=None
        )
        
        V = latents.to(latents_raw.dtype).to(latents_raw.device)
        torch.set_grad_enabled(False)
        
        for step_idx in range(self.num_fixed_point_steps):
            noise_pred = self.apply_transformer(V, self.latent_image_ids, self.timestep)
            V_predict = V + (self.pipe.scheduler.sigmas[-1] - self.pipe.scheduler.sigmas[self.num_predict_steps]) * noise_pred
            V = latents - ( (self.pipe.scheduler.sigmas[-1] - self.pipe.scheduler.sigmas[self.num_predict_steps]) * noise_pred )
            
            self.visualize_latents(V_predict, f"fixed_point_{step_idx}.png")
        
        # Initialize the parameters to be trained
        self.visualize_latents(latents, 'target.png')
        self.noise_optim = torch.nn.Parameter(noise_pred.to(latents_raw.dtype), requires_grad=True)
        self.P_optim = torch.nn.Parameter(V_predict.to(latents_raw.dtype), requires_grad=True)
        self.mask = torch.nn.Parameter(self.mask, requires_grad=True)
        
        self.transformer = self.pipe.transformer
        del self.pipe.text_encoder
        del self.pipe.text_encoder_2
        torch.cuda.empty_cache()
        for param in self.pipe.transformer.parameters():
            param.requires_grad = False
        for param in self.d_model.parameters():
            param.requires_grad = True
            
        torch.set_grad_enabled(True)
        
    def compute_dirichlet_loss(self, points):
        points.requires_grad = True
        offsets = self.warping_net(points)
        
        gradients = torch.autograd.grad(outputs=offsets, inputs=points,
                              grad_outputs=torch.ones_like(offsets),
                              create_graph=True, retain_graph=True)[0]
        
        dirichlet_energy = torch.sum(gradients ** 2, dim=-1).mean()
        return dirichlet_energy
        
        
    def training_step(self, idx_iteration, depth_target=None):
        _, current_latents = self.image_to_latents(self.get_current_image())
        latents_with_noise = current_latents - (self.pipe.scheduler.sigmas[-1] - self.pipe.scheduler.sigmas[self.num_predict_steps]) * self.noise_optim
        
        noise_predict = self.apply_transformer(latents_with_noise, self.latent_image_ids, self.timestep)
        latents_predicted = latents_with_noise + (self.pipe.scheduler.sigmas[-1] - self.pipe.scheduler.sigmas[self.num_predict_steps]) * noise_predict
        
        image_predicted = self.latents_to_image(latents_predicted)
        
        self.visualize_latents(latents_predicted, f'iter_{idx_iteration}.png')
        mask_pil = Image.fromarray((self.mask.detach().cpu() * 255).numpy().astype(np.uint8))
        mask_pil.save(os.path.join(self.logdir, f'mask_{idx_iteration}.png'))
        
        loss = torch.norm(image_predicted - self.image_target)
        # lambda_mask_loss = getattr(self, 'lambda_mask_loss', 1.0)
        # if lambda_mask_loss > 0:
        #     loss += torch.norm(mask - mask_target) * lambda_mask_loss
        if depth_target is not None:
            depth = self.image_to_depth(image_predicted)
            loss += torch.norm(depth - depth_target) * getattr(self, 'lambda_depth_loss', 1.0)
        return loss

    def fit(self, mask_target=None, depth_target=None, step_align=None):
        
        if step_align is None:
            step_align = self.num_align_steps
            
        for step_idx in range(step_align):
            loss = self.training_step(step_idx, depth_target=depth_target)
            print(f"Step {step_idx}: Loss {loss}")
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.noise_optim, 'lr': 0.1},
                # {'params': self.P_optim, 'lr': 0.01},
                {'params': self.mask, 'lr': 0.0}
            ]
        )
    
if __name__ == '__main__':
    assert len(sys.argv) == 2, "Usage: python main.py <config_file>"
    if __ENABLE_VRAM_TRACKING__: torch.cuda.memory._record_memory_history(max_entries=100000, enabled='all')
    with open(sys.argv[1], 'r') as config_file:
        config_yaml = yaml.safe_load(config_file)
    config = DepthAlignConfig(**config_yaml['model'])
    model = DepthAlign(config)
    model.initialize_model()
    model.configure_optimizers()
    model.fit()
    if __ENABLE_VRAM_TRACKING__: torch.cuda.memory._record_memory_history(enabled=None)