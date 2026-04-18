


# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import PIL
from PIL import Image
import numpy as np
import torch
import transformers
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput


import torch.nn as nn
import torch.utils.checkpoint as checkpoint


PIL_INTERPOLATION = {
    "linear": PIL.Image.Resampling.BILINEAR,
    "bilinear": PIL.Image.Resampling.BILINEAR,
    "bicubic": PIL.Image.Resampling.BICUBIC,
    "lanczos": PIL.Image.Resampling.LANCZOS,
    "nearest": PIL.Image.Resampling.NEAREST,
}


import torchvision
from torchvision import transforms


import matplotlib.pyplot as plt
import matplotlib.cm as cm


from PIL import Image



if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps




def image2tensor(imagepath):
    # image process setting and module
    size = 512
    interpolation="bicubic"
    flip_p=1
    center_crop=False

    interpolation = {
        "linear": PIL_INTERPOLATION["linear"],
        "bilinear": PIL_INTERPOLATION["bilinear"],
        "bicubic": PIL_INTERPOLATION["bicubic"],
        "lanczos": PIL_INTERPOLATION["lanczos"],
    }[interpolation]

    flip_transform = transforms.RandomHorizontalFlip(flip_p)

    # load image and preprocess 
    image = Image.open(imagepath)

    if not image.mode == "RGB":
        image = image.convert("RGB")

    img = np.array(image).astype(np.uint8)

    if center_crop:
        crop = min(img.shape[0], img.shape[1])
        (
            h,
            w,
        ) = (
            img.shape[0],
            img.shape[1],
        )
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]


    image = Image.fromarray(img)
    image = image.resize((size, size), resample=interpolation)

    image = flip_transform(image)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)

    example = {}
    example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1).to(device)
    #print(example["pixel_values"].shape) # torch.Size([3, 512, 512])

    imagetensor =  example["pixel_values"]

    return imagetensor


def image_pt2pil(img): # input image tensor shoud be of shape 3,512,512
    input_img_torch_resized = img.permute(1, 2, 0)
    input_img_np = input_img_torch_resized.detach().cpu().numpy()
    input_img_np = (input_img_np * 255).astype(np.uint8)
    return Image.fromarray((input_img_np).astype(np.uint8)) 





def flux_optim(pils_source, pils_target, pils_target_2, text_prompt, optim_iters, pils_sde=None): 

    from diffusers_local.pipelines.flux.pipeline_flux import FluxPipeline
    model_id = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")

    # model_id = "black-forest-labs/FLUX.1-dev" #you can also use `black-forest-labs/FLUX.1-dev`
    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    
    
    device = pipe.device
    #pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    #d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')
    # def d(im):
    #     return d_model.infer_pil(im)

    with torch.no_grad(): 
        height = 512
        width = 512

        # pipe prompt preparation
        prompt = text_prompt
        output_type = "pil"
        seed = 42
        testimage = pipe(
            prompt,
            guidance_scale=3.5,
            height = height,
            width = width, 
            output_type="pil",
            num_inference_steps=2, #use a larger number if you are using [dev]
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]
        #testimage.save("testimage.png")

        #breakpoint()


        # 4. Prepare latent variables
        num_channels_latents = pipe.transformer.config.in_channels // 4
        generator=torch.Generator("cuda").manual_seed(seed)
        latents_rand, latent_image_ids = pipe.prepare_latents(
            1 * 1, # batch_size * num_images_per_prompt
            num_channels_latents,
            height,
            width,
            pipe.prompt_embeds.dtype,
            pipe.device,
            generator,
            None,
        )

        num_inference_steps = 100


        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = 16 #latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.base_image_seq_len,
            pipe.scheduler.config.max_image_seq_len,
            pipe.scheduler.config.base_shift,
            pipe.scheduler.config.max_shift,
        )
        timesteps = None
        timesteps, num_inference_steps = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
        pipe._num_timesteps = len(timesteps)


    def display_latent(latent):
        latent_ = pipe._unpack_latents(latent, height, width, pipe.vae_scale_factor)
        latent_ = (latent_ / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latent_, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type=output_type)
        #display(image[0])

        return image[0]


    guidance = None


    #########################################################################################

    del pipe.text_encoder
    del pipe.text_encoder_2
    # #del transformer_backup
    torch.cuda.empty_cache()

    for param in pipe.transformer.parameters():
        param.requires_grad = False


    ipviews_edited_images = [] # after following process, the returned should be a 2d-array list
    for img_input, img_target, img_target_2 in zip(pils_source, pils_target, pils_target_2):
        
        #########################################################################################

        with torch.no_grad(): 
            device = pipe.device

            # file_path_input = os.path.join('/home/zy3724/git_repos/dualdreamer/harmonydreamer/local_visualization_cache/export2ipadapter/input/ipview_1_hor_-90.png')
            # img = Image.open(file_path_input).convert('RGB').resize((512,512))  # Ensure the image has 3 channels (RGB)
            
            #######################################################################################
            img = img_target
            width, height = img.size
            image_processor = VaeImageProcessor()
            image_target = image_processor.preprocess(img, height=height, width=width).to(torch.bfloat16).to('cuda') #  torch.Size([1, 3, 512, 512])
        
            latents = pipe.vae.encode(image_target)[0].sample() # torch.Size([1, 16, 64, 64])
            latents = (latents - pipe.vae.config.shift_factor)*pipe.vae.config.scaling_factor
            latents_target = pipe._pack_latents(latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
            latents_target_loss = latents_target


            img = img_target_2
            width, height = img.size
            image_processor = VaeImageProcessor()
            image_target_2 = image_processor.preprocess(img, height=height, width=width).to(torch.bfloat16).to('cuda') #  torch.Size([1, 3, 512, 512])
        
            latents = pipe.vae.encode(image_target_2)[0].sample() # torch.Size([1, 16, 64, 64])
            latents = (latents - pipe.vae.config.shift_factor)*pipe.vae.config.scaling_factor
            latents_target = pipe._pack_latents(latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
            latents_target_loss_2 = latents_target




            # idx = 55
            # latents = latents_target_loss
            # latents_target = torch.clone(latents).to(latents.dtype).to(latents.device)

            # torch.manual_seed(0)
            # V = (latents_target).to(latents.dtype).to(latents.device)
            # #display_latent(V)
            # idx = idx

            # V_history = []
            # noise_pred_history = []
            # V_predict_history = []

            # for b in range(10): # 30
            #     # print(torch.norm(V))
            #     timestep = timesteps[idx].to(latents.dtype)
            #     timestep = timestep.repeat(latents.shape[0])
            #     noise_pred = pipe.transformer(
            #         hidden_states=V,
            #         # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
            #         timestep=timestep / 1000,
            #         guidance=guidance,
            #         pooled_projections=pipe.pooled_prompt_embeds,
            #         encoder_hidden_states=pipe.prompt_embeds,
            #         txt_ids=pipe.text_ids,
            #         img_ids=latent_image_ids,
            #         joint_attention_kwargs=pipe.joint_attention_kwargs,
            #         return_dict=False,
            #     )[0]

            #     V_predict = V + (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred
                
            #     V_history.append(V)
            #     noise_pred_history.append(noise_pred)
            #     V_predict_history.append(V_predict)

            #     V = latents_target_loss - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 


            # timestep = timesteps[idx].expand(latents.shape[0]).to(latents.dtype)
            # V_even = V_history
            # # V_even = V_history
            # V_mean = torch.stack(V_even).mean(dim=0)
            # noise_pred_mean = pipe.transformer(
            #     hidden_states= V_mean,
            #     timestep=timestep / 1000,  # Dividing timestep by 1000 for testing as per your code
            #     guidance=guidance,
            #     pooled_projections=pipe.pooled_prompt_embeds,
            #     encoder_hidden_states=pipe.prompt_embeds,
            #     txt_ids=pipe.text_ids,
            #     img_ids=latent_image_ids,
            #     joint_attention_kwargs=pipe.joint_attention_kwargs,
            #     return_dict=False,
            # )[0]
            # V_predict_mean = V_mean + (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred_mean
            # V_mean_ = V_mean 
            # V_predict_mean_ = V_predict_mean
            # #######################################################################################



            img = img_input
            width, height = img.size
            image_processor = VaeImageProcessor()
            image = image_processor.preprocess(img, height=height, width=width).to(torch.bfloat16).to('cuda') #  torch.Size([1, 3, 512, 512])
            latents = pipe.vae.encode(image)[0].sample() # torch.Size([1, 16, 64, 64])
            latents = (latents - pipe.vae.config.shift_factor)*pipe.vae.config.scaling_factor
            latents_target = pipe._pack_latents(latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
            latents_target_1 = latents_target


            idx = 55
            latents = latents_target_1
            latents_target = torch.clone(latents).to(latents.dtype).to(latents.device)



            torch.manual_seed(0)
            V = (latents_target).to(latents.dtype).to(latents.device)
            #display_latent(V)
            idx = idx

            V_history = []
            noise_pred_history = []
            V_predict_history = []

            for b in range(10): # 30
                # print(torch.norm(V))
                timestep = timesteps[idx].to(latents.dtype)
                timestep = timestep.repeat(latents.shape[0])
                noise_pred = pipe.transformer(
                    hidden_states=V,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pipe.pooled_prompt_embeds,
                    encoder_hidden_states=pipe.prompt_embeds,
                    txt_ids=pipe.text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=pipe.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                V_predict = V + (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred
                
                V_history.append(V)
                noise_pred_history.append(noise_pred)
                V_predict_history.append(V_predict)

                V = latents_target_1 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 


            timestep = timesteps[idx].expand(latents.shape[0]).to(latents.dtype)
            V_even = V_history
            # V_even = V_history
            V_mean = torch.stack(V_even).mean(dim=0)
            noise_pred_mean = pipe.transformer(
                hidden_states= V_mean,
                timestep=timestep / 1000,  # Dividing timestep by 1000 for testing as per your code
                guidance=guidance,
                pooled_projections=pipe.pooled_prompt_embeds,
                encoder_hidden_states=pipe.prompt_embeds,
                txt_ids=pipe.text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=pipe.joint_attention_kwargs,
                return_dict=False,
            )[0]
            V_predict_mean = V_mean + (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred_mean



            # file_path_output = os.path.join('/home/zy3724/git_repos/dualdreamer/harmonydreamer/local_visualization_cache/export2ipadapter/output_ipadapter/ipview_1_hor_-90_pred_colored.png')
            # img = Image.open(file_path_output).convert('RGB').resize((512,512))  # Ensure the image has 3 channels (RGB)
            



        with torch.enable_grad():

            img = img_target
            width, height = img.size
            image_processor = VaeImageProcessor()
            image_target = image_processor.preprocess(img, height=height, width=width).to(torch.bfloat16).to('cuda') #  torch.Size([1, 3, 512, 512])


            V_even = V_even
            #V_even = V_data
            torch.manual_seed(6)
            import torch.nn as nn
            import torch.utils.checkpoint as checkpoint
            # Set transformer parameters to not require gradients by default
            for param in pipe.transformer.parameters():
                param.requires_grad = False

            # Set only query, key, value parameters to require gradients
            for name, param in pipe.transformer.named_parameters():
                if "single_transformer_blocks" in name and (".attn.to_q" in name or ".attn.to_k" in name):
                    print('set true in single_transformer_blocks')
                    param.requires_grad = False
                    # breakpoint()

            # Prepare noise and prediction optimizable parameters
            noise_optim = torch.clone(noise_pred_mean).to(latents.dtype).to(latents.device) + 0*torch.randn_like(noise_pred_history[4]).to(latents.dtype).to(latents.device)
            noise_optim = nn.Parameter(noise_optim)
            P_optim = torch.clone(V_predict_mean).to(latents.dtype).to(latents.device)
            P_optim = nn.Parameter(P_optim)

            # Prepare optimizer with trainable transformer parameters and optimizable tensors
            optimizer = torch.optim.AdamW([
                #{'params': [p for n, p in pipe.transformer.named_parameters() if p.requires_grad], 'lr': 0.0001},
                {'params': noise_optim, 'lr': 0.03},
                {'params': P_optim, 'lr': 0.01},
            ])

            noise_predict_track = []
            P_track = []

            num_opt_iter = 0
            R = torch.norm(P_optim.data)
            for iter in range( optim_iters ):

                P_predict = P_optim
                P = P_predict - ((pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_optim)


                noise_pred = pipe.transformer(
                    hidden_states=P,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pipe.pooled_prompt_embeds,
                    encoder_hidden_states=pipe.prompt_embeds,
                    txt_ids=pipe.text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=pipe.joint_attention_kwargs,
                    return_dict=False,
                )[0]


                noise_predict = P + (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred
                noise_predict_track.append(noise_predict.detach())
                P_track.append(P.detach())


                V = noise_predict


                V_ = pipe._unpack_latents(V, height, width, pipe.vae_scale_factor)
                V_ = (V_ / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                images = pipe.vae.decode(V_, return_dict=False)[0]


                #loss_0 = torch.norm(V.float() - V_predict_mean_.detach().float())
                loss_00 = torch.norm(V.float() - latents_target_loss.detach().float()) 
                loss_01 = torch.norm(V.float() - latents_target_loss_2.detach().float())
                
                #loss_0 = 0
                #loss_1 = torch.norm(images.float() - image_target.float())
                loss_1 = 0
                #loss_2 = torch.norm(P_optim - V_mean_)
                loss_2 = 0

                loss = loss_00 + 1.5*loss_01

                loss.backward()
                optimizer.step()

                
                # # Now project updated p_optim onto an L2-ball of radius R
                # with torch.no_grad():
                #     current_norm = P_optim.norm(p=2)
                #     R = R
                #     P_optim.mul_(R / (current_norm + 1e-8))
                


                optimizer.zero_grad()


                print(f'iter {iter}, loss_00 {loss_00}, loss_01 {loss_01}, loss {loss}')
        


        #noise_predict_track.append(V_predict_mean_.detach())

        edited_images = []
        with torch.no_grad(): 
            for p in range(len(noise_predict_track)):
                edited_pil = display_latent( noise_predict_track[p] )
                edited_images.append( edited_pil )
        ipviews_edited_images.append(edited_images)



        # going_images = []
        # # latents = torch.clone(P_track[-1]).to(latents.device).to(latents.dtype)
        # for kkk, P in enumerate(P_track):
        #     latents = torch.clone(P).to(latents.device).to(latents.dtype)
        #     with torch.no_grad():
        #         for i, t in enumerate(timesteps[idx:-1]):
        #             # if self.interrupt:
        #             #     continue
        #             print(f'timestep is {t}')
        #             # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        #             timestep = t.expand(latents.shape[0]).to(latents.dtype)

        #             # handle guidance
        #             if pipe.transformer.config.guidance_embeds:
        #                 guidance = torch.tensor([pipe.guidance_scale], device=device)
        #                 guidance = guidance.expand(latents.shape[0])
        #             else:
        #                 guidance = None



        #             noise_pred = pipe.transformer(
        #                 hidden_states=latents,
        #                 # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
        #                 timestep=timestep / 1000,
        #                 guidance=guidance,
        #                 pooled_projections=pipe.pooled_prompt_embeds,
        #                 encoder_hidden_states=pipe.prompt_embeds,
        #                 txt_ids=pipe.text_ids,
        #                 img_ids=latent_image_ids,
        #                 joint_attention_kwargs=pipe.joint_attention_kwargs,
        #                 return_dict=False,
        #             )[0]



        #             # compute the previous noisy sample x_t -> x_t-1
        #             latents_dtype = latents.dtype
        #             #latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        #             # Upcast to avoid precision issues when computing prev_sample
        #             latents = latents.to(torch.float32)
        #             sigma = pipe.scheduler.sigmas[idx+i]
        #             sigma_next = pipe.scheduler.sigmas[idx+i+1]

                    

        #             # predict for last
        #             latents_predict = latents + (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx+i]) * noise_pred
        #             latents_predict = latents_predict.to(noise_pred.dtype)
        #             print(pipe.scheduler.sigmas[-1] , pipe.scheduler.sigmas[i+idx])


        #             latents = latents + (sigma_next - sigma) * noise_pred
        #             # Cast sample back to model compatible dtype
        #             latents = latents.to(noise_pred.dtype)
        #             print(sigma_next, sigma)

        #             # print('---')


        #             if latents.dtype != latents_dtype:
        #                 if torch.backends.mps.is_available():
        #                     # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
        #                     latents = latents.to(latents_dtype)

        #             if XLA_AVAILABLE:
        #                 xm.mark_step()

            
        #             latents_ = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        #             latents_ = (latents_ / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        #             image = pipe.vae.decode(latents_, return_dict=False)[0]
        #             image = pipe.image_processor.postprocess(image, output_type=output_type)
        #             #display(image[0])

        #             # print(latents)
        #             print(' ----------------------------------------------------------------------- ')
                    
        #             latents_predict_ = pipe._unpack_latents(latents_predict, height, width, pipe.vae_scale_factor)
        #             latents_predict_ = (latents_predict_ / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        #             image = pipe.vae.decode(latents_predict_, return_dict=False)[0]
        #             image = pipe.image_processor.postprocess(image, output_type=output_type)
        #             #display(image[0])
        #             # Split the save_path into the filename and the extension


        #         # Now use this modified path when saving the image
        #         going_images.append( image[0] )


    return ipviews_edited_images 

#############################################################################################################################




#############################################################################################################################





def flux_sde(text_prompt, sde_idx=80, pils_sde=None): 

    from diffusers_local.pipelines.flux.pipeline_flux import FluxPipeline
    # model_id = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`
    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    
    model_id = "black-forest-labs/FLUX.1-dev" #you can also use `black-forest-labs/FLUX.1-dev`
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    
    
    
    device = pipe.device
    #pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    #d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')
    # def d(im):
    #     return d_model.infer_pil(im)

    with torch.no_grad(): 
        height = 512
        width = 512

        # pipe prompt preparation
        prompt = text_prompt
        output_type = "pil"
        seed = 42
        testimage = pipe(
            prompt,
            guidance_scale=3.5,
            height = height,
            width = width, 
            output_type="pil",
            num_inference_steps=2, #use a larger number if you are using [dev]
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]
        #testimage.save("testimage.png")

        #breakpoint()


        # 4. Prepare latent variables
        num_channels_latents = pipe.transformer.config.in_channels // 4
        generator=torch.Generator("cuda").manual_seed(seed)
        latents_rand, latent_image_ids = pipe.prepare_latents(
            1 * 1, # batch_size * num_images_per_prompt
            num_channels_latents,
            height,
            width,
            pipe.prompt_embeds.dtype,
            pipe.device,
            generator,
            None,
        )

        num_inference_steps = 100


        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = 16 #latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.base_image_seq_len,
            pipe.scheduler.config.max_image_seq_len,
            pipe.scheduler.config.base_shift,
            pipe.scheduler.config.max_shift,
        )
        timesteps = None
        timesteps, num_inference_steps = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
        pipe._num_timesteps = len(timesteps)


    def display_latent(latent):
        latent_ = pipe._unpack_latents(latent, height, width, pipe.vae_scale_factor)
        latent_ = (latent_ / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latent_, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type=output_type)
        #display(image[0])

        return image[0]


    guidance = None


    #########################################################################################

    del pipe.text_encoder
    del pipe.text_encoder_2
    # #del transformer_backup
    torch.cuda.empty_cache()

    for param in pipe.transformer.parameters():
        param.requires_grad = False

 
    if True:
        sde_edited_images = [] # after following process, the returned should be a 2d-array list
        for img_input in pils_sde:
            #########################################################################################
            with torch.no_grad(): 

                img = img_input
                width, height = img.size
                image_processor = VaeImageProcessor()
                image = image_processor.preprocess(img, height=height, width=width).to(torch.bfloat16).to('cuda') #  torch.Size([1, 3, 512, 512])
                latents = pipe.vae.encode(image)[0].sample() # torch.Size([1, 16, 64, 64])
                latents = (latents - pipe.vae.config.shift_factor)*pipe.vae.config.scaling_factor
                latents_target = pipe._pack_latents(latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
                latents_target_1 = latents_target



                idx = 80
                timestep = timesteps[idx].expand(latents.shape[0]).to(latents.dtype)
                torch.manual_seed(2)
                # Create a linear combination of torch normal and latents_target_1
                blend_ratio = idx / 100.0
                # Generate random noise with the same shape as latents_target_1
                random_noise = torch.randn_like(latents_target_1).to(latents.device).to(latents.dtype)
                # Create the linear combination
                V_test = 1*(1-blend_ratio) * (random_noise)  + (blend_ratio) * latents_target_1

                noise_pred_test = pipe.transformer(
                    hidden_states= V_test,
                    timestep=timestep / 1000,  # Dividing timestep by 1000 for testing as per your code
                    guidance=None,
                    pooled_projections=pipe.pooled_prompt_embeds,
                    encoder_hidden_states=pipe.prompt_embeds,
                    txt_ids=pipe.text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=pipe.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                V_predict_test = V_test + (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred_test
                image_sde = display_latent(V_predict_test)
                #image_sde.save(f'/home/zy3724/4Dprojects/4Dprojects/harmonydreamer/local_visualization_cache/predict_{t}.png')

            sde_edited_images.append(image_sde)

    return sde_edited_images


