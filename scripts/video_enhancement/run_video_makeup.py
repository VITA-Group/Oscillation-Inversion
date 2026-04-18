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
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set externally
import argparse

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


from diffusers_local.pipelines.flux.pipeline_flux import FluxPipeline

# from huggingface_hub import login

# # Log in using your Hugging Face token
# login(token="YOUR_HF_TOKEN", add_to_git_credential=True)

model_id = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
#pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

#d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')

# def d(im):
#     return d_model.infer_pil(im)




with torch.no_grad(): 
    device = pipe.device







def main(file_path_blur, file_path_blend, prompt_input, idx, save_path, centeridx):
    # (Your existing code here, using the provided arguments instead of hardcoded paths/values)

    from PIL import Image

    with torch.no_grad(): 
        img = Image.open(file_path_blur).convert('RGB')
        img = img.resize((512, 512))
        #img.save(file_path)
        original_width, original_height = img.size
        image_processor = VaeImageProcessor()
        image = image_processor.preprocess(img, height=original_height, width=original_width).to(torch.bfloat16).to('cuda')
        latents = pipe.vae.encode(image)[0].sample()
        latents = (latents - pipe.vae.config.shift_factor)*pipe.vae.config.scaling_factor
        latents_target = pipe._pack_latents(latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
        latents_target_1 = latents_target


        #file_path = os.path.join('/home/zy3724/git_repos/flatten/metric_dataset/recolor/data/wedding/image_blend.png')

        img = Image.open(file_path_blend).convert('RGB')
        img = img.resize((512, 512))
        original_width, original_height = img.size
        image_processor = VaeImageProcessor()
        image = image_processor.preprocess(img, height=original_height, width=original_width).to(torch.bfloat16).to('cuda')
        latents = pipe.vae.encode(image)[0].sample()
        latents = (latents - pipe.vae.config.shift_factor)*pipe.vae.config.scaling_factor
        latents_target = pipe._pack_latents(latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
        latents_target_2 = latents_target



    with torch.no_grad(): 
        height = original_height
        width = original_width

        # pipe prompt preparation
        prompt = prompt_input
        output_type = "pil"
        seed = 42
        testimage = pipe(
            prompt,
            guidance_scale=3.5,
            height = height,
            width = width, 
            output_type="pil",
            num_inference_steps=10, #use a larger number if you are using [dev]
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]
        testimage.save("testimage.png")


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
        image_seq_len = latents.shape[1]
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
        display(image[0])

        return image[0]

    guidance = None





    latents = latents_target_2
    latents_target = torch.clone(latents).to(latents.dtype).to(latents.device)


    with torch.no_grad(): 
        torch.manual_seed(0)
        V = (latents_target).to(latents.dtype).to(latents.device)
        idx = idx

        V_history = []
        noise_pred_history = []
        V_predict_history = []

        
        
        import time

        start_time = time.time()
        for b in range(30):

            print(torch.norm(V))


            timestep = timesteps[idx].to(latents.dtype)
            timestep = timestep.repeat(latents.shape[0])
            print(timestep)


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


            print( '---- below is V --------------------------------------' )
            #display_latent( V )
            # V_ = pipe._unpack_latents(V, height, width, pipe.vae_scale_factor)
            # V_ = (V_ / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            # image = pipe.vae.decode(V_, return_dict=False)[0]
            # image = pipe.image_processor.postprocess(image, output_type=output_type)
            # display(image[0])

            #print( f'---- below is V_predict {len(V_predict_history)-1}------------------------------' )
            # display_latent( V_predict )
            V_predict_ = pipe._unpack_latents(V_predict, height, width, pipe.vae_scale_factor)
            # V_predict_ = (V_predict_ / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            # image = pipe.vae.decode(V_predict_, return_dict=False)[0]
            # image = pipe.image_processor.postprocess(image, output_type=output_type)
            # display(image[0])

            print( '----------------------------------------------------------' )
            print( 'next iteration' )
            print( '----------------------------------------------------------' )


            V = latents_target_1 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred )         # if b == 0:




            # start_mean = 4
            # if b > start_mean:
            #     V_anchor = torch.stack(V_predict_history[1::2]).mean(dim=0)

            #     if b % 2 == 0:
            #         V = latents_target_1 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 
            #     if b % 2 == 1:
            #         V = V_anchor - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 



            # if b % 2 == 0:
            #     V = latents_target_1 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 
            # if b % 2 == 1:
            #     V = latents_target_2 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 



            if b % 3 == 0:
                V = latents_target_1 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 
            if b % 3 == 1:
                V = latents_target_1 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 
            if b % 3 == 2:
                V = latents_target_2 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 


            # if b % 4 == 0:
            #     V = latents_target_1 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 
            # if b % 4 == 1:
            #     V = latents_target_2 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 
            # if b % 4 == 2:
            #     V = latents_target_2 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 
            # if b % 4 == 3:
            #     V = latents_target_2 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 


            # if b % 5== 0:
            #     V = latents_target_1 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 
            # if b % 5 == 1:
            #     V = latents_target_2 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 
            # if b % 5 == 2:
            #     V = latents_target_3 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 
            # if b % 5 == 3:
            #     V = latents_target_1 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred )
            # if b % 5 == 4:
            #     V = latents_target_3 - ( (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx]) * noise_pred ) 
    
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")


    ############### prepare center for reversion #####################################################
    with torch.no_grad(): 
        timestep = timesteps[idx].expand(latents.shape[0]).to(latents.dtype)

        # # Manually specify indices and take elements from V_history
        # selected_indices = [7, 13, 16, 25, 31, 37, 43, 49]  # Example indices
        # V_selected = [V_history[i] for i in selected_indices if i < len(V_history)]
        # # Convert the list to a tensor
        # V_mean = torch.stack(V_selected).mean(dim=0)

        V_mean_list = []
        for centeridx in range(8):

            if centeridx == 0:
                V_even = V_history[0::3]

            if centeridx == 1:
                V_even = V_history[1::3]  

            if centeridx == 2:
                V_even = V_history[2::3]  

            # if centeridx == 0:
            #     V_even = V_history[0::6]

            # if centeridx == 1:
            #     V_even = V_history[1::6]  

            # if centeridx == 2:
            #     V_even = V_history[2::6]  

            if centeridx == 3:
                V_even = V_history[0::3] + V_history[1::3]

            if centeridx == 4:
                V_even = V_history[0::3] + V_history[2::3]

            if centeridx == 5:
                V_even = V_history[2::3] + V_history[3::3]

            if centeridx == 6:
                V_even = V_history[:] 
            # if centeridx == 6:
            #     V_even = V_history[1::6] + V_history[2::6]

            # if centeridx == 7:
            #     V_even = V_history[4::6] + V_history[5::6]

            # if centeridx == 8:
            #     V_even = V_history[2::6] + V_history[5::6]

            V_mean = torch.stack(V_even).mean(dim=0)
            V_mean_list.append(torch.clone(V_mean).detach())
        





    ############### reversion #####################################################
    # 6. Denoising loop
    # with self.progress_bar(total=num_inference_steps) as progress_bar:

    # latents = torch.clone(P_track[-1]).to(latents.device).to(latents.dtype)
    for kkk, V_mean in enumerate(V_mean_list):
        latents = torch.clone(V_mean).to(latents.device).to(latents.dtype)
        with torch.no_grad():
            for i, t in enumerate(timesteps[idx:-1]):
                # if self.interrupt:
                #     continue

                print(f'timestep is {t}')
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # handle guidance
                if pipe.transformer.config.guidance_embeds:
                    guidance = torch.tensor([pipe.guidance_scale], device=device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None



                noise_pred = pipe.transformer(
                    hidden_states=latents,
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



                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                #latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # Upcast to avoid precision issues when computing prev_sample
                latents = latents.to(torch.float32)
                sigma = pipe.scheduler.sigmas[idx+i]
                sigma_next = pipe.scheduler.sigmas[idx+i+1]

                

                # predict for last
                latents_predict = latents + (pipe.scheduler.sigmas[-1] - pipe.scheduler.sigmas[idx+i]) * noise_pred
                latents_predict = latents_predict.to(noise_pred.dtype)
                print(pipe.scheduler.sigmas[-1] , pipe.scheduler.sigmas[i+idx])


                latents = latents + (sigma_next - sigma) * noise_pred
                # Cast sample back to model compatible dtype
                latents = latents.to(noise_pred.dtype)
                print(sigma_next, sigma)

                # print('---')


                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # if callback_on_step_end is not None:
                #     callback_kwargs = {}
                #     for k in callback_on_step_end_tensor_inputs:
                #         callback_kwargs[k] = locals()[k]
                #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                #     latents = callback_outputs.pop("latents", latents)
                #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # # call the callback, if provided
                # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                #     progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        
                latents_ = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
                latents_ = (latents_ / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                image = pipe.vae.decode(latents_, return_dict=False)[0]
                image = pipe.image_processor.postprocess(image, output_type=output_type)
                #display(image[0])

                # print(latents)
                print(' ----------------------------------------------------------------------- ')
                
                latents_predict_ = pipe._unpack_latents(latents_predict, height, width, pipe.vae_scale_factor)
                latents_predict_ = (latents_predict_ / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                image = pipe.vae.decode(latents_predict_, return_dict=False)[0]
                image = pipe.image_processor.postprocess(image, output_type=output_type)
                #display(image[0])
                # Split the save_path into the filename and the extension
                print(save_path)
                save_path_root, save_path_ext = os.path.splitext(save_path)
                image[0].save(f"{save_path_root}_timestep_{int(t)}_center{kkk}{save_path_ext}")

                break


            # Split the save_path into the filename and the extension
            save_path_root, save_path_ext = os.path.splitext(save_path)

            # Append the center{k} suffix before the extension
            save_path_with_center = f"{save_path_root}_center{kkk}{save_path_ext}"

            # Now use this modified path when saving the image
            image[0].save(save_path_with_center)













if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FluxPipeline")
    parser.add_argument('--file_path_blur', type=str, required=True, help="Path to the blurred image file")
    parser.add_argument('--file_path_blend', type=str, required=True, help="Path to the blended image file")
    parser.add_argument('--prompt_input', type=str, required=True, help="Prompt for image generation")
    parser.add_argument('--idx', type=int, required=True, help="Index parameter")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the resulting image")
    parser.add_argument('--centeridx', type=int, required=False, help="Center index for processing")

    args = parser.parse_args()
    
    # Call the main function with arguments
    main(args.file_path_blur, args.file_path_blend, args.prompt_input, args.idx, args.save_path, args.centeridx)
