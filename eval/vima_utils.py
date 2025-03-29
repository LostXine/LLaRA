import os
import re
import json
import random
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2 as T
from vima_bench import make, ALL_PARTITIONS, PARTITION_TO_SPECS
from llara_prompts import action_prompts, detection_prompts, localization_prompts
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


# -------------------------------
# Utility Functions
# -------------------------------

def save_json(path: str, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def pix2pos_front(px, py):
    j, i = float(px - 3) / 251 - 0.5, float(py - 34) / 178 + 0.25
    return np.clip(i, 0.25, 0.75), np.clip(j, -0.5, 0.5)

def pos2pix_front(i, j):
    return int((j + 0.5) * 251 + 3), int((i - 0.25) * 178 + 34)

def get_bounding_box(mask):
    if np.sum(mask) == 0:
        return None
    rows, cols = np.where(mask)
    return (np.min(rows), np.min(cols), np.max(rows), np.max(cols))

def xyxy_to_normalized_bbox(x1, y1, x2, y2, mask):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h = x2 - x1, y2 - y1
    return cx / mask.shape[1], cy / mask.shape[0], w / mask.shape[1], h / mask.shape[0]

def get_center_bbox_from_obs(mask):
    y1, x1, y2, x2 = get_bounding_box(mask)
    return xyxy_to_normalized_bbox(x1, y1, x2, y2, mask)

def parse_coordinate(coord_str: str):
    x_str, y_str = coord_str[1:-1].split(',')
    return float(x_str) * 256, float(y_str) * 128

def format_bbox(bbox):
    cx, cy, w, h = bbox
    return f'<b>({cx:.3f}, {cy:.3f}), {{{w:.3f}, {h:.3f}}}</b>'

def format_object_description(obj):
    return '<p>' + (obj.get('obj_color') or obj.get('texture_name')) + ' ' + obj['obj_name'] + '</p>'

def describe_object_list(obj_list):
    return '\n'.join([f"{desc} at {format_bbox(bbox)}." for bbox, desc in obj_list])

# -------------------------------
# Core Evaluation Logic
# -------------------------------

def model_generation(tokenizer, model, image_processor, image_list, prompt):
    images = [Image.fromarray(np.moveaxis(img, 0, -1)) for img in image_list]
    image_sizes = [img.size for img in images]
    image_tensor = process_images(images, image_processor, model.config).half()

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=256,
            use_cache=True
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    conv.messages[-1][-1] = output_text
    return output_text, conv

def object_detector_inference(image, detector):
    assert detector
    model, transform, device, cls_data, threshold = (
        detector['obj_model'], detector['tf'], detector['device'], detector['cls_data'], detector['detector_thre']
    )

    with torch.no_grad():
        transformed_image = transform(torch.Tensor(image.copy())) / 255
        results = model([transformed_image.to(device)])

    objects = []
    for i, mask in enumerate(results[0]['masks']):
        score = results[0]['scores'][i].item()
        if score < threshold:
            continue
        label = cls_data['color'][results[0]['labels'][i].item()]
        second_label = cls_data['cls'][results[0]['second_labels'][i].item()]
        bbox = results[0]['boxes'][i].cpu().numpy() # x1, y1, x2, y2
        name = format_object_description({'obj_color': label, 'obj_name': second_label})
        box_str = format_bbox(xyxy_to_normalized_bbox(*bbox, mask[0]))
        objects.append(f"{name} at {box_str}")
    return objects

def prepare_prompt(tokenizer, model, image_processor, prompt: str, mode: str,
                   prompt_assets: dict = {}, spatula: bool = False,
                   detector=None, uid: int = -1) -> tuple[str, list]:
    """
    Prepares the prompt string for the VLM model and gathers related images.

    Args:
        tokenizer, model, image_processor: LLaVA model components.
        prompt (str): Original text prompt with placeholders like {obj_name}.
        mode (str): Prompt mode string (e.g. "h", "s", "d", "e", "o").
        prompt_assets (dict): Dict containing reference images and segmentations.
        spatula (bool): Whether the robot uses a spatula (vs suction).
        detector (dict or None): Object detection model (if applicable).
        uid (int): Prompt template index. If <0, choose randomly.

    Returns:
        Tuple[str, list]: Processed prompt string and corresponding image list.
    """
    image_list = []
    image_idx = 0
    refer_objs = re.findall(r'\{.+?\}', prompt)

    for ref_obj in refer_objs:
        obj_key = ref_obj[1:-1]
        asset = prompt_assets[obj_key]
        obj_info = asset['segm']['obj_info']
        ref_image = asset['rgb']['front']

        # Multiple objects (scene)
        if isinstance(obj_info, list):
            obj_desc = obj_key
            if 'e' in mode:
                obj_desc = '\n'.join(object_detector_inference(ref_image, detector))
            elif 'd' in mode:
                det_prompt = '<image>\n' + random.choice(detection_prompts)
                obj_desc, _ = model_generation(tokenizer, model, image_processor, [ref_image], det_prompt)
            elif 'o' in mode:
                objs = [(get_center_bbox_from_obs(np.array(asset['segm']['front']) == int(j['obj_id'])), format_object_description(j)) for j in obj_info]
                obj_desc = describe_object_list(objs)

        # Single object
        else:
            obj_desc = format_object_description(obj_info)
            if 'e' in mode:
                candidates = object_detector_inference(ref_image, detector)
                obj_desc = next((o for o in candidates if obj_desc.lower() in o.lower()), obj_desc)
            elif 'd' in mode:
                loc_prompt = '<image>\n' + random.choice(localization_prompts).replace('{object}', obj_desc)
                obj_desc, _ = model_generation(tokenizer, model, image_processor, [ref_image], loc_prompt)
            elif 'o' in mode:
                obj = [(get_center_bbox_from_obs(np.array(asset['segm']['front']) == int(obj_info['obj_id'])), format_object_description(obj_info))]
                obj_desc = describe_object_list(obj)

        # Add image tag if needed
        if isinstance(obj_info, list):
            obj_desc = '<scene>' + re.sub('<.*?scene>', '', obj_desc) + '</scene>'
        if obj_desc.endswith('.'):
            obj_desc = obj_desc[:-1]
        if 'v' in mode:
            obj_desc = f'<image{image_idx}>\n' + obj_desc
            image_list.append(ref_image.copy())
            image_idx += 1

        prompt = prompt.replace(ref_obj, obj_desc)

    task_prompt = f'<task>{prompt}</task>'

    # Select user instruction
    if uid < 0:
        user_prompt = random.choice(action_prompts)
    elif uid < len(action_prompts):
        user_prompt = action_prompts[uid]
    else:
        user_prompt = ''

    # Format spec for position + rotation encoding
    format_prompt = "Every action you take must include two locations in the format of <b>(x, y)</b> and one clockwise rotation angle in the format of <r>[r]</r>. "
    if spatula:
        format_prompt += "The first location is the image coordinate where you start to sweep the object using a spatula, and the second location is where you stop sweeping. "
        format_prompt += "The image coordinate ranges from 0 to 1. The rotation angle indicates how many degrees you rotate the spatula clockwise, and it ranges from -359 to 359."
    else:
        format_prompt += "The first location is the image coordinate where you use a suction cup to pick up the object, and the second location is where you place the object."
        format_prompt += "The image coordinate ranges from 0 to 1. The rotation angle indicates how many degrees you rotate the object clockwise, and it ranges from -359 to 359."

    # Final prompt formatting
    if len(user_prompt) == 0:
        full_prompt = '\n'.join([f'<image{image_idx}>', task_prompt, format_prompt])
    else:
        full_prompt = '\n'.join([f'<image{image_idx}>', task_prompt, user_prompt, format_prompt])

    return full_prompt, image_list


def eval_episode(args, gen_action_fn, parse_action_fn):
    """
    Main evaluation loop for running VIMA+LLaVA-based task policy inference.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        gen_action_fn (callable): Function to generate high-level action from prompt.
        parse_action_fn (callable): Function to convert VLM output to structured action.
    """
    seed_range = range(args.seed, args.seed + args.num_env)

    # Load detector if needed
    detector = None
    if 'e' in args.prompt_mode:
        if not os.path.exists(args.detector):
            print(f'Could not find object detector checkpoint at {args.detector}')
            return

        cls_data = json.load(open('classes.json'))
        tf = T.Compose([T.ToDtype(torch.float, scale=True), T.ToPureTensor()])
        detector = {
            'obj_model': torch.load(args.detector).to('cuda').eval(),
            'tf': tf,
            'device': torch.device('cuda'),
            'cls_data': cls_data,
            'detector_thre': args.detector_thre
        }

    # Load LLaVA model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name, use_flash_attn=True
    )

    # Determine output filename
    if args.prompt_id < 0:
        prompt_suffix = args.prompt_mode
    elif args.prompt_id >= len(action_prompts):
        prompt_suffix = args.prompt_mode + '_no'
    else:
        prompt_suffix = args.prompt_mode + f'{args.prompt_id:02d}'

    partition_list = [args.partition] if args.partition else ALL_PARTITIONS
    filename = f"[{prompt_suffix}]{args.filename}"
    if args.partition:
        filename = f"({args.partition}){filename}"
    filename = os.path.join(args.output_path, filename)
    if not filename.endswith('.json'):
        filename += '.json'

    # Load or initialize results file
    try:
        result = json.load(open(filename))
    except FileNotFoundError:
        result = {}
    result['global'] = vars(args)
    save_json(filename, result)

    for partition in partition_list:
        for task in tqdm(PARTITION_TO_SPECS["test"][partition].keys(), desc=f'Partition: {partition}'):
            for seed in seed_range:
                tid = f'{partition}/{task}/{seed}'
                if tid in result:
                    continue

                # Set seeds
                random.seed(seed)
                np.random.seed(seed)

                # Setup environment
                env = make(task, task_kwargs=PARTITION_TO_SPECS["test"][partition][task], seed=seed)
                obs = env.reset()
                prompt, prompt_assets = env.prompt, env.prompt_assets

                prompt_hist, answer_hist, action_hist = [], [], []
                action_queue = []

                for step in range(args.max_length):
                    retry = 0
                    while not action_queue and retry < 1:
                        parsed_action, prompt_txt, answer_txt = gen_action_fn(
                            tokenizer, model, image_processor, prompt,
                            args.prompt_mode, prompt_assets, action_hist,
                            obs, detector, args.prompt_id
                        )
                        action_queue.extend(parsed_action)
                        prompt_hist.append(prompt_txt)
                        answer_hist.append(answer_txt)
                        retry += 1

                    # If no valid action, fallback to random sample
                    if not action_queue:
                        action = env.action_space.sample()
                        del action['pose1_rotation']
                    else:
                        try:
                            action = parse_action_fn(action_queue.pop(0))
                        except Exception:
                            action = env.action_space.sample()
                            del action['pose1_rotation']

                    action['pose0_rotation'] = np.array([0, 0, 0, 1], dtype=np.float32)
                    if 'pose1_rotation' not in action:
                        action['pose1_rotation'] = np.array([0, 0, 0, 1], dtype=np.float32)

                    action_hist.append(action)
                    obs, reward, done, info = env.step(action)
                    if info['success'] or info['failure']:
                        break

                env.close()

                result[tid] = {
                    'tid': tid,
                    'level': partition,
                    'task': task,
                    'seed': seed,
                    'prompt': prompt,
                    'lm_prompt_hist': prompt_hist,
                    'lm_answer_hist': answer_hist,
                    'step': step + 1,
                    'success': info['success'],
                    'failure': info['failure']
                }
                save_json(filename, result)
