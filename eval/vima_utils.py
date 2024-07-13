import json
import numpy as np
import torch
from PIL import Image
import re
import random

from vima_bench import make, ALL_PARTITIONS, PARTITION_TO_SPECS
from tqdm import tqdm
import random
import json
from torchvision.transforms import v2 as T
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


action_prompts = [
        "Could you write down what needs to be done to complete the task on this scene?",
        "List out the actions needed to accomplish the task in this scene.",
        "What actions are necessary to perform the task on this scene?",
        "Can you describe what needs to be done on this scene to complete the task?",
        "What steps are required to perform the task shown in this scene?",
        "List the actions needed to perform the task given below.",
        "On the following scene, could you list what actions are required to perform the task?",
        "Describe what actions are needed on this scene to complete the task.",
        "What do you need to do on this scene to accomplish the task?",
        "List the actions required to perform the task given on this scene.",
        "Could you please describe the steps needed to perform the task on this scene?",
        "Write down the actions required to perform the task on this scene.",
        "Please write down the actions required to perform the task shown below.",
        "Can you explain what needs to be done to perform the task in this scene?",
        "Describe the actions required to complete the task on this scene.",
        ]


detection_prompts = [
    "Identify and describe each object in the image. For each object, list it in the format <b>(x, y), {w, h}</b>, where x and y represent the coordinates of the bounding box center, and w and h represent the width and height of the bounding box. The image coordinates should start from the top left corner and be normalized between 0 and 1.",
    "Catalog all the objects present in the image. For every object, use the format <b>(x, y), {w, h}</b>, with x and y indicating the center of the object's bounding box coordinates, and w and h specifying the width and height. The coordinates are normalized from the top left corner, ranging from 0 to 1.",
    "List each object in the image and describe it. Use the format <b>(x, y), {w, h}</b> for each object, where x and y denote the center coordinates of the bounding box, and w and h are the width and height of the bounding box. The coordinates should start from the top left corner and be normalized to a scale of 0 to 1.",
    "Provide descriptions for all objects within the image. Each object should be listed using the format <b>(x, y), {w, h}</b>, where x and y are the coordinates of the bounding box center, and w and h are the width and height. The coordinates should be normalized, starting from the top left corner, within a range of 0 to 1.",
    "Enumerate and describe every object found in the image. For each object, utilize the format <b>(x, y), {w, h}</b>, where x, y are the bounding box center coordinates and w, h are the dimensions (width and height) of the bounding box. The coordinates begin at the top left corner and are normalized between 0 and 1.",
    "Detail all the objects within the image, listing each one using the format <b>(x, y), {w, h}</b>. Here, x and y represent the coordinates of the bounding box center, while w and h indicate the width and height. The coordinates start from the top left corner and are normalized to the range of 0 to 1.",
    "Document each object present in the image. For each object, use the format <b>(x, y), {w, h}</b>, where x and y are the coordinates of the center of the bounding box, and w and h are the width and height. The coordinates should be normalized, starting from the top left corner, and range from 0 to 1.",
    "For each object in the image, provide a description using the format <b>(x, y), {w, h}</b>. Here, x and y denote the coordinates of the bounding box center, and w and h represent the width and height of the bounding box. The coordinates are normalized to a scale of 0 to 1, starting from the top left corner.",
    "Describe all the objects seen in the image, and list them using the format <b>(x, y), {w, h}</b>. The x and y values are the coordinates for the center of the bounding box, while w and h represent its width and height. The coordinates should be normalized from the top left corner, within a range of 0 to 1.",
    "Identify and list each object found in the image. For each one, use the format <b>(x, y), {w, h}</b>. In this format, x and y are the coordinates for the bounding box center, and w and h are the width and height. The coordinates are to be normalized starting from the top left corner, ranging from 0 to 1.",
    "List and describe each object in the image using the format <b>(x, y), {w, h}</b>. Here, x and y correspond to the coordinates of the bounding box center, and w and h specify the width and height of the bounding box. The coordinates should start from the top left corner and be normalized to the range of 0 to 1.",
    "Provide a description for each object in the image, formatted as <b>(x, y), {w, h}</b>. The x and y values indicate the center coordinates of the bounding box, while w and h represent the width and height. The coordinates start from the top left corner and are normalized between 0 and 1.",
    "Catalog each object within the image, using the format <b>(x, y), {w, h}</b> for each one. In this format, x and y are the coordinates for the center of the bounding box, and w and h are the width and height. The coordinates should be normalized, beginning at the top left corner and ranging from 0 to 1.",
    "Enumerate all the objects in the image, providing descriptions for each using the format <b>(x, y), {w, h}</b>. The x and y values represent the center coordinates of the bounding box, while w and h indicate its width and height. The coordinates are normalized starting from the top left corner, within a range of 0 to 1.",
    "Describe each object in the image, listing them in the format <b>(x, y), {w, h}</b>. Here, x and y denote the center coordinates of the bounding box, and w and h specify the width and height. The coordinates should be normalized from the top left corner, ranging from 0 to 1."
]


localization_prompts = [
    "Where is {object} located in the image? Please use the format <b>(x, y), {w, h}</b> where x and y represent the center coordinates of the bounding box, and w and h are the width and height. The coordinates start from the top left corner and are normalized to a scale of 0 to 1.",
    "Can you provide the location of {object} in the image? Format it as <b>(x, y), {w, h}</b>, with x and y as the center coordinates of the bounding box and w and h as the width and height. The coordinates should begin at the top left corner and be normalized from 0 to 1.",
    "What are the coordinates of {object} in the image? Use the format <b>(x, y), {w, h}</b>, where x and y are the center of the bounding box, and w and h represent the width and height. Coordinates should start at the top left corner and be normalized to a range of 0 to 1.",
    "Please specify the location of {object} in the image. List it in the format <b>(x, y), {w, h}</b>, where x and y denote the bounding box center coordinates, and w and h are the width and height. The coordinates begin from the top left corner and should be normalized to 0 to 1.",
    "What is the position of {object} within the image? Use the format <b>(x, y), {w, h}</b> to describe it, with x and y as the center coordinates of the bounding box, and w and h as the width and height. The coordinates start at the top left corner and are normalized to a scale of 0 to 1.",
    "Describe the location of {object} in the image using the format <b>(x, y), {w, h}</b>. In this format, x and y denote the center coordinates of the bounding box, while w and h represent its width and height. Coordinates should be normalized from the top left corner, ranging from 0 to 1.",
    "Can you detail the location of {object} in the image? Format it as <b>(x, y), {w, h}</b>, where x and y indicate the bounding box center, and w and h represent the width and height. The coordinates should be normalized to a scale of 0 to 1 starting from the top left corner.",
    "Provide the location of {object} in the image using the format <b>(x, y), {w, h}</b>. Here, x and y are the center coordinates of the bounding box, and w and h are the width and height. The coordinates begin at the top left corner and are normalized from 0 to 1.",
    "Where is {object} positioned in the image? Use the format <b>(x, y), {w, h}</b>, where x and y denote the center coordinates of the bounding box, and w and h are the width and height. The coordinates should be normalized to a range of 0 to 1 starting from the top left corner.",
    "Specify the location of {object} in the image in the format <b>(x, y), {w, h}</b>. In this format, x and y represent the bounding box center, and w and h are the width and height. The coordinates should start from the top left corner and be normalized between 0 and 1.",
    "What is the exact position of {object} in the image? Format the coordinates as <b>(x, y), {w, h}</b>, where x and y are the center of the bounding box and w and h denote its width and height. The coordinates start from the top left corner and are normalized to a scale of 0 to 1.",
    "Describe where {object} is located in the image using the format <b>(x, y), {w, h}</b>. Here, x and y indicate the bounding box center coordinates, and w and h specify its width and height. The coordinates should be normalized starting from the top left corner, within the range of 0 to 1.",
    "Could you tell me the location of {object} in the image? Use the format <b>(x, y), {w, h}</b>, where x and y denote the center of the bounding box and w and h are the width and height. Coordinates start at the top left corner and should be normalized between 0 and 1.",
    "Provide the coordinates of {object} in the image in the format <b>(x, y), {w, h}</b>. Here, x and y are the center of the bounding box, while w and h represent its width and height. The coordinates should start from the top left corner and be normalized to 0 to 1.",
    "How is the {object} located in the image? List its coordinates using the format <b>(x, y), {w, h}</b>, where x and y are the center coordinates of the bounding box, and w and h indicate its width and height. The coordinates begin at the top left corner and are normalized to a range of 0 to 1."
]


def save_json(file: str, data):
    with open(file, 'w') as f:
        json.dump(data, f, indent=2)

def pix2pos_front(px, py):
    j, i = float(px - 3) / 251 - 0.5, float(py - 34) / 178 + 0.25
    return np.clip(i, 0.25, 0.75), np.clip(j, -0.5, 0.5)

def pos2pix_front(i, j):
    return int((j + 0.5) * 251 + 3), int((i - 0.25) * 178 + 34)

def get_bounding_box(arr):
    if np.sum(arr) == 0:
        return None
    # Find the indices of the array where the elements are True
    rows, cols = np.where(arr)
    # Determine the bounding box by the minimum and maximum indices
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    # Return the bounding box as (top_left, bottom_right)
    # Format: (row_min, col_min, row_max, col_max)
    return (min_row, min_col, max_row, max_col)

def xyxy2xywh(x1, y1, x2, y2, mask):
    cx, cy, w, h = (x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1
    assert len(mask.shape) == 2
    iw = mask.shape[1]
    ih = mask.shape[0]
    return cx / iw, cy / ih, w / iw, h / ih

def get_center_bbox_from_obs(mask):
    y1, x1, y2, x2 = get_bounding_box(mask)
    return xyxy2xywh(x1, y1, x2, y2, mask)

def obj_format(obj):
    if 'obj_color' in obj:
        txt = obj['obj_color'] + ' ' + obj['obj_name'] 
    else:
        txt = obj['texture_name'] + ' ' + obj['obj_name']
    return '<p>' + txt + '</p>'

def get_obj_list_desc(obj_list):
    return '\n'.join([f'{obj_desc} at {bbox_format(bbox)}.' for bbox, obj_desc in obj_list])

def bbox_format(bbox):
    cx, cy, iw, ih = bbox
    return f'<b>({cx:.3f}, {cy:.3f}), {{{iw:.3f}, {ih:.3f}}}</b>'

def parse_coor(s: str):
    # remove ( )
    l = s[1:-1].split(',')
    assert len(l) == 2
    px, py = float(l[0]), float(l[1])
    px *= 256
    py *= 128
    return px, py

def model_generation(tokenizer, model, image_processor, image_list, inp):
    # move axies and convert to pil image
    image = [Image.fromarray(np.moveaxis(i, 0, -1)) for i in image_list]
    image_size = [i.size for i in image]
    # Similar operation in model_worker.py
    image_tensor = process_images(image, image_processor, model.config).half()
    conv = conv_templates['v1'].copy() # !! use 'v1' as vicuna_v1 since we are trained on this system prompt; llava_v1 is different: user's vs human's
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_size,
            do_sample=False,
            max_new_tokens=256,
            use_cache=True)
        
    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    conv.messages[-1][-1] = outputs
    return outputs, conv

def object_detector_inference(image, detector):
    assert detector
    obj_model = detector['obj_model']
    tf = detector['tf']
    device = detector['device']
    cls_data = detector['cls_data']
    detector_thre = detector['detector_thre']
    with torch.no_grad():
        # this is not ideal because the position of the object in the assert image may not refect
        # the actual position of the object in the observation, this will be fixed in the future version.
        inf_img = tf(torch.Tensor(image.copy())) / 255
        result = obj_model([inf_img.to(device)])
        
    obj_list = []
    for i, mask in enumerate(result[0]['masks']):
        label = cls_data['color'][result[0]['labels'][i].item()]
        second_label = cls_data['cls'][result[0]['second_labels'][i].item()]
        score = result[0]['scores'][i].item()
        box = result[0]['boxes'][i].cpu().numpy() # x1, y1, x2, y2
        
        obj_name = obj_format(
            {
                'obj_color': label,
                'obj_name': second_label}
            )
        # print(obj_name)
        if score < detector_thre:
            continue
        obj_box = bbox_format(xyxy2xywh(box[0], box[1], box[2], box[3], mask[0]))
        obj_list.append(obj_name + ' at ' + obj_box)
    return obj_list

def prepare_prompt(tokenizer, model, image_processor, p:str, mode:str, prompt_assets:dict={}, spatula:bool=False, detector=None, uid=-1) -> str:
        refer_objs = re.findall(r'\{.+?\}', p)
        image_list = []
        image_idx = 0
        
        for ref_obj in refer_objs:
            obj_info = prompt_assets[ref_obj[1:-1]]['segm']['obj_info']
            ref_image = prompt_assets[ref_obj[1:-1]]['rgb']['front']
            if isinstance(obj_info, list):
                obj_desc = ref_obj[1:-1] # multiple objects in the reference image
                if 'e' in mode:
                    obj_list = object_detector_inference(ref_image, detector)
                    obj_desc = '\n'.join(obj_list)
                elif 'd' in mode:
                    det_prompt = '<image>\n' + random.choice(detection_prompts)
                    obj_desc, _ = model_generation(tokenizer, model, image_processor, [prompt_assets[ref_obj[1:-1]]['rgb']['front']], det_prompt)
                elif 'o' in mode:
                    # use oracle object information
                    obj_list = [(get_center_bbox_from_obs(np.array(prompt_assets[ref_obj[1:-1]]['segm']['front']) == int(j['obj_id'])), obj_format(j)) for j in obj_info]
                    obj_desc = get_obj_list_desc(obj_list)
            else:
                obj_desc = obj_format(obj_info)
                if 'e' in mode:
                    obj_list = object_detector_inference(ref_image, detector)
                    for detected_obj in obj_list:
                        if obj_desc.lower() in detected_obj.lower():
                            obj_desc = detected_obj
                            break
                    else:
                        print('Could not find the object: ', obj_desc, obj_list)
                elif 'd' in mode:
                    loc_prompt = '<image>\n' + random.choice(localization_prompts).replace('{object}', obj_format(obj_info))
                    obj_desc, _ = model_generation(tokenizer, model, image_processor, [prompt_assets[ref_obj[1:-1]]['rgb']['front']], loc_prompt)
                elif 'o' in mode:
                    # use oracle object information
                    obj_list = [(get_center_bbox_from_obs(np.array(prompt_assets[ref_obj[1:-1]]['segm']['front']) == int(j['obj_id'])), obj_format(j)) for j in [obj_info]]
                    obj_desc = get_obj_list_desc(obj_list)
            if isinstance(obj_info, list):
                obj_desc = '<scene>' + re.sub('<.*?scene>', '', obj_desc) + '</scene>'
            if obj_desc.endswith('.'):
                obj_desc = obj_desc[:-1]
            if 'v' in mode:
                obj_desc = f'<image{image_idx}>\n' + obj_desc
                image_list.append(prompt_assets[ref_obj[1:-1]]['rgb']['front'].copy())
                image_idx += 1
            p = p.replace(ref_obj, obj_desc)
        task_prompt = f'<task>{p}</task>'
        if uid < 0:
            user_prompt = random.choice(action_prompts)
        elif uid < len(action_prompts):
            user_prompt = action_prompts[uid]
        else:
            user_prompt = ''
        
        format_prompt = "Every action you take must include two locations in the format of <b>(x, y)</b> and one clockwise rotation angle in the format of <r>[r]</r>. "
        if spatula:
            format_prompt += "The first location is the image coordinate where you start to sweep the object using a spatula, and the second location is where you stop sweeping. "
            format_prompt += "The image coordinate ranges from 0 to 1. The rotation angle indicates how many degrees you rotate the spatula clockwise, and it ranges from -359 to 359."
        else:
            format_prompt += "The first location is the image coordinate where you use a suction cup to pick up the object, and the second location is where you place the object."
            format_prompt += "The image coordinate ranges from 0 to 1. The rotation angle indicates how many degrees you rotate the object clockwise, and it ranges from -359 to 359."
        
        assert len(image_list) == image_idx
        if len(user_prompt) == 0:
            return '\n'.join([f'<image{image_idx}>', task_prompt, format_prompt]), image_list
        return '\n'.join([f'<image{image_idx}>', task_prompt, user_prompt, format_prompt]), image_list


def eval_episode(args, gen_action, parse_action):
    seed_start = args.seed
    seed_end = args.seed + args.num_env
    max_episode_length = args.max_length
    prompt_mode = args.prompt_mode
    prompt_id = args.prompt_id
    
    # load external object detector
    if 'e' in prompt_mode:
        obj_det = args.detector
        if not os.path.exists(obj_det):
            print('Could not find the object detector checkpoint at ' + obj_det)
            return
        cls_data = json.load(open('classes.json'))
        device = torch.device('cuda')

        def get_transform():
            transforms = []
            transforms.append(T.ToDtype(torch.float, scale=True))
            transforms.append(T.ToPureTensor())
            return T.Compose(transforms)

        tf = get_transform()
        obj_model = torch.load(obj_det).to(device)
        obj_model.eval()
        
        detector = {
            'obj_model': obj_model,
            'tf': tf,
            'device': device,
            'cls_data': cls_data,
            'detector_thre': args.detector_thre
        }
    else:
        detector = None
        
    # init llava model
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, None, model_name, use_flash_attn=True)
    
    if prompt_id < 0:
        print_prompt_mode = prompt_mode
    elif prompt_id >= len(action_prompts):
        print_prompt_mode = prompt_mode + '_no'
    else:
        print_prompt_mode = prompt_mode + f'{prompt_id:02d}'
    if len(args.partition):
        filename = os.path.join(args.output_path, f'({args.partition})[{print_prompt_mode}]' + args.filename)
        plist = [args.partition]
    else:
        filename = os.path.join(args.output_path,f'[{print_prompt_mode}]' + args.filename)
        plist = ALL_PARTITIONS
    if not filename.endswith('.json'):
        filename = filename + '.json'
    try:
        result = json.load(open(filename, 'r'))
    except FileNotFoundError:
        result = {}

    result['global'] = vars(args)
    print(result['global'])
    save_json(filename, result)
    
    for partition in plist:
        for task in tqdm(PARTITION_TO_SPECS["test"][partition].keys()):
            print('Eval Task: ' + task)
            for seed in range(seed_start, seed_end):
                tid = f'{partition}/{task}/{seed}'
                if tid in result:
                    continue
                
                random.seed(seed)
                np.random.seed(seed)
                env = make(task, task_kwargs=PARTITION_TO_SPECS["test"][partition][task], seed=seed)
                obs = env.reset()
                prompt, prompt_assets = env.prompt, env.prompt_assets
                
                prompt_hist = []
                answer_hist = []
                action_queue = []
                action_hist = []
                
                # print(prompt)
                for i in range(max_episode_length):
                    retry = 0 # retry
                    retry_thre = 1 # if we enable sampling we can retry if the model does not generate desired output
                    while len(action_queue) == 0 and retry < retry_thre:
                        paresed_action, prepared_prompt, ans = gen_action(tokenizer, model, image_processor,
                                                                          prompt, prompt_mode, prompt_assets, action_hist,
                                                                          obs, detector, prompt_id)
                        action_queue.extend(paresed_action)
                        prompt_hist.append(prepared_prompt)# send to VLM to solve the query
                        answer_hist.append(ans)
                        retry += 1
                    
                    if len(action_queue) == 0:
                        action = env.action_space.sample()
                        del action['pose1_rotation']
                    else:
                        pick_place_point = action_queue.pop(0)
                        try:
                            action = parse_action(pick_place_point)
                        except:
                            # if there is any error when parsing the result, randomly sample the action
                            action = env.action_space.sample()
                            del action['pose1_rotation']
                    
                    # set the initial rotation
                    action['pose0_rotation'] = np.array([0, 0, 0, 1], dtype=np.float32)
                    if 'pose1_rotation' not in action:
                        action['pose1_rotation'] = np.array([0, 0, 0, 1], dtype=np.float32)
                    
                    # record action
                    action_hist.append(action)
                    obs, reward, done, info = env.step(action)
                    success = info['success']
                    failure = info['failure']
                    if success or failure:
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
                    'step': i + 1,
                    'success': success,
                    'failure': failure
                }
                save_json(filename, result)
