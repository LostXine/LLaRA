import numpy as np
import argparse

from scipy.spatial.transform import Rotation as R
from model import *
from vima_utils import *


def quantize_float_to_text(num: np.ndarray, tokenizer, token_pool_start=31000, token_num=256):
    assert np.all(num <= 1) and np.all(num >= 0) 
    ids = token_pool_start + np.round(num * (token_num - 1))
    ids = ids.astype(int)
    return tokenizer.decode(ids.astype(int))

def text_to_float(s: str, tokenizer, token_pool_start=31000, token_num=256):
    ids = tokenizer.encode(s)
    return [float(i - token_pool_start) / (token_num - 1) for i in ids if i >= token_pool_start and i < token_pool_start + token_num]


def action_to_text(action: dict, tokenizer) -> str:
    act_pos_start = action['pose0_position']
    act_pos_end = action['pose1_position']
    act_rot_start = action['pose0_rotation']
    act_rot_end = action['pose1_rotation']
    # Convert to Euler angles (in radians)
    e1 = R.from_quat(act_rot_start).as_euler('xyz', degrees=True)
    e2 = R.from_quat(act_rot_end).as_euler('xyz', degrees=True)

    de = int(e2[-1] - e1[-1])
    
    norm_action = []
    norm_action.append((act_pos_start[0] - 0.25) * 2) # 0.25, 0.75 => 0, 1
    norm_action.append(act_pos_start[1] + 0.5) # -0.5, 0.5 => 0, 1
    norm_action.append((act_pos_end[0] - 0.25) * 2) # 0.25, 0.75 => 0, 1
    norm_action.append(act_pos_end[1] + 0.5) # -0.5, 0.5 => 0, 1
    norm_action.append((float(de) + 180) / 360) # -180, 180 => 0, 1
            
    return quantize_float_to_text(np.array(norm_action), tokenizer)


def query_rt2(tokenizer, model, image_processor, prompt, prompt_mode, prompt_assets, action_hist, obs, detector, prompt_id):
    prepared_prompt, image_list = prepare_prompt(tokenizer, model, image_processor, prompt, mode=prompt_mode, prompt_assets=prompt_assets, spatula=obs['ee'] > 0, detector=detector, uid=prompt_id)
    
    # tag for action history
    if 'h' in prompt_mode and len(action_hist):
        # append history action to the prompt
        prepared_prompt += '\nYou have finished: '
        prepared_prompt += '\n'.join([f'Step {acti + 1}: ' + action_to_text(acth, tokenizer) for acti, acth in enumerate(action_hist)])
    
    image_list.append(obs['rgb']['front'].copy())
    ans, _ = model_generation(tokenizer, model, image_processor, image_list, prepared_prompt)
    
    norm_acts = text_to_float(ans, tokenizer)
    m_len = len(norm_acts) // 5
    if 's' in prompt_mode:
        m_len = min(m_len, 1)
        
    return [norm_acts[idx * 5: idx * 5 + 5]  for idx in range(m_len)], prepared_prompt, ans


def parse_rt2(pick_place_point):
    def unnorm(v):
        i = v[0] * 0.5 + 0.25
        j = v[1] - 0.5
        return [i, j]
    
    action = {
        'pose0_position': np.array(unnorm(pick_place_point[:2]), dtype=np.float32),
        'pose1_position': np.array(unnorm(pick_place_point[2:]), dtype=np.float32),
    }
    
    rotation = float(pick_place_point[4]) * 360 - 180
    rot_quat = np.array(R.from_euler('z', rotation, degrees=True).as_quat(), dtype=np.float32)
    action['pose1_rotation'] = rot_quat
    return action
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--output-path', type=str, default='../results/')
    parser.add_argument('--prompt-mode', type=str, default='hs')
    parser.add_argument('--seed', type=int, default=200000)
    parser.add_argument('--num-env', type=int, default=20)
    parser.add_argument('--max-length', type=int, default=8)
    parser.add_argument('--partition', type=str, default='')
    parser.add_argument('--detector', type=str, default='')
    parser.add_argument('--detector-thre', type=float, default=0.6)
    
    args = parser.parse_args()
    assert 'RT' in args.model_path
    eval_episode(args, query_rt2, parse_rt2)
