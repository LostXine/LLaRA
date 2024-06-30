import numpy as np
import argparse
import re

from scipy.spatial.transform import Rotation as R
from model import *
from vima_utils import *


def action_to_text(obs_img: np.ndarray, action: dict, spatula:bool) -> str:
    act_pos_start = action['pose0_position']
    act_pos_end = action['pose1_position']
    act_rot_start = action['pose0_rotation']
    act_rot_end = action['pose1_rotation']
    # Convert to Euler angles (in radians)
    e1 = R.from_quat(act_rot_start).as_euler('xyz', degrees=True)
    e2 = R.from_quat(act_rot_end).as_euler('xyz', degrees=True)

    de = int(e2[-1] - e1[-1])
    
    px, py = pos2pix_front(act_pos_start[0], act_pos_start[1])
    tx, ty = pos2pix_front(act_pos_end[0], act_pos_end[1])

    obj_text = 'object'
    
    assert len(obs_img.shape) == 3
    w = obs_img.shape[2]
    h = obs_img.shape[1]
    if spatula:
        return f'Sweep the {obj_text} at <b>({px / w :.3f}, {py / h: .3f})</b>, rotate <r>[{-de}]</r> degrees, and stop at <b>({tx / w :.3f}, {ty / h :.3f})</b>.'
    else:
        return f'Pick up the {obj_text} at <b>({px / w :.3f}, {py / h: .3f})</b>, rotate <r>[{-de}]</r> degrees, and drop it at <b>({tx / w :.3f}, {ty / h :.3f})</b>.'


def query_bc(tokenizer, model, image_processor, prompt, prompt_mode, prompt_assets, action_hist, obs, detector):
    prepared_prompt, image_list = prepare_prompt(tokenizer, model, image_processor, prompt, mode=prompt_mode, prompt_assets=prompt_assets, spatula=obs['ee'] > 0, detector=detector)
                        
    # tag for action history
    if 'h' in prompt_mode and len(action_hist):
        # append history action to the prompt
        prepared_prompt += '\nYou have finished: '
        prepared_prompt += '\n'.join([f'Step {acti + 1}: ' + action_to_text(obs['rgb']['front'], acth, spatula=obs['ee'] > 0) for acti, acth in enumerate(action_hist)])
    
    image_list.append(obs['rgb']['front'].copy())
    ans, _ = model_generation(tokenizer, model, image_processor, image_list, prepared_prompt)
    
    # parse positions and rotations
    str_actions = re.findall(f'\(.+?,.+?\)', ans)
    str_rotation = re.findall(f'\[(-*\d*)\]', ans)
    m_len = min(len(str_actions) // 2, len(str_rotation))
    if 's' in prompt_mode:
        m_len = min(m_len, 1)
    return [str_actions[idx * 2: idx * 2 + 2] + str_rotation[idx: idx + 1]  for idx in range(m_len)], prepared_prompt, ans
    

def parse_bc(pick_place_point):
    p0 = parse_coor(pick_place_point[0])
    p1 = parse_coor(pick_place_point[1])

    action = {
        'pose0_position': np.array(pix2pos_front(*p0), dtype=np.float32),
        'pose1_position': np.array(pix2pos_front(*p1), dtype=np.float32),
    }
    rotation = -float(pick_place_point[2])
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
    assert 'RT' not in args.model_path
    eval_episode(args, query_bc, parse_bc)
