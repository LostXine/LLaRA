import numpy as np
import argparse

from scipy.spatial.transform import Rotation as R
from vima_utils import prepare_prompt, model_generation, eval_episode
from model_maskrcnn import TwoHeadRoIHeads, TwoHeadsMaskRCNN, TwoHeadsFastRCNNPredictor

def quantize_float_to_text(num: np.ndarray, tokenizer, token_pool_start=31000, token_num=256):
    """
    Quantizes normalized float values to a discrete token string.
    """
    assert np.all(num <= 1) and np.all(num >= 0) 
    ids = token_pool_start + np.round(num * (token_num - 1))
    return tokenizer.decode(ids.astype(int))


def text_to_float(s: str, tokenizer, token_pool_start=31000, token_num=256):
    """
    Converts a token string back to normalized float values.
    """
    ids = tokenizer.encode(s)
    return [float(i - token_pool_start) / (token_num - 1)
            for i in ids if token_pool_start <= i < token_pool_start + token_num]


def action_to_text(action: dict, tokenizer) -> str:
    """
    Converts an action dictionary to a quantized string using a tokenizer.
    """
    act_pos_pick = action['pose0_position']
    act_pos_place = action['pose1_position']
    act_rot_pick = action['pose0_rotation']
    act_rot_place = action['pose1_rotation']

    euler1 = R.from_quat(act_rot_pick).as_euler('xyz', degrees=True)
    euler2 = R.from_quat(act_rot_place).as_euler('xyz', degrees=True)
    euler_z_diff = int(euler2[-1] - euler1[-1])  # Z-axis rotation difference

    # Normalize values to [0, 1] range
    norm_action = [
        (act_pos_pick[0] - 0.25) * 2,       # X pick: [0.25, 0.75] -> [0, 1]
        act_pos_pick[1] + 0.5,              # Y pick: [-0.5, 0.5] -> [0, 1]
        (act_pos_place[0] - 0.25) * 2,      # X place
        act_pos_place[1] + 0.5,             # Y place
        (float(euler_z_diff) + 180) / 360    # Rotation: [-180, 180] -> [0, 1]
    ]

    return quantize_float_to_text(np.array(norm_action), tokenizer)


def query_rt2(tokenizer, model, image_processor, prompt, prompt_mode, prompt_assets, action_hist, obs, detector, prompt_id):
    """
    Generates quantized text representation of actions using RT-2 style prompting.
    """
    spatula = obs['ee'] > 0
    prepared_prompt, image_list = prepare_prompt(
        tokenizer, model, image_processor, prompt,
        mode=prompt_mode, prompt_assets=prompt_assets,
        spatula=spatula, detector=detector, uid=prompt_id
    )

    if 'h' in prompt_mode and action_hist:
        prepared_prompt += '\nYou have finished: '
        prepared_prompt += '\n'.join([
            f'Step {i + 1}: {action_to_text(act, tokenizer)}'
            for i, act in enumerate(action_hist)
        ])

    image_list.append(obs['rgb']['front'].copy())
    ans, _ = model_generation(tokenizer, model, image_processor, image_list, prepared_prompt)

    norm_acts = text_to_float(ans, tokenizer)
    m_len = len(norm_acts) // 5
    if 's' in prompt_mode:
        m_len = min(m_len, 1)

    return [norm_acts[i * 5: i * 5 + 5] for i in range(m_len)], prepared_prompt, ans


def parse_rt2(pick_place_point):
    """
    Converts normalized float values back into a structured action dictionary.
    """
    def unnorm(v):
        return [v[0] * 0.5 + 0.25, v[1] - 0.5]

    action = {
        'pose0_position': np.array(unnorm(pick_place_point[:2]), dtype=np.float32),
        'pose1_position': np.array(unnorm(pick_place_point[2:4]), dtype=np.float32),
    }

    rotation = float(pick_place_point[4]) * 360 - 180
    rot_quat = R.from_euler('z', rotation, degrees=True).as_quat()
    action['pose1_rotation'] = np.array(rot_quat, dtype=np.float32)

    return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RT-2 style evaluation using quantized text tokens.')

    # Required
    parser.add_argument('filename', help='Name of the output file.')

    # Paths
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the RT-2 compatible model checkpoint.')
    parser.add_argument('--output-path', type=str, default='../results/',
                        help='Path to the output directory (default: ../results/).')

    # Prompt options
    parser.add_argument('--prompt-mode', type=str, default='hs',
                        help=(
                            'Set of operational flags to customize prompt behavior:\n'
                            '  h: Enable action history.\n'
                            '  s: Query VLM for each observation and only perform a single step.\n'
                            '  d: Enable object detection using VLM.\n'
                            '  e: Enable object detection using MaskRCNN.\n'
                            '  o: Enable oracle object detection.'
                        ))

    # Prompt selection options
    parser.add_argument('--prompt-id', type=int, default=-1,
                        help=(
                            'Which prompt to use when generating actions:\n'
                            '  -1 or any negative value: Randomly select from 15 available prompts (default).\n'
                            '  0 to 14: Use the fixed prompt at that index.\n'
                            '  100 or any number >14: Skip prompt.'
                        ))
    
    # Evaluation control
    parser.add_argument('--seed', type=int, default=200000,
                        help='Random seed for reproducibility.')
    parser.add_argument('--num-env', type=int, default=20,
                        help='Number of episodes per task.')
    parser.add_argument('--max-length', type=int, default=8,
                        help='Maximum number of steps per episode (episodes exceeding this are considered failures).')
    parser.add_argument('--partition', type=str, default='',
                        help='Specific VIMABench partition to test (L1–L4). Tests all if unspecified.')

    # Detector options
    parser.add_argument('--detector', type=str, default='',
                        help='Path to the MaskRCNN checkpoint. Only used if "e" is enabled in prompt-mode.')
    parser.add_argument('--detector-thre', type=float, default=0.6,
                        help='Minimum score threshold for valid object detection proposals (default: 0.6).')

    args = parser.parse_args()

    assert 'rt' in args.model_path.lower(), "This script is only compatible with RT-2 models."

    eval_episode(args, query_rt2, parse_rt2)
