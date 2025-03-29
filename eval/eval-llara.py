import numpy as np
import argparse
import re

from scipy.spatial.transform import Rotation as R
from vima_utils import pos2pix_front, pix2pos_front, parse_coordinate, prepare_prompt, model_generation, eval_episode
from model_maskrcnn import TwoHeadRoIHeads, TwoHeadsMaskRCNN, TwoHeadsFastRCNNPredictor

def action_to_text(obs_img: np.ndarray, action: dict, spatula: bool) -> str:
    """
    Convert an action in VIMA format (two positions and two rotations) into 
    a human-readable LLaRA text description.

    Args:
        obs_img (np.ndarray): Observation image (3D RGB array).
        action (dict): Dictionary containing start/end positions and rotations.
        spatula (bool): Whether the tool is a spatula (affects text wording).

    Returns:
        str: Natural language description of the action.
    """
    # Extract start and end positions and rotations
    act_pos_start = action['pose0_position']
    act_pos_end = action['pose1_position']
    act_rot_start = action['pose0_rotation']
    act_rot_end = action['pose1_rotation']

    # Convert quaternion rotations to Euler angles (degrees)
    euler1 = R.from_quat(act_rot_start).as_euler('xyz', degrees=True)
    euler2 = R.from_quat(act_rot_end).as_euler('xyz', degrees=True)
    euler_z_diff = int(euler2[-1] - euler1[-1])  # Z-axis rotation difference

    # Convert world positions to pixel coordinates
    px, py = pos2pix_front(act_pos_start[0], act_pos_start[1])
    tx, ty = pos2pix_front(act_pos_end[0], act_pos_end[1])

    # Normalize coordinates by image size
    assert len(obs_img.shape) == 3
    h, w = obs_img.shape[1], obs_img.shape[2]

    obj_text = 'object'

    if spatula:
        return (
            f'Sweep the {obj_text} at <b>({px / w:.3f}, {py / h:.3f})</b>, '
            f'rotate <r>[{-euler_z_diff}]</r> degrees, and stop at <b>({tx / w:.3f}, {ty / h:.3f})</b>.'
        )
    else:
        return (
            f'Pick up the {obj_text} at <b>({px / w:.3f}, {py / h:.3f})</b>, '
            f'rotate <r>[{-euler_z_diff}]</r> degrees, and drop it at <b>({tx / w:.3f}, {ty / h:.3f})</b>.'
        )


def query_bc(tokenizer, model, image_processor, prompt, prompt_mode, prompt_assets, action_hist, obs, detector, prompt_id):
    """
    Generate an action plan based on the prompt and current observation using LLaRA.

    Args:
        tokenizer, model, image_processor: Model components.
        prompt (str): Input instruction.
        prompt_mode (str): Prompt customization flags.
        prompt_assets: Optional media for prompt support.
        action_hist (list): History of previous actions.
        obs (dict): Current observation dictionary.
        detector (str): Object detector configuration.
        prompt_id (int): Prompt unique identifier.

    Returns:
        tuple: Parsed actions, prepared prompt string, raw model output.
    """
    spatula = obs['ee'] > 0
    prepared_prompt, image_list = prepare_prompt(
        tokenizer, model, image_processor, prompt,
        mode=prompt_mode, prompt_assets=prompt_assets,
        spatula=spatula, detector=detector, uid=prompt_id
    )

    # Append action history if prompt mode includes 'h'
    if 'h' in prompt_mode and action_hist:
        prepared_prompt += '\nYou have finished: '
        prepared_prompt += '\n'.join([
            f'Step {i + 1}: {action_to_text(obs["rgb"]["front"], act, spatula=spatula)}'
            for i, act in enumerate(action_hist)
        ])

    # Append current image
    image_list.append(obs['rgb']['front'].copy())

    # Generate model output
    answer, _ = model_generation(tokenizer, model, image_processor, image_list, prepared_prompt)

    # Extract and parse action components
    str_points = re.findall(r'\(.+?,.+?\)', answer)
    str_rotation = re.findall(r'\[(-*\d*)\]', answer)
    m_len = min(len(str_points) // 2, len(str_rotation))
    if 's' in prompt_mode:
        m_len = min(m_len, 1)

    parsed = [
        str_points[i * 2: i * 2 + 2] + str_rotation[i: i + 1]
        for i in range(m_len)
    ]

    return parsed, prepared_prompt, answer


def parse_bc(pick_place_point_and_rotation):
    """
    Parse a pick-and-place + rotation action from text into VIMA format.

    Args:
        pick_place_point_and_rotation (list[str]): List containing start and end coordinate strings, and rotation.

    Returns:
        dict: Action dictionary with structured positions and rotation in quaternion format.
    """
    p0 = parse_coordinate(pick_place_point_and_rotation[0])
    p1 = parse_coordinate(pick_place_point_and_rotation[1])
    
    # Convert pixel to world position
    action = {
        'pose0_position': np.array(pix2pos_front(*p0), dtype=np.float32),
        'pose1_position': np.array(pix2pos_front(*p1), dtype=np.float32),
    }

    # Convert rotation to quaternion
    rotation = -float(pick_place_point_and_rotation[2])
    rot_quat = R.from_euler('z', rotation, degrees=True).as_quat()
    action['pose1_rotation'] = np.array(rot_quat, dtype=np.float32)

    return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a behavior cloning policy using a visual language model.')

    # Required argument
    parser.add_argument('filename', help='Name of the output file.')

    # Model and output paths
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the LLaRA checkpoint.')
    parser.add_argument('--output-path', type=str, default='../results/',
                        help='Path to the output directory (default: ../results/).')

    # Prompt customization options
    parser.add_argument('--prompt-mode', type=str, default='hso',
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

    # Evaluation parameters
    parser.add_argument('--seed', type=int, default=200000,
                        help='Random seed for reproducibility.')
    parser.add_argument('--num-env', type=int, default=20,
                        help='Number of episodes per task.')
    parser.add_argument('--max-length', type=int, default=8,
                        help='Maximum number of steps per episode (episodes exceeding this are considered failures).')

    # Optional benchmarking scope
    parser.add_argument('--partition', type=str, default='',
                        help='Specific VIMABench partition to test (L1–L4). Tests all if unspecified.')

    # Object detection configuration
    parser.add_argument('--detector', type=str, default='',
                        help='Path to the MaskRCNN checkpoint. Only used if "e" is enabled in prompt-mode.')
    parser.add_argument('--detector-thre', type=float, default=0.6,
                        help='Minimum score threshold for valid object detection proposals (default: 0.6).')

    # Parse arguments and run evaluation
    args = parser.parse_args()

    assert 'RT' not in args.model_path, "RT models are not supported."

    eval_episode(args, query_bc, parse_bc)
