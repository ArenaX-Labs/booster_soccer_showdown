import argparse
import time
import numpy as np
import mujoco
from mujoco import viewer
from huggingface_hub import hf_hub_download
import pathlib

HERE = pathlib.Path(__file__).parent
def main():
    parser = argparse.ArgumentParser(description="Play qpos from NPZ in MuJoCo.")
    parser.add_argument(
        "--robot",
        choices=["booster_t1", "booster_lower_t1"],
        default="booster_t1",
    )
    parser.add_argument("--npz", required=True, help="Path to .npz with qpos (T, nq).")
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS (overrides any 'fps' in the NPZ).")
    args = parser.parse_args()

    # --- Load trajectory ---
    try:
        data_npz = np.load(args.npz, allow_pickle=False)
    except:
        file_name = hf_hub_download(
                    repo_id="SaiResearch/booster_dataset",
                    filename=f"soccer/{args.robot}/{args.npz}",
                    repo_type="dataset")
        data_npz = np.load(file_name, allow_pickle=False)
        
    key = "qpos"

    if key not in data_npz:
        raise KeyError(f"'{key}' not found in {args.npz}. Available: {list(data_npz.keys())}")
    
    qpos_traj = np.array(data_npz[key], dtype=float)  # (T, nq)

    if qpos_traj.ndim != 2:
        raise ValueError(f"qpos must be 2D (T, nq). Got shape {qpos_traj.shape}")

    # Optional fps in file
    file_fps = float(data_npz["fps"]) if ("fps" in data_npz and args.fps is None) else None
    fps = args.fps if args.fps is not None else (file_fps if file_fps and file_fps > 0 else 30.0)
    dt_frame = 1.0 / fps

    # --- Load model & data ---
    model = mujoco.MjModel.from_xml_path(f"{HERE}/assets/booster_t1/{args.robot}.xml")
    data = mujoco.MjData(model)

    T, nq = qpos_traj.shape
    if nq != model.nq:
        raise ValueError(f"qpos width ({nq}) != model.nq ({model.nq}).")

    # Start from first pose
    data.qpos[:] = qpos_traj[0]
    mujoco.mj_forward(model, data)

    # --- Launch viewer and play ---
    print(f"Playing {T} frames at {fps:.2f} FPS...")
    start_time = time.time()

    with viewer.launch_passive(model, data) as v:
        print(T)
        for t in range(T):

            # Set pose and forward
            data.qpos[:] = qpos_traj[t]
            mujoco.mj_forward(model, data)

            # Render & pace to FPS
            v.sync()
            # Wall-clock pacing (simple)
            target = start_time + (t + 1) * dt_frame
            now = time.time()
            if target > now:
                time.sleep(target - now)

    print("Done.")

if __name__ == "__main__":
    main()
 