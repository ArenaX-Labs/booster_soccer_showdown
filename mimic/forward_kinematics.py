#!/usr/bin/env python3
import argparse
import numpy as np
import mujoco
from huggingface_hub import hf_hub_download
import pathlib

HERE = pathlib.Path(__file__).parent

def extract_fk_package(model, qpos, qvel, target_hz=100.0):
    
    qpos = np.atleast_2d(qpos)
    qvel = np.atleast_2d(qvel)
    assert qpos.shape[1] == model.nq, f"qpos shape {qpos.shape}, expected nq={model.nq}"
    assert qvel.shape[1] == model.nv, f"qvel shape {qvel.shape}, expected nv={model.nv}"
    T = qpos.shape[0]

    data = mujoco.MjData(model)

    # Names
    joint_names = np.array(
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}"
         for j in range(model.njnt)], dtype="<U64")
    body_names = np.array(
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"b{b}"
         for b in range(model.nbody)], dtype="<U64")
    site_names = np.array(
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or f"s{s}"
         for s in range(model.nsite)], dtype="<U64")

    njnt = np.array(model.njnt, dtype=np.int64)
    jnt_type = np.array(model.jnt_type, dtype=np.int32)

    body_rootid = np.array(model.body_rootid, dtype=np.int32, copy=True)
    body_weldid = np.array(model.body_weldid, dtype=np.int32, copy=True)
    body_ipos   = np.array(model.body_ipos, dtype=np.float32, copy=True)
    body_iquat  = np.array(model.body_iquat, dtype=np.float32, copy=True)
    site_bodyid = np.array(model.site_bodyid, dtype=np.int32, copy=True)
    site_pos    = np.array(model.site_pos, dtype=np.float32, copy=True)
    site_quat   = np.array(model.site_quat, dtype=np.float32, copy=True)

    frequency = np.array(float(target_hz), dtype=np.float64)
    metadata  = np.array(None, dtype=object)

    xpos        = np.zeros((T, model.nbody, 3), dtype=np.float32)
    xquat       = np.zeros((T, model.nbody, 4), dtype=np.float32)
    cvel        = np.zeros((T, model.nbody, 6), dtype=np.float32)
    subtree_com = np.zeros((T, model.nbody, 3), dtype=np.float32)
    site_xpos   = np.zeros((T, model.nsite, 3), dtype=np.float32)
    site_xmat   = np.zeros((T, model.nsite, 9), dtype=np.float32)

    for t in range(T):
        data.qpos[:] = qpos[t]
        data.qvel[:] = qvel[t]
        mujoco.mj_forward(model, data)

        xpos[t]        = data.xpos
        xquat[t]       = data.xquat
        cvel[t]        = data.cvel
        subtree_com[t] = data.subtree_com
        if model.nsite > 0:
            site_xpos[t] = data.site_xpos
            site_xmat[t] = data.site_xmat

    split_points = np.array([0, T - 1], dtype=np.int32)

    return {
        "xpos": xpos, "xquat": xquat, "cvel": cvel, "subtree_com": subtree_com,
        "site_xpos": site_xpos, "site_xmat": site_xmat,
        "split_points": split_points,
        "joint_names": joint_names, "body_names": body_names, "site_names": site_names,
        "njnt": njnt, "jnt_type": jnt_type,
        "frequency": frequency, "metadata": metadata,
        "nbody": np.array(model.nbody, dtype=object),
        "body_rootid": body_rootid, "body_weldid": body_weldid,
        "body_pos": body_ipos, "body_quat": body_iquat,
        "body_ipos": body_ipos, "body_iquat": body_iquat,
        "nsite": np.array(model.nsite, dtype=object),
        "site_bodyid": site_bodyid, "site_pos": site_pos, "site_quat": site_quat,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=["booster_t1", "booster_lower_t1"],
        default="booster_t1",
    )
    parser.add_argument("--npz", required=True, help="Path to input npz file containing qpos, qvel")
    parser.add_argument("--out", required=False, help="Path to output npz (default: input name + _fk.npz)")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(f"{HERE}/assets/booster_t1/{args.robot}.xml")

    try:
        data_in = np.load(args.npz, allow_pickle=False)
    except:
        file_name = hf_hub_download(
                    repo_id="SaiResearch/booster_dataset",
                    filename=f"soccer/{args.robot}/{args.npz}",
                    repo_type="dataset")
        data_in = np.load(file_name, allow_pickle=False)

    if "qpos" not in data_in or "qvel" not in data_in:
        raise ValueError("Input npz must contain fields 'qpos' and 'qvel'")

    qpos, qvel = data_in["qpos"], data_in["qvel"]
    fk = extract_fk_package(model, qpos, qvel)

    if args.out is not None:
        out_path = args.out or args.npz.replace(".npz", "_fk.npz")
        np.savez(out_path, **fk)
        print(f"✅ Saved forward kinematics to {out_path}")

if __name__ == "__main__":
    main()
