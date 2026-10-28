# 파일: map_source_clips.py
import json, pathlib, shutil, tqdm

DATA_ROOT = pathlib.Path("/mnt/d/agent_projects/video_grounding/UniTime/unitime_datasets")
SOURCE_ROOT = pathlib.Path("/mnt/d/agent_projects/video_grounding/UniTime/unitime_sources")
OUT_ROOT = pathlib.Path("/mnt/d/agent_projects/video_grounding/UniTime/unitime_mapped")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def load_json(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def find_clip_file(source_name, clip_uid):
    """
    각 레포의 일반적인 클립 위치 패턴:
    - ego4d   : clips/{clip_uid}.mp4
    - anet    : videos/{clip_uid}.mp4
    - charades: clips/{clip_uid}.mp4
    - qvhl    : clips/{clip_uid}.mp4
    - tacos   : clips/{clip_uid}.mp4
    """
    patterns = {
        "ego4d":   f"clips/{clip_uid}.mp4",
        "anet":    f"videos/{clip_uid}.mp4",
        "charades":f"clips/{clip_uid}.mp4",
        "qvhl":    f"clips/{clip_uid}.mp4",
        "tacos":   f"clips/{clip_uid}.mp4",
        "pretrain":f"clips/{clip_uid}.mp4",
    }
    rel_path = patterns.get(source_name)
    if not rel_path:
        return None
    cand = SOURCE_ROOT / source_name / rel_path
    return cand if cand.is_file() else None

def main():
    for split in ("train", "val", "test"):
        json_path = DATA_ROOT / f"nlq_{split}.json"
        data = load_json(json_path)
        for video in tqdm.tqdm(data["videos"], desc=f"Mapping {split}"):
            for clip in video["clips"]:
                src_uid = clip["source_clip_uid"]
                # 원본 종류는 video_uid 에서 추론 가능 (예: ego4d, anet …)
                # 여기서는 간단히 `video_uid` 앞 4글자를 사용
                source_name = video["video_uid"][:4].lower()
                src_file = find_clip_file(source_name, src_uid)
                if src_file:
                    dst_dir = OUT_ROOT / split / video["video_uid"] / clip["clip_uid"]
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_dir / src_file.name)
                else:
                    tqdm.tqdm.write(f"⚠️ Missing: {src_uid} (src={source_name})")

if __name__ == "__main__":
    main()
