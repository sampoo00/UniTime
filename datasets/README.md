목표
unitime_datasets 에 있는 UniTime‑Data 를 그대로 쓰는 것뿐 아니라, 각 어노테이션이 가리키는 원본 비디오/클립을 실제로 확보하고 싶으신 거죠.
nlq_*.json 파일 안의 source_clip_uid 는 원본 데이터셋(ego4d, ANet, Charades, QVHL, Tacos 등) 안에 존재하는 클립을 의미합니다.
아래 단계대로 진행하면

원본 데이터셋(재료) 다운로드
JSON‑파일을 파싱해 source_clip_uid 로 해당 클립을 매핑
필요 시 로컬에 복사 / 구조화
을 손쉽게 할 수 있습니다.

1️⃣ 원본 데이터셋 원본 URL 파악하기
unitime_datasets 의 최상위 metadata 필드에 원본 레포가 적혀 있습니다.

json
{
  "metadata": "s3://ego4d-consortium-sharing/public/v2/ego4d.json",
  ...
}
위와 같이 s3://… 혹은 HuggingFace datasets 레포가 명시돼 있습니다.
대부분은 아래와 같은 HuggingFace 레포에 매핑됩니다.

원본 레포 (HuggingFace)	레포 ID (repo‑type = dataset)	비고
Ego4D	ego4d/ego4d	영상·클립 메타데이터
ActivityNet (ANet)	activitynet/anet	비디오/클립 메타
Charades	charades/charades	
QVHL (QV-Holder‑Lang)	qvhl/qvhl	
Tacos	tacos/tacos	
Pretrain (large text‑video)	zeqianli/UniTime-Pretrain	
Tip – 실제 레포가 존재하지 않을 경우(프라이빗/게이트) → HuggingFace에 로그인(huggingface-cli login)하고 Access Token을 발급받아야 합니다.

원본 레포를 확인하는 방법
bash
# 예시: Ego4D 메타데이터 확인
hf download ego4d/ego4d --repo-type dataset --local-dir temp_ego4d
hf download 로 성공하면 해당 레포가 공개·접근 가능한 상태라는 의미입니다.

2️⃣ 전체 원본 레포 한 번에 내려받기
아래 스크립트를 프로젝트 루트(UniTime/)에 저장하고 실행하면, 필요한 모든 원본 레포를 unitime_sources/ 폴더에 한 번에 받아옵니다.

bash
# 파일: download_original_sources.sh
#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(pwd)/unitime_sources"
mkdir -p "$BASE_DIR"
declare -A REPOS=(
  ["ego4d"]="ego4d/ego4d"
  ["anet"]="activitynet/anet"
  ["charades"]="charades/charades"
  ["qvhl"]="qvhl/qvhl"
  ["tacos"]="tacos/tacos"
  ["pretrain"]="zeqianli/UniTime-Pretrain"
)
for name repo in "${!REPOS[@]}"; do
  echo "🔽 Downloading $name → ${REPOS[$name]}"
  hf download "${REPOS[$name]}" --repo-type dataset \
      --local-dir "$BASE_DIR/$name"
done
echo "✅ All sources downloaded under $BASE_DIR"
bash
chmod +x download_original_sources.sh
./download_original_sources.sh
핵심: --repo-type dataset 를 반드시 명시합니다. (model 로 인식하면 404 오류가 발생합니다.)

3️⃣ source_clip_uid → 실제 파일 매핑하기
nlq_*.json 에는 source_clip_uid 만 들어 있습니다. 원본 레포 안에서는 보통 clips/{clip_uid}.mp4 혹은 videos/{video_uid}.mp4 와 같은 경로 구조를 사용합니다.
아래 파이썬 스크립트는 각 어노테이션을 읽어 원본 파일 경로를 찾아 복사해 줍니다.

python
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
실행 방법

bash
python map_source_clips.py
source_name 추론 로직은 프로젝트에 맞게 바꿔 주세요.
(video_uid 가 216e3f0e‑… 형태라면, 사전에 video_uid ↔ source 매핑 테이블을 만든 뒤 사용하면 정확합니다.)

4️⃣ 전체 파이프라인 요약
HuggingFace 로그인 (필요 시) → hf login
원본 레포 한 번에 다운로드 (download_original_sources.sh)
JSON → 클립 매핑 (map_source_clips.py)
결과물은
/mnt/d/agent_projects/video_grounding/UniTime/unitime_mapped/
├─ train/
│   ├─ <video_uid>/
│   │   ├─ <clip_uid>/clip.mp4
│   │   └─ …
├─ val/
└─ test/
와 같이 구조화됩니다. 이후 train.sh 혹은 inference.py 에서 unitime_mapped/ 를 데이터 루트로 지정하면, 원본 비디오와 UniTime 어노테이션을 동시에 사용할 수 있습니다.

5️⃣ 자주 묻는 질문 (FAQ)
질문	답변
레포가 비공개라 다운로드가 안돼요	huggingface-cli login 후 export HF_TOKEN=your_token 를 설정하고 다시 실행합니다.
source_clip_uid 로도 파일을 못 찾겠어요	레포마다 파일 구조가 다를 수 있습니다. ls unitime_sources/<source_name> 로 실제 폴더와 파일명을 확인하고 find_clip_file() 에 패턴을 맞춰 주세요.
디스크 용량이 부족해요	--cache-dir 옵션을 이용해 ~/.cache/huggingface 로 캐시를 옮기거나, 필요 없는 split(test 등)만 선택적으로 다운로드 (hf download … --local-dir … --download-mode force_ignore) 가능합니다.
스크립트 실행 중 Permission denied	실행 파일에 실행 권한을 부여 (chmod +x download_original_sources.sh)하고, 필요한 경우 sudo 없이 현재 사용자 권한 안에서 작업하세요.
6️⃣ 바로 실행해 보기
bash
# 1️⃣ 로그인 (필요 시)
huggingface-cli login   # 토큰 입력
# 2️⃣ 원본 데이터셋 전체 다운로드
./download_original_sources.sh
# 3️⃣ 어노테이션 ↔ 원본 클립 매핑
python map_source_clips.py
이렇게 하면 각 데이터셋의 원본 비디오와 UniTime‑Data 어노테이션이 1:1로 연결된 로컬 폴더가 완성됩니다. 필요한 경우 train.sh 스크립트에 --data_root unitime_mapped 옵션을 추가해 학습·평가 파이프라인을 그대로 진행하시면 됩니다. 🚀