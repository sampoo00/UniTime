import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 현재 디렉터리를 sys.path에 추가하여 inference 모듈 임포트 가능하도록 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# inference.py의 기존 함수 및 모델 임포트 (기존 함수 재사용)
from inference import (
    PAD_IDX,
    VideoDataset,
    collate_fn,
    extract_time,
    to_window_list_pred,
    to_window_list_pred_vr,
    construct_messages_mr_fps,
    timed_run_inference,
    run_inference,
    print_inference_time_stats,
    reset_inference_time_stats,
    INFERENCE_TIME_STATS,
)
from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor


def process_json_file(model, processor, json_file_path, args, device):
    """
    지정된 JSON 파일의 비디오 샘플들을 로드하여 추론을 수행하고 결과를 저장한다.
    
    Args:
        model: 로드된 Qwen2VLMR 모델
        processor: 로드된 Qwen2VLMRProcessor
        json_file_path: 처리할 .json 파일 경로
        args: 추론 인자 (fps, nf_short, output_dir 등)
        device: 추론 디바이스
    
    Returns:
        list: 생성된 추론 결과 리스트 (실패 시 None)
    """
    resolved_path = json_file_path.strip()

    # 경로 자동 탐색: 입력 경로에 없으면 data/ 하위 경로도 탐색
    if not os.path.exists(resolved_path):
        candidate = os.path.join(CURRENT_DIR, "data", resolved_path)
        if os.path.exists(candidate):
            resolved_path = candidate
        else:
            print(f"\n[오류] 입력한 파일을 찾을 수 없습니다: {json_file_path}")
            return None

    # JSON 데이터 로드
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"\n[오류] JSON 파일 파싱 실패 ({resolved_path}): {e}")
        return None

    # 단일 dict 형태인 경우 list로 래핑
    if isinstance(dataset, dict):
        dataset = [dataset]

    if not isinstance(dataset, list) or len(dataset) == 0:
        print(f"\n[경고] JSON 파일에 처리할 데이터 항목이 없습니다: {resolved_path}")
        return None

    # DataLoader 구성 (기존 inference.py의 Dataset 및 collate_fn 사용)
    video_dataset = VideoDataset(dataset)
    dataloader = DataLoader(
        video_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    reset_inference_time_stats()
    all_results = []

    print(f"\n" + "=" * 65)
    print(f"[*] 추론 시작: {resolved_path} (총 {len(dataset)}개 항목)")
    print(f"=" * 65)

    for data in tqdm(dataloader, desc=f"Processing {os.path.basename(resolved_path)}"):
        print(f"\n=== Processing video: {data.get('id', 'N/A')} ===")
        print(f"QID: {data.get('qid', 'N/A')}")
        print(f"Video ID: {data.get('id', 'N/A')}")
        print(f"Duration: {data.get('duration', 'N/A')}")
        print(f"Video path: {data.get('video_path', 'N/A')}")
        print(f"Feature path: {data.get('feature_path', 'N/A')}")
        print(f"Video start: {data.get('video_start', 0)}")
        print(f"Video end: {data.get('video_end', data.get('duration', 0))}")
        print(f"Retrieval mode: {data.get('retrieval_mode', 'N/A')}")
        if "annos" in data:
            print(f"Number of annotations: {len(data['annos'])}")
            for i, anno in enumerate(data["annos"][:3]):
                print(f"  Annotation {i}: Query='{anno.get('query', '')[:50]}...'")
        print(f"=== End of data info ===\n")

        # 기존 inference.py의 timed_run_inference 함수 실행
        results, execution_time = timed_run_inference(model, processor, data, args, device)
        all_results.extend(results)

    # 결과 JSON 저장
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(resolved_path))[0]
    output_file = os.path.join(args.output_dir, f"{base_name}_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # 실행 시간 통계 저장
    timing_file = os.path.join(args.output_dir, f"{base_name}_timing.json")
    with open(timing_file, "w", encoding="utf-8") as f:
        json.dump(INFERENCE_TIME_STATS, f, indent=4)

    # 예측 요약 출력
    print(f"\n" + "=" * 65)
    print(f"[+] 추론 완료! 결과 파일: {output_file}")
    print(f"=" * 65)
    for res in all_results:
        annos = res.get("annos", [{}])
        gt_window = annos[0].get("window", "N/A") if annos else "N/A"
        query_text = annos[0].get("query", "N/A") if annos else "N/A"
        print(f"  - [QID: {res.get('qid')}] Video: {res.get('id')}")
        print(f"    Query      : {query_text}")
        print(f"    GT Window  : {gt_window}")
        print(f"    Pred Window: {res.get('pred_relevant_windows')}")

    print_inference_time_stats()
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen2-VL inference on Temporal Grounding (Persistent Model / Interactive Wait Mode)"
    )
    parser.add_argument(
        "--model_local_path",
        type=str,
        default="./checkpoints/qwen2vl7B",
        help="Path to the local base model (e.g. checkpoints/qwen2vl7B)",
    )
    parser.add_argument(
        "--model_finetune_path",
        type=str,
        default="./checkpoints/unitime",
        help="Path to the finetuned model checkpoint (e.g. checkpoints/unitime)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save output json files",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Initial JSON data path to process upon startup (optional)",
    )
    parser.add_argument("--video_root", type=str, default=None, help="Root path to videos directory")
    parser.add_argument("--feat_folder", type=str, default="./tmp_feature", help="Folder for cached features")
    parser.add_argument("--fps", type=int, default=2, help="FPS sampling rate")
    parser.add_argument("--clip_length", default=32, type=int, help="Clip length for segmentation")
    parser.add_argument("--nf_short", default=128, type=int, help="Threshold length for short vs long video")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to load model on (default: cuda:0 if available, else cpu)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # -------------------------------------------------------------------------
    # 1. 모델 가중치는 처음에 단 한 번만 로드 (Persistent Model)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"[*] 모델 가중치 로딩 시작 (Device: {device})")
    print(f"    Base model: {args.model_local_path}")
    print(f"    Finetuned : {args.model_finetune_path}")
    print("=" * 65)

    if args.model_finetune_path:
        model = Qwen2VLMRForConditionalGeneration.from_pretrained(
            args.model_finetune_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
    else:
        model = Qwen2VLMRForConditionalGeneration.from_pretrained(
            args.model_local_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
    model.eval()

    processor = Qwen2VLMRProcessor.from_pretrained(args.model_local_path)
    print(f"\n[성공] 모델 및 프로세서 메모리 적재 완료! (가중치 1회 로드 유지)")

    # -------------------------------------------------------------------------
    # 2. 실행 시 인자로 주어진 초기 파일이 있다면 먼저 처리
    # -------------------------------------------------------------------------
    if args.data_path:
        process_json_file(model, processor, args.data_path, args, device)

    # -------------------------------------------------------------------------
    # 3. 입력 .json 파일을 계속 대기하는 인터랙티브 루프
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("[대기 모드] 입력 JSON 파일을 기다립니다.")
    print("  예시 입력: data/test.json 또는 test.json")
    print("  종료 명령: 'q', 'quit', 또는 'exit' 입력")
    print("=" * 65)

    while True:
        try:
            prompt_text = "\n[UniTime] 처리할 JSON 파일 경로를 입력하세요: "
            user_input = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n프로그램을 종료합니다.")
            break

        if not user_input:
            continue

        # 종료 명령 검사
        if user_input.lower() in ["q", "quit", "exit"]:
            print("프로그램을 종료합니다. 수고하셨습니다.")
            break

        # 따옴표 감싸진 경로 처리 (드래그 앤 드롭 등)
        clean_path = user_input.strip("'\"")

        # 추론 수행
        process_json_file(model, processor, clean_path, args, device)
        print("\n다음 파일 입력을 대기합니다...")


if __name__ == "__main__":
    main()

