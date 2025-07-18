import os
import torch
import numpy as np
import soundfile as sf
import librosa
import csv
from moviepy import VideoFileClip, AudioFileClip
import cv2

from audioset.pytorch.models import Cnn14

# 1. 加载类别标签
def load_labels(label_csv):
    labels = []
    with open(label_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row['display_name'])
    return labels

# 2. 音频切片生成器
def audio_stream_chunks(y, sample_rate=32000, chunk_sec=1):
    total_samples = y.shape[0]
    chunk_samples = int(chunk_sec * sample_rate)
    start = 0
    while start + chunk_samples <= total_samples:
        yield y[start:start + chunk_samples]
        start += chunk_samples

# 3. 对视频音频分段分类，返回每段top1类别
def classify_audio_chunks(audio, sample_rate, model, labels, device='cuda', chunk_sec=1):
    results = []  # [(start_sec, end_sec, top1_label, top1_prob)]
    idx = 0
    for chunk in audio_stream_chunks(audio, sample_rate=sample_rate, chunk_sec=chunk_sec):
        waveform = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output_dict = model(waveform)
            clipwise_output = torch.sigmoid(output_dict['clipwise_output'])[0].cpu().numpy()
        top_idx = np.argmax(clipwise_output)
        top_label = labels[top_idx]
        top_prob = clipwise_output[top_idx]
        results.append((idx*chunk_sec, (idx+1)*chunk_sec, top_label, top_prob))
        idx += 1
    return results

# 4. 处理视频帧并叠加文字
def overlay_labels_on_video(video_path, output_path, chunk_results, chunk_sec=1):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    frame_idx = 0

    chunk_idx = 0
    num_chunks = len(chunk_results)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 当前帧所在秒数
        sec = frame_idx / fps
        # 找到当前帧对应的chunk
        chunk_idx = min(int(sec // chunk_sec), num_chunks - 1)
        label, prob = chunk_results[chunk_idx][2], chunk_results[chunk_idx][3]
        text = f'{label} ({prob:.2f})'
        # 叠加文字
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()

if __name__ == '__main__':
    # 参数配置
    video_path = r"G:\hrnet-48w-\your_video.mp4"  # 你的输入视频路径
    output_path = r"G:\hrnet-48w-\your_video_with_labels.mp4"  # 输出视频路径
    label_csv = r"G:\hrnet-48w-\audioset\metadata\class_labels_indices.csv"
    panns_weight = r"G:\hrnet-48w-\Cnn14_mAP=0.431.pth"
    chunk_sec = 1

    # 环境准备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = load_labels(label_csv)
    model = Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=len(labels)
    )
    checkpoint = torch.load(panns_weight, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 1. 提取音频
    audio_temp = 'temp_audio.wav'
    videoclip = VideoFileClip(video_path)
    videoclip.audio.write_audiofile(audio_temp, fps=32000,logger=None)

    # 2. 加载音频
    y, sr = sf.read(audio_temp)
    if sr != 32000:
        y = librosa.resample(y, orig_sr=sr, target_sr=32000)
        sr = 32000

    # 3. 分类
    chunk_results = classify_audio_chunks(y, sr, model, labels, device=device, chunk_sec=chunk_sec)

    # 4. 视频写入
    overlay_labels_on_video(video_path, output_path, chunk_results, chunk_sec=chunk_sec)

    # 5. 清理临时文件
    os.remove(audio_temp)

    print(f"带声音标签的视频已保存至: {output_path}")