"""PyAV decoding, Silero VAD and local faster-whisper transcription."""
import argparse, json, os, sys, wave
from pathlib import Path

def listen(request):
    import av
    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    source=Path(request['source']); folder=Path(request['output_dir']);folder.mkdir(parents=True,exist_ok=True)
    output=folder/'voice.wav'
    sample_count=0
    # Decode directly to WAV instead of retaining every frame plus an int16 copy.
    with wave.open(str(output),'wb') as wav, av.open(str(source)) as container:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
        if not container.streams.audio:raise ValueError('视频中没有音轨，请另外上传录音，或上传物体照片。')
        resampler=av.AudioResampler(format='s16',layout='mono',rate=16000)
        def write_frame(converted):
            nonlocal sample_count
            chunk=converted.to_ndarray().reshape(-1)
            sample_count+=len(chunk)
            if sample_count>16000*1800:raise ValueError('请上传 30 分钟以内的录音或视频。')
            wav.writeframesraw(chunk.tobytes())
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):write_frame(converted)
        for converted in resampler.resample(None):write_frame(converted)
    if not sample_count:raise ValueError('未能提取有效音频。')
    from faster_whisper.audio import decode_audio
    audio=decode_audio(str(output),sampling_rate=16000)
    speech=get_speech_timestamps(audio,VadOptions())
    resource=Path(os.getenv('MEMORYMADE_RESOURCE_ROOT',str(Path(__file__).resolve().parents[2])))
    roots=[Path(os.getenv('MEMORYMADE_WRITE_ROOT',str(resource))),'D:/1223456789',resource]
    model_path=next((Path(root)/'assets/models/faster-whisper-small' for root in roots if (Path(root)/'assets/models/faster-whisper-small/model.bin').exists()),None)
    if model_path is None:raise ValueError('听力模型尚未下载完成，请运行模型安装程序。')
    # CPU int8 leaves the GPU available for visual reconstruction on this laptop.
    model=WhisperModel(str(model_path),device='cpu',compute_type='int8',cpu_threads=8,local_files_only=True)
    segments,info=model.transcribe(audio,beam_size=5,vad_filter=True,word_timestamps=True)
    segments=[{'start':s.start,'end':s.end,'text':s.text.strip(),'words':[{'word':w.word,'start':w.start,'end':w.end,'probability':w.probability} for w in (s.words or [])]} for s in segments]
    result={'ok':True,'audio':str(output),'duration':len(audio)/16000,'language':info.language,
            'language_probability':info.language_probability,'text':' '.join(s['text'] for s in segments),
            'segments':segments,'speech_regions':[{'start':s['start']/16000,'end':s['end']/16000} for s in speech],
            'backend':'faster-whisper small / CTranslate2 int8 + Silero VAD + PyAV','device':'cpu'}
    (folder/'transcript.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'transcript.txt').write_text(result['text'] or '未检测到清晰人声；原录音仍可播放。',encoding='utf-8')
    return result

if __name__=='__main__':
    from disk_guard import start_watchdog
    start_watchdog()
    parser=argparse.ArgumentParser();parser.add_argument('operation');parser.add_argument('request');parser.add_argument('response');args=parser.parse_args()
    try:result=listen(json.loads(Path(args.request).read_text(encoding='utf-8')))
    except Exception as exc:
        import traceback;traceback.print_exc();result={'ok':False,'error':str(exc)}
    Path(args.response).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    sys.exit(0 if result['ok'] else 1)
