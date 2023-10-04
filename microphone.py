import argparse
import os
import wave

import grpc
import pyaudio
from dotenv import load_dotenv

import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc

# Настройки потокового распознавания
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
CHUNK = 4096
WAVE_OUTPUT_FILENAME = "audio.wav"

audio = pyaudio.PyAudio()


def gen(record_seconds):
    # Настройки распознавания
    recognize_options = stt_pb2.StreamingOptions(
        recognition_model=stt_pb2.RecognitionModelOptions(
            audio_format=stt_pb2.AudioFormatOptions(
                raw_audio=stt_pb2.RawAudio(
                    audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                    sample_rate_hertz=8000,
                    audio_channel_count=1,
                )
            ),
            text_normalization=stt_pb2.TextNormalizationOptions(
                text_normalization=stt_pb2.TextNormalizationOptions
                                          .TEXT_NORMALIZATION_ENABLED,
                profanity_filter=True,
                literature_text=False,
            ),
            language_restriction=stt_pb2.LanguageRestrictionOptions(
                restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                language_code=["ru-RU"],
            ),
            audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
        )
    )

    # Отправка сообщения с настройками распознавания
    yield stt_pb2.StreamingRequest(session_options=recognize_options)

    # Запись голоса
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    print("Запись пошла...")
    frames = []

    # Распознавание речи по порциям
    for i in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=data))
        frames.append(data)
    print("Запись окончена")

    # Остановка записи
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Создание файла *.WAV с записанным голосом
    wave_file = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b"".join(frames))
    wave_file.close()


def run(secret, record_seconds):
    # Установка соединения с сервером
    cred = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel("stt.api.cloud.yandex.net:443", cred)
    stub = stt_service_pb2_grpc.RecognizerStub(channel)

    # Отправка данных для распознавания
    it = stub.RecognizeStreaming(gen(record_seconds),
                                 metadata=(("authorization",
                                            f"Api-Key {secret}"),)
                                 )

    # Обработка ответов сервера и вывод результата в консоль
    try:
        for r in it:
            event_type, alternatives = r.WhichOneof("Event"), None
            if event_type == "final_refinement":
                alternatives = [a.text for a in r.final_refinement
                                                 .normalized_text
                                                 .alternatives]
                print("\n".join(alternatives))
    except grpc._channel._Rendezvous as err:
        print(f"Error code {err._state.code}, message: {err._state.details}")
        raise err


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--time", required=False, help="Record length (seconds)")
    args = parser.parse_args()

    API_KEY = os.getenv("API_KEY")
    RECORD_SECONDS = args.time or 5
    run(API_KEY, int(RECORD_SECONDS))
