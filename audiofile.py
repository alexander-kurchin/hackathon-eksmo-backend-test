import argparse
import os

import grpc
from dotenv import load_dotenv

import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc

CHUNK_SIZE = 4000


def gen(audio_file_name):
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

    # Чтение аудиофайла и отправка его содержимого порциями
    with open(audio_file_name, "rb") as f:
        data = f.read(CHUNK_SIZE)
        while data != b"":
            yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=data))
            data = f.read(CHUNK_SIZE)


def run(secret, audio_file_name):
    # Установка соединения с сервером
    cred = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel("stt.api.cloud.yandex.net:443", cred)
    stub = stt_service_pb2_grpc.RecognizerStub(channel)

    # Отправка данных для распознавания
    it = stub.RecognizeStreaming(gen(audio_file_name),
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
    parser.add_argument("--path", required=False, help="Audio file path")
    args = parser.parse_args()

    API_KEY = os.getenv("API_KEY")
    PATH = args.path or "example_audiofile.wav"
    run(API_KEY, PATH)
