import asyncio
from typing import Optional

from cartesia import AsyncCartesiaTTS
from cartesia.tts import AudioOutputFormat

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import SynthesizerConfig, SynthesizerType
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
)

CARTESIA_DEFAULT_VOICE = "Barbershop Man"
CARTESIA_DATA_RTYPE = "bytes"
CARTESIA_DEFAULT_OUTPUT_FORMAT = AudioOutputFormat.PCM
CARTESIA_DEFAULT_MODEL_ID = "upbeat-moon"


class CartesiaSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.CARTESIA.value):
    voice_name: str = CARTESIA_DEFAULT_VOICE
    output_format: str | AudioOutputFormat = CARTESIA_DEFAULT_OUTPUT_FORMAT
    data_rtype: str = CARTESIA_DATA_RTYPE
    model_id: str = CARTESIA_DEFAULT_MODEL_ID
    api_key: str = str(getenv("CARTESIA_API_KEY"))


class CartesiaSynthesizer(BaseSynthesizer[CartesiaSynthesizerConfig]):
    def __init__(self, synthesizer_config: CartesiaSynthesizerConfig):
        super().__init__(synthesizer_config=synthesizer_config)

        if self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            self.synthesizer_config.output_format = AudioOutputFormat.MULAW_8000
        elif self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            self.synthesizer_config.output_format = AudioOutputFormat.PCM

        self.catesia_client = AsyncCartesiaTTS(api_key=self.synthesizer_config.api_key)
        voices = self.catesia_client.get_voices()
        voice_id = voices[self.synthesizer_config.voice_name]["id"]
        self.voice = self.catesia_client.get_voice_embedding(voice_id=voice_id)

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        self.total_chars += len(message.text)
        chunk_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

        try:
            chunk_generator = await self.catesia_client.generate(
                transcript=message.text,
                voice=self.voice,
                stream=True,
                data_rtype=self.synthesizer_config.data_rtype,
                model_id=self.synthesizer_config.model_id,
                output_format=self.synthesizer_config.output_format,
            )
            async for data in chunk_generator:  # type: ignore
                chunk = data["audio"]
                chunk_queue.put_nowait(chunk)

        except asyncio.CancelledError:
            pass
        finally:
            await chunk_queue.put(None)

        return SynthesisResult(
            chunk_generator=self.chunk_result_generator_from_queue(chunk_queue),
            get_message_up_to=lambda seconds: self.get_message_cutoff_from_voice_speed(
                message=message, seconds=seconds, words_per_minute=150
            ),
        )

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: CartesiaSynthesizerConfig):
        output_format = (
            synthesizer_config.output_format
            if isinstance(synthesizer_config.output_format, str)
            else synthesizer_config.output_format.value
        )

        return ":".join(
            (
                SynthesizerType.CARTESIA.value,
                synthesizer_config.model_id,
                synthesizer_config.audio_encoding,
                output_format,
            )
        )
