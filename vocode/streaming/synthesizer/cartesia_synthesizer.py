import io
import wave

from cartesia.tts import AsyncCartesiaTTS

from vocode import getenv
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import CartesiaSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult


class CartesiaSynthesizer(BaseSynthesizer[CartesiaSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: CartesiaSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        self.api_key = getenv("CARTESIA_API_KEY")
        self.model_id = synthesizer_config.model_id
        self.voice_id = synthesizer_config.voice_id
        self.sampling_rate = synthesizer_config.sampling_rate
        self.output_format = synthesizer_config.output_format
        self.client = AsyncCartesiaTTS(api_key=self.api_key)
        self.voice_embedding = self.client.get_voice_embedding(voice_id=self.voice_id)

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        generator = await self.client.generate(
            transcript=message.text,
            voice=self.voice_embedding,
            stream=True,
            model_id=self.model_id,
            data_rtype='bytes',
            output_format=self.output_format
        )

        sample_rate = self.sampling_rate
        audio_file = io.BytesIO()

        with wave.open(audio_file, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            async for chunk in generator:
                raw_data = chunk['audio']
                wav_file.writeframes(raw_data)
        audio_file.seek(0)

        result = self.create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=audio_file,
            message=message,
            chunk_size=chunk_size,
        )

        return result
