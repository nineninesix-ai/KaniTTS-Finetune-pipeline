import torch
from nemo.collections.tts.models import AudioCodecModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np
from config_loader import config_loader



class NemoAudioPlayer:
    def __init__(self, config=None, text_tokenizer_name: str = None) -> None:
        """
        Initialize NemoAudioPlayer

        Args:
            config: Optional config object for backward compatibility. If None, uses centralized config.
            text_tokenizer_name: Optional tokenizer name for text decoding
        """
        # Load centralized configs
        model_cfg = config_loader.get_model_config()

        # For backward compatibility, support old config object
        if config is not None:
            self.conf = config
            # Use values from passed config if available, otherwise use centralized config
            self.tokeniser_length = getattr(config, 'tokeniser_length', model_cfg.tokenizer.vocab_size)
            self.start_of_text = getattr(config, 'start_of_text', model_cfg.special_tokens.start_of_text)
            self.end_of_text = getattr(config, 'end_of_text', model_cfg.special_tokens.end_of_text)
        else:
            # Use centralized config
            self.tokeniser_length = model_cfg.tokenizer.vocab_size
            self.start_of_text = model_cfg.special_tokens.start_of_text
            self.end_of_text = model_cfg.special_tokens.end_of_text

        # Load codec model
        self.nemo_codec_model = AudioCodecModel\
                .from_pretrained(model_cfg.codec.model_name).eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nemo_codec_model.to(self.device)

        # Text tokenizer
        self.text_tokenizer_name = text_tokenizer_name
        if self.text_tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_tokenizer_name)

        # Load special tokens from centralized config
        self.start_of_speech = model_cfg.special_tokens.start_of_speech
        self.end_of_speech = model_cfg.special_tokens.end_of_speech
        self.start_of_human = model_cfg.special_tokens.start_of_human
        self.end_of_human = model_cfg.special_tokens.end_of_human
        self.start_of_ai = model_cfg.special_tokens.start_of_ai
        self.end_of_ai = model_cfg.special_tokens.end_of_ai
        self.pad_token = model_cfg.special_tokens.pad_token
        self.audio_tokens_start = model_cfg.codec.audio_tokens_start
        self.codebook_size = model_cfg.codec.codebook_size

    def output_validation(self, out_ids):
        start_of_speech_flag = self.start_of_speech in out_ids
        end_of_speech_flag = self.end_of_speech in out_ids
        if not (start_of_speech_flag and end_of_speech_flag):
            raise ValueError('Special speech tokens not exist!')

    def get_nano_codes(self, out_ids):
        model_cfg = config_loader.get_model_config()
        num_codebooks = model_cfg.codec.num_codebooks

        start_a_idx = (out_ids == self.start_of_speech).nonzero(as_tuple=True)[0].item()
        end_a_idx   = (out_ids == self.end_of_speech).nonzero(as_tuple=True)[0].item()
        if start_a_idx >= end_a_idx:
            raise ValueError('Invalid audio codes sequence!')

        audio_codes = out_ids[start_a_idx+1 : end_a_idx]
        if len(audio_codes) % num_codebooks:
            raise ValueError(f'The length of the sequence must be a multiple of {num_codebooks}!')
        audio_codes = audio_codes.reshape(-1, num_codebooks)
        audio_codes = audio_codes - torch.tensor([self.codebook_size * i for i in range(num_codebooks)])
        audio_codes = audio_codes - self.audio_tokens_start
        if (audio_codes < 0).sum().item() > 0:
            raise ValueError('Invalid audio tokens!')

        audio_codes = audio_codes.T.unsqueeze(0)
        len_ = torch.tensor([audio_codes.shape[-1]])
        return audio_codes, len_

    def get_text(self, out_ids):
        start_t_idx = (out_ids == self.start_of_text).nonzero(as_tuple=True)[0].item()
        end_t_idx   = (out_ids == self.end_of_text).nonzero(as_tuple=True)[0].item()
        txt_tokens = out_ids[start_t_idx : end_t_idx+1]
        text = self.tokenizer.decode(txt_tokens, skip_special_tokens=True)
        return text

    def get_waveform(self, out_ids):
        out_ids = out_ids.flatten()
        self.output_validation(out_ids)
        audio_codes, len_ = self.get_nano_codes(out_ids)
        audio_codes, len_ = audio_codes.to(self.device), len_.to(self.device)
        with torch.inference_mode():
            reconstructed_audio, _ = self.nemo_codec_model.decode(tokens=audio_codes, tokens_len=len_)
            output_audio = reconstructed_audio.cpu().detach().numpy().squeeze()

        if self.text_tokenizer_name:
            text = self.get_text(out_ids)
            return output_audio, text
        else:
            return output_audio, None



class KaniModel:
    def __init__(self, config=None, model_name=None, player:NemoAudioPlayer=None)->None:
        """
        Initialize KaniModel

        Args:
            config: Optional config object for backward compatibility. If None, uses centralized config.
            model_name: Model name or path
            player: NemoAudioPlayer instance
        """
        # Load centralized configs
        inference_cfg = config_loader.get_inference_config()

        # For backward compatibility, support old config object
        if config is not None:
            self.conf = config
            device_map = getattr(config, 'device_map', inference_cfg.model.device_map)
            torch_dtype = getattr(config, 'torch_dtype', inference_cfg.model.torch_dtype)
        else:
            device_map = inference_cfg.model.device_map
            torch_dtype = inference_cfg.model.torch_dtype

        # Convert torch_dtype string to actual dtype
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)

        self.player = player
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                torch_dtype=torch_dtype,
                                device_map=device_map,
                            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_input_ids(self, text_promt:str)->tuple[torch.tensor]:
        START_OF_HUMAN = self.player.start_of_human
        END_OF_TEXT = self.player.end_of_text
        END_OF_HUMAN = self.player.end_of_human

        input_ids = self.tokenizer(text_promt, return_tensors="pt").input_ids
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        attention_mask = torch.ones(1, modified_input_ids.shape[1], dtype=torch.int64)
        return modified_input_ids, attention_mask


    def model_request(self, input_ids:torch.tensor,
                          attention_mask:torch.tensor)->torch.tensor:
        # Load generation config
        inference_cfg = config_loader.get_inference_config()

        # For backward compatibility, support old config object
        if hasattr(self, 'conf') and self.conf is not None:
            max_new_tokens = getattr(self.conf, 'max_new_tokens', inference_cfg.generation.max_new_tokens)
        else:
            max_new_tokens = inference_cfg.generation.max_new_tokens

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=inference_cfg.generation.do_sample,
                temperature=inference_cfg.generation.temperature,
                top_p=inference_cfg.generation.top_p,
                repetition_penalty=inference_cfg.generation.repetition_penalty,
                num_return_sequences=inference_cfg.generation.num_return_sequences,
                eos_token_id=self.player.end_of_speech,
            )
        return generated_ids.to('cpu')

    def run_model(self, text:str):
        input_ids, attention_mask = self.get_input_ids(text)
        model_output = self.model_request(input_ids, attention_mask)
        audio, _ = self.player.get_waveform(model_output)
        return audio, text