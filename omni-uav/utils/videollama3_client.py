"""
VideoLLaMA3 客户端 - 视频理解多模态大模型接口
基于 DAMO-NLP-SG/VideoLLaMA3
"""
from typing import List, Union, Optional
from PIL import Image
import torch
import os
from pathlib import Path


class VideoLLaMA3Client:
    """VideoLLaMA3 客户端"""

    # Configuration options (can be modified)
    USE_FLASH_ATTENTION = False
    USE_8BIT_QUANTIZATION = False

    def __init__(self, model_name: str = "DAMO-NLP-SG/VideoLLaMA3-7B", device: str = "cuda:0"):
        """
        初始化VideoLLaMA3客户端

        Args:
            model_name: 模型名称或路径
            device: 设备，如 "cuda:0" 或 "cpu"
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._initialized = False
        self.model_dtype = None  # 保存模型使用的 dtype

    def release(self):
        """释放模型资源，清理 GPU 内存"""
        if self.model is not None:
            try:
                # 删除模型
                del self.model
                self.model = None

                # 清理 GPU 缓存
                if str(self.device).startswith("cuda"):
                    import torch
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                print("VideoLLaMA3 model released")
            except Exception as e:
                print(f"Error releasing model: {e}")

        self.processor = None
        self._initialized = False
        self.model_dtype = None

    def initialize(self):
        """初始化模型"""
        if self._initialized:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            print(f"Loading VideoLLaMA3 model: {self.model_name}")

            # 检查 GPU 内存
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU Memory: {gpu_memory:.1f} GB")
                if "7B" in self.model_name and gpu_memory < 16:
                    print("WARNING: 7B model requires ~16GB GPU memory. Consider using 2B model.")
                free_memory = (torch.cuda.get_device_properties(0).total_memory -
                              torch.cuda.memory_allocated(0)) / 1024**3
                print(f"Available GPU Memory: {free_memory:.1f} GB")

            print("This may take a while...")

            # 尝试使用 flash_attention_2（显存占用更小），失败则回退到 sdpa
            attn_implementation = "sdpa"
            use_flash_attn = False

            if self.USE_FLASH_ATTENTION and str(self.device).startswith("cuda"):
                try:
                    import flash_attn
                    # 检查 flash_attn 是否可用
                    attn_implementation = "flash_attention_2"
                    use_flash_attn = True
                    print("Using flash_attention_2 for lower memory usage")
                    print("Note: If you encounter segmentation fault, set USE_FLASH_ATTENTION=False")
                except ImportError:
                    print("flash_attn not available, using default attention (may use more memory)")
                    print("Install with: pip install flash-attn --no-build-isolation")
                    attn_implementation = "sdpa"
            else:
                attn_implementation = "sdpa"
                if not self.USE_FLASH_ATTENTION:
                    print("Flash attention disabled, using default attention")

            # dtype 策略：CUDA 用 float16
            if str(self.device).startswith("cuda"):
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            # 8-bit 量化配置（进一步减少显存）
            quantization_config = None
            if self.USE_8BIT_QUANTIZATION and str(self.device).startswith("cuda"):
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                    )
                    print("Using 8-bit quantization for lower memory usage")
                except ImportError:
                    print("bitsandbytes not available for 8-bit quantization")
                    print("Install with: pip install bitsandbytes")

            # 使用更安全的加载方式
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
                "attn_implementation": attn_implementation,
            }

            if quantization_config:
                load_kwargs["quantization_config"] = quantization_config

            # 先加载到 CPU，然后安全地移动到 GPU
            load_kwargs_cpu = load_kwargs.copy()
            load_kwargs_cpu.pop("device_map", None)  # 确保不使用 device_map

            print("Loading model (manual device placement for stability)...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs_cpu
            )

            # 手动移动到设备（更安全，避免 device_map 相关崩溃）
            if str(self.device).startswith("cuda"):
                print(f"Moving model to {self.device} (this may take a moment)...")
                try:
                    # 先转换精度，再移动设备（确保 dtype 一致）
                    if torch_dtype == torch.float16:
                        print("Converting to float16...")
                        self.model = self.model.half()
                    elif torch_dtype == torch.bfloat16:
                        print("Converting to bfloat16...")
                        self.model = self.model.to(torch.bfloat16)

                    # 清理 CPU 缓存
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    # 移动到 GPU
                    print(f"Moving to {self.device}...")
                    self.model = self.model.to(self.device)

                    # 再次清理缓存
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    print("Model moved to GPU successfully")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        raise RuntimeError(f"GPU out of memory: {e}\n"
                                         "Solutions: 1) Use 2B model instead of 7B, "
                                         "2) Close other GPU applications, "
                                         "3) Reduce video resolution")
                    raise
                except Exception as e:
                    print(f"Warning: Error moving model to GPU: {e}")
                    raise RuntimeError(f"Failed to move model to GPU: {e}")

            # 保存模型使用的 dtype
            try:
                for param in self.model.parameters():
                    if param.is_cuda or param.device.type == 'cuda':
                        self.model_dtype = param.dtype
                        print(f"Detected model dtype: {self.model_dtype}")
                        break
                if self.model_dtype is None:
                    self.model_dtype = torch_dtype
            except Exception as e:
                print(f"Warning: Could not detect model dtype: {e}, using {torch_dtype}")
                self.model_dtype = torch_dtype

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            self._initialized = True
            print(f"VideoLLaMA3 model loaded successfully (attention: {attn_implementation}, dtype: {self.model_dtype})")

        except ImportError as e:
            print(f"Failed to import required libraries: {e}")
            print("Please install: pip install transformers torch accelerate")
            print("Video decoding deps: pip install decord ffmpeg-python")
            print("Also ensure system ffmpeg is installed (e.g. Ubuntu: sudo apt-get install ffmpeg)")
            print("Optional (advanced): pip install flash-attn --no-build-isolation (for faster inference)")
            raise
        except Exception as e:
            print(f"Failed to load VideoLLaMA3 model: {e}")
            raise

    def analyze_single_image(
        self,
        image: Union[str, Image.Image],
        question: str,
        max_new_tokens: int = 1024
    ) -> str:
        """
        分析单张图片

        Args:
            image: 图片路径或PIL Image
            question: 问题
            max_new_tokens: 最大生成token数

        Returns:
            模型回答
        """
        if not self._initialized:
            self.initialize()

        # 如果是路径，转换为绝对路径
        if isinstance(image, str):
            image_path = os.path.abspath(image)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
        else:
            # PIL Image，需要先保存为临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                image_path = tmp.name

        # 构建对话格式
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ]
            },
        ]

        # 处理输入
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # 移动到设备
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # 转换像素值为模型使用的 dtype（确保类型匹配）
        if "pixel_values" in inputs:
            if self.model_dtype is not None:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model_dtype)
            else:
                # 回退方案：根据设备选择
                if str(self.device).startswith("cuda"):
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
                else:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

        # 生成回答
        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        except RuntimeError as e:
            if "CUDA" in str(e) or "memory" in str(e).lower():
                raise RuntimeError(f"GPU memory error during generation: {e}\n"
                                 "Try: 1) Use smaller model (2B), 2) Reduce video resolution")
            raise
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

        # 解码输出
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # 清理临时文件
        if not isinstance(image, str) and os.path.exists(image_path):
            try:
                os.unlink(image_path)
            except:
                pass

        return response

    def generate_text_only(
        self,
        prompt: str,
        max_new_tokens: int = 1024
    ) -> str:
        """
        纯文本生成（不包含视频或图像输入）

        Args:
            prompt: 文本提示
            max_new_tokens: 最大生成token数

        Returns:
            生成的文本
        """
        if not self._initialized:
            self.initialize()

        # 构建纯文本对话格式
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            },
        ]

        # 处理输入
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # 移动到设备
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # 转换像素值为模型使用的 dtype（如果有）
        if "pixel_values" in inputs:
            if self.model_dtype is not None:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model_dtype)
            else:
                if str(self.device).startswith("cuda"):
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
                else:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

        # 生成回答
        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        except Exception as e:
            raise RuntimeError(f"Error during text generation: {e}")

        # 解码输出
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return response
