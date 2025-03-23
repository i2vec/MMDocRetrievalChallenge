import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch.nn.functional as F
from PIL import Image
import gc
from qwen_vl_utils import process_vision_info

class ContentJudger:
    def __init__(self, qwen25vl_path = "/mnt/vepfs/fs_ckps/xumj/llms/Qwen2.5-VL-72B-Instruct-AWQ"):
    # def __init__(self, qwen25vl_path = "/mnt/vepfs/fs_ckps/xumj/llms/Qwen2.5-VL-7B-Instruct/"):
        """初始化内容判断器"""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen25vl_path,
            # torch_dtype="auto",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            # device_map={"": "cuda"},
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(qwen25vl_path)
    
    def judge_relevance(self, image, question, ocr_text=None):
        """
        判断图像和OCR文本是否与问题相关
        
        参数:
        - image: PIL图像对象或图像路径
        - question: 问题文本
        - ocr_text: 可选的OCR文本
        
        返回:
        - result: 'yes'或'no'的判断结果
        """
        try:
            # 处理图像输入
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            # 构建消息格式
            content = []
            content.append({"type": "image", "image": image})
            
            # 构建提示文本
            prompt = f"Is this document page can provide information to answer the following question? Answer with only 'yes' or 'no'."
            if ocr_text:
                prompt += f"\n\nDocument text: {ocr_text}"
            
            prompt += f"\n\nQuestion: {question}"
            
            content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "user", "content": content}]
            
            # 准备输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # 将输入移至GPU
            inputs = inputs.to(self.model.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    # do_sample=False, 
                    # top_k=10
                )
                response = self.processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # 清理回答，只保留yes或no
                result = response.strip().lower()
                if 'yes' in result:
                    result = 'yes'
                elif 'no' in result:
                    result = 'no'
                else:
                    result = 'yes'  # 当模型回答不明确时默认为yes
                # 清理内存
                torch.cuda.empty_cache()
                gc.collect()
                
                return result, response
        except Exception as e:
            print(f"判断过程中出错: {str(e)}")
            return 'no'  # 出错时默认返回no



