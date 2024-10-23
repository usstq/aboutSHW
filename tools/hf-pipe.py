import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from optimum.intel.openvino import OVModelForCausalLM


import inspect, types, time
import psutil
process = psutil.Process()

torch.random.manual_seed(0)


class HookForward:
    def __init__(self, model, tokenizer=None) -> None:
        model._org_forward = model.forward
        model._latencies = []
        model._input_ids_shapes = []
        model._prompt = []
        def new_forward(self, *args, **kwargs):
            # Call the original method
            # print(args, kwargs)
            if 'input_ids' in kwargs:
                input_ids_shape = kwargs["input_ids"].shape
                model._input_ids_shapes.append(input_ids_shape)
                if input_ids_shape[1] > 1:
                    model._prompt.append(kwargs["input_ids"])
            t0 = time.time()
            ret = self._org_forward(*args, **kwargs)
            t1 = time.time()
            self._latencies.append(t1 - t0)
            return ret
        # https://stackoverflow.com/questions/1409295/set-function-signature-in-python
        # package transformers uses inspect.signature to detect exact signature of
        # model's forward method and behave differently based on that, for example
        # only auto-generate attention-mask when signature contain such named parameter
        new_forward.__signature__ = inspect.signature(model.forward)
        model.forward = types.MethodType(new_forward, model)
        self.model = model
        self.tokenizer = tokenizer

    def __enter__(self):
        # print("Entering the context")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        #print("Leaving the context:", exc_type, exc_value, exc_tb)
        self.update()
        self.model.forward = types.MethodType(model._org_forward, self.model)

    def update(self):
        if len(self.model._input_ids_shapes) > 0 and len(self.model._latencies) > 0:
            prompt_shape = self.model._input_ids_shapes[0]
            l = self.model._latencies
            second_tok_latency = sum(l[1:])/(len(l)-1) if len(l) > 1 else 0
            mem_info = process.memory_info()
            if self.tokenizer and len(self.model._prompt) > 0:
                # print(self.model._prompt[0])
                promopt_text = self.tokenizer.decode(self.model._prompt[0][0,:], skip_special_tokens=False)
                print(f" prompt:  {promopt_text}")
            print(f" prompt:{prompt_shape[0]}x{prompt_shape[1]}  {l[0]*1e3:6.1f} ms + {second_tok_latency*1e3:6.1f} ms x {len(l)-1}   RSS/VMS {mem_info.rss*1e-9: .3f}/{mem_info.vms*1e-9: .3f} GB")
        self.model._latencies = []
        self.model._input_ids_shapes = []
        self.model._prompt = []


model_id = "microsoft/Phi-3.5-mini-instruct"
model_id = "./Phi-3.5-mini-instruct/"
model_id = "./Phi-3.5-mini-instruct-bf16SQ/"
model_id = "/mnt/llm_irs/models_original/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "", "AFFINITY":"CORE", "DYNAMIC_QUANTIZATION_GROUP_SIZE":"409600"}

# model = OVModelForCausalLM.from_pretrained( model_id,  device_map="cpu",  torch_dtype="auto", ov_config=ov_config, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id,  device_map="cpu",  torch_dtype="auto", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    #{"role": "system", "content": "You are a helpful AI assistant."},
    #{"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    #{"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    #{"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    {"role": "user", "content": "What's Oxygen?"},
    #{"role": "user", "content": "如何解方程 2x + 3 = 7 ?"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}


with HookForward(model, tokenizer) as h:
    for round in range(1):
        output = pipe(messages, **generation_args)
        h.update()

    print(output[0]['generated_text'])
