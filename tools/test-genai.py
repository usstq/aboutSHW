import time
import openvino_genai as ov_genai
import shutil
import os


'''
$ xONEDNN_VERBOSE=2 xMOE_DEBUG=1 GENAI_REMOTE_TENSOR=0 NO_MOE=1 python test2.py 
/home/openvino-ci-87/tingqian/test2.py:35: DeprecationWarning: 'config' parameters is deprecated, please use kwargs to pass config properties instead.
  pipe = ov_genai.LLMPipeline(model_id, "CPU", scheduler_config=scheduler_config, config={'ATTENTION_BACKEND':"PA", "INFERENCE_PRECISION_HINT": "FP32"})
compile cost  77.65 seconds
prompt=(['你是谁'],)
？你为什么在这里？ 我是一个人工智能助手，旨在帮助用户解答问题 
gen cost 5.65 seconds
？你为什么在这里？ 我是一个人工智能助手，旨在帮助用户解答问题 
gen 0 cost 4.15 seconds

？你为什么在这里？ 我是一个人工智能助手，旨在帮助用户解答问题 
gen 1 cost 3.97 seconds

？你为什么在这里？ 我是一个人工智能助手，旨在帮助用户解答问题 
gen 2 cost 4.31 seconds

Generate duration: 4311.77
TTFT: 359.13 ms
TPOT: 263.48 ms/token
Throughput: 3.80 tokens/s

'''

#input(os.getpid())

scheduler_config = ov_genai.SchedulerConfig()
# cache params
#scheduler_config.cache_size = 2
scheduler_config.dynamic_split_fuse = True
scheduler_config.max_num_batched_tokens = 8192*2
#scheduler_config.max_num_seqs = 256
#scheduler_config.num_kv_blocks = 34
scheduler_config.enable_prefix_caching = False

def streamer(subword): 
        print(subword, end='', flush=True) 
        # Return flag corresponds whether generation should be stopped. 
        # False means continue generation. 
        return False
properties = {
     #'EXECUTION_MODE_HINT': 'ACCURACY'
}

#model_id = '/mnt/luocheng/2025.1/model/qwen3-15b-a2b-base/pytorch/ov/OV_FP16-4BIT_DEFAULT/'
#model_id = '/home/openvino-ci-85/river/models/qwen3-15b-a2b-base/pytorch/ov/OV_FP16-4BIT_DEFAULT'
#model_id = '/home/openvino-ci-96/qwen3-moe-L2-int4/'
#model_id = '/home/devuser/luocheng/models/Qwen3-0.6B-Base-ov-with-past/'
#model_id = r'C:\Users\openvino-ci-73\models\qwen3-15b-a2b-base\pytorch\ov\OV_FP16-4BIT_DEFAULT'
model_id = r'C:\Users\openvino-adlh\river\model\qwen2-7b-instruct\pytorch\ov\OV_FP16-4BIT_DEFAULT'

if 0:
    beg = time.time()
    pipe = ov_genai.LLMPipeline(model_id, "CPU", scheduler_config=scheduler_config, config={'ATTENTION_BACKEND':"PA", "INFERENCE_PRECISION_HINT": "BF16"}) 
    end = time.time()
    print(f'compile cost {end - beg: .2f} seconds')
else:
    beg = time.time()
    #pipe = ov_genai.LLMPipeline(model_id, "CPU", config={'ATTENTION_BACKEND':"PA", "INFERENCE_PRECISION_HINT": "FP32"})
    pipe = ov_genai.LLMPipeline(model_id, "GPU", scheduler_config=scheduler_config, config={'ATTENTION_BACKEND':"PA", "INFERENCE_PRECISION_HINT": "FP16"}) #, "CACHE_DIR": "gpu_cache2"})
    end = time.time()
    print(f'compile cost {end - beg: .2f} seconds')

config = ov_genai.GenerationConfig()
config.max_new_tokens = 8 #128
config.ignore_eos = True
#config.do_sample = False
# config.top_p = 2.0
# config.top_k = 1
config.apply_chat_template = False
#config.rng_seed = 10
# Speculative decoding generation parameters like `num_assistant_tokens` and `assistant_confidence_threshold` are mutually excluded
# add parameter to enable speculative decoding to generate `num_assistant_tokens` candidates by draft_model per iteration
#config.num_assistant_tokens = 5

prompts=[
    #r"天",
    #r"你是谁",
    "Hi " * 8192,
    #r"请问中国十个最著名的景点及其特点是什么？",
    #r'''波吉亚的职业生涯开始于1445年，当时14岁的他被舅舅指派为瓦伦西亚主教堂的看守，而这名舅舅是以前曾经被安日诺四世任命为枢机主教的阿方索-波吉亚。1448年，他成为瓦伦西亚、巴塞罗那和塞戈尔韦主教座堂的咏祷司铎，并在舅舅与尼各老五世的沟通下得以在缺席的情况下担任这个职位而仍然领取薪水，使得波吉亚有机会游历罗马。[8]在罗马，他先是跟随一名叫做加斯帕雷-达-维罗纳的人文主义者学习，其后在博洛尼亚大学学习法律，并获得了教会法学博士学位，被人认为“是一名卓越的、头脑清晰的法律学家”。[9]1456年，在他舅舅成为教宗后，25岁的他领受圣职成为执事级枢机，领圣尼各老监狱执事区执事衔。一年后，被任命为圣座的副秘书长，这是那个时代典型的裙带关系。1457年，嘉礼三世派波吉亚作为教皇特使前往安科纳镇压叛乱。波吉亚很好地完成了任务，至1492年他自己担任教皇为止，他一共担任这个职务长达35年。1457年末，他的哥哥佩德罗患病，他临时填补了佩德罗教皇军队总司令的职务直至佩德罗康复[8]。1458年，他协助教宗庇护二世当选，受到了庇护二世的优待，1460年，他因参加一场私人宴会受到庇护二世的指责，庇护二世听说这次宴会最终变成了一场狂欢，波吉亚后来就此事道歉，但否认宴会出现狂欢，至今宴会的真实情况亦不得而知。1463年，波吉亚响应庇护二世对枢机主教的号召，参与资助一场新的十字军东征。[8] 1464年，保罗二世当选教皇，他是波吉亚的好友，因此波吉亚得以继续保持较高地位。1468年，领受圣铎成为神父，然后在1471年，祝圣为天主教阿尔巴诺罗马城郊教区主教 。[10]1482年，教皇西克斯图斯四世开始任命波吉亚7岁的孩子凯撒担任教会职务，与此同时，波吉亚担任的圣职继续增加，并在1483年成为最富有的红衣主教。[8] 1484年西克斯图斯四世去世，波吉亚原本打算竞选教皇职位，但后来从竞争中退出，教皇英诺森八世即位[8]，他继续扮演教皇的忠实支持者。在为五位教宗服务之后，劳得力·波吉亚积攒了充分的经验、影响力与财富。'''
    #"What is OpenVINO?",
    # "How are you?",
    # "Tell me something about Canada?",
    #"Who is Mark Twain?"
    #'Who is Barbara Cartland?'
    #'Who is the most famous inventor?',
    #'Who is the most famous mathematician?'
    #'What is C++?'
    #'Who is Georges Simenon?'
    #"Who is Harold Robbins?"
],
try:
    shutil.rmtree('gpu_cache2', True)
    print(f'prompt={prompts}')
    beg = time.time()
    result = pipe.generate(inputs=prompts[0], generation_config=config)
    end = time.time()
    print(result, f'\ngen cost {end - beg:.2f} seconds')
    
    # import os
    # input(f'pid= {os.getpid()} hit to continue...')
    for i in range(3):
        beg = time.time()
        result = pipe.generate(inputs=prompts[0], generation_config=config)
        end = time.time()
        print(result, f'\ngen {i} cost {end - beg:.2f} seconds\n')
finally:
    shutil.rmtree('gpu_cache2', True)
    pass

perf_metrics = result.perf_metrics

print(f'Generate duration: {perf_metrics.get_generate_duration().mean:.2f}')
print(f'TTFT: {perf_metrics.get_ttft().mean:.2f} ms')
print(f'TPOT: {perf_metrics.get_tpot().mean:.2f} ms/token')
print(f'Throughput: {perf_metrics.get_throughput().mean:.2f} tokens/s')
