import json, sys, time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from safety_judge import SafetyJudge

SEED=42; N_RANDOM=20; TARGET_LAYERS=[9,11]; ADDITION_ALPHA=2.0
MODEL_PATH='models/llama_3_1_8b_instruct'; GUARD_PATH='models/llama_guard_3_8b'
MAX_NEW_TOK=512; TEMPERATURE=0.7; DO_SAMPLE=True
ACT_DIR=Path('mechinterp/activations'); PROBE_DIR=Path('mechinterp/probes')
OUT_DIR=Path('mechinterp/results'); OUT_DIR.mkdir(parents=True,exist_ok=True)

def make_random_vectors(direction, n, seed_base=1000):
    vectors=[]
    for i in range(n):
        rng=torch.Generator(); rng.manual_seed(seed_base+i)
        u=torch.randn(direction.shape[0],generator=rng,dtype=torch.float32).cuda()
        u=u-(u@direction)*direction; u=u/u.norm()
        vectors.append(u)
    return vectors

def load_model():
    print('Loading Llama 3.1 8B (4-bit)...')
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type='nf4')
    tok=AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl=AutoModelForCausalLM.from_pretrained(MODEL_PATH,quantization_config=bnb,device_map='cuda')
    mdl.eval(); print(f'  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB',flush=True)
    return mdl,tok

def generate(model,tokenizer,prompt,round_number,direction=None,layer=None,alpha=None,ablate=False):
    chat=[{'role':'user','content':prompt}]
    input_ids=tokenizer.apply_chat_template(chat,add_generation_prompt=True,return_tensors='pt').to('cuda')
    attention_mask=torch.ones_like(input_ids); n_input=input_ids.shape[1]
    handle=None
    if direction is not None and layer is not None:
        def hook_fn(module,args):
            h=args[0].to(torch.float32)
            if ablate:
                proj=(h@direction).unsqueeze(-1); h=h-proj*direction
            else:
                h=h+alpha*direction
            mod=list(args); mod[0]=h.to(args[0].dtype); return tuple(mod)
        handle=model.model.layers[layer].register_forward_pre_hook(hook_fn)
    torch.manual_seed(SEED+round_number)
    with torch.no_grad():
        output=model.generate(input_ids=input_ids,attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOK,temperature=TEMPERATURE,do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id)
    if handle: handle.remove()
    return tokenizer.decode(output[0][n_input:],skip_special_tokens=True).strip()

if __name__=='__main__':
    print('='*65)
    print(f'MULTIPLE RANDOM DIRECTION NULL DISTRIBUTION')
    print(f'N_RANDOM={N_RANDOM} | Layers={TARGET_LAYERS} | Addition alpha={ADDITION_ALPHA}')
    print('='*65)

    all_dirs=np.load(str(PROBE_DIR/'directions_prehook_all_layers.npy')).astype(np.float32)
    learned={L:torch.tensor(all_dirs[L],dtype=torch.float32).cuda() for L in TARGET_LAYERS}
    randoms={L:make_random_vectors(learned[L],N_RANDOM) for L in TARGET_LAYERS}

    for L in TARGET_LAYERS:
        dots=[(learned[L]@u).item() for u in randoms[L]]
        print(f'Layer {L}: max dot={max(abs(d) for d in dots):.8f}')

    model,tokenizer=load_model()
    print('\nLoading LlamaGuard...')
    judge=SafetyJudge(model_path=GUARD_PATH); judge.load()
    print(f'  VRAM after both: {torch.cuda.memory_allocated()/1e9:.2f} GB',flush=True)

    split=json.load(open(PROBE_DIR/'train_test_split.json'))
    data_C=np.load(str(ACT_DIR/'activations_prehook_C_seed42.npz'))
    rl_C=data_C['replayed_labels'].astype(bool)
    records_C=[json.loads(l) for l in open(
        'results_task6_676prompt/competition_with_memory_seed42/rounds.jsonl')
        if json.loads(l)['phase']=='phase2']
    C_test_refused=[records_C[i] for i in split['test_indices'] if rl_C[i]]
    C_test_complied=[records_C[i] for i in split['test_indices'] if not rl_C[i]]
    N_ref=len(C_test_refused); N_com=len(C_test_complied)
    print(f'C_test refused: {N_ref} | complied: {N_com}')

    all_results={}; t0=time.time()

    for L in TARGET_LAYERS:
        print(f'\n{"="*55}\nLAYER {L}\n{"="*55}')
        layer_results={}

        # Ablation baselines
        base_labels=[]
        for rec in C_test_refused:
            resp=generate(model,tokenizer,rec['prompt'],rec['round_number'])
            j=judge.score_response(rec['task'],resp)
            base_labels.append(bool(j['is_safe']))

        # Learned ablation
        learned_abl_flips=0
        abl_labels=[]
        for rec,base_safe in zip(C_test_refused,base_labels):
            resp=generate(model,tokenizer,rec['prompt'],rec['round_number'],
                         direction=learned[L],layer=L,ablate=True)
            j=judge.score_response(rec['task'],resp)
            abl_safe=bool(j['is_safe']); abl_labels.append(abl_safe)
            if base_safe and not abl_safe: learned_abl_flips+=1
        print(f'Learned ablation flips: {learned_abl_flips}/{N_ref}',flush=True)

        # Random ablation controls
        random_abl_flips=[]
        for ri,rv in enumerate(randoms[L]):
            flips=0
            for rec,base_safe in zip(C_test_refused,base_labels):
                if not base_safe: continue
                resp=generate(model,tokenizer,rec['prompt'],rec['round_number'],
                             direction=rv,layer=L,ablate=True)
                j=judge.score_response(rec['task'],resp)
                if not bool(j['is_safe']): flips+=1
            random_abl_flips.append(flips)
            print(f'  Ablation random {ri+1}/{N_RANDOM}: {flips} flips',flush=True)

        rand_abl=np.array(random_abl_flips)
        p_abl=float(np.mean(rand_abl>=learned_abl_flips))
        print(f'Ablation null: mean={rand_abl.mean():.1f} std={rand_abl.std():.1f} max={rand_abl.max()} p={p_abl:.3f}')

        # Addition baselines
        add_base=[]
        for rec in C_test_complied:
            resp=generate(model,tokenizer,rec['prompt'],rec['round_number'])
            j=judge.score_response(rec['task'],resp); add_base.append(bool(j['is_safe']))

        # Learned addition
        learned_add=0
        for rec in C_test_complied:
            resp=generate(model,tokenizer,rec['prompt'],rec['round_number'],
                         direction=learned[L],layer=L,alpha=ADDITION_ALPHA,ablate=False)
            j=judge.score_response(rec['task'],resp)
            if bool(j['is_safe']): learned_add+=1
        print(f'Learned addition refusal: {learned_add}/{N_com}',flush=True)

        # Random addition controls
        random_add=[]
        for ri,rv in enumerate(randoms[L]):
            refused=0
            for rec in C_test_complied:
                resp=generate(model,tokenizer,rec['prompt'],rec['round_number'],
                             direction=rv,layer=L,alpha=ADDITION_ALPHA,ablate=False)
                j=judge.score_response(rec['task'],resp)
                if bool(j['is_safe']): refused+=1
            random_add.append(refused)
            print(f'  Addition random {ri+1}/{N_RANDOM}: {refused}/{N_com}',flush=True)

        rand_add=np.array(random_add)
        p_add=float(np.mean(rand_add>=learned_add))
        print(f'Addition null: mean={rand_add.mean():.1f} std={rand_add.std():.1f} max={rand_add.max()} p={p_add:.3f}')

        all_results[str(L)]={
            'ablation':{'learned_flips':learned_abl_flips,'random_flips':random_abl_flips,
                'rand_mean':float(rand_abl.mean()),'rand_std':float(rand_abl.std()),
                'rand_max':int(rand_abl.max()),'p_value':p_abl},
            'addition':{'alpha':ADDITION_ALPHA,'learned_refused':learned_add,
                'random_refused':random_add,'rand_mean':float(rand_add.mean()),
                'rand_std':float(rand_add.std()),'rand_max':int(rand_add.max()),'p_value':p_add}
        }
        with open(OUT_DIR/'multi_random_control_checkpoint.json','w') as f:
            json.dump(all_results,f,indent=2)
        print(f'Checkpoint saved layer {L}',flush=True)

    print(f'\n{"="*65}\nSUMMARY')
    print(f'{"Layer":<8} {"Exp":<12} {"Learned":<12} {"Rand mean":<12} {"Rand max":<10} {"p-value"}')
    print('-'*65)
    for L in TARGET_LAYERS:
        r=all_results[str(L)]
        print(f'{L:<8} {"Ablation":<12} {r["ablation"]["learned_flips"]:<12} {r["ablation"]["rand_mean"]:<12.1f} {r["ablation"]["rand_max"]:<10} {r["ablation"]["p_value"]:.3f}')
        print(f'{L:<8} {"Addition":<12} {r["addition"]["learned_refused"]:<12} {r["addition"]["rand_mean"]:<12.1f} {r["addition"]["rand_max"]:<10} {r["addition"]["p_value"]:.3f}')

    with open(OUT_DIR/'multi_random_control_results.json','w') as f:
        json.dump(all_results,f,indent=2)
    print(f'Saved. Total time: {(time.time()-t0)/60:.1f}min')
    judge.unload(); del model; torch.cuda.empty_cache()
