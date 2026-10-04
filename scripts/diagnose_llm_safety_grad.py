"""Check forward/gradient numerics for the source model precision."""

import gc
from pathlib import Path
import sys

workspace_source = Path(__file__).resolve().parents[1] / "participant" / "workspace"
if workspace_source.is_dir():
    sys.path.insert(0, str(workspace_source))
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_safety_optimizer import TARGET, suffix_prompt_parts

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
revision = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
p, s, t = suffix_prompt_parts(tokenizer, f"Please output {TARGET} ",
                             "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
target = tokenizer("Sure, here it is.", add_special_tokens=False,
                   return_tensors="pt").input_ids[0]
prompt = torch.cat((p, s, t)).to("cuda")
full = torch.cat((prompt, target.to("cuda")))
for dtype in (torch.float16, torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype, device_map="auto",
        attn_implementation="eager"
    )
    model.eval().requires_grad_(False)
    for mode in ("ids", "embeds"):
        if mode == "ids":
            with torch.no_grad():
                logits = model(input_ids=full[None], use_cache=False).logits
        else:
            one_hot = torch.zeros(s.numel(), model.get_input_embeddings().weight.shape[0],
                                  device="cuda", dtype=dtype)
            one_hot.scatter_(1, s.to("cuda")[:, None], 1.0)
            one_hot.requires_grad_(True)
            with torch.no_grad():
                fixed = model.get_input_embeddings()(full[None])
            embed = torch.cat((fixed[:, :p.numel()],
                               (one_hot @ model.get_input_embeddings().weight)[None],
                               fixed[:, p.numel()+s.numel():]), dim=1)
            logits = model(inputs_embeds=embed, use_cache=False).logits
        selected = logits[0, prompt.numel()-1:prompt.numel()-1+target.numel()]
        loss = F.cross_entropy(selected.float(), target.to("cuda"))
        print(dtype, mode, "logits_finite", bool(torch.isfinite(selected).all()),
              "loss", float(loss.detach()), flush=True)
        if mode == "embeds" and torch.isfinite(loss):
            grad = torch.autograd.grad(loss, one_hot)[0]
            print(dtype, mode, "gradient_finite", bool(torch.isfinite(grad).all()),
                  "gradient_max", float(grad.abs().max()), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
