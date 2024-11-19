from functools import partial
from io import open_code
from unibench import Evaluator
from unibench.models_zoo.wrappers.clip import ClipModel
import open_clip

model, _, _ = open_clip.create_model_and_transforms(
    "ViTamin-L", pretrained="datacomp1b"
)

tokenizer = open_clip.get_tokenizer("ViTamin-L")

model = partial(
    ClipModel,
    model=model,
    model_name="vitamin_l_comp1b",
    tokenizer=tokenizer,
    input_resolution=model.visual.image_size[0],
    logit_scale=model.logit_scale,
)


eval = Evaluator(
    benchmarks_dir="/home/ubuntu/data/unibench_eval_sets",
    num_workers=2,  # check with torch.multiprocessing.cpu_count()
    output_dir="eval_results",
)

eval.add_model(model=model)
eval.update_benchmark_list(["imagenet1k"])
eval.update_model_list(["vitamin_l_comp1b"])
eval.evaluate()