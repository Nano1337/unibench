import pandas as pd
from unibench.output import OutputHandler

# NOTE: this output dir is where the model dir and inside are the feather files that contain the eval results
output_handler = OutputHandler(output_dir="eval_results") 

# TODO: I basically want a version of this to be launched per gpu for each different checkpoint during evaluation.
# TODO: we have to get it working end to end for one checkpoint before we integrate into science and work on SLURM
output_handler.write_eval_results(model_name="vitamin_l_comp1b")
