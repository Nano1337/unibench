import pandas as pd
from unibench.output import OutputHandler


output_handler = OutputHandler(output_dir="eval_results")
output_handler.load_model_csvs_and_calculate(model_name="vitamin_l_comp1b")
print(output_handler._model_csv.columns)
