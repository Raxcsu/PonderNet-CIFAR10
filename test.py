import wandb
api = wandb.Api()

file = "./wandb/run-20220227_160047-x37jl1s7/run-x37jl1s7.wandb"
run = api.run(file)
if run.state == "finished":
   for i, row in run.history().iterrows():
      print(row["_timestamp"], row["accuracy"])
