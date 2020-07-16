## POS probe

Results of the POS probe saved in this directory as:

~~~~
  {model_spec}_{DATE}/
  | info.txt
  | probe.state_dict
~~~~

- `info.txt` printout of ARGS used
- `probe.state_dict` saved parameters of probe (using `torch.save(probe.state_dict(), ...)`)
