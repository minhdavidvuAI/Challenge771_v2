import torch
if torch.cuda.is_available():
  print("GPU used")
else:
  print("CPU used, slower execution")
  print("Try to get a GPU runtime for faster progress")