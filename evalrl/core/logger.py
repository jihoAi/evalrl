import os
import csv

class Logger:
  def __init__(self, run_dir):
    self.run_dir = run_dir
    os.makedirs(run_dir,exist_ok=True)
    self._files = {}
    self._writers = {}
    
  def log(self,name,step,metrics):
    path = os.path.join(self.run_dir,f"{name}.csv")
    row = {"step":step,**metrics}
    if name not in self._files:
      self._files[name] = open(path,"w",newline="")
      self._writers[name] = csv.DictWriter(self._files[name],fieldnames=list(row.keys()))
      self._writers[name].writeheader()
    self._writers[name].writerow(row)
    self._files[name].flush()
    output = [f"{metric}: {value:.4f}" for metric,value in metrics.items()]
    line = f"[{name}] step {step:<8} | " + " | ".join(output)
    print(line)

  def close(self):
    for file in self._files.values():
      file.close()
    self._files = {}
    self._writers = {}
