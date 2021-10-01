import numpy as np
import pandas as pd

def chunk_data(samples, duration=2, r_sample=16000):
  
  data=[]
  for offset in range(0, len(samples), r_sample):
    start = offset
    end   = offset + duration*r_sample
    chunk = samples[start:end]
    
    if(len(chunk)==duration*r_sample):
      data.append(chunk)
    
  return data