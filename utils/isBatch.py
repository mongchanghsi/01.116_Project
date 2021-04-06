def isBatch(x, model):
  # If the batch can be detected as per normal
  if (x[0:len(model)] == model and len(x) == 10):
    return True
  return False

def batchSimilarity(x, model):
  # x is the batch word
  # model is the model detected 
  model_word = x[:len(model)]
  count = 0
  i, j = 0, 0
  while i < len(model_word) and j < len(model):
    if model_word[i] == model[i]:
      count += 1
    i += 1
    j += 1
  score = (count / len(model)) * 100
  return x, score