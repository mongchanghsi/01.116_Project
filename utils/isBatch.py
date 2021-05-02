def isBatch(x, model):
  # If the batch can be detected as per normal
  if (x[0:len(model)] == model and len(x) == 10):
    return True
  return False

def batchSimilarity(x, model):
  # x is the batch word
  # model is the model detected
  if (isREF(x)):
    x = cleanREF(x)

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

# Check if there is a REF word in it, if there is, conduct a clean up on the first few alphabets
def isREF(x):
  if 'REF' not in x:
    return False
  return True

def cleanREF(x):
  x = x.replace('REF', '').replace('I', '')
  clean_x = ''
  for i in x:
    if i.isalnum() == True:
      clean_x += i
  return clean_x