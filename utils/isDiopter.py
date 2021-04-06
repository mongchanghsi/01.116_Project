def isDiopter(x):
  # if x[0] != '+' or x[len(x)-1] != 'D':
  #   return False
  if x[0] != '+':
    return False
  return True

# Process Diopter checks if there is an additional 'D' or other stuff in the string, if there is, remove it
def processDiopter(x):
  while True:
    if x[len(x)-1] == 'D' or x[len(x)-1].isdigit() == False:
      x = x[:-1]
    else:
      break
  return x