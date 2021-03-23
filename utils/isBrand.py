brands = ['TECNIS', 'SENSAR', 'ACRYSOF']
# need to implement similarity score
def isBrand(x):
  x = x.upper()
  if x not in brands:
    return False
  return True