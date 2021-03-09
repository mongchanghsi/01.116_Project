def isDate(x):
  if len(x) != 10:
    return False
  if x[:3] != '202':
    return False
  return True