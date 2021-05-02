def isDate(x):
  if len(x) != 10:
    return False
  if x[:3] != '202':
    return False
  if int(x[:4]) < 2021:
    return False
  
  month = int(x[5:7])
  day = int(x[len(x)-2:])
  if month > 13 or day > 31:
    return False
  return True