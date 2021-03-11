SENSAR = 10
TECNIS = 10
ACRYSOF = 11

def isSerial(x, b):
  # assuming that your serial number is in one piece
  if len(x) < 10:
    return False
  if x.isdecimal() == False:
    return False

  if b == 'SENSAR' or b =='TECNIS':
    if len(x) > 10:
      return False

  if b == 'ACRYSOF':
    if len(x) == 10 or len(x) > 11:
      return False
  return True

def isSerial_2(x1, x2, b):
  if x1.isdecimal() == False or x2.isdecimal() == False:
    return False
  combined_serial = x1 + x2

  if b == 'SENSAR' or b =='TECNIS':
    if len(combined_serial) > 10:
      return False

  if b == 'ACRYSOF':
    if len(combined_serial) == 10 or len(combined_serial) > 11:
      return False
  return True