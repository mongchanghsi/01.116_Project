import csv_utils

# brands = ['TECNIS', 'SENSAR', 'ACRYSOF']
BRANDS = csv_utils.get_brands()

# TODO: Include uppercase and lowercase variants too

# need to implement similarity score
def isBrand(x):
  x = x.upper()
  if x not in BRANDS:
    return False
  return True