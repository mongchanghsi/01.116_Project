import csv_utils
acrysof_models = ['TFNT00', 'TFNT20-T60', 'SN6AT2-T9', 'AU00T0', 'SA60WF', 'SA6AT2-T9', 'MA60MA']

tecnis_3_pc_models = ['ZA9003']
tecnis_1_models = ['ZCB00']
tecnis_eyhance_models = ['ICB00']
tecnis_toric_models = ['ZCT100', 'ZCT150', 'ZCT225', 'ZCT300', 'ZCT375', 'ZCT450', 'ZCT525', 'ZCT600', 'ZCT700', 'ZCT800']
tecnis_multifocal_models = ['ZMA00']
tecnis_multifocal_toric_models = ['ZMT150', 'ZMT225', 'ZMT300', 'ZMT400']
tecnis_symfony_models = ['ZXR00']
tecnis_symfony_toric_models = ['ZXT100', 'ZXT150', 'ZXT225', 'ZXT300', 'ZXT375', 'ZXT450', 'ZXT525', 'ZXT600']
tecnis_synergy_models = ['ZFR00V']

sensar_1_models = ['AAB00']
sensar_models = ['AR40M', 'AR40E', 'AR40e']

BRANDS = csv_utils.get_brands()

def isModel(x, b):
  if b == 'TECNIS':
    if x in tecnis_3_pc_models:
      return True
    if x in tecnis_1_models:
      return True
    if x in tecnis_eyhance_models:
      return True
    if x in tecnis_toric_models:
      return True
    if x in tecnis_multifocal_models:
      return True
    if x in tecnis_multifocal_toric_models:
      return True
    if x in tecnis_symfony_models:
      return True
    if x in tecnis_symfony_toric_models:
      return True
    if x in tecnis_synergy_models:
      return True
  elif b == 'SENSAR':
    if x in sensar_1_models:
      return True
    if x in sensar_models:
      return True
  elif b == 'ALCON':
    if x in alcon_models:
      return True
  return False