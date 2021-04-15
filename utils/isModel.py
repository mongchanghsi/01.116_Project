# import csv_utils
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

tecnis_collated_model = []
tecnis_collated_model.extend(tecnis_3_pc_models)
tecnis_collated_model.extend(tecnis_1_models)
tecnis_collated_model.extend(tecnis_eyhance_models)
tecnis_collated_model.extend(tecnis_toric_models)
tecnis_collated_model.extend(tecnis_multifocal_models)
tecnis_collated_model.extend(tecnis_multifocal_toric_models)
tecnis_collated_model.extend(tecnis_symfony_models)
tecnis_collated_model.extend(tecnis_symfony_toric_models)
tecnis_collated_model.extend(tecnis_synergy_models)

sensar_collated_model = []
sensar_collated_model.extend(sensar_1_models)
sensar_collated_model.extend(sensar_models)

# BRANDS = csv_utils.get_brands()

def isModel(x, b):
  if b == 'TECNIS':
    if x in tecnis_collated_model:
      return True
  elif b == 'SENSAR':
    if x in sensar_collated_model:
      return True
  elif b == 'ALCON':
    if x in alcon_models:
      return True
  return False

def modelSimilarity(x,b):
  b = b.upper()
  similarityScore = {}
  if b == 'TECNIS':
    for b2 in tecnis_collated_model:
      word, score = similarityFunction(x, b2)
      if word not in similarityScore.keys():
        similarityScore[word] = score
      else:
        if similarityScore[word] < score:
          similarityScore[word] = score
  elif b == 'SENSAR':
    for b2 in sensar_collated_model:
      word, score = similarityFunction(x, b2)
      if score > 0:
        if word not in similarityScore.keys():
          similarityScore[word] = score
        else:
          if similarityScore[word] < score:
            similarityScore[word] = score
  # print(f'The word is {x}, the Similarity Score is {similarityScore}')
  if similarityScore != {}:
    most_similar_model= (max(similarityScore, key=similarityScore.get))
    highest_score = max(similarityScore.values())
    return most_similar_model, highest_score

  # if there is no similarity at all, return blank modeland 0 score
  return '', 0

def similarityFunction(word1, word2):
  # word1 is the OCR
  # word2 is the 'ground-truth' word
  count = 0
  i, j = 0, 0

  while i < len(word1) and j < len(word2):
    if word1[i] == word2[i]:
      count += 1
    i += 1
    j += 1
  score = (count / len(word2)) * 100
  return word2, score