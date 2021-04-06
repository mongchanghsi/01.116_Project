# TODO: Include uppercase and lowercase variants too
brand_GT = ['TECNIS', 'Sensar', 'AcrySof']
brands = ['TECNIS', 'SENSAR', 'ACRYSOF']
import csv_utils

# brands = ['TECNIS', 'SENSAR', 'ACRYSOF']
BRANDS = csv_utils.get_brands()

# need to implement similarity score
def isBrand(x):
  x = x.upper()
  if x not in brands:
    return False
  return True

def brandSimilarity(x):
  similarityScore = {}
  x = x.upper()
  for i in brands:
    word, score = similarityFunction(x, i)
    if word not in similarityScore.keys():
      similarityScore[word] = score
  if similarityScore != {}:
    most_similar_brand = (max(similarityScore, key=similarityScore.get))
    highest_score = max(similarityScore.values())
    return most_similar_brand, highest_score

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
  score = (count / len(word1)) * 100
  return word2, score