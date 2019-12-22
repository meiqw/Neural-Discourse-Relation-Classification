#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Data Loader/Pre-processor Functions

Feel free to change/restructure the code below
"""
import codecs
import json
import os
import numpy as np

from pymagnitude import Magnitude

__author__ = 'Jayeol Chun'


"""Useful constants when processing `relations.json`"""
ARG1  = 'Arg1'
ARG2  = 'Arg2'
CONN  = 'Connective'
SENSE = 'Sense'
TYPE  = 'Type'
TOKEN = 'TokenList'
KEYS  = [ARG1, ARG2, CONN, SENSE]
TEXT  = 'RawText'
ID = 'DocID'
TYPE = 'Type'
FEATURE = "Feature"
ARG1_LEN  = 20
ARG2_LEN  = 20
CONN_LEN  = 3
sense_dict = {}
sense_count = 0

def featurize(rel, vectors):
  """Featurizes a relation dict into feature vector

  TODO: `rel` is a dict object for a single relation in `relations.json`, where
    `Arg1`, `Arg2`, `Connective` and `Sense` are all strings (`Conn` may be an
    empty string). Implement a featurization function that transforms this into
    a feature vector. You may use word embeddings.

  Args:
    rel: dict, see `preprocess` below

  Returns:

  """
  arg1_feature = np.zeros((ARG1_LEN, vectors.dim))
  field_featurize(arg1_feature, ARG1, ARG1_LEN, rel, vectors)
  conn_feature = np.zeros((CONN_LEN, vectors.dim))
  field_featurize(conn_feature, CONN, CONN_LEN, rel, vectors)
  arg2_feature = np.zeros((ARG2_LEN, vectors.dim))
  field_featurize(arg2_feature, ARG2, ARG2_LEN, rel, vectors)

  rel[ARG1][FEATURE] = arg1_feature
  rel[CONN][FEATURE] = conn_feature
  rel[ARG2][FEATURE] = arg2_feature

  rel[SENSE] = sense_dict[rel[SENSE]]
  rel[FEATURE] = np.concatenate((rel[ARG1][FEATURE],rel[CONN][FEATURE],rel[ARG2][FEATURE]))
  return rel

def field_featurize(field_matrix, FIELD, FIELD_LEN, rel, vectors):
  field_lst = rel[FIELD][TEXT].split()
  field_len = len(field_lst)
  ZERO_EMBED = np.zeros((vectors.dim,))
  if (field_len < FIELD_LEN):
    for counter, value in enumerate(field_lst):
      field_matrix[counter] = vectors.query(value)
    for i in range(field_len, FIELD_LEN):
      field_matrix[i] = ZERO_EMBED
  else:
    for counter, value in enumerate(field_lst[:FIELD_LEN]):
      field_matrix[counter] = vectors.query(value)

def preprocess(rel, vectors):
  """Preprocesses a single relation in `relations.json`

  Args:
    rel: dict

  Returns:
    see `featurize` above
  """
  rel_dict = {}
  for key in KEYS:

    if key in [ARG1, ARG2, CONN]:
      # for `Arg1`, `Arg2`, `Connective`, we only keep tokens of `RawText`
      rel_dict[key] = {TEXT:"", TOKEN:[]}
      rel_dict[key][TEXT] = rel[key][TEXT]
      for lst in rel[key][TOKEN]:
        rel_dict[key][TOKEN].append(lst[2]) if lst is not None else lst[0]
    elif key == SENSE:
      # `Sense` is the target label. For relations with multiple senses, we
      # assume (for simplicity) that the first label is the gold standard.
      rel_dict[key] = rel[key][0]
  rel_dict[ID] = rel[ID]
  rel_dict[TYPE] = rel[TYPE]
  # into feature vector/matrix
  feat_rel = featurize(rel_dict, vectors)
  #print("done")
  return feat_rel

def load_relations(data_file, vectors):
  """Loads a single `relations.json` file

  Args:
    data_file: str, path to a single data file

  Returns:
    list, where each item is of type dict
  """
  rel_path = os.path.join(data_file, "relations.json")
  assert os.path.exists(rel_path), \
    "{} does not exist in `load_relations.py".format(rel_path)

  rels = []
  with codecs.open(rel_path, encoding='utf-8') as pdtb:
    for pdtb_line in pdtb:
      rel = json.loads(pdtb_line)
      #print(rel)
      rel = preprocess(rel, vectors)
      rels.append(rel)

  return rels

def load_data(data_dir='./data'):
  """Loads all data in `data_dir` as a dict

  Each of `dev`, `train` and `test` contains (1) `raw` folder (2)
    `relations.json`. We don't need to worry about `raw` folder, and instead
    focus on `relations.json` which contains all the information we need for our
    classification task.

  Args:
    data_dir: str, the root directory of all data

  Returns:
    dict, where the keys are: `dev`, `train` and `test` and the values are lists
      of relations data in `relations.json`
  """
  assert os.path.exists(data_dir), "`data_dir` does not exist in `load_data`"

  data = {}
  vectors = Magnitude("glove.6B.50d.magnitude")
  #vectors = Magnitude("glove.6B.300d.magnitude")
  get_sense_dict(os.path.join(data_dir,"train"))
  #print(sense_dict)

  for folder in os.listdir(data_dir):
    #print(folder)
    print("Loading", folder)
    folder_path = os.path.join(data_dir, folder)
    #print(folder_path)
    data[folder] = load_relations(folder_path, vectors)
  '''
  print("Loading", "dev")
  folder_path = os.path.join(data_dir, "dev")
  data["dev"] = load_relations(folder_path, vectors)
  '''
  return data

#calculate the sense label
def get_sense_dict(folder_path):
  #global sense_dict
  global sense_count

  rel_path = os.path.join(folder_path, "relations.json")
  with codecs.open(rel_path, encoding='utf-8') as pdtb:
    for pdtb_line in pdtb:
      rel = json.loads(pdtb_line)
      sense = rel[SENSE][0]
      if sense not in sense_dict:
        sense_dict[sense] = sense_count
        sense_count += 1

  #return sense_dict

#if __name__ == "__main__":
  #vectors = Magnitude("glove.6B.300d.magnitude")
  #print(vectors.dim)
  #load_data()
  #print(data['dev'][0])
  #print(data["dev"][0])
  #print(data["dev"][0][FEATURE])
  #print(sense_dict)
