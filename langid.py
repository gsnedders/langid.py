#!/usr/bin/env python
"""
langid.py - 
Language Identifier by Marco Lui April 2011

Based on research by Marco Lui and Tim Baldwin.

Copyright 2011 Marco Lui <saffsd@gmail.com>. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of the copyright holder.
"""

import itertools
import array
import base64
import bz2
import json
import optparse
import logging
import weakref
import sys
from math import log, pi, lgamma
from cPickle import loads
from collections import defaultdict

import numpy as np


__all__ = ["Identifier"]


_default_model = weakref.WeakValueDictionary()


class WeakRefableList(list):
  pass


class WeakRefableDict(dict):
  pass


_fv_dtype = "uint%u" % (log(sys.maxsize + 1, 2) + 1)


lgamma_ufunc = np.frompyfunc(lgamma, 1, 1)
def logfac(a):
  return lgamma_ufunc(a + 1)


class Identifier(object):
  def __init__(self, model=None, languages=None, norm_probs=True):
    """
    Create a language identifier with a given language model, optionally
    normalizing probabilities so that they fall within [0, 1].
    """
    self._norm_probs = norm_probs

    if not model:
      try:
        self.nb_ptc = _default_model["nb_ptc"]
        self.nb_pc = _default_model["nb_pc"]
        self.nb_classes = _default_model["nb_classes"]
        self.tk_nextmove = _default_model["tk_nextmove"]
        self.tk_output = _default_model["tk_output"]
        self.nb_numfeats = len(self.nb_ptc) / len(self.nb_pc)
      except KeyError:
        with bz2.BZ2File("model.bz2", "rb") as fp:
          self._unpack(fp)
        _default_model["nb_ptc"] = self.nb_ptc
        _default_model["nb_pc"] = self.nb_pc
        _default_model["nb_classes"] = self.nb_classes
        _default_model["tk_nextmove"] = self.tk_nextmove
        _default_model["tk_output"] = self.tk_output
    else:
      self._unpack(model)

    if languages:
      self._set_languages(languages)

  def _unpack(self, fp):
    """
    Unpack a model that has been compressed into a file
    NOTE: nb_ptc and nb_pc are array.array('f') instances.
          nb_ptc is packed into a 1-dimensional array, each term is represented by
          len(nb_pc) continuous entries
    """
    # Reading the whole file into memory and calling loads is far quicker. Go figure.
    model = loads(fp.read())
    self.nb_ptc, self.nb_pc, self.nb_classes, self.tk_nextmove, self.tk_output = model
    self.nb_numfeats = len(self.nb_ptc) / len(self.nb_pc)

    # reconstruct pc and ptc
    self.nb_pc = np.array(self.nb_pc)
    self.nb_ptc = np.array(self.nb_ptc).reshape(len(self.nb_ptc)/len(self.nb_pc), len(self.nb_pc))

    # Make sure everything can be weakref'd
    self.nb_classes = WeakRefableList(self.nb_classes)
    self.tk_output = WeakRefableDict(self.tk_output)

  def _set_languages(self, langs):
    # We were passed a restricted set of languages. Trim the arrays accordingly
    # to speed up processing.
    for lang in langs:
      if lang not in nb_classes:
        raise ValueError, "Unknown language code %s" % lang

    subset_mask = np.fromiter((l in langs for l in self.nb_classes), dtype=bool)
    self.nb_classes = [ c for c in self.nb_classes if c in langs ]
    self.nb_ptc = self.nb_ptc[:,subset_mask]
    self.nb_pc = self.nb_pc[subset_mask]

  def tokenize(self, text, arr):
    """
    Tokenize text into a feature vector stored in arr.
    """
    # Convert the text to a sequence of ascii values
    ords = map(ord, text)

    # Count the number of times we enter each state
    state = 0
    statecount = defaultdict(int)
    for letter in ords:
      state = self.tk_nextmove[(state << 8) + letter]
      statecount[state] += 1

    # Update all the productions corresponding to the state
    for state in statecount:
      for index in self.tk_output.get(state, []):
        arr[index] += statecount[state]

    return arr

  def nb_classprobs(self, fv):
    # compute the log-factorial of each element of the vector
    logfv = logfac(fv).astype(float)
    # compute the probability of the document given each class
    pdc = np.dot(fv,self.nb_ptc) - logfv.sum()
    # compute the probability of the document in each class
    pd = pdc + self.nb_pc
    return pd

  def norm_probs(self, pd):
    """
    Renormalize log-probs into a proper distribution (sum 1)
    The technique for dealing with underflow is described in
    http://jblevins.org/log/log-sum-exp
    """
    if self._norm_probs:
      pd = (1/np.exp(pd[None,:] - pd[:,None]).sum(1))
      return pd
    else:
      return pd

  def instance2fv(self, instance):
    """
    Map an instance into the feature space of the trained model.
    """
    if isinstance(instance, unicode):
      instance = instance.encode('utf8')

    fv = self.tokenize(instance, 
                       np.zeros((self.nb_numfeats,), dtype=_fv_dtype))
    return fv

  def classify(self, instance):
    """
    Classify an instance.
    """
    fv = self.instance2fv(instance)
    probs = self.norm_probs(self.nb_classprobs(fv))
    cl = np.argmax(probs)
    conf = probs[cl]
    pred = self.nb_classes[cl]
    return pred, conf

  def rank(self, instance):
    """
    Return a list of languages in order of likelihood.
    """
    fv = self.instance2fv(instance)
    probs = self.norm_probs(self.nb_classprobs(fv))
    return sorted(zip(self.nb_classes, probs), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
  parser = optparse.OptionParser()
  parser.add_option('-v', action='count', dest='verbosity', help='increase verbosity (repeat for greater effect)')
  parser.add_option('-m', dest='model', help='load model from file')
  parser.add_option('-l', '--langs', dest='langs', help='comma-separated set of target ISO639 language codes (e.g en,de)')
  parser.add_option('-u', '--url', help='langid of URL')
  options, args = parser.parse_args()

  if options.verbosity:
    logging.basicConfig(level=max((5-options.verbosity)*10, 0))
  else:
    logging.basicConfig()

  identifier = Identifier(options.model)

  if options.langs:
    langs = options.langs.split(",")
    identifier.set_languages(langs)

  if options.url:
    import urllib2
    import contextlib
    with contextlib.closing(urllib2.urlopen(options.url)) as url:
      text = url.read()
      lang, conf = identifier.classify(text)
      print options.url, len(text), lang
    
  import sys
  if sys.stdin.isatty():
    # Interactive mode
    while True:
      try:
        print ">>>",
        text = raw_input()
      except Exception:
        print
        break
      print identifier.classify(text)
  else:
    # Redirected
    print identifier.classify(sys.stdin.read())
