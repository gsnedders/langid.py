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
from math import log
from cPickle import loads, dumps
from collections import defaultdict

__USE_NUMPY__ = False
try:
  import numpy as np
  __USE_NUMPY__ = True
except ImportError:
  pass

logger = logging.getLogger(__name__)

with open("model.bz2", "rb") as fp:
  _default_model = base64.b64encode(fp.read())


__logfac = {}
def logfac(a):
  if a not in __logfac:
    __logfac[a] = np.sum(np.log(np.arange(1,a+1)))
  return __logfac[a]
logfac = np.frompyfunc(logfac, 1, 1)


class Identifier(object):
  def __init__(self, model=None, norm_probs=True):
    """
    Create a language identifier with a given language model, optionally
    normalizing probabilities so that they fall within [0, 1].
    """
    if model:
      self.model = self.unpack(model)
    else:
      self.model = self.unpack(_default_model)

    self.__full_model = None
    self._norm_probs = norm_probs

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

  if __USE_NUMPY__:
    logger.debug('using numpy implementation')
    def unpack(self, data):
      """
      Unpack a model that has been compressed into a string
      NOTE: nb_ptc and nb_pc are array.array('f') instances.
            nb_ptc is packed into a 1-dimensional array, each term is represented by
            len(nb_pc) continuous entries
      """
      model = loads(bz2.decompress(base64.b64decode(data)))
      self.nb_ptc, self.nb_pc, self.nb_classes, self.tk_nextmove, self.tk_output = model
      self.nb_numfeats = len(self.nb_ptc) / len(self.nb_pc)

      # reconstruct pc and ptc
      self.nb_pc = np.array(self.nb_pc)
      self.nb_ptc = np.array(self.nb_ptc).reshape(len(self.nb_ptc)/len(self.nb_pc), len(self.nb_pc))

    def set_languages(self, langs):
      logger.debug("restricting languages to: %s", langs)

      # Maintain a reference to the full model, in case we change our language set
      # multiple times.
      if self.__full_model is None:
        self.__full_model = self.nb_ptc, self.nb_pc, self.nb_numfeats, self.nb_classes
      else:
        self.nb_ptc, self.nb_pc, self.nb_numfeats, self.nb_classes = self.__full_model

      # We were passed a restricted set of languages. Trim the arrays accordingly
      # to speed up processing.
      for lang in langs:
        if lang not in nb_classes:
          raise ValueError, "Unknown language code %s" % lang

      subset_mask = np.fromiter((l in langs for l in self.nb_classes), dtype=bool)
      self.nb_classes = [ c for c in self.nb_classes if c in langs ]
      self.nb_ptc = self.nb_ptc[:,subset_mask]
      self.nb_pc = self.nb_pc[subset_mask]

    def argmax(self, x):
      return np.argmax(x)

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

  else: # if __USE_NUMPY__:
    # This is a stub for a potential future numpy-less implementation.
    # I will not implement this unless there is a clear demand for it.
    raise NotImplementedError, "langid.py needs numpy to run - please contact the author if you need to use langid.py without numpy"
    logger.debug('using python native implementation')

  def instance2fv(self, instance):
    """
    Map an instance into the feature space of the trained model.
    """
    if isinstance(instance, unicode):
      instance = instance.encode('utf8')

    if __USE_NUMPY__:
      fv = self.tokenize(instance, 
                         np.zeros((self.nb_numfeats,), dtype='uint32'))
    else:
      fv = self.tokenize(instance,
                         array.array('L', itertootls.repeat(0, self.nb_numfeats)))
    return fv

  def classify(self, instance):
    """
    Classify an instance.
    """
    fv = self.instance2fv(instance)
    probs = self.norm_probs(self.nb_classprobs(fv))
    cl = self.argmax(probs)
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
        break
      print identifier.classify(text)
  else:
    # Redirected
    print identifier.classify(sys.stdin.read())
