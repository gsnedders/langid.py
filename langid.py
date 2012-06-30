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

# Defaults for inbuilt server
HOST = None #leave as none for auto-detect
PORT = 9008
FORCE_NATIVE = False
FORCE_WSGIREF = False
NORM_PROBS = True # Normalize optput probabilities.

# NORM_PROBS can be set to False for a small speed increase. It does not
# affect the relative ordering of the predicted classes. 

import itertools
import array
import base64
import bz2
import json
import optparse
import logging
from math import log
from cPickle import loads, dumps
from wsgiref.simple_server import make_server
from wsgiref.util import shift_path_info
from urlparse import parse_qs
from collections import defaultdict

logger = logging.getLogger(__name__)
model_loaded = False
_full_model = None

with open("model.bz2", "rb") as fp:
	model = base64.b64encode(fp.read())


def tokenize(text, arr):
  """
  Tokenize text into a feature vector stored in arr.
  """
  # Convert the text to a sequence of ascii values
  ords = map(ord, text)

  # Count the number of times we enter each state
  state = 0
  statecount = defaultdict(int)
  for letter in ords:
    state = tk_nextmove[(state << 8) + letter]
    statecount[state] += 1

  # Update all the productions corresponding to the state
  for state in statecount:
    for index in tk_output.get(state, []):
      arr[index] += statecount[state]

  return arr

try:
  if FORCE_NATIVE: raise ImportError
  # Numpy implementation
  import numpy as np

  def unpack(data):
    """
    Unpack a model that has been compressed into a string
    NOTE: nb_ptc and nb_pc are array.array('f') instances.
          nb_ptc is packed into a 1-dimensional array, each term is represented by
          len(nb_pc) continuous entries
    """
    global nb_ptc, nb_pc, nb_numfeats, nb_classes, tk_nextmove, tk_output, model_loaded
    model = loads(bz2.decompress(base64.b64decode(data)))
    nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = model
    nb_numfeats = len(nb_ptc) / len(nb_pc)

    # reconstruct pc and ptc
    nb_pc = np.array(nb_pc)
    nb_ptc = np.array(nb_ptc).reshape(len(nb_ptc)/len(nb_pc), len(nb_pc))

    model_loaded = True

  def set_languages(langs):
    global nb_ptc, nb_pc, nb_numfeats, nb_classes
    global _full_model
    logger.debug("restricting languages to: %s", langs)

    # Maintain a reference to the full model, in case we change our language set
    # multiple times.
    if _full_model is None:
      _full_model = nb_ptc, nb_pc, nb_numfeats, nb_classes
    else:
      nb_ptc, nb_pc, nb_numfeats, nb_classes = _full_model

    # We were passed a restricted set of languages. Trim the arrays accordingly
    # to speed up processing.
    for lang in langs:
      if lang not in nb_classes:
        raise ValueError, "Unknown language code %s" % lang

    subset_mask = np.fromiter((l in langs for l in nb_classes), dtype=bool)
    nb_classes = [ c for c in nb_classes if c in langs ]
    nb_ptc = nb_ptc[:,subset_mask]
    nb_pc = nb_pc[subset_mask]


  __logfac = {}
  def logfac(a):
    if a not in __logfac:
      __logfac[a] = np.sum(np.log(np.arange(1,a+1)))
    return __logfac[a]
  logfac = np.frompyfunc(logfac, 1, 1)

  def argmax(x):
    return np.argmax(x)

  def nb_classprobs(fv):
    # compute the log-factorial of each element of the vector
    logfv = logfac(fv).astype(float)
    # compute the probability of the document given each class
    pdc = np.dot(fv,nb_ptc) - logfv.sum()
    # compute the probability of the document in each class
    pd = pdc + nb_pc
    return pd

  if NORM_PROBS:
    def norm_probs(pd):
      """
      Renormalize log-probs into a proper distribution (sum 1)
      The technique for dealing with underflow is described in
      http://jblevins.org/log/log-sum-exp
      """
      pd = (1/np.exp(pd[None,:] - pd[:,None]).sum(1))
      return pd
  else:
    def norm_probs(pd):
      return pd
    
  logger.debug('using numpy implementation')
  __USE_NUMPY__ = True

except ImportError:
  # Pure python implementation
  # This is a stub for a potential future numpy-less implementation.
  # I will not implement this unless there is a clear demand for it.
  raise NotImplementedError, "langid.py needs numpy to run - please contact the author if you need to use langid.py without numpy"
  def unpack(data):
    """
    Unpack a model that has been compressed into a string
    NOTE: nb_ptc and nb_pc are array.array('f') instances.
          nb_ptc is packed into a 1-dimensional array, each term is represented by
          len(nb_pc) continuous entries
    """
    global nb_ptc, nb_pc, nb_numfeats, nb_classes, tk_nextmove, tk_output
    model = loads(bz2.decompress(base64.b64decode(data)))
    nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = model
    nb_numfeats = len(nb_ptc) / len(nb_pc)

  def nb_classprobs(fv):
    raise NotImplementedError, "don't have pure python implementation yet"

  logger.debug('using python native implementation')
  __USE_NUMPY__ = False

def instance2fv(instance):
  """
  Map an instance into the feature space of the trained model.
  """
  if isinstance(instance, unicode):
    instance = instance.encode('utf8')

  if __USE_NUMPY__:
    fv = tokenize(instance, 
          np.zeros((nb_numfeats,), dtype='uint32'))
  else:
    fv = tokenize(instance,
        array.array('L', itertootls.repeat(0, nb_numfeats)))
  return fv

def classify(instance):
  """
  Classify an instance.
  """
  fv = instance2fv(instance)
  probs = norm_probs(nb_classprobs(fv))
  cl = argmax(probs)
  conf = probs[cl]
  pred = nb_classes[cl]
  return pred, conf

def rank(instance):
  """
  Return a list of languages in order of likelihood.
  """
  fv = instance2fv(instance)
  probs = norm_probs(nb_classprobs(fv))
  return [(k,v) for (v,k) in sorted(zip(probs, nb_classes), reverse=True)]

# Based on http://www.ubacoda.com/index.php?p=8
query_form = """
<html>
  <head>
    <meta http-equiv="Content-type" content="text/html; charset=utf-8">
    <title>Language Identifier</title>
    <script src="//ajax.googleapis.com/ajax/libs/jquery/1.7.2/jquery.min.js" type="text/javascript"></script>
    <script type="text/javascript" charset="utf-8">
      $(document).ready(function() {{
        $("#typerArea").keyup(displayType);
      
        function displayType(){{
          var contents = $("#typerArea").val();
          if (contents.length != 0) {{
            $.post(
              "/rank",
              {{q:contents}},
              function(data){{
                for(i=0;i<5;i++) {{
                  $("#lang"+i).html(data.responseData[i][0]);
                  $("#conf"+i).html(data.responseData[i][1]);
                }}
                $("#rankTable").show();
              }},
              "json"
            );
          }}
          else {{
            $("#rankTable").hide();
          }}
        }}
        $("#manualSubmit").remove();
        $("#rankTable").hide();
      }});
    </script>
  </head>
  <body>
    <form method=post>
      <center><table>
        <tr>
          <td>
            <textarea name="q" id="typerArea" cols=40 rows=6></textarea></br>
          </td>
        </tr>
        <tr>
          <td>
            <table id="rankTable">
              <tr>
                <td id="lang0">
                  <p>Unable to load jQuery, live update disabled.</p>
                </td><td id="conf0"/>
              </tr>
              <tr><td id="lang1"/><td id="conf1"></tr>
              <tr><td id="lang2"/><td id="conf2"></tr>
              <tr><td id="lang3"/><td id="conf3"></tr>
              <tr><td id="lang4"/><td id="conf4"></tr>
            </table>
            <input type=submit id="manualSubmit" value="submit">
          </td>
        </tr>
      </table></center>
    </form>

  </body>
</html>
"""
def application(environ, start_response):
  """
  WSGI-compatible langid web service.
  """
  try:
    path = shift_path_info(environ)
  except IndexError:
    # Catch shift_path_info's failure to handle empty paths properly
    path = ''

  if path == 'detect' or path == 'rank':
    data = None

    # Extract the data component from different access methods
    if environ['REQUEST_METHOD'] == 'PUT':
      data = environ['wsgi.input'].read(int(environ['CONTENT_LENGTH']))
    elif environ['REQUEST_METHOD'] == 'GET':
      try:
        data = parse_qs(environ['QUERY_STRING'])['q'][0]
      except KeyError:
        # No query, provide a null response.
        status = '200 OK' # HTTP Status
        response = {
          'responseData': None,
          'responseStatus': 200, 
          'responseDetails': None,
        }
    elif environ['REQUEST_METHOD'] == 'POST':
      input_string = environ['wsgi.input'].read(int(environ['CONTENT_LENGTH']))
      try:
        data = parse_qs(input_string)['q'][0]
      except KeyError:
        # No key 'q', process the whole input instead
        data = input_string
    else:
      # Unsupported method
      status = '405 Method Not Allowed' # HTTP Status
      response = { 
        'responseData': None, 
        'responseStatus': 405, 
        'responseDetails': '%s not allowed' % environ['REQUEST_METHOD'] 
      }

    if data is not None:
      if path == 'detect':
        pred,conf = classify(data)
        responseData = {'language':pred, 'confidence':conf}
      elif path == 'rank':
        responseData = rank(data)

      status = '200 OK' # HTTP Status
      response = {
        'responseData': responseData,
        'responseStatus': 200, 
        'responseDetails': None,
      }
  elif path == 'demo':
    status = '200 OK' # HTTP Status
    headers = [('Content-type', 'text/html; charset=utf-8')] # HTTP Headers
    start_response(status, headers)
    return [query_form.format(**environ)]
    
  else:
    # Incorrect URL
    status = '404 Not Found'
    response = {'responseData': None, 'responseStatus':404, 'responseDetails':'Not found'}

  headers = [('Content-type', 'text/javascript; charset=utf-8')] # HTTP Headers
  start_response(status, headers)
  return [json.dumps(response)]

if __name__ == "__main__":
  parser = optparse.OptionParser()
  parser.add_option('-s','--serve',action='store_true', default=False, dest='serve', help='launch web service')
  parser.add_option('--host', default=HOST, dest='host', help='host/ip to bind to')
  parser.add_option('--port', default=PORT, dest='port', help='port to listen on')
  parser.add_option('-v', action='count', dest='verbosity', help='increase verbosity (repeat for greater effect)')
  parser.add_option('-m', dest='model', help='load model from file')
  parser.add_option('-l', '--langs', dest='langs', help='comma-separated set of target ISO639 language codes (e.g en,de)')
  parser.add_option('-r', '--remote',action="store_true", default=False, help='auto-detect IP address for remote access')
  parser.add_option('--demo',action="store_true", default=False, help='launch an in-browser demo application')
  parser.add_option('-u', '--url', help='langid of URL')
  options, args = parser.parse_args()

  if options.verbosity:
    logging.basicConfig(level=max((5-options.verbosity)*10, 0))
  else:
    logging.basicConfig()

  # unpack a model 
  if options.model:
    try:
      with open(options.model) as f:
        unpack(f.read())
      logger.info("Using external model: %s", options.model)
    except IOError, e:
      logger.warning("Failed to load %s: %s" % (options.model,e))
  
  if not model_loaded:
    unpack(model)
    logger.info("Using internal model")

  if options.langs:
    langs = options.langs.split(",")
    set_languages(langs)

  if options.url:
    import urllib2
    import contextlib
    with contextlib.closing(urllib2.urlopen(options.url)) as url:
      text = url.read()
      lang, conf = classify(text)
      print options.url, len(text), lang
    
  elif options.serve or options.demo:
    # from http://stackoverflow.com/questions/166506/finding-local-ip-addresses-in-python
    if options.remote and options.host is None:
      # resolve the external ip address
      import socket
      s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      s.connect(("google.com",80))
      hostname = s.getsockname()[0]
    elif options.host is None:
      # resolve the local hostname
      import socket
      hostname = socket.gethostbyname(socket.gethostname())
    else:
      hostname = options.host

    if options.demo:
      import webbrowser
      webbrowser.open('http://{0}:{1}/demo'.format(hostname, options.port))
    try:
      if FORCE_WSGIREF: raise ImportError
      # Use fapws3 if available
      import fapws._evwsgi as evwsgi
      from fapws import base
      evwsgi.start(hostname,str(options.port))
      evwsgi.set_base_module(base)
      evwsgi.wsgi_cb(('', application))
      evwsgi.set_debug(0)
      evwsgi.run()
    except ImportError:
      print "Listening on %s:%d" % (hostname, int(options.port))
      print "Press Ctrl+C to exit"
      httpd = make_server(hostname, int(options.port), application)
      try:
        httpd.serve_forever()
      except KeyboardInterrupt:
        pass
  else:
    import sys
    if sys.stdin.isatty():
      # Interactive mode
      while True:
        try:
          print ">>>",
          text = raw_input()
        except Exception:
          break
        print classify(text)
    else:
      # Redirected
      print classify(sys.stdin.read())
     
else:
  # Running as an imported module; unpack the internal model
  unpack(model)
