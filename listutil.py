"""
@author: Kevin S. Brown, University of Connecticut

This source code is provided under the BSD-3 license, duplicated as follows:

Copyright (c) 2013, Kevin S. Brown
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or other
materials provided with the distribution.

3. Neither the name of the University of Connecticut  nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import collections,operator

def unique(seq, idfun=repr):
    """
    Returns a list of unique items in a sequence of items.  There are lots of ways to
    do this; here is one.
    """
    seen = {}
    return [seen.setdefault(idfun(e),e) for e in seq if idfun(e) not in seen]


def flatten(l):
    """
    Generator that flattens a list.
    """
    for el in l:
        if isinstance(el,collections.Iterable) and not isinstance(el,basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def sort_by_value(D,reverse=False):
    """
    There are many ways to sort a dictionary by value and return lists/tuples/etc.
    This is one recommended in PEP265.
    """
    return sorted(D.iteritems(),key=operator.itemgetter(1),reverse=reverse)


def circshift(s,n):
    """
    Circularly shifts the input string or list s by n positions.  n > 0 will do
    a shift to the left, and n < 0 to the right.
    """
    return s[n:]+s[:n]
