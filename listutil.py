import collections

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
