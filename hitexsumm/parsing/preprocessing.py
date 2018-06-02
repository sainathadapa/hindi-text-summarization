#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains methods for parsing and preprocessing strings. Let's consider the most noticeable:

* :func:`~hitexsumm.parsing.preprocessing.remove_stopwords` - remove all stopwords from string
* :func:`~hitexsumm.parsing.preprocessing.preprocess_string` -  preprocess string (in default NLP meaning)

Examples:
---------
>>> from hitexsumm.parsing.preprocessing import remove_stopwords
>>> remove_stopwords("Better late than never, but better never late.")
u'Better late never, better late.'
>>>
>>> preprocess_string("<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?")
[u'hel', u'rld', u'weather', u'todai', u'isn']


Data:
-----

.. data:: STOPWORDS - Set of stopwords from Stone, Denis, Kwantes (2010).
.. data:: RE_PUNCT - Regexp for search an punctuation.
.. data:: RE_TAGS - Regexp for search an tags.
.. data:: RE_NUMERIC - Regexp for search an numbers.
.. data:: RE_NONALPHA - Regexp for search an non-alphabetic character.
.. data:: RE_AL_NUM - Regexp for search a position between letters and digits.
.. data:: RE_NUM_AL - Regexp for search a position between digits and letters .
.. data:: RE_WHITESPACE - Regexp for search space characters.
.. data:: DEFAULT_FILTERS - List of function for string preprocessing.

"""

import re
import string
import glob

from hitexsumm import utils
from hitexsumm.parsing.porter import PorterStemmer


# from https://github.com/stopwords-iso/stopwords-hi
# Released under The MIT License (MIT)
# https://github.com/stopwords-iso/stopwords-hi/blob/master/LICENSE
STOPWORDS = frozenset([
    "अंदर", "अत", "अदि", "अप", "अपना", "अपनि", "अपनी", "अपने", "अभि", "अभी", "आदि",
    "आप", "इंहिं", "इंहें", "इंहों", "इतयादि", "इत्यादि", "इन", "इनका", "इन्हीं", "इन्हें", "इन्हों",
    "इस", "इसका", "इसकि", "इसकी", "इसके", "इसमें", "इसि", "इसी", "इसे", "उंहिं", "उंहें",
    "उंहों", "उन", "उनका", "उनकि", "उनकी", "उनके", "उनको", "उन्हीं", "उन्हें", "उन्हों", "उस",
    "उसके", "उसि", "उसी", "उसे", "एक", "एवं", "एस", "एसे", "ऐसे", "ओर", "और", "कइ",
    "कई", "कर", "करता", "करते", "करना", "करने", "करें", "कहते", "कहा", "का", "काफि",
    "काफ़ी", "कि", "किंहें", "किंहों", "कितना", "किन्हें", "किन्हों", "किया", "किर", "किस", "किसि", "किसी",
    "किसे", "की", "कुछ", "कुल", "के", "को", "कोइ", "कोई", "कोन", "कोनसा", "कौन", "कौनसा",
    "गया", "घर", "जब", "जहाँ", "जहां", "जा", "जिंहें", "जिंहों", "जितना", "जिधर", "जिन", "जिन्हें",
    "जिन्हों", "जिस", "जिसे", "जीधर", "जेसा", "जेसे", "जैसा", "जैसे", "जो", "तक", "तब", "तरह",
    "तिंहें", "तिंहों", "तिन", "तिन्हें", "तिन्हों", "तिस", "तिसे", "तो", "था", "थि", "थी", "थे", "दबारा",
    "दवारा", "दिया", "दुसरा", "दुसरे", "दूसरे", "दो", "द्वारा", "न", "नहिं", "नहीं", "ना", "निचे",
    "निहायत", "नीचे", "ने", "पर", "पहले", "पुरा", "पूरा", "पे", "फिर", "बनि", "बनी", "बहि",
    "बही", "बहुत", "बाद", "बाला", "बिलकुल", "भि", "भितर", "भी", "भीतर", "मगर", "मानो", "मे",
    "में", "यदि", "यह", "यहाँ", "यहां", "यहि", "यही", "या", "यिह", "ये", "रखें", "रवासा", "रहा",
    "रहे", "ऱ्वासा", "लिए", "लिये", "लेकिन", "व", "वगेरह", "वरग", "वर्ग", "वह", "वहाँ", "वहां",
    "वहिं", "वहीं", "वाले", "वुह", "वे", "वग़ैरह", "संग", "सकता", "सकते", "सबसे", "सभि", "सभी",
    "साथ", "साबुत", "साभ", "सारा", "से", "सो", "हि", "ही", "हुअ", "हुआ", "हुइ", "हुई", "हुए",
    "हे", "हें", "है", "हैं", "हो", "होता", "होति", "होती", "होते", "होना", "होने"
])


RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
RE_NONALPHA = re.compile(r"\W", re.UNICODE)
RE_AL_NUM = re.compile(r"([a-z]+)([0-9]+)", flags=re.UNICODE)
RE_NUM_AL = re.compile(r"([0-9]+)([a-z]+)", flags=re.UNICODE)
RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)


def remove_stopwords(s):
    """Remove :const:`~hitexsumm.parsing.preprocessing.STOPWORDS` from `s`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without :const:`~hitexsumm.parsing.preprocessing.STOPWORDS`.

    Examples
    --------
    >>> from hitexsumm.parsing.preprocessing import remove_stopwords
    >>> remove_stopwords("Better late than never, but better never late.")
    u'Better late never, better late.'

    """
    s = utils.to_unicode(s)
    return " ".join(w for w in s.split() if w not in STOPWORDS)


def strip_punctuation(s):
    """Replace punctuation characters with spaces in `s` using :const:`~hitexsumm.parsing.preprocessing.RE_PUNCT`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without punctuation characters.

    Examples
    --------
    >>> from hitexsumm.parsing.preprocessing import strip_punctuation
    >>> strip_punctuation("A semicolon is a stronger break than a comma, but not as much as a full stop!")
    u'A semicolon is a stronger break than a comma  but not as much as a full stop '

    """
    s = utils.to_unicode(s)
    return RE_PUNCT.sub(" ", s)


strip_punctuation2 = strip_punctuation


def strip_tags(s):
    """Remove tags from `s` using :const:`~hitexsumm.parsing.preprocessing.RE_TAGS`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without tags.

    Examples
    --------
    >>> from hitexsumm.parsing.preprocessing import strip_tags
    >>> strip_tags("<i>Hello</i> <b>World</b>!")
    u'Hello World!'

    """
    s = utils.to_unicode(s)
    return RE_TAGS.sub("", s)


def strip_short(s, minsize=3):
    """Remove words with length lesser than `minsize` from `s`.

    Parameters
    ----------
    s : str
    minsize : int, optional

    Returns
    -------
    str
        Unicode string without short words.

    Examples
    --------
    >>> from hitexsumm.parsing.preprocessing import strip_short
    >>> strip_short("salut les amis du 59")
    u'salut les amis'
    >>>
    >>> strip_short("one two three four five six seven eight nine ten", minsize=5)
    u'three seven eight'

    """
    s = utils.to_unicode(s)
    return " ".join(e for e in s.split() if len(e) >= minsize)


def strip_numeric(s):
    """Remove digits from `s` using :const:`~hitexsumm.parsing.preprocessing.RE_NUMERIC`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode  string without digits.

    Examples
    --------
    >>> from hitexsumm.parsing.preprocessing import strip_numeric
    >>> strip_numeric("0text24hitexsumm365test")
    u'texthitexsummtest'

    """
    s = utils.to_unicode(s)
    return RE_NUMERIC.sub("", s)


def strip_non_alphanum(s):
    """Remove non-alphabetic characters from `s` using :const:`~hitexsumm.parsing.preprocessing.RE_NONALPHA`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string with alphabetic characters only.

    Notes
    -----
    Word characters - alphanumeric & underscore.

    Examples
    --------
    >>> from hitexsumm.parsing.preprocessing import strip_non_alphanum
    >>> strip_non_alphanum("if-you#can%read$this&then@this#method^works")
    u'if you can read this then this method works'

    """
    s = utils.to_unicode(s)
    return RE_NONALPHA.sub(" ", s)


def strip_multiple_whitespaces(s):
    r"""Remove repeating whitespace characters (spaces, tabs, line breaks) from `s`
    and turns tabs & line breaks into spaces using :const:`~hitexsumm.parsing.preprocessing.RE_WHITESPACE`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without repeating in a row whitespace characters.

    Examples
    --------
    >>> from hitexsumm.parsing.preprocessing import strip_multiple_whitespaces
    >>> strip_multiple_whitespaces("salut" + '\r' + " les" + '\n' + "         loulous!")
    u'salut les loulous!'

    """
    s = utils.to_unicode(s)
    return RE_WHITESPACE.sub(" ", s)


def split_alphanum(s):
    """Add spaces between digits & letters in `s` using :const:`~hitexsumm.parsing.preprocessing.RE_AL_NUM`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string with spaces between digits & letters.

    Examples
    --------
    >>> from hitexsumm.parsing.preprocessing import split_alphanum
    >>> split_alphanum("24.0hours7 days365 a1b2c3")
    u'24.0 hours 7 days 365 a 1 b 2 c 3'

    """
    s = utils.to_unicode(s)
    s = RE_AL_NUM.sub(r"\1 \2", s)
    return RE_NUM_AL.sub(r"\1 \2", s)


def stem_text(text):
    """Transform `s` into lowercase and stem it.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Unicode lowercased and porter-stemmed version of string `text`.

    Examples
    --------
    >>> from hitexsumm.parsing.preprocessing import stem_text
    >>> stem_text("While it is quite useful to be able to search a large collection of documents almost instantly.")
    u'while it is quit us to be abl to search a larg collect of document almost instantly.'

    """
    text = utils.to_unicode(text)
    p = PorterStemmer()
    return ' '.join(p.stem(word) for word in text.split())


stem = stem_text


DEFAULT_FILTERS = [
    lambda x: x.lower(), strip_tags, strip_punctuation,
    strip_multiple_whitespaces, strip_numeric,
    remove_stopwords, strip_short, stem_text
]


def preprocess_string(s, filters=DEFAULT_FILTERS):
    """Apply list of chosen filters to `s`.

    Default list of filters:

    * :func:`~hitexsumm.parsing.preprocessing.strip_tags`,
    * :func:`~hitexsumm.parsing.preprocessing.strip_punctuation`,
    * :func:`~hitexsumm.parsing.preprocessing.strip_multiple_whitespaces`,
    * :func:`~hitexsumm.parsing.preprocessing.strip_numeric`,
    * :func:`~hitexsumm.parsing.preprocessing.remove_stopwords`,
    * :func:`~hitexsumm.parsing.preprocessing.strip_short`,
    * :func:`~hitexsumm.parsing.preprocessing.stem_text`.

    Parameters
    ----------
    s : str
    filters: list of functions, optional

    Returns
    -------
    list of str
        Processed strings (cleaned).

    Examples
    --------
    >>> from hitexsumm.parsing.preprocessing import preprocess_string
    >>> preprocess_string("<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?")
    [u'hel', u'rld', u'weather', u'todai', u'isn']
    >>>
    >>> s = "<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?"
    >>> CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation]
    >>> preprocess_string(s, CUSTOM_FILTERS)
    [u'hel', u'9lo', u'wo9', u'rld', u'th3', u'weather', u'is', u'really', u'g00d', u'today', u'isn', u't', u'it']

    """
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s.split()


def preprocess_documents(docs):
    """Apply :const:`~hitexsumm.parsing.preprocessing.DEFAULT_FILTERS` to the documents strings.

    Parameters
    ----------
    docs : list of str

    Returns
    -------
    list of list of str
        Processed documents split by whitespace.

    Examples
    --------
    >>> from hitexsumm.parsing.preprocessing import preprocess_documents
    >>> preprocess_documents(["<i>Hel 9lo</i> <b>Wo9 rld</b>!", "Th3     weather_is really g00d today, isn't it?"])
    [[u'hel', u'rld'], [u'weather', u'todai', u'isn']]

    """
    return [preprocess_string(d) for d in docs]


def read_file(path):
    with utils.smart_open(path) as fin:
        return fin.read()


def read_files(pattern):
    return [read_file(fname) for fname in glob.glob(pattern)]
