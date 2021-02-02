#!/usr/bin/env python
# coding: utf-8

import logging

logging.basicConfig(level=logging.DEBUG)
g_debugLevel = logging.DEBUG


### SET logging.basicConfig() BEFORE



def setDebugLevel(level):
    """
    Set debug level: logging.DEBUG ... logging.CRITICAL

    We use a the global logging module, so this is not only this class
    """
    global g_debugLevel

    level = int(level)

    g_debugLevel = level
    logging.basicConfig(level=level)

    print("setDebugLevel(): Set global logging level to", g_debugLevel)


def printDebugLevel():
    """
    Print debug level set by setDebugLevel()

    Inform about levels as well
    """
    global g_debugLevel

    print("debugLevel(): level is", g_debugLevel)
    print("logging.DEBUG=", logging.DEBUG)
    print("logging.INFO=", logging.INFO)
    print("logging.WARNING=", logging.WARNING)
    print("logging.ERROR=", logging.ERROR)
    print("logging.CRITICAL=", logging.CRITICAL)

def debug(*msg):
    """ Return anchor object with highest hitrate

    @return ANCHOROBJ (type1)
    """
    global g_debugLevel

    if g_debugLevel > logging.DEBUG:     # for IPtython
        return

    msgTotal = ""
    for arg in msg:
        msgTotal += str(arg) +" "
    logging.debug(msgTotal)

def info(*msg):
    global g_debugLevel

    if g_debugLevel > logging.INFO:     # for IPtython
        return

    msgTotal = ""
    for arg in msg:
        msgTotal += str(arg) +" "
    logging.info(msgTotal)
    
def warning(*msg):
    global g_debugLevel

    if g_debugLevel > logging.WARNING:     # for IPtython
        return

    msgTotal = ""
    for arg in msg:
        msgTotal += str(arg) +" "
    logging.warning(msgTotal)

def error(*msg):
    global g_debugLevel

    if g_debugLevel > logging.ERROR:     # for IPtython
        return

    msgTotal = ""
    for arg in msg:
        msgTotal += str(arg) +" "
    logging.error(msgTotal)

def critical(*msg):
    global g_debugLevel

    if g_debugLevel > logging.CRITICAL:     # for IPtython
        return

    msgTotal = ""
    for arg in msg:
        msgTotal += str(arg) +" "
    logging.critical(msgTotal)



###   For printing text and list of text .......................

STRING_LENGTH = 30             # length of string to output
ITERABLE_LENGTH = 10           # number of elements to output

def getSnippet(textable):
    """
    Return first 10 elements or some characters of string

    textable is iterable or string
    """
    def isIterable(obj):
        """
        Helper to check the obj is iterable
        """
        try:
            iter(obj)
        except Exception:
            return False
        else:
            return True

    def shorten(string):
        """
        Helper to cut down to max of 40 characters
        """
        #ret = ""
        string = str(string)
        if len(string) > STRING_LENGTH:
            return string[:STRING_LENGTH] +".."
        return string

    if isinstance(textable,str):
        return shorten(textable)
    elif isIterable(textable):
        ret = "["
        nMax = min(ITERABLE_LENGTH,len(textable))

        for i in range(nMax):
            if i != 0:
                ret += ","
            ret += shorten(textable[i])
        if nMax != len(textable):  # shortened?
            ret += ",.."
        ret += "]"
        return ret
    else:   # textable?
        string = shorten(textable)
        return string


