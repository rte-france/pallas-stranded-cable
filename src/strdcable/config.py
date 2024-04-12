#!/usr/bin/env python
# -*- coding: utf-8 -*-
import configparser
import os


# %%
__PATH_STRDCABLE__ = os.path.dirname(os.path.realpath(__file__))
cfg = configparser.ConfigParser()
cfg.read(__PATH_STRDCABLE__ + '/config.ini')
