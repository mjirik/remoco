#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.

"""
Module for testing format rawiv
"""
import logging
logger = logging.getLogger(__name__)
import unittest
import os.path as op
import shutil
import sys
import numpy as np

import xml.etree.ElementTree as etree
import pvcamctrl as anse
# import io3d

from nose.plugins.attrib import attr

class LiverMouseTest(unittest.TestCase):

    # @attr('actual')
    # @attr('interactive')
    def test_xml_parser_simple(self):
        test_xmlstr = "<?xml version='1.0' encoding='UTF-8' standalone='yes' ?><NodeId>0</NodeId><Accelerometer><Accelerometer1>0.670318603515625</Accelerometer1><Accelerometer2>0.05816650390625</Accelerometer2><Accelerometer3>9.442153930664062</Accelerometer3></Accelerometer><Gyroscope><Gyroscope1>-0.001861572265625</Gyroscope1><Gyroscope2>0.0047454833984375</Gyroscope2><Gyroscope3>8.23974609375E-4</Gyroscope3></Gyroscope><Magnetometer><Magnetometer1>3.1328201293945312</Magnetometer1><Magnetometer2>-39.75334167480469</Magnetometer2><Magnetometer3>-26.612472534179688</Magnetometer3></Magnetometer><RotationVector><RotationVector1>0.03671461343765259</RotationVector1><RotationVector2>8.875191560946405E-4</RotationVector2><RotationVector3>0.9978557229042053</RotationVector3></RotationVector><TimeStamp>1535453234555</TimeStamp>"
        # test_xmlstr = "<NodeId>0</NodeId>"
        # test_xmlstr = "<?xml version='1.0' encoding='UTF-8' standalone='yes' ?><NodeId>0</NodeId><Accelerometer></Accelerometer>"

        # tree = etree.parse('examples/feed.xml')
        # root = tree.getroot()
        fixed_xmlstr = anse.fix_xml(test_xmlstr)
        root = etree.fromstring(fixed_xmlstr)
        self.assertNotNone(root)

    def test_xml_parse_sensors(self):
        test_xmlstr = "<?xml version='1.0' encoding='UTF-8' standalone='yes' ?><NodeId>0</NodeId><Accelerometer><Accelerometer1>0.670318603515625</Accelerometer1><Accelerometer2>0.05816650390625</Accelerometer2><Accelerometer3>9.442153930664062</Accelerometer3></Accelerometer><Gyroscope><Gyroscope1>-0.001861572265625</Gyroscope1><Gyroscope2>0.0047454833984375</Gyroscope2><Gyroscope3>8.23974609375E-4</Gyroscope3></Gyroscope><Magnetometer><Magnetometer1>3.1328201293945312</Magnetometer1><Magnetometer2>-39.75334167480469</Magnetometer2><Magnetometer3>-26.612472534179688</Magnetometer3></Magnetometer><RotationVector><RotationVector1>0.03671461343765259</RotationVector1><RotationVector2>8.875191560946405E-4</RotationVector2><RotationVector3>0.9978557229042053</RotationVector3></RotationVector><TimeStamp>1535453234555</TimeStamp>"
        fixed_xmlstr = anse.fix_xml(test_xmlstr)
        root = etree.fromstring(fixed_xmlstr)
        rotation = anse.get_rotation_vector(root)
        rotation


if __name__ == "__main__":
    unittest.main()
