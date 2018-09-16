

models = ['BaseLighter_SolidWorks_center.stl',
'batteryCMM_center.stl',
'knob1_proto_center.stl',
'OilSealDisc_SolidWorks_center.stl',
'Screw_SLDW_center.stl',
'sg-r41ac8600-mp-1566915-01_C.stl',
'sg-r41ac8600-mp-1568037-01_C.stl',
'sg-r41ac8600-mp-1573710-01_C.stl',
'sg-r41ac8600-mp-1576207-01_C.stl',
'sg-r41ac8600-mp-1620498-01_C.stl',
'sg-r41ac8610-mp-1618508-01_C.stl',
'sg-r41ac8610-mp-1618514-01_C.stl',
'sg-r41ac8610-mp-1618608-01_C.stl',
'sg-r41ad1710-mp-1559596-01_C.stl',
'sg-r41ad1710-mp-1559654-01_C.stl',
'sg-r41ad2840-mp-2143464-01_C.stl',
'Zigzag_SolidWorks_center.stl']

folders = ['BaseLighter_SolidWorks_center_dim', 'batteryCMM_center', 'knob1_proto_center', 'OilSealDisk_SolidWorks_center', 'Screw_SLDW_center', '1566915', '1568037', '1573710', '1576207', '1620498', '1618508', '1618514', '1618608', '1559596', '1559654', '2143464', 'Zigzag_SolidWorks_center']
import numpy as np

symmetries_maya_world = [[0,0,0], [0,2,0], [0,0,3], [0,2,0], [6,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,12], [0,0,0], [0,0,0], [0,0,0]]
# The one with 12, could be actually 6,12 or inf


def get_correspondances():
  return models, symmetries_maya_world, folders
