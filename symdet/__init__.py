"""
__init__ file for the symdet package
"""
import os
from symdet.data.double_well_potential import DoubleWellPotential
from symdet.ml_models.dense_model import DenseModel
from symdet.symmetry_group_extraction.group_detection import GroupDetection
from symdet.generator_extraction.generators import GeneratorExtraction
from symdet.data.so2_data import SO2
from symdet.data.so3_data import SO3

__all__ = ['DoubleWellPotential', 'DenseModel', 'GroupDetection', 'SO2', 'SO3']
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
