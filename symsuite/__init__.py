"""
__init__ file for the symsuite package
"""
import os

from symsuite.data.double_well_potential import DoubleWellPotential
from symsuite.data.so2_data import SO2
from symsuite.data.so3_data import SO3
from symsuite.generator_extraction.generators import GeneratorExtraction
from symsuite.ml_models.dense_model import DenseModel
from symsuite.symmetry_group_extraction.group_detection import GroupDetection

__all__ = ["DoubleWellPotential", "DenseModel", "GroupDetection", "SO2", "SO3"]
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
