# ###################################################
# =*= COMMON FUNCTIONS SHARED ACCROSS THE PROJECT =*=
# ###################################################
__author__ = "Hedi Boukamcha - hedi.boukamcha.1@ulaval.ca, Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

def to_bool(v: str) -> bool:
    return v.lower() in ['true', 't', 'yes', '1']