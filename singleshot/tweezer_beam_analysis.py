from analysislib.common.tweezer_finding import TweezerFinder
from analysislib.common.image_preprocessor import ImagePreprocessor
from analysislib.common.analysis_config import manta_tweezer_system
from analysislib.common.image import Image

preprocessor = ImagePreprocessor(imaging_setup=manta_tweezer_system, load_type='lyse')

# make tweezerfinder
TweezerFinder(images=[Image(preprocessor.exposures[0])])


# find site rois

# fit site rois

# plot amplitude/position/blah vs index
