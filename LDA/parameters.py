IN_DIR = '../data'
OUT_DIR = '../out'
FILENAME = 'MyA_TY2018_all.csv'

TEXT_COL = "Please tell us why you were dissatisfied with today's experience. Please be as specific as possible."
RENAMED_TEXT_COL = "dissatisfaction_reason"

# TEXT_COL = "Do you have any suggestions for improvements to information or services in My Account?"
# RENAMED_TEXT_COL = "suggestions"

LANGUAGE = 'EN' #'FR' , 'ALL'


####- NLP PREPROCESSING PARAMETERS- ####
EXTRA_STOP_WORDS = ['cra', 'say', 'want' , 'like']

NUM_TOPICS = 14