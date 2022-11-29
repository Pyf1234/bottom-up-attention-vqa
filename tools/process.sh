# Process data

#this script create the dictionary and pick the word embedding which will be used
python tools/create_dictionary.py
#this script create the preprocessed dataset which contains the following fields
#'question_id'
#'question_type'
#'image_id'
#'label_counts'
#'labels'
#'scores'
python tools/compute_softscore.py v2
python tools/compute_softscore.py cp_v2
#the feature of all the image was collected in a single tsv file
#this script convert the single tsv file to multiple bin files(a single bin file per image)
python tools/detection_features_converter_fs.py
