from typing import Counter
import os
import json

import numpy as np
from textblob import TextBlob
import textstat
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

from AP_Bots.utils.output_parser import extract_bfi_scores

class FeatureProcessor():

    def __init__(self, dataset) -> None:
        
        self.dataset = dataset
        nltk.download('punkt', quiet=True)      
        nltk.download('stopwords', quiet=True)  
        nltk.download('averaged_perceptron_tagger', quiet=True)
        self.save_loc = os.path.join("files", "features")
        os.makedirs(self.save_loc, exist_ok=True)

    @staticmethod
    def get_sentence_length(texts):
        if isinstance(texts, list):
            return [len(text.split(" ")) for text in texts]
        else:
            return [len(texts.split(" "))]

    @staticmethod
    def get_sentiment_polarity(texts):
        if isinstance(texts, list):
            return [TextBlob(text).sentiment.polarity for text in texts]
        else:
            return [TextBlob(texts).sentiment.polarity]

    @staticmethod
    def get_subjectivity(texts):
        if isinstance(texts, list):
            return [TextBlob(text).sentiment.subjectivity for text in texts]
        else:
            return [TextBlob(texts).sentiment.subjectivity]
    
    @staticmethod
    def get_smog_index(texts):
        if isinstance(texts, list):
            return [textstat.smog_index(text) for text in texts]
        else:
            return [textstat.smog_index(texts)]
                            
    @staticmethod
    def get_adverb_usage(texts):
        res = []
        if not isinstance(texts, list):
            texts = [texts]

        for text in texts:
            words = word_tokenize(text.lower())
            pos_tags = pos_tag(words)
            adverbs = [word for word, pos in pos_tags if pos.startswith('RB')]
            res.append(round((len(adverbs) / len(words)) * 100, 2))

        return res
    
    @staticmethod
    def get_adjective_usage(texts):
        res = []
        if not isinstance(texts, list):
            texts = [texts]

        for text in texts:
            words = word_tokenize(text.lower())
            pos_tags = pos_tag(words)
            adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
            res.append(round((len(adjectives) / len(words)) * 100, 2))

        return res
    
    @staticmethod
    def get_pronoun_usage(texts):
        res = []
        if not isinstance(texts, list):
            texts = [texts]

        for text in texts:
            words = word_tokenize(text.lower())
            pos_tags = pos_tag(words)
            pronouns = [word for word, pos in pos_tags if pos.startswith('PRP')]
            res.append(round((len(pronouns) / len(words)) * 100, 2))

        return res

    @staticmethod
    def get_word_frequency(texts):
        texts = " ".join(texts)
        words = word_tokenize(texts.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        word_freq = Counter(filtered_words)
        total_words = sum(word_freq.values())

        word_freq_percentage = {word: round((count / total_words) * 100, 3) for word, count in word_freq.items()} if total_words > 0 else {} 
        sorted_word_freq_percentage = list(sorted(word_freq_percentage.items(), key=lambda item: item[1], reverse=True))   

        return sorted_word_freq_percentage

    def get_named_entity_freqency(self, texts):
        texts = " ".join(texts)
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(texts)
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        entity_counter = Counter(named_entities)
        total_entities = sum(entity_counter.values())
        
        sorted_entities = [
            ((ent, label), round((count / total_entities) * 100, 2))
            for (ent, label), count in entity_counter.most_common()
        ]
        
        return sorted_entities
    
    def get_dep_pattern_frequency(self, texts):
        texts = " ".join(texts)
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(texts)
        
        dependency_patterns = [(token.text, token.dep_) for token in doc]
        pattern_counter = Counter(dependency_patterns)
        total_patterns = sum(pattern_counter.values())
        
        sorted_patterns = [
            (pattern, round((count / total_patterns) * 100, 2))
            for pattern, count in pattern_counter.most_common()
        ]

        return sorted_patterns
    
    @staticmethod
    def get_bfi_scores(texts):
        if not isinstance(texts, list):
            texts = [texts]

        bfi_scores = [extract_bfi_scores(text) for text in texts]

        return bfi_scores

    def feat_name_mappings(self):
        return {
            "MSWC": {
                "full_name": "Mean Sentence Word Count",
                "desc": "The number of words the writer uses in a sentence on average",
                "func": self.get_sentence_length
            },
            "SP": {
                "full_name": "Sentiment Polarity",
                "desc": "Average sentiment polarity for the writer (between -1 to 1)",
                "func": self.get_sentiment_polarity
            },
            "SUBJ": {
                "full_name": "Subjectivity",
                "desc": "Average subjectivity for the writer (between 0 to 1, 1 being the most subjective)",
                "func": self.get_subjectivity
            },
            "SMOG": {
                "full_name": "SMOG Index",
                "desc": "Average SMOG Index of the writer, which measures how many years of education is needed to understand the text",
                "func": self.get_smog_index
            },
            "ADVU": {
                "full_name": "Adverb Usage Percentage",
                "desc": "The percentage (between 0-100) of adverbs the writer uses on average",
                "func": self.get_adverb_usage
            },
            "ADJU": {
                "full_name": "Adjective Usage Percentage",
                "desc": "The percentage (between 0-100) of adjectives the writer uses on average",
                "func": self.get_adjective_usage
            },
            "PU": {
                "full_name": "Pronoun Usage Percentage",
                "desc": "The percentage (between 0-100) of pronouns the writer uses on average",
                "func": self.get_pronoun_usage
            },
            "NEF": {
                "full_name": "Named Entity Frequencies",
                "desc": "Most frequently used named entities of the writer",
                "func": self.get_named_entity_freqency
            },
            "DPF": {
                "full_name": "Dependency Pattern Frequencies",
                "desc": "Most frequently used dependency patterns of the writer",
                "func": self.get_dep_pattern_frequency
            },
            "WF": {
                "full_name": "Word Frequencies",
                "desc": "Most frequently used words of the writer",
                "func": self.get_word_frequency
            },
            "BFI": {
                "full_name": "Big Five Inventory Scores",
                "desc": "Scores for the Big Five Inventory traits of the writer (from 1 to 5)",
                "func": self.get_bfi_scores
            }
        }

    def get_feat_file(self, file_name):
        file_path = os.path.join(self.save_loc, f"{file_name}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            return dict()

    def save_feat_file(self, file_name, obj):
        file_path = os.path.join(self.save_loc, f"{file_name}.json")
        with open(file_path, "w") as f:
            json.dump(obj, f)

    def get_all_features(self, feature_list):

        file_name = f"{self.dataset.tag}_feats"
        _, retr_texts, retr_gts = self.dataset.get_retr_data() 

        author_texts = retr_gts
        full_auth_texts = retr_texts if retr_texts != retr_gts else author_texts
        author_features = self.get_feat_file(file_name)
        feature_mappings = self.feat_name_mappings()

        for feature in feature_list:
            if feature not in author_features.keys():
                print(f"Preparing {feature}")
                if feature in ["NEF", "DPF"]:
                    author_features[feature] = [feature_mappings[feature]["func"](text) for text in full_auth_texts]
                elif feature == "BFI":
                    bfi_path = "personality_analysis/files/inferred_bfi/amazon_Grocery_and_Gourmet_Food_2018_UP_BFI_GEMMA-2-27B.json"
                    with open(bfi_path, "r") as f:
                        bfi_texts = json.load(f)
                    author_features[feature] = [feature_mappings[feature]["func"](text) for text in bfi_texts]
                else:
                    if feature == "WF" or feature == "BFI":
                        author_features[feature] = [feature_mappings[feature]["func"](text) for text in author_texts]
                    else:
                        author_features[feature] = [np.mean(feature_mappings[feature]["func"](text)) for text in author_texts]
            self.save_feat_file(file_name, author_features)
        return author_features
    
    def prepare_features(self, feature_list, top_k=10):

        all_features = self.get_all_features(feature_list)

        all_author_features = []
        for i in range(len(all_features[list(all_features.keys())[0]])):
            proc_author_features = []
            for feature in feature_list:
                if not feature.endswith("F") and feature != "BFI":
                    pers_value = round(all_features[feature][i], 3)
                elif feature == "BFI":
                    pers_value = all_features[feature][i]
                else:
                    pers_value = [w for w, _ in all_features[feature][i][:top_k]]
                feat_desc = f"-{self.feat_name_mappings()[feature]['desc']} is: {pers_value}"
                proc_author_features.append(feat_desc)
            all_author_features.append(proc_author_features)
        return all_author_features
