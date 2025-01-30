from AP_Bots.feature_processor import FeatureProcessor

def get_sentiment_polarity(text):
    return FeatureProcessor.get_sentiment_polarity(text) 

def get_vader_sent_polarity(text):
    return FeatureProcessor.get_vader_sent_polarity(text)

def get_bert_sentiment(text):
    return FeatureProcessor.get_bert_sentiment(text)