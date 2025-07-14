from sklearn.pipeline import Pipeline
def get_features(pipeline: Pipeline):
    raw_features=pipeline.get_feature_names_out()
    features=[]
    for i in raw_features:
        features.append(i.split("__")[-1])
    return features